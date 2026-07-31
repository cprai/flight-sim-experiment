//! Rendering a single frame with no window behind it.
//!
//! The application draws into a swapchain, which needs a display server to
//! present to. Plenty of machines that can run the GPU cannot present from it --
//! a container given `/dev/dri` but no X or Wayland socket is the case that
//! prompted this -- and on those the whole renderer is unreachable even though
//! the hardware is right there. Drawing into a plain texture and reading it back
//! asks nothing of the windowing system, so it works anywhere the device does.
//!
//! Nothing here is a second renderer: [`Scene`] already takes a viewport and a
//! format rather than a surface, so this is only the target and the readback
//! that [`crate::renderer::Renderer`] gets from the swapchain instead.

use std::path::Path;
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use glam::{UVec2, Vec3};

use crate::camera::Camera;
use crate::scene::{Scene, create_depth_view};

/// Format of the captured frame.
///
/// sRGB, like the format [`crate::renderer::Renderer`] picks out of the surface
/// capabilities, so the shaders' output is encoded the same way here as it is on
/// screen. It also means the readback bytes are already what a PNG holds and
/// need no conversion. Callers must build their [`Scene`] with this format: the
/// pipelines bake their colour target format in.
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Where to put the camera, as `x,y,z,yaw,pitch`.
///
/// Position in metres, angles in degrees. Roll is not offered because a still of
/// terrain has no use for a tilted horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub position: Vec3,
    pub yaw_degrees: f32,
    pub pitch_degrees: f32,
}

impl Placement {
    /// Moves `camera` here, leaving its aspect and projection alone.
    pub fn apply(&self, camera: &mut Camera) {
        camera.position = self.position;
        camera.orientation = Camera::from_yaw_pitch_roll(
            self.yaw_degrees.to_radians(),
            self.pitch_degrees.to_radians(),
            0.0,
        );
    }
}

impl FromStr for Placement {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        let numbers: Vec<f32> = text
            .split(',')
            .map(|part| {
                part.trim()
                    .parse()
                    .with_context(|| format!("{part:?} is not a number"))
            })
            .collect::<Result<_>>()?;
        let [x, y, z, yaw, pitch] = numbers[..] else {
            bail!(
                "expected x,y,z,yaw,pitch -- five numbers, got {}",
                numbers.len()
            );
        };
        Ok(Self {
            position: Vec3::new(x, y, z),
            yaw_degrees: yaw,
            pitch_degrees: pitch,
        })
    }
}

/// A GPU device and queue with no surface behind them.
///
/// Requests exactly what [`crate::renderer::Renderer`] does, minus the
/// `compatible_surface` it has no surface to name. Asking for the same features
/// and limits is what makes a frame captured here evidence about the frame the
/// application would draw, rather than about some other configuration.
///
/// The power preference matters more here than it looks: with the default,
/// [`wgpu`] takes whichever adapter enumerates first and does no sorting, which
/// on a machine with a discrete GPU, an integrated one and a software fallback
/// is a coin toss. `WGPU_POWER_PREF` decides it.
pub fn device() -> Result<(wgpu::Device, wgpu::Queue)> {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::from_env().unwrap_or_default(),
        ..Default::default()
    }))
    .context("no wgpu adapter available")?;
    // Which device this ran on decides whether the timings mean anything and
    // whether a difference in the image is the change or the driver, so say it
    // rather than leave it to be assumed.
    log::info!("using adapter: {:?}", adapter.get_info());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("headless device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .context("failed to create device")?;
    Ok((device, queue))
}

/// Draws `scene` into a fresh texture and returns it as tightly packed RGBA8.
///
/// `scene` must have been built with [`CAPTURE_FORMAT`] and a viewport of
/// `size`, and [`Scene::update`] must have run since the camera last moved.
pub fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &Scene,
    size: UVec2,
) -> Result<Vec<u8>> {
    let extent = wgpu::Extent3d {
        width: size.x,
        height: size.y,
        depth_or_array_layers: 1,
    };
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("capture"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: CAPTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let depth = create_depth_view(device, size.x, size.y);

    // A texture-to-buffer copy wants its rows on a 256-byte stride, which only
    // widths that are a multiple of 64 give for free. Padding the buffer and
    // dropping the slack afterwards is what lets any width be asked for.
    let packed_row = size.x * 4;
    let bytes_per_row = packed_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: u64::from(bytes_per_row) * u64::from(size.y),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("capture"),
    });
    scene.draw(&mut encoder, &view, &depth);
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(size.y),
            },
        },
        extent,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    // The map callback only runs from inside a poll, so this is also what waits
    // for the frame itself.
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver
        .recv()
        .context("the buffer was never mapped")?
        .context("failed to map the readback buffer")?;

    let mapped = readback.get_mapped_range(..).context("not mapped")?;
    let mut pixels = Vec::with_capacity((packed_row * size.y) as usize);
    for row in mapped.chunks_exact(bytes_per_row as usize) {
        pixels.extend_from_slice(&row[..packed_row as usize]);
    }
    drop(mapped);
    readback.unmap();

    Ok(pixels)
}

/// Writes tightly packed RGBA8 pixels out as a PNG.
pub fn write_png(path: &Path, size: UVec2, pixels: &[u8]) -> Result<()> {
    let expected = (size.x * size.y * 4) as usize;
    if pixels.len() != expected {
        bail!(
            "{} pixel bytes for a {}x{} image, which wants {expected}",
            pixels.len(),
            size.x,
            size.y
        );
    }

    let file = std::fs::File::create(path)
        .with_context(|| format!("failed to create {}", path.display()))?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), size.x, size.y);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // [`CAPTURE_FORMAT`] already encoded these, so mark them encoded rather than
    // letting a viewer take them for linear and brighten the lot.
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    encoder
        .write_header()
        .and_then(|mut writer| writer.write_image_data(pixels))
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

/// Renders one frame of `terrain_root` and writes it to `output`.
///
/// Without a `placement` the scene's own opening view is kept, which frames the
/// whole extent.
pub fn run(
    terrain_root: &Path,
    size: UVec2,
    placement: Option<Placement>,
    output: &Path,
) -> Result<()> {
    let (device, queue) = device()?;

    // Each stage is timed on its own because they fail differently: a scene that
    // builds quickly can still stall reading tiles, and a frame that draws
    // slowly says something about the GPU that neither of the others does.
    let started = std::time::Instant::now();
    let mut scene = Scene::new(&device, CAPTURE_FORMAT, size, terrain_root)?;
    log::info!("built the scene in {:.2?}", started.elapsed());

    if let Some(placement) = placement {
        placement.apply(&mut scene.camera);
    }
    log::info!(
        "camera at {} facing {:?}",
        scene.camera.position,
        scene.camera.orientation
    );

    let started = std::time::Instant::now();
    scene.settle(&queue);
    log::info!("filled every level in {:.2?}", started.elapsed());

    let started = std::time::Instant::now();
    let pixels = capture(&device, &queue, &scene, size)?;
    log::info!("rendered one frame in {:.2?}", started.elapsed());

    write_png(output, size, &pixels)?;
    log::info!("wrote {}", output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_reads_five_numbers() {
        assert_eq!(
            "1,-2.5, 3 ,90,-15".parse::<Placement>().unwrap(),
            Placement {
                position: Vec3::new(1.0, -2.5, 3.0),
                yaw_degrees: 90.0,
                pitch_degrees: -15.0,
            }
        );
    }

    #[test]
    fn placement_rejects_the_wrong_shape() {
        assert!("1,2,3,4".parse::<Placement>().is_err());
        assert!("1,2,3,4,5,6".parse::<Placement>().is_err());
        assert!("1,2,3,4,north".parse::<Placement>().is_err());
    }
}
