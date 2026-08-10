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
use crate::scene::Scene;

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
        required_features: crate::profile::timer_features(&adapter),
        required_limits: crate::deferred::limits(&adapter.limits()),
        ..Default::default()
    }))
    .context("failed to create device")?;
    Ok((device, queue))
}

/// How far the camera is taken to move between one frame and the next, per
/// metre per second of [`Flight::speed`].
///
/// A nominal sixtieth of a second rather than the frame's own measured time.
/// The path flown has to come out the same on a fast machine and a slow one, or
/// two runs would be looking at two different flights and could not be
/// compared -- which is the only reason either mode can fly at all.
const STEP_SECONDS: f32 = 1.0 / 60.0;

/// A camera that moves, for the modes that draw more than one frame.
///
/// One frame says nothing about anything carried between frames, so both
/// headless modes can fly forward instead of standing still. The default is
/// still, which is what every run before this existed did.
#[derive(Clone, Copy, Default, Debug)]
pub struct Flight {
    /// Frames to draw. The last one is the one that counts.
    pub frames: u32,
    /// Metres per second along the camera's forward vector.
    pub speed: f32,
}

impl Flight {
    /// Moves the camera on by one frame.
    ///
    /// Called *between* frames rather than before the first, so the opening
    /// frame is always at the camera that was asked for and a single-frame run
    /// is unaffected by the speed.
    fn advance(self, scene: &mut Scene) {
        if self.speed != 0.0 {
            let forward = scene.camera.ray_basis()[2];
            scene.camera.position += forward * self.speed * STEP_SECONDS;
        }
    }
}

/// Draws `scene` into a fresh texture and returns it as tightly packed RGBA8.
///
/// `scene` must have been built with [`CAPTURE_FORMAT`] and a viewport of
/// `size` -- its G-buffer is that size and the shading pass looks pixels up
/// by coordinate -- and [`Scene::update`] must have run since the camera
/// last moved.
///
/// Draws `flight.frames` frames and reads back the last. Each gets its own
/// submit rather than several passes on one encoder, because
/// `queue.write_buffer` is ordered at submit: batched, every frame would march
/// against the last frame's camera instead of its own.
pub fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &mut Scene,
    size: UVec2,
    flight: Flight,
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

    let profiler = crate::profile::profiler(device, false);
    for _ in 1..flight.frames.max(1) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("capture"),
        });
        {
            let mut gpu = profiler.scope("gpu", &mut encoder);
            scene.draw(&mut gpu, &view);
        }
        queue.submit(std::iter::once(encoder.finish()));
        flight.advance(scene);
        scene.update(device, queue);
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("capture"),
    });
    {
        let mut gpu = profiler.scope("gpu", &mut encoder);
        scene.draw(&mut gpu, &view);
    }
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

/// Opens `terrain_root`, points the camera, and fills every level.
///
/// The shared prologue of both headless modes. Each stage is timed on its own
/// because they fail differently: a scene that builds quickly can still stall
/// reading tiles off disk.
fn settled(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    terrain_root: &Path,
    size: UVec2,
    placement: Option<Placement>,
) -> Result<Scene> {
    let started = std::time::Instant::now();
    let mut scene = Scene::new(device, CAPTURE_FORMAT, size, terrain_root)?;
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
    scene.settle(device, queue);
    log::info!("filled every level in {:.2?}", started.elapsed());
    Ok(scene)
}

/// Renders one frame of `terrain_root` and writes it to `output`.
///
/// Deliberately silent about timing. This mode exists to produce an image to
/// look at, and one cold frame is not a measurement of anything: it carries
/// first-use pipeline compilation and whatever the tile reads left behind.
/// [`profile`] is the mode that answers what it costs.
///
/// Without a `placement` the scene's own opening view is kept, which frames the
/// whole extent.
///
/// `flight` is what makes this able to show anything a frame carries over from
/// the one before it: with a single frame there is no previous frame, so the
/// reprojection has nothing to work from and the image is the march's alone.
pub fn render(
    terrain_root: &Path,
    size: UVec2,
    placement: Option<Placement>,
    flight: Flight,
    output: &Path,
) -> Result<()> {
    let (device, queue) = device()?;
    let mut scene = settled(&device, &queue, terrain_root, size, placement)?;

    let pixels = capture(&device, &queue, &mut scene, size, flight)?;
    write_png(output, size, &pixels)?;
    log::info!("wrote {}", output.display());
    Ok(())
}

/// Frames drawn and thrown away before any are counted.
///
/// The first use of a pipeline compiles it, the first use of a texture may move
/// it, and neither is what a steady frame costs. Small because after
/// [`Scene::settle`] there is nothing left to warm but the draw itself.
const WARMUP: u32 = 8;

/// Measures `frames` frames of a settled scene and prints where the time went.
///
/// Writes no image. What it draws into is a texture nobody reads back, because
/// the readback is not part of the frame being measured -- the copy and the
/// buffer map that follow it in [`capture`] cost more than the draw does.
///
/// The scene is settled first, so the tile streaming rows read near zero here:
/// nothing is pending once every level is whole. That is the point of a
/// measurement mode -- it holds the one variable that would otherwise swamp the
/// others still -- but it does mean this cannot tell you what streaming costs.
/// The windowed overlay while flying, or `FLIGHT_SIM_WALK` in
/// `dump_installed_terrain`, is what shows that.
pub fn profile(
    terrain_root: &Path,
    size: UVec2,
    placement: Option<Placement>,
    flight: Flight,
) -> Result<()> {
    let frames = flight.frames;
    let (device, queue) = device()?;
    let mut scene = settled(&device, &queue, terrain_root, size, placement)?;
    scene.profile(true);

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("profile target"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: CAPTURE_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut profiler = crate::profile::profiler(&device, true);
    let mut measured: Vec<crate::profile::Frame> = Vec::with_capacity(frames as usize);
    let mut last = std::time::Instant::now();

    for index in 0..WARMUP + frames {
        let mut frame = crate::profile::Frame::default();

        if index > 0 {
            flight.advance(&mut scene);
        }
        scene.update(&device, &queue);
        scene.record(&mut frame);

        let clock = crate::profile::Clock::start(true);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("profile frame"),
        });
        {
            let mut gpu = profiler.scope("gpu", &mut encoder);
            scene.draw(&mut gpu, &view);
        }
        profiler.resolve_queries(&mut encoder);
        frame.cpu.encode = clock.elapsed();

        let clock = crate::profile::Clock::start(true);
        queue.submit(std::iter::once(encoder.finish()));
        frame.cpu.submit = clock.elapsed();

        // Nothing presents here, so there is no vsync and nothing else to wait
        // on the GPU. Blocking makes each iteration a whole frame rather than a
        // queue of them, which is what makes the interval mean anything and
        // what lets the timestamps come back on the very next call.
        device.poll(wgpu::PollType::wait_indefinitely())?;

        let now = std::time::Instant::now();
        frame.interval = now.duration_since(last);
        last = now;

        profiler
            .end_frame()
            .context("the profiler was left with a scope open")?;
        if let Some(results) = profiler.process_finished_frame(queue.get_timestamp_period()) {
            frame.take_gpu(&results);
        }

        if index >= WARMUP {
            measured.push(frame);
        }
    }

    // The last frames' timestamps are still in flight; drain them so they are
    // not silently dropped, and attach them to the frames still missing theirs.
    for frame in measured.iter_mut().filter(|frame| frame.gpu.is_empty()) {
        device.poll(wgpu::PollType::wait_indefinitely())?;
        if let Some(results) = profiler.process_finished_frame(queue.get_timestamp_period()) {
            frame.take_gpu(&results);
        }
    }

    print!("{}", crate::profile::table(&measured));

    let coverage = coverage(&device, &queue, &mut scene, &view, flight)?;
    let pixels = size.x * size.y;
    // Every pixel takes exactly one path through the compaction, so anything
    // else means one of them stopped counting -- which would otherwise show up
    // only as three percentages that quietly fail to add up.
    if coverage.total() != pixels {
        log::warn!(
            "the compaction accounted for {} pixels of {pixels}",
            coverage.total()
        );
    }
    let share = |count: u32| 100.0 * f64::from(count) / f64::from(pixels);
    println!(
        "{pixels} pixels: {:.1}% reprojected from the last frame, {:.1}% sky, \
         {:.1}% marched",
        share(coverage.reprojected),
        share(coverage.sky),
        share(coverage.marched),
    );
    // Subsets of the marched share rather than paths of their own, and both
    // near zero unless the march is failing, so they are only worth a line when
    // there is something to say.
    if coverage.abandoned > 0 || coverage.spent > 0 {
        println!(
            "of which {} pixels were abandoned and {} ran out of steps",
            coverage.abandoned, coverage.spent,
        );
    }
    // The march covering the list it was handed is the invariant the whole
    // uncleared G-buffer rests on, and a frame it skips looks like a frame
    // rather than like a failure, so say so rather than leave it to be assumed.
    if coverage.wrote != coverage.marched {
        log::warn!(
            "the march wrote {} of the {} pixels it was handed, over {} workgroups",
            coverage.wrote,
            coverage.marched,
            coverage.groups,
        );
    }
    Ok(())
}

/// Draws one more frame and reads back how each of its pixels was settled.
///
/// After the measured run rather than during it: the copy and the buffer map
/// cost more than the frame does, and a measurement that waits on a readback is
/// measuring the readback. The extra frame is drawn on the settled, already
/// warm scene, so it is representative of the ones just timed.
///
/// One more step along the same flight, not a redraw of the last measured
/// frame. Standing still is the reprojection's best case -- every surviving
/// point lands back on the pixel it came from -- so a coverage frame that did
/// not move would report roughly the same share however fast the run it follows
/// was flying, which is the one thing this number must not do.
fn coverage(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &mut Scene,
    view: &wgpu::TextureView,
    flight: Flight,
) -> Result<crate::reproject::Coverage> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("coverage readback"),
        size: crate::reproject::Coverage::BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    flight.advance(scene);
    scene.update(device, queue);
    let profiler = crate::profile::profiler(device, false);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("coverage"),
    });
    {
        let mut gpu = profiler.scope("gpu", &mut encoder);
        scene.draw(&mut gpu, view);
    }
    encoder.copy_buffer_to_buffer(
        scene.tally(),
        0,
        &readback,
        0,
        crate::reproject::Coverage::BYTES,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let (sender, receiver) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    receiver
        .recv()
        .context("the buffer was never mapped")?
        .context("failed to map the coverage tally")?;
    let mapped = readback.get_mapped_range(..).context("not mapped")?;
    let coverage = crate::reproject::Coverage::from_bytes(&mapped);
    drop(mapped);
    readback.unmap();
    Ok(coverage)
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
