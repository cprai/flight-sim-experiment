use anyhow::Result;
use glam::UVec2;
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::terrain::clipmap::ClipmapConfig;
use crate::terrain::gpu::Terrain;

/// Sky the terrain is drawn against.
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.30,
    g: 0.55,
    b: 0.85,
    a: 1.0,
};

/// Format of the depth buffer the scene draws against.
///
/// Float depth rather than the more compact `Depth24Plus` because the camera
/// projects with reversed depth, which only pays off when the buffer's own
/// precision is concentrated near zero the way a float exponent is.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Depth value the buffer is cleared to.
///
/// Reversed depth puts the far distance at 0, so "nothing drawn yet" is 0 and
/// fragments pass when their depth is [`wgpu::CompareFunction::Greater`].
const DEPTH_CLEAR: f32 = 0.0;

/// Creates a depth buffer view sized to match a render target.
///
/// Lives here beside the pipeline state that has to agree with it, and is
/// called both by the renderer on resize and by the offscreen tests.
pub fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth buffer"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// Mirrors the `Camera` uniform block in `terrain.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    /// `w` is unused padding; uniform members are aligned to 16 bytes anyway.
    position: [f32; 4],
    /// [`Camera::ray_basis`], one vector per row, `w` unused on each.
    ///
    /// Carried alongside the matrix rather than derived from it because the
    /// raymarched far field needs a ray per pixel and inverting `view_proj` on
    /// the GPU to get one would cost far more than three vectors of uniform.
    ray_right: [f32; 4],
    ray_up: [f32; 4],
    ray_forward: [f32; 4],
}

impl CameraUniform {
    fn new(camera: &Camera) -> Self {
        let [right, up, forward] = camera.ray_basis();
        Self {
            view_proj: camera.view_projection().to_cols_array_2d(),
            position: camera.position.extend(1.0).to_array(),
            ray_right: right.extend(0.0).to_array(),
            ray_up: up.extend(0.0).to_array(),
            ray_forward: forward.extend(0.0).to_array(),
        }
    }
}

/// The terrain plus the camera looking at it, and the GPU state to draw them.
pub struct Scene {
    pub camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    terrain: Terrain,
}

impl Scene {
    /// Opens the terrain tile pyramid and frames the camera on it.
    ///
    /// `viewport` is the target's size in pixels, not merely its aspect: how
    /// much ground the clipmap keeps resident at each level is chosen so that
    /// a texel of it lands on about a pixel, and that needs the pixel count.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        viewport: UVec2,
        terrain_root: &std::path::Path,
    ) -> Result<Self> {
        let mut config = ClipmapConfig {
            pixel_angle: crate::terrain::clipmap::pixel_angle(
                viewport.y,
                f64::from(crate::camera::FOV_Y_DEGREES).to_radians(),
            ),
            ..ClipmapConfig::default()
        };
        config.window_texels = config.window_for();
        Self::with_config(device, format, viewport, terrain_root, config)
    }

    /// As [`Scene::new`], but over a clipmap configured by the caller.
    ///
    /// Only [`dump_installed_terrain`] uses this, to time one view against
    /// several shapes of clipmap; the application takes the default.
    pub fn with_config(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        viewport: UVec2,
        terrain_root: &std::path::Path,
        config: ClipmapConfig,
    ) -> Result<Self> {
        let (camera_buffer, camera_layout, camera_bind_group) = camera_binding(device);
        let terrain = Terrain::from_tiles(device, format, &camera_layout, config, terrain_root)?;
        Ok(Self::assemble(
            camera_buffer,
            camera_bind_group,
            terrain,
            viewport.x as f32 / viewport.y.max(1) as f32,
        ))
    }

    /// Frames the camera on an already-built terrain.
    ///
    /// Kept separate from [`Scene::new`] so tests can supply rasters directly
    /// instead of depending on files that are not in version control.
    #[cfg(test)]
    pub fn from_terrain(
        device: &wgpu::Device,
        terrain: impl FnOnce(&wgpu::BindGroupLayout) -> Terrain,
        aspect: f32,
    ) -> Self {
        let (camera_buffer, camera_layout, camera_bind_group) = camera_binding(device);
        let terrain = terrain(&camera_layout);
        Self::assemble(camera_buffer, camera_bind_group, terrain, aspect)
    }

    fn assemble(
        camera_buffer: wgpu::Buffer,
        camera_bind_group: wgpu::BindGroup,
        terrain: Terrain,
        aspect: f32,
    ) -> Self {
        let camera = Camera::overlooking(terrain.world_extent(), terrain.height_range().1, aspect);
        Self {
            camera,
            camera_buffer,
            camera_bind_group,
            terrain,
        }
    }

    /// Uploads the current camera and moves the clipmap to follow it.
    ///
    /// Call once per frame before [`Scene::draw`].
    pub fn update(&mut self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::new(&self.camera)),
        );
        self.terrain.update(queue, self.camera.position);
    }

    /// Records a sky clear plus the terrain into `view`.
    ///
    /// `depth` must be a [`DEPTH_FORMAT`] view matching `view`'s dimensions; see
    /// [`create_depth_view`].
    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(DEPTH_CLEAR),
                    // Nothing reads the depth buffer after the pass yet, but
                    // discarding it would break as soon as anything does.
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        self.terrain.draw(&mut pass);
        drop(pass);

        // The far field goes in a second pass loading what the first left, not
        // in more draws at the end of it, because a pipeline's depth state is
        // fixed and this one writes its depth from the fragment stage. Loading
        // the depth buffer is the whole mechanism: the near field is already in
        // it, so the two sort against each other with nothing else to arrange.
        let mut far = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("far terrain pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        far.set_bind_group(0, &self.camera_bind_group, &[]);
        self.terrain.draw_far(&mut far);
    }
}

/// The camera uniform, its layout, and the bind group tying them together.
///
/// The layout is handed to the terrain pipeline as well, so both agree on what
/// group 0 holds.
fn camera_binding(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera uniform"),
        contents: bytemuck::bytes_of(&CameraUniform {
            view_proj: [[0.0; 4]; 4],
            position: [0.0; 4],
            ray_right: [0.0; 4],
            ray_up: [0.0; 4],
            ray_forward: [0.0; 4],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera bind group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (buffer, layout, bind_group)
}

/// A headless device and queue for the offscreen tests.
///
/// Requests exactly what [`crate::renderer::Renderer`] does, so a test passing
/// here is evidence the application will run on the same baseline rather than on
/// whatever the test machine happens to offer.
#[cfg(test)]
pub fn test_device() -> (wgpu::Device, wgpu::Queue) {
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no wgpu adapter available");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("failed to create device")
}

/// The camera bind group layout alone, for tests that build a terrain without a
/// whole scene around it. The real one, so group 0 cannot drift out of step.
#[cfg(test)]
pub fn test_camera_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    camera_binding(device).1
}

/// Where a world point lands on screen, in pixels, with (0, 0) at the top left.
///
/// Only used by tests, but it belongs next to the projection it inverts.
#[cfg(test)]
fn to_pixels(view_proj: glam::Mat4, point: glam::Vec3, width: u32, height: u32) -> (f32, f32) {
    let clip = view_proj * point.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    (
        (ndc.x + 1.0) * 0.5 * width as f32,
        (1.0 - ndc.y) * 0.5 * height as f32,
    )
}

#[cfg(test)]
mod tests {
    use glam::{IVec2, UVec2, Vec2, Vec3};

    use super::*;
    use crate::terrain::geotiff::Georeferencing;
    use crate::terrain::pyramid::{Level, Pyramid, RasterSource, Srgb8};

    /// Side of the offscreen render target.
    const SIZE: u32 = 256;
    /// Side of the synthetic rasters, in texels.
    const RASTER: u32 = 128;
    const METRES_PER_TEXEL: f64 = 30.0;

    /// A deliberately small clipmap, so the software rasterizer stays quick.
    fn test_config() -> ClipmapConfig {
        ClipmapConfig {
            block_verts: 16,
            window_texels: 64,
            // A far coarser pixel than any real viewport, because the rule for
            // giving up a level compares its texels to one. This raster's are
            // thirty metres, which a 256-pixel frame still resolves from
            // thirteen kilometres up -- four times further than the raster is
            // wide, so no camera that can see it would ever drop a level and
            // the tests for dropping one would have nothing to say. Three and a
            // half degrees a pixel puts the handover at a kilometre instead,
            // which the altitudes below fly through.
            pixel_angle: 0.06,
            ..Default::default()
        }
    }

    fn placement() -> Georeferencing {
        Georeferencing::square(RASTER, RASTER, METRES_PER_TEXEL)
    }

    const GREEN: Srgb8 = Srgb8([60, 140, 50, 255]);
    const RED: Srgb8 = Srgb8([220, 30, 30, 255]);
    /// Deliberately unlike the sky, the ground and the red: nothing else in a
    /// frame has a high red *and* blue with almost no green.
    const MAGENTA: Srgb8 = Srgb8([220, 20, 220, 255]);

    fn is_magenta([r, g, b, _]: [u8; 4]) -> bool {
        r > 100 && b > 100 && g < 80
    }

    fn flat_ground() -> Vec<Srgb8> {
        vec![GREEN; (RASTER * RASTER) as usize]
    }

    /// Builds terrain from raw texels and renders one frame of it.
    fn render(heights: Vec<f32>, colours: Vec<Srgb8>, aim: impl FnOnce(&mut Camera)) -> Vec<u8> {
        render_after(heights, colours, aim, &[])
    }

    /// As [`render`], but stepping the camera through `path` first so the
    /// clipmap has to update incrementally before the frame that is captured.
    fn render_after(
        heights: Vec<f32>,
        colours: Vec<Srgb8>,
        aim: impl FnOnce(&mut Camera),
        path: &[Vec3],
    ) -> Vec<u8> {
        render_probed(heights, colours, aim, path).0
    }

    /// As [`render_after`], but also reporting the base level the clipmap chose.
    ///
    /// The base level is how much detail the camera's height above the ground
    /// bought: everything below it was dropped. A test that means to look at
    /// more than one level has to say so, because a camera high enough leaves
    /// only the coarsest and the test would pass on an empty promise.
    fn render_probed(
        heights: Vec<f32>,
        colours: Vec<Srgb8>,
        aim: impl FnOnce(&mut Camera),
        path: &[Vec3],
    ) -> (Vec<u8>, u32) {
        render_config(test_config(), heights, colours, aim, path)
    }

    /// The same clipmap with room around its grid.
    ///
    /// The mesh draws the same rings either way; the extra texels are there for
    /// the far field, which reads whatever is resident rather than only what the
    /// mesh covers. Anything the near field draws has to come out the same.
    fn wide_config() -> ClipmapConfig {
        ClipmapConfig {
            window_texels: 128,
            ..test_config()
        }
    }

    /// The clipmap of [`test_config`] cut at a given radius.
    ///
    /// Infinity rasterizes the whole frame and zero raymarches it, which is how
    /// the two halves of the renderer are held against each other.
    fn cut_at(near_rings: f32) -> ClipmapConfig {
        ClipmapConfig {
            near_rings,
            ..test_config()
        }
    }

    /// As [`render_probed`], but over a clipmap configured by the caller.
    fn render_config(
        config: ClipmapConfig,
        heights: Vec<f32>,
        colours: Vec<Srgb8>,
        aim: impl FnOnce(&mut Camera),
        path: &[Vec3],
    ) -> (Vec<u8>, u32) {
        let (device, queue) = test_device();

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = create_depth_view(&device, SIZE, SIZE);

        let mut scene = Scene::from_terrain(
            &device,
            |camera_layout| {
                Terrain::new(
                    &device,
                    format,
                    camera_layout,
                    config,
                    placement(),
                    Box::new(Pyramid::build(Level::new(RASTER, RASTER, heights))),
                    Box::new(Pyramid::build(Level::new(RASTER, RASTER, colours))),
                )
            },
            1.0,
        );
        aim(&mut scene.camera);

        // Walk the requested path first, so the windows arrive at the captured
        // frame through a series of incremental updates.
        let destination = scene.camera.position;
        for step in path {
            scene.camera.position = *step;
            scene.update(&queue);
        }
        scene.camera.position = destination;
        scene.update(&queue);

        // `SIZE * 4` is already a multiple of the 256-byte copy alignment.
        let bytes_per_row = SIZE * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        scene.draw(&mut encoder, &view, &depth);
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");

        let pixels = readback
            .get_mapped_range(..)
            .expect("buffer not mapped")
            .to_vec();
        readback.unmap();
        (pixels, scene.terrain.base_level())
    }

    fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i..i + 4].try_into().unwrap()
    }

    fn is_sky([r, g, b, _]: [u8; 4]) -> bool {
        b > r && b > g
    }

    /// The exact bytes a pixel nothing drew over is left holding.
    ///
    /// Stricter than [`is_sky`], and it has to be for counting holes: the
    /// imagery has water in it, which is bluer than it is red or green, so a
    /// test on the channels alone finds every lake and river as well.
    fn untouched(pixel: [u8; 4]) -> bool {
        let clear = [CLEAR_COLOR.r, CLEAR_COLOR.g, CLEAR_COLOR.b]
            .map(|channel| terrain_tiles::linear_to_srgb(channel as f32));
        pixel[..3] == clear
    }

    /// Pixels nothing drew that have ground both above and below them.
    ///
    /// Sky above a ridge is honest; sky enclosed by ground is a ray that should
    /// have found something and did not.
    fn holes(pixels: &[u8]) -> Vec<(u32, u32)> {
        (0..SIZE)
            .flat_map(|x| {
                let drawn: Vec<bool> = (0..SIZE).map(|y| !untouched(pixel(pixels, x, y))).collect();
                (0..SIZE)
                    .filter(|&y| {
                        !drawn[y as usize]
                            && drawn[..y as usize].iter().any(|hit| *hit)
                            && drawn[y as usize..].iter().any(|hit| *hit)
                    })
                    .map(move |y| (x, y))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Looks straight down from high enough to see most of the raster.
    fn straight_down(camera: &mut Camera) {
        camera.position = Vec3::new(0.0, 3000.0, 0.0);
        camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
    }

    /// World position of the centre of a raster texel, on the ground.
    fn world_of(col: f64, row: f64) -> Vec3 {
        let (x, z) = placement().world_of_texel(0, col, row);
        Vec3::new(x as f32, 0.0, z as f32)
    }

    /// How many pixels of a frame are sky.
    fn count_sky(pixels: &[u8]) -> usize {
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| is_sky(pixel(pixels, x, y)))
            .count()
    }

    /// Mean absolute difference between two frames, per colour byte.
    fn mean_difference(a: &[u8], b: &[u8]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / a.len() as f64
    }

    /// An oblique view from low enough that the mesh is not blending vertically.
    ///
    /// `detail_base` starts blending the finest level into the one outside it as
    /// soon as a pixel covers more than one of its texels -- 500 m above the
    /// ground for this raster and this test's deliberately coarse pixel -- and
    /// the march does not reproduce that blend. Six hundred metres over ground
    /// standing at about 180 leaves it at zero, so a comparison from here is
    /// measuring the traversal rather than measuring a mismatch that is already
    /// known and accepted.
    fn low_and_looking_out(camera: &mut Camera) {
        camera.position = Vec3::new(70.0, 600.0, -110.0);
        camera.orientation = Camera::from_yaw_pitch_roll(0.0, -20f32.to_radians(), 0.0);
    }

    /// A ray that runs out of steps must not leave a hole in a ridge.
    ///
    /// The expensive ray in any maximum-mipmap traversal is the one running
    /// along a slope just above the surface: too close to skip a cell, too far
    /// to hit one. A grazing view is made of them, and a whole column of pixels
    /// can be doing it at once, so when the budget was too small for the window
    /// the failure was not a scattering of pinholes but vertical bands of sky
    /// through solid ground.
    ///
    /// Two things stop it and this checks the pair: a budget that scales with
    /// the traversal it bounds, and a march that reports where it had got to
    /// rather than reporting sky when the budget does run out.
    #[test]
    fn a_grazing_ray_never_leaves_a_hole_in_the_ground() {
        let (heights, colours) = rugged();
        let grazing = |camera: &mut Camera| {
            camera.position = Vec3::new(-1500.0, 400.0, -1500.0);
            camera.orientation =
                Camera::from_yaw_pitch_roll(45f32.to_radians(), -1.5f32.to_radians(), 0.0);
        };
        let frame = |cells: u32| {
            render_config(
                ClipmapConfig {
                    // Nothing rasterized, so every pixel of ground is a ray's.
                    near_rings: 0.0,
                    march_cells: cells,
                    ..wide_config()
                },
                heights.clone(),
                colours.clone(),
                grazing,
                &[],
            )
            .0
        };

        // A budget far below what the traversal needs, so that rays really do
        // run out and what is being looked at is what happens when they do.
        // This raster is too small to exhaust the shipped budget from any
        // camera, which is why the starving is deliberate rather than hoped
        // for: without it the test would pass on an empty promise.
        let starved = frame(3);
        let holes = holes(&starved);
        assert!(
            holes.is_empty(),
            "{} pixels of ground came out as sky, first at {:?}",
            holes.len(),
            holes.first()
        );

        // ... and where it had got to is close enough to where it was going
        // that the picture barely notices.
        let whole = frame(ClipmapConfig::default().march_cells);
        let difference = mean_difference(&starved, &whole);
        assert!(
            difference < 3.0,
            "giving up early moved the frame by {difference:.2} of 255"
        );
    }

    /// A wider window is the only thing that buys the far field more detail.
    ///
    /// The whole arrangement rests on this and nothing else measures it. The
    /// ground is flat and painted in a one-texel check, so a ray's hit position
    /// is identical either way and the only thing that can differ is which
    /// level's colours it reads there. Coarse levels are box filters, so the
    /// check averages towards a flat wash; a window that keeps finer levels
    /// resident further out reads it before it has been averaged away.
    ///
    /// Measured as the contrast between horizontally adjacent pixels, which is
    /// what a check surviving looks like and what a wash does not.
    #[test]
    fn a_wider_window_reads_finer_ground_at_the_same_distance() {
        let check: Vec<Srgb8> = (0..RASTER * RASTER)
            .map(|index| {
                let (x, y) = (index % RASTER, index / RASTER);
                if (x + y) % 2 == 0 { RED } else { GREEN }
            })
            .collect();

        let contrast = |config: ClipmapConfig| {
            let (pixels, _) = render_config(
                ClipmapConfig {
                    // Nothing rasterized, so every pixel of ground was found by
                    // a ray and the comparison is of the march alone.
                    near_rings: 0.0,
                    ..config
                },
                vec![0.0; (RASTER * RASTER) as usize],
                check.clone(),
                low_and_looking_out,
                &[],
            );
            let mut total = 0u64;
            let mut ground = 0u64;
            for y in 0..SIZE {
                for x in 0..SIZE - 1 {
                    let (here, next) = (pixel(&pixels, x, y), pixel(&pixels, x + 1, y));
                    if is_sky(here) || is_sky(next) {
                        continue;
                    }
                    total += u64::from(here[0].abs_diff(next[0]));
                    ground += 1;
                }
            }
            assert!(ground > 10_000, "only {ground} pixels of ground to measure");
            total as f64 / ground as f64
        };

        let narrow = contrast(test_config());
        // Four times the width, so that level zero's reach covers the whole of
        // the raster this camera can see rather than only the near half of it.
        let wide = contrast(ClipmapConfig {
            window_texels: 256,
            ..test_config()
        });
        assert!(
            wide > narrow * 1.8,
            "widening the window resolved {wide:.2} of contrast against {narrow:.2}, \
             which is not the detail it was supposed to buy"
        );
    }

    /// The far field on its own draws the ground the mesh would have drawn.
    ///
    /// The strongest statement available about the traversal: with the radius at
    /// zero the mesh draws nothing at all and every lit pixel in the frame was
    /// found by a ray, so holding it against the fully rasterized frame compares
    /// the two halves of the renderer directly. They agree because they read the
    /// same data at the same level -- a point belongs to the finest level whose
    /// grid contains it, which is exactly the level whose ring the mesh would
    /// have used.
    ///
    /// What is left over is the ring blend. The mesh fades each ring's outer
    /// quarter into the level outside it and the march reads each level's texels
    /// as they are, so the two disagree across those bands and nowhere else.
    /// That is the accepted mismatch, not a defect being tolerated: the bound
    /// below is set just above where it currently sits, so if it ever grows --
    /// or if the traversal starts finding a different surface altogether -- this
    /// fails.
    #[test]
    fn raymarching_the_whole_frame_matches_rasterizing_it() {
        let (heights, colours) = rugged_painted();
        let (rastered, base) = render_config(
            cut_at(f32::INFINITY),
            heights.clone(),
            colours.clone(),
            low_and_looking_out,
            &[],
        );
        assert_eq!(base, 0, "the camera has to be low enough not to blend");
        let marched = render_config(cut_at(0.0), heights, colours, low_and_looking_out, &[]).0;

        // Guard against the happy case where both frames are empty sky and any
        // comparison between them passes.
        let (sky, marched_sky) = (count_sky(&rastered), count_sky(&marched));
        let pixels = (SIZE * SIZE) as usize;
        assert!(
            pixels - marched_sky > pixels / 4,
            "the marched frame should hold real terrain, got {marched_sky} sky pixels"
        );

        // The horizon has to land in the same place, which a mean cannot say:
        // a few hundred pixels of sky where there should be terrain barely move
        // one, and that is what a hole in the acceleration structure looks like.
        assert!(
            marched_sky.abs_diff(sky) * 50 < pixels,
            "sky covers {marched_sky} pixels marched against {sky} rasterized"
        );

        let difference = mean_difference(&marched, &rastered);
        assert!(
            difference < 5.0,
            "the two halves should draw the same ground, mean |difference| {difference:.3}"
        );
    }

    /// Across the range it is actually used at, the radius costs nothing.
    ///
    /// Which level covers a point does not depend on where the cut falls, so
    /// moving it changes how a frame was computed rather than what it shows --
    /// and that is what makes the radius safe to tune by frame time alone. The
    /// bound here is tight, a fiftieth of what the previous test allows, because
    /// at these radii the mesh still owns the ring blend bands and the two halves
    /// have nothing left to disagree about.
    ///
    /// It stops at the shipped default rather than sweeping to zero. Below it the
    /// march takes over ground the mesh was still blending, and the disagreement
    /// climbs to the several units the previous test measures; that is the
    /// mismatch's shape, and pinning it here as well would only say it twice.
    #[test]
    fn no_choice_of_near_radius_changes_what_the_frame_shows() {
        let (heights, colours) = rugged_painted();
        let rastered = render_config(
            cut_at(f32::INFINITY),
            heights.clone(),
            colours.clone(),
            low_and_looking_out,
            &[],
        )
        .0;
        let sky = count_sky(&rastered);

        for rings in [32.0, 16.0, 8.0, ClipmapConfig::default().near_rings] {
            let frame = render_config(
                cut_at(rings),
                heights.clone(),
                colours.clone(),
                low_and_looking_out,
                &[],
            )
            .0;
            let difference = mean_difference(&frame, &rastered);
            assert!(
                difference < 0.1,
                "cutting at {rings} rings moved the frame by {difference:.3}"
            );
            // A gap at the join, or a ray slipping between two cells, both show
            // up here as sky that the mesh did not put there.
            assert!(
                count_sky(&frame).abs_diff(sky) * 500 < (SIZE * SIZE) as usize,
                "cutting at {rings} rings left {} sky pixels against {sky}",
                count_sky(&frame)
            );
        }
    }

    /// Looking straight down at rough ground, a ray must find it.
    ///
    /// The failure this is here for is a max pyramid whose cells bound only the
    /// samples at their corners rather than the ground between them: rays then
    /// slip through ridges and the frame fills with pinholes of sky.
    #[test]
    fn the_far_field_does_not_let_pinholes_of_sky_through() {
        let (heights, _) = rugged();
        let rastered = render_config(
            cut_at(f32::INFINITY),
            heights.clone(),
            flat_ground(),
            straight_down,
            &[],
        )
        .0;
        let marched = render_config(cut_at(0.0), heights, flat_ground(), straight_down, &[]).0;

        // Not zero either way: the frame's corners reach past the raster, and
        // that ground is cut by both halves alike. What matters is that marching
        // does not add to it.
        let (sky, marched_sky) = (count_sky(&rastered), count_sky(&marched));
        assert!(
            marched_sky < sky + 200,
            "marching showed {marched_sky} sky pixels where the mesh showed {sky}"
        );
    }

    #[test]
    fn the_opening_view_looks_out_over_terrain_under_sky() {
        let pixels = render(vec![0.0; (RASTER * RASTER) as usize], flat_ground(), |_| {});

        let sky = pixel(&pixels, SIZE / 2, 4);
        assert_eq!(sky[3], 255, "sky should be opaque");
        assert!(is_sky(sky), "top of frame should be sky, got {sky:?}");

        let ground = pixel(&pixels, SIZE / 2, SIZE - 4);
        assert!(
            !is_sky(ground),
            "bottom of frame should be ground, got {ground:?}"
        );
        assert!(
            ground[1] > ground[0] && ground[1] > ground[2],
            "ground should show the colour raster, got {ground:?}"
        );
    }

    /// Run against a window that just fits the grid and one with room to spare.
    ///
    /// Registration is what a margin could break: the vertex stage offsets grid
    /// coordinates into window coordinates before reading either texture, so a
    /// margin applied to the heights and not to the colours -- or to either and
    /// not to the world position -- would slide the imagery off the ground it
    /// belongs to. The wide window puts thirty-two texels between the grid and
    /// the window's edge, so any such slip is far larger than a pixel.
    #[test]
    fn the_colour_raster_lands_where_the_georeferencing_puts_it() {
        for config in [test_config(), wide_config()] {
            // A patch of a distinct colour, well away from the raster's centre
            // so that getting the axes or the origin wrong would move it
            // visibly.
            let (patch_col, patch_row) = (32u32, 96u32);
            let half = 8u32;
            let mut colours = flat_ground();
            for row in patch_row - half..patch_row + half {
                for col in patch_col - half..patch_col + half {
                    colours[(row * RASTER + col) as usize] = RED;
                }
            }

            let mut camera = None;
            let (pixels, _) = render_config(
                config,
                vec![0.0; (RASTER * RASTER) as usize],
                colours,
                |c| {
                    straight_down(c);
                    camera = Some(*c);
                },
                &[],
            );
            let camera = camera.expect("camera captured");
            let window = config.window_texels;

            let centre = world_of(f64::from(patch_col), f64::from(patch_row));
            let (x, y) = to_pixels(camera.view_projection(), centre, SIZE, SIZE);
            let found = pixel(&pixels, x.round() as u32, y.round() as u32);

            assert!(
                found[0] > found[1] + 40 && found[0] > found[2] + 40,
                "window {window}: expected the red patch at ({x:.0}, {y:.0}), got {found:?}"
            );

            // ... and the rest of the ground is still the background colour, so
            // the patch has not simply been smeared over everything.
            let elsewhere = world_of(f64::from(patch_col), f64::from(RASTER - patch_row));
            let (x, y) = to_pixels(camera.view_projection(), elsewhere, SIZE, SIZE);
            let found = pixel(&pixels, x.round() as u32, y.round() as u32);
            assert!(
                found[1] > found[0],
                "window {window}: expected background at ({x:.0}, {y:.0}), got {found:?}"
            );
        }
    }

    /// Tiles with nothing under them are never written, so a survey's ragged
    /// edge arrives as nodata in the middle of the raster rather than only at
    /// its border. Without the shader's test those texels would draw as a pit
    /// thirty kilometres deep; with it the sky shows through instead.
    #[test]
    fn a_hole_in_the_middle_of_the_data_is_cut_out_rather_than_drawn() {
        const NODATA: f32 = -32767.0;

        let with_hole = |hole: bool| {
            let mut heights = vec![0.0f32; (RASTER * RASTER) as usize];
            if hole {
                for row in 56..72 {
                    for col in 56..72 {
                        heights[(row * RASTER + col) as usize] = NODATA;
                    }
                }
            }
            heights
        };
        let count_sky = |pixels: &[u8]| {
            (0..SIZE)
                .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
                .filter(|&(x, y)| is_sky(pixel(pixels, x, y)))
                .count()
        };

        let solid = count_sky(&render(with_hole(false), flat_ground(), straight_down));
        assert_eq!(
            solid, 0,
            "looking straight down at unbroken ground should show no sky"
        );

        let punched = count_sky(&render(with_hole(true), flat_ground(), straight_down));
        assert!(
            punched > 200,
            "the hole should show sky through it, got {punched} pixels"
        );
    }

    #[test]
    fn a_near_ridge_hides_what_is_behind_it() {
        // Two plateaus across the view. The far one is low enough that the
        // camera looks down onto its top; the near one is tall enough to block
        // the line of sight to it entirely.
        let ridges = |near: bool| {
            let mut heights = vec![0.0f32; (RASTER * RASTER) as usize];
            let mut colours = flat_ground();
            for row in 0..RASTER {
                let (height, colour) = match row {
                    66..=73 if near => (900.0, RED),
                    46..=53 => (250.0, MAGENTA),
                    _ => continue,
                };
                for col in 0..RASTER {
                    heights[(row * RASTER + col) as usize] = height;
                    colours[(row * RASTER + col) as usize] = colour;
                }
            }
            (heights, colours)
        };

        let aim = |camera: &mut Camera| {
            camera.position = Vec3::new(0.0, 400.0, world_of(64.0, 76.0).z + 400.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -10f32.to_radians(), 0.0);
        };
        let count_far = |pixels: &[u8]| {
            (0..SIZE)
                .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
                .filter(|&(x, y)| is_magenta(pixel(pixels, x, y)))
                .count()
        };

        let (heights, colours) = ridges(false);
        let alone = count_far(&render(heights, colours, aim));
        assert!(
            alone > 500,
            "the far plateau should be plainly in shot on its own, got {alone} pixels"
        );

        let (heights, colours) = ridges(true);
        let occluded = count_far(&render(heights, colours, aim));
        assert_eq!(
            occluded, 0,
            "the near ridge should have depth-rejected every fragment behind it"
        );
    }

    /// The same two plateaus, found by rays rather than drawn as triangles.
    ///
    /// Exercises occlusion inside the traversal: with the radius at zero the
    /// depth buffer has nothing in it to reject the far plateau, so the only
    /// thing that can hide it is the march stopping at the near ridge first.
    #[test]
    fn a_near_ridge_hides_what_is_behind_it_in_the_far_field() {
        let ridges = |near: bool| {
            let mut heights = vec![0.0f32; (RASTER * RASTER) as usize];
            let mut colours = flat_ground();
            for row in 0..RASTER {
                let (height, colour) = match row {
                    66..=73 if near => (900.0, RED),
                    46..=53 => (250.0, MAGENTA),
                    _ => continue,
                };
                for col in 0..RASTER {
                    heights[(row * RASTER + col) as usize] = height;
                    colours[(row * RASTER + col) as usize] = colour;
                }
            }
            (heights, colours)
        };
        let aim = |camera: &mut Camera| {
            camera.position = Vec3::new(0.0, 400.0, world_of(64.0, 76.0).z + 400.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -10f32.to_radians(), 0.0);
        };
        let count_far = |pixels: &[u8]| {
            (0..SIZE)
                .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
                .filter(|&(x, y)| is_magenta(pixel(pixels, x, y)))
                .count()
        };

        let (heights, colours) = ridges(false);
        let alone = count_far(&render_config(cut_at(0.0), heights, colours, aim, &[]).0);
        assert!(
            alone > 500,
            "the far plateau should be plainly in shot on its own, got {alone} pixels"
        );

        let (heights, colours) = ridges(true);
        let occluded = count_far(&render_config(cut_at(0.0), heights, colours, aim, &[]).0);
        assert_eq!(
            occluded, 0,
            "every ray should have stopped at the near ridge"
        );
    }

    /// Nodata and the edge of the raster are holes to a ray as much as to a
    /// triangle, and for the same reason: there is no ground there to draw.
    #[test]
    fn the_far_field_cuts_holes_and_the_data_edge_out_too() {
        const NODATA: f32 = -32767.0;

        let with_hole = |hole: bool| {
            let mut heights = vec![0.0f32; (RASTER * RASTER) as usize];
            if hole {
                for row in 56..72 {
                    for col in 56..72 {
                        heights[(row * RASTER + col) as usize] = NODATA;
                    }
                }
            }
            heights
        };

        let solid = count_sky(
            &render_config(
                cut_at(0.0),
                with_hole(false),
                flat_ground(),
                straight_down,
                &[],
            )
            .0,
        );
        assert_eq!(solid, 0, "unbroken ground should show no sky");

        let punched = count_sky(
            &render_config(
                cut_at(0.0),
                with_hole(true),
                flat_ground(),
                straight_down,
                &[],
            )
            .0,
        );
        // Sized, not merely present. The hole is sixteen texels of 30 m, so
        // 480 m across; from 3000 m up, over a frame spanning 3464 m in 256
        // pixels, it projects to about 35 pixels a side and so 1250 of them. A
        // ray refuses the whole quad it is standing in whenever any corner is
        // nodata, which at the level this is marched at widens the cut by one
        // 120 m quad on each side, to about 44 pixels a side.
        //
        // The bound matters because both ways of getting this wrong land inside
        // a loose one. Cutting nothing but the exact quads leaves the ground
        // closing back over the hole, and cutting whatever the ray met after
        // dropping through it -- which is what a hit reported from under the
        // surface amounts to -- shrank this to 49 pixels.
        assert!(
            (1200..2600).contains(&punched),
            "the hole should show sky through it, got {punched} pixels"
        );

        // Climbing until the raster no longer fills the frame puts its edge in
        // shot. Rings reach past it and reads out there repeat the border texel,
        // so a march that did not cut at the data bounds would draw a plateau of
        // invented ground rather than sky.
        let beyond = count_sky(
            &render_config(
                cut_at(0.0),
                with_hole(false),
                flat_ground(),
                |camera| {
                    camera.position = Vec3::new(0.0, 6000.0, 0.0);
                    camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
                },
                &[],
            )
            .0,
        );
        assert!(
            beyond > 2000,
            "the ground should stop at the raster's edge, got {beyond} sky pixels"
        );
    }

    #[test]
    fn walking_the_camera_there_looks_the_same_as_arriving_directly() {
        // The incremental toroidal update path is only correct if it agrees
        // with the trivially-correct full refresh. Walk far enough that every
        // level's window moves and the finest wraps around its texture.
        let heights: Vec<f32> = (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as f32, (i / RASTER) as f32);
                120.0 * ((x * 0.21).sin() + (y * 0.17).cos()) + 60.0 * (x * 0.05 + y * 0.03).sin()
            })
            .collect();
        let colours: Vec<Srgb8> = (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as u8, (i / RASTER) as u8);
                Srgb8([x.wrapping_mul(3), y.wrapping_mul(5), x ^ y, 255])
            })
            .collect();

        let aim = |camera: &mut Camera| {
            camera.position = Vec3::new(400.0, 900.0, 300.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -30f32.to_radians(), 0.0);
        };

        let direct = render(heights.clone(), colours.clone(), aim);

        let steps: Vec<Vec3> = (0..200)
            .map(|i| {
                let t = f32::from(i as u16);
                Vec3::new(-1400.0 + t * 9.0, 900.0, 1500.0 - t * 6.0)
            })
            .collect();
        let walked = render_after(heights, colours, aim, &steps);

        assert_eq!(
            direct, walked,
            "incremental clipmap updates diverged from a full refresh"
        );
    }

    /// Renders a real tile pyramid and writes the frame out.
    ///
    /// Ignored because no pyramid is in version control -- one covering a few
    /// kilometres is hundreds of megabytes -- and because this is a look-at-it
    /// check rather than an assertion. Run it with
    /// `FLIGHT_SIM_TERRAIN=/tmp/terrain cargo test --release -- --ignored dump_installed`
    /// and open the file.
    ///
    /// `FLIGHT_SIM_CAMERA` overrides the opening view, as
    /// `x,y,z,yaw,pitch` -- position in metres from the pyramid's centre, then
    /// two angles in degrees. Without it the scene's own opening camera is
    /// used, which frames the whole extent and therefore looks at whatever is
    /// most of the box. That is the wrong tool for checking one corner of it:
    /// a change confined to ground the default view does not reach renders
    /// byte-identical frames and looks like it did nothing.
    ///
    /// `FLIGHT_SIM_NEAR_RINGS` overrides [`ClipmapConfig::near_rings`], so the
    /// same view can be timed and dumped with the near field cut at different
    /// radii -- including infinity, which rasterizes the lot, and zero, which
    /// raymarches it. That comparison is the only way to choose the default.
    #[test]
    #[ignore = "requires a tile pyramid, which is not in version control"]
    fn dump_installed_terrain() {
        const WIDE: u32 = 960;
        const TALL: u32 = 540;

        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("preview"),
            size: wgpu::Extent3d {
                width: WIDE,
                height: TALL,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let depth = create_depth_view(&device, WIDE, TALL);

        // The clipmap reports what it chose and what that costs through `log`,
        // which is most of what this test exists to read.
        let _ = env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or("warn,flight_sim=info"),
        )
        .try_init();

        let started = std::time::Instant::now();
        let root = std::path::PathBuf::from(
            std::env::var("FLIGHT_SIM_TERRAIN")
                .expect("set FLIGHT_SIM_TERRAIN to a directory terrain-download wrote"),
        );
        let mut config = ClipmapConfig {
            pixel_angle: crate::terrain::clipmap::pixel_angle(
                TALL,
                f64::from(crate::camera::FOV_Y_DEGREES).to_radians(),
            ),
            ..ClipmapConfig::default()
        };
        config.window_texels = config.window_for();
        if let Ok(rings) = std::env::var("FLIGHT_SIM_NEAR_RINGS") {
            config.near_rings = rings
                .parse()
                .expect("FLIGHT_SIM_NEAR_RINGS must be a number");
        }
        // Overriding the window is how the detail this whole arrangement buys
        // is measured against what it costs, which is why it is a knob here and
        // nowhere else.
        if let Ok(window) = std::env::var("FLIGHT_SIM_WINDOW") {
            config.window_texels = window
                .parse()
                .expect("FLIGHT_SIM_WINDOW must be a power of two");
        }
        eprintln!(
            "rasterizing out to {} ring reaches, windows of {} texels",
            config.near_rings, config.window_texels
        );
        let mut scene = Scene::with_config(&device, format, UVec2::new(WIDE, TALL), &root, config)
            .expect("failed to open the terrain pyramid");
        eprintln!("built the scene in {:.2?}", started.elapsed());

        if let Ok(aim) = std::env::var("FLIGHT_SIM_CAMERA") {
            let n: Vec<f32> = aim
                .split(',')
                .map(|p| p.trim().parse().expect("FLIGHT_SIM_CAMERA wants numbers"))
                .collect();
            assert_eq!(n.len(), 5, "FLIGHT_SIM_CAMERA wants x,y,z,yaw,pitch");
            scene.camera.position = Vec3::new(n[0], n[1], n[2]);
            scene.camera.orientation =
                Camera::from_yaw_pitch_roll(n[3].to_radians(), n[4].to_radians(), 0.0);
        }

        eprintln!(
            "camera at {} facing {:?}",
            scene.camera.position, scene.camera.orientation
        );
        // Timed separately from the frame below it, because this is where the
        // tile reads and the pyramid reductions happen: a frame that draws
        // quickly can still stall here, and the two want telling apart.
        let started = std::time::Instant::now();
        if let Ok(walk) = std::env::var("FLIGHT_SIM_WALK") {
            let steps: u32 = walk.parse().expect("FLIGHT_SIM_WALK wants a count");
            let home = scene.camera.position;
            scene.camera.position = home - Vec3::new(steps as f32, 0.0, steps as f32);
            for _ in 0..steps {
                scene.camera.position += Vec3::new(1.0, 0.0, 1.0);
                scene.update(&queue);
            }
            scene.camera.position = home;
        }
        scene.update(&queue);
        eprintln!(
            "filled the windows in {:.2?}, finest level {}",
            started.elapsed(),
            scene.terrain.base_level()
        );

        let bytes_per_row = WIDE * 4;
        assert_eq!(bytes_per_row % 256, 0, "readback rows must stay aligned");
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row * TALL),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let started = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        scene.draw(&mut encoder, &view, &depth);
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(TALL),
                },
            },
            wgpu::Extent3d {
                width: WIDE,
                height: TALL,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        eprintln!("rendered one frame in {:.2?}", started.elapsed());

        let pixels = readback.get_mapped_range(..).expect("not mapped").to_vec();
        let mut ppm = format!("P6\n{WIDE} {TALL}\n255\n").into_bytes();
        for y in 0..TALL {
            for x in 0..WIDE {
                let i = ((y * WIDE + x) * 4) as usize;
                ppm.extend_from_slice(&pixels[i..i + 3]);
            }
        }
        readback.unmap();

        let path = std::env::temp_dir().join("terrain.ppm");
        std::fs::write(&path, ppm).expect("failed to write the preview");
        eprintln!("wrote {}", path.display());
    }

    /// Rough terrain, so that neighbouring clipmap levels genuinely disagree
    /// about where the surface is and any seam between them would show.
    fn rugged() -> (Vec<f32>, Vec<Srgb8>) {
        let heights = (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as f32, (i / RASTER) as f32);
                300.0 * ((x * 0.31).sin() + (y * 0.27).cos()) + 150.0 * (x * 0.11 - y * 0.09).sin()
            })
            .collect();
        (heights, flat_ground())
    }

    /// As [`rugged`], but painted rather than uniformly green.
    ///
    /// Looking straight down at flat colour, geometry is nearly invisible: the
    /// frame is the same green wherever the surface happens to be. A test that
    /// means to see which level drew a patch of ground needs the ground to look
    /// different from place to place, so that both the shape and the texel it is
    /// coloured from show up in the pixels.
    fn rugged_painted() -> (Vec<f32>, Vec<Srgb8>) {
        let (heights, _) = rugged();
        let colours = (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as f32, (i / RASTER) as f32);
                // A few texels per cycle: fine enough that a coarser level's
                // averaging of it is plainly a different colour, coarse enough
                // not to alias into noise that would drown the difference.
                let wave = |f: f32| (128.0 + 110.0 * f.sin()) as u8;
                Srgb8([wave(x * 0.7), wave(y * 0.6), wave((x + y) * 0.45), 255])
            })
            .collect();
        (heights, colours)
    }

    #[test]
    fn no_sky_shows_through_the_joins_between_levels() {
        // Looking straight down, across a ring boundary. A T-junction between
        // two levels would let the sky through as a pinhole or a hairline.
        //
        // The altitude is chosen so the frame stays well inside the raster even
        // at its corners: terrain below the camera's own height projects
        // outwards, so the ground visible at the frame edge comes from further
        // out than the frustum's footprint alone suggests. Straying past the
        // data would show sky for the honest reason that the terrain ends
        // there, and mask the seams this is looking for.
        //
        // It also has to stay under the height at which the finest level is
        // dropped for being too far below the camera to be worth drawing --
        // around 1060 m here, where the ground is a hundred-odd metres up and
        // this small clipmap's finest level reaches only 480 m. Above that there
        // are fewer levels left to have joins between, and the test would go
        // quiet rather than fail. Hence the assertion on the base level below.
        let (heights, colours) = rugged();
        let (pixels, base) = render_probed(
            heights,
            colours,
            |camera| {
                camera.position = Vec3::new(70.0, 900.0, -110.0);
                camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
            },
            &[],
        );
        assert_eq!(base, 0, "the frame under test has to hold every level");

        let holes: Vec<(u32, u32)> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| is_sky(pixel(&pixels, x, y)))
            .collect();

        assert!(
            holes.is_empty(),
            "{} pixels of sky came through the terrain, first at {:?}",
            holes.len(),
            holes.first()
        );
    }

    #[test]
    fn the_terrain_stops_at_the_edge_of_the_data() {
        // Clipmap rings deliberately reach past the raster so there is always a
        // level coarse enough to cover the horizon. Out there every read clamps
        // to the border texel, which would otherwise draw the edge row smeared
        // outwards as a plateau indistinguishable from real ground.
        //
        // Flat ground, so that a point's screen position depends only on where
        // it is: over rough terrain a tall peak inside the raster projects onto
        // the same pixel as a spot outside it, and the two cannot be told apart.
        let mut camera = None;
        let pixels = render(vec![0.0; (RASTER * RASTER) as usize], flat_ground(), |c| {
            // High enough that the raster's edge sits well inside the frame.
            c.position = Vec3::new(0.0, 6000.0, 0.0);
            c.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
            camera = Some(*c);
        });
        let camera = camera.expect("camera captured");
        let at = |world: Vec3| {
            let (x, y) = to_pixels(camera.view_projection(), world, SIZE, SIZE);
            pixel(&pixels, x.round() as u32, y.round() as u32)
        };

        let ((min_x, min_z), (max_x, max_z)) = placement().data_bounds();
        let (min_x, min_z) = (min_x as f32, min_z as f32);
        let (max_x, max_z) = (max_x as f32, max_z as f32);

        // Sampled at ground level, where a point's screen position does not
        // depend on the terrain's own height.
        for corner in [
            Vec3::new(min_x, 0.0, min_z),
            Vec3::new(max_x, 0.0, min_z),
            Vec3::new(min_x, 0.0, max_z),
            Vec3::new(max_x, 0.0, max_z),
        ] {
            let outside = corner + Vec3::new(corner.x.signum(), 0.0, corner.z.signum()) * 150.0;
            assert!(
                is_sky(at(outside)),
                "{outside} lies beyond the raster but was drawn as terrain: {:?}",
                at(outside)
            );
        }

        // ... and the data itself is still drawn right up to its edge, so this
        // has not simply clipped the terrain away.
        let inside = Vec3::new(max_x - 150.0, 0.0, max_z - 150.0);
        assert!(
            !is_sky(at(inside)),
            "{inside} is inside the raster but was cut away: {:?}",
            at(inside)
        );
    }

    #[test]
    fn crossing_a_ring_boundary_does_not_make_the_terrain_jump() {
        // Window origins snap to even texels, so a level's grid shifts by a
        // whole two texels at a time. Without the morph those shifts would land
        // as visible pops; with it, each step should change the picture about as
        // much as any other.
        let (heights, colours) = rugged();
        let aim = |camera: &mut Camera| {
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -25f32.to_radians(), 0.0);
        };

        // Advance by a fraction of a texel at a time, so several steps fall
        // between each snap of the finest window.
        let step = (METRES_PER_TEXEL / 3.0) as f32;
        let frames: Vec<Vec<u8>> = (0..12)
            .map(|i| {
                let moved = aim;
                let z = 900.0 - f32::from(i as u16) * step;
                render(heights.clone(), colours.clone(), move |camera| {
                    moved(camera);
                    camera.position = Vec3::new(0.0, 700.0, z);
                })
            })
            .collect();

        assert_no_step_stands_out(&frames, 4.0);
    }

    /// Asserts that no one step between consecutive frames changed the picture
    /// far more than its neighbours did, which is what a pop looks like.
    ///
    /// The frames are expected to come from a camera moving steadily, so the
    /// change between any two of them is roughly the same. A level snapping into
    /// place, or vanishing, shows up as a single outlier.
    ///
    /// `tolerance` is how many times the typical step the worst one is allowed
    /// to be. How tight it can be depends on how evenly the sweep changes the
    /// frame to begin with: a camera flying along sees the picture turn over
    /// steadily and needs room, one climbing straight up mostly zooms and can be
    /// held to much less.
    fn assert_no_step_stands_out(frames: &[Vec<u8>], tolerance: f64) {
        let differences: Vec<f64> = frames
            .windows(2)
            .map(|pair| {
                let total: u64 = pair[0]
                    .iter()
                    .zip(&pair[1])
                    .map(|(a, b)| u64::from(a.abs_diff(*b)))
                    .sum();
                total as f64 / pair[0].len() as f64
            })
            .collect();

        let worst = differences.iter().copied().fold(0.0, f64::max);
        let typical = differences.iter().sum::<f64>() / differences.len() as f64;
        assert!(
            worst < typical * tolerance + 1.0,
            "one step changed the frame far more than the others, which is what \
             a pop looks like: worst {worst:.2}, typical {typical:.2}, all {differences:?}"
        );
    }

    /// Looks straight down from `altitude` over the same spot every time.
    fn from_altitude(altitude: f32) -> impl FnOnce(&mut Camera) {
        move |camera: &mut Camera| {
            camera.position = Vec3::new(70.0, altitude, -110.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
        }
    }

    #[test]
    fn climbing_away_from_the_ground_gives_up_the_finest_levels() {
        // Levels are chosen by how far the ground they cover is from the camera,
        // and a camera in the air is far from the ground directly below it as
        // well as from the horizon. Drawing the finest level from high up spends
        // full-resolution triangles on ground that covers a fraction of a pixel,
        // and a fine window's worth of tile reads on fetching it.
        let (heights, colours) = rugged();

        let (_, low) = render_probed(heights.clone(), colours.clone(), from_altitude(900.0), &[]);
        let (pixels, high) = render_probed(heights, colours, from_altitude(4000.0), &[]);

        assert_eq!(low, 0, "close to the ground every level is worth drawing");
        assert!(
            high > low,
            "climbing should have given up at least one level, still at {high}"
        );

        // ... and what the dropped levels used to draw is still drawn, by the
        // level that took over. The middle of the frame is well inside the
        // raster at this height; its edges are not, and the sky past the data is
        // honest there.
        let holes: Vec<(u32, u32)> = (SIZE / 4..SIZE * 3 / 4)
            .flat_map(|y| (SIZE / 4..SIZE * 3 / 4).map(move |x| (x, y)))
            .filter(|&(x, y)| is_sky(pixel(&pixels, x, y)))
            .collect();
        assert!(
            holes.is_empty(),
            "dropping the finest levels left {} pixels of sky, first at {:?}",
            holes.len(),
            holes.first()
        );
    }

    #[test]
    fn climbing_past_the_height_a_level_is_dropped_does_not_make_the_terrain_jump() {
        // A level does not simply vanish when the camera gets far enough from
        // the ground for it to stop being worth drawing: it is blended into the
        // level outside it on the way up, so that by the time it goes it is
        // already drawing that level's surface and colour exactly. Without the
        // blend, the whole middle of the frame would snap to a coarser shape in
        // one frame.
        //
        // The sweep spans the height where this clipmap's finest level goes,
        // around 1060 m over the hundred-odd metres of ground below the camera.
        let (heights, colours) = rugged_painted();
        let probed: Vec<(Vec<u8>, u32)> = (0..13)
            .map(|i| {
                let altitude = 900.0 + f32::from(i as u16) * 30.0;
                render_probed(
                    heights.clone(),
                    colours.clone(),
                    from_altitude(altitude),
                    &[],
                )
            })
            .collect();

        let first = probed.first().expect("frames rendered").1;
        assert!(
            probed.iter().any(|(_, base)| *base != first),
            "the sweep never crossed the height a level is dropped at"
        );

        let frames: Vec<Vec<u8>> = probed.into_iter().map(|(pixels, _)| pixels).collect();
        assert_no_step_stands_out(&frames, 2.0);
    }

    /// A raster source that notes which levels are read from it.
    struct Counted {
        inner: Box<dyn RasterSource>,
        levels: std::rc::Rc<std::cell::RefCell<Vec<u32>>>,
    }

    impl RasterSource for Counted {
        fn level_count(&self) -> u32 {
            self.inner.level_count()
        }

        fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]) {
            self.levels.borrow_mut().push(level);
            self.inner.read_rect(level, origin, size, out);
        }
    }

    #[test]
    fn a_level_too_fine_to_draw_is_not_streamed_either() {
        // The saving that matters most is not the triangles: it is the tiles.
        // A window that is not drawn still follows the camera, and at altitude
        // the camera covers ground fast, so leaving the finest levels streaming
        // would keep reading detail nobody can see. They stop entirely instead,
        // and are refilled whole when the camera comes back down to them --
        // their textures having gone stale in the meantime.
        let (device, queue) = test_device();
        let (heights, colours) = rugged();
        let reads = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let mut scene = Scene::from_terrain(
            &device,
            |camera_layout| {
                Terrain::new(
                    &device,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    camera_layout,
                    test_config(),
                    placement(),
                    Box::new(Counted {
                        inner: Box::new(Pyramid::build(Level::new(RASTER, RASTER, heights))),
                        levels: reads.clone(),
                    }),
                    Box::new(Pyramid::build(Level::new(RASTER, RASTER, colours))),
                )
            },
            1.0,
        );

        let mut read_levels = |at: Vec3| {
            reads.borrow_mut().clear();
            scene.camera.position = at;
            scene.update(&queue);
            let seen: std::collections::HashSet<u32> = reads.borrow().iter().copied().collect();
            (seen, scene.terrain.base_level())
        };

        // High enough that the finest level is gone. Note this is the very first
        // update, so nothing is resident and every level still being drawn has
        // to be read in full -- what is missing is missing because it was
        // dropped, not because it happened to have nothing new.
        let (high, base) = read_levels(Vec3::new(70.0, 4000.0, -110.0));
        assert!(base > 0, "the sweep needs an altitude that drops a level");
        assert_eq!(
            high.iter().copied().min(),
            Some(base),
            "levels below {base} should not have been streamed: read {high:?}"
        );

        // ... and coming back down brings them straight back.
        let (low, base) = read_levels(Vec3::new(70.0, 900.0, -110.0));
        assert_eq!(base, 0, "the descent has to reach the finest level again");
        assert!(
            low.contains(&0),
            "the finest level did not come back on descent: read {low:?}"
        );
    }

    #[test]
    fn the_camera_opens_above_the_terrain_looking_at_all_of_it() {
        let extent = Vec2::new(4000.0, 9000.0);
        let camera = Camera::overlooking(extent, 2500.0, 16.0 / 9.0);

        assert!(
            camera.position.y > 2500.0,
            "the viewpoint must clear the highest ground, got {}",
            camera.position.y
        );

        // Both far corners of the terrain fall inside the frustum.
        for corner in [
            Vec3::new(-extent.x * 0.5, 0.0, -extent.y * 0.5),
            Vec3::new(extent.x * 0.5, 0.0, -extent.y * 0.5),
        ] {
            let clip = camera.view_projection() * corner.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!(
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc.z),
                "{corner} projects outside the view at {ndc}"
            );
        }

        // ... and the middle of the view lands on the terrain rather than
        // beyond it. Having the corners in shot is not enough on its own: they
        // can sit along the very bottom edge with the rest of the frame sky,
        // which is what a pitch that does not follow the extent produces.
        let forward = camera.orientation * Vec3::NEG_Z;
        assert!(forward.y < 0.0, "the view must slope downwards");
        let ground = camera.position + forward * (-camera.position.y / forward.y);
        assert!(
            (-extent.y * 0.5..=extent.y * 0.5).contains(&ground.z),
            "the centre of the view meets the ground at z {}, outside the \
             terrain's {}..{}",
            ground.z,
            -extent.y * 0.5,
            extent.y * 0.5
        );
    }
}
