use anyhow::Result;
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
}

impl CameraUniform {
    fn new(camera: &Camera) -> Self {
        Self {
            view_proj: camera.view_projection().to_cols_array_2d(),
            position: camera.position.extend(1.0).to_array(),
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
    /// Loads the terrain rasters from disk and frames the camera on them.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, aspect: f32) -> Result<Self> {
        let (camera_buffer, camera_layout, camera_bind_group) = camera_binding(device);
        let terrain = Terrain::from_files(
            device,
            format,
            &camera_layout,
            ClipmapConfig::default(),
            crate::terrain::HEIGHT_RASTER_PATH,
            crate::terrain::COLOUR_RASTER_PATH,
        )?;
        Ok(Self::assemble(
            camera_buffer,
            camera_bind_group,
            terrain,
            aspect,
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
    use glam::{Vec2, Vec3};

    use super::*;
    use crate::terrain::geotiff::Georeferencing;
    use crate::terrain::pyramid::{Level, Pyramid, Srgb8};

    /// Side of the offscreen render target.
    const SIZE: u32 = 256;
    /// Side of the synthetic rasters, in texels.
    const RASTER: u32 = 128;
    const METRES_PER_TEXEL: f64 = 30.0;

    /// A deliberately small clipmap, so the software rasterizer stays quick.
    fn test_config() -> ClipmapConfig {
        ClipmapConfig {
            block_verts: 16,
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

    /// A headless device asking for the same limits the application does.
    fn test_device() -> (wgpu::Device, wgpu::Queue) {
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
                    test_config(),
                    placement(),
                    Pyramid::build(Level::new(RASTER, RASTER, heights)),
                    Pyramid::build(Level::new(RASTER, RASTER, colours)),
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
        pixels
    }

    fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i..i + 4].try_into().unwrap()
    }

    fn is_sky([r, g, b, _]: [u8; 4]) -> bool {
        b > r && b > g
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

    #[test]
    fn the_colour_raster_lands_where_the_georeferencing_puts_it() {
        // A patch of a distinct colour, well away from the raster's centre so
        // that getting the axes or the origin wrong would move it visibly.
        let (patch_col, patch_row) = (32u32, 96u32);
        let half = 8u32;
        let mut colours = flat_ground();
        for row in patch_row - half..patch_row + half {
            for col in patch_col - half..patch_col + half {
                colours[(row * RASTER + col) as usize] = RED;
            }
        }

        let mut camera = None;
        let pixels = render(vec![0.0; (RASTER * RASTER) as usize], colours, |c| {
            straight_down(c);
            camera = Some(*c);
        });
        let camera = camera.expect("camera captured");

        let centre = world_of(f64::from(patch_col), f64::from(patch_row));
        let (x, y) = to_pixels(camera.view_projection(), centre, SIZE, SIZE);
        let found = pixel(&pixels, x.round() as u32, y.round() as u32);

        assert!(
            found[0] > found[1] + 40 && found[0] > found[2] + 40,
            "expected the red patch at ({x:.0}, {y:.0}), got {found:?}"
        );

        // ... and the rest of the ground is still the background colour, so the
        // patch has not simply been smeared over everything.
        let elsewhere = world_of(f64::from(patch_col), f64::from(RASTER - patch_row));
        let (x, y) = to_pixels(camera.view_projection(), elsewhere, SIZE, SIZE);
        let found = pixel(&pixels, x.round() as u32, y.round() as u32);
        assert!(
            found[1] > found[0],
            "expected background at ({x:.0}, {y:.0}), got {found:?}"
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

    /// Renders the rasters actually on disk and writes the frame out.
    ///
    /// Ignored because the assets are not in version control, and because this
    /// is a look-at-it check rather than an assertion. Run it with
    /// `cargo test --release -- --ignored dump_installed` and open the file.
    #[test]
    #[ignore = "requires the raster assets, which are not in version control"]
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

        let started = std::time::Instant::now();
        let mut scene = Scene::new(&device, format, WIDE as f32 / TALL as f32)
            .expect("failed to load the installed terrain");
        eprintln!("built the scene in {:.2?}", started.elapsed());
        eprintln!("camera opens at {}", scene.camera.position);
        scene.update(&queue);

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
        let (heights, colours) = rugged();
        let pixels = render(heights, colours, |camera| {
            camera.position = Vec3::new(70.0, 2000.0, -110.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
        });

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
            worst < typical * 4.0 + 1.0,
            "one step changed the frame far more than the others, which is what \
             a pop looks like: worst {worst:.2}, typical {typical:.2}, all {differences:?}"
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
    }
}
