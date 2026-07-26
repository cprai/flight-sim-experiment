use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::camera::Camera;

/// Sky the ground plane is drawn against.
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.30,
    g: 0.55,
    b: 0.85,
    a: 1.0,
};

/// Half the side length of the ground plane, in world units (metres).
///
/// Large enough that its far edge sits within a third of a degree of the true
/// horizon from normal viewing heights, so the seam is not obvious.
const GROUND_HALF_EXTENT: f32 = 5_000.0;

/// Where the camera starts: a little above the ground, nose slightly down.
const EYE_HEIGHT: f32 = 25.0;
const EYE_PITCH_DEGREES: f32 = -12.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x3];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Mirrors the `Camera` uniform block in `shader.wgsl`.
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

/// The ground plane plus the camera looking at it, and the GPU state to draw them.
pub struct Scene {
    pub camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl Scene {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, aspect: f32) -> Self {
        let camera = Camera::new(
            Vec3::new(0.0, EYE_HEIGHT, 0.0),
            Camera::from_yaw_pitch_roll(0.0, EYE_PITCH_DEGREES.to_radians(), 0.0),
            aspect,
        );

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform"),
            contents: bytemuck::bytes_of(&CameraUniform::new(&camera)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // A single quad on the y = 0 plane. The fragment shader derives the
        // grass pattern from world position, so no per-vertex data beyond
        // position is needed.
        let e = GROUND_HALF_EXTENT;
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground vertices"),
            contents: bytemuck::cast_slice(&[
                Vertex {
                    position: [-e, 0.0, -e],
                },
                Vertex {
                    position: [e, 0.0, -e],
                },
                Vertex {
                    position: [e, 0.0, e],
                },
                Vertex {
                    position: [-e, 0.0, e],
                },
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ground indices"),
            contents: bytemuck::cast_slice::<u16, u8>(&[0, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scene pipeline layout"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(Vertex::layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            // Backface culling stays off so the ground is still drawn when the
            // camera passes below it.
            primitive: wgpu::PrimitiveState::default(),
            // Only one opaque object so far; nothing to sort against.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            camera,
            camera_buffer,
            camera_bind_group,
            pipeline,
            vertices,
            indices,
            index_count: 6,
        }
    }

    /// Uploads the current camera transform. Call once per frame before [`Scene::draw`].
    pub fn upload_camera(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::new(&self.camera)),
        );
    }

    /// Records a sky clear plus the ground plane into `view`.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
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
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

/// Where a world point lands on screen, in pixels, with (0, 0) at the top left.
///
/// Only used by tests, but it belongs next to the projection it inverts.
#[cfg(test)]
fn to_pixels(view_proj: glam::Mat4, point: Vec3, width: u32, height: u32) -> (f32, f32) {
    let clip = view_proj * point.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    (
        (ndc.x + 1.0) * 0.5 * width as f32,
        (1.0 - ndc.y) * 0.5 * height as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: u32 = 256;

    /// Renders the scene to an offscreen texture and returns the RGBA8 pixels.
    ///
    /// Lets CI verify the camera, shader and pipeline without a display server.
    fn render_offscreen() -> (Vec<u8>, Camera) {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no wgpu adapter available");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to create device");

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

        // `SIZE * 4` is already a multiple of the 256-byte copy alignment.
        let bytes_per_row = SIZE * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let scene = Scene::new(&device, format, 1.0);
        scene.upload_camera(&queue);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        scene.draw(&mut encoder, &view);
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
        (pixels, scene.camera)
    }

    fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i..i + 4].try_into().unwrap()
    }

    fn is_grass([r, g, b, _]: [u8; 4]) -> bool {
        g > r && g > b
    }

    #[test]
    fn the_camera_looks_down_at_a_green_ground_plane_under_a_blue_sky() {
        let (pixels, _) = render_offscreen();

        let sky = pixel(&pixels, SIZE / 2, 8);
        let [r, g, b, a] = sky;
        assert_eq!(a, 255, "sky should be opaque");
        assert!(
            b > r && b > g,
            "top of frame should be blue sky, got {sky:?}"
        );

        let ground = pixel(&pixels, SIZE / 2, SIZE - 8);
        assert!(
            is_grass(ground),
            "bottom of frame should be green grass, got {ground:?}"
        );
    }

    #[test]
    fn the_horizon_lands_where_the_camera_projection_predicts() {
        let (pixels, camera) = render_offscreen();

        // A point on the ground far enough away to sit essentially on the horizon.
        let expected = to_pixels(
            camera.view_projection(),
            Vec3::new(0.0, 0.0, -GROUND_HALF_EXTENT),
            SIZE,
            SIZE,
        )
        .1;

        let actual = (0..SIZE)
            .find(|&y| is_grass(pixel(&pixels, SIZE / 2, y)))
            .expect("no ground visible in the frame");

        assert!(
            (f32::from(actual as u16) - expected).abs() <= 2.0,
            "horizon at row {actual}, projection predicts {expected:.1}"
        );
    }

    #[test]
    fn perspective_makes_nearer_ground_cells_larger() {
        let (pixels, _) = render_offscreen();

        // Count the checker transitions down the middle column in the top and
        // bottom halves of the ground. Distant cells crowd together, so the
        // upper half must contain more of them.
        let column: Vec<bool> = (0..SIZE)
            .map(|y| pixel(&pixels, SIZE / 2, y)[1] > pixel(&pixels, SIZE / 2, y)[0])
            .collect();
        let first_ground = column.iter().position(|&g| g).expect("no ground visible");

        let edges = |range: std::ops::Range<usize>| {
            range
                .clone()
                .filter(|&y| {
                    let a = pixel(&pixels, SIZE / 2, y as u32);
                    let b = pixel(&pixels, SIZE / 2, y as u32 + 1);
                    a[1].abs_diff(b[1]) > 4
                })
                .count()
        };

        let midpoint = (first_ground + SIZE as usize) / 2;
        let near_horizon = edges(first_ground..midpoint);
        let near_camera = edges(midpoint..SIZE as usize - 1);
        assert!(
            near_horizon > near_camera,
            "cells should crowd toward the horizon: {near_horizon} vs {near_camera} edges"
        );
    }
}
