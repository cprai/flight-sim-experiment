//! The G-buffer between the terrain march and the shading that colours it.
//!
//! Rendering is two passes. The geometry pass raymarches the terrain once per
//! pixel and writes what it found -- which material, where in the world, how
//! far away -- into screen-sized buffers. The shading pass reads those buffers
//! back and produces the image. Splitting them means the march, by far the
//! expensive half, runs exactly once however elaborate the shading becomes,
//! and the shading can grow -- lighting, material texture, atmosphere --
//! without the traversal being touched.
//!
//! For now the shading is one flat colour per material from
//! [`crate::palette`]. The world-space position and the surface normal are
//! written and bound but not yet read; they are the inputs every later shading
//! feature starts from, so the plumbing is laid now while the pipeline is
//! being shaped.
//!
//! The three colour targets cost 4, 16 and 8 bytes a sample, and the alignment
//! rule rounds each up to its own component size, so the pass sits at 28 of
//! the 32 bytes `wgpu::Limits::default` guarantees. A fourth target does not
//! fit; the thing to reclaim first is the position, which depth and the
//! camera's inverse projection can reconstruct.
//!
//! Sky is a pixel the march never wrote: the buffers clear to depth zero,
//! which a real hit can never produce -- the camera projects with reversed
//! infinite depth, where zero is the far plane itself -- so the shading pass
//! tests exactly that. A hit whose material is `Null` is different: the ray
//! met real ground that OpenStreetMap says nothing about, and it draws as
//! magenta, the colour of missing data.

use glam::UVec2;
use wgpu::util::DeviceExt;

/// One material id per pixel, straight from the march.
pub const MATERIAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// World-space hit position per pixel; `w` is 1 where a hit was written.
pub const POSITION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;

/// Unit surface normal per pixel, in world space.
///
/// Half floats rather than the two bytes the stored normals themselves take:
/// `Rg8Snorm` is not a format anything is guaranteed to be able to render to,
/// and this buffer holds a decoded direction rather than the packed one.
pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Format of the depth buffer the geometry pass writes.
///
/// Float depth rather than the more compact `Depth24Plus` because the camera
/// projects with reversed depth, which only pays off when the buffer's own
/// precision is concentrated near zero the way a float exponent is.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// The screen-sized targets the geometry pass writes and the shading reads.
///
/// Rebuilt whenever the target size changes, like any depth buffer; the
/// shading pass's bind group has to be rebuilt with it, which is
/// [`Shading::rebind`].
pub struct GBuffer {
    pub material: wgpu::TextureView,
    pub position: wgpu::TextureView,
    pub normal: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    /// The normal target itself, rather than a view of it.
    ///
    /// Nothing draws with the normals yet, so the only way to see whether the
    /// march writes the right ones is to copy the buffer back and look, and a
    /// copy needs the texture. The same reasoning as the `COPY_SRC` on the max
    /// pyramid's texture: cheap enough to pay for always rather than build a
    /// differently-shaped G-buffer under `cfg(test)`.
    #[allow(dead_code, reason = "read only by the G-buffer readback test")]
    pub normal_target: wgpu::Texture,
}

impl GBuffer {
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        let target = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.x.max(1),
                    height: size.y.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let normal_target = target("gbuffer normal", NORMAL_FORMAT);
        Self {
            material: view(&target("gbuffer material", MATERIAL_FORMAT)),
            position: view(&target("gbuffer position", POSITION_FORMAT)),
            normal: view(&normal_target),
            depth: view(&target("gbuffer depth", DEPTH_FORMAT)),
            normal_target,
        }
    }
}

/// The pass that turns the G-buffer into the image.
pub struct Shading {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    palette: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Shading {
    /// Builds the shading pipeline against the target `format` and the
    /// buffers it will read.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, gbuffer: &GBuffer) -> Self {
        // Uploaded once: the table is a pure function of the material enum,
        // so nothing ever rewrites it.
        let palette = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("material palette"),
            contents: bytemuck::cast_slice(&crate::palette::build()),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let texture = |sample_type| wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shading layout"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, texture(wgpu::TextureSampleType::Uint)),
                entry(2, texture(wgpu::TextureSampleType::Float { filterable: false })),
                entry(3, texture(wgpu::TextureSampleType::Depth)),
                entry(4, texture(wgpu::TextureSampleType::Float { filterable: false })),
            ],
        });
        let bind_group = Self::bind(device, &layout, &palette, gbuffer);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shading shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shading.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shading pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shading pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_shade"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_shade"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            // Every pixel is written from the G-buffer's depth, so the pass
            // itself has nothing to test or carry.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            layout,
            pipeline,
            palette,
            bind_group,
        }
    }

    /// Points the pass at a rebuilt G-buffer, after a resize.
    pub fn rebind(&mut self, device: &wgpu::Device, gbuffer: &GBuffer) {
        self.bind_group = Self::bind(device, &self.layout, &self.palette, gbuffer);
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        palette: &wgpu::Buffer,
        gbuffer: &GBuffer,
    ) -> wgpu::BindGroup {
        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shading bind group"),
            layout,
            entries: &[
                entry(0, palette.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(&gbuffer.material)),
                entry(2, wgpu::BindingResource::TextureView(&gbuffer.position)),
                entry(3, wgpu::BindingResource::TextureView(&gbuffer.depth)),
                entry(4, wgpu::BindingResource::TextureView(&gbuffer.normal)),
            ],
        })
    }

    /// Records the shading into an already-started render pass.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
