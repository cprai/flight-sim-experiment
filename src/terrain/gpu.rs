//! The GPU side of the clipmap: textures, pipeline, and per-frame updates.

use std::ops::Range;

use std::path::Path;

use anyhow::{Context, Result};
use glam::{DVec2, IVec2, UVec2, Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::terrain::clipmap::{
    ClipmapConfig, detail_base, exposed_regions, grid_origin, split_across_seam, window_origin,
};
use crate::terrain::geotiff::Georeferencing;
use crate::terrain::mesh::{self, PatchKind};
use crate::terrain::pyramid::{RasterSource, Resident, Srgb8};
use crate::terrain::tiles::TileStore;

/// Must match `MAX_LEVELS` in the shader.
const MAX_LEVELS: usize = 16;

/// Vertical scale applied to the height raster, for when terrain needs
/// exaggerating to read clearly. One means true to the data.
const VERTICAL_EXAGGERATION: f32 = 1.0;

/// Side length of a compute workgroup, in texels of the mip it writes.
const WORKGROUP: u32 = 8;

/// The coarsest mip of the max pyramid: one texel covering the whole window.
///
/// A window is a power of two texels across, so this is just its exponent.
const fn max_mip(config: &ClipmapConfig) -> u32 {
    config.window_texels.trailing_zeros()
}

/// A grid vertex: nothing but its integer position within the shared grid.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridVertex {
    position: [u16; 2],
}

/// One patch of geometry to draw.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PatchInstance {
    origin: [u32; 2],
    level: u32,
    padding: u32,
}

/// Mirrors `Level` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LevelUniform {
    origin: [f32; 2],
    spacing: [f32; 2],
    torus: [u32; 2],
    coarse_offset: [f32; 2],
}

/// Mirrors `Terrain` in the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainUniform {
    levels: [LevelUniform; MAX_LEVELS],
    level_count: u32,
    window_mask: u32,
    morph_band: f32,
    grid_quads: f32,
    data_min: [f32; 2],
    data_max: [f32; 2],
    base_level: u32,
    base_morph: f32,
    near_radius: f32,
    max_mip: u32,
    window_quads: f32,
    grid_offset: f32,
    // The struct's alignment is that of its widest member, so its size has to
    // stay a multiple of sixteen to match the shader's layout.
    padding: [u32; 2],
}

/// A height raster and a matching colour raster, drawn as a geometry clipmap.
pub struct Terrain {
    config: ClipmapConfig,
    placement: Georeferencing,
    heights: Box<dyn RasterSource>,
    colours: Box<dyn RasterSource>,
    height_range: (f32, f32),

    /// One window origin per level, in that level's own texel coordinates.
    origins: Vec<IVec2>,
    /// Whether each level's window holds the ground its origin claims.
    ///
    /// Per level rather than one flag for the lot, because a level dropped for
    /// altitude stops being uploaded while its origin keeps following the
    /// camera. When it comes back its texture holds ground from wherever it was
    /// abandoned, which no incremental update can repair, so it is refilled
    /// whole.
    filled: Vec<bool>,
    /// The finest level being drawn, and how far it has blended into the next.
    detail: (u32, f32),
    /// A CPU mirror of the coarsest level's height window.
    ///
    /// The clipmap's whole point is that the ground under the camera is already
    /// resident, so height above terrain costs a copy of what `upload` is
    /// passing through anyway rather than a read back out of the tile store --
    /// which reopens and reparses a GeoTIFF per call and could not be asked
    /// every frame. The coarsest level is the one mirrored because it is the one
    /// level never dropped: the level chosen from this height must not depend on
    /// data whose residency that choice controls.
    ground: Vec<f32>,
    /// Reused between uploads so a moving camera allocates nothing.
    staging: Vec<u8>,

    height_texture: wgpu::Texture,
    colour_texture: wgpu::Texture,
    /// The max pyramid itself, kept only so a test can copy it back.
    ///
    /// Drawing does not need the handle: the bind groups below hold views of it,
    /// and wgpu resources are reference counted, so the texture outlives this
    /// struct field whether or not it exists.
    #[cfg(test)]
    maxima: wgpu::Texture,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    /// Draws everything past the near radius by raymarching the max pyramid.
    far_pipeline: wgpu::RenderPipeline,
    far_group: wgpu::BindGroup,

    /// Builds the quadtree the far field is raymarched through.
    ///
    /// Layer `l` mip `m` texel `(i, j)` ends up an upper bound on level `l`'s
    /// surface across the closed square `[i*2^m, (i+1)*2^m]` on each axis of its
    /// window, so a ray staying above that value skips the whole square in one
    /// step. `cell_max` writes the finest mip from the height windows and
    /// `reduce` folds each mip into the next.
    cell_max_pipeline: wgpu::ComputePipeline,
    reduce_pipeline: wgpu::ComputePipeline,
    cell_max_group: wgpu::BindGroup,
    /// One per mip above the finest: the mip below it read, this one written.
    ///
    /// Built once, because the views never change; only how many layers are
    /// dispatched varies, with the base level.
    reduce_groups: Vec<wgpu::BindGroup>,

    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_ranges: Vec<Range<u32>>,
    instances: wgpu::Buffer,
    /// Which index range to draw, and for which run of instances.
    draws: Vec<(usize, Range<u32>)>,
}

impl Terrain {
    /// Opens both tile pyramids and builds the clipmap around them.
    ///
    /// `root` holds one directory per product. Nothing is decoded here beyond
    /// two manifests -- the tiles themselves are read a window at a time while
    /// drawing, which is what keeps residency independent of how much ground the
    /// pyramid covers.
    pub fn from_tiles(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
        config: ClipmapConfig,
        root: &Path,
    ) -> Result<Self> {
        let elevation = crate::terrain::ELEVATION_PRODUCTS
            .iter()
            .map(|product| root.join(product))
            .find(|directory| directory.is_dir())
            .with_context(|| {
                format!(
                    "{} holds no {} directory",
                    root.display(),
                    crate::terrain::ELEVATION_PRODUCTS.join(" or ")
                )
            })?;
        let colour = root.join(crate::terrain::COLOUR_PRODUCT);

        let heights = TileStore::<f32>::open(&elevation)?;
        let colours = TileStore::<Srgb8>::open(&colour)?;

        // Structural rather than approximate: both manifests come from one
        // download over one snapped extent, so they either describe the same
        // ground exactly or one of them is from a different run.
        anyhow::ensure!(
            heights.manifest().covers_same_ground_as(colours.manifest()),
            "{} and {} do not cover the same ground",
            elevation.display(),
            colour.display()
        );

        let placement = heights.placement();
        log::info!(
            "terrain: {} x {} texels at {} m, levels up to {}",
            placement.width,
            placement.height,
            heights.manifest().base_metres_per_texel,
            heights.manifest().max_level()
        );

        // The downloader writes as many levels as it was asked for, which is
        // rarely enough to span the whole raster: five here where seven are
        // needed. Continuing the chain in memory is what lets the outermost ring
        // -- and so the far field marched through it -- reach the edge of the
        // data rather than a couple of kilometres of it.
        let raster = UVec2::new(placement.width, placement.height);
        Ok(Self::new(
            device,
            format,
            camera_layout,
            config,
            placement,
            Box::new(Resident::<f32>::over(Box::new(heights), raster)),
            Box::new(Resident::<Srgb8>::over(Box::new(colours), raster)),
        ))
    }

    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
        config: ClipmapConfig,
        placement: Georeferencing,
        heights: Box<dyn RasterSource>,
        colours: Box<dyn RasterSource>,
    ) -> Self {
        let raster = UVec2::new(placement.width, placement.height);
        let available = heights.level_count().min(colours.level_count());
        let level_count = config.level_count(raster, available).min(MAX_LEVELS as u32);
        let window = config.window_texels;

        let layers = wgpu::Extent3d {
            width: window,
            height: window,
            depth_or_array_layers: level_count,
        };
        let clip_texture = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: layers,
                // The array *is* the mip chain -- layer N holds mip N -- so the
                // textures themselves need only one level each.
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let height_texture = clip_texture("terrain heights", wgpu::TextureFormat::R32Float, usage);
        let colour_texture = clip_texture(
            "terrain colours",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            usage,
        );

        // Unlike the clipmap's own textures, this one is a real mip chain: the
        // array indexes clipmap level, the mips index quadtree depth within a
        // level's window. `COPY_SRC` is not needed to draw, only so a test can
        // read the pyramid back and check it against a reference built on the
        // CPU -- which is cheap enough to be worth paying for always rather than
        // building a differently-shaped texture under `cfg(test)`.
        let maxima = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain maxima"),
            size: layers,
            mip_level_count: max_mip(&config) + 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let array_view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("terrain colour sampler"),
            // Windows wrap around their texture, so the sampler has to as well
            // for filtering to stay correct across the seam.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            // Ground is nearly always seen at a glancing angle, which is the
            // case isotropic filtering blurs worst.
            anisotropy_clamp: 16,
            ..Default::default()
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain levels"),
            size: size_of::<TerrainUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let terrain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT
                        .union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    // Every stage: the vertex stage places the mesh on it,
                    // compute builds the max pyramid over it, and the fragment
                    // stage reads it at the leaf of a raymarch.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT
                        .union(wgpu::ShaderStages::COMPUTE),
                    ty: wgpu::BindingType::Texture {
                        // Only ever `textureLoad`ed, so filtering support for
                        // 32-bit floats is not needed on the device.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain bind group"),
            layout: &terrain_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&array_view(&height_texture)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&array_view(&colour_texture)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::include_wgsl!("../terrain.wgsl"));

        // Two layouts rather than one with an unused entry, because a bind group
        // has to supply everything its layout declares: a shared layout would
        // force the first pass to name some view as its source, and the only
        // view it could name is the mip it is about to write. Sampling and
        // storing the same subresource in one dispatch is a validation error.
        let storage_target = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        };
        let cell_max_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain cell max layout"),
            entries: &[storage_target],
        });
        let reduce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain pyramid reduce layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_target,
            ],
        });

        // A view of one mip, across every layer. Single-mip on both sides of a
        // reduction: a storage view is required to be one mip, and an all-mip
        // view bound beside a storage view of a mip it contains would overlap.
        let mip_view = |mip: u32, label: &str| {
            maxima.create_view(&wgpu::TextureViewDescriptor {
                label: Some(label),
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };

        let cell_max_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain cell max"),
            layout: &cell_max_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&mip_view(0, "terrain maxima mip 0")),
            }],
        });
        let reduce_groups = (1..=max_mip(&config))
            .map(|mip| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain pyramid reduce"),
                    layout: &reduce_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&mip_view(
                                mip - 1,
                                "terrain maxima source",
                            )),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&mip_view(
                                mip,
                                "terrain maxima target",
                            )),
                        },
                    ],
                })
            })
            .collect();

        // Group 3 rather than 0, so that the groups the render pipelines share
        // -- the camera at 0 and the terrain at 1 -- keep their numbering in a
        // shader module every pipeline is compiled from.
        let compute_pipeline = |entry: &str, group: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain pyramid pipeline layout"),
                bind_group_layouts: &[None, Some(&terrain_layout), None, Some(group)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let cell_max_pipeline = compute_pipeline("cs_cell_max", &cell_max_layout);
        let reduce_pipeline = compute_pipeline("cs_reduce", &reduce_layout);

        // The far pass reads every mip at once, which is why this view is
        // separate from the single-mip pair the reductions run between.
        let far_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain far bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let far_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain far"),
            layout: &far_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&array_view(&maxima)),
            }],
        });

        let grid: Vec<GridVertex> = mesh::grid_vertices(&config)
            .into_iter()
            .map(|position| GridVertex { position })
            .collect();
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain grid vertices"),
            contents: bytemuck::cast_slice(&grid),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let (index_data, index_ranges) = mesh::grid_indices(&config);
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain grid indices"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Twelve blocks, four seams, and either a trim pair or a centre, per
        // level; rounded up so the buffer never has to grow.
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain patches"),
            size: (level_count as u64 + 1) * 20 * size_of::<PatchInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&terrain_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<GridVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<PatchInstance>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Uint32x2, 2 => Uint32],
                    }),
                ],
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
            // Culling stays off so terrain is still drawn from below ground.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::scene::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                // Reversed depth, so nearer fragments compare greater.
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let far_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain far pipeline layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(&terrain_layout),
                Some(&far_layout),
            ],
            immediate_size: 0,
        });
        let far_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain far pipeline"),
            layout: Some(&far_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_far"),
                compilation_options: Default::default(),
                // One triangle covering the viewport, generated from the vertex
                // index alone.
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_far"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::scene::DEPTH_FORMAT,
                // The fragment stage writes its own depth, from the distance the
                // ray actually met the ground, so the far field sorts against
                // the near field rather than simply losing to it.
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let height_range = Self::coarsest_height_range(heights.as_ref(), &placement);
        let window_bytes = (window * window) as usize;
        Self {
            config,
            placement,
            heights,
            colours,
            height_range,
            origins: vec![IVec2::ZERO; level_count as usize],
            filled: vec![false; level_count as usize],
            detail: (0, 0.0),
            ground: vec![0.0; window_bytes],
            staging: vec![0; window_bytes * size_of::<Srgb8>().max(size_of::<f32>())],
            height_texture,
            colour_texture,
            #[cfg(test)]
            maxima,
            uniform,
            bind_group,
            pipeline,
            far_pipeline,
            far_group,
            cell_max_pipeline,
            reduce_pipeline,
            cell_max_group,
            reduce_groups,
            vertices,
            indices,
            index_ranges,
            instances,
            draws: Vec::new(),
        }
    }

    /// The terrain's ground size in metres.
    pub fn world_extent(&self) -> Vec2 {
        let (x, z) = self.placement.world_extent();
        Vec2::new(x as f32, z as f32)
    }

    /// The lowest and highest elevation in the raster, after exaggeration.
    pub fn height_range(&self) -> (f32, f32) {
        (
            self.height_range.0 * VERTICAL_EXAGGERATION,
            self.height_range.1 * VERTICAL_EXAGGERATION,
        )
    }

    /// The finest level currently being drawn.
    ///
    /// Zero on the ground, rising as the camera climbs away from the terrain and
    /// the levels below it stop being worth their triangles.
    #[cfg(test)]
    pub fn base_level(&self) -> u32 {
        self.detail.0
    }

    /// One level's resident window, in window order and after exaggeration.
    ///
    /// Read back from the raster source rather than from the texture, so a test
    /// comparing the max pyramid against it is comparing against the ground the
    /// upload was asked for and not against the same GPU state twice.
    #[cfg(test)]
    pub fn window_heights(&self, level: u32) -> Vec<f32> {
        let window = self.config.window_texels;
        let mut heights = vec![0f32; (window * window) as usize];
        self.heights.read_rect(
            level,
            self.origins[level as usize],
            UVec2::splat(window),
            bytemuck::cast_slice_mut(&mut heights),
        );
        for height in &mut heights {
            *height *= VERTICAL_EXAGGERATION;
        }
        heights
    }

    /// Moves every level's window to follow the camera, uploading only the
    /// ground that has come into view since the last call.
    pub fn update(&mut self, queue: &wgpu::Queue, camera: Vec3) {
        let camera_texels = self
            .placement
            .texel_of_world(f64::from(camera.x), f64::from(camera.z));
        let window = self.config.window_texels;

        let levels = self.origins.len();
        let coarsest = levels - 1;

        // Every window's new position is settled before any of them is
        // described, because each level's morph needs to know where the level
        // outside it ended up.
        let placed: Vec<IVec2> = (0..levels as u32)
            .map(|level| window_origin(&self.config, level, camera_texels))
            .collect();

        // The coarsest level is refreshed first, out of turn, because the height
        // above terrain that decides how many of the finer levels are worth
        // uploading at all is read back out of its window. Taking it from the one
        // level that is never dropped keeps the decision independent of what the
        // decision itself makes resident, and means the very first frame already
        // has ground to measure against.
        self.refresh(queue, coarsest, placed[coarsest]);
        let ground = self.ground_height(camera_texels);
        let metres_per_texel = self
            .placement
            .metres_per_texel_x
            .min(self.placement.metres_per_texel_z);
        self.detail = detail_base(
            &self.config,
            metres_per_texel,
            f64::from(camera.y - ground),
            levels as u32,
        );
        let (base, base_morph) = self.detail;
        let near_radius = self.config.near_radius(metres_per_texel, base);

        for (level, &new) in placed.iter().enumerate().take(coarsest) {
            if level as u32 >= base {
                self.refresh(queue, level, new);
            } else {
                // Dropped for altitude. Its origin still follows the camera, so
                // that the level outside it can place its trim against a window
                // that has not gone stale, but nothing is uploaded and whatever
                // its texture holds is no longer the ground its origin claims.
                self.origins[level] = new;
                self.filled[level] = false;
            }
        }

        let (data_min, data_max) = self.placement.data_bounds();
        let mut uniform = TerrainUniform {
            levels: [LevelUniform::default(); MAX_LEVELS],
            level_count: levels as u32,
            window_mask: window - 1,
            morph_band: self.config.morph_band,
            grid_quads: self.config.grid_quads() as f32,
            data_min: [data_min.0 as f32, data_min.1 as f32],
            data_max: [data_max.0 as f32, data_max.1 as f32],
            base_level: base,
            base_morph,
            near_radius: near_radius as f32,
            max_mip: max_mip(&self.config),
            window_quads: self.config.window_quads() as f32,
            grid_offset: self.config.margin() as f32,
            padding: [0; 2],
        };

        for (level, &new) in placed.iter().enumerate() {
            let (origin_x, origin_z) =
                self.placement
                    .world_of_texel(level as u32, f64::from(new.x), f64::from(new.y));
            let scale = f64::from(1u32 << level);
            // Halving is exact because window origins are always even, which is
            // what keeps a fine vertex landing on a whole or half coarse texel.
            let coarse_offset = placed
                .get(level + 1)
                .map_or(IVec2::ZERO, |coarse| new / 2 - *coarse);
            uniform.levels[level] = LevelUniform {
                origin: [origin_x as f32, origin_z as f32],
                spacing: [
                    (self.placement.metres_per_texel_x * scale) as f32,
                    (self.placement.metres_per_texel_z * scale) as f32,
                ],
                torus: [
                    new.x.rem_euclid(window as i32) as u32,
                    new.y.rem_euclid(window as i32) as u32,
                ],
                coarse_offset: [coarse_offset.x as f32, coarse_offset.y as f32],
            };
        }

        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        self.rebuild_patches(queue, camera, near_radius);
    }

    /// Moves one level's window to `new`, uploading the ground that exposes.
    ///
    /// A level whose texture no longer matches its origin -- because it has
    /// never been filled, or because it was dropped for altitude while its
    /// origin carried on following the camera -- is refilled whole. There is
    /// nothing for an incremental update to keep in that case.
    fn refresh(&mut self, queue: &wgpu::Queue, level: usize, new: IVec2) {
        let window = self.config.window_texels;
        let regions = if self.filled[level] {
            exposed_regions(self.origins[level], new, window)
        } else {
            vec![crate::terrain::clipmap::Rect {
                x: new.x,
                y: new.y,
                width: window,
                height: window,
            }]
        };

        for region in regions {
            for (piece, destination) in split_across_seam(region, window) {
                self.upload(queue, level as u32, piece, destination);
            }
        }
        self.origins[level] = new;
        self.filled[level] = true;
    }

    /// The elevation of the ground under the camera, in world units.
    ///
    /// Read from the mirror of the coarsest level, so it is the ground averaged
    /// over kilometres rather than the peak the camera happens to be over. That
    /// is the right shape of answer for choosing a level that covers kilometres,
    /// and where it is least accurate -- close to the ground, where the relief it
    /// smooths away is a large fraction of the distance -- the finest level is
    /// being drawn anyway.
    fn ground_height(&self, camera_texels: DVec2) -> f32 {
        let level = self.origins.len() - 1;
        let window = self.config.window_texels as i32;
        let texels = camera_texels / f64::from(1u32 << level);
        let offset =
            IVec2::new(texels.x.floor() as i32, texels.y.floor() as i32) - self.origins[level];
        let x = offset.x.rem_euclid(window) as usize;
        let y = offset.y.rem_euclid(window) as usize;

        let height = self.ground[y * window as usize + x];
        // The camera can legally be over ground the raster says nothing about:
        // past the edge of the survey, or over a hole in it. Sea level is the
        // same fallback the terrain itself draws there.
        if height > crate::terrain::NODATA_BELOW {
            height
        } else {
            0.0
        }
    }

    /// Copies one seam-free piece of a window into both clip textures.
    fn upload(
        &mut self,
        queue: &wgpu::Queue,
        level: u32,
        piece: crate::terrain::clipmap::Rect,
        destination: UVec2,
    ) {
        let size = piece.size();
        let extent = wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        };
        let origin = wgpu::Origin3d {
            x: destination.x,
            y: destination.y,
            z: level,
        };

        // `write_texture` has no row-alignment requirement -- unlike a buffer
        // copy -- so an arbitrarily narrow strip uploads directly.
        let copy = |texture: &wgpu::Texture, bytes: u32, data: &[u8]| {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size.x * bytes),
                    rows_per_image: Some(size.y),
                },
                extent,
            );
        };

        let texels = (size.x * size.y) as usize;

        let bytes = texels * size_of::<f32>();
        self.heights
            .read_rect(level, piece.origin(), size, &mut self.staging[..bytes]);
        if VERTICAL_EXAGGERATION != 1.0 {
            for height in bytemuck::cast_slice_mut::<u8, f32>(&mut self.staging[..bytes]) {
                *height *= VERTICAL_EXAGGERATION;
            }
        }
        copy(&self.height_texture, 4, &self.staging[..bytes]);
        if level as usize == self.origins.len() - 1 {
            self.mirror_ground(size, destination, bytes);
        }

        let bytes = texels * size_of::<Srgb8>();
        self.colours
            .read_rect(level, piece.origin(), size, &mut self.staging[..bytes]);
        copy(&self.colour_texture, 4, &self.staging[..bytes]);
    }

    /// Copies a just-uploaded piece of the coarsest window into the CPU mirror.
    ///
    /// Written from the same staging buffer, at the same destination, as the
    /// texture copy immediately above it, so the mirror wraps around the torus
    /// exactly as the texture does and cannot drift out of step with it.
    fn mirror_ground(&mut self, size: UVec2, destination: UVec2, bytes: usize) {
        let heights: &[f32] = bytemuck::cast_slice(&self.staging[..bytes]);
        let window = self.config.window_texels as usize;
        for row in 0..size.y as usize {
            let source = row * size.x as usize;
            let target = (destination.y as usize + row) * window + destination.x as usize;
            self.ground[target..target + size.x as usize]
                .copy_from_slice(&heights[source..source + size.x as usize]);
        }
    }

    /// The lowest and highest elevation, read from a coarse level of the pyramid.
    ///
    /// Read coarse rather than from the base: every texel of a coarse level is a
    /// box filter of everything beneath it, so the range is representative without
    /// anything being scanned that is not already cheap to fetch. Peaks are averaged
    /// down a little, which only matters for framing the camera at startup -- the one
    /// thing this is used for.
    ///
    /// The level is chosen by how much raster it leaves rather than by counting from
    /// the top, because the top is now a single texel: [`Resident`] carries the chain
    /// all the way down to one, and the mean of the whole dataset says nothing about
    /// how high its mountains are.
    fn coarsest_height_range(source: &dyn RasterSource, placement: &Georeferencing) -> (f32, f32) {
        /// Side length to aim for. A quarter of a million texels is a fraction of a
        /// megabyte to read and plenty of samples to take a range over.
        const SAMPLES: u32 = 512;

        let widest = placement.width.max(placement.height);
        let level = widest
            .div_ceil(SAMPLES)
            .next_power_of_two()
            .trailing_zeros()
            .min(source.level_count().saturating_sub(1));
        let size = UVec2::new(
            (placement.width >> level).max(1),
            (placement.height >> level).max(1),
        );

        let mut heights = vec![0f32; (size.x as usize) * (size.y as usize)];
        source.read_rect(
            level,
            IVec2::ZERO,
            size,
            bytemuck::cast_slice_mut(&mut heights),
        );

        let range = heights
            .iter()
            .filter(|height| **height > crate::terrain::NODATA_BELOW)
            .fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(low, high), &height| (low.min(height), high.max(height)),
            );
        // Ground with no data anywhere is legal; sea level is as good a frame as any.
        if range.0.is_finite() {
            range
        } else {
            (0.0, 0.0)
        }
    }

    /// Rebuilds the instance buffer and the draw ranges that index it.
    fn rebuild_patches(&mut self, queue: &wgpu::Queue, camera: Vec3, near_radius: f64) {
        let (data_min, data_max) = self.placement.data_bounds();
        // Patches are laid out in grid coordinates, which start a margin in from
        // the window the textures are addressed in.
        let grids: Vec<IVec2> = self
            .origins
            .iter()
            .map(|origin| grid_origin(&self.config, *origin))
            .collect();
        let patches: Vec<mesh::Patch> = mesh::patches(&self.config, &grids, self.detail.0)
            .into_iter()
            .filter(|patch| {
                let level = patch.level as usize;
                let scale = f64::from(1u32 << level);
                let (near_x, near_z) = self.placement.world_of_texel(
                    patch.level,
                    f64::from(grids[level].x + patch.origin.x as i32),
                    f64::from(grids[level].y + patch.origin.y as i32),
                );
                let size = patch.kind.size_quads(&self.config);
                let far_x = near_x + f64::from(size.x) * scale * self.placement.metres_per_texel_x;
                let far_z = near_z + f64::from(size.y) * scale * self.placement.metres_per_texel_z;

                // Rings reach beyond the raster, and the fragment stage cuts
                // away whatever falls outside it. A patch lying wholly out there
                // would have every one of its fragments thrown away, so drop it
                // here instead and never rasterize it at all. The saving is
                // largest exactly where it matters: a coarse ring viewed from
                // over a corner of the data.
                let inside_data = far_x >= data_min.0
                    && near_x <= data_max.0
                    && far_z >= data_min.1
                    && near_z <= data_max.1;

                // Likewise for the patches the raymarched pass has taken over.
                // The fragment stage cuts at the exact radius in three
                // dimensions; this only has to save the vertex work without ever
                // dropping a patch that stage would have kept, so it measures
                // horizontally to the nearest point of the patch's box. Three
                // dimensional distance is never less than horizontal distance,
                // which makes that exactly conservative and needs no bound on
                // how high the ground inside the box reaches. No such bound is
                // available anyway: `height_range` comes from the coarsest level
                // of the pyramid, which has already averaged the peaks down, so
                // using it here would drop patches whose fragments were well
                // inside the radius and leave the far field to fill a hole it
                // starts too far out to see.
                let x = (near_x - f64::from(camera.x)).max(f64::from(camera.x) - far_x);
                let z = (near_z - f64::from(camera.z)).max(f64::from(camera.z) - far_z);
                let nearest = x.max(0.0).powi(2) + z.max(0.0).powi(2);

                inside_data && nearest <= near_radius * near_radius
            })
            .collect();

        let instances: Vec<PatchInstance> = patches
            .iter()
            .map(|patch| PatchInstance {
                origin: patch.origin.to_array(),
                level: patch.level,
                padding: 0,
            })
            .collect();
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&instances));

        // Patches arrive grouped by kind, so each run is one draw call.
        self.draws.clear();
        let mut start = 0;
        while start < patches.len() {
            let kind = patches[start].kind;
            let end = patches[start..]
                .iter()
                .position(|patch| patch.kind != kind)
                .map_or(patches.len(), |offset| start + offset);
            let index = PatchKind::ALL
                .iter()
                .position(|&candidate| candidate == kind)
                .expect("every patch kind is listed");
            self.draws.push((index, start as u32..end as u32));
            start = end;
        }
    }

    /// Rebuilds the max pyramid over the windows [`Terrain::update`] left.
    ///
    /// Records its own compute pass, so it has to be called on the frame's
    /// encoder between `update` and [`Terrain::draw`]. `update` has only a
    /// queue, and that is fine: wgpu runs a submission's queued writes before
    /// its encoder's commands, so the uploads are in place by the first
    /// dispatch.
    ///
    /// The whole chain is rebuilt every frame rather than patched where the
    /// windows moved. Almost every frame moves almost every window, and at a
    /// third of a million texels a level this is a fraction of a millisecond --
    /// far less than working out which cells an incremental update would have to
    /// touch.
    pub fn build_pyramid(&self, encoder: &mut wgpu::CommandEncoder) {
        // Levels below the base are neither uploaded nor drawn nor marched, so
        // there is nothing to bound. At altitude that is most of them.
        let layers = self.origins.len() as u32 - self.detail.0;
        let window = self.config.window_texels;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("terrain pyramid pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.cell_max_pipeline);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_bind_group(3, &self.cell_max_group, &[]);
        let groups = window.div_ceil(WORKGROUP);
        pass.dispatch_workgroups(groups, groups, layers);

        // Consecutive dispatches in one pass are ordered against each other, so
        // each mip sees the one below it finished without a pass boundary.
        pass.set_pipeline(&self.reduce_pipeline);
        pass.set_bind_group(1, &self.bind_group, &[]);
        for (mip, group) in self.reduce_groups.iter().enumerate() {
            let size = (window >> (mip + 1)).max(1);
            pass.set_bind_group(3, group, &[]);
            let groups = size.div_ceil(WORKGROUP);
            pass.dispatch_workgroups(groups, groups, layers);
        }
    }

    /// Records the terrain into an already-started render pass.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_vertex_buffer(1, self.instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);

        for (kind, instances) in &self.draws {
            pass.draw_indexed(self.index_ranges[*kind].clone(), 0, instances.clone());
        }
    }

    /// Records the raymarched far field into an already-started render pass.
    ///
    /// Wants its own pass, after [`Terrain::draw`]'s and loading rather than
    /// clearing both attachments: it writes its own depth, so the near field
    /// already being in the buffer is what keeps the two from drawing over each
    /// other where they meet.
    pub fn draw_far(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_pipeline(&self.far_pipeline);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_bind_group(2, &self.far_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::geotiff::Georeferencing;
    use crate::terrain::pyramid::{Level, Pyramid};

    /// Small enough to keep the readback cheap, large enough that the pyramid
    /// has six mips to disagree over.
    fn test_config() -> ClipmapConfig {
        ClipmapConfig {
            block_verts: 8,
            window_texels: 32,
            ..Default::default()
        }
    }

    const RASTER: u32 = 64;

    /// Ridged enough that neighbouring texels disagree, so a cell's maximum is
    /// a real choice rather than whichever corner happened to be picked.
    fn rugged() -> Vec<f32> {
        (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as f32, (i / RASTER) as f32);
                40.0 * (x * 0.7).sin() + 25.0 * (y * 0.9).cos() + 10.0 * (x * 0.13 - y * 0.21).sin()
            })
            .collect()
    }

    /// What the compute passes are supposed to produce, in plain Rust.
    ///
    /// Mip 0 takes the four corners of each quad rather than the one sample the
    /// cell is named after, which is what makes a cell an upper bound over the
    /// ground *between* samples and not just at them.
    fn reference_pyramid(window: &[f32], side: u32) -> Vec<Vec<f32>> {
        let last = side - 1;
        let at = |x: u32, y: u32| window[(y * side + x) as usize];

        let mut base = vec![0f32; (side * side) as usize];
        for y in 0..side {
            for x in 0..side {
                let (fx, fy) = ((x + 1).min(last), (y + 1).min(last));
                base[(y * side + x) as usize] =
                    at(x, y).max(at(fx, y)).max(at(x, fy)).max(at(fx, fy));
            }
        }

        let mut mips = vec![base];
        let mut size = side;
        while size > 1 {
            let finer = mips.last().expect("the base is always there");
            let half = size / 2;
            let mut coarse = vec![0f32; (half * half) as usize];
            for y in 0..half {
                for x in 0..half {
                    let cell =
                        |dx: u32, dy: u32| finer[((2 * y + dy) * size + 2 * x + dx) as usize];
                    coarse[(y * half + x) as usize] =
                        cell(0, 0).max(cell(1, 0)).max(cell(0, 1)).max(cell(1, 1));
                }
            }
            mips.push(coarse);
            size = half;
        }
        mips
    }

    #[test]
    fn a_cell_bounds_every_sample_it_covers() {
        let side = 32;
        let window: Vec<f32> = (0..side * side)
            .map(|i| {
                let (x, y) = ((i % side) as f32, (i / side) as f32);
                (x * 1.1).sin() * 70.0 + (y * 0.7).cos() * 45.0
            })
            .collect();
        let mips = reference_pyramid(&window, side);
        assert_eq!(mips.len(), 6, "32 texels should reduce to one in six mips");

        // The property the whole traversal rests on, stated over the *closed*
        // square. Bounding only the half-open one -- the samples strictly inside
        // -- would leave a ray free to pass between the last sample of one cell
        // and the first of the next, straight through any ridge standing there.
        for (mip, cells) in mips.iter().enumerate() {
            let span = 1u32 << mip;
            let across = side >> mip;
            for j in 0..across {
                for i in 0..across {
                    let bound = cells[(j * across + i) as usize];
                    for y in (j * span)..=((j + 1) * span).min(side - 1) {
                        for x in (i * span)..=((i + 1) * span).min(side - 1) {
                            let sample = window[(y * side + x) as usize];
                            assert!(
                                bound >= sample,
                                "mip {mip} cell ({i}, {j}) bounds {bound} \
                                 but ({x}, {y}) stands at {sample}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_max_pyramid_on_the_gpu_matches_the_one_on_the_cpu() {
        let (device, queue) = crate::scene::test_device();
        let camera_layout = crate::scene::test_camera_layout(&device);
        let config = test_config();
        let window = config.window_texels;

        let mut terrain = Terrain::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &camera_layout,
            config,
            Georeferencing::square(RASTER, RASTER, 30.0),
            Box::new(Pyramid::build(Level::new(RASTER, RASTER, rugged()))),
            Box::new(Pyramid::build(Level::new(
                RASTER,
                RASTER,
                vec![Srgb8([0, 0, 0, 255]); (RASTER * RASTER) as usize],
            ))),
        );

        // Low enough that no level is dropped for altitude, so every layer of
        // the pyramid is dispatched and every one of them is checked.
        terrain.update(&queue, Vec3::new(0.0, 100.0, 0.0));
        assert_eq!(terrain.base_level(), 0, "the test wants every level built");
        let levels = terrain.origins.len() as u32;

        // Rows in a texture-to-buffer copy are padded to 256 bytes, which the
        // finer mips of a 32-texel window are well short of.
        let stride = |size: u32| (size * 4).div_ceil(256) * 256;
        let readbacks: Vec<wgpu::Buffer> = (0..=max_mip(&config))
            .map(|mip| {
                let size = (window >> mip).max(1);
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("pyramid readback"),
                    size: u64::from(stride(size) * size * levels),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        terrain.build_pyramid(&mut encoder);
        for (mip, readback) in readbacks.iter().enumerate() {
            let size = (window >> mip).max(1);
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &terrain.maxima,
                    mip_level: mip as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(stride(size)),
                        rows_per_image: Some(size),
                    },
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: levels,
                },
            );
        }
        queue.submit(std::iter::once(encoder.finish()));

        for readback in &readbacks {
            readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        }
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");

        for level in 0..levels {
            let expected = reference_pyramid(&terrain.window_heights(level), window);
            for (mip, readback) in readbacks.iter().enumerate() {
                let size = (window >> mip).max(1);
                let stride = stride(size) as usize;
                let bytes = readback.get_mapped_range(..).expect("buffer not mapped");
                let layer = &bytes[level as usize * stride * size as usize..];
                for y in 0..size as usize {
                    let row: &[f32] =
                        bytemuck::cast_slice(&layer[y * stride..y * stride + size as usize * 4]);
                    for (x, &got) in row.iter().enumerate() {
                        let want = expected[mip][y * size as usize + x];
                        assert_eq!(
                            got, want,
                            "level {level} mip {mip} texel ({x}, {y}): {got} not {want}"
                        );
                    }
                }
            }
        }
    }
}
