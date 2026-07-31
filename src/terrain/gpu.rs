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
use crate::terrain::maxima::ceiling_half;
use crate::terrain::mesh::{self, PatchKind};
use crate::terrain::pyramid::{RasterSource, Resident, Srgb8};
use crate::terrain::tiles::TileStore;
use terrain_tiles::maxima::highest;

/// Must match `MAX_LEVELS` in the shader.
const MAX_LEVELS: usize = 16;

/// Vertical scale applied to the height raster, for when terrain needs
/// exaggerating to read clearly. One means true to the data.
const VERTICAL_EXAGGERATION: f32 = 1.0;

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
    ceiling: f32,
    padding: [f32; 3],
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
    ceiling: f32,
    march_steps: u32,
}

/// The three rasters a clipmap is built over, all describing one piece of
/// ground.
///
/// Grouped rather than passed alongside each other because they are only
/// meaningful together, and because `maxima` is the one whose levels do not mean
/// what the others' do -- naming it beside them is where that is easiest to get
/// wrong. See [`Terrain::maxima`].
pub struct Sources {
    pub heights: Box<dyn RasterSource>,
    pub colours: Box<dyn RasterSource>,
    pub maxima: Box<dyn RasterSource>,
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

    /// The quadtree the far field is raymarched through, written by
    /// `terrain-process` and read exactly as the heights are.
    ///
    /// **Its levels are not clipmap levels.** Level `m` of this source is the
    /// maximum over squares of `2^m` raster samples, so clipmap level `l`'s
    /// depth `m` is level `l + m` of it; see [`crate::terrain::maxima`]. That is
    /// what lets one pyramid serve every level, and why this source's level
    /// count must never be folded into the count the clipmap builds.
    maxima: Box<dyn RasterSource>,
    /// A mirror of the coarsest cells of each level, in window order.
    ///
    /// The far field's cheapest early out is "above everything and climbing",
    /// and answering it needs the highest ground anywhere resident. Kept on the
    /// CPU because there is no single texel to read it from: the pyramid stops
    /// three depths short of covering a whole window in one cell.
    ceilings: Vec<Vec<f32>>,

    height_texture: wgpu::Texture,
    colour_texture: wgpu::Texture,
    maxima_texture: wgpu::Texture,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    /// Draws everything past the near radius by raymarching the max pyramid.
    far_pipeline: wgpu::RenderPipeline,
    far_group: wgpu::BindGroup,

    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_ranges: Vec<Range<u32>>,
    instances: wgpu::Buffer,
    /// Which index range to draw, and for which run of instances.
    draws: Vec<(usize, Range<u32>)>,
}

impl Terrain {
    /// Opens the tile pyramids and builds the clipmap around them.
    ///
    /// `root` holds one directory per product: elevation, colour, and the max
    /// pyramid `terrain-process` reduced from the elevation. Nothing is decoded
    /// here beyond three manifests -- the tiles themselves are read a window at a
    /// time while drawing, which is what keeps residency independent of how much
    /// ground the pyramid covers.
    pub fn from_tiles(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
        config: ClipmapConfig,
        root: &Path,
    ) -> Result<Self> {
        let product = crate::terrain::ELEVATION_PRODUCTS
            .iter()
            .find(|product| root.join(product).is_dir())
            .with_context(|| {
                format!(
                    "{} holds no {} directory",
                    root.display(),
                    crate::terrain::ELEVATION_PRODUCTS.join(" or ")
                )
            })?;
        let elevation = root.join(product);
        let colour = root.join(crate::terrain::COLOUR_PRODUCT);
        // Named after the elevation it was reduced from, because `dtm` and `dsm`
        // are different surfaces and a bound over one does not cover the other.
        let ceilings = root.join(terrain_tiles::maxima_product(product));

        let heights = TileStore::<f32>::open(&elevation)?;
        let colours = TileStore::<Srgb8>::open(&colour)?;
        let maxima = TileStore::<f32>::open(&ceilings).with_context(|| {
            format!(
                "{} holds no max pyramid for {product}; run terrain-process over the download",
                root.display()
            )
        })?;

        // Structural rather than approximate: every manifest here descends from
        // one download over one snapped extent, so they either describe the same
        // ground exactly or one of them is from a different run.
        for (directory, manifest) in [
            (&colour, colours.manifest()),
            (&ceilings, maxima.manifest()),
        ] {
            anyhow::ensure!(
                heights.manifest().covers_same_ground_as(manifest),
                "{} and {} do not cover the same ground",
                elevation.display(),
                directory.display()
            );
        }

        let placement = heights.placement();
        log::info!(
            "terrain: {} x {} texels at {} m, levels up to {}",
            placement.width,
            placement.height,
            heights.manifest().base_metres_per_texel,
            heights.manifest().max_level()
        );

        // The tools write as many levels as they were asked for, which is rarely
        // enough to span the whole raster: nine here where sixteen are wanted.
        // Continuing the chain in memory is what lets the outermost ring -- and
        // so the far field marched through it -- reach the edge of the data
        // rather than a couple of kilometres of it. The max chain continues with
        // a maximum rather than a mean, or its coarsest cells would sit below
        // the ground they are supposed to bound.
        let raster = UVec2::new(placement.width, placement.height);
        Ok(Self::new(
            device,
            format,
            camera_layout,
            config,
            placement,
            Sources {
                heights: Box::new(Resident::<f32>::over(Box::new(heights), raster)),
                colours: Box::new(Resident::<Srgb8>::over(Box::new(colours), raster)),
                maxima: Box::new(Resident::<f32>::over_with(
                    Box::new(maxima),
                    raster,
                    terrain_tiles::maxima::highest,
                )),
            },
        ))
    }

    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_layout: &wgpu::BindGroupLayout,
        config: ClipmapConfig,
        placement: Georeferencing,
        sources: Sources,
    ) -> Self {
        let Sources {
            heights,
            colours,
            maxima,
        } = sources;
        let raster = UVec2::new(placement.width, placement.height);
        // Only the sources whose levels *are* clipmap levels. The max pyramid's
        // are depths of a quadtree over the same raster and run further, so
        // folding it in here would cost real levels of terrain.
        let available = heights.level_count().min(colours.level_count());
        // The caller asks for the window the screen would like; how much of it
        // is affordable depends on the raster, which only becomes known here.
        let config = ClipmapConfig {
            window_texels: config.fit_window(raster, available),
            ..config
        };
        let level_count = config.level_count(raster, available).min(MAX_LEVELS as u32);
        let window = config.window_texels;

        // The coarsest thing that will ever be asked for: the outermost level's
        // coarsest depth. A source that stops short of it would answer with its
        // own top level clamped, which bounds a smaller square than the cell
        // covers and so is not a bound at all.
        let deepest = level_count - 1 + config.max_mip();
        assert!(
            maxima.level_count() > deepest,
            "the max pyramid reaches level {} but level {} is needed for {level_count} \
             clipmap levels of {window} texels",
            maxima.level_count() - 1,
            deepest
        );
        log::info!(
            "clipmap: {level_count} levels of {window} texels, \
             {:.0} MiB of texture, reaching {} texels from the camera",
            config.texture_bytes(level_count) as f64 / (1 << 20) as f64,
            (config.window_quads() / 2) << (level_count - 1),
        );

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
        // array indexes clipmap level, the mips index quadtree depth. It stops
        // three short of a single texel per layer; see `ClipmapConfig::max_mip`.
        // `COPY_SRC` is not needed to draw, only so a test can read the pyramid
        // back and check it against a reference built on the CPU -- which is
        // cheap enough to be worth paying for always rather than building a
        // differently-shaped texture under `cfg(test)`.
        let maxima_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain maxima"),
            size: layers,
            mip_level_count: config.max_mip() + 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Half precision, which is worth three hundred megabytes at the
            // widest window. `ceiling_half` rounds towards positive infinity so
            // that a cell stays an upper bound on the ground under it; see the
            // reasoning there.
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
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
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    // Both stages: the vertex stage places the mesh on it, and
                    // the fragment stage reads it at the leaf of a raymarch.
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                resource: wgpu::BindingResource::TextureView(&array_view(&maxima_texture)),
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
        let ceiling_texels = ((window >> config.max_mip()) as usize).pow(2);
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
            maxima,
            ceilings: vec![vec![f32::NEG_INFINITY; ceiling_texels]; level_count as usize],
            height_texture,
            colour_texture,
            maxima_texture,
            uniform,
            bind_group,
            pipeline,
            far_pipeline,
            far_group,
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
            max_mip: self.config.max_mip(),
            window_quads: self.config.window_quads() as f32,
            grid_offset: self.config.margin() as f32,
            // Across every level being marched, because a ray leaving a fine
            // window carries on into a coarse one and could meet either.
            ceiling: self.ceilings[base as usize..]
                .iter()
                .map(|level| highest(level))
                .fold(f32::NEG_INFINITY, f32::max),
            march_steps: self.config.march_steps(levels as u32),
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
                ceiling: highest(&self.ceilings[level]),
                padding: [0.0; 3],
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
        let filled = self.filled[level];
        let whole = |origin: IVec2, span: u32| crate::terrain::clipmap::Rect {
            x: origin.x,
            y: origin.y,
            width: span,
            height: span,
        };
        let regions = if filled {
            exposed_regions(self.origins[level], new, window)
        } else {
            vec![whole(new, window)]
        };

        for region in regions {
            for (piece, destination) in split_across_seam(region, window) {
                self.upload(queue, level as u32, piece, destination);
            }
        }

        // The pyramid's cells are anchored to the raster, so depth `m`'s window
        // is the level's own origin divided down -- an arithmetic shift, which
        // floors for negative origins exactly as the cells do. That is what
        // makes the same incremental machinery work for every depth: each has
        // its own origin, its own span, and its own seam, and nothing about it
        // depends on where the window happens to have stopped.
        //
        // A depth only moves once the origin crosses a multiple of `2^m`, so
        // most frames touch only the finest one or two.
        for mip in 0..=self.config.max_mip() {
            let (was, now) = (self.origins[level] >> mip, new >> mip);
            let span = window >> mip;
            let regions = if filled {
                exposed_regions(was, now, span)
            } else {
                vec![whole(now, span)]
            };
            for region in regions {
                for (piece, destination) in split_across_seam(region, span) {
                    self.upload_maxima(queue, level as u32, mip, piece, destination);
                }
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

    /// Copies one seam-free piece of one depth of the max pyramid.
    fn upload_maxima(
        &mut self,
        queue: &wgpu::Queue,
        level: u32,
        mip: u32,
        piece: crate::terrain::clipmap::Rect,
        destination: UVec2,
    ) {
        let size = piece.size();
        let texels = (size.x * size.y) as usize;
        let bytes = texels * size_of::<f32>();
        // `level + mip`, not `mip`: the pyramid is one chain over the raster
        // rather than one per clipmap level, and level `l`'s depth `m` cell is
        // exactly the cell level `l + m` of it holds, at the same index. See
        // [`crate::terrain::maxima`].
        self.maxima.read_rect(
            level + mip,
            piece.origin(),
            size,
            &mut self.staging[..bytes],
        );
        // Narrowed in place, forwards, so that the half float for a cell is
        // written over bytes the full-precision one has already been read out
        // of. Two bytes never overtake four.
        for cell in 0..texels {
            let read = f32::from_le_bytes(
                self.staging[cell * 4..cell * 4 + 4]
                    .try_into()
                    .expect("four bytes"),
            );
            let narrowed = ceiling_half(read * VERTICAL_EXAGGERATION).to_bits();
            self.staging[cell * 2..cell * 2 + 2].copy_from_slice(&narrowed.to_le_bytes());
        }
        let bytes = texels * size_of::<u16>();

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.maxima_texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: destination.x,
                    y: destination.y,
                    z: level,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &self.staging[..bytes],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.x * size_of::<u16>() as u32),
                rows_per_image: Some(size.y),
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );

        // The coarsest depth is small enough to mirror whole, and mirroring it
        // from the same staging buffer at the same destination keeps it wrapped
        // around the torus exactly as the texture is. Read back out of the half
        // floats rather than kept at full precision, so that the scalar the far
        // field takes its early out from is the same bound the texture holds.
        if mip == self.config.max_mip() {
            let cells: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&self.staging[..bytes])
                .iter()
                .map(|bits| half::f16::from_bits(*bits).to_f32())
                .collect();
            let span = (self.config.window_texels >> mip) as usize;
            for row in 0..size.y as usize {
                let from = row * size.x as usize;
                let to = (destination.y as usize + row) * span + destination.x as usize;
                self.ceilings[level as usize][to..to + size.x as usize]
                    .copy_from_slice(&cells[from..from + size.x as usize]);
            }
        }
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
    /// has three depths to disagree over.
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

    /// The highest sample of one level across the closed square a cell covers.
    ///
    /// An oracle deliberately independent of everything the upload path does:
    /// it knows only which ground a cell is named after, and asks the source
    /// what stands on it. This is what the *march* needs a cell to bound when
    /// that level is the one reading it.
    fn cell_ceiling(source: &dyn RasterSource, level: u32, mip: u32, cell: IVec2) -> f32 {
        let span = 1i32 << mip;
        let side = span as u32 + 1;
        let mut samples = vec![0f32; (side * side) as usize];
        source.read_rect(
            level,
            cell * span,
            UVec2::splat(side),
            bytemuck::cast_slice_mut(&mut samples),
        );
        terrain_tiles::maxima::highest(&samples)
    }

    /// What a depth is defined to hold: the greatest of [`cell_ceiling`] over
    /// every level that could read it.
    ///
    /// The bound each level asks for differs, and a cell has to satisfy all of
    /// them at once, because which level reads a given depth depends on where
    /// the camera is rather than on anything the pyramid knows.
    fn cell_defined(source: &dyn RasterSource, depth: u32, cell: IVec2) -> f32 {
        (0..=depth.min(source.level_count() - 1))
            .map(|level| cell_ceiling(source, level, depth - level, cell))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Every texel of the pyramid bounds the ground it claims, and sits where
    /// the raymarcher will go looking for it.
    ///
    /// Both halves matter and they fail differently. A wrong *value* lets rays
    /// through ridges; a wrong *position* is worse, because the cells are
    /// anchored to the raster while the window is not, so a version that walked
    /// the cells in window coordinates would read the ceiling of a neighbouring
    /// cell and be wrong by one cell everywhere except when the origin happened
    /// to land on a multiple of the cell size.
    #[test]
    fn every_cell_of_the_pyramid_bounds_the_ground_it_covers() {
        let (device, queue) = crate::scene::test_device();
        let camera_layout = crate::scene::test_camera_layout(&device);
        let config = test_config();
        let window = config.window_texels;

        // The raster the cells are defined over, kept aside so the oracle can
        // look their squares up in it directly rather than through anything the
        // upload path touched.
        let raster = Pyramid::build(Level::new(RASTER, RASTER, rugged()));
        let mut terrain = Terrain::new(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &camera_layout,
            config,
            Georeferencing::square(RASTER, RASTER, 30.0),
            Sources {
                heights: Box::new(Pyramid::build(Level::new(RASTER, RASTER, rugged()))),
                colours: Box::new(Pyramid::build(Level::new(
                    RASTER,
                    RASTER,
                    vec![Srgb8([0, 0, 0, 255]); (RASTER * RASTER) as usize],
                ))),
                maxima: Box::new(crate::terrain::pyramid::max_pyramid(&raster)),
            },
        );
        let raster: &dyn RasterSource = &raster;

        // Off any round number, so that a window origin is not a multiple of a
        // coarse cell and reading the cells in window order would be visibly
        // wrong. Low enough that no level is dropped for altitude, so every
        // layer is uploaded and every one of them is checked.
        terrain.update(&queue, Vec3::new(137.0, 100.0, -71.0));
        assert_eq!(terrain.base_level(), 0, "the test wants every level built");
        let levels = terrain.origins.len() as u32;

        // Rows in a texture-to-buffer copy are padded to 256 bytes, which the
        // coarser mips of a 32-texel window are well short of.
        let stride = |size: u32| (size * 2).div_ceil(256) * 256;
        let readbacks: Vec<wgpu::Buffer> = (0..=config.max_mip())
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
        for (mip, readback) in readbacks.iter().enumerate() {
            let size = (window >> mip).max(1);
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &terrain.maxima_texture,
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

        let (mut checked, mut exact) = (0, 0);
        for level in 0..levels {
            for (mip, readback) in readbacks.iter().enumerate() {
                let mip = mip as u32;
                let span = (window >> mip).max(1) as i32;
                let stride = stride(span as u32) as usize;
                let bytes = readback.get_mapped_range(..).expect("buffer not mapped");
                let layer = &bytes[level as usize * stride * span as usize..];

                // Where the window's first cell falls on the raster. An
                // arithmetic shift, which floors for the negative origins a
                // window near the raster's edge legally has.
                let base = terrain.origins[level as usize] >> mip;
                for j in 0..span {
                    for i in 0..span {
                        let cell = base + IVec2::new(i, j);
                        let (x, y) = (cell.x.rem_euclid(span), cell.y.rem_euclid(span));
                        let row: &[u16] = bytemuck::cast_slice(
                            &layer[y as usize * stride..y as usize * stride + span as usize * 2],
                        );
                        let got = half::f16::from_bits(row[x as usize]).to_f32();

                        // The definition: the greatest bound any level reading
                        // depth `level + mip` asks for. Equality pins both the
                        // values and the `level + mip` indexing, which is the
                        // part a version reading its own level would get right
                        // only where the two happen to coincide.
                        //
                        // Asked only of cells whose square lands on real ground.
                        // A window legitimately hangs off the raster, and out
                        // there both the pyramid and this oracle repeat the
                        // border -- but at their own granularities, so they
                        // agree on the bound without agreeing on the figure.
                        let reach = (cell + IVec2::ONE) << (level + mip);
                        if cell.min_element() >= 0 && reach.max_element() < RASTER as i32 {
                            let want = cell_defined(raster, level + mip, cell);
                            assert_eq!(
                                got,
                                ceiling_half(want).to_f32(),
                                "level {level} mip {mip} cell {cell} at ({x}, {y}) \
                                 holds {got} where the levels reading it want {want}"
                            );
                            exact += 1;
                        }

                        // ... and the property the march actually leans on,
                        // stated in the terms it works in: the cell is at or
                        // above every height sample of *this* level over the
                        // closed square it covers. Below would be a ridge a ray
                        // can pass through; above only costs a descent.
                        let meets = cell_ceiling(terrain.heights.as_ref(), level, mip, cell);
                        assert!(
                            got >= meets,
                            "level {level} mip {mip} cell {cell} holds {got} \
                             for level-{level} ground at {meets}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 1000, "only {checked} cells were checked");
        assert!(
            exact > 100,
            "only {exact} cells sat wholly on the raster, which is too few to pin the definition"
        );
    }
}
