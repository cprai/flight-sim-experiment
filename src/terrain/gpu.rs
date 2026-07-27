//! The GPU side of the clipmap: textures, pipeline, and per-frame updates.

use std::ops::Range;

use std::path::Path;

use anyhow::{Context, Result};
use glam::{IVec2, UVec2, Vec2, Vec3};
use wgpu::util::DeviceExt;

use crate::terrain::clipmap::{ClipmapConfig, exposed_regions, split_across_seam, window_origin};
use crate::terrain::geotiff::Georeferencing;
use crate::terrain::mesh::{self, PatchKind};
use crate::terrain::pyramid::{RasterSource, Srgb8};
use crate::terrain::tiles::TileStore;

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
    /// Cleared until the first update has filled every window.
    windows_filled: bool,
    /// Reused between uploads so a moving camera allocates nothing.
    staging: Vec<u8>,

    height_texture: wgpu::Texture,
    colour_texture: wgpu::Texture,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,

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

        Ok(Self::new(
            device,
            format,
            camera_layout,
            config,
            placement,
            Box::new(heights),
            Box::new(colours),
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
        let window = config.window_texels();

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
                    visibility: wgpu::ShaderStages::VERTEX,
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

        let shader = device.create_shader_module(wgpu::include_wgsl!("../terrain.wgsl"));
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

        let height_range = Self::coarsest_height_range(heights.as_ref(), &placement);
        let window_bytes = (window * window) as usize;
        Self {
            config,
            placement,
            heights,
            colours,
            height_range,
            origins: vec![IVec2::ZERO; level_count as usize],
            windows_filled: false,
            staging: vec![0; window_bytes * size_of::<Srgb8>().max(size_of::<f32>())],
            height_texture,
            colour_texture,
            uniform,
            bind_group,
            pipeline,
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

    /// Moves every level's window to follow the camera, uploading only the
    /// ground that has come into view since the last call.
    pub fn update(&mut self, queue: &wgpu::Queue, camera: Vec3) {
        let camera_texels = self
            .placement
            .texel_of_world(f64::from(camera.x), f64::from(camera.z));
        let window = self.config.window_texels();

        let levels = self.origins.len();
        let (data_min, data_max) = self.placement.data_bounds();
        let mut uniform = TerrainUniform {
            levels: [LevelUniform::default(); MAX_LEVELS],
            level_count: levels as u32,
            window_mask: window - 1,
            morph_band: self.config.morph_band,
            grid_quads: self.config.grid_quads() as f32,
            data_min: [data_min.0 as f32, data_min.1 as f32],
            data_max: [data_max.0 as f32, data_max.1 as f32],
        };

        // Every window's new position is settled before any of them is
        // described, because each level's morph needs to know where the level
        // outside it ended up.
        let placed: Vec<IVec2> = (0..levels as u32)
            .map(|level| window_origin(&self.config, level, camera_texels))
            .collect();

        for (level, &new) in placed.iter().enumerate() {
            let regions = if self.windows_filled {
                exposed_regions(self.origins[level], new, window)
            } else {
                // Nothing is resident yet, so every window is entirely new.
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
        self.windows_filled = true;

        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        self.rebuild_patches(queue);
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

        let bytes = texels * size_of::<Srgb8>();
        self.colours
            .read_rect(level, piece.origin(), size, &mut self.staging[..bytes]);
        copy(&self.colour_texture, 4, &self.staging[..bytes]);
    }

    /// The lowest and highest elevation in the coarsest level of the pyramid.
    ///
    /// Read from the top rather than the base: the coarsest level is one or two
    /// tiles rather than the whole dataset, and every texel in it is a box filter of
    /// everything beneath, so the range is representative without anything being
    /// scanned that is not already resident for a moment. Peaks are averaged down a
    /// little, which only matters for framing the camera at startup -- the one thing
    /// this is used for.
    fn coarsest_height_range(source: &dyn RasterSource, placement: &Georeferencing) -> (f32, f32) {
        let level = source.level_count().saturating_sub(1);
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
    fn rebuild_patches(&mut self, queue: &wgpu::Queue) {
        // Rings reach beyond the raster, and the fragment stage cuts away
        // whatever falls outside it. A patch lying wholly out there would have
        // every one of its fragments thrown away, so drop it here instead and
        // never rasterize it at all. The saving is largest exactly where it
        // matters: a coarse ring viewed from over a corner of the data.
        let (data_min, data_max) = self.placement.data_bounds();
        let patches: Vec<mesh::Patch> = mesh::patches(&self.config, &self.origins)
            .into_iter()
            .filter(|patch| {
                let level = patch.level as usize;
                let scale = f64::from(1u32 << level);
                let (near_x, near_z) = self.placement.world_of_texel(
                    patch.level,
                    f64::from(self.origins[level].x + patch.origin.x as i32),
                    f64::from(self.origins[level].y + patch.origin.y as i32),
                );
                let size = patch.kind.size_quads(&self.config);
                let far_x = near_x + f64::from(size.x) * scale * self.placement.metres_per_texel_x;
                let far_z = near_z + f64::from(size.y) * scale * self.placement.metres_per_texel_z;

                far_x >= data_min.0
                    && near_x <= data_max.0
                    && far_z >= data_min.1
                    && near_z <= data_max.1
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
}
