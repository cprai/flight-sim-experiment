//! The GPU side of the terrain: three texture arrays, one pipeline, and the
//! tiles that stream into them.
//!
//! Every level holds a square of whole tiles around the camera, and drawing is
//! one compute dispatch that raymarches them. There is no mesh, no near field
//! and no sliding window: the camera moving within a tile costs nothing, and
//! crossing out of one costs a bounded number of whole-tile reads. See
//! [`crate::terrain::residency`] for how the squares move and `src/terrain.wgsl`
//! for what the march does with them.
//!
//! The dispatch writes the G-buffer rather than the screen: each pixel gets the
//! material id and world position of the ground its ray met, and
//! [`crate::deferred`]'s shading pass turns those into colour afterwards.

use std::path::Path;
use std::time::Duration;

use crate::terrain::geotiff::Georeferencing;
use crate::terrain::pyramid::RasterSource;
use crate::terrain::residency::{Residency, TileResidency, Wanted, detail_base};
use crate::terrain::tiles::{MaterialId, TileStore};
use anyhow::{Context, Result};
use glam::{DVec2, IVec2, UVec2, Vec2, Vec3};

/// Must match `MAX_LEVELS` in the shader.
const MAX_LEVELS: usize = 16;

/// Vertical scale applied to the height raster, for when terrain needs
/// exaggerating to read clearly. One means true to the data.
const VERTICAL_EXAGGERATION: f32 = 1.0;

/// Mirrors `Level` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LevelUniform {
    valid_low: [i32; 2],
    valid_high: [i32; 2],
    ceiling: f32,
    padding: f32,
    /// All ones for a resident level, whose index is its texture coordinate.
    mask: [i32; 2],
}

/// Which mip of the heights is mirrored on the CPU, as an offset from the base.
///
/// Four, so 128 m texels on an 8 m base: 768 x 896 for this raster, 2.75 MiB
/// against the 64 MiB the old mirror of a whole clipmap level cost. Coarse
/// enough to be small and to answer with the ground *averaged* over kilometres,
/// which is the right shape of answer for choosing a level that covers
/// kilometres; fine enough that a camera in a valley is not told it is standing
/// on the ridge beside it.
const GROUND_MIP: u32 = 4;

/// Mirrors `Terrain` in the shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainUniform {
    levels: [LevelUniform; MAX_LEVELS],
    origin: [f32; 2],
    metres_per_texel: [f32; 2],
    data_min: [f32; 2],
    data_max: [f32; 2],
    level_count: u32,
    base_level: u32,
    resident_base: u32,
    march_steps: u32,
    ceiling: f32,
    wall_nudge: f32,
    /// The target's size in pixels, which turns a pixel index into a ray.
    ///
    /// Occupies what was tail padding, so the uniform is the same size and
    /// every other member sits where it did.
    viewport: [u32; 2],
}

/// Mirrors `DetailJob` in the shader: one rectangle of one level to generate.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DetailJob {
    origin: [i32; 2],
    size: [u32; 2],
    level: u32,
    octaves: u32,
    wavelength: f32,
    relief: f32,
}

/// Threads per side of a `cs_detail` workgroup.
///
/// Must match `@workgroup_size` on `cs_detail` in `src/terrain.wgsl`.
const DETAIL_GROUP: u32 = 8;

/// Mirrors `MaximaJob` in the shader: one rectangle of one level to derive.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MaximaJob {
    origin: [i32; 2],
    size: [u32; 2],
    level: u32,
    carry: u32,
    below_low: [i32; 2],
    below_high: [i32; 2],
    stride: u32,
    padding: u32,
}

/// Side of the widest rectangle one derive job covers, in cells.
///
/// Bounds the scratch buffer the cells go out through, which is the only thing
/// this costs: a rectangle wider than this is cut into several jobs, and a
/// square being derived whole is cut into a few dozen. A thousand and
/// twenty-four cells is a two mebibyte buffer at the stride below.
const DERIVE_CHUNK: u32 = 1024;

/// Threads per side of a `cs_maxima` workgroup.
///
/// Must match `@workgroup_size` on `cs_maxima` in `src/terrain.wgsl`.
const DERIVE_GROUP: u32 = 8;

/// Bytes between the slots of the job uniform.
///
/// A dynamic uniform offset must be a multiple of
/// `min_uniform_buffer_offset_alignment`, which is 256 at the limits this
/// device asks for, so a job occupies far more room than it fills.
const JOB_SLOT: u64 = 256;

/// Bytes a row of a buffer being copied into a texture must be a multiple of.
const COPY_ALIGN: u32 = 256;

/// Bytes one row of `derive_scratch` occupies for a rectangle this wide.
///
/// Two bytes a cell, packed in pairs because a storage buffer cannot be written
/// in anything smaller than four, and rounded up to what a copy demands.
fn derive_row_bytes(cells: u32) -> u32 {
    (cells.div_ceil(2) * 4).div_ceil(COPY_ALIGN) * COPY_ALIGN
}

/// How much of `[at, end)` one derive job may take along an axis.
///
/// As much as the scratch buffer holds, and never past the next wrap: a
/// rectangle straddling one is two rectangles in the texture, and the copy out
/// of the buffer would have to start partway along a row at an offset it is not
/// allowed to start at. An identity mask is a level that does not wrap at all,
/// which is every resident one.
fn span(at: i32, end: i32, mask: i32) -> u32 {
    let to_wrap = if mask == -1 {
        i32::MAX
    } else {
        mask + 1 - (at & mask)
    };
    (end - at).min(to_wrap).min(DERIVE_CHUNK as i32) as u32
}

/// How far past a cell wall the march has to put a ray for the next step to
/// land in the next cell, in level-0 texels.
///
/// A fixed fraction of a texel is not enough, and that is not obvious until the
/// raster is large. The march works in texel indices measured from the raster's
/// north-west corner, which on a survey a hundred kilometres across reach six
/// figures, and consecutive `f32` values up there are 0.008 texels apart. The
/// thousandth of a texel this used to add rounded straight back off: `t` did not
/// change, the ray landed exactly on the wall it had just left, and whether it
/// escaped came down to which way the multiply happened to round. It took about
/// sixteen iterations to leave one cell, so a ray crossed 8 metres per step
/// instead of the 128 its level was worth, ran out of budget 33 km out, and was
/// painted as ground where it stopped -- the smeared wall across the distance.
///
/// So take the step from the size of the numbers. `f32::EPSILON` times the
/// largest index in play is one to two of those gaps; eight times that is
/// comfortably clear of them and still a tenth of a texel, far below the finest
/// ground drawn and far below the pixel it lands in.
///
/// This holds while the two bounds have room between them, which on a `f32` is
/// up to a raster of about a quarter of a million texels a side, a bit over
/// twice the one installed. Past that the nudge a ray needs to leave a wall is a
/// sizeable fraction of the texel it is leaving, and the traversal would have to
/// carry the cell as an integer and step it, the way a Bresenham-style DDA does,
/// rather than deriving it from a position each time. That is a larger change
/// than this bug is worth on its own, and it would still want a rule for what
/// happens when the level changes mid-cell.
fn wall_nudge(raster: UVec2) -> f32 {
    8.0 * raster.max_element() as f32 * f32::EPSILON
}

/// The two rasters the terrain is drawn from, both describing one piece of
/// ground.
///
/// There is no max pyramid here any more. It was a third product carried
/// alongside these, and it is now derived on the GPU from the heights once they
/// are resident -- see [`Terrain::build_pyramid`].
pub struct Sources {
    pub heights: Box<dyn RasterSource>,
    pub materials: Box<dyn RasterSource>,
}

/// A height raster and a matching material raster, raymarched through a max
/// pyramid.
///
/// All three live in GPU memory whole, as mip chains over the raster from
/// [`Residency::resident_base`] upwards. Nothing streams and nothing moves, so
/// most of what used to be here -- the square of tiles per level, the queue of
/// what to load next, the CPU mirrors that had to be kept in step with it -- is
/// gone.
pub struct Terrain {
    residency: Residency,
    placement: Georeferencing,
    height_range: (f32, f32),

    /// The level mip zero holds, after [`Residency::fit_base`] has had its say.
    resident_base: u32,
    /// Size of mip zero, in its own texels.
    base_size: UVec2,
    /// Mips in each chain, so levels `resident_base .. resident_base + this`.
    mips: u32,
    /// The finest level worth descending to, from [`detail_base`].
    base: u32,

    /// A CPU mirror of one coarse mip of the heights.
    ///
    /// Height above terrain decides the finest level worth descending to, and
    /// asking the GPU would mean a readback and a stall. This is the same
    /// answer a level of the chain holds, kept on the way past during the load;
    /// [`GROUND_MIP`] says why that level and not another.
    ground: Vec<f32>,
    /// Size of the mirrored mip, and the level it is.
    ground_size: UVec2,
    ground_level: u32,
    /// The highest ground anywhere, off the top of the max pyramid.
    ///
    /// A ray above it and climbing is sky, which is most of a horizon view.
    /// One figure rather than one per level: every level bounds the same whole
    /// raster now, so their maxima differ only by how coarsely each closes it.
    ceiling: f32,
    /// Where the chain is read from, until it has been.
    ///
    /// Taken at the first update and dropped there. Nothing reads a tile after
    /// that, which is the whole point: a file handle held past the load is a
    /// file handle something might be tempted to use.
    sources: Option<Sources>,

    /// Where an update's time went, when a run asked to be told.
    ///
    /// [`None`] on an unprofiled run, so the clock is not read at all.
    spans: Option<crate::profile::Terrain>,
    /// The target's size in pixels, passed through to the march so it can turn
    /// a pixel into the ray through it. Followed by [`Terrain::resize`].
    viewport: UVec2,

    /// Which tiles each generated level holds, and what to fill next.
    windows: TileResidency,
    /// Texels across one generated level's window.
    detail_across: u32,
    /// Tiles generated since the last derive, waiting to be turned into jobs.
    generated: Vec<Wanted>,
    /// This update's generation rectangles.
    detail_jobs_cpu: Vec<DetailJob>,

    /// The chain's rectangles, coarsest last.
    ///
    /// Ordered by level because a cell carries from the level below, which must
    /// therefore be finished across every rectangle before this one starts.
    jobs: Vec<MaximaJob>,
    /// How many rectangles [`Terrain::derive_jobs`] has room for.
    job_slots: usize,

    height_texture: wgpu::Texture,
    material_texture: wgpu::Texture,
    maxima_texture: wgpu::Texture,
    /// The generated levels: one array layer per level below the base.
    ///
    /// Kept alive by the views the bind groups hold, so nothing outside a test
    /// reaches for the texture itself -- but owning it here is what says the
    /// terrain owns the memory.
    #[allow(dead_code, reason = "read by the tests that check what was generated")]
    detail_height_texture: wgpu::Texture,
    detail_maxima_texture: wgpu::Texture,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Fills one rectangle of one generated level.
    generate: wgpu::ComputePipeline,
    /// Group 1 for [`Terrain::generate`], which is the terrain *without* the
    /// generated textures: it writes one of them, and a texture may not be
    /// bound as writable storage and read as a texture in the same dispatch.
    generate_terrain_group: wgpu::BindGroup,
    /// Group 3 for [`Terrain::generate`]: the job, and the layer it writes.
    generate_bind_group: wgpu::BindGroup,
    /// One generation job per 256-byte slot, addressed by dynamic offset.
    generate_jobs: wgpu::Buffer,
    /// Derives one rectangle of one level from the heights and the level below.
    derive: wgpu::ComputePipeline,
    /// Group 3 for [`Terrain::derive`]: the job, and the cells it writes.
    derive_bind_group: wgpu::BindGroup,
    /// One job per 256-byte slot, addressed by dynamic offset.
    derive_jobs: wgpu::Buffer,
    /// Where one rectangle's cells land on their way to the texture.
    derive_scratch: wgpu::Buffer,
    /// Settles every pixel it can from the reprojection and lists the rest.
    compact: wgpu::ComputePipeline,
    /// Turns that list's length into a dispatch size for `march`.
    args: wgpu::ComputePipeline,
    /// Casts a ray for each pixel on the list, and nothing for any other.
    march: wgpu::ComputePipeline,
    /// Reduces the finished motion field to one number per dither cell.
    risk: wgpu::ComputePipeline,
    /// Spreads that field outwards over the distance each cell's motion covers.
    reach: wgpu::ComputePipeline,
}

/// Side of the square of pixels one workgroup of the compaction covers.
///
/// Must match `@workgroup_size` on `cs_compact` in `src/terrain.wgsl`.
const COMPACT_TILE: u32 = 8;

impl Terrain {
    /// Opens the tile pyramids and builds the terrain around them.
    ///
    /// `root` holds one directory per product: elevation, materials, and the
    /// max pyramid `terrain-process` reduced from the elevation. Nothing is
    /// decoded here beyond three manifests -- the tiles themselves are read as
    /// the camera reaches them.
    pub fn from_tiles(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        storage_layout: &wgpu::BindGroupLayout,
        work_layout: &wgpu::BindGroupLayout,
        args_layout: &wgpu::BindGroupLayout,
        risk_layout: &wgpu::BindGroupLayout,
        reach_layout: &wgpu::BindGroupLayout,
        residency: Residency,
        viewport: UVec2,
        root: &Path,
    ) -> Result<Self> {
        let product = crate::terrain::ELEVATION_PRODUCT;
        let elevation = root.join(product);
        anyhow::ensure!(
            elevation.is_dir(),
            "{} holds no {product} directory",
            root.display()
        );
        let material = root.join(terrain_tiles::MATERIAL_PRODUCT);

        let heights = TileStore::<f32>::open(&elevation)?;
        let materials = TileStore::<MaterialId>::open(&material).with_context(|| {
            format!(
                "{} holds no ground-cover materials; run terrain-process over a download \
                 with an osm extract",
                root.display()
            )
        })?;
        // Structural rather than approximate: both manifests descend from one
        // download over one snapped extent, so they either describe the same
        // ground exactly or one of them is from a different run.
        anyhow::ensure!(
            heights.manifest().covers_same_ground_as(materials.manifest()),
            "{} and {} do not cover the same ground",
            elevation.display(),
            material.display()
        );
        // `<product>-max` is no longer opened at all. The pyramid is built on
        // the GPU out of the heights once they are resident, which is one
        // spelling of the recurrence instead of two and 57 GB of tiles that no
        // longer have to exist. See `build_pyramid`.

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
            camera_layout,
            storage_layout,
            work_layout,
            args_layout,
            risk_layout,
            reach_layout,
            residency,
            viewport,
            placement,
            Sources {
                heights: Box::new(heights),
                materials: Box::new(materials),
            },
        ))
    }

    pub fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        storage_layout: &wgpu::BindGroupLayout,
        work_layout: &wgpu::BindGroupLayout,
        args_layout: &wgpu::BindGroupLayout,
        risk_layout: &wgpu::BindGroupLayout,
        reach_layout: &wgpu::BindGroupLayout,
        residency: Residency,
        viewport: UVec2,
        placement: Georeferencing,
        sources: Sources,
    ) -> Self {
        let Sources { heights, materials } = sources;
        let raster = UVec2::new(placement.width, placement.height);
        let available = heights.level_count().min(materials.level_count());
        // The caller asks for the resolution it would like; whether the device
        // and the budget will take it depends on the raster, which only becomes
        // known here.
        let residency = Residency {
            resident_base: residency.fit_base(raster, available, device.limits().max_texture_dimension_2d),
            ..residency
        };
        let resident_base = residency.resident_base;
        let base_size = residency.base_size(raster);
        let mips = Residency::mip_count(base_size).min(MAX_LEVELS as u32 - resident_base);
        log::info!(
            "terrain: {} x {} texels at level {resident_base}, {mips} mips, {:.0} MiB of \
             texture, the whole raster resident",
            base_size.x,
            base_size.y,
            Residency::texture_bytes(base_size, mips) as f64 / (1 << 20) as f64,
        );
        assert!(
            available > resident_base,
            "the products reach level {} but level {resident_base} is being held",
            available.saturating_sub(1)
        );

        let chain = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: base_size.x,
                    height: base_size.y,
                    depth_or_array_layers: 1,
                },
                // The chain *is* the quadtree now. It could not be while a
                // level was a window of fixed texel count over doubling ground
                // -- a mip halves resolution at a fixed extent, which is the
                // other thing entirely -- and it can be the moment the whole
                // raster is resident.
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        // `COPY_SRC` on the maxima is not needed to draw, only so the ceilings
        // can be read back off the top of the chain and so a test can check the
        // pyramid against a reference built on the CPU.
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        // `COPY_SRC` so a test can read a level back; see the maxima below.
        let height_texture = chain(
            "terrain heights",
            wgpu::TextureFormat::R32Float,
            usage | wgpu::TextureUsages::COPY_SRC,
        );
        // Sixteen bits rather than thirty-two. Ids reach 0x080c and the palette
        // is 2304 entries, so the top half was never carrying anything, and at
        // this size that half was 470 MB of it.
        let material_texture = chain("terrain materials", wgpu::TextureFormat::R16Uint, usage);
        let maxima_texture = chain(
            "terrain maxima",
            // Half precision, which is worth half the memory. `ceiling_half`
            // rounds towards positive infinity so a texel stays an upper bound on
            // the ground under it; see the reasoning there.
            wgpu::TextureFormat::R16Float,
            usage | wgpu::TextureUsages::COPY_SRC,
        );

        // The generated levels. One layer per level below the base, all the
        // same width, wrapped onto their slots -- which is the clipmap the
        // resident chain replaced, kept for exactly the levels that still have
        // to move. `resident_base` of zero means there is nothing under the
        // base to generate, and a texture may not have no layers, so it
        // degenerates to one texel that nothing ever reads.
        let detail_levels = resident_base.max(1);
        let detail_across = if resident_base > 0 {
            residency.detail_tiles * residency.detail_tile_texels
        } else {
            // Nothing under the base to generate, and a texture may not be
            // empty. One texel that nothing ever reads.
            1
        };
        let window = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: detail_across,
                    height: detail_across,
                    depth_or_array_layers: detail_levels,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        // `COPY_SRC` on both is not needed to draw, only so a test can read a
        // window back and check the pyramid over it against its own heights.
        let detail_height_texture = window(
            "terrain detail heights",
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
        );
        let detail_maxima_texture = window(
            "terrain detail maxima",
            wgpu::TextureFormat::R16Float,
            wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
        );
        if resident_base > 0 {
            log::info!(
                "terrain: {resident_base} generated levels of {} x {} tiles, {detail_across} \
                 texels each, {:.0} MiB, reaching {} texels from the camera",
                residency.detail_tiles,
                residency.detail_tiles,
                (detail_across as usize).pow(2) * 6 * resident_base as usize / (1 << 20),
                residency.detail_reach() << (resident_base - 1),
            );
        }

        let array_view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let layer_view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain uniform"),
            size: size_of::<TerrainUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let generated_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let layout_entries = [
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    // Integer texels: material ids can only be loaded, never
                    // sampled, so there is no sampler anywhere in this layout.
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
        ];
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain layout"),
            entries: &[
                layout_entries[0],
                layout_entries[1],
                layout_entries[2],
                layout_entries[3],
                generated_entry(4),
                generated_entry(5),
            ],
        });
        // The same terrain without the generated arrays. `cs_detail` writes one
        // of them, and wgpu will not have a texture bound as writable storage
        // and read as a texture in one dispatch -- so the pass that generates a
        // level binds only what it reads, which is the base.
        let base_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain base layout"),
            entries: &layout_entries[..4],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain bind group"),
            layout: &layout,
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
                    resource: wgpu::BindingResource::TextureView(&array_view(&material_texture)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&array_view(&maxima_texture)),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&layer_view(
                        &detail_height_texture,
                    )),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&layer_view(
                        &detail_maxima_texture,
                    )),
                },
            ],
        });
        let generate_terrain_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain base bind group"),
            layout: &base_layout,
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
                    resource: wgpu::BindingResource::TextureView(&array_view(&material_texture)),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&array_view(&maxima_texture)),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../terrain.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain pipeline layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(&layout),
                Some(storage_layout),
                Some(work_layout),
            ],
            immediate_size: 0,
        });
        // `cs_args` gets its own, holding only the count and the dispatch size
        // it writes. It cannot share the one above: that binds `march_args` as
        // writable storage, and the march reads the same buffer as its indirect
        // argument, which wgpu refuses to allow in one dispatch.
        let args_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain args pipeline layout"),
            bind_group_layouts: &[None, None, None, Some(args_layout)],
            immediate_size: 0,
        });
        // Likewise its own, and for a related reason: it reads the motion
        // target the march writes, which cannot be bound both ways at once. It
        // does need the terrain, for the viewport.
        let risk_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain risk pipeline layout"),
            bind_group_layouts: &[None, Some(&layout), None, Some(risk_layout)],
            immediate_size: 0,
        });
        // And its own, for the same reason once more: it reads the risk texture
        // `cs_risk` writes. It needs the terrain for the viewport, which is what
        // tells it how many cells there are.
        let reach_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain reach pipeline layout"),
                bind_group_layouts: &[None, Some(&layout), None, Some(reach_layout)],
                immediate_size: 0,
            });
        // Compute rather than a fullscreen triangle. The march never needed
        // anything the raster pipeline provides -- no interpolation, no
        // blending, and a depth test that could reject nothing because each
        // pixel was covered exactly once -- and a dispatch is the only shape
        // that can be given a list of pixels rather than a screen to cover.
        //
        // All three share one layout. `args` needs almost none of it and
        // `march` never touches the carried textures, but a pipeline may leave
        // its layout's entries alone, and one description the three agree on
        // cannot drift between them.
        let stage = |label, entry_point, layout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let compact = stage("terrain compact pipeline", "cs_compact", &pipeline_layout);
        let args = stage("terrain args pipeline", "cs_args", &args_pipeline_layout);
        let march = stage("terrain march pipeline", "cs_march", &pipeline_layout);
        let risk = stage("terrain risk pipeline", "cs_risk", &risk_pipeline_layout);
        let reach = stage("terrain reach pipeline", "cs_reach", &reach_pipeline_layout);

        // Deriving the pyramid reads the terrain, so it takes group 1 whole,
        // and writes through a group of its own. Its own layout again, for the
        // reason the three above have theirs: group 3 is where a pass's private
        // bindings go, four groups being all the device promises, and this pass
        // shares none of its with the march.
        let derive_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain derive layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        // One rectangle per dispatch, chosen by offset, so the
                        // whole update's work goes into one buffer written once.
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(size_of::<MaximaJob>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        // Every mip is swept whole in rectangles no wider than the scratch, and
        // a chain is four thirds of its base, so this is a little over four
        // thirds of what mip zero alone takes.
        let sweep: usize = (0..mips)
            .map(|mip| {
                let level = UVec2::new(base_size.x >> mip, base_size.y >> mip).max(UVec2::ONE);
                (level.x.div_ceil(DERIVE_CHUNK) * level.y.div_ceil(DERIVE_CHUNK)) as usize
            })
            .sum();
        // A generated tile raises every level above it, and each of those
        // rectangles is cut at the wrap into at most four pieces.
        let cascade = residency.detail_per_update as usize
            * (resident_base + mips) as usize
            * 4;
        let job_slots = sweep.max(cascade);
        let derive_jobs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain derive jobs"),
            size: JOB_SLOT * job_slots as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let widest = base_size.x.max(base_size.y).max(detail_across);
        let chunk = DERIVE_CHUNK.min(widest);
        let derive_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain derive cells"),
            size: u64::from(derive_row_bytes(chunk)) * u64::from(chunk),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let derive_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain derive bind group"),
            layout: &derive_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &derive_jobs,
                        offset: 0,
                        size: wgpu::BufferSize::new(size_of::<MaximaJob>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: derive_scratch.as_entire_binding(),
                },
            ],
        });
        let derive_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain derive pipeline layout"),
                bind_group_layouts: &[None, Some(&layout), None, Some(&derive_layout)],
                immediate_size: 0,
            });
        let derive = stage("terrain derive pipeline", "cs_maxima", &derive_pipeline_layout);

        // Generating a level: reads the base through group 1, writes one layer
        // of the generated heights through group 3.
        let generate_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain generate layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(size_of::<DetailJob>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });
        // A window moves by whole tiles and hands out at most
        // `detail_per_update` of them, and a tile is cut at the wrap into at
        // most four rectangles.
        let generate_slots = (residency.detail_per_update as usize * 4).max(4);
        let generate_jobs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain generate jobs"),
            size: JOB_SLOT * generate_slots as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let generate_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain generate bind group"),
            layout: &generate_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &generate_jobs,
                        offset: 0,
                        size: wgpu::BufferSize::new(size_of::<DetailJob>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: wgpu::BindingResource::TextureView(&layer_view(
                        &detail_height_texture,
                    )),
                },
            ],
        });
        let generate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("terrain generate pipeline layout"),
                bind_group_layouts: &[None, Some(&base_layout), None, Some(&generate_layout)],
                immediate_size: 0,
            });
        let generate = stage(
            "terrain generate pipeline",
            "cs_detail",
            &generate_pipeline_layout,
        );

        let height_range = Self::coarsest_height_range(heights.as_ref(), &placement);
        let ground_level = (resident_base + GROUND_MIP).min(resident_base + mips - 1);
        let ground_size = UVec2::new(
            (base_size.x >> GROUND_MIP).max(1),
            (base_size.y >> GROUND_MIP).max(1),
        );
        Self {
            residency,
            placement,
            height_range,
            resident_base,
            base_size,
            mips,
            base: resident_base,
            ground: Vec::new(),
            ground_size,
            ground_level,
            ceiling: f32::INFINITY,
            sources: Some(Sources { heights, materials }),
            windows: TileResidency::new(residency, resident_base),
            detail_across,
            generated: Vec::new(),
            detail_jobs_cpu: Vec::new(),
            spans: None,
            viewport,
            jobs: Vec::new(),
            job_slots,
            height_texture,
            material_texture,
            maxima_texture,
            detail_height_texture,
            detail_maxima_texture,
            uniform,
            bind_group,
            generate,
            generate_terrain_group,
            generate_bind_group,
            generate_jobs,
            derive,
            derive_bind_group,
            derive_jobs,
            derive_scratch,
            compact,
            args,
            march,
            risk,
            reach,
        }
    }

    /// Follows the render target to a new size.
    ///
    /// Only the pixel count the march turns into rays; how much ground is worth
    /// keeping resident is [`Residency::pixel_angle`]'s business and is decided
    /// when the scene is built.
    pub fn resize(&mut self, viewport: UVec2) {
        self.viewport = viewport;
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

    /// The finest level currently being kept.
    ///
    /// Only tests ask: it is what says whether a camera was low enough for the
    /// levels a test means to exercise to have been loaded at all.
    #[cfg(test)]
    pub fn base_level(&self) -> u32 {
        self.base
    }

    /// The levels the terrain is drawn with, finest first.
    fn levels(&self) -> std::ops::Range<u32> {
        self.resident_base..self.resident_base + self.mips
    }

    /// The coarsest level, whose single-figure ceiling clears most of the sky.
    fn coarsest(&self) -> u32 {
        self.resident_base + self.mips - 1
    }

    /// The texels of a level a ray may read, as a half-open range.
    ///
    /// Two conventions, because the two halves fail differently past their
    /// edge. A resident level advertises the whole of itself and `slot` clamps,
    /// so the last texel's patch repeats the border -- past the raster there is
    /// nothing, and repeating is what the tile store did too. A window
    /// advertises itself one texel short, because past *its* edge sits a real
    /// texel of somewhere else that the wrap would fold in.
    fn level_valid(&self, level: u32) -> (IVec2, IVec2) {
        if level >= self.resident_base {
            return (IVec2::ZERO, self.level_size(level).as_ivec2());
        }
        if level < self.base {
            return (IVec2::ZERO, IVec2::ZERO);
        }
        let tile = self.residency.detail_tile_texels as i32;
        let (low, high) = self.windows.level(level).valid(self.residency.detail_tiles);
        if low == high {
            return (IVec2::ZERO, IVec2::ZERO);
        }
        (low * tile, high * tile - IVec2::ONE)
    }

    /// What wraps a level's texel index onto its texture coordinate.
    fn level_mask(&self, level: u32) -> IVec2 {
        if level >= self.resident_base {
            IVec2::NEG_ONE
        } else {
            IVec2::splat(self.detail_across as i32 - 1)
        }
    }

    /// The size of one level, in its own texels.
    fn level_size(&self, level: u32) -> UVec2 {
        let mip = level - self.resident_base;
        UVec2::new(
            (self.base_size.x >> mip).max(1),
            (self.base_size.y >> mip).max(1),
        )
    }

    /// The highest ground anywhere, which a climbing ray above is sky.
    ///
    /// What `ground_at` and `cs_compact` test against to settle a pixel without
    /// walking anything. Off the coarsest cell of the max pyramid, so it is the
    /// bound the chain itself carries rather than a second opinion about it.
    pub fn ceiling(&self) -> f32 {
        self.ceiling
    }

    /// Whether the chain has still to be read in.
    ///
    /// True exactly once, before the first update. It is not the queue of a
    /// moving window any more -- there is no window -- but `settle` still wants
    /// to know whether a frame drawn now would draw anything.
    pub fn pending(&self) -> bool {
        self.sources.is_some() || self.windows.pending()
    }

    /// Reads the chain in on the first call, then follows the camera.
    ///
    /// The load is one-off and unbounded: it is the whole raster, and there is
    /// no frame to protect while it happens. Every update after it is a little
    /// arithmetic and one small uniform write, because nothing streams.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, camera: Vec3) {
        // Cleared rather than accumulated: a row in the readout is what this
        // frame cost, not what every frame since the run began cost.
        if let Some(spans) = self.spans.as_mut() {
            *spans = crate::profile::Terrain::default();
        }

        if let Some(sources) = self.sources.take() {
            self.load(device, queue, &sources);
        }

        let camera_texels = self
            .placement
            .texel_of_world(f64::from(camera.x), f64::from(camera.z));
        let timed = self.spans.is_some();
        let clock = crate::profile::Clock::start(timed);
        let ground = self.ground_height(camera_texels);
        let metres_per_texel = self
            .placement
            .metres_per_texel_x
            .min(self.placement.metres_per_texel_z);
        // Never finer than what is held. `detail_base` answers in absolute
        // levels, so below the base it is answering about ground that is not
        // there.
        // No longer clamped to the base: the levels under it exist again, and
        // this is what decides how many of them are worth generating.
        self.base = detail_base(
            &self.residency,
            metres_per_texel,
            f64::from(camera.y - ground),
            self.resident_base + self.mips,
        );
        // Below the base nothing is generated, and a window that has been given
        // up is refilled whole when the camera comes back down to it.
        let work = self
            .windows
            .advance(camera_texels, self.base.min(self.resident_base));
        let advance = clock.elapsed();

        // Before anything is generated or derived, because both read the chain
        // and the windows through this uniform -- the level masks above all.
        // Deriving against a stale one wrote every texel of every generated
        // level to slot zero, which reads as a shader bug and is an ordering
        // one. The same mistake, one level down, as building the chain's own
        // pyramid before its uniform existed.
        let clock = crate::profile::Clock::start(timed);
        self.write_uniform(queue);
        let uniform = clock.elapsed();

        let clock = crate::profile::Clock::start(timed);
        self.generate_tiles(device, queue, &work);
        self.raise_pyramid(device, queue);
        let generate = clock.elapsed();

        if let Some(spans) = self.spans.as_mut() {
            spans.advance += advance;
            spans.generate += generate;
            spans.write += uniform;
        }
    }

    /// Fills the tiles a window has just asked for.
    ///
    /// One pass for all of them: a tile is a whole slot, tiles are disjoint, so
    /// nothing here overlaps anything else and there is no ordering to keep.
    fn generate_tiles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, work: &[Wanted]) {
        if work.is_empty() {
            return;
        }
        let tile = self.residency.detail_tile_texels;
        // The first octave is the base's own texel, so the base cannot hold it
        // -- a level can only carry features larger than two of its texels --
        // and every level under the base carries one halving more than the one
        // outside it. That is what makes the fractal *restore* what the box
        // filter took out rather than paint something new over it.
        let base_metres = self
            .placement
            .metres_per_texel_x
            .min(self.placement.metres_per_texel_z) as f32
            * (1u32 << self.resident_base) as f32;
        self.detail_jobs_cpu.clear();
        for wanted in work {
            self.detail_jobs_cpu.push(DetailJob {
                origin: (wanted.tile * tile as i32).to_array(),
                size: [tile, tile],
                level: wanted.level,
                // One octave per level below the base: the first is the base's
                // own texel, which its Nyquist already excludes, and each level
                // under it can hold one more halving.
                octaves: self.resident_base - wanted.level,
                wavelength: base_metres,
                relief: self.residency.detail_relief,
            });
        }
        for (index, job) in self.detail_jobs_cpu.iter().enumerate() {
            queue.write_buffer(
                &self.generate_jobs,
                index as u64 * JOB_SLOT,
                bytemuck::bytes_of(job),
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain generate"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain generate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.generate);
            pass.set_bind_group(1, &self.generate_terrain_group, &[]);
            for (index, job) in self.detail_jobs_cpu.iter().enumerate() {
                pass.set_bind_group(
                    3,
                    &self.generate_bind_group,
                    &[(index as u64 * JOB_SLOT) as u32],
                );
                pass.dispatch_workgroups(
                    job.size[0].div_ceil(DETAIL_GROUP),
                    job.size[1].div_ceil(DETAIL_GROUP),
                    1,
                );
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
        self.generated.extend_from_slice(work);
    }

    /// Derives the pyramid over the tiles just generated, and raises every
    /// level above them.
    ///
    /// A tile changes the pyramid at every level from its own upwards: its
    /// heights feed the cells of its own level directly, and those cells are
    /// carried up into every coarser cell above them -- through the generated
    /// levels and on into the resident chain, whose cells were derived before
    /// this ground had any detail on it. Miss that and a coarse cell reads too
    /// low, which is a ray passing through a ridge.
    fn raise_pyramid(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.jobs.clear();
        let generated = std::mem::take(&mut self.generated);
        if generated.is_empty() {
            return;
        }
        let tile = self.residency.detail_tile_texels as i32;
        for level in self.base..self.resident_base + self.mips {
            let (valid_low, valid_high) = self.level_valid(level);
            for wanted in generated.iter().filter(|w| w.level <= level) {
                let shift = level - wanted.level;
                let side = (tile >> shift).max(1);
                let corner = IVec2::new(
                    (wanted.tile.x * tile) >> shift,
                    (wanted.tile.y * tile) >> shift,
                );
                // A tile's own slots are its own whatever the window says: its
                // heights went into them a moment ago, and the square only
                // admits them once the step finishes. Coarser cells are not its
                // to write, and one outside will be derived by whichever tile
                // makes it valid.
                let (mut low, high) = if shift == 0 {
                    (corner, corner + IVec2::splat(side))
                } else {
                    (
                        corner.max(valid_low),
                        (corner + IVec2::splat(side)).min(valid_high),
                    )
                };
                // One cell back along each axis, because a cell is closed by a
                // sample past itself: the last row and column of the ground
                // before this tile could not be closed until it arrived.
                let back = low - IVec2::ONE;
                if back.x >= valid_low.x && back.x < valid_high.x {
                    low.x = back.x;
                }
                if back.y >= valid_low.y && back.y < valid_high.y {
                    low.y = back.y;
                }
                self.emit(level, low, high);
            }
        }
        debug_assert!(
            self.jobs.len() <= self.job_slots,
            "{} rectangles to derive against room for {}",
            self.jobs.len(),
            self.job_slots
        );
        self.run_jobs(device, queue);
    }

    /// Starts or stops accounting for where an update's time goes.
    pub fn profile(&mut self, on: bool) {
        self.spans = on.then(crate::profile::Terrain::default);
    }

    /// What the last [`Terrain::update`] spent, if it was being watched.
    pub fn spans(&self) -> Option<crate::profile::Terrain> {
        self.spans
    }

    /// Describes the chain to the shader.
    fn write_uniform(&self, queue: &wgpu::Queue) {
        let (data_min, data_max) = self.placement.data_bounds();
        let (origin_x, origin_z) = self.placement.world_of_texel(0, 0.0, 0.0);

        let mut uniform = TerrainUniform {
            levels: [LevelUniform::default(); MAX_LEVELS],
            origin: [origin_x as f32, origin_z as f32],
            metres_per_texel: [
                self.placement.metres_per_texel_x as f32,
                self.placement.metres_per_texel_z as f32,
            ],
            data_min: [data_min.0 as f32, data_min.1 as f32],
            data_max: [data_max.0 as f32, data_max.1 as f32],
            // The coarsest level plus one, because the shader counts levels
            // from zero even though nothing below the base exists: a level is
            // an absolute power-of-two texel size, and the march works in
            // level-0 texels throughout.
            level_count: self.resident_base + self.mips,
            base_level: self.base,
            resident_base: self.resident_base,
            march_steps: self.residency.march_steps(self.mips),
            ceiling: self.ceiling,
            wall_nudge: wall_nudge(UVec2::new(self.placement.width, self.placement.height)),
            viewport: self.viewport.to_array(),
        };

        for level in self.base..self.resident_base + self.mips {
            let (low, high) = self.level_valid(level);
            uniform.levels[level as usize] = LevelUniform {
                valid_low: low.to_array(),
                valid_high: high.to_array(),
                // One figure for every level: they all bound the same whole
                // raster, so the highest cell of one is the highest cell of any
                // to within how coarsely it closes its squares. A generated
                // level cannot exceed it either -- it is the base plus detail
                // the base's own cell already bounds.
                ceiling: self.ceiling,
                padding: 0.0,
                mask: self.level_mask(level).to_array(),
            };
        }

        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// The elevation of the ground under the camera, in world units.
    ///
    /// Read from the mirror of a coarse mip, so it is the ground averaged over
    /// kilometres rather than the peak the camera happens to be over. That is
    /// the right shape of answer for choosing a level that covers kilometres,
    /// and where it is least accurate -- close to the ground, where the relief
    /// it smooths away is a large fraction of the distance -- the finest level
    /// is being drawn anyway.
    fn ground_height(&self, camera_texels: DVec2) -> f32 {
        if self.ground.is_empty() {
            return 0.0;
        }
        let texels = camera_texels / f64::from(1u32 << self.ground_level);
        let at = IVec2::new(texels.x.floor() as i32, texels.y.floor() as i32).clamp(
            IVec2::ZERO,
            self.ground_size.as_ivec2() - IVec2::ONE,
        );
        let height = self.ground[at.y as usize * self.ground_size.x as usize + at.x as usize];
        // The camera can legally be over ground the raster says nothing about:
        // past the edge of the survey, or over a hole in it. Sea level is the
        // same fallback the terrain itself draws there.
        if height > crate::terrain::NODATA_BELOW {
            height
        } else {
            0.0
        }
    }

    /// Reads every mip of the heights and the ground cover in, then builds the
    /// max pyramid over them.
    ///
    /// A block of rows at a time rather than a level at a time: mip zero of
    /// this raster is 704 MB of heights, and staging it whole would cost more
    /// resident memory than the texture it is on its way into.
    fn load(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, sources: &Sources) {
        /// Rows of one mip staged at once. A hundred and forty megabytes at the
        /// widest, which is a fifth of what a whole level would be.
        const LOAD_ROWS: u32 = 512;

        let timed = self.spans.is_some();
        let started = std::time::Instant::now();
        let mut staging: Vec<u8> = Vec::new();
        let (mut read, mut convert, mut write) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);

        for mip in 0..self.mips {
            let level = self.resident_base + mip;
            let size = self.level_size(level);
            let mirrored = level == self.ground_level;
            if mirrored {
                self.ground = vec![0.0; (size.x as usize) * (size.y as usize)];
            }

            let mut y = 0;
            while y < size.y {
                let rows = LOAD_ROWS.min(size.y - y);
                let count = (size.x as usize) * (rows as usize);
                let origin = IVec2::new(0, y as i32);
                let block = UVec2::new(size.x, rows);

                staging.resize(count * size_of::<f32>(), 0);
                let clock = crate::profile::Clock::start(timed);
                sources.heights.read_rect(level, origin, block, &mut staging);
                read += clock.elapsed();
                let clock = crate::profile::Clock::start(timed);
                if VERTICAL_EXAGGERATION != 1.0 {
                    for height in bytemuck::cast_slice_mut::<u8, f32>(&mut staging) {
                        *height *= VERTICAL_EXAGGERATION;
                    }
                }
                convert += clock.elapsed();
                if mirrored {
                    let from: &[f32] = bytemuck::cast_slice(&staging);
                    let at = (y as usize) * (size.x as usize);
                    self.ground[at..at + count].copy_from_slice(from);
                }
                let clock = crate::profile::Clock::start(timed);
                Self::write_rows(queue, &self.height_texture, mip, y, block, 4, &staging);
                write += clock.elapsed();

                staging.resize(count * size_of::<MaterialId>(), 0);
                let clock = crate::profile::Clock::start(timed);
                sources
                    .materials
                    .read_rect(level, origin, block, &mut staging);
                read += clock.elapsed();
                let clock = crate::profile::Clock::start(timed);
                // Narrowed in place, forwards, so the sixteen bit id is written
                // over bytes the thirty-two bit one has already been read out
                // of. Two bytes never overtake four. An id that did not fit
                // becomes `Null`, which draws as the magenta that means nothing
                // is known -- visible, rather than silently some other material.
                for cell in 0..count {
                    let wide = u32::from_le_bytes(
                        staging[cell * 4..cell * 4 + 4]
                            .try_into()
                            .expect("four bytes"),
                    );
                    let narrow = u16::try_from(wide).unwrap_or(0);
                    staging[cell * 2..cell * 2 + 2].copy_from_slice(&narrow.to_le_bytes());
                }
                convert += clock.elapsed();
                let clock = crate::profile::Clock::start(timed);
                Self::write_rows(
                    queue,
                    &self.material_texture,
                    mip,
                    y,
                    block,
                    2,
                    &staging[..count * 2],
                );
                write += clock.elapsed();

                y += rows;
            }
        }

        // Before the pyramid, because `cs_maxima` reads the chain through the
        // same uniform the march does -- the level masks, the base, the sizes.
        // Deriving against an unwritten uniform reads every cell of every level
        // at texel zero, which is a pyramid of one number and looks like a
        // shader bug rather than an ordering one.
        self.write_uniform(queue);
        self.build_pyramid(device, queue);
        self.ceiling = self.read_ceiling(device, queue);
        log::info!(
            "terrain: read {} mips and built the pyramid in {:.2?}, highest ground {:.0} m",
            self.mips,
            started.elapsed(),
            self.ceiling
        );

        if let Some(spans) = self.spans.as_mut() {
            spans.read += read;
            spans.convert += convert;
            spans.write += write;
        }
    }

    /// Writes a block of rows into one mip.
    fn write_rows(
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        mip: u32,
        y: u32,
        block: UVec2,
        bytes: u32,
        data: &[u8],
    ) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d { x: 0, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // Tightly packed. `write_texture` waives the row alignment a
                // buffer copy demands, which is what lets a six-texel mip go in
                // without a padded staging copy of its own.
                bytes_per_row: Some(block.x * bytes),
                rows_per_image: Some(block.y),
            },
            wgpu::Extent3d {
                width: block.x,
                height: block.y,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Builds the max pyramid over the whole chain, finest mip first.
    ///
    /// One sweep, once. There is nothing incremental about it because there is
    /// nothing incremental left: the heights under it never change.
    fn build_pyramid(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.jobs.clear();
        for level in self.levels() {
            let size = self.level_size(level).as_ivec2();
            self.emit(level, IVec2::ZERO, size);
        }
        debug_assert!(
            self.jobs.len() <= self.job_slots,
            "{} rectangles to derive against room for {}",
            self.jobs.len(),
            self.job_slots
        );
        self.run_jobs(device, queue);
    }

    /// Cuts one rectangle of one level into jobs and records them.
    ///
    /// Two things bound a job: the scratch buffer its cells go out through, and
    /// the wrap. A rectangle straddling a wrap is two rectangles in the texture
    /// and cannot be copied in one go, so it is cut there rather than copied in
    /// pieces at offsets a copy would refuse to start at. A resident level has
    /// no wrap to cut at, which is what the identity mask says.
    fn emit(&mut self, level: u32, low: IVec2, high: IVec2) {
        if low.x >= high.x || low.y >= high.y {
            return;
        }
        let mask = self.level_mask(level).x;
        // The level below, bounded exactly as the march bounds it, so a child
        // is carried if and only if a ray could descend into it.
        let carry = level > self.base;
        let (below_low, below_high) = if carry {
            self.level_valid(level - 1)
        } else {
            (IVec2::ZERO, IVec2::ZERO)
        };

        let mut y = low.y;
        while y < high.y {
            let rows = span(y, high.y, mask);
            let mut x = low.x;
            while x < high.x {
                let columns = span(x, high.x, mask);
                self.jobs.push(MaximaJob {
                    origin: [x, y],
                    size: [columns, rows],
                    level,
                    carry: u32::from(carry),
                    below_low: below_low.to_array(),
                    below_high: below_high.to_array(),
                    stride: derive_row_bytes(columns) / 4,
                    padding: 0,
                });
                x += columns as i32;
            }
            y += rows as i32;
        }
    }

    /// Dispatches every planned rectangle and copies its cells into the chain.
    fn run_jobs(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.jobs.is_empty() {
            return;
        }
        for (index, job) in self.jobs.iter().enumerate() {
            queue.write_buffer(
                &self.derive_jobs,
                index as u64 * JOB_SLOT,
                bytemuck::bytes_of(job),
            );
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain derive"),
        });
        for (index, job) in self.jobs.iter().enumerate() {
            {
                // A pass of its own per rectangle, which is what orders a
                // coarse level's reads after the fine level's copy: the cells
                // go out through one scratch buffer, so nothing here overlaps
                // anything else anyway.
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("terrain derive"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.derive);
                pass.set_bind_group(1, &self.bind_group, &[]);
                pass.set_bind_group(
                    3,
                    &self.derive_bind_group,
                    &[(index as u64 * JOB_SLOT) as u32],
                );
                pass.dispatch_workgroups(
                    job.size[0].div_ceil(2).div_ceil(DERIVE_GROUP),
                    job.size[1].div_ceil(DERIVE_GROUP),
                    1,
                );
            }
            // A resident level is a mip of the chain at its own index; a
            // generated one is a layer of the window, wrapped onto its slot the
            // same way the shader wrapped it.
            let resident = job.level >= self.resident_base;
            let mask = self.level_mask(job.level).x as u32;
            let (texture, mip, layer) = if resident {
                (&self.maxima_texture, job.level - self.resident_base, 0)
            } else {
                (&self.detail_maxima_texture, 0, job.level)
            };
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.derive_scratch,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(job.stride * 4),
                        rows_per_image: Some(job.size[1]),
                    },
                },
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d {
                        x: job.origin[0] as u32 & mask,
                        y: job.origin[1] as u32 & mask,
                        z: layer,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: job.size[0],
                    height: job.size[1],
                    depth_or_array_layers: 1,
                },
            );
        }
        queue.submit(std::iter::once(encoder.finish()));
        self.jobs.clear();
    }

    /// The highest cell of the coarsest mip, which bounds every cell under it.
    ///
    /// This is what `tile_ceilings` used to be, and it is now a property of the
    /// chain rather than a tally the upload path had to keep in step with it:
    /// `M[d] >= reduce_max(M[d - 1])` at every depth, so the top of the chain
    /// bounds the whole of it and one readback of a few dozen texels answers.
    fn read_ceiling(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> f32 {
        let size = self.level_size(self.coarsest());
        let stride = (size.x * 2).div_ceil(COPY_ALIGN) * COPY_ALIGN;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain ceiling readback"),
            size: u64::from(stride * size.y),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("terrain ceiling"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.maxima_texture,
                mip_level: self.mips - 1,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(size.y),
                },
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        readback.map_async(wgpu::MapMode::Read, .., |result| {
            result.expect("ceiling readback failed")
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let bytes = readback.get_mapped_range(..).expect("buffer not mapped");
        let mut ceiling = f32::NEG_INFINITY;
        for row in 0..size.y as usize {
            let at = row * stride as usize;
            let cells: &[u16] = bytemuck::cast_slice(&bytes[at..at + size.x as usize * 2]);
            for cell in cells {
                ceiling = ceiling.max(half::f16::from_bits(*cell).to_f32());
            }
        }
        ceiling
    }

    /// The lowest and highest elevation in the raster, from a coarse level.
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

    /// Settles every pixel the reprojection answered, and lists the rest.
    ///
    /// The caller has set group 0 (the camera), group 2 (the G-buffer being
    /// written) and group 3 (the carried buffers and the work list); each of
    /// these adds group 1, which is the terrain itself.
    pub fn compact(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.compact);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.dispatch_workgroups(
            self.viewport.x.div_ceil(COMPACT_TILE),
            self.viewport.y.div_ceil(COMPACT_TILE),
            1,
        );
    }

    /// Sizes the march's dispatch from the list [`Terrain::compact`] left.
    ///
    /// Binds nothing itself: its whole layout is the one group the caller sets.
    pub fn args(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.args);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Reduces the motion field to one risk value per dither cell.
    ///
    /// After the march, so the field it reads is the finished one. What it
    /// writes is read by the *next* frame's splat, which is a frame late and no
    /// worse for it: how fast the picture is changing does not change fast.
    pub fn risk(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.risk);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.dispatch_workgroups(
            self.viewport.x.div_ceil(COMPACT_TILE),
            self.viewport.y.div_ceil(COMPACT_TILE),
            1,
        );
    }

    /// Spreads the risk field outwards over the distance each cell's motion
    /// covers, which is what says whether a cell of sky is still sky.
    ///
    /// After [`Terrain::risk`], which it reads whole: one thread per cell, so
    /// the dispatch is over cells of cells.
    pub fn reach(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.reach);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.dispatch_workgroups(
            self.viewport
                .x
                .div_ceil(COMPACT_TILE)
                .div_ceil(COMPACT_TILE),
            self.viewport
                .y
                .div_ceil(COMPACT_TILE)
                .div_ceil(COMPACT_TILE),
            1,
        );
    }

    /// Casts a ray for each pixel on that list.
    ///
    /// Indirect, because how many there are is decided on the GPU and never
    /// travels back to the CPU -- which is the whole point: a round trip would
    /// cost a stall per frame and buy nothing.
    pub fn march(&self, pass: &mut wgpu::ComputePass<'_>, args: &wgpu::Buffer) {
        pass.set_pipeline(&self.march);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.dispatch_workgroups_indirect(args, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::maxima::ceiling_half;
    use crate::terrain::pyramid::{Level, Pyramid};

    /// The whole test raster, held at whatever base the test asks for.
    ///
    /// Generated windows of four tiles of eight texels rather than the shipped
    /// eight of five hundred and twelve, because a raster a test can afford to
    /// build is smaller than one real tile and none of the wrapping would be
    /// exercised at all.
    fn test_residency(resident_base: u32) -> Residency {
        Residency {
            resident_base,
            detail_relief: 0.0,
            detail_tiles: 4,
            detail_tile_texels: 8,
            // Whole windows per update, so a test never has to drain a queue.
            detail_per_update: 4096,
            ..Default::default()
        }
    }

    const RASTER: u32 = 64;

    /// A nudge only moves a ray off a wall if it survives being added to the
    /// numbers the march is working in, and those numbers are texel indices
    /// measured from the raster's corner.
    ///
    /// This is the property the fixed thousandth of a texel did not have. On the
    /// installed raster an index reaches 114688, where consecutive `f32` values
    /// are 0.0078 apart, so the nudge rounded away entirely: the ray came back
    /// to the wall it had just left and took roughly sixteen iterations to
    /// escape one cell. Everything visible about that -- the smeared wall across
    /// the distance, the streaks over near ground -- followed from a step too
    /// small to be a step.
    #[test]
    fn a_nudge_still_moves_a_ray_at_the_far_corner_of_the_raster() {
        for raster in [
            UVec2::splat(RASTER),
            UVec2::new(98304, 114688),
            UVec2::splat(1 << 18),
        ] {
            let corner = raster.max_element() as f32;
            let spacing = f32::from_bits(corner.to_bits() + 1) - corner;
            let nudge = wall_nudge(raster);
            assert!(
                nudge >= 4.0 * spacing,
                "a {raster:?} raster nudges by {nudge} where the float step is {spacing}",
            );
            // And still a fraction of the finest texel, so a ray that steps into
            // a cell has not stepped over the ground at the near edge of it.
            assert!(
                nudge < 0.5,
                "a {raster:?} raster nudges by {nudge} of a level-0 texel",
            );
        }
    }

    /// Ridged enough that neighbouring texels disagree, so a maximum is a real
    /// choice rather than whichever corner happened to be picked.
    fn rugged() -> Vec<f32> {
        (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as f32, (i / RASTER) as f32);
                40.0 * (x * 0.7).sin() + 25.0 * (y * 0.9).cos() + 10.0 * (x * 0.13 - y * 0.21).sin()
            })
            .collect()
    }

    /// The highest sample of one level across the closed square a texel covers,
    /// where the texel is `span` of that level's samples across.
    ///
    /// An oracle deliberately independent of everything the upload path does:
    /// it knows only which ground a texel is named after, and asks the source
    /// what stands on it. This is what the march needs a texel to bound when
    /// that level is the one reading it.
    fn cell_ceiling(source: &dyn RasterSource, level: u32, cell: IVec2, span: i32) -> f32 {
        let side = span as u32 + 1;
        let mut samples = vec![0f32; (side * side) as usize];
        source.read_rect(
            level,
            cell * span,
            UVec2::splat(side),
            bytemuck::cast_slice_mut(&mut samples),
        );
        samples.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// What a level is defined to hold: the greatest of [`cell_ceiling`] over
    /// every level that could read it.
    ///
    /// The bound each level asks for differs, and one texel has to satisfy all
    /// of them at once, because which level reads a given point depends on
    /// where the camera is rather than on anything the pyramid knows.
    fn cell_defined(source: &dyn RasterSource, level: u32, cell: IVec2) -> f32 {
        (0..=level.min(source.level_count() - 1))
            .map(|under| cell_ceiling(source, under, cell, 1 << (level - under)))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// The bound a *derived* cell has to hold.
    ///
    /// Narrower than [`cell_defined`], and deliberately so. The product the
    /// tools write answers for every level under this one, because it is built
    /// once over the whole raster and has no idea which of them will be
    /// resident. A cell derived from what is loaded answers only for the levels
    /// a ray could actually reach from it: its own closed square, and every
    /// finer cell the march would be allowed to descend into. So this is the
    /// same recursion the march performs, with the same residency test.
    fn cell_reachable(source: &dyn RasterSource, terrain: &Terrain, level: u32, cell: IVec2) -> f32 {
        let mut top = cell_ceiling(source, level, cell, 1);
        if level <= terrain.resident_base {
            return top;
        }
        let finer = level - 1;
        // The march's own bound, which is the whole level.
        let high = terrain.level_size(finer).as_ivec2();
        for dy in 0..2 {
            for dx in 0..2 {
                let child = cell * 2 + IVec2::new(dx, dy);
                if child.cmpge(IVec2::ZERO).all() && child.cmplt(high).all() {
                    top = top.max(cell_reachable(source, terrain, finer, child));
                }
            }
        }
        top
    }

    /// A terrain over a rugged raster, held from `resident_base` upwards.
    fn terrain_over(device: &wgpu::Device, resident_base: u32) -> Terrain {
        let camera_layout = crate::scene::test_camera_layout(device);
        let storage_layout = crate::deferred::storage_layout(device);
        let work_layout = crate::reproject::work_layout(device);
        let args_layout = crate::reproject::args_layout(device);
        let risk_layout = crate::reproject::risk_layout(device);
        let reach_layout = crate::reproject::reach_layout(device);
        Terrain::new(
            device,
            &camera_layout,
            &storage_layout,
            &work_layout,
            &args_layout,
            &risk_layout,
            &reach_layout,
            test_residency(resident_base),
            UVec2::splat(RASTER),
            Georeferencing::square(RASTER, RASTER, 30.0),
            Sources {
                heights: Box::new(Pyramid::build(Level::new(RASTER, RASTER, rugged()))),
                materials: Box::new(Pyramid::build(Level::new(
                    RASTER,
                    RASTER,
                    vec![MaterialId(0); (RASTER * RASTER) as usize],
                ))),
            },
        )
    }

    /// One mip of the max pyramid, with the padded row stride it came back at.
    ///
    /// A mip at a time because they are all different sizes now, which is the
    /// whole difference between a chain and the array of equal layers this
    /// replaced.
    fn read_mip(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &Terrain,
        level: u32,
    ) -> (Vec<u8>, u32) {
        let size = terrain.level_size(level);
        // Rows in a texture-to-buffer copy are padded to 256 bytes, which a
        // sixty-four texel row of half floats is well short of.
        let stride = (size.x * 2).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pyramid readback"),
            size: u64::from(stride * size.y),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &terrain.maxima_texture,
                mip_level: level - terrain.resident_base,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(size.y),
                },
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let bytes = readback.get_mapped_range(..).expect("buffer not mapped")[..].to_vec();
        (bytes, stride)
    }

    /// One cell out of what [`read_mip`] returned.
    fn mip_cell(bytes: &[u8], stride: u32, cell: IVec2) -> f32 {
        let at = cell.y as usize * stride as usize + cell.x as usize * 2;
        half::f16::from_bits(u16::from_le_bytes([bytes[at], bytes[at + 1]])).to_f32()
    }

    /// Checks every cell of every level against both bounds it sits between,
    /// and returns how many were asked.
    ///
    /// The lower bound is the failure this structure exists to prevent: a cell
    /// below ground a ray could reach is a ridge the ray passes through. The
    /// upper bound is the promise the derivation makes in return -- it carries
    /// only from levels that are held, so it can only ever be tighter than the
    /// chain a tool would have written over the whole raster, and a cell above
    /// that would be slack the march has to pay descents for.
    ///
    /// The two coincide exactly when the base is level zero, because then every
    /// level a cell is defined over is a level that is held. That is what makes
    /// the base-zero case an equality rather than a sandwich.
    fn check_pyramid(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &Terrain,
        raster: &dyn RasterSource,
    ) -> u32 {
        let mut checked = 0;
        for level in terrain.levels() {
            let (bytes, stride) = read_mip(device, queue, terrain, level);
            let size = terrain.level_size(level).as_ivec2();
            for cell_y in 0..size.y {
                for cell_x in 0..size.x {
                    let cell = IVec2::new(cell_x, cell_y);
                    // Asked only of texels whose square lands on real ground.
                    // A closed square reaches one sample past the cell, and out
                    // there the chain and this oracle both repeat the border,
                    // at their own granularities -- so they agree on the bound
                    // without agreeing on the figure.
                    let reach = (cell + IVec2::ONE) << level;
                    if reach.max_element() >= RASTER as i32 {
                        continue;
                    }
                    let got = mip_cell(&bytes, stride, cell);
                    let needs = ceiling_half(cell_reachable(raster, terrain, level, cell)).to_f32();
                    let allowed = ceiling_half(cell_defined(raster, level, cell)).to_f32();
                    assert!(
                        got >= needs,
                        "level {level} cell {cell} holds {got} where a ray reaching \
                         through it can meet ground at {needs}"
                    );
                    assert!(
                        got <= allowed,
                        "level {level} cell {cell} holds {got}, above the {allowed} \
                         a chain over the whole raster would have"
                    );
                    checked += 1;
                }
            }
        }
        checked
    }

    /// A terrain over a rugged raster with the relief turned up.
    fn terrain_with_relief(device: &wgpu::Device, resident_base: u32, relief: f32) -> Terrain {
        Terrain::new(
            device,
            &crate::scene::test_camera_layout(device),
            &crate::deferred::storage_layout(device),
            &crate::reproject::work_layout(device),
            &crate::reproject::args_layout(device),
            &crate::reproject::risk_layout(device),
            &crate::reproject::reach_layout(device),
            Residency {
                detail_relief: relief,
                ..test_residency(resident_base)
            },
            UVec2::splat(RASTER),
            Georeferencing::square(RASTER, RASTER, 30.0),
            Sources {
                heights: Box::new(Pyramid::build(Level::new(RASTER, RASTER, rugged()))),
                materials: Box::new(Pyramid::build(Level::new(
                    RASTER,
                    RASTER,
                    vec![MaterialId(0); (RASTER * RASTER) as usize],
                ))),
            },
        )
    }

    /// The detail is real, it stays inside the relief it was given, and it
    /// moves no sample the survey actually measured.
    ///
    /// Measured against the same terrain with the relief at zero, so what is
    /// left is the fractal and nothing else -- no second spelling of the
    /// interpolation under it, which is the whole reason to difference two runs
    /// rather than model one.
    ///
    /// The third assertion is the surprising one and it is why there is no seam
    /// where a generated level hands over to the base. The fractal's first
    /// octave is the base's own texel, so its lattice *is* the base grid, and
    /// gradient noise is zero at every lattice point of every octave. The
    /// detail therefore interpolates between the measured samples without ever
    /// moving one of them -- a property of where the octaves were put rather
    /// than of anything that enforces it, which is exactly the kind that stops
    /// being true by accident.
    #[test]
    fn generated_detail_is_there_bounded_and_pinned_to_the_survey() {
        const RELIEF: f32 = 12.0;

        let (device, queue) = crate::scene::test_device();
        let mut flat = terrain_with_relief(&device, 2, 0.0);
        let mut rough = terrain_with_relief(&device, 2, RELIEF);
        let at = Vec3::new(137.0, 100.0, -71.0);
        flat.update(&device, &queue, at);
        rough.update(&device, &queue, at);

        for level in 0..rough.resident_base {
            let smooth = read_level(&device, &queue, &flat, &flat.detail_height_texture, level, 4);
            let detailed =
                read_level(&device, &queue, &rough, &rough.detail_height_texture, level, 4);
            let step = 1 << (rough.resident_base - level);
            let (low, high) = rough.level_valid(level);
            let (mut moved, mut worst, mut nodes) = (0u32, 0.0f32, 0u32);
            for cell_y in low.y..high.y {
                for cell_x in low.x..high.x {
                    let cell = IVec2::new(cell_x, cell_y);
                    let difference = detailed(cell) - smooth(cell);
                    worst = worst.max(difference.abs());
                    moved += u32::from(difference != 0.0);
                    if cell.x % step == 0 && cell.y % step == 0 {
                        // Zero to a micron rather than to the bit. The octave
                        // lands on its own lattice point exactly, but the
                        // reciprocal of the wavelength does not divide exactly,
                        // so the fraction comes out a few ulps off zero and the
                        // gradient is dotted with that rather than with nothing.
                        assert!(
                            difference.abs() < 1e-3,
                            "level {level} moved base node {} by {difference} m",
                            cell / step
                        );
                        nodes += 1;
                    }
                }
            }
            assert!(moved > 100, "level {level} moved only {moved} texels");
            assert!(nodes > 16, "level {level} shared only {nodes} nodes");
            assert!(worst > 0.05, "level {level} moved by at most {worst} m");
            assert!(
                worst <= RELIEF,
                "level {level} moved a texel {worst} m against a {RELIEF} m relief"
            );
        }

        // And a coarser level is never *louder* than the one under it, at the
        // points the two share. It carries a subset of the same octaves at the
        // same amplitudes, so it cannot be -- unless the sum is renormalised
        // over the octaves that survive the band limit, which would make every
        // level as loud as the finest. A ray crossing a window edge would then
        // step onto ground of the same roughness at half the resolution, which
        // draws as the terrain breathing as the ring goes by and which every
        // other check here passes happily. Measured: 0.052 m against 0.034 m
        // renormalised, and equal both ways when it is not.
        //
        // Equal, not quieter, and the reason is worth writing down because it
        // looks like the test failing to see anything. The points the two
        // levels share are every other texel of the finer one, which is exactly
        // the lattice of the octave the coarser one dropped -- and gradient
        // noise is zero at its own lattice points. The dropped octave
        // contributes nothing *at these particular points*, so what is left is
        // the same sum on both sides. It is the normalisation that differs.
        //
        // Compared at matched world points rather than level against level: a
        // coarser window covers twice the ground, and on ground this uneven
        // that difference swamps the one being looked for.
        for level in 1..rough.resident_base {
            let pair = |terrain: &Terrain, at| {
                read_level(&device, &queue, terrain, &terrain.detail_height_texture, at, 4)
            };
            let (fine_flat, fine_rough) = (pair(&flat, level - 1), pair(&rough, level - 1));
            let (coarse_flat, coarse_rough) = (pair(&flat, level), pair(&rough, level));
            // Only where the two windows overlap. A coarser window covers
            // twice the ground, so most of it has no finer texel to compare
            // against -- and reading one anyway wraps onto a tile of somewhere
            // else, which is ground rather than nonsense and so goes unnoticed.
            let (low, high) = rough.level_valid(level);
            let (under_low, under_high) = rough.level_valid(level - 1);
            let low = low.max((under_low + IVec2::ONE) / 2);
            let high = high.min(under_high / 2);
            let (mut fine, mut coarse, mut counted) = (0.0f64, 0.0f64, 0u32);
            for cell_y in low.y..high.y {
                for cell_x in low.x..high.x {
                    let cell = IVec2::new(cell_x, cell_y);
                    let here = f64::from(coarse_rough(cell) - coarse_flat(cell));
                    let under = f64::from(fine_rough(cell * 2) - fine_flat(cell * 2));
                    coarse += here * here;
                    fine += under * under;
                    counted += 1;
                }
            }
            let (fine, coarse) = (
                (fine / f64::from(counted)).sqrt(),
                (coarse / f64::from(counted)).sqrt(),
            );
            assert!(
                coarse <= fine * 1.02,
                "level {level} came back at {coarse:.4} m against {fine:.4} m for the level \
                 under it, so dropping an octave made it louder"
            );
        }
    }

    /// Every cell of the pyramid bounds the ground it claims, at every level.
    ///
    /// With the base at level zero the derivation has every level a cell is
    /// defined over, so this is an equality against the CPU oracle rather than
    /// a bound: the GPU chain must be exactly what `terrain-process` used to
    /// write. That is the strongest statement available about it, and it is
    /// what says the shader's closed squares, its carry and its rounding all
    /// agree with the definition in `crates/terrain-tiles/src/maxima.rs`.
    ///
    /// Teeth: nothing writes these cells but the derivation, so a chain that
    /// failed to run reads as zeroes, and this raster is ridged either side of
    /// zero.
    #[test]
    fn every_cell_of_the_pyramid_bounds_the_ground_it_covers() {
        let (device, queue) = crate::scene::test_device();
        let raster = Pyramid::build(Level::new(RASTER, RASTER, rugged()));
        let mut terrain = terrain_over(&device, 0);
        terrain.update(&device, &queue, Vec3::new(137.0, 100.0, -71.0));
        assert_eq!(terrain.base_level(), 0, "the test wants every level built");

        let raster: &dyn RasterSource = &raster;
        let checked = check_pyramid(&device, &queue, &terrain, raster);
        assert!(
            checked > 1000,
            "only {checked} cells sat wholly on the raster, which is too few"
        );
    }

    /// One level of a texture read back, indexed by texel rather than by slot.
    ///
    /// A resident level is a mip of a chain and its index *is* its coordinate;
    /// a generated one is a layer of a window and wraps. The closure hides
    /// which, so a check can be written once against both.
    fn read_level(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &Terrain,
        texture: &wgpu::Texture,
        level: u32,
        bytes: u32,
    ) -> impl Fn(IVec2) -> f32 + use<> {
        let resident = level >= terrain.resident_base;
        let (size, mip, layer) = if resident {
            (terrain.level_size(level), level - terrain.resident_base, 0)
        } else {
            (UVec2::splat(terrain.detail_across), 0, level)
        };
        let stride = (size.x * bytes).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("level readback"),
            size: u64::from(stride * size.y),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(size.y),
                },
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let data = readback.get_mapped_range(..).expect("buffer not mapped")[..].to_vec();
        let mask = if resident {
            IVec2::NEG_ONE
        } else {
            IVec2::splat(terrain.detail_across as i32 - 1)
        };
        let last = size.as_ivec2() - IVec2::ONE;
        move |cell: IVec2| {
            let at = (cell & mask).clamp(IVec2::ZERO, last);
            let byte = at.y as usize * stride as usize + at.x as usize * bytes as usize;
            if bytes == 4 {
                f32::from_le_bytes(data[byte..byte + 4].try_into().expect("four bytes"))
            } else {
                half::f16::from_bits(u16::from_le_bytes(
                    data[byte..byte + 2].try_into().expect("two bytes"),
                ))
                .to_f32()
            }
        }
    }

    /// Every cell of the pyramid is exactly the recurrence over the heights
    /// that are actually in the textures.
    ///
    /// Read back rather than modelled. The oracle above knows what the survey
    /// holds, which is the right thing to check while every level is measured
    /// and the wrong thing the moment one is generated: a generated level is
    /// whatever the shader put there, and Catmull-Rom legitimately overshoots
    /// the samples it passes through. Checking against the textures asks the
    /// only question that stays meaningful either way -- does the pyramid bound
    /// the surface the march will actually solve against -- and it tests the
    /// closed square, the carry, the residency test and the rounding at once.
    #[test]
    fn the_pyramid_is_the_recurrence_over_the_heights_that_are_there() {
        let (device, queue) = crate::scene::test_device();
        let mut terrain = terrain_over(&device, 1);
        terrain.update(&device, &queue, Vec3::new(137.0, 100.0, -71.0));
        assert_eq!(terrain.base_level(), 0, "the test wants the generated level");

        let mut checked = 0;
        for level in terrain.base..terrain.resident_base + terrain.mips {
            let heights = read_level(
                &device,
                &queue,
                &terrain,
                if level >= terrain.resident_base {
                    &terrain.height_texture
                } else {
                    &terrain.detail_height_texture
                },
                level,
                4,
            );
            let ceilings = read_level(
                &device,
                &queue,
                &terrain,
                if level >= terrain.resident_base {
                    &terrain.maxima_texture
                } else {
                    &terrain.detail_maxima_texture
                },
                level,
                2,
            );
            let below = (level > terrain.base).then(|| {
                read_level(
                    &device,
                    &queue,
                    &terrain,
                    if level > terrain.resident_base {
                        &terrain.maxima_texture
                    } else {
                        &terrain.detail_maxima_texture
                    },
                    level - 1,
                    2,
                )
            });
            let (low, high) = terrain.level_valid(level);
            let (child_low, child_high) = terrain.level_valid(level - 1.min(level));

            for cell_y in low.y..high.y {
                for cell_x in low.x..high.x {
                    let cell = IVec2::new(cell_x, cell_y);
                    // This level's own samples over the cell's closed square.
                    let mut want = f32::NEG_INFINITY;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            want = want.max(heights(cell + IVec2::new(dx, dy)));
                        }
                    }
                    // Every child a ray could descend into, and no other.
                    if let Some(below) = &below {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let child = cell * 2 + IVec2::new(dx, dy);
                                if child.cmpge(child_low).all() && child.cmplt(child_high).all() {
                                    want = want.max(below(child));
                                }
                            }
                        }
                    }
                    let got = ceilings(cell);
                    assert_eq!(
                        got,
                        ceiling_half(want).to_f32(),
                        "level {level} cell {cell} holds {got} where the heights under \
                         it come to {want}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 1000, "only {checked} cells checked");
    }

    /// A generated level agrees with the base wherever the two grids meet.
    ///
    /// This is the whole of what stage one gave up and this puts back, stated
    /// as a property rather than as a picture: level `l`'s sample `i` sits at
    /// level-0 texel `i * 2^l`, so every base node is also a node of every
    /// level under it, and Catmull-Rom passes through the nodes. If it did not,
    /// the ground would step wherever a ray handed over between a generated
    /// level and the base -- and the window's edge is exactly where that
    /// happens, several kilometres out, where a step is hardest to notice and
    /// hardest to attribute.
    #[test]
    fn a_generated_level_meets_the_base_at_every_shared_node() {
        let (device, queue) = crate::scene::test_device();
        // Relief off, from `test_residency`, so what is left is the smooth read
        // alone. With detail on, a level carries an octave the base does not
        // and the two are meant to differ; the test below is what bounds that.
        let mut terrain = terrain_over(&device, 2);
        terrain.update(&device, &queue, Vec3::new(137.0, 100.0, -71.0));
        assert_eq!(terrain.base_level(), 0, "the test wants both generated levels");

        let base = read_level(
            &device,
            &queue,
            &terrain,
            &terrain.height_texture,
            terrain.resident_base,
            4,
        );
        for level in 0..terrain.resident_base {
            let generated = read_level(
                &device,
                &queue,
                &terrain,
                &terrain.detail_height_texture,
                level,
                4,
            );
            let step = 1 << (terrain.resident_base - level);
            let (low, high) = terrain.level_valid(level);
            let mut shared = 0;
            for cell_y in low.y..high.y {
                for cell_x in low.x..high.x {
                    let cell = IVec2::new(cell_x, cell_y);
                    if cell.x % step != 0 || cell.y % step != 0 {
                        continue;
                    }
                    let node = cell / step;
                    assert_eq!(
                        generated(cell),
                        base(node),
                        "level {level} cell {cell} sits on base node {node} and disagrees"
                    );
                    shared += 1;
                }
            }
            assert!(shared > 16, "level {level} shared only {shared} nodes");
        }
    }

}
