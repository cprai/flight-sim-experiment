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
use crate::terrain::maxima::ceiling_half;
use crate::terrain::pyramid::{RasterSource, Resident};
use crate::terrain::residency::{Residency, TileResidency, Wanted, detail_base};
use crate::terrain::tiles::{MaterialId, TileStore};
use anyhow::{Context, Result};
use glam::{DVec2, IVec2, UVec2, Vec2, Vec3};
use terrain_tiles::maxima::highest;

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
    more_padding: [f32; 2],
}

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
    texel_mask: u32,
    march_steps: u32,
    ceiling: f32,
    wall_nudge: f32,
    /// The target's size in pixels, which turns a pixel index into a ray.
    ///
    /// Occupies what was tail padding, so the uniform is the same size and
    /// every other member sits where it did.
    viewport: [u32; 2],
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

/// The three rasters the terrain is drawn from, all describing one piece of
/// ground.
///
/// Grouped rather than passed alongside each other because they are only
/// meaningful together, and because `maxima` is the one whose texels do not
/// mean what the others' do -- naming it beside them is where that is easiest
/// to get wrong. See [`Terrain::maxima`].
pub struct Sources {
    pub heights: Box<dyn RasterSource>,
    pub materials: Box<dyn RasterSource>,
    pub maxima: Box<dyn RasterSource>,
}

/// A height raster and a matching material raster, raymarched through a max
/// pyramid.
pub struct Terrain {
    residency: Residency,
    placement: Georeferencing,
    heights: Box<dyn RasterSource>,
    materials: Box<dyn RasterSource>,
    /// The quadtree the march is walked through, written by `terrain-process`.
    ///
    /// Level `l` holds one ceiling per level-`l` texel, bounding every surface
    /// the renderer might draw across the closed square that texel covers. The
    /// level array is the quadtree, so nothing here carries a mip chain of its
    /// own and climbing means reading the next level out.
    maxima: Box<dyn RasterSource>,
    height_range: (f32, f32),

    /// Which tiles each level holds, and what to load next.
    tiles: TileResidency,
    /// The finest level being kept, from [`detail_base`].
    base: u32,
    /// Whether anything has been loaded yet, which decides whether an update is
    /// allowed to take as long as it needs.
    started: bool,

    /// A CPU mirror of the coarsest level's resident square.
    ///
    /// Height above terrain decides how many of the finer levels are worth
    /// keeping, so it has to be known before the levels are loaded, and it
    /// cannot come from the tile store -- that reopens and reparses a GeoTIFF
    /// per call. Mirroring the coarsest level costs a copy of what the upload
    /// is passing through anyway, and the coarsest is the one level never
    /// dropped: the level chosen from this height must not depend on data whose
    /// residency that choice controls.
    ground: Vec<f32>,
    /// The highest ceiling in each resident tile, by level and slot.
    ///
    /// A ray clears a whole level with one comparison against the largest of
    /// these, which is most of what a ray heading for the horizon does. Kept per
    /// tile so a swap costs one tile's maximum rather than a sweep of the
    /// square.
    tile_ceilings: Vec<Vec<f32>>,
    /// Reused between uploads so a moving camera allocates nothing.
    staging: Vec<u8>,
    /// Where an update's time went, when a run asked to be told.
    ///
    /// [`None`] on an unprofiled run, so the clock is not read at all. This is
    /// the only account there is of the streaming cost: the uploads leave
    /// through `queue.write_texture` onto the staging belt rather than through
    /// a command encoder, so no GPU timestamp scope can be put around them.
    spans: Option<crate::profile::Terrain>,
    /// The target's size in pixels, passed through to the march so it can turn
    /// a pixel into the ray through it. Followed by [`Terrain::resize`].
    viewport: UVec2,

    height_texture: wgpu::Texture,
    material_texture: wgpu::Texture,
    maxima_texture: wgpu::Texture,
    uniform: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
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
        let material = root.join(terrain_tiles::MATERIAL_PRODUCT);
        // Named after the elevation it was reduced from, because `dtm` and `dsm`
        // are different surfaces and a bound over one does not cover the other.
        let ceilings = root.join(terrain_tiles::maxima_product(product));

        let heights = TileStore::<f32>::open(&elevation)?;
        let materials = TileStore::<MaterialId>::open(&material).with_context(|| {
            format!(
                "{} holds no ground-cover materials; run terrain-process over a download \
                 with an osm extract",
                root.display()
            )
        })?;
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
            (&material, materials.manifest()),
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
        // enough to span the whole raster. Continuing the chain in memory is
        // what lets the outermost square reach the edge of the data rather than
        // a couple of kilometres of it. The max chain continues with a maximum
        // rather than a mean, or its coarsest cells would sit below the ground
        // they are supposed to bound.
        let raster = UVec2::new(placement.width, placement.height);
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
                heights: Box::new(Resident::<f32>::over(Box::new(heights), raster)),
                materials: Box::new(Resident::<MaterialId>::over(Box::new(materials), raster)),
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
        let Sources {
            heights,
            materials,
            maxima,
        } = sources;
        let raster = UVec2::new(placement.width, placement.height);
        // Only the sources whose levels *are* terrain levels. The max pyramid's
        // run further -- it is a quadtree over the same raster -- and folding it
        // in here would cost real levels of terrain.
        let available = heights.level_count().min(materials.level_count());
        // The caller asks for the square the screen would like; how much of it
        // is affordable depends on the raster, which only becomes known here.
        let residency = Residency {
            tiles_across: residency.fit_tiles(raster, available),
            ..residency
        };
        let level_count = residency
            .level_count(raster, available)
            .min(MAX_LEVELS as u32);
        let across = residency.texels_across();
        log::info!(
            "terrain: {level_count} levels of {} x {} tiles, {} texels each, {:.0} MiB of \
             texture, reaching {} texels from the camera",
            residency.tiles_across,
            residency.tiles_across,
            across,
            residency.texture_bytes(level_count) as f64 / (1 << 20) as f64,
            residency.reach_texels() << (level_count - 1),
        );

        // The max pyramid has to answer for the coarsest level the march can
        // climb to. A source that stopped short would answer with its own top
        // level clamped, which bounds a smaller square than the texel covers and
        // so is not a bound at all.
        assert!(
            maxima.level_count() >= level_count,
            "the max pyramid reaches level {} but {level_count} levels are being drawn",
            maxima.level_count().saturating_sub(1)
        );
        let layers = wgpu::Extent3d {
            width: across,
            height: across,
            depth_or_array_layers: level_count,
        };
        let layer_texture = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: layers,
                // The array is the quadtree -- layer `l` is level `l` -- so
                // nothing needs a mip chain of its own.
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        // `COPY_SRC` on the maxima is not needed to draw, only so a test can
        // read the pyramid back and check it against a reference built on the
        // CPU -- cheap enough to be worth paying for always rather than building
        // a differently-shaped texture under `cfg(test)`.
        let usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
        let height_texture = layer_texture("terrain heights", wgpu::TextureFormat::R32Float, usage);
        let material_texture =
            layer_texture("terrain materials", wgpu::TextureFormat::R32Uint, usage);
        let maxima_texture = layer_texture(
            "terrain maxima",
            // Half precision, which is worth a third of the memory. `ceiling_half`
            // rounds towards positive infinity so a texel stays an upper bound on
            // the ground under it; see the reasoning there.
            wgpu::TextureFormat::R16Float,
            usage | wgpu::TextureUsages::COPY_SRC,
        );

        let array_view = |texture: &wgpu::Texture| {
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

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain layout"),
            entries: &[
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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
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
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
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

        let height_range = Self::coarsest_height_range(heights.as_ref(), &placement);
        let slots = (residency.tiles_across * residency.tiles_across) as usize;
        let square = (across as usize).pow(2);
        Self {
            residency,
            placement,
            heights,
            materials,
            maxima,
            height_range,
            tiles: TileResidency::new(residency, level_count),
            base: 0,
            started: false,
            ground: vec![0.0; square],
            tile_ceilings: vec![vec![f32::NEG_INFINITY; slots]; level_count as usize],
            staging: vec![0; (residency.tile_texels as usize).pow(2) * size_of::<MaterialId>()],
            spans: None,
            viewport,
            height_texture,
            material_texture,
            maxima_texture,
            uniform,
            bind_group,
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

    /// How many levels the terrain is drawn with.
    fn level_count(&self) -> u32 {
        self.tile_ceilings.len() as u32
    }

    /// The highest ground anywhere resident, across every level being marched.
    ///
    /// What `ground_at` and `cs_compact` test a climbing ray against to settle
    /// it as sky without walking anything. Taken across every *slot* of each
    /// level rather than across the square in use, so a tile of somewhere else
    /// that has not yet been written over still counts -- which makes this a
    /// bound rather than the highest ground on screen, and worth being able to
    /// read while flying.
    pub fn ceiling(&self) -> f32 {
        (self.base..self.level_count())
            .map(|level| highest(&self.tile_ceilings[level as usize]))
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Whether any level is still short of the tiles it wants.
    pub fn pending(&self) -> bool {
        self.tiles.pending()
    }

    /// Brings residency up to date with where the camera is, and writes the
    /// uniform the march reads.
    ///
    /// Bounded: at most [`Residency::tiles_per_update`] tiles are read, so
    /// crossing a tile boundary costs a known amount rather than a stall. The
    /// exception is the first update, which has nothing resident at all and
    /// takes as long as it needs -- there is no frame to protect yet, and a
    /// single-frame render would otherwise draw an empty world.
    pub fn update(&mut self, queue: &wgpu::Queue, camera: Vec3) {
        // Cleared rather than accumulated: a row in the readout is what this
        // frame cost, not what every frame since the run began cost.
        if let Some(spans) = self.spans.as_mut() {
            *spans = crate::profile::Terrain::default();
        }

        let camera_texels = self
            .placement
            .texel_of_world(f64::from(camera.x), f64::from(camera.z));

        // The coarsest level is brought up first, out of turn, because the
        // height above terrain that decides how many finer levels are worth
        // keeping is read back out of its square. Taking it from the one level
        // that is never dropped keeps the decision independent of what the
        // decision itself makes resident.
        // Accumulated into a local and merged once at the end: `advance` is
        // measured either side of calls that borrow `self` mutably, so holding
        // a borrow of `self.spans` across them would not compile.
        let timed = self.spans.is_some();
        let mut advance = Duration::ZERO;

        let coarsest = self.level_count() - 1;
        loop {
            let clock = crate::profile::Clock::start(timed);
            let work = self.tiles.advance(camera_texels, coarsest);
            advance += clock.elapsed();
            if work.is_empty() {
                break;
            }
            for wanted in work {
                self.upload(queue, wanted);
            }
        }

        let ground = self.ground_height(camera_texels);
        let metres_per_texel = self
            .placement
            .metres_per_texel_x
            .min(self.placement.metres_per_texel_z);
        self.base = detail_base(
            &self.residency,
            metres_per_texel,
            f64::from(camera.y - ground),
            self.level_count(),
        );

        loop {
            let clock = crate::profile::Clock::start(timed);
            let work = self.tiles.advance(camera_texels, self.base);
            advance += clock.elapsed();
            let more = !work.is_empty();
            for wanted in work {
                self.upload(queue, wanted);
            }
            // Every level whole before the first frame; after that, whatever the
            // budget allowed, with the rest picked up next time.
            if !more || self.started {
                break;
            }
        }
        self.started = true;

        let clock = crate::profile::Clock::start(timed);
        self.write_uniform(queue);
        let uniform = clock.elapsed();

        if let Some(spans) = self.spans.as_mut() {
            spans.advance += advance;
            // The uniform is one small `write_buffer`, the same kind of work as
            // the texture uploads and far too small to earn a row of its own.
            spans.write += uniform;
        }
    }

    /// Starts or stops accounting for where an update's time goes.
    pub fn profile(&mut self, on: bool) {
        self.spans = on.then(crate::profile::Terrain::default);
    }

    /// What the last [`Terrain::update`] spent, if it was being watched.
    pub fn spans(&self) -> Option<crate::profile::Terrain> {
        self.spans
    }

    /// Describes the current residency to the shader.
    fn write_uniform(&self, queue: &wgpu::Queue) {
        let levels = self.level_count();
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
            level_count: levels,
            base_level: self.base,
            texel_mask: self.residency.texel_mask(),
            march_steps: self.residency.march_steps(levels),
            ceiling: f32::NEG_INFINITY,
            wall_nudge: wall_nudge(UVec2::new(self.placement.width, self.placement.height)),
            viewport: self.viewport.to_array(),
        };

        for level in self.base..levels {
            let (low, high) = self.tiles.level(level).valid(self.residency.tiles_across);
            let ceiling = highest(&self.tile_ceilings[level as usize]);
            uniform.levels[level as usize] = LevelUniform {
                valid_low: (low * self.residency.tile_texels as i32).to_array(),
                // One short, because the bilinear patch at the last texel of the
                // square reads the texel after it, which belongs to whatever
                // else shares that slot.
                valid_high: (high * self.residency.tile_texels as i32 - IVec2::ONE).to_array(),
                ceiling,
                padding: 0.0,
                more_padding: [0.0; 2],
            };
        }
        uniform.ceiling = self.ceiling();

        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    /// The elevation of the ground under the camera, in world units.
    ///
    /// Read from the mirror of the coarsest level, so it is the ground averaged
    /// over kilometres rather than the peak the camera happens to be over. That
    /// is the right shape of answer for choosing a level that covers kilometres,
    /// and where it is least accurate -- close to the ground, where the relief
    /// it smooths away is a large fraction of the distance -- the finest level
    /// is being drawn anyway.
    fn ground_height(&self, camera_texels: DVec2) -> f32 {
        let level = self.level_count() - 1;
        let across = self.residency.texels_across() as usize;
        let texels = camera_texels / f64::from(1u32 << level);
        let slot = IVec2::new(texels.x.floor() as i32, texels.y.floor() as i32)
            & IVec2::splat(self.residency.texel_mask() as i32);

        let height = self.ground[slot.y as usize * across + slot.x as usize];
        // The camera can legally be over ground the raster says nothing about:
        // past the edge of the survey, or over a hole in it. Sea level is the
        // same fallback the terrain itself draws there.
        if height > crate::terrain::NODATA_BELOW {
            height
        } else {
            0.0
        }
    }

    /// Reads one tile of every product and writes it into its slot.
    ///
    /// A whole tile, always. That is 512 consecutive rows of one file, which is
    /// what the one-row-per-strip layout the tiles are written in is fastest at
    /// -- and it is why nothing here has to care which way the camera moved.
    fn upload(&mut self, queue: &wgpu::Queue, wanted: Wanted) {
        let tile = self.residency.tile_texels;
        let texels = wanted.tile * tile as i32;
        let slot = texels & IVec2::splat(self.residency.texel_mask() as i32);
        let extent = wgpu::Extent3d {
            width: tile,
            height: tile,
            depth_or_array_layers: 1,
        };
        let origin = wgpu::Origin3d {
            x: slot.x as u32,
            y: slot.y as u32,
            z: wanted.level,
        };
        let size = UVec2::splat(tile);
        let count = (tile as usize).pow(2);

        let timed = self.spans.is_some();
        let (mut read, mut convert) = (Duration::ZERO, Duration::ZERO);
        // A `Cell` because `copy` is a closure called four times and the total
        // has to outlive each call; it captures nothing of `self`, which is
        // what lets the reads borrow `self.staging` mutably alongside it.
        let write = std::cell::Cell::new(Duration::ZERO);

        let copy = |texture: &wgpu::Texture, bytes: u32, data: &[u8]| {
            let clock = crate::profile::Clock::start(timed);
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
                    bytes_per_row: Some(tile * bytes),
                    rows_per_image: Some(tile),
                },
                extent,
            );
            write.set(write.get() + clock.elapsed());
        };

        let bytes = count * size_of::<f32>();
        let clock = crate::profile::Clock::start(timed);
        self.heights
            .read_rect(wanted.level, texels, size, &mut self.staging[..bytes]);
        read += clock.elapsed();
        let clock = crate::profile::Clock::start(timed);
        if VERTICAL_EXAGGERATION != 1.0 {
            for height in bytemuck::cast_slice_mut::<u8, f32>(&mut self.staging[..bytes]) {
                *height *= VERTICAL_EXAGGERATION;
            }
        }
        convert += clock.elapsed();
        copy(&self.height_texture, 4, &self.staging[..bytes]);
        if wanted.level == self.level_count() - 1 {
            self.mirror_ground(slot, bytes);
        }

        let bytes = count * size_of::<MaterialId>();
        let clock = crate::profile::Clock::start(timed);
        self.materials
            .read_rect(wanted.level, texels, size, &mut self.staging[..bytes]);
        read += clock.elapsed();
        copy(&self.material_texture, 4, &self.staging[..bytes]);

        let bytes = count * size_of::<f32>();
        let clock = crate::profile::Clock::start(timed);
        self.maxima
            .read_rect(wanted.level, texels, size, &mut self.staging[..bytes]);
        read += clock.elapsed();
        let clock = crate::profile::Clock::start(timed);
        // Narrowed in place, forwards, so the half float for a cell is written
        // over bytes the full-precision one has already been read out of. Two
        // bytes never overtake four.
        let mut ceiling = f32::NEG_INFINITY;
        for cell in 0..count {
            let read = f32::from_le_bytes(
                self.staging[cell * 4..cell * 4 + 4]
                    .try_into()
                    .expect("four bytes"),
            );
            let narrowed = ceiling_half(read * VERTICAL_EXAGGERATION);
            ceiling = ceiling.max(narrowed.to_f32());
            self.staging[cell * 2..cell * 2 + 2].copy_from_slice(&narrowed.to_bits().to_le_bytes());
        }
        convert += clock.elapsed();
        copy(&self.maxima_texture, 2, &self.staging[..count * 2]);

        if let Some(spans) = self.spans.as_mut() {
            spans.read += read;
            spans.convert += convert;
            spans.write += write.get();
            spans.tiles += 1;
        }

        // Read back out of the half floats rather than kept at full precision,
        // so the figure a ray clears a whole level with is the same bound the
        // texture holds.
        let across = self.residency.tiles_across as usize;
        let slot_index =
            (slot.y as usize / tile as usize) * across + slot.x as usize / tile as usize;
        self.tile_ceilings[wanted.level as usize][slot_index] = ceiling;
    }

    /// Copies a just-uploaded coarsest-level tile into the CPU mirror.
    ///
    /// Written from the same staging buffer, at the same slot, as the texture
    /// copy immediately above it, so the mirror wraps exactly as the texture
    /// does and cannot drift out of step with it.
    fn mirror_ground(&mut self, slot: IVec2, bytes: usize) {
        let heights: &[f32] = bytemuck::cast_slice(&self.staging[..bytes]);
        let across = self.residency.texels_across() as usize;
        let tile = self.residency.tile_texels as usize;
        for row in 0..tile {
            let to = (slot.y as usize + row) * across + slot.x as usize;
            self.ground[to..to + tile].copy_from_slice(&heights[row * tile..(row + 1) * tile]);
        }
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
    use crate::terrain::pyramid::{Level, Pyramid, max_pyramid};

    /// Tiles small enough that a test raster is several of them across.
    ///
    /// The store's tiles are 512 texels, which is bigger than any raster a test
    /// can afford to build, so residency is told to work at a scale where its
    /// own arithmetic -- squares, slots, wraps -- is actually exercised.
    fn test_residency() -> Residency {
        Residency {
            tiles_across: 4,
            tile_texels: 8,
            // Whole squares per update, so a test never has to drain a queue.
            tiles_per_update: 4096,
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
        highest(&samples)
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

    /// Every texel of the pyramid bounds the ground it claims, and sits where
    /// the march will go looking for it.
    ///
    /// Both halves matter and they fail differently. A wrong *value* lets rays
    /// through ridges; a wrong *position* is worse, because a slot's address is
    /// a mask of the raster-relative index, so a version that wrapped against
    /// the square's own origin instead would read a neighbouring tile's ceiling
    /// and be wrong everywhere except when the square happened to start on a
    /// multiple of its own width.
    #[test]
    fn every_texel_of_the_pyramid_bounds_the_ground_it_covers() {
        let (device, queue) = crate::scene::test_device();
        let camera_layout = crate::scene::test_camera_layout(&device);
        let storage_layout = crate::deferred::storage_layout(&device);
        let work_layout = crate::reproject::work_layout(&device);
        let args_layout = crate::reproject::args_layout(&device);
        let risk_layout = crate::reproject::risk_layout(&device);
        let reach_layout = crate::reproject::reach_layout(&device);
        let residency = test_residency();
        let across = residency.texels_across();

        // The raster the texels are defined over, kept aside so the oracle can
        // look their squares up directly rather than through anything the
        // upload path touched.
        let raster = Pyramid::build(Level::new(RASTER, RASTER, rugged()));
        let mut terrain = Terrain::new(
            &device,
            &camera_layout,
            &storage_layout,
            &work_layout,
            &args_layout,
            &risk_layout,
            &reach_layout,
            residency,
            UVec2::splat(RASTER),
            Georeferencing::square(RASTER, RASTER, 30.0),
            Sources {
                heights: Box::new(Pyramid::build(Level::new(RASTER, RASTER, rugged()))),
                materials: Box::new(Pyramid::build(Level::new(
                    RASTER,
                    RASTER,
                    vec![MaterialId(0); (RASTER * RASTER) as usize],
                ))),
                maxima: Box::new(max_pyramid(&raster)),
            },
        );
        let raster: &dyn RasterSource = &raster;

        // Off any round number, so no square starts on a multiple of its own
        // width and reading the slots the wrong way round would be visible.
        terrain.update(&queue, Vec3::new(137.0, 100.0, -71.0));
        assert_eq!(terrain.base_level(), 0, "the test wants every level built");
        let levels = terrain.level_count();

        // Rows in a texture-to-buffer copy are padded to 256 bytes, which a
        // thirty-two texel square of half floats is well short of.
        let stride = (across * 2).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pyramid readback"),
            size: u64::from(stride * across * levels),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &terrain.maxima_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(across),
                },
            },
            wgpu::Extent3d {
                width: across,
                height: across,
                depth_or_array_layers: levels,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let bytes = readback.get_mapped_range(..).expect("buffer not mapped");

        let mut checked = 0;
        for level in 0..levels {
            let (low, high) = terrain.tiles.level(level).valid(residency.tiles_across);
            let (low, high) = (
                low * residency.tile_texels as i32,
                high * residency.tile_texels as i32,
            );
            let layer = &bytes[level as usize * stride as usize * across as usize..];

            for cell_y in low.y..high.y {
                for cell_x in low.x..high.x {
                    let cell = IVec2::new(cell_x, cell_y);
                    let slot = cell & IVec2::splat(residency.texel_mask() as i32);
                    let row: &[u16] = bytemuck::cast_slice(
                        &layer[slot.y as usize * stride as usize
                            ..slot.y as usize * stride as usize + across as usize * 2],
                    );
                    let got = half::f16::from_bits(row[slot.x as usize]).to_f32();

                    // Asked only of texels whose square lands on real ground.
                    // A resident square legitimately hangs off the raster, and
                    // out there the pyramid and this oracle both repeat the
                    // border, but at their own granularities -- so they agree
                    // on the bound without agreeing on the figure.
                    let reach = (cell + IVec2::ONE) << level;
                    if cell.min_element() < 0 || reach.max_element() >= RASTER as i32 {
                        continue;
                    }
                    // Half precision, rounded towards positive infinity, so a
                    // texel is at or a little above the ground it covers and
                    // never below it. Below would be a ridge a ray can pass
                    // through; above only costs a descent.
                    let want = cell_defined(raster, level, cell);
                    assert_eq!(
                        got,
                        ceiling_half(want).to_f32(),
                        "level {level} texel {cell} in slot {slot} holds {got} \
                         where the levels reading it want {want}"
                    );
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 1000,
            "only {checked} texels sat wholly on the raster, which is too few"
        );
    }
}
