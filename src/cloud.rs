//! The noise a cloud is carved out of.
//!
//! Two 3D volumes, built once at load and read for the rest of the run: a
//! shape, which says where cloud is at all, and a detail, which says what its
//! edges look like. Both tile, which is what lets eight megabytes cover a
//! hundred kilometres of sky; see `src/cloud.wgsl` for how the lattice is
//! folded to make that true.
//!
//! Generated rather than shipped. A 128-cubed volume is 8 MB of file that would
//! have to live somewhere, be found at run time, and be kept in step with the
//! shader that reads it -- against a third of a second of compute that produces
//! the same bytes on every machine, every run. The same trade the scattering
//! tables in [`crate::sky`] make, and for the same reasons.

/// Side of the shape volume, in texels. Must match `SHAPE_SIZE` in
/// `src/cloud.wgsl`.
///
/// A hundred and twenty-eight, tiled over four kilometres, puts a texel at
/// about thirty metres -- fine enough that the billows have edges and coarse
/// enough that the whole volume is eight megabytes.
pub const SHAPE_SIZE: u32 = 128;

/// Side of the detail volume, in texels. Must match `DETAIL_SIZE` in
/// `src/cloud.wgsl`.
///
/// Tiled over a couple of hundred metres, so a texel is a few metres. It is
/// only ever read where the shape has already put cloud, so it does not need
/// the resolution the shape does -- what it needs is to be small enough to sit
/// in cache while the march is stepping through it.
pub const DETAIL_SIZE: u32 = 32;

/// Side of the weather map, in texels. Must match `WEATHER_SIZE` in
/// `src/cloud.wgsl`.
///
/// Two hundred and fifty-six over sixty kilometres is a texel every 234 m,
/// which is far coarser than a cloud and exactly right for what this holds:
/// not cloud, but where cloud is *allowed*. The finest thing in it is the
/// third octave, about five kilometres across, so a feature still spans twenty
/// texels. It was 512 first, and that cost 0.62 ms a frame to say the same
/// thing at four times the resolution of anything in it.
pub const WEATHER_SIZE: u32 = 256;

/// How many decks the weather describes. Must match `DECKS` in
/// `src/cloud.wgsl`.
///
/// Three, at the three heights cloud actually forms at: a low deck that
/// mountains reach into, a middle one, and cirrus. They are layers of one array
/// texture rather than three textures, so the march reads them with one binding
/// and an index.
pub const DECKS: usize = 3;

/// Cells across the ceiling cache, in X and Z. Must match `CEILING_ACROSS` in
/// `src/cloud_march.wgsl`.
///
/// Exactly two weather texels to a cell, so the cache tiles with the map it is
/// built from and one fold serves both. A cell is 469 m across, which is coarse
/// against a cloud and about right for a thing whose only job is to say whether
/// a ray may skip the next half kilometre without looking.
pub const CEILING_ACROSS: u32 = 128;

/// Cells up the ceiling cache. Must match `CEILING_SLICES` in
/// `src/cloud_march.wgsl`.
///
/// Twenty-four over [`CEILING_TOP`] is a cell every 500 m, which is a fifth of
/// the low deck's thickness -- enough for a ray to skip the air between the
/// decks and the air under them, which is most of what there is to skip.
pub const CEILING_SLICES: u32 = 24;

/// How high the ceiling cache reaches, in metres. Must match `CEILING_TOP` in
/// `src/cloud_march.wgsl`.
///
/// Above the highest deck, so a ray that climbs out of the grid has left every
/// cloud behind rather than merely left the table.
const CEILING_TOP: f32 = 12000.0;

/// How much world one tile of the weather map covers, in metres. Must match
/// `WEATHER_TILE` in `src/cloud_march.wgsl`.
///
/// Sixty kilometres, which is the scale weather systems come at and far enough
/// that no flight crosses the seam twice in a way anyone would notice.
///
/// Nothing on this side addresses the map -- the march does, in world metres --
/// so Rust holds this only so a test can check that the shader still says the
/// same number. See `the_shader_and_rust_agree_on_the_cloud_grid`.
#[allow(
    dead_code,
    reason = "mirrored from the shader for the test comparing them"
)]
const WEATHER_TILE: f32 = 60_000.0;

/// Where each deck sits: base, top, how far its base may lift, and how dense.
///
/// Three heights cloud actually forms at, and they deliberately do not overlap
/// -- a height belongs to at most one deck, including after the base has lifted
/// by its whole swing, which is what lets a sample in the march cost one weather
/// fetch rather than three. The low deck runs from 700 m, which is below the
/// tops of the mountains this flies over: that is the point, and it is what the
/// terrain coupling later has to survive.
///
/// The densities fall with height because the cloud does: cirrus is ice crystals
/// where cumulus is water, and what it takes out of a beam is a quarter of what
/// the same thickness of cumulus would.
const DECK_SLABS: [[f32; 4]; DECKS] = [
    [700.0, 3000.0, 400.0, 1.0],
    [3500.0, 6000.0, 500.0, 0.65],
    [8500.0, 10000.0, 300.0, 0.25],
];

/// Texels across each light volume, and slices up it. Must match `LIGHT_ACROSS`
/// and `LIGHT_SLICES` in `src/cloud_march.wgsl`.
///
/// Over [`LIGHT_EXTENT`] and [`CEILING_TOP`] that is 312 m across and 250 m up.
/// Coarse against a cloud, and about right for what these hold: not the cloud
/// but the shadow of it, which is soft by the time it has been cast through a
/// kilometre of anything. Fourteen megabytes apiece.
pub const LIGHT_ACROSS: u32 = 192;
pub const LIGHT_SLICES: u32 = 48;

/// How much world the light volumes span horizontally, in metres.
///
/// Camera-centred, and snapped to a whole texel so the shadows do not crawl
/// under the camera: the volume is a fact about the world, and a fact about the
/// world that moved when the eye did would shimmer. Sixty kilometres is far
/// enough that what falls outside is behind most of a hundred kilometres of
/// haze, where the sampler's clamp is as good an answer as any.
const LIGHT_EXTENT: f32 = 60_000.0;

/// The lowest the sun may be taken to be when the volume's columns are leaned
/// along it.
///
/// The lean is `sun.xz / sun.y`, which runs away as the sun sets: at one degree
/// a column crossing twelve kilometres of height would cross seven hundred of
/// ground. Floored at about eight and a half degrees, below which the shadows
/// are geometrically wrong and it does not matter, because what they are
/// shadowing has by then been taken out by the air instead -- the transmittance
/// table has the direct sun at a fraction of a per cent there. It also keeps a
/// sun below the horizon from putting an infinity in the uniform.
const SHEAR_FLOOR: f32 = 0.15;

/// The format the half-resolution cloud buffer is held in.
///
/// Three channels of scattered radiance and one of transmittance, which is more
/// than eight bits a channel can hold: sunlit cloud and cloud in its own shadow
/// are two orders of magnitude apart, and the transmittance is multiplied
/// through everything behind it, so a step in it is a step in the whole
/// background. The same format and the same reasoning as `LUT_FORMAT` in
/// `src/sky.rs`.
const CLOUD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The format of the ceiling cache and of the cloud's own depth.
///
/// Both hold one number in metres over a range of a hundred kilometres, and
/// `R32Float` is the one single-channel float format that is storage-writable in
/// core WebGPU. See the note on [`FORMAT`].
const DISTANCE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Threads per workgroup for the ceiling build, in each axis. Must match
/// `@workgroup_size` on `cs_cloud_ceiling`.
const CEILING_GROUP: u32 = 4;

/// The same for the march, which is a flat image and takes a flat group. Must
/// match `@workgroup_size` on `cs_cloud_march`.
const MARCH_GROUP: u32 = 8;

/// How long the weather takes to come back round to where it was, in seconds.
///
/// Ten minutes. Long enough that a front does not visibly cycle within a
/// flight, short enough that a sky left alone is not the same sky an hour
/// later.
const WEATHER_PERIOD: f32 = 600.0;

/// Threads per workgroup, in each axis. Must match `@workgroup_size` on the two
/// noise entry points in `src/cloud.wgsl`.
const GROUP: u32 = 4;

/// The same, for the weather, which is a flat map and takes a flat group. Must
/// match `@workgroup_size` on `cs_cloud_weather`.
const WEATHER_GROUP: u32 = 8;

/// The format both volumes are held in.
///
/// `Rgba8Unorm` is storage-writable and filterable in core WebGPU, which is the
/// pair of properties that rules almost everything else out -- see
/// `FIELD_FORMAT` in `src/air.rs` and `LUT_FORMAT` in `src/sky.rs`, which hit
/// the same wall from different sides. Eight bits a channel is ample: this is
/// noise being thresholded, not a measurement, and the quantisation is far
/// below the softest edge the march can draw.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// What kind of day it is.
///
/// A name rather than a set of numbers, because the numbers are not
/// independent: a sky with solid low cloud does not also have crisp cirrus over
/// it, and a scattered fair-weather sky has cumulus that heap rather than
/// stratus that lie flat. What each one means to each deck is [`Preset::decks`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Preset {
    /// Nothing at all, at any height. Exactly nothing; see [`Deck::look`].
    Clear,
    /// Scattered fair-weather cumulus with a little cirrus over them.
    #[default]
    Fair,
    /// Cumulus grown together into a broken sheet, with more above.
    Broken,
    /// A solid grey lid, low and flat.
    Overcast,
    /// The lid, lower and much thicker, heaped up underneath.
    Storm,
}

/// What one preset asks of one deck.
///
/// Mirrors `Deck` in `src/cloud.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeckUniform {
    /// The field values that map to no cloud and to solid cloud, then the lean
    /// -- nothing for flat stratus, one for heaped cumulus -- and the density.
    look: [f32; 4],
    /// Where this deck's fields are drawn from, and three spare.
    seed: [u32; 4],
    /// This deck's entry from [`DECK_SLABS`].
    ///
    /// A constant of the world rather than of the preset -- what a name like
    /// `storm` changes is how much cloud there is and how dense, not what
    /// altitude cumulus forms at. It rides in the uniform anyway because the
    /// march wants it beside the rest, and one buffer beats two.
    slab: [f32; 4],
}

/// Mirrors the `Weather` uniform block in `src/cloud.wgsl` and in
/// `src/cloud_march.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WeatherUniform {
    decks: [DeckUniform; DECKS],
    /// Seconds since the world started, then [`WEATHER_PERIOD`].
    clock: [f32; 4],
    /// The lowest and highest metres any deck can put cloud at, then two spare.
    ///
    /// What the march clips its ray against before it steps at all: everything
    /// outside this is sky, and for a camera looking down at the ground it is
    /// most of the ray.
    span: [f32; 4],
    /// The light volumes' near corner in x and z, the height of their lowest
    /// slice, and the metres one of their texels covers across.
    light_origin: [f32; 4],
    /// How far a sun column leans per metre it climbs, then the metres of ray
    /// one slice is worth towards the sun, then the metres of height one slice
    /// is worth -- which is what the same slice is worth straight up.
    light_walk: [f32; 4],
}

/// Where each deck's fields are drawn from.
///
/// Far apart, so that the three fields one deck takes from consecutive seeds --
/// cover, lean, density, base -- cannot collide with another deck's.
const DECK_SEEDS: [u32; DECKS] = [0x4c6f7700, 0x4d696400, 0x48696700];

impl Preset {
    /// What this preset asks of the low, middle and high decks.
    ///
    /// The four numbers are the two ends of the cover ramp, the lean and the
    /// density. A ramp whose ends both sit above one is a deck with no cloud in
    /// it at all: the field cannot reach them, so the cover is a hard zero
    /// rather than a small number, which is what lets `clear` be clear.
    ///
    /// The ramps narrow as the sky fills. Fair weather wants cumulus with sky
    /// between them, so its ramp is high and short -- only the peaks of the
    /// field become cloud. Overcast wants the opposite: a ramp so low that
    /// almost the whole field clears it, leaving breaks only where the field
    /// dips hardest.
    fn decks(self) -> [[f32; 4]; DECKS] {
        const NONE: [f32; 4] = [2.0, 3.0, 0.0, 0.0];
        match self {
            Self::Clear => [NONE, NONE, NONE],
            Self::Fair => [[0.50, 0.72, 1.0, 0.75], NONE, [0.60, 0.85, 0.0, 0.35]],
            Self::Broken => [
                [0.42, 0.66, 0.8, 0.9],
                [0.58, 0.82, 0.4, 0.5],
                [0.52, 0.80, 0.0, 0.4],
            ],
            Self::Overcast => [[0.16, 0.44, 0.2, 1.0], [0.40, 0.70, 0.2, 0.7], NONE],
            Self::Storm => [
                [0.04, 0.34, 0.9, 1.0],
                [0.24, 0.56, 0.6, 1.0],
                [0.45, 0.75, 0.0, 0.5],
            ],
        }
    }

    fn uniform(
        self,
        elapsed: std::time::Duration,
        eye: glam::Vec3,
        sun: glam::Vec3,
    ) -> WeatherUniform {
        let looks = self.decks();
        let across = LIGHT_EXTENT / LIGHT_ACROSS as f32;
        let up = CEILING_TOP / LIGHT_SLICES as f32;
        // Centred on the eye and then snapped to a whole texel. Both halves
        // matter: centred, so the resolution is spent where the camera is, and
        // snapped, so the lattice stands still in the world while the camera
        // moves across it. A volume that slid with the eye would put a cloud's
        // shadow in a slightly different place every frame, which reads as the
        // whole sky crawling.
        let centre = (glam::Vec2::new(eye.x, eye.z) / across).round() * across;
        let corner = centre - LIGHT_EXTENT * 0.5;
        // The sun's own lean, floored so a setting sun cannot run it away or a
        // set one invert it. See [`SHEAR_FLOOR`].
        let climb = sun.y.max(SHEAR_FLOOR);
        let lean = glam::Vec2::new(sun.x, sun.z) / climb;
        WeatherUniform {
            decks: std::array::from_fn(|deck| DeckUniform {
                look: looks[deck],
                seed: [DECK_SEEDS[deck], 0, 0, 0],
                slab: DECK_SLABS[deck],
            }),
            clock: [elapsed.as_secs_f32(), WEATHER_PERIOD, 0.0, 0.0],
            span: [cloud_span().0, cloud_span().1, 0.0, 0.0],
            light_origin: [corner.x, corner.y, 0.0, across],
            // A slice of height is a longer piece of a leaning ray than of a
            // plumb one, by exactly the sun's climb.
            light_walk: [lean.x, lean.y, up / climb, up],
        }
    }
}

/// The lowest and highest a cloud can be, over every deck.
///
/// Derived from [`DECK_SLABS`] rather than written down beside it, because a
/// second copy of a bound is a bound that can disagree with what it bounds. A
/// deck's top lifts with its base, so the highest it reaches is its top plus its
/// whole swing.
fn cloud_span() -> (f32, f32) {
    DECK_SLABS
        .iter()
        .fold((f32::MAX, 0.0f32), |(low, high), s| {
            (low.min(s[0]), high.max(s[1] + s[2]))
        })
}

/// The two volumes, the weather over them, and the pipelines that fill them.
pub struct Cloud {
    #[allow(dead_code, reason = "read only by the noise readback tests")]
    shape: wgpu::Texture,
    #[allow(dead_code, reason = "read only by the noise readback tests")]
    detail: wgpu::Texture,
    /// One texel per patch of sky per deck: how much cloud it may hold, which
    /// way that cloud leans, how dense it is and where its base sits.
    ///
    /// Rewritten every frame rather than built once, because it moves. It is
    /// the cheapest thing in the frame by a wide margin -- see the `weather`
    /// row -- and evolving it is what stops a sky being the same sky for the
    /// length of a flight.
    #[allow(dead_code, reason = "read only by the weather readback tests")]
    weather: wgpu::Texture,
    shape_view: wgpu::TextureView,
    detail_view: wgpu::TextureView,
    weather_view: wgpu::TextureView,
    uniform: wgpu::Buffer,
    weather_group: wgpu::BindGroup,
    storage_group: wgpu::BindGroup,
    /// Fills the weather map. Kept, unlike the two below: it runs every frame.
    weather_pipeline: wgpu::ComputePipeline,
    /// Dropped once the volumes are filled; see [`Build`].
    build: Option<Build>,
}

/// Everything the build needs and the frame does not.
///
/// Dropped the moment the two dispatches have run, exactly as `Build` in
/// `src/sky.rs` and `src/air.rs` are. There is nothing large in it -- the
/// volumes themselves outlive it -- but a pipeline that can never run again is
/// a pipeline someone can be tempted to run.
struct Build {
    shape: wgpu::ComputePipeline,
    detail: wgpu::ComputePipeline,
}

impl Cloud {
    pub fn new(device: &wgpu::Device) -> Self {
        let volume = |label, size| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: FORMAT,
                // `COPY_SRC` so a test can read the volume back and say whether
                // it tiles, which is the one property that cannot be seen by
                // looking at a frame: a seam in a cloud looks like a cloud.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let shape = volume("cloud shape noise", SHAPE_SIZE);
        let detail = volume("cloud detail noise", DETAIL_SIZE);
        let weather = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud weather"),
            size: wgpu::Extent3d {
                width: WEATHER_SIZE,
                height: WEATHER_SIZE,
                depth_or_array_layers: DECKS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            // Three layers of one 2D texture, not a volume: the decks sit at
            // different heights and nothing ever interpolates between them.
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let (shape_view, detail_view) = (view(&shape), view(&detail));
        let weather_view = weather.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud weather uniform"),
            size: std::mem::size_of::<WeatherUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let weather_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud weather layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let weather_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud weather group"),
            layout: &weather_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });

        let written = |binding, dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: FORMAT,
                view_dimension: dimension,
            },
            count: None,
        };
        // All three written things in one layout, and one bind group for all
        // three kernels. A kernel need only be given the bindings it uses, so
        // the two noise builds could have had a narrower group of their own --
        // but three descriptions of the same three textures is three things to
        // keep in step, and none of them is ever bound for reading here.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud storage layout"),
            entries: &[
                written(0, wgpu::TextureViewDimension::D3),
                written(1, wgpu::TextureViewDimension::D3),
                written(2, wgpu::TextureViewDimension::D2Array),
            ],
        });
        let storage_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud storage group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shape_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&detail_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&weather_view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cloud.wgsl").into()),
        });
        // Group 1 is the domain uniform and group 3 is what is written, which
        // is the convention everywhere else here. There is no camera and
        // nothing shared to read: the noise is a function of position and a
        // seed, and the weather of position, a seed and the clock.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud pipeline layout"),
            bind_group_layouts: &[None, Some(&weather_layout), None, Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            shape_view,
            detail_view,
            weather_view,
            uniform,
            weather_group,
            storage_group,
            weather_pipeline: pipeline("cloud weather", "cs_cloud_weather"),
            build: Some(Build {
                shape: pipeline("cloud shape", "cs_cloud_shape"),
                detail: pipeline("cloud detail", "cs_cloud_detail"),
            }),
            shape,
            detail,
            weather,
        }
    }

    /// Fills both volumes, once.
    ///
    /// Called from [`crate::scene::Scene::update`] on the first update, for the
    /// reason `Sky::ensure_built` is: filling them needs a queue, and a
    /// constructor has none. Waits for the GPU, so the cost lands at load with
    /// the tile reads and the scattering tables rather than being smeared
    /// across the opening frames of a flight, where it would read as the frame
    /// being slow.
    pub fn ensure_built(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(build) = self.build.take() else {
            return;
        };
        let started = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cloud noise"),
        });
        {
            // One pass for both: they write different textures and neither
            // reads what the other wrote, so there is nothing for a pass
            // boundary to order.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cloud noise"),
                timestamp_writes: None,
            });
            pass.set_bind_group(1, &self.weather_group, &[]);
            pass.set_bind_group(3, &self.storage_group, &[]);
            let groups = |size: u32| size.div_ceil(GROUP);
            pass.set_pipeline(&build.shape);
            pass.dispatch_workgroups(groups(SHAPE_SIZE), groups(SHAPE_SIZE), groups(SHAPE_SIZE));
            pass.set_pipeline(&build.detail);
            pass.dispatch_workgroups(
                groups(DETAIL_SIZE),
                groups(DETAIL_SIZE),
                groups(DETAIL_SIZE),
            );
        }
        queue.submit(std::iter::once(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        log::info!(
            "cloud: built {}^3 shape and {}^3 detail noise, {:.1} MiB, in {:.2?}",
            SHAPE_SIZE,
            DETAIL_SIZE,
            bytes() as f64 / (1024.0 * 1024.0),
            started.elapsed(),
        );
    }

    /// Says what kind of day this frame is and how far into it the world is.
    ///
    /// Uploaded every frame rather than when it changes, for the reason the sky
    /// uploads its sun every frame: the clock moves whatever else does, and a
    /// branch to avoid rewriting eighty bytes costs more than the write.
    pub fn set_frame(
        &self,
        queue: &wgpu::Queue,
        preset: Preset,
        elapsed: std::time::Duration,
        eye: glam::Vec3,
        sun: glam::Vec3,
    ) {
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&preset.uniform(elapsed, eye, sun)),
        );
    }

    /// Records the weather into an already-started compute pass.
    ///
    /// Every frame, because the weather moves. Its own dispatch rather than
    /// part of the march's, so the readout says what it costs -- a quarter of a
    /// million texels of three-octave noise is not nothing, and the only way to
    /// find out whether it matters is for it to have a row.
    pub fn draw_weather(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.weather_pipeline);
        pass.set_bind_group(1, &self.weather_group, &[]);
        pass.set_bind_group(3, &self.storage_group, &[]);
        let across = WEATHER_SIZE.div_ceil(WEATHER_GROUP);
        pass.dispatch_workgroups(across, across, DECKS as u32);
    }

    /// The two volumes and the weather over them, for the march to bind.
    pub fn views(&self) -> (&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView) {
        (&self.shape_view, &self.detail_view, &self.weather_view)
    }
}

/// What the two volumes cost in video memory.
fn bytes() -> u64 {
    let cube = |size: u64| size * size * size * 4;
    cube(u64::from(SHAPE_SIZE)) + cube(u64::from(DETAIL_SIZE))
}

/// The coarse bound on where cloud can be, and the march that reads it.
///
/// Separate from [`Cloud`] because the two answer different questions and are
/// built at different times. `Cloud` holds fields: what the sky is made of and
/// what kind of day it is, neither of which knows anything about a camera. This
/// holds a view of them: screen-sized buffers that follow a resize, bound
/// against a G-buffer and a set of scattering tables that belong to other
/// modules. A resize throws all of this away and none of that.
pub struct March {
    /// An upper bound on the extinction anywhere in each cell of a coarse world
    /// grid, rebuilt every frame from the weather.
    #[allow(dead_code, reason = "read through its view")]
    ceiling: wgpu::Texture,
    /// Scattered radiance and transmittance, at half the frame's resolution.
    ///
    /// Nothing reads it yet; the composite lands next. `COPY_SRC` for the reason
    /// the G-buffer's targets carry it -- a frame this does not appear in says
    /// nothing about what is in it.
    colour: wgpu::Texture,
    /// How far along each of those rays the cloud it found actually was.
    depth: wgpu::Texture,
    /// How much of the sun reaches each point of a coarse world grid, and how
    /// much of the sky does. One shape, two leans; see `light_position` in
    /// `src/cloud_march.wgsl`.
    #[allow(dead_code, reason = "read through their views")]
    sun_light: wgpu::Texture,
    #[allow(dead_code, reason = "read through their views")]
    sky_light: wgpu::Texture,
    colour_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    ceiling_view: wgpu::TextureView,
    sun_light_view: wgpu::TextureView,
    sky_light_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    edge_sampler: wgpu::Sampler,
    ceiling_group: wgpu::BindGroup,
    /// One group per volume, against one layout: the only difference between
    /// the two dispatches is which texture they write and which entry point
    /// they run.
    light_groups: [wgpu::BindGroup; 2],

    march_layout: wgpu::BindGroupLayout,
    march_group: wgpu::BindGroup,
    ceiling_pipeline: wgpu::ComputePipeline,
    sun_light_pipeline: wgpu::ComputePipeline,
    sky_light_pipeline: wgpu::ComputePipeline,
    march_pipeline: wgpu::ComputePipeline,
    /// The half-resolution size every buffer above was built at.
    size: glam::UVec2,
}

/// The world-fixed things the march reads, and the two samplers it reads them
/// with.
///
/// A bundle rather than six parameters, because every one of them outlives a
/// resize and is named again unchanged when the screen-sized half does change --
/// which is exactly the split [`March::resize`] has to make.
struct Lit<'a> {
    ceiling: &'a wgpu::TextureView,
    sun: &'a wgpu::TextureView,
    sky: &'a wgpu::TextureView,
    tiling: &'a wgpu::Sampler,
    edge: &'a wgpu::Sampler,
}

impl March {
    /// Builds the two passes against the fields, tables and G-buffer they read.
    ///
    /// Every layout but its own comes from whoever owns it -- the camera from
    /// the scene, the sun and its tables from [`crate::sky::Sky`] -- for the
    /// reason `Shading::new` takes the same three: one description of what a
    /// group holds, held in one place.
    pub fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        sky_layout: &wgpu::BindGroupLayout,
        sun_tables_layout: &wgpu::BindGroupLayout,
        cloud: &Cloud,
        gbuffer: &crate::deferred::GBuffer,
    ) -> Self {
        // A deck reaching above the cache is a deck the march never finds: the
        // ceiling reads as empty above its own top, and the ray skips straight
        // past. Checked rather than trusted because the failure is silent -- the
        // cloud is simply not drawn, and nothing says why.
        let (_, highest) = cloud_span();
        assert!(
            highest <= CEILING_TOP,
            "a deck reaches {highest} m, above the {CEILING_TOP} m the ceiling cache covers"
        );
        let ceiling = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud ceiling"),
            size: wgpu::Extent3d {
                width: CEILING_ACROSS,
                height: CEILING_SLICES,
                depth_or_array_layers: CEILING_ACROSS,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: DISTANCE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let ceiling_view = ceiling.create_view(&Default::default());

        let volume = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: LIGHT_ACROSS,
                    height: LIGHT_ACROSS,
                    depth_or_array_layers: LIGHT_SLICES,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: CLOUD_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        // `Rgba16Float` for one channel apiece, which is three quarters wasted
        // and is still the right format: it is the only core format that is
        // both storage-writable and *filterable*, and these are read between
        // their texels every time. An eight-bit unorm would fit and would band
        // -- a shadow gradient across a deck is exactly the smooth ramp that
        // shows quantisation. See `FORMAT` above for the same wall from the
        // other side.
        let sun_light = volume("cloud sun light");
        let sky_light = volume("cloud sky light");
        let sun_light_view = sun_light.create_view(&Default::default());
        let sky_light_view = sky_light.create_view(&Default::default());

        // Repeating in every axis. All three fields the march reads tile, and
        // reading them wrapped is the whole reason eight megabytes of noise
        // covers a hundred kilometres of sky. Its own sampler rather than the
        // sky's, which clamps: a table runs to the ends of its range and stops,
        // where these come back round.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud field sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // The light volumes do not tile, so they get the other kind. Clamped, so
        // a point outside reads the nearest column rather than one from the far
        // side of the world.
        let edge_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cloud light sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let uniform = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sampled = |binding, dimension, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: dimension,
                multisampled: false,
            },
            count: None,
        };
        let written = |binding, format, dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: dimension,
            },
            count: None,
        };

        // The ceiling build reads the weather and writes the cache, and holds
        // nothing else: it is a function of the forecast and of the constants
        // the decks are described by. Deliberately *not* the layout below with
        // the extra entries left unbound -- the cache is bound writable here and
        // sampled there, and wgpu tracks that across a whole pass. See the note
        // on `Build` in `src/sky.rs`.
        let ceiling_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud ceiling layout"),
            entries: &[
                uniform,
                sampled(1, wgpu::TextureViewDimension::D2Array, true),
                written(9, DISTANCE_FORMAT, wgpu::TextureViewDimension::D3),
            ],
        });
        let (_, _, weather_view) = cloud.views();
        let ceiling_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud ceiling group"),
            layout: &ceiling_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cloud.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(weather_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&ceiling_view),
                },
            ],
        });

        let march_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud march layout"),
            entries: &[
                uniform,
                sampled(1, wgpu::TextureViewDimension::D2Array, true),
                sampled(2, wgpu::TextureViewDimension::D3, true),
                sampled(3, wgpu::TextureViewDimension::D3, true),
                // Never filtered, and the layout says so: interpolating between
                // maxima returns less than the true maximum of the cell a sample
                // is in, which is a hole in a cloud.
                sampled(4, wgpu::TextureViewDimension::D3, false),
                sampled(5, wgpu::TextureViewDimension::D2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                written(7, CLOUD_FORMAT, wgpu::TextureViewDimension::D2),
                written(8, DISTANCE_FORMAT, wgpu::TextureViewDimension::D2),
                sampled(11, wgpu::TextureViewDimension::D3, true),
                sampled(12, wgpu::TextureViewDimension::D3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Everything a light column needs to know what is in front of it, and
        // the one texture it writes. The volumes it reads are deliberately
        // absent: each is bound writable here, and wgpu tracks that across a
        // whole pass.
        let light_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud light layout"),
            entries: &[
                uniform,
                sampled(1, wgpu::TextureViewDimension::D2Array, true),
                sampled(2, wgpu::TextureViewDimension::D3, true),
                sampled(3, wgpu::TextureViewDimension::D3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                written(10, CLOUD_FORMAT, wgpu::TextureViewDimension::D3),
            ],
        });
        let (shape_view, detail_view, _) = cloud.views();
        let light_group = |label, into: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &light_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: cloud.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(weather_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(shape_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(detail_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(into),
                    },
                ],
            })
        };
        let light_groups = [
            light_group("cloud sun light group", &sun_light_view),
            light_group("cloud sky light group", &sky_light_view),
        ];

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud march shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cloud_march.wgsl").into()),
        });
        let pipeline = |label, entry, layouts: &[Option<&wgpu::BindGroupLayout>]| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        // The ceiling build needs no camera, no sun and no tables: what it says
        // is true of the world rather than of a view of it.
        let ceiling_pipeline = pipeline(
            "cloud ceiling",
            "cs_cloud_ceiling",
            &[None, None, None, Some(&ceiling_layout)],
        );
        // The light builds need neither camera nor tables either: what reaches
        // a point is a fact about the cloud between it and the sun, which is
        // the same fact from anywhere it is looked at. That is the whole reason
        // there is a volume rather than a march per shaded sample.
        let sun_light_pipeline = pipeline(
            "cloud sun light",
            "cs_cloud_sun_light",
            &[None, None, None, Some(&light_layout)],
        );
        let sky_light_pipeline = pipeline(
            "cloud sky light",
            "cs_cloud_sky_light",
            &[None, None, None, Some(&light_layout)],
        );
        // The march needs all four. Group 2 is the reduced read of the two
        // build-once tables rather than the whole set: the sky-view table and
        // the two aerial volumes are what the *composite* wants, and leaving
        // them out here is what keeps this inside the sampled-texture budget.
        let march_pipeline = pipeline(
            "cloud march",
            "cs_cloud_march",
            &[
                Some(camera_layout),
                Some(sky_layout),
                Some(sun_tables_layout),
                Some(&march_layout),
            ],
        );

        let size = half_of(gbuffer.size);
        let (colour, depth) = Self::buffers(device, size);
        let colour_view = colour.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let lit = Lit {
            ceiling: &ceiling_view,
            sun: &sun_light_view,
            sky: &sky_light_view,
            tiling: &sampler,
            edge: &edge_sampler,
        };
        let march_group = Self::bind(
            device,
            &march_layout,
            cloud,
            gbuffer,
            &lit,
            &colour_view,
            &depth_view,
        );

        Self {
            ceiling,
            sun_light,
            sky_light,
            colour,
            depth,
            colour_view,
            depth_view,
            ceiling_view,
            sun_light_view,
            sky_light_view,
            sampler,
            edge_sampler,
            ceiling_group,
            light_groups,
            march_layout,
            march_group,
            ceiling_pipeline,
            sun_light_pipeline,
            sky_light_pipeline,
            march_pipeline,
            size,
        }
    }

    /// Follows the render target to a new size.
    ///
    /// The cache does not move -- it is a fact about the world, at a resolution
    /// of its own -- so only the two screen-sized buffers and the group naming
    /// them are rebuilt. Called from [`crate::scene::Scene::resize`], which also
    /// hands over the rebuilt G-buffer this reads the depth of.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        cloud: &Cloud,
        gbuffer: &crate::deferred::GBuffer,
    ) {
        self.size = half_of(gbuffer.size);
        let (colour, depth) = Self::buffers(device, self.size);
        self.colour_view = colour.create_view(&Default::default());
        self.depth_view = depth.create_view(&Default::default());
        self.colour = colour;
        self.depth = depth;
        self.march_group = Self::bind(
            device,
            &self.march_layout,
            cloud,
            gbuffer,
            &Lit {
                ceiling: &self.ceiling_view,
                sun: &self.sun_light_view,
                sky: &self.sky_light_view,
                tiling: &self.sampler,
                edge: &self.edge_sampler,
            },
            &self.colour_view,
            &self.depth_view,
        );
    }

    fn buffers(device: &wgpu::Device, size: glam::UVec2) -> (wgpu::Texture, wgpu::Texture) {
        let buffer = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        (
            buffer("cloud colour", CLOUD_FORMAT),
            buffer("cloud depth", DISTANCE_FORMAT),
        )
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        cloud: &Cloud,
        gbuffer: &crate::deferred::GBuffer,
        lit: &Lit<'_>,
        colour: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let (shape, detail, weather) = cloud.views();
        let texture = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        let sampler = |binding, which| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Sampler(which),
        };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud march group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cloud.uniform.as_entire_binding(),
                },
                texture(1, weather),
                texture(2, shape),
                texture(3, detail),
                texture(4, lit.ceiling),
                texture(5, &gbuffer.depth),
                sampler(6, lit.tiling),
                texture(7, colour),
                texture(8, depth),
                texture(11, lit.sun),
                texture(12, lit.sky),
                sampler(13, lit.edge),
            ],
        })
    }

    /// Records the ceiling build into an already-started compute pass.
    ///
    /// Every frame, after the weather it is built from and before the march that
    /// reads it -- and in a pass of its own on both counts, because a pass
    /// boundary is what makes one dispatch's writes visible to the next.
    pub fn draw_ceiling(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.ceiling_pipeline);
        pass.set_bind_group(3, &self.ceiling_group, &[]);
        let across = CEILING_ACROSS.div_ceil(CEILING_GROUP);
        pass.dispatch_workgroups(across, CEILING_SLICES.div_ceil(CEILING_GROUP), across);
    }

    /// Records both light volumes into an already-started compute pass.
    ///
    /// After the ceiling and before the march, though only the second ordering
    /// is forced: these read the weather and the noise, and the march reads
    /// these. Both in the one pass, because neither reads what the other writes
    /// -- one is the sun's own line through the cloud and the other is the
    /// plumb line, and no column of either crosses a column of the other.
    pub fn draw_light(&self, pass: &mut wgpu::ComputePass<'_>) {
        let across = LIGHT_ACROSS.div_ceil(MARCH_GROUP);
        for (pipeline, group) in [
            (&self.sun_light_pipeline, &self.light_groups[0]),
            (&self.sky_light_pipeline, &self.light_groups[1]),
        ] {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(3, group, &[]);
            pass.dispatch_workgroups(across, across, 1);
        }
    }

    /// Records the march. The caller has set group 0 to the camera.
    pub fn draw(&self, pass: &mut wgpu::ComputePass<'_>, sky: &crate::sky::Sky) {
        pass.set_pipeline(&self.march_pipeline);
        pass.set_bind_group(1, sky.bind_group(), &[]);
        pass.set_bind_group(2, sky.sun_tables_bind_group(), &[]);
        pass.set_bind_group(3, &self.march_group, &[]);
        pass.dispatch_workgroups(
            self.size.x.div_ceil(MARCH_GROUP),
            self.size.y.div_ceil(MARCH_GROUP),
            1,
        );
    }

    /// The two buffers the composite reads.
    pub fn views(&self) -> (&wgpu::TextureView, &wgpu::TextureView) {
        (&self.colour_view, &self.depth_view)
    }

    /// The textures themselves, for the tests that read them back.
    #[cfg(test)]
    pub fn buffers_for_test(
        &self,
    ) -> (
        &wgpu::Texture,
        &wgpu::Texture,
        &wgpu::Texture,
        &wgpu::Texture,
    ) {
        (&self.colour, &self.depth, &self.ceiling, &self.sun_light)
    }

    /// The half-resolution size the buffers were last built at.
    #[cfg(test)]
    pub fn size(&self) -> glam::UVec2 {
        self.size
    }
}

/// The half-resolution size a viewport marches at.
///
/// Rounded up, so a viewport with an odd side still has a half-resolution texel
/// standing over its last column rather than one short of it.
fn half_of(viewport: glam::UVec2) -> glam::UVec2 {
    ((viewport + glam::UVec2::ONE) / 2).max(glam::UVec2::ONE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One volume read back off the GPU, one byte per channel.
    struct Volume {
        size: u32,
        texels: Vec<[u8; 4]>,
    }

    impl Volume {
        fn at(&self, x: u32, y: u32, z: u32) -> [u8; 4] {
            let size = self.size;
            self.texels[((z * size + y) * size + x) as usize]
        }

        /// The same, wrapping, so a caller can ask for the texel one past the
        /// end and be given the one the tiling says is there.
        fn wrapped(&self, x: i32, y: i32, z: i32) -> [u8; 4] {
            let fold = |v: i32| (v.rem_euclid(self.size as i32)) as u32;
            self.at(fold(x), fold(y), fold(z))
        }
    }

    fn read_volume(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        size: u32,
    ) -> Volume {
        // A texture-to-buffer copy wants its rows on a 256-byte stride. The
        // shape's rows are 512 bytes and need no padding; the detail's are 128
        // and do, so the slack is dropped on the way out -- the same
        // arrangement `crate::headless::capture` makes for a narrow frame.
        let packed = size * 4;
        let bytes_per_row = packed.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud readback"),
            size: u64::from(bytes_per_row * size * size),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: size,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.get_mapped_range(..).unwrap();

        let mut texels = Vec::with_capacity((size * size * size) as usize);
        for row in 0..size * size {
            let start = (row * bytes_per_row) as usize;
            for x in 0..size as usize {
                let at = start + x * 4;
                texels.push([mapped[at], mapped[at + 1], mapped[at + 2], mapped[at + 3]]);
            }
        }
        drop(mapped);
        readback.unmap();
        Volume { size, texels }
    }

    fn built() -> (Volume, Volume) {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let mut cloud = Cloud::new(&device);
        cloud.ensure_built(&device, &queue);
        (
            read_volume(&device, &queue, &cloud.shape, SHAPE_SIZE),
            read_volume(&device, &queue, &cloud.detail, DETAIL_SIZE),
        )
    }

    /// The largest step between neighbouring texels along one axis, and the
    /// largest step across the seam where the volume meets itself.
    ///
    /// Both measured over the same channel and the same volume, so they are
    /// directly comparable: if the field tiles, the seam is not a special place
    /// and its steps sit inside the range the interior's do.
    fn steps(volume: &Volume, channel: usize, axis: usize) -> (u32, u32) {
        let mut interior = 0;
        let mut seam = 0;
        let size = volume.size;
        for a in 0..size {
            for b in 0..size {
                for c in 0..size {
                    let at = |offset: i32| {
                        let mut p = [c as i32, a as i32, b as i32];
                        p[axis] += offset;
                        i32::from(volume.wrapped(p[0], p[1], p[2])[channel])
                    };
                    let step = (at(1) - at(0)).unsigned_abs();
                    // The step out of the last texel is the one that crosses
                    // the seam; every other is interior.
                    if c == size - 1 {
                        seam = seam.max(step);
                    } else {
                        interior = interior.max(step);
                    }
                }
            }
        }
        (interior, seam)
    }

    /// The volumes meet themselves without a seam, in all three axes.
    ///
    /// This is the property the whole build is arranged around, and it is
    /// invisible in a frame: a discontinuity in a cloud looks like a cloud. So
    /// it is measured rather than looked at -- the largest jump across the wrap
    /// must sit inside the largest jump the field makes anywhere else. A
    /// lattice that failed to fold would put a fresh, uncorrelated set of
    /// gradients on the far side of the boundary and the seam step would run to
    /// most of the range.
    #[test]
    fn the_noise_volumes_tile_seamlessly_in_every_axis() {
        let (shape, detail) = built();
        for (name, volume) in [("shape", &shape), ("detail", &detail)] {
            for channel in 0..4 {
                for axis in 0..3 {
                    let (interior, seam) = steps(volume, channel, axis);
                    assert!(
                        seam <= interior,
                        "{name} channel {channel} axis {axis}: the seam steps by \
                         {seam} where the interior never steps by more than {interior}"
                    );
                }
            }
        }
    }

    /// Every channel spreads across the range it is stored in.
    ///
    /// Two things at once. It says the dispatch ran -- an unwritten storage
    /// texture reads as zeroes, which would tile perfectly and pass the test
    /// above without a murmur. And it says the field is worth eight bits:
    /// what is stored here is thresholded to decide where cloud is, so a
    /// channel crowded into a narrow band spends its whole signal on a few of
    /// the 255 levels it has, and a cloud edge drawn through a few levels
    /// bands.
    ///
    /// The bound is on the standard deviation rather than on the extremes,
    /// because extremes are what a handful of outlying texels can supply on
    /// their own. It is set at 0.10, against a measured 0.15 for the shape's
    /// Perlin-Worley and 0.17 to 0.18 for every single-frequency Worley
    /// channel -- and against 0.078 for the first version of `perlin_worley`,
    /// which lerped towards solid the way Schneider's own does and which this
    /// bound is set to reject. The one channel that is deliberately narrower
    /// is the detail's fractal, at 0.12, because averaging three octaves is
    /// what it is for.
    #[test]
    fn every_channel_of_the_noise_uses_the_range_it_is_stored_in() {
        let (shape, detail) = built();
        for (name, volume) in [("shape", &shape), ("detail", &detail)] {
            for channel in 0..4 {
                let count = volume.texels.len() as f64;
                let value = |texel: &[u8; 4]| f64::from(texel[channel]) / 255.0;
                let mean = volume.texels.iter().map(value).sum::<f64>() / count;
                let spread = (volume
                    .texels
                    .iter()
                    .map(|t| (value(t) - mean).powi(2))
                    .sum::<f64>()
                    / count)
                    .sqrt();
                assert!(
                    spread > 0.10,
                    "{name} channel {channel} has a spread of {spread:.3}, which is \
                     too narrow for eight bits to hold without banding"
                );
                // And it sits in the middle of the range rather than against an
                // end, so a threshold sweeping the field meets it gradually
                // rather than all at once.
                assert!(
                    (0.3..0.7).contains(&mean),
                    "{name} channel {channel} averages {mean:.3}, which is off centre"
                );
            }
        }
    }

    /// Writes a slice of each volume out to look at.
    ///
    /// Ignored because it asserts nothing: it is the check the tests above
    /// cannot make. They say the field tiles and that it uses its range, and a
    /// field that did both and still looked like static rather than like cloud
    /// would pass them all. Run it with
    /// `cargo test --release -- --ignored dump_noise --nocapture` and open the
    /// PNGs it names.
    ///
    /// Two slices of the shape a quarter of the volume apart, so the tiling can
    /// be judged as well: the right edge of each is what meets its own left.
    #[test]
    #[ignore = "writes an image to look at rather than asserting anything"]
    fn dump_noise() {
        let (shape, detail) = built();
        let out = std::env::temp_dir();

        let write = |name: &str, volume: &Volume, slice: u32| {
            let size = volume.size;
            // One channel per row of the image, so all four are seen at once:
            // the shape's Perlin-Worley over its three Worley octaves, or the
            // detail's three frequencies over their fractal.
            let mut pixels = vec![0u8; (size * size * 4 * 4) as usize];
            for channel in 0..4usize {
                for y in 0..size {
                    for x in 0..size {
                        let value = volume.at(x, y, slice)[channel];
                        let row = channel as u32 * size + y;
                        let at = ((row * size + x) * 4) as usize;
                        pixels[at] = value;
                        pixels[at + 1] = value;
                        pixels[at + 2] = value;
                        pixels[at + 3] = 255;
                    }
                }
            }
            let path = out.join(name);
            crate::headless::write_png(&path, glam::UVec2::new(size, size * 4), &pixels)
                .expect("failed to write");
            eprintln!("wrote {}", path.display());
        };

        for (name, volume) in [("shape", &shape), ("detail", &detail)] {
            for channel in 0..4usize {
                let values: Vec<f64> = volume
                    .texels
                    .iter()
                    .map(|t| f64::from(t[channel]) / 255.0)
                    .collect();
                let n = values.len() as f64;
                let mean = values.iter().sum::<f64>() / n;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let at = |q: f64| sorted[((n - 1.0) * q) as usize];
                eprintln!(
                    "{name}[{channel}] mean={mean:.3} sd={:.3} p05={:.3} p50={:.3} \
                     p95={:.3} max={:.3}",
                    variance.sqrt(),
                    at(0.05),
                    at(0.50),
                    at(0.95),
                    at(1.00)
                );
            }
        }

        write("cloud-shape-0.png", &shape, 0);
        write("cloud-shape-32.png", &shape, SHAPE_SIZE / 4);
        write("cloud-detail-0.png", &detail, 0);
    }

    /// The weather map for one preset at one moment, read back per deck.
    fn forecast(preset: Preset, elapsed: std::time::Duration) -> Vec<Volume> {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let mut cloud = Cloud::new(&device);
        cloud.ensure_built(&device, &queue);
        // The weather does not depend on where the eye is or where the sun is;
        // only the light volumes do, and this reads none of them.
        cloud.set_frame(
            &queue,
            preset,
            elapsed,
            glam::Vec3::ZERO,
            crate::sky::Sun::default().direction,
        );

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            cloud.draw_weather(&mut pass);
        }
        queue.submit(std::iter::once(encoder.finish()));

        (0..DECKS)
            .map(|deck| read_layer(&device, &queue, &cloud.weather, deck as u32))
            .collect()
    }

    /// One layer of the weather array, as a square of texels.
    fn read_layer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        layer: u32,
    ) -> Volume {
        let size = WEATHER_SIZE;
        // 512 texels of four bytes is 2048, already a multiple of the 256-byte
        // copy alignment, so there is no padding to drop.
        let bytes_per_row = size * 4;
        assert_eq!(bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weather readback"),
            size: u64::from(bytes_per_row * size),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
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
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.get_mapped_range(..).unwrap();
        let texels = mapped
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        drop(mapped);
        readback.unmap();
        Volume { size, texels }
    }

    /// The share of the sky one deck covers, from nothing to all of it.
    fn cover(deck: &Volume) -> f64 {
        deck.texels
            .iter()
            .map(|texel| f64::from(texel[0]) / 255.0)
            .sum::<f64>()
            / deck.texels.len() as f64
    }

    /// A clear sky has no cloud in it anywhere, at any height.
    ///
    /// Exactly none, not nearly none. It is the setting every other test in the
    /// tree will reach for to say "and this changes nothing", so a stray texel
    /// of cover would turn up later as an unexplained difference in a frame
    /// that was supposed to be untouched. The ramp a clear preset asks for sits
    /// above anything the field can reach, so the subtraction is negative
    /// everywhere and `clamp` makes it a hard zero.
    #[test]
    fn a_clear_sky_has_no_cloud_at_any_height() {
        for deck in forecast(Preset::Clear, std::time::Duration::ZERO) {
            let worst = deck.texels.iter().map(|t| t[0]).max().unwrap_or(0);
            assert_eq!(worst, 0, "a clear sky covered part of a deck");
        }
    }

    /// The presets are an order, and the sky fills as they go.
    ///
    /// Each name is a claim about how much sky is covered, and the numbers that
    /// back the claim are four per deck in a table that is easy to edit and
    /// easy to get subtly wrong. This is what says the table still means what
    /// the names say -- measured over the low deck, which is the one every
    /// preset is mostly about.
    #[test]
    fn each_preset_covers_more_sky_than_the_one_before_it() {
        let order = [
            Preset::Clear,
            Preset::Fair,
            Preset::Broken,
            Preset::Overcast,
            Preset::Storm,
        ];
        let covers: Vec<f64> = order
            .iter()
            .map(|preset| cover(&forecast(*preset, std::time::Duration::ZERO)[0]))
            .collect();
        for pair in order.iter().zip(&covers).collect::<Vec<_>>().windows(2) {
            let [(before, less), (after, more)] = pair else {
                unreachable!()
            };
            assert!(
                more > less,
                "{after:?} covers {more:.3} of the sky, which is no more than {before:?} at {less:.3}"
            );
        }
        // And the ends mean what they say: nothing at all, and very nearly
        // everything. Without this the order could hold across five presets
        // that were all the same drizzle.
        assert_eq!(covers[0], 0.0);
        assert!(covers[1] < 0.35, "fair weather covered {:.3}", covers[1]);
        assert!(
            covers[4] > 0.85,
            "a storm covered only {:.3} of the sky",
            covers[4]
        );
    }

    /// The weather moves, and moves slowly.
    ///
    /// Both halves matter. A field that did not move at all would be a static
    /// sky dressed up as a changing one; a field that moved fast would flicker
    /// between frames rather than evolve over minutes. So: a frame apart is
    /// almost the same sky, and a third of the period apart is a different one.
    #[test]
    fn the_weather_evolves_over_minutes_rather_than_frames() {
        let at =
            |seconds: f32| forecast(Preset::Broken, std::time::Duration::from_secs_f32(seconds));
        let difference = |a: &Volume, b: &Volume| {
            a.texels
                .iter()
                .zip(&b.texels)
                .map(|(x, y)| (f64::from(x[0]) - f64::from(y[0])).abs() / 255.0)
                .sum::<f64>()
                / a.texels.len() as f64
        };

        let start = at(0.0);
        let frame_later = at(1.0 / 60.0);
        let much_later = at(WEATHER_PERIOD / 3.0);

        let flicker = difference(&start[0], &frame_later[0]);
        let drift = difference(&start[0], &much_later[0]);
        assert!(
            flicker < 0.002,
            "the sky changed by {flicker:.4} in a single frame"
        );
        assert!(
            drift > 0.05,
            "the sky changed by only {drift:.4} in {:.0} seconds",
            WEATHER_PERIOD / 3.0
        );
    }

    /// Writes each preset's low deck out to look at.
    ///
    /// Ignored, like `dump_noise`, and for the same reason: the tests say the
    /// presets are an order and that a clear sky is clear, and a table that
    /// satisfied both could still produce cover shaped like nothing in the sky.
    /// Run with `cargo test --release -- --ignored dump_weather --nocapture`.
    ///
    /// Cover on the left, lean in the middle, base on the right, so a preset
    /// can be judged on all three at once.
    #[test]
    #[ignore = "writes an image to look at rather than asserting anything"]
    fn dump_weather() {
        let out = std::env::temp_dir();
        for preset in [
            Preset::Clear,
            Preset::Fair,
            Preset::Broken,
            Preset::Overcast,
            Preset::Storm,
        ] {
            let decks = forecast(preset, std::time::Duration::ZERO);
            let size = WEATHER_SIZE;
            let wide = size * 3;
            let mut pixels = vec![0u8; (wide * size * DECKS as u32 * 4) as usize];
            for (index, deck) in decks.iter().enumerate() {
                for y in 0..size {
                    for x in 0..size {
                        let texel = deck.at(x, y, 0);
                        // Cover, lean, base -- density is the one channel that
                        // says nothing about the shape of the sky.
                        for (column, channel) in [texel[0], texel[1], texel[3]].iter().enumerate() {
                            let row = index as u32 * size + y;
                            let at = ((row * wide + column as u32 * size + x) * 4) as usize;
                            pixels[at] = *channel;
                            pixels[at + 1] = *channel;
                            pixels[at + 2] = *channel;
                            pixels[at + 3] = 255;
                        }
                    }
                }
            }
            let path = out.join(format!("weather-{preset:?}.png").to_lowercase());
            crate::headless::write_png(&path, glam::UVec2::new(wide, size * DECKS as u32), &pixels)
                .expect("failed to write");
            eprintln!("wrote {} ({:.3} cover)", path.display(), cover(&decks[0]));
        }
    }

    /// One frame of the march, and the cache it was walked over.
    struct Marched {
        /// Half-resolution scattered radiance and transmittance.
        colour: Vec<[f32; 4]>,
        /// Where along each of those rays the cloud was, in metres.
        depth: Vec<f32>,
        size: glam::UVec2,
        /// The ceiling cache, indexed `(x, slice, z)`.
        ceiling: Vec<f32>,
        /// How much of the sun reaches each texel of the light volume, indexed
        /// `(x, z, slice)`.
        sun_light: Vec<f32>,
        /// The weather the cache was built from, one layer per deck.
        weather: Vec<Volume>,
    }

    impl Marched {
        fn cell(&self, x: u32, slice: u32, z: u32) -> f32 {
            let across = CEILING_ACROSS;
            self.ceiling[((z * CEILING_SLICES + slice) * across + x) as usize]
        }

        /// How much of the sun reaches one texel of the light volume.
        fn lit(&self, x: u32, z: u32, slice: u32) -> f32 {
            self.sun_light[((slice * LIGHT_ACROSS + z) * LIGHT_ACROSS + x) as usize]
        }

        /// The weather over a world point, sampled the way the march samples it.
        ///
        /// Bilinear over a map that wraps, with a texel standing for its own
        /// middle: the Rust twin of the `textureSampleLevel` the shader makes,
        /// which is what the oracle below needs and what a `textureLoad` would
        /// not give.
        fn forecast_at(&self, deck: usize, x: f32, z: f32) -> [f32; 4] {
            let size = WEATHER_SIZE as f32;
            let at = glam::Vec2::new(x, z) / WEATHER_TILE * size - 0.5;
            let corner = at.floor();
            let f = at - corner;
            let layer = &self.weather[deck];
            let tap = |dx: i32, dy: i32| {
                let texel = layer.wrapped(corner.x as i32 + dx, corner.y as i32 + dy, 0);
                std::array::from_fn::<f32, 4, _>(|c| f32::from(texel[c]) / 255.0)
            };
            let mix = |a: [f32; 4], b: [f32; 4], t: f32| {
                std::array::from_fn::<f32, 4, _>(|c| a[c] + (b[c] - a[c]) * t)
            };
            mix(
                mix(tap(0, 0), tap(1, 0), f.x),
                mix(tap(0, 1), tap(1, 1), f.x),
                f.y,
            )
        }

        /// How much of the frame the cloud covers, from nothing to all of it.
        fn opacity(&self) -> f64 {
            self.colour
                .iter()
                .map(|texel| 1.0 - f64::from(texel[3]))
                .sum::<f64>()
                / self.colour.len() as f64
        }
    }

    /// Marches one frame of one preset, from a camera of the caller's choosing.
    ///
    /// No terrain: the G-buffer is left as it was made, which is zeroes, and
    /// zero depth is exactly what the march reads as "this ray found no ground".
    /// So every ray runs its full length, which is the case worth measuring and
    /// the only one where what the march did can be read off the buffer without
    /// a mountain in the way. What the depth clip does with real ground is a
    /// question for a real frame; `src/scene.rs` asks it there.
    fn marched(preset: Preset, camera: &crate::camera::Camera, size: glam::UVec2) -> Marched {
        marched_under(preset, camera, size, crate::sky::Sun::default())
    }

    /// The same, under a sun of the caller's choosing.
    fn marched_under(
        preset: Preset,
        camera: &crate::camera::Camera,
        size: glam::UVec2,
        sun: crate::sky::Sun,
    ) -> Marched {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let (camera_layout, camera_group) = crate::scene::test_camera(&device, &queue, camera);

        let mut sky = crate::sky::Sky::new(&device, &camera_layout);
        sky.ensure_built(&device, &queue);
        sky.set_frame(
            &queue,
            sun,
            camera.position,
            crate::sky::pixel_angle(camera.fov_y, size.y),
        );

        let mut cloud = Cloud::new(&device);
        cloud.ensure_built(&device, &queue);
        cloud.set_frame(
            &queue,
            preset,
            std::time::Duration::ZERO,
            camera.position,
            sun.direction,
        );

        let gbuffer = crate::deferred::GBuffer::new(&device, size);
        let march = March::new(
            &device,
            &camera_layout,
            sky.layout(),
            sky.sun_tables_layout(),
            &cloud,
            &gbuffer,
        );

        let mut encoder = device.create_command_encoder(&Default::default());
        // Three passes, as `Scene::draw` records them, and for the reason it
        // does: each reads what the one before it wrote.
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            cloud.draw_weather(&mut pass);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            march.draw_ceiling(&mut pass);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            march.draw_light(&mut pass);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &camera_group, &[]);
            march.draw(&mut pass, &sky);
        }
        queue.submit(std::iter::once(encoder.finish()));

        let (colour, depth, ceiling, _) = march.buffers_for_test();
        let half = march.size();
        Marched {
            colour: read_texels(&device, &queue, colour, half.x, half.y, 1, 8)
                .chunks_exact(8)
                .map(|texel| {
                    std::array::from_fn(|c| {
                        half::f16::from_le_bytes([texel[c * 2], texel[c * 2 + 1]]).to_f32()
                    })
                })
                .collect(),
            depth: read_texels(&device, &queue, depth, half.x, half.y, 1, 4)
                .chunks_exact(4)
                .map(|texel| f32::from_le_bytes(texel.try_into().unwrap()))
                .collect(),
            size: half,
            ceiling: read_texels(
                &device,
                &queue,
                ceiling,
                CEILING_ACROSS,
                CEILING_SLICES,
                CEILING_ACROSS,
                4,
            )
            .chunks_exact(4)
            .map(|texel| f32::from_le_bytes(texel.try_into().unwrap()))
            .collect(),
            sun_light: read_texels(
                &device,
                &queue,
                march.buffers_for_test().3,
                LIGHT_ACROSS,
                LIGHT_ACROSS,
                LIGHT_SLICES,
                8,
            )
            .chunks_exact(8)
            .map(|texel| half::f16::from_le_bytes([texel[0], texel[1]]).to_f32())
            .collect(),
            // Read back from this very run rather than rebuilt beside it: the
            // field is deterministic, so a second build would give the same
            // bytes -- but then the oracle would be checking the cache against a
            // forecast it was not made from, and the day that stopped being true
            // the test would be measuring nothing.
            weather: (0..DECKS)
                .map(|deck| read_layer(&device, &queue, &cloud.weather, deck as u32))
                .collect(),
        }
    }

    /// A texture read back into tightly packed bytes, whatever its row stride.
    ///
    /// The copy wants rows on a 256-byte stride and none of these buffers has
    /// one naturally, so the slack is dropped on the way out -- the same
    /// arrangement [`read_volume`] makes and for the same reason.
    fn read_texels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        width: u32,
        height: u32,
        depth: u32,
        stride: u32,
    ) -> Vec<u8> {
        let packed = width * stride;
        let bytes_per_row = packed.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud march readback"),
            size: u64::from(bytes_per_row * height * depth),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: depth,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.get_mapped_range(..).unwrap();
        let mut out = Vec::with_capacity((packed * height * depth) as usize);
        for row in 0..height * depth {
            let start = (row * bytes_per_row) as usize;
            out.extend_from_slice(&mapped[start..start + packed as usize]);
        }
        drop(mapped);
        readback.unmap();
        out
    }

    /// Three whole weather tiles from the origin, and negative.
    ///
    /// Every camera below flies from here rather than from nothing, so that
    /// every ray they cast is at world coordinates the fields have to be folded
    /// to reach. Both folds are on that path -- the cache's cell index and the
    /// repeating sampler the three noise fields are read through -- and either
    /// one missing puts the whole march outside its data. At the origin that is
    /// invisible: an unfolded index into the first tile is the right index.
    const AWAY: f32 = -3.0 * WEATHER_TILE;

    /// A camera out at [`AWAY`], at a height and a pitch.
    fn looking(height: f32, pitch_degrees: f32) -> crate::camera::Camera {
        crate::camera::Camera::new(
            glam::Vec3::new(AWAY, height, AWAY),
            crate::camera::Camera::from_yaw_pitch_roll(0.0, pitch_degrees.to_radians(), 0.0),
            1.0,
        )
    }

    /// A clear sky marches to no cloud at any pixel, and marches every pixel.
    ///
    /// Two claims in one readback, and the second is why the first is worth
    /// making at all. Transmittance one and no scattered light is what a ray
    /// that met nothing leaves behind -- and a texel the dispatch never reached
    /// reads as zero, which is *opaque black cloud*. So the exact ones here say
    /// the march covered the buffer as well as that it found nothing in it.
    ///
    /// This is the setting every later test reaches for to say "and this changes
    /// nothing", so it has to be exact rather than nearly so.
    #[test]
    fn a_clear_sky_marches_to_nothing_at_every_pixel() {
        let frame = marched(
            Preset::Clear,
            &looking(1500.0, 0.0),
            glam::UVec2::splat(128),
        );
        assert_eq!(frame.colour.len(), (frame.size.x * frame.size.y) as usize);
        for (index, texel) in frame.colour.iter().enumerate() {
            assert_eq!(
                *texel,
                [0.0, 0.0, 0.0, 1.0],
                "texel {index} of a clear sky reads {texel:?}"
            );
        }
        // ... and the cache agrees, which is the other half of "clear is clear":
        // a cell that bounded something would have the march sampling inside it.
        let worst = frame.ceiling.iter().copied().fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "a clear sky bounded {worst} of extinction");
    }

    /// A sky with cloud in it draws cloud, at a distance the ray reached.
    ///
    /// The counterpart to the test above, and the one that says the march does
    /// anything at all: every assertion there is satisfied by a dispatch that
    /// returns immediately. A level view under a broken sky should be most cloud
    /// and some gap, and where there is cloud the recorded distance has to be a
    /// distance the ray actually walked -- in front of the eye and no further
    /// than the march is allowed to look.
    #[test]
    fn a_broken_sky_draws_cloud_at_the_distance_it_was_found() {
        let frame = marched(
            Preset::Broken,
            &looking(1500.0, 0.0),
            glam::UVec2::splat(128),
        );
        let opacity = frame.opacity();
        assert!(
            (0.2..0.98).contains(&opacity),
            "a broken sky came out {opacity:.3} opaque, which is not broken"
        );
        for (index, texel) in frame.colour.iter().enumerate() {
            assert!(
                texel.iter().all(|v| v.is_finite()),
                "texel {index} holds {texel:?}"
            );
            assert!(
                (0.0..=1.0).contains(&texel[3]),
                "texel {index} has transmittance {}",
                texel[3]
            );
            // Cloud is lit by a sun and by a sky, so anything it stopped it also
            // put something back. Scattering nothing while stopping light is
            // what a bug in the source term looks like.
            if texel[3] < 0.5 {
                assert!(
                    texel[0] > 0.0 && texel[1] > 0.0 && texel[2] > 0.0,
                    "texel {index} stopped light and scattered {texel:?}"
                );
                let at = frame.depth[index];
                assert!(
                    at > 0.0 && at <= 100_000.0,
                    "texel {index} put its cloud {at} m away"
                );
            }
        }
    }

    /// How much of a deck's slab is filled, at a height fraction through it.
    ///
    /// The Rust twin of `vertical` in `src/cloud_march.wgsl`. A second copy of
    /// the arithmetic rather than a text comparison of it, because what is being
    /// checked here is not that the two agree -- it is that the *cache* bounds
    /// what the march would find, and an oracle that cannot be wrong in the same
    /// way the shader is wrong is the whole point of writing one.
    fn vertical(h: f32, lean: f32) -> f32 {
        let smoothstep = |edge0: f32, edge1: f32, x: f32| {
            let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let mix = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let rise = mix(0.10, 0.35, lean);
        let fall = mix(0.88, 0.60, lean);
        smoothstep(0.0, rise, h) * (1.0 - smoothstep(fall, 1.0, h))
    }

    /// What the march would find at a point, at its very largest.
    ///
    /// The extinction the march computes is this with the shape field in place
    /// of the one the carve is evaluated at, and the field cannot exceed one --
    /// so this is the number the cell containing the point has to bound. Must
    /// stay in step with `cloud_extinction` and `cell_bound` in
    /// `src/cloud_march.wgsl`; `EXTINCTION` and `EDGE` are read out of the
    /// shader rather than restated, so a change to either fails the test that
    /// compares them rather than quietly moving both sides at once.
    fn most_at(frame: &Marched, p: glam::Vec3) -> f32 {
        let extinction = shader_constant("EXTINCTION");
        let edge = shader_constant("EDGE");
        let mut most: f32 = 0.0;
        for (deck, slab) in DECK_SLABS.iter().enumerate() {
            if p.y < slab[0] || p.y > slab[1] + slab[2] {
                continue;
            }
            let w = frame.forecast_at(deck, p.x, p.z);
            let base = slab[0] + w[3] * slab[2];
            let coverage = w[0] * vertical((p.y - base) / (slab[1] - slab[0]), w[1]);
            most = most.max(extinction * w[2] * slab[3] * (coverage / edge).clamp(0.0, 1.0));
        }
        most
    }

    /// What the march shader gives as the value of a named constant.
    fn shader_says(name: &str) -> String {
        let source = include_str!("cloud_march.wgsl");
        let declaration = format!("const {name}: ");
        let line = source
            .lines()
            .find(|line| line.starts_with(&declaration))
            .unwrap_or_else(|| panic!("src/cloud_march.wgsl declares no {name}"));
        let (_, value) = line
            .split_once(" = ")
            .unwrap_or_else(|| panic!("src/cloud_march.wgsl says {line:?}"));
        value.trim_end_matches(';').to_owned()
    }

    /// The same, as the number it is.
    fn shader_constant(name: &str) -> f32 {
        let value = shader_says(name);
        value
            .parse()
            .unwrap_or_else(|_| panic!("src/cloud_march.wgsl says {name} is {value:?}"))
    }

    /// No cell of the cache ever claims less cloud than is inside it.
    ///
    /// The one property the empty-space skipping rests on, and the one that
    /// fails invisibly: a cell whose bound is too low is a cell the march walks
    /// straight through, and what that leaves is a bite out of a cloud that
    /// looks like a gap in the cloud. There is no frame in which it reads as a
    /// bug.
    ///
    /// So it is measured against an oracle instead. A grid of world points is
    /// walked -- deliberately not on the cache's own lattice, because the
    /// interesting points are the ones near a cell's corners, where the march's
    /// bilinear reach into the neighbouring weather texels is what the build's
    /// margin has to have covered. Points are taken well outside the tile too,
    /// where the fold is what has to be right.
    #[test]
    fn no_cell_of_the_ceiling_claims_less_cloud_than_it_holds() {
        let frame = marched(
            Preset::Broken,
            &looking(1500.0, 0.0),
            glam::UVec2::splat(32),
        );
        // A stride that shares no factor with the cell size, so the points walk
        // across cells rather than landing at the same place in each.
        let step = 61.0;
        let mut worst: f32 = 0.0;
        let mut checked = 0u32;
        for iz in -40..40 {
            for ix in -40..40 {
                for slice in 0..CEILING_SLICES {
                    let p = glam::Vec3::new(
                        AWAY + ix as f32 * step,
                        (f32::from(slice as u16) + 0.37) * (CEILING_TOP / CEILING_SLICES as f32),
                        AWAY + iz as f32 * step,
                    );
                    let across = CEILING_ACROSS as i32;
                    let cell = |v: f32| {
                        let raw = (v / (WEATHER_TILE / CEILING_ACROSS as f32)).floor() as i32;
                        ((raw % across) + across) % across
                    };
                    let bound = frame.cell(cell(p.x) as u32, slice, cell(p.z) as u32);
                    let most = most_at(&frame, p);
                    worst = worst.max(most - bound);
                    checked += 1;
                }
            }
        }
        // The tolerance is for the two bilinears, not for the bound: the shader
        // filters an eight-bit map in hardware and the oracle does it in `f32`,
        // and the two disagree by about a seventh of a texel's last bit. That
        // measures 1.1e-4 here. Dropping the margin measures 2.4e-3, so the
        // bound below sits between them with four times' room on each side.
        assert!(
            worst < 5e-4,
            "the cache fell {worst:.5} short of the cloud in it, over {checked} points"
        );
    }

    /// The light volume gets darker the further down a deck it goes.
    ///
    /// The defining property of the walk, and the one an off-by-one in it
    /// destroys: a column is filled from the top down, so every texel holds the
    /// cloud above it and no texel may be brighter than the one over its head.
    /// Walking the other way instead -- which is one character -- would fill a
    /// deck's *top* with its own shadow and leave its base in full sun, and the
    /// image that produces is a cloud lit from below, which reads as strange
    /// rather than as wrong.
    ///
    /// Measured over an overcast sky, which is the one that fills the volume:
    /// the highest slice keeps almost all of the sun and the lowest almost none.
    #[test]
    fn the_light_volume_darkens_downwards_through_a_deck() {
        let frame = marched(
            Preset::Overcast,
            &looking(1500.0, 0.0),
            glam::UVec2::splat(32),
        );
        let mut worst: f32 = 0.0;
        let mut darkest: f32 = 1.0;
        let mut brightest: f32 = 0.0;
        for z in 0..LIGHT_ACROSS {
            for x in 0..LIGHT_ACROSS {
                brightest = brightest.max(frame.lit(x, z, LIGHT_SLICES - 1));
                darkest = darkest.min(frame.lit(x, z, 0));
                for slice in 0..LIGHT_SLICES - 1 {
                    // Above may never be darker than below.
                    worst = worst.max(frame.lit(x, z, slice) - frame.lit(x, z, slice + 1));
                }
            }
        }
        // Half floats, so two texels that should be equal can differ by a
        // thousandth of what they hold.
        assert!(
            worst < 1e-3,
            "a texel of the light volume is {worst} brighter than the one above it"
        );
        assert!(
            brightest > 0.99,
            "the top of the volume keeps only {brightest} of the sun, \
             with nothing above it to take any"
        );
        assert!(
            darkest < 0.01,
            "the bottom of an overcast deck still keeps {darkest} of the sun"
        );
    }

    /// A sun below the horizon puts no direct light in the clouds.
    ///
    /// Not because the volume says so -- it says nothing about where the sun is,
    /// only about what stands between here and there -- but because the planet
    /// itself is in the way, which is a test the march has to make and which the
    /// transmittance table will not make for it. Its parameterisation happily
    /// returns a number for a ray that would have to come up through the ground.
    ///
    /// Made at each sample's own altitude rather than the eye's, which is what
    /// leaves the highest deck lit after the lowest has gone dark. That is the
    /// last light of the day and it is worth having.
    #[test]
    fn a_sun_below_the_horizon_puts_no_direct_light_in_the_clouds() {
        let camera = looking(1500.0, 0.0);
        let size = glam::UVec2::splat(64);
        let total = |sun: crate::sky::Sun| {
            let frame = marched_under(Preset::Broken, &camera, size, sun);
            let sum: f64 = frame
                .colour
                .iter()
                .map(|texel| {
                    assert!(
                        texel.iter().all(|v| v.is_finite()),
                        "{texel:?} is not finite"
                    );
                    f64::from(texel[1])
                })
                .sum();
            sum / frame.colour.len() as f64
        };
        // Down to the horizon and past it, and never back up: the light a cloud
        // takes from the sun has to fall away as the sun sets rather than
        // reappear from some part of the parameterisation that was never meant
        // to be asked. Measured: 0.0018 at two degrees up, 0.00023 level,
        // 5e-6 at four down, and nothing at all at eight.
        let mut before = f64::INFINITY;
        for elevation in [30.0f32, 2.0, 0.0, -2.0, -4.0, -8.0, -20.0] {
            let lit = total(crate::sky::Sun::from_angles(elevation, 140.0));
            assert!(
                lit <= before,
                "the sun at {elevation} degrees puts {lit:.6} into the cloud, \
                 more than the {before:.6} it put in from higher up"
            );
            before = lit;
        }
        assert_eq!(before, 0.0, "a sun twenty degrees down still lit the cloud");
        let day = total(crate::sky::Sun::from_angles(30.0, 140.0));
        assert!(day > 1e-3, "daylight only put {day:.6} into the cloud");
    }

    /// The light volume stands still while the camera moves across it.
    ///
    /// It is placed on the camera, so that its resolution is spent where the
    /// eye is; it is snapped to a whole texel, so that being placed on a moving
    /// thing does not make it a moving thing. Without the snap every cloud's
    /// shadow would land a fraction of a texel differently each frame, and a
    /// whole sky of shadows shifting under a walking camera is exactly the kind
    /// of crawl a still image cannot show and a flight cannot hide.
    #[test]
    fn the_light_volume_stands_still_while_the_camera_moves() {
        let across = LIGHT_EXTENT / LIGHT_ACROSS as f32;
        let sun = crate::sky::Sun::default().direction;
        let corner = |x: f32| {
            Preset::Fair
                .uniform(
                    std::time::Duration::ZERO,
                    glam::Vec3::new(x, 2000.0, 0.0),
                    sun,
                )
                .light_origin
        };
        // Two eyes a tenth of a texel apart put the volume in the same place...
        assert_eq!(corner(0.0), corner(across * 0.1));
        // ... and two a whole texel apart move it by exactly one texel, rather
        // than by a texel and a fraction.
        let moved = corner(across);
        assert_eq!(moved[0] - corner(0.0)[0], across);
        assert_eq!(moved[1], corner(0.0)[1]);
    }

    /// Writes what the march drew, and the cache it walked, out to look at.
    ///
    /// Ignored, like `dump_noise` and `dump_weather`, and for the reason those
    /// are: the tests above say the buffer is covered, that a clear sky is clear
    /// and that a broken one is neither -- and a field satisfying all three
    /// could still look like fog, or like static, or like nothing at all. Run
    /// with `cargo test --release -- --ignored dump_cloud --nocapture`.
    ///
    /// Transmittance on the left and the scattered light on the right, tonemapped
    /// the way the frame will shortly tonemap it, so what is written here is what
    /// the composite is about to put on the screen.
    #[test]
    #[ignore = "writes an image to look at rather than asserting anything"]
    fn dump_cloud() {
        let out = std::env::temp_dir();
        for (name, camera) in [
            ("level", looking(1500.0, 0.0)),
            ("down", looking(6000.0, -35.0)),
            ("up", looking(500.0, 25.0)),
        ] {
            for preset in [
                Preset::Fair,
                Preset::Broken,
                Preset::Overcast,
                Preset::Storm,
            ] {
                let frame = marched(preset, &camera, glam::UVec2::splat(512));
                let (w, h) = (frame.size.x, frame.size.y);
                let mut pixels = vec![0u8; (w * 2 * h * 4) as usize];
                for y in 0..h {
                    for x in 0..w {
                        let texel = frame.colour[(y * w + x) as usize];
                        let lit =
                            crate::sky::tonemap(glam::Vec3::new(texel[0], texel[1], texel[2]));
                        let alpha = ((1.0 - texel[3]) * 255.0) as u8;
                        let put = |pixels: &mut Vec<u8>, column: u32, rgb: [u8; 3]| {
                            let at = ((y * w * 2 + column) * 4) as usize;
                            pixels[at] = rgb[0];
                            pixels[at + 1] = rgb[1];
                            pixels[at + 2] = rgb[2];
                            pixels[at + 3] = 255;
                        };
                        put(&mut pixels, x, [alpha, alpha, alpha]);
                        put(
                            &mut pixels,
                            w + x,
                            [
                                (lit.x * 255.0) as u8,
                                (lit.y * 255.0) as u8,
                                (lit.z * 255.0) as u8,
                            ],
                        );
                    }
                }
                let path = out.join(format!("cloud-{name}-{preset:?}.png").to_lowercase());
                crate::headless::write_png(&path, glam::UVec2::new(w * 2, h), &pixels)
                    .expect("failed to write");
                // What the cloud is worth in radiance, against what the ground
                // under it is: a sunlit grass slope leaves about 0.07, and a
                // sunlit cloud top should be several times that.
                let mut lit: Vec<f32> = frame.colour.iter().map(|t| t[1]).collect();
                lit.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let at = |q: f64| lit[((lit.len() - 1) as f64 * q) as usize];
                eprintln!(
                    "wrote {} ({:.3} opaque, {:.0} m mean, green p50={:.3} \
                     p95={:.3} max={:.3})",
                    path.display(),
                    frame.opacity(),
                    frame.depth.iter().sum::<f32>() / frame.depth.len() as f32,
                    at(0.50),
                    at(0.95),
                    at(1.00),
                );
            }
        }
    }

    /// The two copies of the mixer are the same mixer.
    ///
    /// Nothing would break if they differed -- they seed different fields -- but
    /// a reader finding two spellings has to work out whether the difference is
    /// deliberate, and it is not. The same text comparison
    /// `both_shaders_map_the_sky_the_same_way` makes between the sky's two
    /// copies of its own arithmetic.
    #[test]
    fn both_shaders_mix_their_noise_the_same_way() {
        let body = |source: &str| {
            let start = source
                .find("fn noise_mix(bits: u32) -> u32 {")
                .expect("no noise_mix");
            let end = source[start..].find("\n}").expect("unterminated");
            source[start..start + end].to_owned()
        };
        assert_eq!(
            body(include_str!("cloud.wgsl")),
            body(include_str!("terrain.wgsl")),
            "src/cloud.wgsl and src/terrain.wgsl spell `noise_mix` differently"
        );
    }

    /// Rust and the shader agree about the grid the march walks.
    ///
    /// Every one of these is a number Rust uses to shape a texture or size a
    /// dispatch and the shader uses to address it. None of them would fail to
    /// compile if they disagreed: the cache would be addressed at the wrong
    /// scale, or two thirds of it would go unwritten, and the result would be a
    /// sky with holes in it that looked like a sky with gaps in it.
    #[test]
    fn the_shader_and_rust_agree_on_the_cloud_grid() {
        for (name, value) in [
            ("WEATHER_SIZE", WEATHER_SIZE),
            ("DECKS", DECKS as u32),
            ("CEILING_ACROSS", CEILING_ACROSS),
            ("CEILING_SLICES", CEILING_SLICES),
            ("LIGHT_ACROSS", LIGHT_ACROSS),
            ("LIGHT_SLICES", LIGHT_SLICES),
        ] {
            assert_eq!(shader_says(name), format!("{value}u"), "{name} differs");
        }
        for (name, value) in [("CEILING_TOP", CEILING_TOP), ("WEATHER_TILE", WEATHER_TILE)] {
            assert_eq!(shader_constant(name), value, "{name} differs");
        }
        // One cubic kernel -- the ceiling -- and three flat ones: the march and
        // the two light columns, which are one kernel written twice because
        // only their lean differs.
        let source = include_str!("cloud_march.wgsl");
        for (group, flat, count) in [(CEILING_GROUP, CEILING_GROUP, 1), (MARCH_GROUP, 1, 3)] {
            assert_eq!(
                source
                    .matches(&format!(
                        "@compute @workgroup_size({group}, {group}, {flat})"
                    ))
                    .count(),
                count,
                "src/cloud_march.wgsl has a kernel this module would dispatch wrongly"
            );
        }
    }

    /// The march reads the tables and rebuilds its rays the way the shading does.
    ///
    /// Seven functions copied into `src/cloud_march.wgsl` from
    /// `src/shading.wgsl`, because there is no preprocessor here and no way to
    /// share them. Two of them decide where a pixel's ray points, and a
    /// last-bit difference there would put the cloud on a slightly different ray
    /// from the ground it is drawn against -- still in a still frame, and
    /// crawling in a moving one. The other five address the scattering tables,
    /// where a difference is a cloud lit by a sun in a marginally different
    /// place from the one lighting the mountain under it.
    ///
    /// Whitespace is normalised and the bodies are compared: a comment may
    /// differ, since each copy says what its own caller wants, and the
    /// arithmetic may not.
    #[test]
    fn the_march_reads_the_sky_the_way_the_shading_does() {
        let body = |source: &str, name: &str| {
            let start = source
                .find(&format!("fn {name}("))
                .unwrap_or_else(|| panic!("no {name}"));
            let end = source[start..].find("\n}").expect("unterminated");
            source[start..start + end]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        };
        let shade = include_str!("shading.wgsl");
        let march = include_str!("cloud_march.wgsl");
        for name in [
            "ray_raw_at",
            "distance_at",
            "to_texture",
            "top_distance",
            "transmittance_uv",
            "sample_transmittance",
            "sample_multiscatter",
        ] {
            assert_eq!(
                body(shade, name),
                body(march, name),
                "{name} differs between src/shading.wgsl and src/cloud_march.wgsl"
            );
        }
        // ... and the constants those seven read.
        for name in [
            "GROUND_RADIUS",
            "TOP_RADIUS",
            "TRANSMITTANCE_WIDTH",
            "TRANSMITTANCE_HEIGHT",
            "MULTISCATTER_SIZE",
            "PI",
        ] {
            let declared = |source: &'static str| {
                source
                    .lines()
                    .find(|line| line.starts_with(&format!("const {name}: ")))
                    .unwrap_or_else(|| panic!("no {name}"))
            };
            assert_eq!(
                declared(shade),
                declared(march),
                "{name} differs between src/shading.wgsl and src/cloud_march.wgsl"
            );
        }
    }

    /// Both cloud shaders describe the uniform they share the same way.
    ///
    /// One buffer, written once by [`Preset::uniform`] and read by two shader
    /// modules that cannot include each other. A field added to one and not the
    /// other does not fail to compile -- it silently shifts every field after it
    /// in the module that is short, so the march would read a deck's density out
    /// of the bytes that hold its seed.
    #[test]
    fn both_cloud_shaders_describe_the_same_uniform() {
        let block = |source: &str, name: &str| {
            let start = source
                .find(&format!("struct {name} {{"))
                .unwrap_or_else(|| panic!("no struct {name}"));
            let end = source[start..].find("\n};").expect("unterminated");
            // Comments differ between the two on purpose -- each says what its
            // own reader does with the fields -- so only the declarations are
            // compared.
            source[start..start + end]
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
                .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let build = include_str!("cloud.wgsl");
        let march = include_str!("cloud_march.wgsl");
        for name in ["Deck", "Weather"] {
            assert_eq!(
                block(build, name),
                block(march, name),
                "{name} differs between src/cloud.wgsl and src/cloud_march.wgsl"
            );
        }
        // ... and Rust writes exactly what both of them expect to read.
        assert_eq!(
            std::mem::size_of::<WeatherUniform>(),
            DECKS * 3 * 16 + 4 * 16,
            "the uniform is not the three decks and four vectors both shaders read"
        );
    }

    /// No height belongs to two decks at once.
    ///
    /// What lets a sample in the march cost one weather fetch rather than three:
    /// it finds the deck a height is in and stops looking. Overlapping slabs
    /// would not fail -- the lower deck would simply win everywhere the two met,
    /// and the upper one would be missing its underside for reasons nothing
    /// records. The swing counts: a deck's base lifts and carries its top with
    /// it, so what it can occupy runs to its top plus its whole swing.
    #[test]
    fn the_decks_never_reach_into_one_another() {
        for pair in DECK_SLABS.windows(2) {
            let [below, above] = pair else { unreachable!() };
            assert!(
                below[1] + below[2] < above[0],
                "a deck reaching {} m sits under one starting at {} m",
                below[1] + below[2],
                above[0]
            );
        }
        // ... and each is a slab with a thickness, rather than a plane or an
        // inversion, which is what the height fraction divides by.
        for slab in DECK_SLABS {
            assert!(
                slab[1] > slab[0],
                "a deck runs from {} to {}",
                slab[0],
                slab[1]
            );
        }
    }

    /// Rust and the shader agree about how big the volumes are.
    ///
    /// The sizes decide the textures on this side and the bounds test and the
    /// texel centres on that one. A disagreement would not fail to compile: it
    /// would write a corner of the volume and leave the rest as zeroes, or
    /// address it half a texel out.
    #[test]
    fn the_shader_and_rust_agree_on_the_noise_volumes() {
        let source = include_str!("cloud.wgsl");
        for (name, value) in [("SHAPE_SIZE", SHAPE_SIZE), ("DETAIL_SIZE", DETAIL_SIZE)] {
            let declaration = format!("const {name}: u32");
            let line = source
                .lines()
                .find(|line| line.trim_start().starts_with(&declaration))
                .unwrap_or_else(|| panic!("src/cloud.wgsl declares no {name}"));
            assert!(
                line.contains(&format!("{value}u")),
                "src/cloud.wgsl says {line:?}, but src/cloud.rs says {name} is {value}"
            );
        }
        assert_eq!(
            source
                .matches(&format!(
                    "@compute @workgroup_size({GROUP}, {GROUP}, {GROUP})"
                ))
                .count(),
            2,
            "src/cloud.wgsl has kernels this module would dispatch wrongly"
        );
    }
}
