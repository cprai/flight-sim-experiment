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
pub const DECKS: usize = 4;

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
    [0.0, 1250.0, 120.0, 1.0],
    [700.0, 3000.0, 400.0, 1.0],
    [3500.0, 6000.0, 500.0, 0.65],
    [8500.0, 10000.0, 300.0, 0.25],
];

/// Which entry of [`DECK_SLABS`] is the low deck.
///
/// The one every preset but `fog` is mostly about, and the one the tests reach
/// for when they want to say what a name means. Named rather than written as a
/// number, because it stopped being the first the day the fog went under it.
#[cfg(test)]
const LOW_DECK: usize = 1;

/// How far each deck's base rides the ground rather than standing at an
/// altitude of its own.
///
/// One for the fog and nothing for the rest. A deck that hugs keeps the top
/// [`DECK_SLABS`] gives it and takes its base from the terrain under it, so it
/// *pools*: it fills every valley floor below its top and leaves every ridge
/// above it in clear air, which is what valley fog is and what a blanket laid
/// over the hills at a constant depth would not be.
///
/// The top staying absolute is what keeps the rest of the system still. A
/// hugging deck occupies exactly the heights [`DECK_SLABS`] says it does -- its
/// base only ever rises into that band -- so `deck_at` needs no terrain, the
/// ceiling cache's slab test needs no terrain, and nothing has to be told how
/// high the mountains are.
const DECK_HUG: [f32; DECKS] = [1.0, 0.0, 0.0, 0.0];

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

/// Which texel of a two-by-two block of the cloud buffer each frame marches.
///
/// The two-by-two Bayer matrix, read as an order rather than as a threshold:
/// `0 2 / 3 1`, so the sequence is a corner, the opposite corner, and then the
/// other two. Each frame's texel is as far as it can be from the last one's,
/// which is what makes a partly-filled buffer look like a coarse image rather
/// than like one half of a fine one -- the same reason the pattern is used for
/// dithering, applied to time instead of tone.
pub const ROTATION: [[u32; 2]; 4] = [[0, 0], [1, 1], [1, 0], [0, 1]];

/// Which quarter of the buffer this frame marches, and how big the buffer is.
///
/// Mirrors `Rotation` in `src/cloud_march.wgsl`. Its own buffer rather than two
/// more fields of anything already bound: the march and the resolve both read
/// it and nothing else does, and the size is here rather than derived from the
/// bound textures so that the two passes cannot disagree about it -- the resolve
/// writes a buffer the march never sees.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RotationUniform {
    at: [u32; 4],
}

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
    /// A still morning: fog lying in the valleys under a clear sky.
    Fog,
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
    /// This deck's entry from [`DECK_HUG`], and three spare.
    hug: [f32; 4],
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
}

/// Where the light volumes stand and how their columns lean.
///
/// Mirrors `Light` in `src/cloud_march.wgsl` and in `src/shading.wgsl`. Its own
/// buffer rather than two more fields of [`WeatherUniform`], because the
/// shading pass reads this and has no business reading that: where a volume
/// stands is a fact about the camera and the sun, and what is in it is a fact
/// about the weather. The shading would otherwise have to declare every deck's
/// cover ramp to reach past them.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    /// The near corner in x and z, the height of the lowest slice, and the
    /// metres one texel covers across.
    origin: [f32; 4],
    /// How far a sun column leans per metre it climbs, then the metres of ray
    /// one slice is worth towards the sun, then the metres of height one slice
    /// is worth -- which is what the same slice is worth straight up.
    walk: [f32; 4],
    /// Where the baked wind's grid stands; see [`crate::air::Air::bounds`].
    air: [f32; 4],
    /// How far the mean wind has carried the weather since the world started,
    /// in metres.
    carried: [f32; 4],
}

impl LightUniform {
    /// Where the volumes go for an eye and a sun, over a wind's own grid and
    /// however far that wind has carried the weather by now.
    fn at(eye: glam::Vec3, sun: glam::Vec3, air: [f32; 4], carried: glam::Vec3) -> Self {
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
        Self {
            origin: [corner.x, corner.y, 0.0, across],
            // A slice of height is a longer piece of a leaning ray than of a
            // plumb one, by exactly the sun's climb.
            walk: [lean.x, lean.y, up / climb, up],
            air,
            carried: [carried.x, carried.y, carried.z, 0.0],
        }
    }

    /// How far the mean wind has carried the weather by now, in metres.
    ///
    /// Folded into one tile of the weather map, which is what keeps a long
    /// flight from walking the offset out of the precision an `f32` addresses a
    /// lattice with: at ten metres a second an hour is thirty-six kilometres,
    /// and a day is nine hundred. The fold is exact for every field the march
    /// reads, because the shape's tile divides the weather's and the detail's
    /// divides the shape's -- so wrapping at the largest wraps all three.
    ///
    /// Computed here rather than in the shader for the same reason: this is an
    /// `f64` seconds count times a speed, and the shader would have to be
    /// handed the product anyway.
    fn carried_by(wind: crate::air::Wind, elapsed: std::time::Duration) -> glam::Vec3 {
        let moved = wind.velocity() * elapsed.as_secs_f32();
        glam::Vec3::new(
            moved.x.rem_euclid(WEATHER_TILE),
            0.0,
            moved.z.rem_euclid(WEATHER_TILE),
        )
    }
}

/// Where each deck's fields are drawn from.
///
/// Far apart, so that the three fields one deck takes from consecutive seeds --
/// cover, lean, density, base -- cannot collide with another deck's.
const DECK_SEEDS: [u32; DECKS] = [0x466f6700, 0x4c6f7700, 0x4d696400, 0x48696700];

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
            Self::Clear => [NONE, NONE, NONE, NONE],
            Self::Fair => [NONE, [0.50, 0.72, 1.0, 0.75], NONE, [0.60, 0.85, 0.0, 0.35]],
            Self::Broken => [
                NONE,
                [0.42, 0.66, 0.8, 0.9],
                [0.58, 0.82, 0.4, 0.5],
                [0.52, 0.80, 0.0, 0.4],
            ],
            Self::Overcast => [NONE, [0.16, 0.44, 0.2, 1.0], [0.40, 0.70, 0.2, 0.7], NONE],
            Self::Storm => [
                NONE,
                [0.04, 0.34, 0.9, 1.0],
                [0.24, 0.56, 0.6, 1.0],
                [0.45, 0.75, 0.0, 0.5],
            ],
            // A wide, low ramp and no lean: fog is the flattest stratus there
            // is, and what decides where it lies is the ground rather than the
            // field. Nothing above it -- a valley-fog morning is a morning
            // under a clear sky, and putting cumulus over it would only hide
            // what this preset is for.
            Self::Fog => [[0.30, 0.62, 0.0, 1.0], NONE, NONE, NONE],
        }
    }

    fn uniform(self, elapsed: std::time::Duration) -> WeatherUniform {
        let looks = self.decks();
        WeatherUniform {
            decks: std::array::from_fn(|deck| DeckUniform {
                look: looks[deck],
                seed: [DECK_SEEDS[deck], 0, 0, 0],
                slab: DECK_SLABS[deck],
                hug: [DECK_HUG[deck], 0.0, 0.0, 0.0],
            }),
            clock: [elapsed.as_secs_f32(), WEATHER_PERIOD, 0.0, 0.0],
            span: [cloud_span().0, cloud_span().1, 0.0, 0.0],
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
    pub fn set_frame(&self, queue: &wgpu::Queue, preset: Preset, elapsed: std::time::Duration) {
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&preset.uniform(elapsed)),
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
    /// Two of them, alternating: the one this frame resolves into, and the one
    /// last frame left for it to carry. They swap roles rather than contents,
    /// which is what keeps this to a pair of bind groups instead of a copy.
    /// `COPY_SRC` for the reason the G-buffer's targets carry it -- a frame this
    /// does not appear in says nothing about what is in it.
    colour: [wgpu::Texture; 2],
    /// How far along each of those rays the cloud it found actually was.
    depth: [wgpu::Texture; 2],
    /// This frame's own quarter of the two above, one texel per two-by-two
    /// block, before the other three quarters are carried in beside it.
    #[allow(dead_code, reason = "read through its view")]
    fresh_colour: wgpu::Texture,
    #[allow(dead_code, reason = "read through its view")]
    fresh_depth: wgpu::Texture,
    /// How much of the sun reaches each point of a coarse world grid, and how
    /// much of the sky does. One shape, two leans; see `light_position` in
    /// `src/cloud_march.wgsl`.
    #[allow(dead_code, reason = "read through their views")]
    sun_light: wgpu::Texture,
    #[allow(dead_code, reason = "read through their views")]
    sky_light: wgpu::Texture,
    colour_view: [wgpu::TextureView; 2],
    depth_view: [wgpu::TextureView; 2],
    fresh_colour_view: wgpu::TextureView,
    fresh_depth_view: wgpu::TextureView,
    ceiling_view: wgpu::TextureView,
    sun_light_view: wgpu::TextureView,
    sky_light_view: wgpu::TextureView,
    /// Where the two volumes stand this frame; see [`LightUniform`]. Read by
    /// the builds, by the march and by the shading.
    light: wgpu::Buffer,
    /// Which quarter this frame marches; see [`RotationUniform`].
    rotation: wgpu::Buffer,
    sampler: wgpu::Sampler,
    edge_sampler: wgpu::Sampler,
    ceiling_group: wgpu::BindGroup,
    /// One group per volume, against one layout: the only difference between
    /// the two dispatches is which texture they write and which entry point
    /// they run.
    light_groups: [wgpu::BindGroup; 2],

    march_layout: wgpu::BindGroupLayout,
    march_group: wgpu::BindGroup,
    /// One group per parity: which of the two buffers is written and which is
    /// read. Chosen here rather than in [`crate::scene::Scene::draw`] because
    /// that takes `&self` and cannot flip anything.
    resolve_layout: wgpu::BindGroupLayout,
    resolve_groups: [wgpu::BindGroup; 2],
    ceiling_pipeline: wgpu::ComputePipeline,
    sun_light_pipeline: wgpu::ComputePipeline,
    sky_light_pipeline: wgpu::ComputePipeline,
    march_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
    /// The half-resolution size every buffer above was built at.
    size: glam::UVec2,
    /// Which frame this is, as [`crate::scene::Scene`] counts them. Its low two
    /// bits pick the quarter marched and its low bit picks the buffer written,
    /// so the whole of the alternation is one number the scene already keeps --
    /// and putting it back after settling is therefore already done.
    frame: u32,
    /// Whether the history has been filled with an empty sky; see
    /// [`March::ensure_cleared`].
    cleared: bool,
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
    light: &'a wgpu::Buffer,
    /// The baked wind's deviation field; see `src/air.rs`.
    drift: &'a wgpu::TextureView,
    /// The ground the wind was solved around, for the deck that rides it.
    ground: &'a wgpu::TextureView,
}

/// Everything the march keeps at the size of the frame, which a resize throws
/// away and remakes.
///
/// Together rather than eight fields threaded through three functions, because
/// they are made together, replaced together, and are exactly the set that
/// [`March::resize`] has to be careful about.
struct Screen {
    colour: [wgpu::Texture; 2],
    depth: [wgpu::Texture; 2],
    fresh_colour: wgpu::Texture,
    fresh_depth: wgpu::Texture,
    colour_view: [wgpu::TextureView; 2],
    depth_view: [wgpu::TextureView; 2],
    fresh_colour_view: wgpu::TextureView,
    fresh_depth_view: wgpu::TextureView,
}

impl Screen {
    /// `size` is the half-resolution size the resolved pair is held at; the
    /// pair the march writes is half of that again.
    fn new(device: &wgpu::Device, size: glam::UVec2) -> Self {
        let buffer = |label, format, size: glam::UVec2| {
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
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let blocks = half_of(size);
        let colour = [
            buffer("cloud colour 0", CLOUD_FORMAT, size),
            buffer("cloud colour 1", CLOUD_FORMAT, size),
        ];
        let depth = [
            buffer("cloud depth 0", DISTANCE_FORMAT, size),
            buffer("cloud depth 1", DISTANCE_FORMAT, size),
        ];
        let fresh_colour = buffer("cloud colour marched", CLOUD_FORMAT, blocks);
        let fresh_depth = buffer("cloud depth marched", DISTANCE_FORMAT, blocks);
        Self {
            colour_view: std::array::from_fn(|i| colour[i].create_view(&Default::default())),
            depth_view: std::array::from_fn(|i| depth[i].create_view(&Default::default())),
            fresh_colour_view: fresh_colour.create_view(&Default::default()),
            fresh_depth_view: fresh_depth.create_view(&Default::default()),
            colour,
            depth,
            fresh_colour,
            fresh_depth,
        }
    }
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
        air: &crate::air::Air,
    ) -> Self {
        let (_, drift, ground) = air.views();
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

        let light = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud light uniform"),
            size: std::mem::size_of::<LightUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let rotation = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud rotation uniform"),
            size: std::mem::size_of::<RotationUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
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
        // The one entry both the builds, the march and the shading want, which
        // is why it is spelled once. Visible to the fragment stage as well: the
        // ground reads these volumes too.
        let light_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
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
                light_entry(14),
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
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: light.as_entire_binding(),
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
                light_entry(14),
                sampled(15, wgpu::TextureViewDimension::D3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    ..uniform
                },
                sampled(21, wgpu::TextureViewDimension::D2, true),
            ],
        });

        // What the resolve reads and writes, and deliberately nothing else. The
        // march's own fields are absent because it has no use for them: it does
        // not evaluate a cloud, it decides where last frame's answer has moved
        // to. The buffer it writes is bound writable here and sampled by the
        // shading, which is a different pass; the buffer it reads is the other
        // one of the pair, which is the whole point of there being two.
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud resolve layout"),
            entries: &[
                // Only for its size: `ray_raw_at` measures the screen in
                // full-resolution pixels, and the half-resolution buffer cannot
                // say whether the frame was an odd number of them wide.
                sampled(5, wgpu::TextureViewDimension::D2, false),
                written(7, CLOUD_FORMAT, wgpu::TextureViewDimension::D2),
                written(8, DISTANCE_FORMAT, wgpu::TextureViewDimension::D2),
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    ..uniform
                },
                // This frame's quarter, read at the texels it was marched at.
                sampled(17, wgpu::TextureViewDimension::D2, false),
                sampled(18, wgpu::TextureViewDimension::D2, false),
                // The history. Its colour is filtered, because it is read at
                // wherever the camera has moved this texel to rather than at a
                // texel centre; its distance is not, and could not be -- an
                // `r32float` is unfilterable in core WebGPU, which here is the
                // restriction agreeing with what was wanted anyway.
                sampled(19, wgpu::TextureViewDimension::D2, true),
                sampled(20, wgpu::TextureViewDimension::D2, false),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                light_entry(14),
                sampled(15, wgpu::TextureViewDimension::D3, true),
                sampled(21, wgpu::TextureViewDimension::D2, true),
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
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::Sampler(&edge_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: light.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: wgpu::BindingResource::TextureView(drift),
                    },
                    wgpu::BindGroupEntry {
                        binding: 21,
                        resource: wgpu::BindingResource::TextureView(ground),
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
        // The camera and nothing else: where a texel used to be on the screen
        // is a question about the two view matrices, and the answer says
        // nothing about what is in the sky.
        let resolve_pipeline = pipeline(
            "cloud resolve",
            "cs_cloud_resolve",
            &[Some(camera_layout), None, None, Some(&resolve_layout)],
        );

        let size = half_of(gbuffer.size);
        let screen = Screen::new(device, size);
        let lit = Lit {
            ceiling: &ceiling_view,
            sun: &sun_light_view,
            sky: &sky_light_view,
            tiling: &sampler,
            edge: &edge_sampler,
            light: &light,
            drift,
            ground,
        };
        let march_group = Self::bind(
            device,
            &march_layout,
            cloud,
            gbuffer,
            &lit,
            &screen,
            &rotation,
        );
        let resolve_groups = Self::resolve_bind(
            device,
            &resolve_layout,
            gbuffer,
            &edge_sampler,
            &screen,
            &rotation,
        );

        let Screen {
            colour,
            depth,
            fresh_colour,
            fresh_depth,
            colour_view,
            depth_view,
            fresh_colour_view,
            fresh_depth_view,
        } = screen;
        Self {
            ceiling,
            sun_light,
            sky_light,
            colour,
            depth,
            fresh_colour,
            fresh_depth,
            colour_view,
            depth_view,
            fresh_colour_view,
            fresh_depth_view,
            ceiling_view,
            sun_light_view,
            sky_light_view,
            sampler,
            edge_sampler,
            light,
            rotation,
            ceiling_group,
            light_groups,
            march_layout,
            march_group,
            resolve_layout,
            resolve_groups,
            ceiling_pipeline,
            sun_light_pipeline,
            sky_light_pipeline,
            march_pipeline,
            resolve_pipeline,
            size,
            frame: 0,
            cleared: false,
        }
    }

    /// Follows the render target to a new size.
    ///
    /// The cache does not move -- it is a fact about the world, at a resolution
    /// of its own -- so only the screen-sized buffers and the groups naming them
    /// are rebuilt. Called from [`crate::scene::Scene::resize`], which also
    /// hands over the rebuilt G-buffer this reads the depth of.
    ///
    /// The history goes with them, and is the zeroes a new texture is created
    /// with. Nothing has to be said about that: the resolve clamps a carried
    /// texel to the range of the answers around it, and on the first frame
    /// after a resize those are the only answers there are.
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        cloud: &Cloud,
        gbuffer: &crate::deferred::GBuffer,
        air: &crate::air::Air,
    ) {
        let (_, drift, ground) = air.views();
        self.size = half_of(gbuffer.size);
        let screen = Screen::new(device, self.size);
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
                light: &self.light,
                drift,
                ground,
            },
            &screen,
            &self.rotation,
        );
        self.resolve_groups = Self::resolve_bind(
            device,
            &self.resolve_layout,
            gbuffer,
            &self.edge_sampler,
            &screen,
            &self.rotation,
        );
        self.colour = screen.colour;
        self.depth = screen.depth;
        self.fresh_colour = screen.fresh_colour;
        self.fresh_depth = screen.fresh_depth;
        self.colour_view = screen.colour_view;
        self.depth_view = screen.depth_view;
        self.fresh_colour_view = screen.fresh_colour_view;
        self.fresh_depth_view = screen.fresh_depth_view;
        // A new pair, so a new empty sky to put in it.
        self.cleared = false;
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        cloud: &Cloud,
        gbuffer: &crate::deferred::GBuffer,
        lit: &Lit<'_>,
        screen: &Screen,
        rotation: &wgpu::Buffer,
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
                // The quarter-sized pair. The march writes one texel per
                // two-by-two block of the frame's own buffer, and writes it
                // packed rather than scattered: a dispatch a quarter the size,
                // with every thread of it marching.
                texture(7, &screen.fresh_colour_view),
                texture(8, &screen.fresh_depth_view),
                texture(11, lit.sun),
                texture(12, lit.sky),
                sampler(13, lit.edge),
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: lit.light.as_entire_binding(),
                },
                texture(15, lit.drift),
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: rotation.as_entire_binding(),
                },
                texture(21, lit.ground),
            ],
        })
    }

    /// One group per parity: which buffer is written and which is carried.
    fn resolve_bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        gbuffer: &crate::deferred::GBuffer,
        edge: &wgpu::Sampler,
        screen: &Screen,
        rotation: &wgpu::Buffer,
    ) -> [wgpu::BindGroup; 2] {
        std::array::from_fn(|parity| {
            let texture = |binding, view| wgpu::BindGroupEntry {
                binding,
                resource: wgpu::BindingResource::TextureView(view),
            };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cloud resolve group"),
                layout,
                entries: &[
                    texture(5, &gbuffer.depth),
                    texture(7, &screen.colour_view[parity]),
                    texture(8, &screen.depth_view[parity]),
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: wgpu::BindingResource::Sampler(edge),
                    },
                    wgpu::BindGroupEntry {
                        binding: 16,
                        resource: rotation.as_entire_binding(),
                    },
                    texture(17, &screen.fresh_colour_view),
                    texture(18, &screen.fresh_depth_view),
                    texture(19, &screen.colour_view[1 - parity]),
                    texture(20, &screen.depth_view[1 - parity]),
                ],
            })
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

    /// Says where the light volumes stand for the frame about to be drawn.
    ///
    /// Every frame, because the camera moves and the sun may. Separate from
    /// [`Cloud::set_frame`] because it answers a different question with
    /// different inputs: that one is asked what kind of day it is, this one
    /// where the eye is and where the sun is.
    pub fn set_frame(
        &self,
        queue: &wgpu::Queue,
        eye: glam::Vec3,
        sun: glam::Vec3,
        air: &crate::air::Air,
        wind: crate::air::Wind,
        elapsed: std::time::Duration,
    ) {
        let placed = LightUniform::at(
            eye,
            sun,
            air.bounds(),
            LightUniform::carried_by(wind, elapsed),
        );
        queue.write_buffer(&self.light, 0, bytemuck::bytes_of(&placed));
    }

    /// Says which quarter of the buffer this frame marches.
    ///
    /// `frame` is [`crate::scene::Scene`]'s own frame counter, which it already
    /// puts back after settling -- and settling is not time passing, so the
    /// rotation must not advance through it either. Its own call rather than
    /// two more arguments above, because the scene has to make it again by
    /// itself when it puts that counter back.
    pub fn set_rotation(&mut self, queue: &wgpu::Queue, frame: u32) {
        self.frame = frame;
        let at = ROTATION[self.phase()];
        queue.write_buffer(
            &self.rotation,
            0,
            bytemuck::bytes_of(&RotationUniform {
                at: [at[0], at[1], self.size.x, self.size.y],
            }),
        );
    }

    /// Fills the history with an empty sky, once, before anything reads it.
    ///
    /// The zeroes a new texture comes with are a transmittance of nothing,
    /// which is opaque black cloud. The resolve clamps a carried texel to the
    /// range of the marched answers around it and that puts almost all of them
    /// right -- but where a texel sits on a silhouette and too few of its
    /// neighbours are looking as far as it is, the range widens to let a stale
    /// answer through, and on the first frame the stale answer is the zeroes.
    /// Measured at fifty-three black specks in the first frame of a flight and
    /// none in the second, which is the rotation coming round.
    ///
    /// Exactly one in the fourth channel, which matters: it is what the
    /// composite's early return reads, and it is the value a clear-sky frame
    /// has to keep bit for bit. See `cloud_at` in `src/shading.wgsl`.
    ///
    /// The same shape as [`Cloud::ensure_built`] and [`crate::sky::Sky`]'s, and
    /// here rather than in the constructor for the same reason: writing a
    /// texture needs a queue, and a constructor has only a device.
    pub fn ensure_cleared(&mut self, queue: &wgpu::Queue) {
        if self.cleared {
            return;
        }
        self.cleared = true;
        let empty = half::f16::ONE.to_le_bytes();
        let sky: Vec<u8> = std::iter::repeat_n(
            [0, 0, 0, 0, 0, 0, empty[0], empty[1]],
            (self.size.x * self.size.y) as usize,
        )
        .flatten()
        .collect();
        for texture in &self.colour {
            queue.write_texture(
                texture.as_image_copy(),
                &sky,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.size.x * 8),
                    rows_per_image: Some(self.size.y),
                },
                wgpu::Extent3d {
                    width: self.size.x,
                    height: self.size.y,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Which texel of a two-by-two block this frame marches, as an index into
    /// [`ROTATION`].
    fn phase(&self) -> usize {
        (self.frame & 3) as usize
    }

    /// Which of the two resolved buffers this frame writes; the other is the
    /// history it carries.
    pub fn parity(&self) -> usize {
        (self.frame & 1) as usize
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
    ///
    /// A quarter of the buffer's texels, packed: see [`ROTATION`].
    pub fn draw(&self, pass: &mut wgpu::ComputePass<'_>, sky: &crate::sky::Sky) {
        let blocks = half_of(self.size);
        pass.set_pipeline(&self.march_pipeline);
        pass.set_bind_group(1, sky.bind_group(), &[]);
        pass.set_bind_group(2, sky.sun_tables_bind_group(), &[]);
        pass.set_bind_group(3, &self.march_group, &[]);
        pass.dispatch_workgroups(
            blocks.x.div_ceil(MARCH_GROUP),
            blocks.y.div_ceil(MARCH_GROUP),
            1,
        );
    }

    /// Records the resolve, which fills the other three quarters from the last
    /// frame's answer. The caller has set group 0 to the camera.
    ///
    /// Its own pass, after the march: it reads what the march wrote, and a pass
    /// boundary is what makes those writes visible.
    pub fn draw_resolve(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(3, &self.resolve_groups[self.parity()], &[]);
        pass.dispatch_workgroups(
            self.size.x.div_ceil(MARCH_GROUP),
            self.size.y.div_ceil(MARCH_GROUP),
            1,
        );
    }

    /// Everything the shading reads of the clouds: the half-resolution buffers
    /// the composite upsamples, and the two light volumes the ground is
    /// shadowed out of.
    ///
    /// Both of the alternating pair, because the shading is built once and the
    /// alternation is per frame; which one is read is [`March::parity`], handed
    /// over at the draw.
    pub fn views(&self) -> crate::deferred::Clouds<'_> {
        crate::deferred::Clouds {
            colour: [&self.colour_view[0], &self.colour_view[1]],
            along: [&self.depth_view[0], &self.depth_view[1]],
            sun: &self.sun_light_view,
            sky: &self.sky_light_view,
            light: &self.light,
        }
    }

    /// The textures themselves, for the tests that read them back.
    ///
    /// The resolved pair is the one this frame wrote, which is what the shading
    /// read: a test asking what the frame showed wants that one and not the
    /// history beside it.
    #[cfg(test)]
    pub fn buffers_for_test(
        &self,
    ) -> (
        &wgpu::Texture,
        &wgpu::Texture,
        &wgpu::Texture,
        &wgpu::Texture,
    ) {
        (
            &self.colour[self.parity()],
            &self.depth[self.parity()],
            &self.ceiling,
            &self.sun_light,
        )
    }

    /// This frame's own quarter, before the rest was carried in beside it.
    #[cfg(test)]
    pub fn marched_for_test(&self) -> (&wgpu::Texture, &wgpu::Texture) {
        (&self.fresh_colour, &self.fresh_depth)
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
        cloud.set_frame(&queue, preset, elapsed);

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
            .map(|preset| cover(&forecast(*preset, std::time::Duration::ZERO)[LOW_DECK]))
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

        let flicker = difference(&start[LOW_DECK], &frame_later[LOW_DECK]);
        let drift = difference(&start[LOW_DECK], &much_later[LOW_DECK]);
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
        /// What each frame of the rotation actually marched, one entry per
        /// frame, at one texel per two-by-two block of `colour`.
        marched: Vec<Vec<[f32; 4]>>,
        /// The size those are held at.
        blocks: glam::UVec2,
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

    /// Marches a full rotation of one preset, from a camera of the caller's
    /// choosing.
    ///
    /// Four frames rather than one, because one frame is a quarter of an answer:
    /// the march fills one texel of every two-by-two block and the resolve
    /// carries the rest in from the frame before. Four is what it takes for
    /// every texel of the buffer to have been marched at its own place, which is
    /// the state a flight is in at every frame but its first three.
    ///
    /// The camera is still and the clock is stopped, so the reprojection between
    /// them is the identity and the four quarters compose into exactly the frame
    /// a march of every texel would have left. That is the point of
    /// [`Marched::marched`], which keeps what each of the four actually did.
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
        marched_at(
            preset,
            camera,
            size,
            sun,
            crate::air::Wind::default(),
            std::time::Duration::ZERO,
        )
    }

    /// The same again, at a moment of a day the wind has been blowing through.
    fn marched_at(
        preset: Preset,
        camera: &crate::camera::Camera,
        size: glam::UVec2,
        sun: crate::sky::Sun,
        wind: crate::air::Wind,
        elapsed: std::time::Duration,
    ) -> Marched {
        marched_over(
            preset,
            &[camera; ROTATION.len()],
            size,
            sun,
            wind,
            elapsed,
            None,
        )
    }

    /// The same again, one camera per frame.
    ///
    /// A frame apiece rather than one for all of them, because what the resolve
    /// does is only visible when the camera has moved: it reads the buffer the
    /// last frame left, at wherever this frame's camera has since put that
    /// texel. Given the same camera four times over -- which is what everything
    /// above hands it -- that reprojection is the identity and the four
    /// quarters compose into the frame a march of every texel would have left.
    #[allow(clippy::too_many_arguments, reason = "one test wants a terrain too")]
    fn marched_over(
        preset: Preset,
        cameras: &[&crate::camera::Camera],
        size: glam::UVec2,
        sun: crate::sky::Sun,
        wind: crate::air::Wind,
        elapsed: std::time::Duration,
        ground: Option<&[f32]>,
    ) -> Marched {
        let camera = cameras[0];
        let (device, queue) = crate::headless::device().expect("no headless device");
        let (camera_buffer, camera_layout, camera_group) =
            crate::scene::test_camera_buffer(&device, &queue, camera);

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
        cloud.set_frame(&queue, preset, elapsed);

        let gbuffer = crate::deferred::GBuffer::new(&device, size);
        // An unsolved wind, which reports a grid of no extent and so puts no
        // drift and no lift anywhere. What the wind does to a cloud is a
        // question about a terrain, and it is asked in `src/scene.rs` over one.
        let mut air = crate::air::Air::new(&device);
        // A terrain for a deck to hang off, and no wind at all: what the ground
        // does to a cloud base and what the wind does to it are two questions,
        // and mixing them would leave neither answered.
        if let Some(heights) = ground {
            air.assume_ground(&queue, glam::Vec2::splat(GROUND_EXTENT), heights);
        }
        let mut march = March::new(
            &device,
            &camera_layout,
            sky.layout(),
            sky.sun_tables_layout(),
            &cloud,
            &gbuffer,
            &air,
        );

        let blocks = half_of(march.size());
        let mut fresh = Vec::new();
        for (frame, aim) in cameras.iter().enumerate() {
            // Where this frame's camera is and where it has just come from,
            // which is the whole of what the resolve carries a texel through.
            queue.write_buffer(
                &camera_buffer,
                0,
                bytemuck::bytes_of(&crate::scene::test_camera_moved(
                    aim,
                    cameras[frame.saturating_sub(1)],
                )),
            );
            march.set_frame(&queue, aim.position, sun.direction, &air, wind, elapsed);
            march.set_rotation(&queue, frame as u32);

            let mut encoder = device.create_command_encoder(&Default::default());
            // Five passes, as `Scene::draw` records them, and for the reason it
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
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &camera_group, &[]);
                march.draw_resolve(&mut pass);
            }
            queue.submit(std::iter::once(encoder.finish()));
            fresh.push(
                read_texels(
                    &device,
                    &queue,
                    march.marched_for_test().0,
                    blocks.x,
                    blocks.y,
                    1,
                    8,
                )
                .chunks_exact(8)
                .map(|texel| {
                    std::array::from_fn(|c| {
                        half::f16::from_le_bytes([texel[c * 2], texel[c * 2 + 1]]).to_f32()
                    })
                })
                .collect(),
            );
        }

        let (colour, depth, ceiling, _) = march.buffers_for_test();
        let half = march.size();
        Marched {
            marched: fresh,
            blocks,
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

    /// How much world the ground the fog tests draw covers, in metres.
    ///
    /// The grid is centred on the origin, as [`crate::air::Air::bounds`] lays
    /// it out, so a camera that wants a terrain under it has to fly here rather
    /// than at [`AWAY`]. Wide enough that a level ray leaves the far side of
    /// it well beyond anything the frame shows.
    const GROUND_EXTENT: f32 = 40_000.0;

    /// A ground field at one height everywhere.
    fn ground_at_height(height: f32) -> Vec<f32> {
        vec![height; (crate::air::CELLS[0] * crate::air::CELLS[2]) as usize]
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

    /// The rotation marches every texel of the buffer, each at its own place.
    ///
    /// The whole of what makes a quarter of a march look like a march. Each
    /// frame marches one texel of every two-by-two block and the resolve puts
    /// it back where it was taken from; over four frames the four sub-positions
    /// between them cover the buffer exactly once. Get the offsets wrong -- a
    /// repeated one, a transposed one, a block index used where a texel index
    /// was wanted -- and some texels are marched twice while others are never
    /// marched at all, and the ones that are never marched hold whatever three
    /// frames of carrying has left there.
    ///
    /// Two claims. The cover is exact and is counted rather than sampled: every
    /// texel claimed by exactly one frame of the rotation. What is in them is
    /// then checked against those very answers, assembled on the CPU -- the
    /// camera is still and the clock is stopped, so a texel carried through
    /// three frames should come back to itself. Measured at 0.0018 of mean
    /// transmittance and 0.062 at worst, which is the round trip through a
    /// filtered fetch and a clamp and not a texel in the wrong place.
    #[test]
    fn the_rotation_marches_every_texel_at_its_own_place() {
        let frame = marched(
            Preset::Broken,
            &looking(1500.0, 0.0),
            glam::UVec2::splat(128),
        );
        assert!(
            frame.opacity() > 0.2,
            "a broken sky covered only {:.3} of the frame, so this measures nothing",
            frame.opacity()
        );

        let mut oracle = vec![[0.0f32; 4]; frame.colour.len()];
        let mut claims = vec![0u32; frame.colour.len()];
        for (phase, marched) in frame.marched.iter().enumerate() {
            let off = ROTATION[phase];
            for by in 0..frame.blocks.y {
                for bx in 0..frame.blocks.x {
                    // `marched_texel` in `src/cloud_march.wgsl`, in Rust.
                    let x = (bx * 2 + off[0]).min(frame.size.x - 1);
                    let y = (by * 2 + off[1]).min(frame.size.y - 1);
                    let at = (y * frame.size.x + x) as usize;
                    oracle[at] = marched[(by * frame.blocks.x + bx) as usize];
                    claims[at] += 1;
                }
            }
        }
        let missed = claims.iter().filter(|c| **c == 0).count();
        let twice = claims.iter().filter(|c| **c > 1).count();
        assert_eq!(
            (missed, twice),
            (0, 0),
            "of {} texels the rotation never marched {missed} and marched {twice} more than once",
            claims.len()
        );

        let apart = oracle
            .iter()
            .zip(&frame.colour)
            .map(|(want, got)| (want[3] - got[3]).abs());
        let mean = apart.clone().map(f64::from).sum::<f64>() / oracle.len() as f64;
        let worst = apart.fold(0.0f32, f32::max);
        assert!(
            mean < 0.005 && worst < 0.12,
            "a texel carried back to itself came back {mean:.6} from its own march \
             on average and {worst:.6} at worst"
        );
    }

    /// A still world leaves the buffer where it was.
    ///
    /// What "the clouds stopped boiling" is, said as a number. Every texel is
    /// re-marched once every four frames and carried through the other three,
    /// and the carry is a fetch of a filtered texel clamped to the range of its
    /// neighbours -- three operations that could each leave a little of
    /// themselves behind, and compound over a flight into a sky that crawls
    /// while nothing moves.
    ///
    /// Three rotations against one, over a camera that does not move and a
    /// clock that does not run, so the only thing that can differ is what the
    /// carrying has done. Measured at 0.0016 of mean transmittance, and 0.049
    /// at the single worst texel.
    #[test]
    fn a_still_world_leaves_the_buffer_where_it_was() {
        let camera = looking(1500.0, 0.0);
        let size = glam::UVec2::splat(128);
        let rounds = |n: usize| {
            marched_over(
                Preset::Broken,
                &vec![&camera; n],
                size,
                crate::sky::Sun::default(),
                crate::air::Wind::default(),
                std::time::Duration::ZERO,
                None,
            )
        };
        let once = rounds(ROTATION.len());
        let thrice = rounds(ROTATION.len() * 3);
        let apart = once
            .colour
            .iter()
            .zip(&thrice.colour)
            .map(|(a, b)| (a[3] - b[3]).abs());
        let mean = apart.clone().map(f64::from).sum::<f64>() / once.colour.len() as f64;
        let worst = apart.fold(0.0f32, f32::max);
        assert!(
            mean < 0.005 && worst < 0.1,
            "two more rotations over a world that did not move shifted the sky by \
             {mean:.6} on average and {worst:.6} at worst"
        );
    }

    /// A camera that turns carries the sky it was looking at with it.
    ///
    /// The reprojection, which is what the other three quarters of every frame
    /// are worth. A texel not marched this frame is read out of the last
    /// frame's buffer at wherever the camera has since moved that texel to; get
    /// that wrong and the buffer holds the *old* view, three quarters of it,
    /// smeared across the new one.
    ///
    /// Four frames of one camera and then a single frame three degrees to the
    /// side, against four frames of the turned camera from cold. Carried, the
    /// two are 0.0040 of mean transmittance apart -- a fifth of the 0.0197 that
    /// separates the old view from the new one, which is what a frame with no
    /// reprojection in it would be showing. The second number is asserted as
    /// well, because without it the first says nothing: two frames of the same
    /// sky are close whatever the pass does with them.
    #[test]
    fn a_camera_that_turns_carries_the_sky_with_it() {
        let size = glam::UVec2::splat(128);
        let still = looking(1500.0, 0.0);
        let turned = crate::camera::Camera::new(
            still.position,
            crate::camera::Camera::from_yaw_pitch_roll(3f32.to_radians(), 0.0, 0.0),
            1.0,
        );
        let at = |cameras: &[&crate::camera::Camera]| {
            marched_over(
                Preset::Broken,
                cameras,
                size,
                crate::sky::Sun::default(),
                crate::air::Wind::default(),
                std::time::Duration::ZERO,
                None,
            )
        };
        let apart = |a: &Marched, b: &Marched| {
            a.colour
                .iter()
                .zip(&b.colour)
                .map(|(x, y)| f64::from((x[3] - y[3]).abs()))
                .sum::<f64>()
                / a.colour.len() as f64
        };

        let cold = at(&[&turned; ROTATION.len()]);
        let carried = at(&[&still, &still, &still, &still, &turned]);
        let left = at(&[&still; ROTATION.len() + 1]);
        let moved = apart(&left, &cold);
        assert!(
            moved > 0.015,
            "turning three degrees changed the sky by only {moved:.6}, so nothing \
             here is being measured"
        );
        let ghost = apart(&carried, &cold);
        assert!(
            ghost < 0.010 && ghost < moved * 0.5,
            "a buffer carried through the turn is {ghost:.6} from one marched at \
             the new camera, against {moved:.6} for one not carried at all"
        );
    }

    /// Fog pools to a level and thins as the ground comes up under it.
    ///
    /// The whole of what a deck that rides the ground is for, and the whole of
    /// what separates it from one laid over the hills at a constant depth. A
    /// hugging deck keeps the top [`DECK_SLABS`] gives it and takes its base
    /// from the terrain, so raising the ground under it does not raise the fog:
    /// it *squeezes* it, until at the top there is none left. A blanket would
    /// come back with the same reading at every height, which is what this
    /// would look like if `hug` were dropped on the floor.
    ///
    /// Measured over a flat ground at three heights, from a camera above the
    /// deck looking down into it: 0.912 of the frame covered at seven hundred
    /// metres, 0.725 at eleven hundred, and nothing at all at fourteen hundred,
    /// where the ground has come up past the highest the top can swing to.
    ///
    /// The last of those is exact and is the one that matters most: a ridge
    /// standing out of the fog has to stand *clear* of it, not in a haze of
    /// it, or every hilltop in the frame wears a halo.
    #[test]
    fn fog_pools_to_a_level_and_thins_as_the_ground_comes_up() {
        // Over the origin rather than at [`AWAY`], because the ground grid is
        // centred there; see [`GROUND_EXTENT`].
        let camera = crate::camera::Camera::new(
            glam::Vec3::new(0.0, 1500.0, 0.0),
            crate::camera::Camera::from_yaw_pitch_roll(0.0, (-35f32).to_radians(), 0.0),
            1.0,
        );
        let size = glam::UVec2::splat(96);
        let over = |height: f32| {
            marched_over(
                Preset::Fog,
                &[&camera; ROTATION.len()],
                size,
                crate::sky::Sun::default(),
                crate::air::Wind::default(),
                std::time::Duration::ZERO,
                Some(&ground_at_height(height)),
            )
            .opacity()
        };

        let (deep, shallow, above) = (over(700.0), over(1100.0), over(1400.0));
        assert!(
            deep > 0.8,
            "fog over ground at seven hundred metres covered only {deep:.3} of the frame"
        );
        assert!(
            shallow < deep * 0.9,
            "the ground rising four hundred metres into the fog left {shallow:.3} \
             of the frame covered against {deep:.3}, which is not a pool being \
             squeezed but a blanket being carried"
        );
        assert_eq!(
            above, 0.0,
            "ground above the highest the fog's top can reach still left \
             {above:.3} of the frame in fog"
        );
    }

    /// The cache bounds the fog too, over ground it is not allowed to read.
    ///
    /// A hugging deck's profile is measured from the terrain, and the cache
    /// cannot know where the terrain is: it tiles every sixty kilometres and
    /// the ground does not, so one cell of it stands over every terrain in the
    /// world at once. So it bounds a hugging deck as though the deck were
    /// filling anywhere in its band -- and this is what says that is still a
    /// bound, walked over a ground that slopes twelve hundred metres across the
    /// grid so that neighbouring cells hold the fog at quite different heights.
    ///
    /// Bounding it the way an ordinary deck is bounded -- by where in the cell's
    /// own heights the profile could peak, measured from a base at sea level --
    /// fails this by 0.05, which is most of the fog.
    #[test]
    fn no_cell_of_the_ceiling_claims_less_fog_than_it_holds() {
        let camera = crate::camera::Camera::new(
            glam::Vec3::new(0.0, 1500.0, 0.0),
            crate::camera::Camera::from_yaw_pitch_roll(0.0, (-35f32).to_radians(), 0.0),
            1.0,
        );
        // A slope rather than a plane, so that the ground varies inside a cell
        // of the cache as well as between cells.
        let across = crate::air::CELLS[0];
        let heights: Vec<f32> = (0..across * crate::air::CELLS[2])
            .map(|i| 400.0 + 1200.0 * (i % across) as f32 / (across - 1) as f32)
            .collect();
        let frame = marched_over(
            Preset::Fog,
            &[&camera; ROTATION.len()],
            glam::UVec2::splat(32),
            crate::sky::Sun::default(),
            crate::air::Wind::default(),
            std::time::Duration::ZERO,
            Some(&heights),
        );

        let step = 61.0;
        let mut worst: f32 = 0.0;
        let mut checked = 0u32;
        for iz in -40..40 {
            for ix in -40..40 {
                for slice in 0..CEILING_SLICES {
                    let p = glam::Vec3::new(
                        ix as f32 * step,
                        (f32::from(slice as u16) + 0.37) * (CEILING_TOP / CEILING_SLICES as f32),
                        iz as f32 * step,
                    );
                    let across = CEILING_ACROSS as i32;
                    let cell = |v: f32| {
                        let raw = (v / (WEATHER_TILE / CEILING_ACROSS as f32)).floor() as i32;
                        ((raw % across) + across) % across
                    };
                    let bound = frame.cell(cell(p.x) as u32, slice, cell(p.z) as u32);
                    let most = most_at(&frame, p, sampled_ground(&heights, p.x, p.z));
                    worst = worst.max(most - bound);
                    checked += 1;
                }
            }
        }
        assert!(
            worst < 5e-4,
            "the cache fell {worst:.5} short of the fog in it, over {checked} points"
        );
    }

    /// The ground under a point, sampled the way `ground_at` samples it.
    ///
    /// Bilinear over the grid `Air` lays out, clamped at its edges, through the
    /// half floats the map is stored in -- all three of which the shader does,
    /// and the last of which is worth a couple of metres at a thousand.
    fn sampled_ground(heights: &[f32], x: f32, z: f32) -> f32 {
        let across = crate::air::CELLS[0] as i32;
        let down = crate::air::CELLS[2] as i32;
        let uv = glam::Vec2::new(x, z) / GROUND_EXTENT + 0.5;
        let at = uv * glam::Vec2::new(across as f32, down as f32) - 0.5;
        let corner = at.floor();
        let f = at - corner;
        let tap = |dx: i32, dy: i32| {
            let ix = (corner.x as i32 + dx).clamp(0, across - 1);
            let iy = (corner.y as i32 + dy).clamp(0, down - 1);
            half::f16::from_f32(heights[(iy * across + ix) as usize]).to_f32()
        };
        let low = tap(0, 0) + (tap(1, 0) - tap(0, 0)) * f.x;
        let high = tap(0, 1) + (tap(1, 1) - tap(0, 1)) * f.x;
        low + (high - low) * f.y
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
    fn most_at(frame: &Marched, p: glam::Vec3, ground: f32) -> f32 {
        let extinction = shader_constant("EXTINCTION");
        let edge = shader_constant("EDGE");
        // Summed rather than maxed, because the march sums: decks may share
        // heights, and where they do a point is in both of them.
        let mut most = 0.0;
        for (deck, slab) in DECK_SLABS.iter().enumerate() {
            if p.y < slab[0] || p.y > slab[1] + slab[2] {
                continue;
            }
            let w = frame.forecast_at(deck, p.x, p.z);
            let mut base = slab[0] + w[3] * slab[2];
            let mut thickness = slab[1] - slab[0];
            if DECK_HUG[deck] > 0.0 {
                base = base.max(ground);
                thickness = (slab[1] + w[3] * slab[2] - base).max(1.0);
            }
            let coverage = w[0] * vertical((p.y - base) / thickness, w[1]);
            most += extinction * w[2] * slab[3] * (coverage / edge).clamp(0.0, 1.0);
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
                    let most = most_at(&frame, p, 0.0);
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

    /// The light volume blocks more the further down a deck it goes.
    ///
    /// The defining property of the walk, and the one an off-by-one in it
    /// destroys: a column is filled from the top down, so every texel holds the
    /// cloud above it and no texel may block less than the one over its head.
    /// Walking the other way instead -- which is one character -- would fill a
    /// deck's *top* with its own shadow and leave its base in full sun, and the
    /// image that produces is a cloud lit from below, which reads as strange
    /// rather than as wrong.
    ///
    /// The volume holds what it blocks rather than what it lets through; see
    /// `walk_light` for why an empty sky has to come back as an exact zero.
    /// Measured over an overcast sky, which is the one that fills it: the
    /// highest slice blocks almost nothing and the lowest almost everything.
    #[test]
    fn the_light_volume_blocks_more_the_further_down_a_deck_it_goes() {
        let frame = marched(
            Preset::Overcast,
            &looking(1500.0, 0.0),
            glam::UVec2::splat(32),
        );
        let mut worst: f32 = 0.0;
        let mut most: f32 = 0.0;
        let mut least: f32 = 1.0;
        for z in 0..LIGHT_ACROSS {
            for x in 0..LIGHT_ACROSS {
                least = least.min(frame.lit(x, z, LIGHT_SLICES - 1));
                most = most.max(frame.lit(x, z, 0));
                for slice in 0..LIGHT_SLICES - 1 {
                    // Below may never block less than above.
                    worst = worst.max(frame.lit(x, z, slice + 1) - frame.lit(x, z, slice));
                }
            }
        }
        // Half floats, so two texels that should be equal can differ by a
        // thousandth of what they hold.
        assert!(
            worst < 1e-3,
            "a texel of the light volume blocks {worst} more than the one below it"
        );
        assert!(
            least < 0.01,
            "the top of the volume blocks {least} of the sun, with nothing \
             above it to block any"
        );
        assert!(
            most > 0.99,
            "the bottom of an overcast deck blocks only {most} of the sun"
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

    /// The wind carries the sky, and only the wind does.
    ///
    /// The bulk drift is the whole of "clouds animate over time": the field is
    /// sampled where it *was* rather than advected forward, so nothing
    /// diffuses, nothing is stored between frames, and a sky an hour old is as
    /// crisp as the one it started as. All it costs is a subtraction.
    ///
    /// Two frames a minute apart under a strong wind, against the same two
    /// under none. The weather evolves on its own -- that is what makes a front
    /// arrive rather than appear -- so the still pair is what says the moving
    /// pair moved because of the wind and not merely because time passed.
    #[test]
    fn the_wind_carries_the_sky_it_is_blowing_through() {
        // Looking down on the deck rather than along it. A level view is mostly
        // saturated cloud near the horizon, opaque before the wind and opaque
        // after it, and the mean of the frame is then a measure of how much sky
        // is not worth measuring.
        let camera = looking(6000.0, -35.0);
        let size = glam::UVec2::splat(64);
        let at = |speed, seconds| {
            marched_at(
                Preset::Broken,
                &camera,
                size,
                crate::sky::Sun::default(),
                crate::air::Wind {
                    speed,
                    from_degrees: 270.0,
                },
                std::time::Duration::from_secs_f32(seconds),
            )
        };
        let apart = |a: &Marched, b: &Marched| {
            a.colour
                .iter()
                .zip(&b.colour)
                .map(|(x, y)| f64::from((x[3] - y[3]).abs()))
                .sum::<f64>()
                / a.colour.len() as f64
        };

        // Two frames of the *same instant*, one under a gale and one under
        // nothing. Comparing two instants instead would compare the weather's
        // own evolution as well, and that turns out to swamp the carry
        // completely: a minute apart, a still sky differs from itself by 0.29
        // where a minute of a twenty-metre wind differs by 0.27. Both numbers
        // are the front moving through, and neither is the wind.
        let still = at(0.0, 60.0);
        let blown = at(20.0, 60.0);
        // Measured at 0.132 of mean transmittance over the frame, against 0.015
        // seen along the deck rather than down onto it.
        let carried = apart(&still, &blown);
        assert!(
            carried > 0.05,
            "a minute of a twenty-metre wind left the sky {carried:.4} from \
             where no wind at all left it"
        );

        // ... and at the very start of the world the two are the same sky, to
        // the last bit, because no time has passed for a wind to carry it
        // through. That is what says the difference above is the carry and not
        // the wind having got into something else.
        let (early, gale) = (at(0.0, 0.0), at(20.0, 0.0));
        assert_eq!(apart(&early, &gale), 0.0, "the wind blew before time began");
    }

    /// The carried offset folds into one tile of the map, exactly.
    ///
    /// A flight is long and an `f32` is not: at ten metres a second a day is
    /// nine hundred kilometres, and addressing a two-hundred-metre detail
    /// lattice at that range has lost most of the precision it needs. The
    /// offset is folded into one tile of the weather map instead -- which is
    /// exact for all three fields at once, because the shape's tile divides the
    /// weather's and the detail's divides the shape's.
    #[test]
    fn the_carried_offset_folds_into_one_tile_of_the_map() {
        let westerly = crate::air::Wind {
            speed: 10.0,
            from_degrees: 270.0,
        };
        let after = |seconds| {
            LightUniform::carried_by(westerly, std::time::Duration::from_secs_f32(seconds))
        };
        // A westerly blows east, so the offset grows along +x and nowhere else.
        // Within a millimetre: a bearing of 270 degrees is a cosine that is not
        // quite zero, and sixty seconds of ten metres a second multiplies it up
        // to a few microns of northward drift.
        let minute = after(60.0);
        assert!(
            minute.distance(glam::Vec3::new(600.0, 0.0, 0.0)) < 1e-3,
            "a minute of a ten-metre westerly carried the sky to {minute}"
        );
        // ... and a tile's worth of it is no offset at all, however many tiles
        // have gone by.
        let tile = WEATHER_TILE / westerly.speed;
        for laps in [1.0f32, 2.0, 500.0] {
            let round = after(tile * laps);
            assert!(
                round.length() < 1.0,
                "{laps} tiles of wind left an offset of {round}"
            );
        }
        // And every fold lands where the unfolded one would, to the metre.
        for seconds in [30.0f32, 3_600.0, 86_400.0] {
            let folded = after(seconds).x;
            let whole = (10.0 * seconds).rem_euclid(WEATHER_TILE);
            assert!(
                (folded - whole).abs() < 1.0,
                "{seconds} s folds to {folded} where it should fold to {whole}"
            );
        }
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
            LightUniform::at(
                glam::Vec3::new(x, 2000.0, 0.0),
                sun,
                [0.0; 4],
                glam::Vec3::ZERO,
            )
            .origin
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

    /// Rust and the shader agree about how high the baked wind reaches.
    ///
    /// The march reads the drift field by a world height, and the number it
    /// divides by lives in `src/air.rs`. A disagreement would read the wind at
    /// the wrong altitude -- lee air where there should be a ridge's updraught,
    /// which is a föhn drawn upside down and looks like nothing in particular.
    #[test]
    fn the_march_and_the_wind_agree_on_how_high_the_air_goes() {
        assert_eq!(shader_constant("AIR_TOP"), crate::air::TOP_METRES);
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
        // One cubic kernel -- the ceiling -- and four flat ones: the march, the
        // resolve that fills in around it, and the two light columns, which are
        // one kernel written twice because only their lean differs.
        let source = include_str!("cloud_march.wgsl");
        for (group, flat, count) in [(CEILING_GROUP, CEILING_GROUP, 1), (MARCH_GROUP, 1, 4)] {
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
            // ... and the three that place a point in a light volume, which the
            // ground now reads as well as the cloud. A shadow the ground draws
            // in a different place from the one the cloud draws it is a cloud
            // floating free of its own shadow. `sun_reaching` and `sky_reaching`
            // are deliberately not in this list: they differ by one word, the
            // sampler, because the march reads three tiling fields through a
            // repeating one and the shading has the tables' clamped one already
            // to hand.
            "light_uvw",
            "beyond_light",
            "sky_share",
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
            "LIGHT_ACROSS",
            "LIGHT_SLICES",
            "BOUNCED",
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
            DECKS * 4 * 16 + 2 * 16,
            "the uniform is not the four decks and two vectors both shaders read"
        );
    }

    /// Only a deck that rides the ground shares heights with another.
    ///
    /// A sample costs one weather fetch per deck it stands in, so two decks
    /// standing in the same heights cost two -- and that is worth paying for
    /// exactly one reason: fog pools at a level the low cumulus already
    /// occupies over terrain whose valleys are at seven hundred metres and
    /// whose ridges are at two and a half thousand. Between two decks that both
    /// stand at fixed altitudes it buys nothing at all and is simply a mistake,
    /// so it is not allowed. The swing counts: a deck's base lifts and carries
    /// its top with it, so what it can occupy runs to its top plus its whole
    /// swing.
    #[test]
    fn only_a_hugging_deck_shares_heights_with_another() {
        for (deck, slab) in DECK_SLABS.iter().enumerate().skip(1) {
            let below = DECK_SLABS[deck - 1];
            assert!(
                slab[0] > below[0],
                "the decks are not in order: one starting at {} m follows one at {} m",
                slab[0],
                below[0]
            );
            if DECK_HUG[deck - 1] > 0.0 {
                continue;
            }
            assert!(
                below[1] + below[2] < slab[0],
                "a deck reaching {} m sits under one starting at {} m, and neither \
                 of them rides the ground",
                below[1] + below[2],
                slab[0]
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
