// The clouds themselves: where they can be, and what they look like.
//
// Two entry points, and the first exists to make the second affordable.
//
// `cs_cloud_ceiling` fills a coarse world grid with an upper bound on how much
// light a cell can take out of a beam crossing it. It is the same idea as the
// terrain's max pyramid and it earns its keep the same way: a ray that finds a
// cell bounded below the threshold can leave the whole cell in one step, without
// sampling anything inside it. Most of the sky is such a cell -- cloud occupies
// a couple of slabs a few kilometres thick out of the twelve the grid spans, and
// within a slab a fair-weather sky is more gap than cloud.
//
// `cs_cloud_march` walks that grid at half resolution and integrates the cloud
// it meets. The recipe is Schneider's ("The Real-time Volumetric Cloudscapes of
// Horizon Zero Dawn", SIGGRAPH 2015 Advances in Real-Time Rendering): a
// low-frequency shape thresholded by a coverage map, eroded by higher-frequency
// noise that only ever subtracts. The lighting is Wrenninge's multiple-octave
// approximation ("Oz: The Great and Volumetric", SIGGRAPH 2013 Talks) over a
// dual-lobe Henyey-Greenstein phase.
//
// What it writes is composited into the frame by `fs_shade`, which upsamples it
// bilaterally against the G-buffer's depth. This pass never touches the air
// between the eye and the cloud: the haze in front of a cloud is applied there,
// at the cloud's own distance, out of the aerial-perspective volume the frame
// already has.

const PI: f32 = 3.14159265358979;

// The planet and the two build-once tables, as `src/sky.wgsl` has them. Copied
// rather than shared because there is no preprocessor here; the functions below
// that read them are compared against `src/shading.wgsl` as text by a test.
const GROUND_RADIUS: f32 = 6360000.0;
const TOP_RADIUS: f32 = 6460000.0;
const TRANSMITTANCE_WIDTH: u32 = 256u;
const TRANSMITTANCE_HEIGHT: u32 = 64u;
const MULTISCATTER_SIZE: u32 = 32u;

// The weather map's shape. Must match `WEATHER_SIZE` and `DECKS` in
// `src/cloud.rs`.
const WEATHER_SIZE: u32 = 256u;
const DECKS: u32 = 4u;

// The ceiling cache's shape. Must match `CEILING_ACROSS`, `CEILING_SLICES` and
// `CEILING_TOP` in `src/cloud.rs`.
//
// It tiles horizontally with the weather map -- exactly two weather texels to a
// cell -- so a cell index is folded rather than clamped and the cache covers the
// whole world for a megabyte and a half. Vertically it stops at twelve
// kilometres, which is above the highest deck: a ray that climbs out of the grid
// has left the weather behind and is done.
const CEILING_ACROSS: u32 = 128u;
const CEILING_SLICES: u32 = 24u;
const CEILING_TOP: f32 = 12000.0;

// How much world one tile of each field covers, in metres. Must match
// `src/cloud.rs`.
//
// The weather is the scale of a front, the shape the scale of a cumulus, the
// detail the scale of its edge. Each is read wrapped, so these are the periods
// at which the sky repeats -- sixty kilometres is far enough that a flight does
// not fly through the same weather twice.
const WEATHER_TILE: f32 = 60000.0;
const SHAPE_TILE: f32 = 4000.0;
const DETAIL_TILE: f32 = 200.0;

// Metres of cloud per cell of the ceiling cache.
const CELL_ACROSS: f32 = WEATHER_TILE / f32(CEILING_ACROSS);
const CELL_UP: f32 = CEILING_TOP / f32(CEILING_SLICES);
// Weather texels to a cache cell, which is what the build's footprint is
// measured in.
const TEXELS_PER_CELL: i32 = i32(WEATHER_SIZE / CEILING_ACROSS);

// How high the baked wind reaches, in metres. Must match `TOP_METRES` in
// `src/air.rs`.
const AIR_TOP: f32 = 7000.0;

// The most of its own deviation a parcel is allowed to carry the cloud by, in
// metres.
//
// The bake's own bound is three times the free stream over the drift window,
// which at twenty-five metres a second is nearly seven kilometres -- a bound,
// not a typical value, and far more than a deformation should be. Clamped to
// something a cloud can be stretched by without ceasing to be where the weather
// says it is.
const MAX_DEVIATION: f32 = 600.0;

// What a parcel's recent climb is worth in cover, the climb that buys all of
// it, and how much of that the lee gets to take away again.
//
// The whole of the orographic lift, and the reason the bake carries a fourth
// channel at all: air that has just been pushed up a windward slope has cooled,
// and cloud forms in it. The same mechanism clears the lee, where the rise is
// negative and this subtracts -- which is the föhn, and it is why a range can
// have a wall of cloud on one side and blue sky on the other.
//
// It multiplies the cover rather than adding to it, and that is not a detail.
// Added, the wind could put cloud where the weather says there is none -- a cap
// over a bare ridge in clear air, which is a real thing -- but the ceiling cache
// would then have to allow for it everywhere, because the cache bounds a cell
// without reading the wind. `saturate(0 + 0.35)` is not zero, so every cell of
// every deck stopped being skippable and the march went from 0.43 ms to 0.94
// under fair weather. Multiplied, a cell with no cover has none however hard
// the air is climbing through it, the cache stays exactly as tight as it was,
// and what the wind does is thicken and thin the cloud the weather allows.
//
// The numbers were arrived at by looking. At 120 m for the full swing a
// twenty-five metre wind swept a broken sky nearly clean; the drift window is a
// minute and a half and vertical excursions of hundreds of metres are ordinary
// over this terrain. Three hundred and fifty leaves it a modulation rather than
// a switch.
//
// The lee is deliberately weaker than the windward side. Air that has risen
// makes cloud promptly; air coming back down has to warm through the cloud it
// already carries before the cloud goes, so the clearing lags the building.
const LIFT_GAIN: f32 = 1.2;
const LIFT_METRES: f32 = 350.0;
const LIFT_LEE: f32 = 0.5;

// What a unit of density takes out of a beam, per metre.
//
// A kilometre of solid cloud at density one is an optical depth of sixty, which
// is opaque several times over -- as a cumulus a kilometre deep is.
const EXTINCTION: f32 = 0.06;

// The share of what a cloud droplet takes out of a beam that it puts back.
//
// Very nearly all of it: water droplets scatter and hardly absorb, which is why
// a cloud is white rather than grey, and why what darkens a cloud base is the
// depth of cloud above it rather than any absorption in it.
const ALBEDO: f32 = 0.95;

// The largest the shape volume's first channel is anywhere in it.
//
// What makes the cache a bound rather than a guess: `carve` below is increasing
// in the field, so the largest the field can be is the largest cloud a cell can
// hold. Named rather than written as a bare one because it is a fact about the
// built volume and not about arithmetic -- and the volume was measured, which is
// how it comes to be exactly one: `perlin_worley` clamps, and it reaches the
// clamp.
const SHAPE_CEILING: f32 = 1.0;

// How hard the detail volume eats into the shape's edges.
const DETAIL_STRENGTH: f32 = 0.35;

// How far a ray is followed, in metres.
//
// The same hundred kilometres the aerial-perspective volume reaches, and for the
// same reason: past it the haze has all but saturated and what is behind it
// cannot be told from what the air in front of it is doing.
const MAX_DISTANCE: f32 = 100000.0;

// The step, as a share of how far along the ray we are, and its bounds.
//
// Proportional to distance because a step should cover about a pixel: a cone a
// hundredth of a radian wide is roughly what a half-resolution pixel subtends
// here, so a step that grows with distance keeps the sampling matched to what
// can be seen rather than spending the same effort on cloud a kilometre away and
// cloud fifty kilometres away.
const STEP_SLOPE: f32 = 0.01;
const MIN_STEP: f32 = 30.0;
const MAX_STEP: f32 = 400.0;

// How far apart two blocks' rays may be stopped and still be taken to be
// looking at the same thing, as a share of the farther of the two.
//
// What the resolve uses to decide which of the marched texels around it are
// entitled to say what it may hold. A tenth is loose enough that a hillside
// running away from the eye is one surface all the way down it, and tight
// enough that a ridge and the sky beyond it are two things -- there the ratio
// is not ten per cent but two orders of magnitude.
const TOGETHER: f32 = 0.1;

// How far outside the range of the marched texels around it a carried texel is
// allowed to sit, as a multiple of that range's own width.
const SLACK: f32 = 1.0;

// How many times round the loop before a ray gives up.
//
// Both kinds of step count against it -- a skip across an empty cell and a
// sample inside a full one -- so a ray running along the underside of a deck for
// fifty kilometres is what sets it. Running out leaves cloud undrawn at the far
// end of the ray rather than anything worse: the transmittance stored is what
// had accumulated, which is what the ray had established.
const MAX_STEPS: u32 = 256u;

// Below this transmittance the ray has established that nothing behind it will
// be seen, and stops.
//
// A millionth, and it is the sun's own disc that sets it. Everything else in a
// frame is of the order of the sky it stands against, so a hundredth would do
// and did: what survives is a per cent of a background nobody is measuring. The
// disc is `SUN_DISC_RADIANCE`, fifteen thousand times the sky, and a hundredth
// of it still tonemaps to solid white -- so an overcast deck left the sun
// shining through it as a clean white spot, which is exactly what an overcast
// deck does not do. A millionth puts the residue below the sky it sits in.
//
// It costs three or four extra steps inside cloud that was already opaque:
// going from a hundredth to a millionth is nine more of optical depth, which at
// this extinction is a hundred and fifty metres. Measured at 0.60 ms against
// 0.64 for an overcast sky, and unchanged for every other -- a thin cloud never
// reaches either threshold and stops by running out of deck.
const CUTOFF: f32 = 1e-6;

// The extinction below which a cell of the cache is treated as empty.
//
// Not zero: a cell whose bound is a millionth of a per-metre extinction would
// contribute nothing over any step this march takes, and stepping through it
// costs the same as stepping through a solid one.
const EMPTY: f32 = 1e-4;

// The light volumes' shape. Must match `LIGHT_ACROSS` and `LIGHT_SLICES` in
// `src/cloud.rs`.
//
// Two volumes of one shape: how much of the sun reaches a point, and how much
// of the sky does. Camera-centred over sixty kilometres and twelve up, which
// puts a texel at 312 m across and 250 m up -- coarse against a cloud and about
// right for what these hold, which is not the cloud but the shadow of it. A
// billow is five texels; the wisp on its edge is none, and the wisp on its edge
// casts no shadow anybody can see.
const LIGHT_ACROSS: u32 = 192u;
const LIGHT_SLICES: u32 = 48u;

// Octaves of Wrenninge's multiple-scattering approximation, and what each one
// halves.
//
// One march towards the sun gives the light that arrives unscattered. Light that
// scattered on the way in arrives from further round, having travelled through
// less cloud and been turned more gently -- so each octave takes half the
// energy, half the extinction and half the phase eccentricity of the one before
// it. Three of them is what turns a cloud from a grey silhouette into something
// that glows where it is thin.
const OCTAVES: u32 = 3u;

// How much of the sky a cloud blocks still arrives, having bounced.
//
// The sky volume says what fraction of the sky's own light reaches a point
// unscattered, and under a deck that is nothing at all -- which would make the
// underside of an overcast sky black. It is not black; it is grey, and it is
// grey because the light the deck stopped was scattered rather than absorbed
// (see `ALBEDO`, which is 0.95) and a good share of it comes out of the bottom
// anyway. Those are the scattering orders this model does not carry: the sun
// term has three octaves of them and the sky term has none, so it gets a
// constant instead.
//
// The same trade the multiple-scattering table in `src/sky.wgsl` makes for the
// atmosphere, and cruder: there the whole infinite series is summed in closed
// form, here it is one number chosen by looking at the underside of a deck.
const BOUNCED: f32 = 0.4;

// The two lobes of the phase function: a strong forward one, which is the
// silver lining when the sun is behind a cloud, and a weak backward one, which
// is why a cloud lit from behind the eye is bright rather than flat.
const HG_FORWARD: f32 = 0.8;
const HG_BACK: f32 = -0.25;

// Mirrors `CameraUniform` in `src/scene.rs`.
struct Camera {
    view_proj: mat4x4<f32>,
    was_view_proj: mat4x4<f32>,
    position: vec4<f32>,
    ray_right: vec4<f32>,
    ray_up: vec4<f32>,
    ray_forward: vec4<f32>,
};

// Mirrors `SkyUniform` in `src/sky.rs`; see `src/sky.wgsl` for what each is.
struct Sky {
    sun: vec4<f32>,
    eye: vec4<f32>,
    up: vec4<f32>,
    sun_tangent: vec4<f32>,
};

// What one deck of one preset looks like. Must match `Deck` in `src/cloud.wgsl`
// character for character: both are views of the one uniform buffer.
struct Deck {
    look: vec4<f32>,
    seed: vec4<u32>,
    slab: vec4<f32>,
    hug: vec4<f32>,
};

// Must match `Weather` in `src/cloud.wgsl`, for the same reason.
struct Weather {
    decks: array<Deck, 4>,
    clock: vec4<f32>,
    span: vec4<f32>,
};

// Where the light volumes sit and how their columns lean.
//
// Its own uniform rather than two more fields of the one above, because the
// shading pass reads this and has no business reading that: where a volume
// stands is a fact about the camera and the sun, and what is in it is a fact
// about the weather. Mirrors `LightUniform` in `src/cloud.rs` and `Light` in
// `src/shading.wgsl`.
struct Light {
    // The near corner in x and z, the height of the lowest slice, and the
    // metres one texel covers across.
    origin: vec4<f32>,
    // How far a sun column drifts horizontally per metre it climbs, then the
    // metres of ray one slice is worth walking towards the sun, then the metres
    // of height one slice is worth -- which is what the same slice is worth
    // walking straight up.
    walk: vec4<f32>,
    // Where the baked wind's own grid stands: its near corner in x and z, then
    // how much world it covers in each. Zero-sized until the wind has been
    // solved, which is what says there is no field to read yet.
    air: vec4<f32>,
    // How far the weather has been carried by the mean wind since the world
    // started, in metres. Folded into one tile of the map, so a long flight
    // cannot walk it out of the range an `f32` holds a lattice index in.
    carried: vec4<f32>,
};

// Which quarter of the resolved buffer this frame actually marches.
//
// Mirrors `RotationUniform` in `src/cloud.rs`.
struct Rotation {
    // In `xy`, the sub-position of the two-by-two block whose ray is marched
    // this frame; in `zw`, the size of the resolved buffer the other three
    // quarters are carried into.
    at: vec4<u32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> sky: Sky;

// The two build-once scattering tables, without the per-frame ones beside them.
// The sky-view table and the aerial-perspective volumes are deliberately absent:
// the haze in front of a cloud is applied where the cloud is composited, not
// here, and leaving them out is what keeps this pass inside the sampled-texture
// budget.
@group(2) @binding(0) var lut_sampler: sampler;
@group(2) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(2) var multiscatter_lut: texture_2d<f32>;

@group(3) @binding(0) var<uniform> cloud: Weather;
@group(3) @binding(1) var weather_map: texture_2d_array<f32>;
@group(3) @binding(2) var shape_noise: texture_3d<f32>;
@group(3) @binding(3) var detail_noise: texture_3d<f32>;
@group(3) @binding(4) var ceiling: texture_3d<f32>;
@group(3) @binding(5) var depth: texture_2d<f32>;
// Repeating in every axis: all three fields above tile, and reading them wrapped
// is the whole reason eight megabytes of noise covers a world.
@group(3) @binding(6) var tile_sampler: sampler;
@group(3) @binding(7) var out_cloud: texture_storage_2d<rgba16float, write>;
@group(3) @binding(8) var out_cloud_depth: texture_storage_2d<r32float, write>;
@group(3) @binding(9) var out_ceiling: texture_storage_3d<r32float, write>;
// Whichever of the two light volumes is being filled. One binding and one
// layout for both, because the only thing that differs between them is which
// way the column walks -- see `cs_cloud_sun_light` and `cs_cloud_sky_light`.
@group(3) @binding(10) var out_light: texture_storage_3d<rgba16float, write>;
@group(3) @binding(11) var sun_light: texture_3d<f32>;
@group(3) @binding(12) var sky_light: texture_3d<f32>;
// Clamped rather than repeating: these do not tile. A point outside reads the
// nearest column, which is a real transmittance from a real place and the most
// plausible thing to continue with -- and everything that far out is behind
// most of a hundred kilometres of haze.
@group(3) @binding(13) var edge_sampler: sampler;
@group(3) @binding(14) var<uniform> light: Light;
// How far the air arriving at each cell of the baked grid has strayed from the
// bulk drift, and in `w` how far it has climbed to get there. Solved once at
// load around the actual mountains; see `src/air.rs`.
@group(3) @binding(15) var air_drift: texture_3d<f32>;
@group(3) @binding(16) var<uniform> rotation: Rotation;
// This frame's march, one texel per two-by-two block of the resolved buffer,
// and the resolved buffer the last frame left. Both read by `cs_cloud_resolve`
// and by nothing else.
//
// The fresh pair is loaded rather than filtered: each texel is one ray's
// answer at one place, and the resolve wants those places exactly. The history
// colour is filtered, because it is read at wherever this texel used to be,
// which is not a texel centre.
@group(3) @binding(17) var fresh_cloud: texture_2d<f32>;
@group(3) @binding(18) var fresh_along: texture_2d<f32>;
@group(3) @binding(19) var was_cloud: texture_2d<f32>;
@group(3) @binding(20) var was_along: texture_2d<f32>;
// The ground under every column of the baked wind's own grid, in metres. Read
// only by a deck whose base rides it; see `ground_at`.
@group(3) @binding(21) var air_ground: texture_2d<f32>;

// The ray through a point on the screen, before it is normalised. Must match
// `ray_raw_at` in `src/shading.wgsl` and `src/terrain.wgsl`, character for
// character; there is a test comparing them as text.
fn ray_raw_at(screen: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(depth));
    let ndc = vec2<f32>(
        screen.x / size.x * 2.0 - 1.0,
        1.0 - screen.y / size.y * 2.0,
    );
    return camera.ray_right.xyz * ndc.x + camera.ray_up.xyz * ndc.y + camera.ray_forward.xyz;
}

// How far along `ray_raw_at` a reversed-Z depth puts a point. Must match
// `distance_at` in `src/terrain.wgsl`; `ray_forward.w` carries the near plane.
fn distance_at(d: f32) -> f32 {
    return camera.ray_forward.w / d;
}

// The half-texel correction. Must match `to_texture` in `src/sky.wgsl`.
fn to_texture(x: f32, n: f32) -> f32 {
    return 0.5 / n + x * (1.0 - 1.0 / n);
}

fn top_distance(r: f32, mu: f32) -> f32 {
    let discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
    return max(-r * mu + sqrt(max(discriminant, 0.0)), 0.0);
}

// Must match `transmittance_uv` in `src/sky.wgsl`.
fn transmittance_uv(r: f32, mu: f32) -> vec2<f32> {
    let horizon = sqrt(max(r * r - GROUND_RADIUS * GROUND_RADIUS, 0.0));
    let atmosphere = sqrt(max(TOP_RADIUS * TOP_RADIUS - GROUND_RADIUS * GROUND_RADIUS, 0.0));
    let distance = top_distance(r, mu);
    let shortest = TOP_RADIUS - r;
    let longest = horizon + atmosphere;
    let span = max(longest - shortest, 1e-6);
    return vec2<f32>(
        to_texture(clamp((distance - shortest) / span, 0.0, 1.0), f32(TRANSMITTANCE_WIDTH)),
        to_texture(clamp(horizon / atmosphere, 0.0, 1.0), f32(TRANSMITTANCE_HEIGHT)),
    );
}

fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    return textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(r, mu), 0.0).rgb;
}

// Must match `multiscatter_uv` in `src/sky.wgsl`.
fn sample_multiscatter(r: f32, mu_s: f32) -> vec3<f32> {
    let altitude = clamp((r - GROUND_RADIUS) / (TOP_RADIUS - GROUND_RADIUS), 0.0, 1.0);
    let uv = vec2<f32>(
        to_texture(clamp(mu_s * 0.5 + 0.5, 0.0, 1.0), f32(MULTISCATTER_SIZE)),
        to_texture(altitude, f32(MULTISCATTER_SIZE)),
    );
    return textureSampleLevel(multiscatter_lut, lut_sampler, uv, 0.0).rgb;
}

// Distance to the ground, or a negative number if the ray misses it. Must match
// `ground_distance` in `src/sky.wgsl`.
//
// A ray pointing at all upwards cannot meet the ground however the arithmetic
// comes out, which is why `mu > 0` is rejected outright rather than left to the
// discriminant: near the horizon the two roots are close and rounding decides.
fn ground_distance(r: f32, mu: f32) -> f32 {
    if mu > 0.0 {
        return -1.0;
    }
    let discriminant = r * r * (mu * mu - 1.0) + GROUND_RADIUS * GROUND_RADIUS;
    if discriminant < 0.0 {
        return -1.0;
    }
    return -r * mu - sqrt(discriminant);
}

// Henyey-Greenstein, normalised over the sphere. The same function
// `mie_phase` in `src/sky.wgsl` is, with the asymmetry a parameter: the cloud
// wants it at two eccentricities at once and at three scales of each.
fn henyey(cos_theta: f32, g: f32) -> f32 {
    let gg = g * g;
    let denominator = 1.0 + gg - 2.0 * g * cos_theta;
    return (1.0 - gg) / (4.0 * PI * denominator * sqrt(max(denominator, 1e-8)));
}

// How much of a deck's slab is filled, at a height fraction through it.
//
// Nothing at the base, rising to solid, holding, then falling away to nothing at
// the top -- and where it rises and falls is what tells one kind of cloud from
// another. Flat stratus fills from just above its base and holds nearly to its
// top; heaped cumulus is narrow underneath and rounds off well below it, which
// is what leaves a cauliflower rather than a sheet once the coverage map has cut
// holes in it.
//
// Unimodal by construction, with a plateau rather than a peak, and the march
// below is not the only caller that depends on it: the ceiling cache takes the
// largest value this can reach over a range of heights, and it does that with a
// single evaluation because the function has exactly one flat top. See
// `vertical_peak`.
fn vertical(h: f32, lean: f32) -> f32 {
    let rise = mix(0.10, 0.35, lean);
    let fall = mix(0.88, 0.60, lean);
    return smoothstep(0.0, rise, h) * (1.0 - smoothstep(fall, 1.0, h));
}

// A height fraction at which `vertical` is at its largest, whatever the lean.
//
// The middle of the plateau, so clamping it into any range gives the point of
// that range where `vertical` is largest: inside the plateau if the range
// reaches it, and otherwise the end nearest to it, which is where a function
// that rises to the plateau and falls from it takes its largest value.
fn vertical_peak(lean: f32) -> f32 {
    return 0.5 * (mix(0.10, 0.35, lean) + mix(0.88, 0.60, lean));
}

// Whether a preset has switched a deck off altogether.
//
// A deck with no cloud in it is described by a cover ramp whose lower end is
// above one, which the field cannot reach -- see `Preset::decks` in
// `src/cloud.rs`, where that is what `clear` is. So its layer of the weather map
// is exactly zero everywhere, and this is what says so without fetching it.
//
// Cheap and worth more than it looks. Every preset but `storm` switches at
// least one deck off, and the fog shares its heights with the low cumulus, so
// without this every sample low in the sky would pay a fetch to be told there
// is no fog today.
fn empty_deck(deck: u32) -> bool {
    return cloud.decks[deck].look.x >= 1.0;
}

// Whether a deck can put cloud at a height at all.
//
// A deck's base lifts by up to its swing, carrying its top with it, so the most
// it can occupy is from its nominal base to its nominal top plus that swing.
//
// Decks may overlap, and the fog and the low cumulus over this terrain do: the
// valleys here stand at seven hundred metres and the ridges at two and a half
// thousand, so fog pools at a level the cumulus deck already occupies. What
// that costs is a weather fetch for each deck a sample stands in rather than
// one for the sample -- and it buys the only arrangement in which both are the
// height they should be. It is also the physical answer: two things in the air
// at the same place add up.

// The weather over a point, for one deck: cover, lean, density, base offset.
fn weather_at(deck: i32, p: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(weather_map, tile_sampler, p.xz / WEATHER_TILE, deck, 0.0);
}

// How much of the noise field is allowed to become cloud here.
//
// The coverage map's own figure, narrowed by where in the deck's slab this point
// sits. Zero is the answer almost everywhere -- above the deck, below it, and
// wherever the front has not reached -- and zero is what the ceiling cache is
// built to find in bulk.
// The cover a patch has, once the ground under it has had its say.
//
// Air pushed up a windward slope cools, and cloud forms in it; air coming back
// down the lee warms, and cloud in it goes away. That is one number out of the
// bake -- how far the parcel arriving here has climbed in the last minute and a
// half -- and it is the whole of the föhn.
//
// Clamped both ways, and the upper clamp is what the ceiling cache stands on:
// it bounds a cell without reading the wind, which it can only do because the
// most the wind can add is written down. See `cell_bound`.
fn covered(cover: f32, risen: f32) -> f32 {
    let lift = clamp(risen / LIFT_METRES, -LIFT_LEE, 1.0);
    return saturate(cover * (1.0 + LIFT_GAIN * lift));
}

// Where in the baked wind's grid a world point sits.
fn air_uvw(p: vec3<f32>) -> vec3<f32> {
    let flat = (p.xz - light.air.xy) / max(light.air.zw, vec2<f32>(1.0));
    return vec3<f32>(flat.x, clamp(p.y / AIR_TOP, 0.0, 1.0), flat.y);
}

// How far the air arriving at a point has strayed, and how far it has climbed.
//
// Zero outside the solved grid, and faded to zero at its sides rather than
// clamped: the grid covers the raster and the march runs a hundred kilometres,
// so most of a level ray is outside it, and a wall of edge values would draw a
// straight line of cloud across the sky at the survey's boundary.
//
// Zero before the bake has run as well. The grid is zero-sized until then and
// the guard above turns that into a coordinate far outside, which the fade
// takes to nothing -- so the frames drawn while the terrain is still loading
// have no wind in them rather than a division by no extent.
fn air_at(p: vec3<f32>) -> vec4<f32> {
    if light.air.z <= 0.0 {
        return vec4<f32>(0.0);
    }
    let uvw = air_uvw(p);
    let out = abs(uvw.xz - vec2<f32>(0.5)) * 2.0;
    let inside = 1.0 - smoothstep(0.9, 1.0, max(out.x, out.y));
    return textureSampleLevel(air_drift, edge_sampler, uvw, 0.0) * inside;
}

// How far above the threshold the shape field has to climb before the cloud it
// makes is solid, in field units.
//
// What sets how sharp a cloud edge is. Schneider's own remap divides by the
// coverage instead, which stretches whatever is above the threshold across the
// whole range -- so a patch with a tenth of the sky covered has edges ten times
// softer than a patch that is solid, which is backwards, and the bound the cache
// takes from it collapses to "there is some cover here" and says nothing about
// how much. A fixed width is both the better-behaved knob and the one that
// leaves the cache something continuous to bound.
const EDGE: f32 = 0.3;

// The shape field thresholded by a coverage.
//
// Where the field climbs above `1 - coverage` there is cloud, and `EDGE` above
// that it is solid. Increasing in both arguments, which is the property the
// cache depends on: the largest cloud a cell can hold is this evaluated at the
// largest coverage in the cell and the largest the field can ever reach.
fn carve(field: f32, coverage: f32) -> f32 {
    return saturate((field - 1.0 + coverage) / EDGE);
}

// What a beam loses per metre at a point, and how much of that is a guess.
//
// `fine` is off for the samples taken along the way to the sun, where the two
// erosions are left out. They only ever subtract, so leaving them out reports
// more cloud between a sample and the sun than there is, which errs towards a
// darker cloud base -- and it halves the cost of the most expensive part of the
// march.
fn cloud_extinction(p: vec3<f32>, fine: bool) -> f32 {
    var total = 0.0;
    for (var deck = 0u; deck < DECKS; deck += 1u) {
        if empty_deck(deck) {
            continue;
        }
        let slab = cloud.decks[deck].slab;
        if p.y < slab.x || p.y > slab.y + slab.z {
            continue;
        }
        total = total + deck_extinction(deck, p, fine);
    }
    return total;
}

// The same, for one deck a point is known to stand in.
fn deck_extinction(deck: u32, p: vec3<f32>, fine: bool) -> f32 {
    // The weather is carried bodily by the mean wind and by nothing else. That
    // is not a simplification for its own sake: the ceiling cache bounds the
    // coverage of a cell without reading the wind, and it can carry a bulk
    // offset exactly where it could not carry a field that varies from cell to
    // cell. A front moves; it does not wrap itself around a mountain.
    let front = p - light.carried.xyz;
    let w = weather_at(i32(deck), front);

    // Where in its slab this point sits, before the wind has had a say. The
    // wind's say is a *multiplier*, so nought times it is still nought -- which
    // is what lets the drift be fetched here rather than above, after the great
    // majority of samples have already turned out to be empty sky. Moving this
    // one fetch below this one test is worth 0.11 ms of the march.
    let slab = cloud.decks[deck].slab;
    var base = slab.x + w.a * slab.z;
    var thickness = max(slab.y - slab.x, 1.0);
    // A deck that rides the ground keeps its top and takes its base from the
    // terrain, so it pools rather than blankets: it fills a valley to a level
    // and leaves the ridges above that level in clear air. Where the ground is
    // already above the top there is no deck at all, which `max` and the
    // clamped thickness between them say.
    //
    // Branched rather than folded into the arithmetic above, which would be
    // shorter: `ground_at` is a fetch, no deck but the fog has any use for it,
    // and the same reasoning put the drift fetch below the cover test.
    let hug = cloud.decks[deck].hug.x;
    if hug > 0.0 {
        let top = slab.y + w.a * slab.z;
        base = mix(base, max(base, ground_at(p)), hug);
        thickness = max(mix(slab.y - slab.x, top - base, hug), 1.0);
    }
    let profile = vertical((p.y - base) / thickness, w.g);
    if w.r * profile <= 0.0 {
        return 0.0;
    }

    let air = air_at(p);
    let coverage = covered(w.r, air.w) * profile;
    if coverage <= 0.0 {
        return 0.0;
    }

    // The cloud's own structure is carried *and* deformed. The deviation is
    // what the bake was for: it stretches a field through a valley and piles it
    // against a slope, and it may move the noise as far as it likes without
    // costing the cache anything, because where the shape is has no bearing on
    // how much cloud a cell is allowed to hold.
    let strayed = clamp_deviation(air.xyz);
    let taken = front - strayed;
    let shape = textureSampleLevel(shape_noise, tile_sampler, taken / SHAPE_TILE, 0.0);
    var density = carve(shape.r, coverage);
    if density <= 0.0 {
        return 0.0;
    }

    if fine {
        // The detail volume, as a remap that can only take cloud away -- which
        // is what keeps the cache's bound a bound. Wispy at the bottom of the
        // deck and billowy at the top, which is Schneider's own inversion: an
        // updraft shreds a cloud where it enters and heaps it where it stops.
        //
        // The shape's own three Worley channels are not read. They were built
        // to erode this field at a scale between the billows and the detail,
        // and the field wants no such thing yet: what is missing from a cloud
        // here is light inside it, not another frequency of hole in it. They
        // cost nothing to leave in the volume they are already in.
        let detail = textureSampleLevel(detail_noise, tile_sampler, taken / DETAIL_TILE, 0.0);
        let h = saturate((p.y - base) / thickness);
        let wisp = mix(detail.a, 1.0 - detail.a, saturate(h * 5.0));
        let eaten = wisp * DETAIL_STRENGTH;
        density = saturate((density - eaten) / max(1.0 - eaten, 1e-3));
    }

    return EXTINCTION * w.b * cloud.decks[deck].slab.w * density;
}

// How high the ground is under a point, in metres.
//
// Sampled from the lattice the wind was solved on, which is the one place the
// terrain's own height mirror has already been read into something a shader can
// address. That is a couple of hundred metres to a texel over this raster --
// coarse against a hillside, and about right for what it is for: the top of a
// pool of fog is a soft thing, and a shoreline drawn to the nearest hundred
// metres is not what anyone looks at.
//
// Sea level outside the grid and before it is solved, which is the honest
// answer in both cases: there is no terrain out there, and a deck that rides
// the ground rides nothing.
fn ground_at(p: vec3<f32>) -> f32 {
    if light.air.z <= 0.0 || light.air.w <= 0.0 {
        return 0.0;
    }
    let uv = (p.xz - light.air.xy) / light.air.zw;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 0.0;
    }
    return textureSampleLevel(air_ground, edge_sampler, uv, 0.0).r;
}

// A deviation, cut down to something a cloud can be stretched by.
fn clamp_deviation(strayed: vec3<f32>) -> vec3<f32> {
    let far = length(strayed);
    return strayed * min(far, MAX_DEVIATION) / max(far, 1e-3);
}

// The largest extinction a cell of the cache may have to bound, from one texel
// of the weather over one range of heights.
//
// The wind is not read here and does not need to be. What it does to the cover
// is bounded by `LIFT_COVER`, so the most a cell can hold is what it would hold
// with the strongest updraught the model allows over it -- and what it does to
// the shape does not enter, because this bounds the field at its own ceiling
// wherever the field is sampled from.
fn cell_bound(deck: u32, w: vec4<f32>, low: f32, high: f32) -> f32 {
    let slab = cloud.decks[deck].slab;
    let base = slab.x + w.a * slab.z;
    let thickness = max(slab.y - slab.x, 1.0);
    var reached = 1.0;
    if cloud.decks[deck].hug.x <= 0.0 {
        // Where in this cell's heights the deck's profile comes nearest to
        // filling, which for most of the sky is nowhere near it.
        let peak = clamp(
            vertical_peak(w.g),
            (low - base) / thickness,
            (high - base) / thickness,
        );
        reached = vertical(peak, w.g);
    } else {
        // A hugging deck's profile is measured from ground the cache cannot
        // read and must not try to. The cache tiles every sixty kilometres and
        // the ground does not, so one cell of it stands over every terrain in
        // the world at once -- there is no ground to measure from. What is left
        // is the honest bound: a deck that rides the ground may be filling
        // anywhere inside its own band.
        //
        // The clamp above cannot stand in for this, and the reason is worth
        // keeping. Riding the ground only ever *lowers* the height fraction a
        // point sits at -- the base rises towards the point while the top stays
        // put -- so a cell that reaches the plateau is bounded either way. A
        // cell entirely above the plateau is not: measured from sea level its
        // fraction is past the falling edge and the clamp reports the profile
        // on the way down, while the same point measured from ground just below
        // it sits in the middle of a thin pool and is filled. A band of five
        // hundred and twenty metres would do it -- the cell at five hundred is
        // then at 0.96 of the way up, where the clamp says 0.15 and the truth
        // is 1.0, and every second cache cell would cut a hole in the fog.
        //
        // The fog's band is a kilometre and a quarter, so no cell of the cache
        // reaches past its plateau and the two agree today. They agree by
        // arithmetic rather than by rule, which is not a thing to build on.
        //
        // What the rule costs is the band and nothing else, and only when there
        // is fog in the forecast: above the cover ramp's reach `w.r` is a hard
        // zero and this is zero with it.
        reached = vertical(vertical_peak(w.g), w.g);
    }
    let coverage = covered(w.r, LIFT_METRES) * reached;
    return EXTINCTION * w.b * slab.w * carve(SHAPE_CEILING, coverage);
}

// A weather texel index folded back into the map, which tiles.
fn wrap_texel(at: vec2<i32>) -> vec2<i32> {
    let size = i32(WEATHER_SIZE);
    return ((at % size) + size) % size;
}

// An upper bound on the extinction anywhere in each cell of a coarse world grid.
//
// Rebuilt every frame, because the weather it is built from moves. It is cheap
// for the reason the weather map is: what it reads is a quarter-million texels
// describing a whole sky, not the sky itself.
//
// The footprint is the two weather texels the cell covers in each axis, one
// either side, and one more for the fraction of a texel the wind has carried
// the map by. The margin is not caution: the march samples the weather map
// bilinearly, so a point just inside a cell reads texels just outside it, and a
// bound taken over the covered texels alone would be a bound on something the
// march never asks for.
@compute @workgroup_size(4, 4, 4)
fn cs_cloud_ceiling(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= CEILING_ACROSS || id.y >= CEILING_SLICES || id.z >= CEILING_ACROSS {
        return;
    }
    let low = f32(id.y) * CELL_UP;
    let high = low + CELL_UP;
    // Where this cell's ground was before the wind carried the weather over it.
    // The offset is a whole number of nothing in particular, so the footprint
    // starts at a texel found by division rather than by multiplying the cell
    // index, and it is one wider on each side to cover the fraction.
    let across = WEATHER_TILE / f32(WEATHER_SIZE);
    let corner = vec2<f32>(vec2<u32>(id.x, id.z)) * CELL_ACROSS - light.carried.xz;
    let first = vec2<i32>(floor(corner / across)) - 1;

    var bound = 0.0;
    for (var deck = 0u; deck < DECKS; deck += 1u) {
        if empty_deck(deck) {
            continue;
        }
        let slab = cloud.decks[deck].slab;
        // A deck that does not reach into these heights cannot put cloud in
        // them, whatever the weather over them says.
        if high < slab.x || low > slab.y + slab.z {
            continue;
        }
        var worst = 0.0;
        for (var j = 0; j <= TEXELS_PER_CELL + 2; j += 1) {
            for (var i = 0; i <= TEXELS_PER_CELL + 2; i += 1) {
                let at = wrap_texel(first + vec2<i32>(i, j));
                let w = textureLoad(weather_map, at, i32(deck), 0);
                worst = max(worst, cell_bound(deck, w, low, high));
            }
        }
        // Summed across decks and maxed across the map. Two decks may stand in
        // one cell -- the fog and the low cumulus do over this terrain -- and
        // the march adds what it finds in each, so a bound on the pair has to
        // add too. Where they do not overlap only one term is ever non-zero and
        // this is the maximum it was.
        bound = bound + worst;
    }
    textureStore(out_ceiling, vec3<i32>(id), vec4<f32>(bound, 0.0, 0.0, 0.0));
}

// The bound on the cell a point stands in, or nothing above the grid.
//
// Loaded, never filtered. Interpolating between maxima returns something below
// the true maximum of the cell a sample is actually in, which is a hole in a
// cloud -- the same reason `ceiling_at` in `src/terrain.wgsl` loads rather than
// samples the terrain's own pyramid.
fn ceiling_at(p: vec3<f32>) -> f32 {
    if p.y < 0.0 || p.y >= CEILING_TOP {
        return 0.0;
    }
    let across = i32(CEILING_ACROSS);
    let column = vec2<i32>(floor(p.xz / CELL_ACROSS));
    let folded = ((column % across) + across) % across;
    let slice = i32(p.y / CELL_UP);
    return textureLoad(ceiling, vec3<i32>(folded.x, slice, folded.y), 0).r;
}

// How far it is to the far side of the cell a point stands in.
//
// A slab test per axis rather than a stepped digital line, because the grid is
// uniform: the exit is the nearest of three plane crossings and the cell indices
// never have to be carried between iterations.
fn cell_exit(p: vec3<f32>, direction: vec3<f32>) -> f32 {
    let size = vec3<f32>(CELL_ACROSS, CELL_UP, CELL_ACROSS);
    let low = floor(p / size) * size;
    let towards = select(low, low + size, direction > vec3<f32>(0.0));
    let crossing = (towards - p) / direction;
    // An axis the ray does not move along is never left through, so it must not
    // be allowed to win the minimum with whatever the division produced.
    let usable = select(vec3<f32>(1e9), crossing, abs(direction) > vec3<f32>(1e-6));
    return max(min(min(usable.x, usable.y), usable.z), 0.0);
}

// What arrives at a sample from the sun, over three scales of scattering.
//
// `tau` is the optical depth between the sample and the sun. Octave zero is the
// light that got here unscattered; each after it stands for light that scattered
// on the way, which arrives having crossed less cloud and been turned less
// sharply. Wrenninge's approximation, and the difference between a cloud that
// glows and a cloud that is a grey cut-out.
fn sunlit(through: f32, cos_theta: f32, sunlight: vec3<f32>) -> vec3<f32> {
    // The octaves take a share of the extinction, which on a transmittance
    // already exponentiated is a power rather than a scaled exponent. Floored,
    // because zero to the power of a half is zero but zero read out of a half
    // float and raised to a power is a place a NaN can start.
    let survived = max(through, 1e-6);
    var sum = vec3<f32>(0.0);
    var energy = 1.0;
    var depth = 1.0;
    var eccentricity = 1.0;
    for (var octave = 0u; octave < OCTAVES; octave += 1u) {
        let phase = mix(
            henyey(cos_theta, HG_FORWARD * eccentricity),
            henyey(cos_theta, HG_BACK * eccentricity),
            0.5,
        );
        sum = sum + sunlight * (energy * phase * pow(survived, depth));
        energy = energy * 0.5;
        depth = depth * 0.5;
        eccentricity = eccentricity * 0.5;
    }
    return sum;
}

// Where a light volume's texel sits in the world.
//
// The whole of the shear. A column of the volume -- one `(x, z)`, every slice --
// is a straight line in the world that leans by `shear` for every metre it
// climbs. Give it the sun's own lean and the column *is* a sun ray, so walking
// it top to bottom accumulates exactly the cloud between each point and the
// sun, once, in one pass. Give it no lean and the column is a plumb line, and
// the same walk accumulates the cloud between each point and the sky.
//
// Indexed by world x and z rather than by anything in the sun's frame, so a
// texel covers the same ground however low the sun gets. A sun-space box
// degenerates exactly where it is most wanted: at five degrees its own plane is
// nearly vertical, which puts a hundred and fifty metres of resolution across
// the two and a half kilometres a cloud deck occupies -- seventeen useful texels
// over the whole interesting half of the sun's arc.
fn light_position(at: vec3<u32>, shear: vec2<f32>) -> vec3<f32> {
    let climbed = (f32(at.z) + 0.5) * light.walk.w;
    let flat = light.origin.xy
        + (vec2<f32>(at.xy) + 0.5) * light.origin.w
        + climbed * shear;
    return vec3<f32>(flat.x, light.origin.z + climbed, flat.y);
}

// The same mapping backwards: where a world point sits in a light volume.
fn light_uvw(p: vec3<f32>, shear: vec2<f32>) -> vec3<f32> {
    let climbed = p.y - light.origin.z;
    let flat = p.xz - light.origin.xy - climbed * shear;
    return vec3<f32>(
        flat / (light.origin.w * f32(LIGHT_ACROSS)),
        climbed / (light.walk.w * f32(LIGHT_SLICES)),
    );
}

// How far outside a light volume a coordinate has strayed, across.
//
// Zero anywhere inside, one at a side, and more beyond. What is beyond is not a
// rare case: the sun columns lean, and the lower the sun the further they lean,
// so a point high above a low sun un-shears to somewhere tens of kilometres
// outside a volume sixty across. Letting the sampler clamp there is what put
// horizontal stripes across a dusk sky -- every slice above four kilometres
// read the same edge column, so the transmittance stepped from slice to slice
// instead of varying along the ray.
//
// So the answer fades to full light over the outermost tenth instead. Nothing
// is lost by it: the sun that leans a column that far is a sun the air has
// already taken out, and the fade is smooth where the clamp was a staircase.
//
// Across only, and that is the point of the `.xy`. The volume's floor is sea
// level and its ceiling is above every deck, so a point outside it vertically
// wants the nearest slice and not a fade -- and the ground, which is the whole
// reason any of this is read twice, sits *on* the floor. Fading there was worth
// a whole commit's effort producing no shadow at all on any terrain.
fn beyond_light(uvw: vec3<f32>) -> f32 {
    let out = abs(uvw.xy - vec2<f32>(0.5)) * 2.0;
    return max(out.x, out.y);
}

// How much of the sun reaches a point, and how much of the sky does.
//
// No half-texel correction and none wanted: a texel of these stands for the
// point at its own centre rather than for a sample of a function over a range,
// so `light_position` and this are already inverses. See `to_texture` in
// `src/sky.wgsl` for the case where the correction is needed.
fn sun_reaching(p: vec3<f32>) -> f32 {
    let uvw = light_uvw(p, light.walk.xy);
    let blocked = textureSampleLevel(sun_light, edge_sampler, uvw, 0.0).r;
    // The fade to full light is a fade of what is blocked to nothing, which is
    // the same arithmetic and leaves an empty volume returning exactly one.
    return 1.0 - blocked * (1.0 - smoothstep(0.9, 1.0, beyond_light(uvw)));
}

fn sky_reaching(p: vec3<f32>) -> f32 {
    let uvw = light_uvw(p, vec2<f32>(0.0));
    let blocked = textureSampleLevel(sky_light, edge_sampler, uvw, 0.0).r;
    return 1.0 - blocked * (1.0 - smoothstep(0.9, 1.0, beyond_light(uvw)));
}

// What a point gets of the sky, given how much of it reaches there unscattered.
//
// Written so that nothing blocked leaves the answer exactly one, for the reason
// above. See `BOUNCED`.
fn sky_share(reaching: f32) -> f32 {
    return 1.0 - (1.0 - BOUNCED) * (1.0 - reaching);
}

// One column of a light volume, walked from the top down.
//
// The cost of the whole technique is here and it is one density sample per
// texel: a thread owns a column and carries the running integral down it, so
// what would be a march per shaded sample becomes a fetch per shaded sample.
// Structurally `cs_aerial` in `src/sky.wgsl`, which walks the frustum's froxel
// columns the same way and for the same reason.
fn walk_light(id: vec3<u32>, shear: vec2<f32>, metres: f32) {
    var tau = 0.0;
    for (var slice = LIGHT_SLICES; slice > 0u; slice -= 1u) {
        let at = vec3<u32>(id.x, id.y, slice - 1u);
        let extinction = cloud_extinction(light_position(at, shear), false);
        let crossed = extinction * metres;
        tau = tau + crossed;
        // What is *blocked*, not what gets through, and that is not a matter of
        // taste. A frame with no cloud in it has to draw the ground exactly as
        // it drew it before there were clouds, and a volume of ones does not
        // filter back to one: the hardware's weights are worth about sixteen
        // bits, so eight identical texels of 1.0 come back a hundred-thousandth
        // short, and a per-cent-of-a-per-cent on the sunlight moves the odd byte
        // of a frame that was supposed to be untouched. Zeroes filter to zero
        // whatever the weights are worth.
        //
        // Stored for the texel's own centre, so half of its own cell is not yet
        // in front of it. Without that the volume reads half a cell dark
        // everywhere, which is a deck's own shadow cast onto its own top.
        textureStore(
            out_light,
            vec3<i32>(at),
            vec4<f32>(1.0 - exp(crossed * 0.5 - tau), 0.0, 0.0, 1.0),
        );
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_cloud_sun_light(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= LIGHT_ACROSS || id.y >= LIGHT_ACROSS {
        return;
    }
    walk_light(id, light.walk.xy, light.walk.z);
}

@compute @workgroup_size(8, 8, 1)
fn cs_cloud_sky_light(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= LIGHT_ACROSS || id.y >= LIGHT_ACROSS {
        return;
    }
    // No lean, and a slice is worth its own height rather than the longer ray a
    // leaning one crosses.
    walk_light(id, vec2<f32>(0.0), light.walk.w);
}

// How far the ground lets a two-by-two block of pixels see, along the view
// axis and in metres.
//
// The farthest of the block's four depths, which is what `cs_cloud_march` takes
// and for the reason given there. Along the view axis rather than along the ray
// -- one axis for every block, so two blocks' answers can be compared, which is
// the only thing this is used for. `cs_cloud_march` wants the same number in
// metres of its own ray instead, and scales as it goes.
fn reach_of(block: vec2<i32>) -> f32 {
    let full = vec2<i32>(textureDimensions(depth));
    var reach = 0.0;
    for (var j = 0; j < 2; j += 1) {
        for (var i = 0; i < 2; i += 1) {
            let d = textureLoad(depth, min(block + vec2<i32>(i, j), full - 1), 0).r;
            if d == 0.0 {
                reach = MAX_DISTANCE;
            } else {
                reach = max(reach, min(distance_at(d), MAX_DISTANCE));
            }
        }
    }
    return reach;
}

// Where a ray enters and leaves a horizontal slab, as distances along it.
//
// Returns an empty range -- the far end before the near one -- when the ray
// never reaches the slab at all, which is what a level ray above the cloud does.
fn slab_range(y: f32, dy: f32, low: f32, high: f32) -> vec2<f32> {
    if abs(dy) < 1e-6 {
        if y < low || y > high {
            return vec2<f32>(1.0, 0.0);
        }
        return vec2<f32>(0.0, 1e9);
    }
    let first = (low - y) / dy;
    let second = (high - y) / dy;
    return vec2<f32>(min(first, second), max(first, second));
}

// Which texel of a two-by-two block of the resolved buffer `block` marches.
//
// Clamped, so the last block of an odd-sized buffer marches a ray that is on
// the screen rather than one texel off the end of it. Both the march and the
// resolve ask this, which is why it is a function: they have to agree about
// where this frame's answers landed or the resolve carries the wrong texel.
fn marched_texel(block: vec2<i32>) -> vec2<i32> {
    let last = vec2<i32>(rotation.at.zw) - 1;
    return min(block * 2 + vec2<i32>(rotation.at.xy), last);
}

// The clouds in front of one texel in four, at half resolution.
//
// Half because the field is smooth and the frame is not: a cloud edge is soft
// over metres where a mountain silhouette is sharp over a pixel, so what is lost
// by marching one ray per two-by-two block is far less than what is saved. What
// clips the ray at the near end of that trade is the G-buffer's own depth, which
// makes the terrain's occlusion of cloud exact and free.
//
// One texel in four because the answer keeps. A cloud is a slow, smooth thing
// and the camera moves a few metres a frame, so a texel marched three frames
// ago and carried through the camera's own motion is very nearly the texel
// marched now -- and it costs a fetch rather than a march. This dispatch is a
// quarter the size of the buffer it fills; `cs_cloud_resolve` fills the rest.
@compute @workgroup_size(8, 8, 1)
fn cs_cloud_march(@builtin(global_invocation_id) id: vec3<u32>) {
    let blocks = textureDimensions(out_cloud);
    if id.x >= blocks.x || id.y >= blocks.y {
        return;
    }
    let at = vec2<i32>(id.xy);
    let block = marched_texel(at) * 2;
    let full = vec2<i32>(textureDimensions(depth));

    // The ray through the corner where the block's four pixels meet, which is
    // the direction their four rays average to.
    let raw = ray_raw_at(vec2<f32>(block) + 1.0);
    let per_step = length(raw);
    let direction = raw / per_step;

    // How far the ray may run before the ground stops it. The four depths of the
    // block can disagree -- that is what a silhouette is -- and the far one is
    // taken, so cloud is drawn behind ground as well as beside it. Erring the
    // other way would leave a hole in the cloud along every ridge line; erring
    // this way leaves cloud to be masked by the depth test the composite makes
    // per full-resolution pixel.
    var reach = 0.0;
    for (var j = 0; j < 2; j += 1) {
        for (var i = 0; i < 2; i += 1) {
            let pixel = min(block + vec2<i32>(i, j), full - 1);
            let d = textureLoad(depth, pixel, 0).r;
            // Zero depth is the reversed-Z far plane, which the march writes
            // where its ray found no ground at all: nothing stops this ray.
            if d == 0.0 {
                reach = MAX_DISTANCE;
            } else {
                reach = max(reach, min(distance_at(d) * per_step, MAX_DISTANCE));
            }
        }
    }

    let eye = camera.position.xyz;
    let range = slab_range(eye.y, direction.y, cloud.span.x, cloud.span.y);
    var t = max(range.x, 0.0);
    let far = min(range.y, reach);

    var scattered = vec3<f32>(0.0);
    var transmitted = 1.0;
    var depth_sum = 0.0;
    var weight_sum = 0.0;

    if t < far {
        // Constant along a straight ray, so both are found once.
        let cos_theta = dot(direction, sky.sun.xyz);
        let radius = max(length(eye + vec3<f32>(0.0, GROUND_RADIUS, 0.0)), GROUND_RADIUS);
        let sun_mu = dot(sky.up.xyz, sky.sun.xyz);
        let ambient = sample_multiscatter(radius, sun_mu);

        for (var taken = 0u; taken < MAX_STEPS; taken += 1u) {
            if t >= far || transmitted < CUTOFF {
                break;
            }
            let p = eye + direction * t;
            let bound = ceiling_at(p);
            if bound < EMPTY {
                // Nothing in this cell can be seen, so leave all of it at once.
                // The metre is what guarantees the next iteration stands in the
                // next cell rather than on the boundary of this one.
                t = t + cell_exit(p, direction) + 1.0;
                continue;
            }

            let step = min(clamp(STEP_SLOPE * t, MIN_STEP, MAX_STEP), far - t);
            let middle = p + direction * (step * 0.5);
            let extinction = cloud_extinction(middle, true);
            if extinction > 0.0 {
                // Two fetches, where this was a five-step march towards the sun
                // and a flat constant for the sky. The second is what gives a
                // cloud an inside: without it every part of a deck too deep for
                // the sun to reach sits at exactly the same ambient, and a deck
                // seen from below is cotton wool.
                // What the air has left of the sun, at this sample's own
                // altitude rather than the eye's. Both halves of that matter at
                // dusk and only then: the air reddens a cloud at one kilometre
                // far harder than one at ten, and the planet's own shadow rises
                // through the decks -- a sun three degrees down is still up as
                // seen from ten kilometres, which is why the last light of the
                // day is on the highest cloud.
                //
                // The occlusion test rather than trusting the table, and for the
                // reason `cs_skyview` gives: the parameterisation happily
                // returns a transmittance for a ray that would have to pass
                // through the planet to get here.
                let centred = middle + vec3<f32>(0.0, GROUND_RADIUS, 0.0);
                let at = max(length(centred), GROUND_RADIUS);
                let facing = dot(centred / at, sky.sun.xyz);
                var sunlight = sample_transmittance(at, facing);
                if ground_distance(at, facing) >= 0.0 {
                    sunlight = vec3<f32>(0.0);
                }
                let lit_by_sky = sky_share(sky_reaching(middle));
                let source = sunlit(sun_reaching(middle), cos_theta, sunlight)
                    + ambient * lit_by_sky;
                let survived = exp(-extinction * step);
                // Hillaire's energy-conserving segment integral, as
                // `cs_skyview` and `cs_aerial` take it: the scattering
                // coefficient is the albedo times the extinction, so the
                // division by extinction that integral carries cancels and what
                // is left is exact for a uniform segment.
                let landed = transmitted * (1.0 - survived);
                scattered = scattered + landed * ALBEDO * source;
                depth_sum = depth_sum + landed * t;
                weight_sum = weight_sum + landed;
                transmitted = transmitted * survived;
            }
            t = t + step;
        }
    }

    textureStore(out_cloud, at, vec4<f32>(scattered, transmitted));
    // Where along the ray the cloud is, weighted by how much of it each step
    // stopped -- which is the distance the haze in front of it should be read
    // at. Zero where the ray met nothing, where it is never asked for: a
    // transparent pixel composites to what was behind it whatever depth is
    // recorded.
    textureStore(
        out_cloud_depth,
        at,
        vec4<f32>(depth_sum / max(weight_sum, 1e-6), 0.0, 0.0, 0.0),
    );
}

// This frame's quarter, and the last frame's answer carried into the other
// three.
//
// One texel in four is `cs_cloud_march`'s own work and is written through
// untouched. The rest are read out of the buffer this pass left last frame, at
// wherever the camera has since moved them to, and then clamped to the range of
// the marched answers around them.
//
// The clamp is what makes a carried texel safe. A cloud swinging out of frame
// leaves its colour behind in the history, and the neighbours are what say
// there is nothing there now; the same clamp catches a history that has never
// been written at all, which is the zeroes a new texture comes with and would
// read as opaque black cloud. And where every neighbour agrees, the range they
// span is a single point and the clamp returns that point exactly, whatever the
// filter it came through made of it -- which is what keeps an empty sky exactly
// empty. A filtered field of ones is not something to take on trust here; see
// what the light volumes hold, and why, in `src/cloud.rs`.
//
// Which neighbours are entitled to speak is the whole of the difficulty, and it
// is the same question the composite's bilateral upsample asks: a marched texel
// whose ray was stopped by a ridge says nothing about a texel beside it looking
// past the ridge. So the range is built from the ones whose ray ran as far as
// this one's, and only from them -- until there are too few of those to make a
// range at all, and then every neighbour is taken rather than none, because a
// wide range that lets a stale answer through is a far smaller fault than a
// point that replaces this texel with its neighbour.
@compute @workgroup_size(8, 8, 1)
fn cs_cloud_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(rotation.at.zw);
    let at = vec2<i32>(id.xy);
    if at.x >= size.x || at.y >= size.y {
        return;
    }

    let last = vec2<i32>(textureDimensions(fresh_cloud)) - 1;
    let own = at / 2;

    // The nine blocks around this texel, twice over: the range of the ones
    // looking as far as it is, and the range of all of them.
    //
    // Nine rather than the four that bracket it, measured: four leaves too many
    // texels along a skyline with one agreeing neighbour or none, and a range
    // built from one neighbour is a point. Widening the search took the frame
    // from 0.54 per cent of its pixels visibly moved by the amortization to
    // 0.27.
    let mine_reach = reach_of(at * 2);
    var low = vec4<f32>(1e30);
    var high = vec4<f32>(-1e30);
    var near = 1e30;
    var far = -1e30;
    var any_low = vec4<f32>(1e30);
    var any_high = vec4<f32>(-1e30);
    var any_near = 1e30;
    var any_far = -1e30;
    var agreeing = 0;
    for (var j = -1; j < 2; j += 1) {
        for (var i = -1; i < 2; i += 1) {
            let tap = clamp(own + vec2<i32>(i, j), vec2<i32>(0), last);
            let lit = textureLoad(fresh_cloud, tap, 0);
            let along = textureLoad(fresh_along, tap, 0).r;
            any_low = min(any_low, lit);
            any_high = max(any_high, lit);
            any_near = min(any_near, along);
            any_far = max(any_far, along);
            let reach = reach_of(marched_texel(tap) * 2);
            if abs(reach - mine_reach) > TOGETHER * max(reach, mine_reach) {
                continue;
            }
            agreeing += 1;
            low = min(low, lit);
            high = max(high, lit);
            near = min(near, along);
            far = max(far, along);
        }
    }
    // A range wants two ends. One neighbour that agrees is not a range but a
    // point, and clamping to a point is replacing this texel with that
    // neighbour -- which shows as flat rectangles cut out of a cloud.
    if agreeing < 2 {
        low = any_low;
        high = any_high;
        near = any_near;
        far = any_far;
    }

    // This frame's own answer, written through and never carried.
    let mine = textureLoad(fresh_cloud, own, 0);
    let mine_along = textureLoad(fresh_along, own, 0).r;
    if all(at == marched_texel(own)) {
        textureStore(out_cloud, at, mine);
        textureStore(out_cloud_depth, at, vec4<f32>(mine_along, 0.0, 0.0, 0.0));
        return;
    }

    // Where this texel's ray was on the screen the last frame drew. The
    // distance is this block's own, which is the nearest answer there is and is
    // never more than one texel away -- and where the block found no cloud at
    // all there is no distance to speak of, so the ray is followed to infinity
    // instead. That is the `w = 0` point, and it is the right answer for a sky
    // whose contents are kilometres off: all that is left of the motion is the
    // camera's own turn, which is exactly what a point at infinity carries.
    let raw = ray_raw_at(vec2<f32>(at * 2) + 1.0);
    let direction = raw / length(raw);
    var probe = vec4<f32>(direction, 0.0);
    if mine_along > 0.0 {
        probe = vec4<f32>(camera.position.xyz + direction * mine_along, 1.0);
    }
    let clip = camera.was_view_proj * probe;

    var carried = mine;
    var carried_along = mine_along;
    if clip.w > 0.0 {
        let ndc = clip.xy / clip.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0)) {
            let texel = min(vec2<i32>(uv * vec2<f32>(size)), size - 1);
            let slack = (high - low) * SLACK;
            carried = clamp(
                textureSampleLevel(was_cloud, edge_sampler, uv, 0.0),
                low - slack,
                high + slack,
            );
            // Loaded rather than filtered, and not only because an `r32float`
            // cannot be. A blended distance addresses the aerial-perspective
            // volume at a place no cloud is; the composite refuses to blend one
            // for the same reason.
            let room = (far - near) * SLACK;
            carried_along =
                clamp(textureLoad(was_along, texel, 0).r, near - room, far + room);
        }
    }
    textureStore(out_cloud, at, carried);
    textureStore(out_cloud_depth, at, vec4<f32>(carried_along, 0.0, 0.0, 0.0));
}
