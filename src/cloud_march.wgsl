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
const DECKS: u32 = 3u;

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
};

// Must match `Weather` in `src/cloud.wgsl`, for the same reason.
struct Weather {
    decks: array<Deck, 3>,
    clock: vec4<f32>,
    span: vec4<f32>,
    // Where the light volumes sit: the near corner in x and z, the height of
    // their lowest slice, and the metres one texel covers across.
    light_origin: vec4<f32>,
    // How far a sun column drifts horizontally per metre it climbs, then the
    // metres of ray one slice is worth walking towards the sun, then the metres
    // of height one slice is worth -- which is what the same slice is worth
    // walking straight up.
    light_walk: vec4<f32>,
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

// Which deck a height belongs to, or nothing.
//
// A deck's base lifts by up to its swing, carrying its top with it, so the most
// it can occupy is from its nominal base to its nominal top plus that swing. The
// three do not overlap, so this is a single answer rather than a set, and it is
// what lets a sample cost one weather fetch instead of three.
fn deck_at(y: f32) -> i32 {
    for (var deck = 0u; deck < DECKS; deck += 1u) {
        let slab = cloud.decks[deck].slab;
        if y >= slab.x && y <= slab.y + slab.z {
            return i32(deck);
        }
    }
    return -1;
}

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
fn coverage_at(deck: i32, w: vec4<f32>, y: f32) -> f32 {
    let slab = cloud.decks[deck].slab;
    let base = slab.x + w.a * slab.z;
    let thickness = max(slab.y - slab.x, 1.0);
    return w.r * vertical((y - base) / thickness, w.g);
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
    let deck = deck_at(p.y);
    if deck < 0 {
        return 0.0;
    }
    let w = weather_at(deck, p);
    let coverage = coverage_at(deck, w, p.y);
    if coverage <= 0.0 {
        return 0.0;
    }

    let shape = textureSampleLevel(shape_noise, tile_sampler, p / SHAPE_TILE, 0.0);
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
        let detail = textureSampleLevel(detail_noise, tile_sampler, p / DETAIL_TILE, 0.0);
        let slab = cloud.decks[deck].slab;
        let base = slab.x + w.a * slab.z;
        let h = saturate((p.y - base) / max(slab.y - slab.x, 1.0));
        let wisp = mix(detail.a, 1.0 - detail.a, saturate(h * 5.0));
        let eaten = wisp * DETAIL_STRENGTH;
        density = saturate((density - eaten) / max(1.0 - eaten, 1e-3));
    }

    return EXTINCTION * w.b * cloud.decks[deck].slab.w * density;
}

// The largest extinction a cell of the cache may have to bound, from one texel
// of the weather over one range of heights.
fn cell_bound(deck: u32, w: vec4<f32>, low: f32, high: f32) -> f32 {
    let slab = cloud.decks[deck].slab;
    let base = slab.x + w.a * slab.z;
    let thickness = max(slab.y - slab.x, 1.0);
    let peak = clamp(
        vertical_peak(w.g),
        (low - base) / thickness,
        (high - base) / thickness,
    );
    let coverage = w.r * vertical(peak, w.g);
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
// The footprint is the two weather texels the cell covers in each axis *and one
// either side*. That margin is not caution: the march samples the weather map
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
    let first = vec2<i32>(vec2<u32>(id.x, id.z)) * TEXELS_PER_CELL - 1;

    var bound = 0.0;
    for (var deck = 0u; deck < DECKS; deck += 1u) {
        let slab = cloud.decks[deck].slab;
        // A deck that does not reach into these heights cannot put cloud in
        // them, whatever the weather over them says.
        if high < slab.x || low > slab.y + slab.z {
            continue;
        }
        for (var j = 0; j <= TEXELS_PER_CELL + 1; j += 1) {
            for (var i = 0; i <= TEXELS_PER_CELL + 1; i += 1) {
                let at = wrap_texel(first + vec2<i32>(i, j));
                let w = textureLoad(weather_map, at, i32(deck), 0);
                bound = max(bound, cell_bound(deck, w, low, high));
            }
        }
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
    let climbed = (f32(at.z) + 0.5) * cloud.light_walk.w;
    let flat = cloud.light_origin.xy
        + (vec2<f32>(at.xy) + 0.5) * cloud.light_origin.w
        + climbed * shear;
    return vec3<f32>(flat.x, cloud.light_origin.z + climbed, flat.y);
}

// The same mapping backwards: where a world point sits in a light volume.
fn light_uvw(p: vec3<f32>, shear: vec2<f32>) -> vec3<f32> {
    let climbed = p.y - cloud.light_origin.z;
    let flat = p.xz - cloud.light_origin.xy - climbed * shear;
    return vec3<f32>(
        flat / (cloud.light_origin.w * f32(LIGHT_ACROSS)),
        climbed / (cloud.light_walk.w * f32(LIGHT_SLICES)),
    );
}

// How far outside a light volume a coordinate has strayed, as a share of it.
//
// Zero anywhere inside, one at a face, and more beyond. What is beyond is not a
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
fn beyond_light(uvw: vec3<f32>) -> f32 {
    let out = abs(uvw - vec3<f32>(0.5)) * 2.0;
    return max(max(out.x, out.y), out.z);
}

// How much of the sun reaches a point, and how much of the sky does.
//
// No half-texel correction and none wanted: a texel of these stands for the
// point at its own centre rather than for a sample of a function over a range,
// so `light_position` and this are already inverses. See `to_texture` in
// `src/sky.wgsl` for the case where the correction is needed.
fn sun_reaching(p: vec3<f32>) -> f32 {
    let uvw = light_uvw(p, cloud.light_walk.xy);
    let inside = 1.0 - smoothstep(0.9, 1.0, beyond_light(uvw));
    return mix(
        1.0,
        textureSampleLevel(sun_light, edge_sampler, uvw, 0.0).r,
        inside,
    );
}

fn sky_reaching(p: vec3<f32>) -> f32 {
    let uvw = light_uvw(p, vec2<f32>(0.0));
    let inside = 1.0 - smoothstep(0.9, 1.0, beyond_light(uvw));
    return mix(
        1.0,
        textureSampleLevel(sky_light, edge_sampler, uvw, 0.0).r,
        inside,
    );
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
        // Stored for the texel's own centre, so half of its own cell is not yet
        // in front of it. Without this the volume reads half a cell dark
        // everywhere, which is a deck's own shadow cast onto its own top.
        textureStore(
            out_light,
            vec3<i32>(at),
            vec4<f32>(exp(crossed * 0.5 - tau), 0.0, 0.0, 1.0),
        );
    }
}

@compute @workgroup_size(8, 8, 1)
fn cs_cloud_sun_light(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= LIGHT_ACROSS || id.y >= LIGHT_ACROSS {
        return;
    }
    walk_light(id, cloud.light_walk.xy, cloud.light_walk.z);
}

@compute @workgroup_size(8, 8, 1)
fn cs_cloud_sky_light(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= LIGHT_ACROSS || id.y >= LIGHT_ACROSS {
        return;
    }
    // No lean, and a slice is worth its own height rather than the longer ray a
    // leaning one crosses.
    walk_light(id, vec2<f32>(0.0), cloud.light_walk.w);
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

// The clouds in front of every pixel, at half resolution.
//
// Half because the field is smooth and the frame is not: a cloud edge is soft
// over metres where a mountain silhouette is sharp over a pixel, so what is lost
// by marching one ray per two-by-two block is far less than what is saved. What
// clips the ray at the near end of that trade is the G-buffer's own depth, which
// makes the terrain's occlusion of cloud exact and free.
@compute @workgroup_size(8, 8, 1)
fn cs_cloud_march(@builtin(global_invocation_id) id: vec3<u32>) {
    let half_size = textureDimensions(out_cloud);
    if id.x >= half_size.x || id.y >= half_size.y {
        return;
    }
    let at = vec2<i32>(id.xy);
    let block = at * 2;
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
                let lit_by_sky = mix(BOUNCED, 1.0, sky_reaching(middle));
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
