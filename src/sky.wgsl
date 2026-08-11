// The atmosphere, and the tables that stand in for raymarching it.
//
// Hillaire's arrangement (Sebastien Hillaire, "A Scalable and Production Ready
// Sky and Atmosphere Rendering Technique", EGSR 2020). The idea the whole thing
// rests on is that scattering integrals are smooth in their arguments, so they
// can be evaluated on a coarse grid and read back with a bilinear fetch instead
// of being marched per pixel.
//
// Which table is built when follows from what each is a function of. The two
// here -- transmittance and multiple scattering -- depend only on the medium's
// own constants: no camera, no sun. So they are built once at load and never
// touched again, and neither entry point below reads the sky uniform at all.
// The tables that do depend on the sun and the eye are per-frame and live
// elsewhere.
//
// Everything is in metres and in units of the sun's irradiance at the top of
// the atmosphere, which is taken as one per channel. One world unit is one
// metre, so no conversion appears anywhere.

// The planet, as a pair of spheres.
//
// Must match `GROUND_RADIUS` and `TOP_RADIUS` in `src/sky.rs`. The world is
// flat and the atmosphere is not, which is the one real approximation here: see
// the module doc of `src/sky.rs` for what that costs and why the alternative --
// stacking flat slabs of air -- is worse.
const GROUND_RADIUS: f32 = 6360000.0;
const TOP_RADIUS: f32 = 6460000.0;

// Rayleigh scattering: air molecules, and the reason the sky is blue.
//
// Scattering and extinction are the same thing here -- molecular scattering
// absorbs nothing -- and the fourth-power dependence on wavelength is what puts
// blue nearly six times higher than red.
const RAYLEIGH_SCATTERING: vec3<f32> = vec3<f32>(5.802e-6, 13.558e-6, 33.100e-6);
const RAYLEIGH_SCALE_HEIGHT: f32 = 8000.0;

// Mie scattering: aerosols, haze, the whiteness near the horizon.
//
// Scalar because particles far larger than the wavelength scatter every colour
// alike. Extinction exceeds scattering because they absorb as well, and the
// scale height is short: this is the part of the air that sits in valleys.
const MIE_SCATTERING: f32 = 3.996e-6;
const MIE_EXTINCTION: f32 = 4.400e-6;
const MIE_SCALE_HEIGHT: f32 = 1200.0;
// Henyey-Greenstein asymmetry: strongly forward, which is the bright aureole
// around the sun and the reason haze looks back-lit.
const MIE_G: f32 = 0.8;

// Ozone: absorbs and does not scatter, in a layer around 25 km.
//
// It never affects the air the camera is flying through -- the layer sits well
// above the twelve kilometres this ever climbs to -- and it is here for one
// reason: it is what keeps a low sun red rather than orange, by taking out the
// green a long horizontal path would otherwise leave. A tent rather than an
// exponential because the layer has a middle, unlike the other two.
const OZONE_ABSORPTION: vec3<f32> = vec3<f32>(0.650e-6, 1.881e-6, 0.085e-6);
const OZONE_CENTRE: f32 = 25000.0;
const OZONE_HALF_WIDTH: f32 = 15000.0;

// What the ground reflects, for the multiple-scattering bounce alone.
//
// One grey number rather than the material palette, because this is light that
// has left the ground, bounced around the sky and come back: by then it has
// averaged over far more ground than any one pixel can see.
const GROUND_ALBEDO: f32 = 0.3;

// Table sizes. Must match `src/sky.rs`.
const TRANSMITTANCE_WIDTH: u32 = 256u;
const TRANSMITTANCE_HEIGHT: u32 = 64u;
const MULTISCATTER_SIZE: u32 = 32u;
const SKYVIEW_WIDTH: u32 = 192u;
const SKYVIEW_HEIGHT: u32 = 108u;
const AERIAL_WIDTH: u32 = 32u;
const AERIAL_HEIGHT: u32 = 32u;
const AERIAL_SLICES: u32 = 64u;

// How far the aerial perspective volume reaches, in metres along the view axis.
//
// Hillaire's is 32 km, which is right for a world you stand in and far too
// short for one seen from a cockpit: this survey is 115 km across and a ridge
// at eighty of them is exactly the thing the haze exists to place. Clamped
// beyond, where the integral has all but saturated anyway.
const AERIAL_FAR: f32 = 100000.0;
// Samples per slice. Four, because the segment integral below is exact for a
// constant medium and only has to follow the density's curve.
const AERIAL_SUBSTEPS: u32 = 4u;

// Steps along a ray when building each table.
//
// Forty is Bruneton's figure for the transmittance and it is cheap: the table
// is built once and is 16384 texels. Twenty for the multiple scattering, which
// runs sixty-four rays per texel and is the expensive one.
const TRANSMITTANCE_STEPS: u32 = 40u;
const MULTISCATTER_STEPS: u32 = 20u;
// Steps along a sky-view ray. Thirty because this one is rebuilt every frame:
// 192 x 108 x 30 is 622k samples, which is a fraction of a millisecond.
const SKYVIEW_STEPS: u32 = 30u;
// Directions sampled per multiple-scattering texel, as an eight-by-eight grid
// over the sphere. Must match the workgroup size of `cs_multiscatter`.
const MULTISCATTER_DIRECTIONS: u32 = 64u;
const MULTISCATTER_ROOT: f32 = 8.0;

const PI: f32 = 3.14159265358979;
// The phase function of something that scatters every way alike.
const UNIFORM_PHASE: f32 = 0.0795774715459477; // 1 / (4 pi)

// Mirrors `SkyUniform` in `src/sky.rs`. Read by the per-frame entry points
// rather than by the two build-once ones, which are functions of the medium
// alone.
struct Sky {
    // The unit vector pointing at the sun. `w` is unused.
    sun: vec4<f32>,
    // The eye in planet space -- world space with the planet's centre at the
    // origin -- and its radius in `w`.
    eye: vec4<f32>,
    // The local up at the eye. `w` is unused.
    up: vec4<f32>,
    // The sun projected into the eye's tangent plane and normalised: the
    // sky-view table's azimuth zero. Built on the CPU so the degenerate case --
    // the sun exactly overhead, where the projection vanishes -- is decided
    // once, there, instead of by a branch in every thread.
    sun_tangent: vec4<f32>,
};

// Mirrors `CameraUniform` in `src/scene.rs`. Read only by `cs_aerial`, whose
// volume is the camera's own frustum; the other three entry points here are
// functions of the atmosphere and the sun alone and bind nothing at group 0.
struct Camera {
    view_proj: mat4x4<f32>,
    was_view_proj: mat4x4<f32>,
    position: vec4<f32>,
    ray_right: vec4<f32>,
    ray_up: vec4<f32>,
    ray_forward: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<uniform> sky: Sky;

// Linear and clamped. The repository's first sampler: everything else reads
// textures with `textureLoad` at exact texel centres, because everything else
// is a raster whose texels mean particular ground. A table is a function
// sampled on a grid, and reading between its texels is the whole point of
// having one.
@group(2) @binding(0) var lut_sampler: sampler;
@group(2) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(2) var multiscatter_lut: texture_2d<f32>;
// Wrapping in `u`, clamped in `v`: the azimuth comes back round and the zenith
// does not. See `skyview_u`.
@group(2) @binding(3) var skyview_sampler: sampler;
@group(2) @binding(4) var skyview_lut: texture_2d<f32>;

@group(3) @binding(0) var out_transmittance: texture_storage_2d<rgba16float, write>;
@group(3) @binding(1) var out_multiscatter: texture_storage_2d<rgba16float, write>;
@group(3) @binding(2) var out_skyview: texture_storage_2d<rgba16float, write>;
@group(3) @binding(3) var out_aerial_scatter: texture_storage_3d<rgba16float, write>;
@group(3) @binding(4) var out_aerial_transmit: texture_storage_3d<rgba16float, write>;

// Where a slice boundary sits, and its inverse.
//
// Quadratic rather than Hillaire's uniform slices, because a hundred kilometres
// split evenly into 64 gives 1.6 km steps and puts the coarsest sampling
// exactly where the gradient is steepest -- the first few kilometres, which is
// most of what a frame contains. Equal steps in the square root of distance put
// roughly equal *visible* change in each slice, because optical depth grows
// about linearly with distance while what the eye notices grows with its
// logarithm. Slice 0 ends at 24 m, slice 15 at 6.25 km, slice 31 at 25 km,
// slice 63 at 100.
//
// Rejected: exponential slicing, which is better still and needs a bias tuned
// to the near plane, and cannot start its first slice at zero.
fn aerial_distance(w: f32) -> f32 {
    return AERIAL_FAR * w * w;
}

fn aerial_w(t: f32) -> f32 {
    return sqrt(clamp(t, 0.0, AERIAL_FAR) / AERIAL_FAR);
}

// Slice `i` holds the integral out to `w = (i+1)/n` but its texel centre is at
// `(i+0.5)/n`, so the depth coordinate is half a slice behind the distance it
// stands for. Clamped at both ends by the sampler.
fn aerial_z(w: f32) -> f32 {
    return w - 0.5 / f32(AERIAL_SLICES);
}

// The half-texel correction, both ways.
//
// A table of `n` texels sampled at uv covers its range with the first texel
// centre at `0.5/n` and the last at `1 - 0.5/n`, so the parameter has to be
// squeezed into that span or the ends of the table are never addressed and the
// bilinear fetch reads a clamped edge instead of the value that belongs there.
//
// This is the single easiest thing to get wrong in the whole technique, and it
// fails quietly: everything still looks nearly right, with the horizon a degree
// out of place. `src/sky.rs` carries the same pair and a test round-trips them.
fn to_texture(x: f32, n: f32) -> f32 {
    return 0.5 / n + x * (1.0 - 1.0 / n);
}

fn to_unit(u: f32, n: f32) -> f32 {
    return (u - 0.5 / n) / (1.0 - 1.0 / n);
}

// Distance to the top of the atmosphere from radius `r` along a ray whose
// cosine against the local up is `mu`. Always reached from inside.
fn top_distance(r: f32, mu: f32) -> f32 {
    let discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
    return max(-r * mu + sqrt(max(discriminant, 0.0)), 0.0);
}

// Distance to the ground, or a negative number if the ray misses it.
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

// How far along a ray it takes to reach whatever it meets first.
fn ray_end(r: f32, mu: f32) -> f32 {
    let ground = ground_distance(r, mu);
    if ground >= 0.0 {
        return ground;
    }
    return top_distance(r, mu);
}

// Radius after travelling `t` along a ray that left radius `r` at cosine `mu`.
fn radius_at(r: f32, mu: f32, t: f32) -> f32 {
    return sqrt(max(r * r + t * t + 2.0 * r * mu * t, 0.0));
}

// What the air at height `h` scatters and absorbs, per metre.
struct Medium {
    rayleigh: vec3<f32>,
    mie: f32,
    // Everything that takes light out of the beam, scattering included.
    extinction: vec3<f32>,
};

fn medium(height: f32) -> Medium {
    // The parameterisations never ask below the ground, but rounding at the
    // horizon can, and a negative height would make the exponentials explode.
    let h = max(height, 0.0);
    let rayleigh_density = exp(-h / RAYLEIGH_SCALE_HEIGHT);
    let mie_density = exp(-h / MIE_SCALE_HEIGHT);
    let ozone_density = max(0.0, 1.0 - abs(h - OZONE_CENTRE) / OZONE_HALF_WIDTH);

    var out: Medium;
    out.rayleigh = RAYLEIGH_SCATTERING * rayleigh_density;
    out.mie = MIE_SCATTERING * mie_density;
    out.extinction = out.rayleigh
        + vec3<f32>(MIE_EXTINCTION * mie_density)
        + OZONE_ABSORPTION * ozone_density;
    return out;
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    // 3/(16 pi) * (1 + cos^2)
    return 0.0596831036594607 * (1.0 + cos_theta * cos_theta);
}

fn mie_phase(cos_theta: f32) -> f32 {
    // Henyey-Greenstein, normalised over the sphere.
    let g = MIE_G;
    let gg = g * g;
    let denominator = 1.0 + gg - 2.0 * g * cos_theta;
    return (1.0 - gg) / (4.0 * PI * denominator * sqrt(max(denominator, 1e-8)));
}

// Where `(r, mu)` sits in the transmittance table.
//
// Bruneton's parameterisation, which Hillaire keeps: the horizontal axis is how
// far the ray runs before it leaves the atmosphere, mapped between the shortest
// such distance at this radius and the longest, and the vertical axis is the
// distance to the horizon. Both put texels where the function actually bends,
// which a plain `(altitude, cos)` grid does not -- near the horizon the
// transmittance falls by orders of magnitude over a fraction of a degree.
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

// The same mapping backwards, which is what building the table needs.
fn transmittance_params(uv: vec2<f32>) -> vec2<f32> {
    let atmosphere = sqrt(max(TOP_RADIUS * TOP_RADIUS - GROUND_RADIUS * GROUND_RADIUS, 0.0));
    let horizon = atmosphere * to_unit(uv.y, f32(TRANSMITTANCE_HEIGHT));
    let r = sqrt(max(horizon * horizon + GROUND_RADIUS * GROUND_RADIUS, 0.0));
    let shortest = TOP_RADIUS - r;
    let longest = horizon + atmosphere;
    let distance = shortest + to_unit(uv.x, f32(TRANSMITTANCE_WIDTH)) * (longest - shortest);
    var mu = 1.0;
    if distance > 0.0 {
        mu = clamp(
            (atmosphere * atmosphere - horizon * horizon - distance * distance)
                / (2.0 * r * distance),
            -1.0,
            1.0,
        );
    }
    return vec2<f32>(r, mu);
}

// How much of the sun survives the air between here and the top, per channel.
fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    let uv = transmittance_uv(r, mu);
    return textureSampleLevel(transmittance_lut, lut_sampler, uv, 0.0).rgb;
}

// Where `(r, mu_s)` sits in the multiple-scattering table.
//
// Plain and linear in both, unlike the transmittance above, and deliberately:
// multiply-scattered light is the smoothest quantity in the model -- it has
// been averaged over every direction -- so it has no sharp horizon feature to
// concentrate texels around. Thirty-two squared is enough for it.
fn multiscatter_uv(r: f32, mu_s: f32) -> vec2<f32> {
    let altitude = clamp((r - GROUND_RADIUS) / (TOP_RADIUS - GROUND_RADIUS), 0.0, 1.0);
    return vec2<f32>(
        to_texture(clamp(mu_s * 0.5 + 0.5, 0.0, 1.0), f32(MULTISCATTER_SIZE)),
        to_texture(altitude, f32(MULTISCATTER_SIZE)),
    );
}

fn sample_multiscatter(r: f32, mu_s: f32) -> vec3<f32> {
    let uv = multiscatter_uv(r, mu_s);
    return textureSampleLevel(multiscatter_lut, lut_sampler, uv, 0.0).rgb;
}

// The optical depth from `(r, mu)` out to the top of the atmosphere.
//
// Midpoint rule. The integrand is a sum of exponentials in the height, which is
// smooth enough that forty steps land within a fraction of a percent of the
// analytic answer -- `src/sky.rs` integrates the same thing at four times the
// step count so the table can be checked against something other than itself.
fn optical_depth(r: f32, mu: f32) -> vec3<f32> {
    let end = top_distance(r, mu);
    let step = end / f32(TRANSMITTANCE_STEPS);
    var depth = vec3<f32>(0.0);
    for (var i = 0u; i < TRANSMITTANCE_STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * step;
        depth = depth + medium(radius_at(r, mu, t) - GROUND_RADIUS).extinction * step;
    }
    return depth;
}

@compute @workgroup_size(8, 8, 1)
fn cs_transmittance(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= TRANSMITTANCE_WIDTH || id.y >= TRANSMITTANCE_HEIGHT {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5)
        / vec2<f32>(f32(TRANSMITTANCE_WIDTH), f32(TRANSMITTANCE_HEIGHT));
    let params = transmittance_params(uv);
    let transmittance = exp(-optical_depth(params.x, params.y));
    textureStore(out_transmittance, vec2<i32>(id.xy), vec4<f32>(transmittance, 1.0));
}

// The zenith angle of the horizon, seen from radius `r`.
//
// Greater than a right angle from anywhere above the ground: from twelve
// kilometres up the horizon is three and a half degrees *below* level, which is
// exactly the feature the sky-view table's vertical mapping is built to
// resolve. On the ground it is a right angle exactly.
fn horizon_zenith(r: f32) -> f32 {
    return PI - asin(clamp(GROUND_RADIUS / max(r, GROUND_RADIUS), -1.0, 1.0));
}

// Where a zenith angle sits down the sky-view table.
//
// Hillaire's mapping: zenith to 0, the horizon to exactly 0.5, straight down to
// 1, with a square root on each side so texels crowd towards the horizon. That
// is where the whole interesting part of a sky is -- the pale band, the sunset
// -- and a linear axis would spend half its rows on the featureless dome
// overhead.
fn skyview_v(r: f32, zenith: f32) -> f32 {
    let horizon = horizon_zenith(r);
    if zenith < horizon {
        return 0.5 * (1.0 - sqrt(max(1.0 - zenith / horizon, 0.0)));
    }
    return 0.5 + 0.5 * sqrt(max((zenith - horizon) / (PI - horizon), 0.0));
}

// The same backwards, which is what building the table needs.
fn skyview_zenith(r: f32, v: f32) -> f32 {
    let horizon = horizon_zenith(r);
    if v < 0.5 {
        let away = 1.0 - 2.0 * v;
        return horizon * (1.0 - away * away);
    }
    let past = 2.0 * v - 1.0;
    return horizon + (PI - horizon) * past * past;
}

// Where a direction sits across the sky-view table: its azimuth from the sun.
//
// A deliberate departure from the paper, which uses `0.5 * cos(azimuth) + 0.5`.
// That mapping's derivative vanishes at the sun itself, so it puts its fewest
// texels exactly where the sky changes fastest -- around the aureole. A signed
// angle is uniform instead, 1.875 degrees a texel over the full circle, and it
// makes no assumption that the sky is symmetric about the sun-zenith plane,
// which stops being true the moment terrain shadowing or a cloud arrives.
//
// It costs a wrapping sampler in `u`, or the seam behind the sun is a line.
//
// No half-texel correction on this axis, and that is not an oversight. The
// correction exists to reach the ends of a range that stops; this range does not
// stop, it comes back round. With `Repeat` addressing the texel centres are at
// `(i + 0.5) / n` and the last interpolates into the first, so an azimuth maps
// straight to a coordinate and the seam behind the sun closes exactly.
fn skyview_u(direction: vec3<f32>) -> f32 {
    let up = sky.up.xyz;
    let forward = sky.sun_tangent.xyz;
    let side = cross(up, forward);
    let flat = normalize(direction - up * dot(up, direction));
    return 0.5 + atan2(dot(side, flat), dot(forward, flat)) * (0.5 / PI);
}

// The sky in a direction, from the table.
//
// Below the horizon as well as above it, and that is worth explaining because
// the obvious move is to clamp there. A ray pointing down past the edge of the
// survey has no ground to meet -- the terrain is a plane that stops -- so the
// question is what to draw instead, and the table's own answer turns out to be
// the right one.
//
// It is the right one because `cs_skyview` marches the *air* and never adds
// what the ground reflects. Below the horizon its rays stop at the ground
// sphere and carry only the haze in between, which is exactly what a downward
// ray over a world with no more terrain should show. Clamping to the horizon
// instead was tried and is plainly worse: it smears one colour flat across the
// whole lower half of the frame, where the unclamped table darkens away
// smoothly like distance haze over water.
fn sample_skyview(direction: vec3<f32>) -> vec3<f32> {
    let zenith = acos(clamp(dot(sky.up.xyz, direction), -1.0, 1.0));
    let uv = vec2<f32>(
        skyview_u(direction),
        to_texture(skyview_v(sky.eye.w, zenith), f32(SKYVIEW_HEIGHT)),
    );
    return textureSampleLevel(skyview_lut, skyview_sampler, uv, 0.0).rgb;
}

// The sky the eye sees in every direction, for this frame's eye and sun.
//
// The first of the two tables that cannot be precomputed. It is a raymarch from
// a particular altitude with the sun in a particular place, and those two facts
// are what let it be two-dimensional rather than four: baking them in as axes
// would make a faithful table gigabytes, and a table small enough to ship would
// throw away the horizon crowding above and band every sunset.
@compute @workgroup_size(8, 8, 1)
fn cs_skyview(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= SKYVIEW_WIDTH || id.y >= SKYVIEW_HEIGHT {
        return;
    }
    // The azimuth axis wraps, so its coordinate is the parameter; the zenith
    // axis stops at both ends, so its is corrected. See `skyview_u`.
    let u = (f32(id.x) + 0.5) / f32(SKYVIEW_WIDTH);
    let v = to_unit((f32(id.y) + 0.5) / f32(SKYVIEW_HEIGHT), f32(SKYVIEW_HEIGHT));

    let r = sky.eye.w;
    let zenith = skyview_zenith(r, v);
    let azimuth = (u - 0.5) * 2.0 * PI;
    let up = sky.up.xyz;
    let forward = sky.sun_tangent.xyz;
    let side = cross(up, forward);
    let flat = forward * cos(azimuth) + side * sin(azimuth);
    let direction = up * cos(zenith) + flat * sin(zenith);

    let mu = cos(zenith);
    let end = ray_end(r, mu);
    let step = end / f32(SKYVIEW_STEPS);

    // Constant along a straight ray, so the two phase functions are evaluated
    // once rather than at every step.
    let cos_theta = dot(direction, sky.sun.xyz);
    let phase = vec2<f32>(rayleigh_phase(cos_theta), mie_phase(cos_theta));

    var scattered = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    for (var i = 0u; i < SKYVIEW_STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * step;
        // In the eye's own frame the start is straight up at radius r, so a
        // point along the ray is this. Only the radius and the sun angle are
        // ever asked for, so no world position is needed.
        let position = up * r + direction * t;
        let radius = length(position);
        let local_up = position / radius;
        let air = medium(radius - GROUND_RADIUS);

        let sun_mu = dot(local_up, sky.sun.xyz);
        var sunlight = sample_transmittance(radius, sun_mu);
        if ground_distance(radius, sun_mu) >= 0.0 {
            sunlight = vec3<f32>(0.0);
        }
        let multiple = sample_multiscatter(radius, sun_mu);

        // Single scattering carries the phase functions -- which way the light
        // was turned decides how much of it comes this way -- and multiple
        // scattering does not, having forgotten.
        let single = (air.rayleigh * phase.x + vec3<f32>(air.mie) * phase.y) * sunlight;
        let bounced = (air.rayleigh + vec3<f32>(air.mie)) * multiple;

        let extinction = max(air.extinction, vec3<f32>(1e-12));
        let survived = exp(-extinction * step);
        // Hillaire's energy-conserving segment integral rather than a midpoint
        // rectangle: exact for a constant medium over the step, which is what
        // lets thirty steps stand in for a smooth integral.
        scattered = scattered
            + throughput * (single + bounced) * (vec3<f32>(1.0) - survived) / extinction;
        throughput = throughput * survived;
    }

    textureStore(out_skyview, vec2<i32>(id.xy), vec4<f32>(scattered, 1.0));
}

// The air in front of every part of the frame, sliced by distance.
//
// The second table that cannot be precomputed, and the one that could not be
// precomputed even in principle: its two horizontal axes *are* the camera's own
// frustum. It is less a lookup table than a cache of this frame at a thirtieth
// of the resolution -- 32 x 32 x 64 marches standing in for a per-pixel
// raymarch at 1280 x 720, which is the whole reason the shading pass can ask
// what a hundred kilometres of air did with two filtered fetches.
//
// Two volumes rather than Hillaire's one, and this is forced by the range. He
// stores a single mean transmittance in the alpha of the scattering volume; over
// a hundred kilometres the Rayleigh split between red and blue is enormous, and
// one number for all three channels turns a distant mountain grey where it
// should go blue -- which is precisely the effect being added.
//
// One thread per froxel column, marching all sixty-four slices and storing the
// running integral at each boundary. A thread per froxel would redo the whole
// march from the eye for every slice.
@compute @workgroup_size(8, 8, 1)
fn cs_aerial(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= AERIAL_WIDTH || id.y >= AERIAL_HEIGHT {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(AERIAL_WIDTH), f32(AERIAL_HEIGHT));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // The same basis the march walks and the shading rebuilds from, so a froxel
    // column stands over the pixels that will read it. Unnormalised, with its
    // component along the view axis exactly one -- which is what makes `t` here
    // the very quantity `distance_at` recovers from a depth, with no normalise
    // at either end.
    let raw = camera.ray_right.xyz * ndc.x + camera.ray_up.xyz * ndc.y + camera.ray_forward.xyz;
    // ... and the metres a unit of `t` covers, which is more than one away from
    // the middle of the frame.
    let per_step = length(raw);
    let direction = raw / per_step;

    // Constant along a straight ray.
    let cos_theta = dot(direction, sky.sun.xyz);
    let phase = vec2<f32>(rayleigh_phase(cos_theta), mie_phase(cos_theta));

    var scattered = vec3<f32>(0.0);
    var transmitted = vec3<f32>(1.0);
    var t = 0.0;
    for (var slice = 0u; slice < AERIAL_SLICES; slice = slice + 1u) {
        let ends = aerial_distance(f32(slice + 1u) / f32(AERIAL_SLICES));
        let stride = (ends - t) / f32(AERIAL_SUBSTEPS);
        for (var k = 0u; k < AERIAL_SUBSTEPS; k = k + 1u) {
            let along = t + stride * (f32(k) + 0.5);
            let position = sky.eye.xyz + raw * along;
            let radius = length(position);
            let up = position / radius;
            let air = medium(radius - GROUND_RADIUS);

            let sun_mu = dot(up, sky.sun.xyz);
            var sunlight = sample_transmittance(radius, sun_mu);
            // Shadowed by the planet, not by the terrain: there is no terrain
            // in this volume and putting it there would need a shadow map.
            if ground_distance(radius, sun_mu) >= 0.0 {
                sunlight = vec3<f32>(0.0);
            }
            let multiple = sample_multiscatter(radius, sun_mu);
            let single = (air.rayleigh * phase.x + vec3<f32>(air.mie) * phase.y) * sunlight;
            let bounced = (air.rayleigh + vec3<f32>(air.mie)) * multiple;

            let metres = stride * per_step;
            let extinction = max(air.extinction, vec3<f32>(1e-12));
            let survived = exp(-extinction * metres);
            scattered = scattered
                + transmitted * (single + bounced) * (vec3<f32>(1.0) - survived) / extinction;
            transmitted = transmitted * survived;
        }
        t = ends;
        let at = vec3<i32>(i32(id.x), i32(id.y), i32(slice));
        textureStore(out_aerial_scatter, at, vec4<f32>(scattered, 1.0));
        textureStore(out_aerial_transmit, at, vec4<f32>(transmitted, 1.0));
    }
}

// One thread per direction, sixty-four directions per texel.
var<workgroup> shared_scattered: array<vec3<f32>, 64>;
var<workgroup> shared_returned: array<vec3<f32>, 64>;

// The light that reaches a point after bouncing more than once.
//
// Hillaire's construction, and the reason the technique is affordable: rather
// than integrating scattering order by order, march the second order once and
// measure what fraction of it a further bounce would return. That fraction is
// the same at every order, so the whole infinite series is the geometric sum
// `L2 / (1 - f)` -- one closed form standing in for every bounce after the
// first.
//
// The result is stored as though the sky were lit from every direction alike,
// which is what makes it a two-dimensional table: by the second bounce the
// light has forgotten which way it came from.
@compute @workgroup_size(1, 1, 64)
fn cs_multiscatter(
    @builtin(global_invocation_id)id: vec3<u32>,
    @builtin(local_invocation_index) thread: u32,
) {
    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(f32(MULTISCATTER_SIZE));
    // The inverse of `multiscatter_uv`, which is linear in both axes.
    let mu_s = clamp(to_unit(uv.x, f32(MULTISCATTER_SIZE)) * 2.0 - 1.0, -1.0, 1.0);
    let altitude = to_unit(uv.y, f32(MULTISCATTER_SIZE)) * (TOP_RADIUS - GROUND_RADIUS);
    // Held just off the ground: a point exactly on it has a horizon at zero and
    // half the sphere of directions ends up marching a zero-length ray.
    let r = clamp(GROUND_RADIUS + altitude, GROUND_RADIUS + 1.0, TOP_RADIUS - 1.0);

    // A frame of our own, with the local up along +Z. Nothing outside this
    // function sees it: the table is indexed by two angles, so where the sun
    // is in azimuth cannot matter.
    let start = vec3<f32>(0.0, 0.0, r);
    let sun = vec3<f32>(0.0, sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s);

    // A uniform grid over the sphere: `phi` from an inverted cosine so the
    // rings carry equal solid angle, `theta` around. Even coverage matters more
    // than a clever sequence here -- sixty-four samples of a function this
    // smooth, and any clumping shows up as a band in the sky.
    let root = u32(MULTISCATTER_ROOT);
    let theta = 2.0 * PI * ((f32(thread / root) + 0.5) / MULTISCATTER_ROOT);
    let phi = acos(1.0 - 2.0 * ((f32(thread % root) + 0.5) / MULTISCATTER_ROOT));
    let direction = vec3<f32>(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));

    let mu = direction.z;
    let end = ray_end(r, mu);
    let step = end / f32(MULTISCATTER_STEPS);

    var scattered = vec3<f32>(0.0);
    var returned = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    for (var i = 0u; i < MULTISCATTER_STEPS; i = i + 1u) {
        let t = (f32(i) + 0.5) * step;
        let position = start + direction * t;
        let radius = length(position);
        let up = position / radius;
        let air = medium(radius - GROUND_RADIUS);
        let scattering = air.rayleigh + vec3<f32>(air.mie);
        // Guarded because ozone alone can be the whole of the extinction high
        // up, and it goes to zero at the top of its tent.
        let extinction = max(air.extinction, vec3<f32>(1e-12));

        // How much sun reaches this sample: the air above it, and nothing at
        // all if the planet itself is in the way. Terrain is not consulted --
        // there is none in this frame of reference and none in the table.
        let sun_mu = dot(up, sun);
        var sunlight = sample_transmittance(radius, sun_mu);
        if ground_distance(radius, sun_mu) >= 0.0 {
            sunlight = vec3<f32>(0.0);
        }

        // The energy-conserving integral of a homogeneous segment, rather than
        // a midpoint rectangle: exact for a constant medium over the step, and
        // what lets twenty steps stand in for a smooth integral.
        let survived = exp(-extinction * step);
        let integrate = (vec3<f32>(1.0) - survived) / extinction;
        scattered = scattered + throughput * scattering * UNIFORM_PHASE * sunlight * integrate;
        // The same integral without the sun in it: the share of light passing
        // through here that this air would scatter onwards. This is the `f`
        // that closes the series.
        returned = returned + throughput * scattering * integrate;
        throughput = throughput * survived;
    }

    // Light that went down, hit the ground and came back up. The ground is a
    // Lambertian grey, so what leaves it is its albedo over pi times what
    // arrived.
    if ground_distance(r, mu) >= 0.0 {
        let landed = start + direction * end;
        let up = normalize(landed);
        let facing = max(dot(up, sun), 0.0);
        scattered = scattered
            + throughput * GROUND_ALBEDO / PI * facing * sample_transmittance(GROUND_RADIUS, facing);
    }

    shared_scattered[thread] = scattered;
    shared_returned[thread] = returned;
    workgroupBarrier();

    // Halving reduction. The directions were sampled uniformly over the sphere
    // and weighted by the isotropic phase, and `4 pi` solid angle against a
    // phase of `1 / 4 pi` cancels exactly -- so the average is the integral and
    // there is no stray factor to carry.
    for (var stride = MULTISCATTER_DIRECTIONS / 2u; stride > 0u; stride = stride / 2u) {
        if thread < stride {
            shared_scattered[thread] = shared_scattered[thread] + shared_scattered[thread + stride];
            shared_returned[thread] = shared_returned[thread] + shared_returned[thread + stride];
        }
        workgroupBarrier();
    }
    if thread != 0u {
        return;
    }

    let count = f32(MULTISCATTER_DIRECTIONS);
    let second_order = shared_scattered[0] / count;
    let fraction = shared_returned[0] / count;
    // The geometric series. `fraction` is a share of light and so is below one
    // for any real medium, but it is clamped rather than trusted: a table with
    // an infinity in it fails a long way from here and says nothing about why.
    let series = vec3<f32>(1.0) / max(vec3<f32>(1.0) - fraction, vec3<f32>(1e-4));
    textureStore(
        out_multiscatter,
        vec2<i32>(id.xy),
        vec4<f32>(second_order * series, 1.0),
    );
}
