// Shafts: the sun's light carved out of the air wherever something stands
// between that air and the sun.
//
// The aerial perspective in `cs_aerial` (`src/sky.wgsl`) lights every metre of
// air by the sun as though nothing but the planet could hide it, which is what
// a sky with nothing in it does. A mountain between the sun and the air leaves
// a wedge of that air unlit; a cloud does the same in a column. Looking towards
// the sun past a peak, the unlit wedges against the lit air either side of them
// are what an eye calls a god ray -- and looking away from it, the same wedges
// are the long shadows a low sun lays across the haze.
//
// So this does not light the air again. It works out what the aerial volume
// counted that the sun never delivered -- the single scattering of the sun, at
// every point of the ray that is in the terrain's shadow or a cloud's -- and the
// shading takes that back out. A sky with nothing in front of the sun loses
// exactly nothing and is drawn to the bit as it was.
//
// Whether a point of the air is lit is a fact about the world: the terrain's
// shadow height (`cs_shadow_sweep` in `src/terrain.wgsl`) and the cloud's light
// volume (`walk_light` in `src/cloud_march.wgsl`), both held in cascades on the
// eye and neither knowing which way it looks. What is per pixel is only the
// eye's own line through that lit and unlit air, which is the thing an eye
// sees. The froxel volume could have carried it instead and was not given it:
// at 32 x 32 columns a froxel is forty to a hundred pixels across, which smears
// a shaft into a smudge and leaks it across every silhouette.
//
// Half resolution, one thread per two-by-two block of the frame, on the same
// lattice the cloud buffer uses: texel `h` marches the ray through
// full-resolution point `2h + 1`, where its block's four pixels meet.

const GROUND_RADIUS: f32 = 6360000.0;
const TOP_RADIUS: f32 = 6460000.0;
const TRANSMITTANCE_WIDTH: u32 = 256u;
const TRANSMITTANCE_HEIGHT: u32 = 64u;
const AERIAL_FAR: f32 = 100000.0;
const PI: f32 = 3.14159265358979;

// The air, which must be the air `cs_aerial` lit: what this takes out has to be
// exactly what that put in. Must match `src/sky.wgsl`; see
// `the_shafts_take_out_what_the_aerial_volume_put_in`.
const RAYLEIGH_SCATTERING: vec3<f32> = vec3<f32>(5.802e-6, 13.558e-6, 33.100e-6);
const RAYLEIGH_SCALE_HEIGHT: f32 = 8000.0;
const MIE_SCATTERING: f32 = 3.996e-6;
const MIE_EXTINCTION: f32 = 4.400e-6;
const MIE_SCALE_HEIGHT: f32 = 1200.0;
const MIE_G: f32 = 0.8;
const OZONE_ABSORPTION: vec3<f32> = vec3<f32>(0.650e-6, 1.881e-6, 0.085e-6);
const OZONE_CENTRE: f32 = 25000.0;
const OZONE_HALF_WIDTH: f32 = 15000.0;

// The light volume's shape. Must match `src/cloud.rs`.
const LIGHT_ACROSS: u32 = 192u;
const LIGHT_SLICES: u32 = 192u;
const LIGHT_CASCADES: u32 = 3u;
const CASCADE_EDGE: f32 = 0.8;
const CASCADE_FADE: f32 = 0.9;
const SUN_CHANNEL: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 0.0);

// The terrain's shadow. Must match `src/terrain.wgsl`.
const SHADOW_ACROSS: u32 = 1024u;
const SHADOW_CASCADES: u32 = 6u;

// Samples along each ray. Spread by the square of their index, so that near
// the eye, where a shaft is widest on screen, they are a few metres apart, and
// at the far end, where the air is behind most of a hundred kilometres of
// itself, they are kilometres apart.
const SHAFT_STEPS: u32 = 48u;

// How high anything can stand between the air and the sun: the top of the
// cloud's light volume, `CEILING_TOP` in `src/cloud.rs`. Above it the terrain is
// far below and the light volume reads clear, so the march stops where its ray
// climbs out through it.
const SHADOW_TOP: f32 = 12000.0;

// Threads per side of a workgroup. Must match `SHAFT_GROUP` in `src/shafts.rs`.
const SHAFT_GROUP: u32 = 8u;

struct Camera {
    view_proj: mat4x4<f32>,
    was_view_proj: mat4x4<f32>,
    position: vec4<f32>,
    ray_right: vec4<f32>,
    ray_up: vec4<f32>,
    ray_forward: vec4<f32>,
    // Where this frame's eye lands in the previous frame's clip space. Mirrors
    // `CameraUniform` in `src/scene.rs`, which says why it is handed over rather
    // than worked out here.
    was_clip: vec4<f32>,
};

// What the world is lit by. Mirrors `SkyUniform` in `src/sky.rs`.
struct Sky {
    sun: vec4<f32>,
    eye: vec4<f32>,
    up: vec4<f32>,
    sun_tangent: vec4<f32>,
};

struct Light {
    cascade: array<vec4<f32>, LIGHT_CASCADES>,
    walk: vec4<f32>,
};

struct Shadow {
    cascade: array<vec4<f32>, SHADOW_CASCADES>,
    sun: vec4<f32>,
    bounds: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> sky: Sky;
@group(2) @binding(0) var lut_sampler: sampler;
@group(2) @binding(1) var transmittance_lut: texture_2d<f32>;

// The frame's depth, for how far each block's rays run.
@group(3) @binding(0) var depth: texture_2d<f32>;
// Where along each block's ray the cloud is, in metres; see `cloud_along` in
// `src/shading.wgsl`.
@group(3) @binding(1) var cloud_along: texture_2d<f32>;
// The cloud's light volume and where it stands.
@group(3) @binding(2) var light_volume: texture_3d<f32>;
@group(3) @binding(3) var<uniform> light: Light;
// The terrain's shadow and where it stands.
@group(3) @binding(4) var shadow_field: texture_2d_array<u32>;
@group(3) @binding(5) var<uniform> shadow: Shadow;
// What the march leaves for the blur: the light lost along the block's ray out
// to its farthest pixel, and out to its nearest, each with in `a` how far that
// pixel is in kilometres along the view axis -- a half float runs out at
// 65504, short of the hundred kilometres the haze reaches. And beside them the
// share of the far half that was lost in front of the cloud on the ray.
//
// Both distances because a block straddling a silhouette has two answers, the
// ridge's and the slope forty kilometres behind it, and each is blurred only
// with the answers of blocks that reach as far as it does. And because the
// shading needs both to know where between the two halves a pixel stands.
@group(3) @binding(6) var out_far: texture_storage_2d<rgba16float, write>;
@group(3) @binding(7) var out_near: texture_storage_2d<rgba16float, write>;
@group(3) @binding(10) var out_share: texture_storage_2d<r32float, write>;
// The blur's own inputs: what the march left.
@group(3) @binding(8) var raw_far: texture_2d<f32>;
@group(3) @binding(9) var raw_near: texture_2d<f32>;
@group(3) @binding(11) var raw_share: texture_2d<f32>;

fn ray_raw_at(screen: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(depth));
    let ndc = vec2<f32>(
        screen.x / size.x * 2.0 - 1.0,
        1.0 - screen.y / size.y * 2.0,
    );
    return camera.ray_right.xyz * ndc.x + camera.ray_up.xyz * ndc.y + camera.ray_forward.xyz;
}

fn distance_at(d: f32) -> f32 {
    return camera.ray_forward.w / d;
}

fn to_texture(x: f32, n: f32) -> f32 {
    return 0.5 / n + x * (1.0 - 1.0 / n);
}

fn top_distance(r: f32, mu: f32) -> f32 {
    let discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
    return max(-r * mu + sqrt(max(discriminant, 0.0)), 0.0);
}

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

fn light_uvw(p: vec3<f32>, shear: vec2<f32>, cascade: u32) -> vec3<f32> {
    let origin = light.cascade[cascade];
    let climbed = p.y - origin.z;
    let flat = p.xz - origin.xy - climbed * shear;
    return vec3<f32>(
        flat / (origin.w * f32(LIGHT_ACROSS)),
        climbed / (light.walk.w * f32(LIGHT_SLICES)),
    );
}

fn stacked_w(w: f32, cascade: u32) -> f32 {
    let slices = f32(LIGHT_SLICES);
    let inside = clamp(w * slices, 0.5, slices - 0.5);
    return (f32(cascade) * slices + inside) / (slices * f32(LIGHT_CASCADES));
}

fn beyond_light(uvw: vec3<f32>) -> f32 {
    let out = abs(uvw.xy - vec2<f32>(0.5)) * 2.0;
    return max(out.x, out.y);
}

fn reaching(
    volume: texture_3d<f32>,
    filtering: sampler,
    p: vec3<f32>,
    shear: vec2<f32>,
    channel: vec4<f32>,
) -> f32 {
    var blocked = 0.0;
    var answered = 0.0;
    for (var cascade = 0u; cascade < LIGHT_CASCADES; cascade += 1u) {
        if answered >= 1.0 {
            break;
        }
        let uvw = light_uvw(p, shear, cascade);
        // The widest one is not handing over to anything, so it holds on to its
        // own edge for longer. See `CASCADE_FADE`.
        var edge = CASCADE_EDGE;
        if cascade + 1u == LIGHT_CASCADES {
            edge = CASCADE_FADE;
        }
        let inside = 1.0 - smoothstep(edge, 1.0, beyond_light(uvw));
        let share = min(inside, 1.0 - answered);
        if share > 0.0 {
            let at = vec3<f32>(uvw.xy, stacked_w(uvw.z, cascade));
            let held = textureSampleLevel(volume, filtering, at, 0.0);
            blocked = blocked + share * (1.0 - exp(-dot(held, channel)));
            answered = answered + share;
        }
    }
    // What is blocked, so an unanswered share leaves the light alone and an
    // empty volume returns exactly one. See `walk_light`.
    //
    // Exponentiated per cascade rather than once over the sum, so a hand-over
    // still blends two transmittances rather than two depths. Averaging depth
    // across a join would be a geometric mean of what the two cascades say, and
    // what they stand for is ground side by side rather than cloud one behind
    // the other -- see `a_coarse_cascade_holds_what_the_fine_one_averages_to`,
    // which is written in the share and would not survive the swap.
    return 1.0 - blocked;
}

// The cloud between a point and the sun, as `sun_reaching` in
// `src/shading.wgsl` reads it: the volume alone, without the march's walk of
// the field ahead of the point, which is a skin a shaft has no use for.
fn sun_reaching(p: vec3<f32>) -> f32 {
    return reaching(light_volume, lut_sampler, p, light.walk.xy, SUN_CHANNEL);
}

// Where a cascade of the terrain's shadow starts handing a point over to the
// one outside it, and where the outermost starts giving up, as shares of the
// way from its middle to its edge. The same pair `reaching` uses, for the same
// reasons: a join is a blend rather than a step, and the outermost has nothing
// to blend into.
const SHADOW_EDGE: f32 = 0.8;
const SHADOW_FADE: f32 = 0.9;
// The thinnest a penumbra is taken to be, in metres of height. The disc's own
// penumbra from an occluder one texel away is centimetres; this only keeps the
// comparison below from dividing by nothing.
const SHADOW_SOFTEST: f32 = 0.05;

// One texel of the terrain's shadow: the shadow height, the distance back to
// the occluder in metres, and how far the texel's own ground clears the shadow.
// See `out_shadow` in `src/terrain.wgsl` for why it is packed the way it is.
//
// Must match `shadow_texel` everywhere else the field is read; see
// `the_shadow_reads_the_same_everywhere` in `src/shadow.rs`.
fn shadow_texel(at: vec2<i32>, cascade: u32) -> vec3<f32> {
    let held = textureLoad(shadow_field, at, i32(cascade), 0).rg;
    let halves = unpack2x16float(held.y);
    return vec3<f32>(bitcast<f32>(held.x), halves.x * 1000.0, halves.y);
}

// How much of the sun one cascade of the terrain's shadow lets through at
// lattice coordinates `w`: to a point at height `y`, or to the ground itself.
//
// Each of the four texels around the point is asked on its own, and the
// answers are blended -- not the heights, which across the edge of a ridge's
// shadow blend to a height nothing stands at. Each answer is a ramp rather than
// a step, as wide as the penumbra the sun's disc casts from as far away as that
// texel's occluder: a shadow from the next ridge over is sharp and one from a
// mountain twenty kilometres off is soft.
//
// The ground is asked differently from a point in the air. A point in the air
// is where it is, and is compared with the shadow height. The ground is asked
// by its own clearance at each texel -- how far the ground the field was swept
// over stands above the shadow there -- and not by the height of the ground the
// march drew, which from far enough away is a coarser level whose surface need
// not be the field's: on a steep slope the two stand hundreds of metres apart,
// and the drawn one would read as buried in its own shadow. Blending the
// texels' own answers is the field's shadow on the field's own ground, which is
// all a field this coarse can say about the ground anyway; the relief finer
// than its texels is lit by the cosine.
//
// Must match `shadow_tap` everywhere else the field is read.
fn shadow_tap(cascade: u32, w: vec2<f32>, y: f32, ground: bool) -> f32 {
    let low = vec2<i32>(floor(w));
    let f = w - vec2<f32>(low);
    let last = vec2<i32>(i32(SHADOW_ACROSS) - 1);
    var clear = vec4<f32>(0.0);
    for (var corner = 0u; corner < 4u; corner += 1u) {
        let offset = vec2<i32>(i32(corner & 1u), i32(corner >> 1u));
        let held = shadow_texel(clamp(low + offset, vec2<i32>(0), last), cascade);
        var above = y - held.x;
        if ground {
            above = held.z;
        }
        let width = max(held.y * shadow.sun.w, SHADOW_SOFTEST);
        clear[corner] = saturate(above / width + 0.5);
    }
    return mix(mix(clear.x, clear.y, f.x), mix(clear.z, clear.w, f.x), f.y);
}

// How much of the sun the terrain lets through to a point, over every cascade
// of its shadow -- or, with `ground`, to the ground under it.
//
// One above the highest ground anywhere: nothing up there can be in the
// terrain's shadow, which is most of the sky and so most of what the cloud and
// the air ask. Below it, the cascades answer innermost first exactly as the
// light volume's do in `reaching`, and what none of them answers for is lit.
//
// Must match `sun_clear` everywhere else the field is read.
fn sun_clear(p: vec3<f32>, ground: bool) -> f32 {
    if p.y > shadow.bounds.x && !ground {
        return 1.0;
    }
    var clear = 0.0;
    var answered = 0.0;
    for (var cascade = 0u; cascade < SHADOW_CASCADES; cascade += 1u) {
        if answered >= 1.0 {
            break;
        }
        let origin = shadow.cascade[cascade];
        let w = (p.xz - origin.xy) / origin.zw;
        let out = abs(w / f32(SHADOW_ACROSS - 1u) - vec2<f32>(0.5)) * 2.0;
        var edge = SHADOW_EDGE;
        if cascade + 1u == SHADOW_CASCADES {
            edge = SHADOW_FADE;
        }
        let inside = 1.0 - smoothstep(edge, 1.0, max(out.x, out.y));
        let share = min(inside, 1.0 - answered);
        if share > 0.0 {
            clear = clear + share * shadow_tap(cascade, w, p.y, ground);
            answered = answered + share;
        }
    }
    return clear + (1.0 - answered);
}

// A fixed offset in `[0, 1)` for each texel of a four-by-four tile.
//
// Forty-eight samples along a ray are forty-eight slabs of air, and a shaft's
// edge crossing them steps forty-eight times; every ray in a region shares the
// same steps, so they line up across the frame as rings about the eye. Starting
// each texel of a tile a different share of a step along turns the rings into a
// fine pattern that the blur below takes back out. Fixed rather than moving with
// the frame: there is no history here to average a moving pattern over, and a
// still camera should see a still sky.
fn interleave(texel: vec2<u32>) -> f32 {
    let bayer = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    return (f32(bayer[(texel.y & 3u) * 4u + (texel.x & 3u)]) + 0.5) / 16.0;
}

// Where along the ray the march's `k`th boundary falls, in view-axis units, for
// a ray of length `reach`.
fn shaft_at(k: f32, reach: f32) -> f32 {
    let w = clamp(k / f32(SHAFT_STEPS), 0.0, 1.0);
    return reach * w * w;
}

@compute @workgroup_size(8, 8, 1)
fn cs_shafts(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(out_far);
    if id.x >= size.x || id.y >= size.y {
        return;
    }
    let block = vec2<i32>(id.xy) * 2;
    let full = vec2<i32>(textureDimensions(depth));

    // How far the block's four pixels reach, nearest and farthest, along the
    // view axis. Sky reaches the end of the aerial volume, which is as far as
    // the haze being corrected was ever counted.
    var near = AERIAL_FAR;
    var far = 0.0;
    for (var j = 0; j < 2; j += 1) {
        for (var i = 0; i < 2; i += 1) {
            let d = textureLoad(depth, min(block + vec2<i32>(i, j), full - 1), 0).r;
            var reach = AERIAL_FAR;
            if d != 0.0 {
                reach = min(distance_at(d), AERIAL_FAR);
            }
            near = min(near, reach);
            far = max(far, reach);
        }
    }

    let raw = ray_raw_at(vec2<f32>(block) + 1.0);
    let per_step = length(raw);
    let direction = raw / per_step;
    let eye = camera.position.xyz;
    // The cloud on this block's ray, in the same view-axis units, or nowhere.
    var cloud = textureLoad(cloud_along, vec2<i32>(id.xy), 0).r / per_step;
    if cloud <= 0.0 {
        cloud = AERIAL_FAR;
    }

    // Out to where the ray climbs above anything that could stand between it
    // and the sun, and no further: past that every sample would be lit, and
    // every one of them would be spent finding that out.
    var reach = far;
    if raw.y > 0.0 {
        reach = min(reach, max(SHADOW_TOP - eye.y, 0.0) / raw.y);
    }
    // ... and a ray that met no ground but points down -- past the edge of the
    // data, or through a hole in it -- shows the sky-view table, whose air ends
    // where the ray meets the planet. Marched on, it would run under the ground,
    // where the terrain's shadow says everything is dark, and take out light
    // that table never counted.
    if raw.y < 0.0 {
        reach = min(reach, max(eye.y, 0.0) / -raw.y);
    }

    let cos_theta = dot(direction, sky.sun.xyz);
    let phase = vec2<f32>(rayleigh_phase(cos_theta), mie_phase(cos_theta));
    let offset = interleave(id.xy);

    var lost = vec3<f32>(0.0);
    var lost_near = vec3<f32>(0.0);
    var lost_cloud = vec3<f32>(0.0);
    var transmitted = vec3<f32>(1.0);
    var t = 0.0;
    var near_taken = false;
    var cloud_taken = false;
    if reach > 0.0 {
        for (var k = 0u; k < SHAFT_STEPS; k += 1u) {
            // The last step runs to the end, so the integral covers all of the
            // ray whatever the offset.
            let end = select(shaft_at(f32(k) + offset, reach), reach, k + 1u == SHAFT_STEPS);
            let stride = end - t;
            if stride <= 0.0 {
                continue;
            }
            // Snapshot what was lost up to the nearer of the block's pixels and
            // up to the cloud as the march passes them.
            if !near_taken && end >= near {
                near_taken = true;
                lost_near = lost;
            }
            if !cloud_taken && end >= cloud {
                cloud_taken = true;
                lost_cloud = lost;
            }
            let along = t + stride * 0.5;
            let position = eye + raw * along;
            let centred = position + vec3<f32>(0.0, GROUND_RADIUS, 0.0);
            let radius = length(centred);
            let air = medium(radius - GROUND_RADIUS);
            let metres = stride * per_step;
            let extinction = max(air.extinction, vec3<f32>(1e-12));
            let survived = exp(-extinction * metres);

            let hidden = 1.0 - sun_clear(position, false) * sun_reaching(position);
            if hidden > 0.0 {
                let sun_mu = dot(centred / radius, sky.sun.xyz);
                var sunlight = sample_transmittance(radius, sun_mu);
                if ground_distance(radius, sun_mu) >= 0.0 {
                    sunlight = vec3<f32>(0.0);
                }
                let single = (air.rayleigh * phase.x + vec3<f32>(air.mie) * phase.y) * sunlight;
                lost = lost + hidden * transmitted * single * (vec3<f32>(1.0) - survived) / extinction;
            }
            transmitted = transmitted * survived;
            t = end;
        }
    }
    if !near_taken {
        lost_near = lost;
    }
    if !cloud_taken {
        lost_cloud = lost;
    }
    // The share lost in front of the cloud, taken off the green channel: the
    // three differ in how much they lose and hardly at all in where.
    var share = 1.0;
    if lost.g > 0.0 {
        share = clamp(lost_cloud.g / lost.g, 0.0, 1.0);
    }
    textureStore(out_far, vec2<i32>(id.xy), vec4<f32>(lost, far / 1000.0));
    textureStore(out_near, vec2<i32>(id.xy), vec4<f32>(lost_near, near / 1000.0));
    textureStore(out_share, vec2<i32>(id.xy), vec4<f32>(share, 0.0, 0.0, 0.0));
}

// How far apart two blocks' reaches are allowed to be and still be taken for
// the same surface, as a share of the nearer.
const BLUR_SAME: f32 = 0.05;

// How much a tap reaching `there` counts towards a texel reaching `here`.
fn blur_weight(there: f32, here: f32) -> f32 {
    let apart = abs(there - here) / (BLUR_SAME * max(min(there, here), 0.001));
    return 1.0 / (1.0 + apart * apart * 100.0);
}

// A five-by-five blur that keeps to one side of every silhouette.
//
// What it is for is the interleave above: neighbouring texels start their
// samples at different points of a step, and the average over a tile of them
// is the smooth answer each one is a sample of. A ridge's near side and the
// slope past it are not samples of one answer, so each half of a texel is
// blurred only with the halves of its neighbours that reach about as far as it
// does: the far half by the far distances, the near half by the near ones.
// Weighing both by the far distance alone left a block straddling a ridge
// averaging its near half with its neighbours' far ones, which drew a dark
// dotted line along every silhouette.
@compute @workgroup_size(8, 8, 1)
fn cs_shafts_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = vec2<i32>(textureDimensions(out_far));
    let at = vec2<i32>(id.xy);
    if at.x >= size.x || at.y >= size.y {
        return;
    }
    let far_here = textureLoad(raw_far, at, 0).a;
    let near_here = textureLoad(raw_near, at, 0).a;
    var far_sum = vec4<f32>(0.0);
    var near_sum = vec3<f32>(0.0);
    var share_sum = 0.0;
    var far_total = 0.0;
    var near_total = 0.0;
    for (var j = -2; j <= 2; j += 1) {
        for (var i = -2; i <= 2; i += 1) {
            let tap = clamp(at + vec2<i32>(i, j), vec2<i32>(0), size - 1);
            let far = textureLoad(raw_far, tap, 0);
            let near = textureLoad(raw_near, tap, 0);
            let far_weight = blur_weight(far.a, far_here);
            let near_weight = blur_weight(near.a, near_here);
            far_sum = far_sum + far_weight * far;
            share_sum = share_sum + far_weight * textureLoad(raw_share, tap, 0).r;
            far_total = far_total + far_weight;
            near_sum = near_sum + near_weight * near.rgb;
            near_total = near_total + near_weight;
        }
    }
    textureStore(out_far, at, vec4<f32>(far_sum.rgb / far_total, far_here));
    textureStore(out_near, at, vec4<f32>(near_sum / near_total, near_here));
    textureStore(out_share, at, vec4<f32>(share_sum / far_total, 0.0, 0.0, 0.0));
}
