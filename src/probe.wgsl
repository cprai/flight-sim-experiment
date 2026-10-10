// Irradiance probes: the light.
//
// `cs_probe_rays` in `src/terrain.wgsl` has found what each ray of this frame's
// probes met. This lights it -- the sky where a ray escaped, and where it met
// the ground, that ground's albedo times the sun reaching it through the
// terrain's shadow and the cloud's and the sky reaching it -- and projects the
// answers onto the first two bands of spherical harmonics, which is as much of
// the light's shape as irradiance keeps.
//
// One bounce. What lights the ground a ray meets is the sun and the same
// stand-in for the sky the shading used before there were probes -- not the
// probes themselves, which would make it every bounce rather than one, a frame
// at a time, and would make what a probe holds depend on what it held before.
// A probe is then a function of where it stands, the sun and the weather and
// nothing else: a still camera sees still light, and a camera that flew here
// sees the light one that jumped here does, once the probes have come round.
// The bounces it gives up are the second and later, each a tenth to a third of
// the one before at the albedos the ground has.
//
// The probes stand in cascades centred on the eye and snapped to their own
// spacing, each twice the spacing and twice the span of the one inside it; see
// `src/probe.rs`. Like the terrain's shadow and the cloud's light volume they
// are a fact about the world around the eye, never about where it is looking.
//
// What the probes hold is the light with the cloud's dimming of the sky left
// out: a ray that escapes takes the sky-view table's radiance as it is. The
// cloud overhead is applied where the light lands, as it always was -- the
// ground by `sky_reaching` at the ground, a cloud by its own at the sample --
// because what dims the sky at a point is the cloud above that point and not
// the cloud above the nearest probe.

const GROUND_RADIUS: f32 = 6360000.0;
const TOP_RADIUS: f32 = 6460000.0;
const TRANSMITTANCE_WIDTH: u32 = 256u;
const TRANSMITTANCE_HEIGHT: u32 = 64u;
const SKYVIEW_HEIGHT: u32 = 108u;
const MULTISCATTER_SIZE: u32 = 32u;
const PI: f32 = 3.14159265358979;
const LIGHT_ACROSS: u32 = 192u;
const LIGHT_SLICES: u32 = 192u;
const LIGHT_CASCADES: u32 = 3u;
const CASCADE_EDGE: f32 = 0.8;
const CASCADE_FADE: f32 = 0.9;
const SUN_CHANNEL: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 0.0);
const SKY_CHANNEL: vec4<f32> = vec4<f32>(0.0, 1.0, 0.0, 0.0);
const SUN_FRAMES: u32 = 0u;
const SKY_FRAMES: u32 = 3u;
const BOUNCED: f32 = 0.4;
const SHADOW_ACROSS: u32 = 1024u;
const SHADOW_CASCADES: u32 = 6u;
const PALETTE_SIZE: u32 = 2304u;
const MATERIAL_MASK: u32 = 0xffffu;
const PROBE_ACROSS: u32 = 32u;
const PROBE_CASCADES: u32 = 8u;
const PROBE_RAYS: u32 = 128u;
const PROBE_PHASES: u32 = 64u;
const NEVER: f32 = 1e30;

// The first two bands of the real spherical harmonics, as the constants they
// are multiplied by: the band-zero constant, and the band-one constant that
// scales each of x, y and z.
const SH_0: f32 = 0.282095;
const SH_1: f32 = 0.488603;

struct Sky {
    sun: vec4<f32>,
    eye: vec4<f32>,
    up: vec4<f32>,
    sun_tangent: vec4<f32>,
};

struct Palette {
    colours: array<vec4<f32>, 2304>,
};

// One cascade of one light volume. Must match `Frame` in `src/cloud_march.wgsl`.
struct Frame {
    // World to texture: each row dotted with `(p, 1)`.
    to_texture: array<vec4<f32>, 3>,
    // Texture to world, the same way round.
    to_world: array<vec4<f32>, 3>,
    // One slice of a column as a vector in the world, and its length.
    slice: vec4<f32>,
    // Which texture axes a cascade hands over at past its edge (one), and
    // which it holds its nearest texel along (zero). See `beyond_light`.
    fade: vec4<f32>,
};

struct Light {
    // The sky's plumb lattice, which its frames are built from: per cascade the
    // near corner in x and z, the height of the lowest slice, and the metres a
    // texel covers across. Read by the tests rather than by anything here.
    cascade: array<vec4<f32>, LIGHT_CASCADES>,
    // The metres of height one plumb slice is worth, in `w`.
    walk: vec4<f32>,
    // Where the baked wind's own grid stands: its near corner in x and z, then
    // how much world it covers in each. Zero-sized until the wind has been
    // solved, which is what says there is no field to read yet.
    air: vec4<f32>,
    // How far the weather has been carried by the mean wind since the world
    // started, in metres. Folded into one tile of the map, so a long flight
    // cannot walk it out of the range an `f32` holds a lattice index in.
    carried: vec4<f32>,
    // The sun's cascades, then the sky's. A texel's red channel is a point of
    // the sun's lattice and its green a point of the sky's: one texture, two
    // lattices, which coincide while the sun's columns climb. See `out_light`.
    frames: array<Frame, 6>,
};

struct Shadow {
    cascade: array<vec4<f32>, SHADOW_CASCADES>,
    sun: vec4<f32>,
    bounds: vec4<f32>,
};

// Where the probe cascades stand. Must match `Probes` in `src/terrain.wgsl`.
struct Probes {
    // Per cascade: the corner of the lattice in the world, and the metres
    // between probes. Probe `i` stands at `corner + (i + 0.5) * spacing`.
    cascade: array<vec4<f32>, PROBE_CASCADES>,
    // The same, as last frame had it.
    was: array<vec4<f32>, PROBE_CASCADES>,
};

struct Phase {
    at: vec4<u32>,
};

@group(0) @binding(0) var<uniform> sky: Sky;

@group(2) @binding(0) var lut_sampler: sampler;
@group(2) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(2) var multiscatter_lut: texture_2d<f32>;
@group(2) @binding(3) var skyview_sampler: sampler;
@group(2) @binding(4) var skyview_lut: texture_2d<f32>;

@group(3) @binding(0) var<uniform> probes: Probes;
@group(3) @binding(1) var<uniform> phase: Phase;
@group(3) @binding(2) var<storage, read> probe_hits: array<vec4<u32>>;
// Every probe as it stood at the end of last frame, and as it will stand at the
// end of this one: three texels a probe, laid across the first axis in three
// blocks of `PROBE_ACROSS` -- block-major, so the filter never blends one
// coefficient into another -- and the cascades stacked up the third. Block zero
// is band zero and the z of band one's red; blocks one and two are band one's
// x and y and the z's green and blue. See `probe_uvw`.
@group(3) @binding(3) var probe_field: texture_3d<f32>;
@group(3) @binding(4) var out_probes: texture_storage_3d<rgba16float, write>;
@group(3) @binding(5) var<uniform> palette: Palette;
@group(3) @binding(6) var light_volume: texture_3d<f32>;
@group(3) @binding(7) var<uniform> light: Light;
@group(3) @binding(8) var shadow_field: texture_2d_array<u32>;
@group(3) @binding(9) var<uniform> shadow: Shadow;

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

fn sample_multiscatter(r: f32, mu_s: f32) -> vec3<f32> {
    let altitude = clamp((r - GROUND_RADIUS) / (TOP_RADIUS - GROUND_RADIUS), 0.0, 1.0);
    let uv = vec2<f32>(
        to_texture(clamp(mu_s * 0.5 + 0.5, 0.0, 1.0), f32(MULTISCATTER_SIZE)),
        to_texture(altitude, f32(MULTISCATTER_SIZE)),
    );
    return textureSampleLevel(multiscatter_lut, lut_sampler, uv, 0.0).rgb;
}

fn horizon_zenith(r: f32) -> f32 {
    return PI - asin(clamp(GROUND_RADIUS / max(r, GROUND_RADIUS), -1.0, 1.0));
}

fn skyview_v(r: f32, zenith: f32) -> f32 {
    let horizon = horizon_zenith(r);
    if zenith < horizon {
        return 0.5 * (1.0 - sqrt(max(1.0 - zenith / horizon, 0.0)));
    }
    return 0.5 + 0.5 * sqrt(max((zenith - horizon) / (PI - horizon), 0.0));
}

fn sample_skyview(direction: vec3<f32>) -> vec3<f32> {
    let up = sky.up.xyz;
    let forward = sky.sun_tangent.xyz;
    let side = cross(up, forward);
    let flat = normalize(direction - up * dot(up, direction));
    // No half-texel correction across: that axis wraps. See `src/sky.wgsl`.
    let u = 0.5 + atan2(dot(side, flat), dot(forward, flat)) * (0.5 / PI);
    let zenith = acos(clamp(dot(up, direction), -1.0, 1.0));
    let v = to_texture(skyview_v(sky.eye.w, zenith), f32(SKYVIEW_HEIGHT));
    return textureSampleLevel(skyview_lut, skyview_sampler, vec2<f32>(u, v), 0.0).rgb;
}

fn light_uvw(p: vec3<f32>, frame: u32) -> vec3<f32> {
    let q = vec4<f32>(p, 1.0);
    let forward = light.frames[frame].to_texture;
    return vec3<f32>(dot(forward[0], q), dot(forward[1], q), dot(forward[2], q));
}

fn stacked_w(w: f32, cascade: u32) -> f32 {
    let slices = f32(LIGHT_SLICES);
    let inside = clamp(w * slices, 0.5, slices - 0.5);
    return (f32(cascade) * slices + inside) / (slices * f32(LIGHT_CASCADES));
}

fn beyond_light(uvw: vec3<f32>, fade: vec3<f32>) -> f32 {
    let out = abs(uvw - vec3<f32>(0.5)) * 2.0 * fade;
    return max(max(out.x, out.y), out.z);
}

fn reaching(
    volume: texture_3d<f32>,
    filtering: sampler,
    p: vec3<f32>,
    frames: u32,
    channel: vec4<f32>,
) -> f32 {
    var blocked = 0.0;
    var answered = 0.0;
    for (var cascade = 0u; cascade < LIGHT_CASCADES; cascade += 1u) {
        if answered >= 1.0 {
            break;
        }
        let frame = frames + cascade;
        let uvw = light_uvw(p, frame);
        // The widest one is not handing over to anything, so it holds on to its
        // own edge for longer. See `CASCADE_FADE`.
        var edge = CASCADE_EDGE;
        if cascade + 1u == LIGHT_CASCADES {
            edge = CASCADE_FADE;
        }
        let inside = 1.0 - smoothstep(edge, 1.0, beyond_light(uvw, light.frames[frame].fade.xyz));
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

// The cloud between a point and the sun, and between it and the sky, read as
// the shading reads them.
fn sun_reaching(p: vec3<f32>) -> f32 {
    return reaching(light_volume, lut_sampler, p, SUN_FRAMES, SUN_CHANNEL);
}

fn sky_reaching(p: vec3<f32>) -> f32 {
    return reaching(light_volume, lut_sampler, p, SKY_FRAMES, SKY_CHANNEL);
}

fn sky_share(reaching: f32) -> f32 {
    return 1.0 - (1.0 - BOUNCED) * (1.0 - reaching);
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

fn noise_mix(bits: u32) -> u32 {
    var b = bits;
    b ^= b >> 16u;
    b = b * 0x7feb352du;
    b ^= b >> 15u;
    b = b * 0x846ca68bu;
    b ^= b >> 16u;
    return b;
}

fn noise_hash(x: i32, y: i32, seed: u32) -> u32 {
    var bits = seed * 0x9e3779b1u;
    bits = noise_mix(bits ^ (u32(x) * 0x3504f333u));
    bits = noise_mix(bits ^ (u32(y) * 0xf1bbcdcbu));
    return bits;
}

fn probe_of(slot: u32, cascade: u32) -> vec3<u32> {
    let corner = probes.cascade[cascade];
    let first = vec3<i32>(round(corner.xyz / corner.w));
    let reduced = vec3<u32>(slot % 8u, (slot / 8u) % 8u, slot / 64u);
    let residue = vec3<i32>(phase.at.xyz);
    let shift = vec3<u32>(((residue - first) % 4 + 4) % 4);
    return reduced * 4u + shift;
}

fn probe_centre(index: vec3<u32>, cascade: u32) -> vec3<f32> {
    let corner = probes.cascade[cascade];
    return corner.xyz + (vec3<f32>(index) + 0.5) * corner.w;
}

fn probe_world(index: vec3<u32>, cascade: u32) -> vec3<i32> {
    let corner = probes.cascade[cascade];
    return vec3<i32>(round(corner.xyz / corner.w)) + vec3<i32>(index);
}

fn probe_direction(world: vec3<i32>, cascade: u32, ray: u32) -> vec3<f32> {
    let golden = 2.39996323;
    let z = 1.0 - (2.0 * f32(ray) + 1.0) / f32(PROBE_RAYS);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let a = f32(ray) * golden;
    let d = vec3<f32>(r * cos(a), z, r * sin(a));
    // A uniformly random rotation from three hashes, as a unit quaternion.
    let seed = cascade * 0x85ebca6bu;
    let h0 = f32(noise_hash(world.x, world.z, seed ^ u32(world.y)) >> 8u) / 16777216.0;
    let h1 = f32(noise_hash(world.z, world.y, seed ^ 0x68e31da4u) >> 8u) / 16777216.0;
    let h2 = f32(noise_hash(world.y, world.x, seed ^ 0xb5297a4du) >> 8u) / 16777216.0;
    let s1 = sqrt(1.0 - h0);
    let s2 = sqrt(h0);
    let q = vec4<f32>(
        s1 * sin(6.28318531 * h1),
        s1 * cos(6.28318531 * h1),
        s2 * sin(6.28318531 * h2),
        s2 * cos(6.28318531 * h2),
    );
    let t = 2.0 * cross(q.xyz, d);
    return d + q.w * t + cross(q.xyz, t);
}

// The octahedral packing `pack_normal` in `src/terrain.wgsl` makes, undone.
fn unpack_normal(packed: u32) -> vec3<f32> {
    let p = unpack2x16snorm(packed);
    var n = vec3<f32>(p.x, 1.0 - abs(p.x) - abs(p.y), p.y);
    if n.y < 0.0 {
        let folded = (1.0 - abs(n.zx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xz >= vec2<f32>(0.0));
        n = vec3<f32>(folded.x, n.y, folded.y);
    }
    return normalize(n);
}

// Where a cascade of probes starts handing a point over to the one outside it,
// and where the outermost gives up, as shares of the way from its middle to its
// edge -- the same pair, for the same reasons, as `CASCADE_EDGE` and
// `CASCADE_FADE` for the light volume. Must match everywhere the probes are read;
// see `the_probes_read_the_same_everywhere` in `src/probe.rs`.
const PROBE_EDGE: f32 = 0.8;
const PROBE_FADE: f32 = 0.9;

// Where a world point falls among one cascade's probes, counted in spacings
// from the cascade's corner, so probe `i` stands at `i + 0.5`.
fn probe_grid(p: vec3<f32>, cascade: u32) -> vec3<f32> {
    let corner = probes.cascade[cascade];
    return (p - corner.xyz) / corner.w;
}

// The texture coordinate of one block of the probes around `g` in a cascade,
// held half a texel in from every side of the block and the cascade, which is
// where a linear filter stops reaching into the next block or the next cascade.
fn probe_uvw(g: vec3<f32>, cascade: u32, block: u32) -> vec3<f32> {
    let n = f32(PROBE_ACROSS);
    let inside = clamp(g, vec3<f32>(0.5), vec3<f32>(n - 0.5));
    return vec3<f32>(
        (f32(block) * n + inside.x) / (3.0 * n),
        inside.y / n,
        (f32(cascade) * n + inside.z) / (n * f32(PROBE_CASCADES)),
    );
}

// How much of a point a cascade answers for: all of it inside, handing over to
// the next one out across its outermost fifth, and giving up across the
// outermost tenth of the widest.
fn probe_share(g: vec3<f32>, cascade: u32) -> f32 {
    let out = abs(g / f32(PROBE_ACROSS) - vec3<f32>(0.5)) * 2.0;
    var edge = PROBE_EDGE;
    if cascade + 1u == PROBE_CASCADES {
        edge = PROBE_FADE;
    }
    return 1.0 - smoothstep(edge, 1.0, max(max(out.x, out.y), out.z));
}

// The irradiance the probes say arrives at a surface at `p` facing `n`, and in
// `a` how much of the point the cascades answered for -- which is all of it
// within sixty-five kilometres of the eye and nothing past them.
//
// Read half a spacing out along the normal, so a surface takes its light from
// the probes in front of it rather than from the ones standing in the hill
// behind it. Irradiance from the first two bands of radiance is Ramamoorthi
// and Hanrahan's: band zero times pi, band one times two pi over three. Clamped
// at zero, which two bands of a light that is all on one side can dip below.
fn probe_irradiance(field: texture_3d<f32>, filtering: sampler, p: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    var sum = vec3<f32>(0.0);
    var answered = 0.0;
    for (var cascade = 0u; cascade < PROBE_CASCADES; cascade += 1u) {
        if answered >= 1.0 {
            break;
        }
        let g = probe_grid(p + n * (0.5 * probes.cascade[cascade].w), cascade);
        let share = min(probe_share(g, cascade), 1.0 - answered);
        if share > 0.0 {
            let a = textureSampleLevel(field, filtering, probe_uvw(g, cascade, 0u), 0.0);
            let b = textureSampleLevel(field, filtering, probe_uvw(g, cascade, 1u), 0.0);
            let c = textureSampleLevel(field, filtering, probe_uvw(g, cascade, 2u), 0.0);
            let along = b.rgb * n.x + c.rgb * n.y + vec3<f32>(a.w, b.w, c.w) * n.z;
            let e = PI * SH_0 * a.rgb + (2.0 * PI / 3.0) * SH_1 * along;
            sum = sum + share * max(e, vec3<f32>(0.0));
            answered = answered + share;
        }
    }
    return vec4<f32>(sum, answered);
}

// The light arriving at `p` averaged over every direction, and in `a` how much
// of the point the cascades answered for: band zero alone, one fetch a cascade.
fn probe_radiance(field: texture_3d<f32>, filtering: sampler, p: vec3<f32>) -> vec4<f32> {
    var sum = vec3<f32>(0.0);
    var answered = 0.0;
    for (var cascade = 0u; cascade < PROBE_CASCADES; cascade += 1u) {
        if answered >= 1.0 {
            break;
        }
        let g = probe_grid(p, cascade);
        let share = min(probe_share(g, cascade), 1.0 - answered);
        if share > 0.0 {
            let a = textureSampleLevel(field, filtering, probe_uvw(g, cascade, 0u), 0.0);
            sum = sum + share * max(SH_0 * a.rgb, vec3<f32>(0.0));
            answered = answered + share;
        }
    }
    return vec4<f32>(sum, answered);
}

// The texel of one block of one probe.
fn probe_texel(index: vec3<i32>, cascade: u32, block: u32) -> vec3<i32> {
    let n = i32(PROBE_ACROSS);
    return vec3<i32>(i32(block) * n + index.x, index.y, i32(cascade) * n + index.z);
}

// Where a probe stood in last frame's lattice, or outside it.
fn probe_was(index: vec3<u32>, cascade: u32) -> vec3<i32> {
    let was = probes.was[cascade];
    return vec3<i32>(round((probe_centre(index, cascade) - was.xyz) / was.w - 0.5));
}

fn probe_held(at: vec3<i32>) -> bool {
    return all(at >= vec3<i32>(0)) && all(at < vec3<i32>(i32(PROBE_ACROSS)));
}

// Every probe carried from where the lattice stood last frame to where it
// stands now.
//
// A cascade is snapped to its own spacing, so when the eye crosses one the
// whole lattice steps by a probe and the slab it stepped into has no history.
// That slab is filled from last frame's next cascade out, which covered it at
// twice the spacing -- a coarser answer, but the right light, where a slab of
// black would be a stripe of dark ground until its probes came round to be
// traced. The widest cascade has nothing outside it and takes its nearest edge.
@compute @workgroup_size(4, 4, 4)
fn cs_probe_shift(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = PROBE_ACROSS;
    if id.x >= n || id.y >= n || id.z >= n * PROBE_CASCADES {
        return;
    }
    let cascade = id.z / n;
    let index = vec3<u32>(id.x, id.y, id.z % n);
    let held = probe_was(index, cascade);
    var blocks: array<vec4<f32>, 3>;
    if probe_held(held) {
        for (var b = 0u; b < 3u; b += 1u) {
            blocks[b] = textureLoad(probe_field, probe_texel(held, cascade, b), 0);
        }
    } else if cascade + 1u < PROBE_CASCADES {
        let outer = probes.was[cascade + 1u];
        let g = (probe_centre(index, cascade) - outer.xyz) / outer.w;
        for (var b = 0u; b < 3u; b += 1u) {
            blocks[b] = textureSampleLevel(probe_field, lut_sampler, probe_uvw(g, cascade + 1u, b), 0.0);
        }
    } else {
        let edge = clamp(held, vec3<i32>(0), vec3<i32>(i32(n) - 1));
        for (var b = 0u; b < 3u; b += 1u) {
            blocks[b] = textureLoad(probe_field, probe_texel(edge, cascade, b), 0);
        }
    }
    for (var b = 0u; b < 3u; b += 1u) {
        textureStore(out_probes, probe_texel(vec3<i32>(index), cascade, b), blocks[b]);
    }
}

var<workgroup> gathered: array<vec4<f32>, 384>;

// One probe's rays lit and projected onto two bands of harmonics.
//
// A ray that escaped sees the sky-view table in its direction, and nothing of
// the sun's own disc: a disc a hundred thousand times the sky, met by one ray
// in a thousand, would be a firefly in one probe in a hundred, and the sun's
// direct light is counted where it lands. A ray that met the ground sees that
// ground lit by the sun through the air, the terrain's shadow and the cloud's,
// and by the sky as the shading lit all ground before there were probes, and
// sends back its albedo's share of both.
@compute @workgroup_size(128, 1, 1)
fn cs_probe_gather(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) ray: u32,
) {
    let cascade = group.y;
    let index = probe_of(group.x, cascade);
    let dir = probe_direction(probe_world(index, cascade), cascade, ray);
    let slot = (cascade * (PROBE_ACROSS * PROBE_ACROSS * PROBE_ACROSS / PROBE_PHASES) + group.x) * PROBE_RAYS + ray;
    let met = probe_hits[slot];
    let t = bitcast<f32>(met.x);
    var origin = probe_centre(index, cascade);
    origin.y = bitcast<f32>(met.w);

    var radiance = vec3<f32>(0.0);
    if t >= 0.5 * NEVER {
        radiance = sample_skyview(dir);
    } else if met.y != 0xffffffffu {
        let p = origin + dir * t;
        let n = unpack_normal(met.z);
        let id = met.y & MATERIAL_MASK;
        var albedo = vec3<f32>(0.0);
        if id < PALETTE_SIZE {
            albedo = palette.colours[id].rgb;
        }
        let centred = p + vec3<f32>(0.0, GROUND_RADIUS, 0.0);
        let radius = max(length(centred), GROUND_RADIUS);
        let up = centred / radius;
        let sun_mu = dot(up, sky.sun.xyz);
        let direct = sample_transmittance(radius, sun_mu)
            * max(dot(n, sky.sun.xyz), 0.0)
            * sun_reaching(p)
            * sun_clear(p, true);
        let open = sample_multiscatter(radius, sun_mu)
            * PI
            * (0.5 + 0.5 * dot(n, up))
            * sky_share(sky_reaching(p));
        radiance = albedo * (direct + open) / PI;
    }

    let base = ray * 3u;
    gathered[base] = vec4<f32>(radiance * SH_0, radiance.r * SH_1 * dir.z);
    gathered[base + 1u] = vec4<f32>(radiance * (SH_1 * dir.x), radiance.g * SH_1 * dir.z);
    gathered[base + 2u] = vec4<f32>(radiance * (SH_1 * dir.y), radiance.b * SH_1 * dir.z);
    workgroupBarrier();
    for (var width = PROBE_RAYS / 2u; width > 0u; width >>= 1u) {
        if ray < width {
            for (var b = 0u; b < 3u; b += 1u) {
                gathered[base + b] = gathered[base + b] + gathered[(ray + width) * 3u + b];
            }
        }
        workgroupBarrier();
    }
    if ray != 0u {
        return;
    }
    // A uniform set of directions, so each stands for an equal share of the
    // sphere.
    let weight = 4.0 * PI / f32(PROBE_RAYS);
    for (var b = 0u; b < 3u; b += 1u) {
        textureStore(out_probes, probe_texel(vec3<i32>(index), cascade, b), gathered[b] * weight);
    }
}
