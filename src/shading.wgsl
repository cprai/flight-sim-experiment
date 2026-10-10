// Shading: the G-buffer in, the image out.
//
// The geometry pass has already done the expensive part -- every pixel holds
// the material id, world position and surface normal of the ground its ray
// met, or depth zero where it met nothing. This pass is one fetch, one table
// lookup and one dot product per pixel: the table gives the material's colour
// and the normal decides how much of the sun reaches it. Real material texture
// belongs here later, which is the point of the split -- it will not touch the
// march.

// Must match `PALETTE_SIZE` in `src/palette.rs`: one slot per id up to the
// last assigned category block.
const PALETTE_SIZE: u32 = 2304u;

// Missing data. Ids at or past the table's end can only come from a corrupt
// tile -- in-range unassigned ids are magenta in the table itself.
const MAGENTA: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

// Must match `MATERIAL_MASK` in `src/terrain.wgsl`.
const MATERIAL_MASK: u32 = 0xffffu;

// The planet and the tables' shapes. Must match `src/sky.rs` and
// `src/sky.wgsl`; see the module doc of `src/sky.rs` for what pinning a round
// atmosphere under a flat world costs.
const GROUND_RADIUS: f32 = 6360000.0;
const TOP_RADIUS: f32 = 6460000.0;
const TRANSMITTANCE_WIDTH: u32 = 256u;
const TRANSMITTANCE_HEIGHT: u32 = 64u;
const MULTISCATTER_SIZE: u32 = 32u;
const SKYVIEW_HEIGHT: u32 = 108u;
const AERIAL_SLICES: u32 = 64u;
const AERIAL_FAR: f32 = 100000.0;

const PI: f32 = 3.14159265358979;

// The light volume's shape. Must match `LIGHT_ACROSS` and `LIGHT_SLICES` in
// `src/cloud.rs`.
const LIGHT_ACROSS: u32 = 192u;
const LIGHT_SLICES: u32 = 192u;

// Cascades of each light volume. Must match `LIGHT_CASCADES` in `src/cloud.rs`,
// which says why there is more than one.
const LIGHT_CASCADES: u32 = 3u;

// Where one hands over to the next. Must match `CASCADE_EDGE` in
// `src/cloud_march.wgsl`.
const CASCADE_EDGE: f32 = 0.8;

// Where the widest gives up instead. Must match `CASCADE_FADE` in
// `src/cloud_march.wgsl`, which says why it is not the same number.
const CASCADE_FADE: f32 = 0.9;

// How much of the sky a cloud blocks still arrives, having bounced. Must match
// `BOUNCED` in `src/cloud_march.wgsl`; see there for what it stands in for.
const BOUNCED: f32 = 0.4;

// How much display brightness a unit of radiance is worth.
//
// Everything upstream is in units of the sun's irradiance at the top of the
// atmosphere, where level ground of albedo `a` under the reference 45-degree
// sun comes out at about `0.209 a`. The number that follows is the one that
// puts that back where the old two-constant light left it -- `0.35 + 0.65 cos
// 45`, or `0.81 a` -- so the change of model is a change of *behaviour* and not
// a change of overall brightness. Must match `EXPOSURE` in `src/sky.rs`.
const EXPOSURE: f32 = 5.0;

// The radiance that maps to white. Must match `WHITE` in `src/sky.rs`.
//
// 1.6 in unexposed units, which is far above sunlit snow at about 0.3 and far
// below the sun's own disc. So the disc clips to white and nothing else in the
// frame does.
const WHITE: f32 = 8.0;

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

struct Palette {
    colours: array<vec4<f32>, 2304>,
};

// What the world is lit by. Mirrors `SkyUniform` in `src/sky.rs`.
//
// `sun` is the unit vector pointing at the sun from the ground; `w` is unused
// padding.
struct Sky {
    sun: vec4<f32>,
    eye: vec4<f32>,
    up: vec4<f32>,
    sun_tangent: vec4<f32>,
};

// Where the light volume stands and how its columns lean. Must match `Light`
// in `src/cloud_march.wgsl` and `LightUniform` in `src/cloud.rs`.
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

// Group 0 is the camera, as it is for every pipeline in this program: the
// shading needs a world position now, and rebuilds it from the depth and the
// same ray basis the march walked.
@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<uniform> sky: Sky;

// The scattering tables and the sampler that reads between their texels. Built
// once at load; see `src/sky.wgsl`.
@group(2) @binding(0) var lut_sampler: sampler;
@group(2) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(2) var multiscatter_lut: texture_2d<f32>;
// Wrapping in `u`; see `skyview_u` in `src/sky.wgsl`.
@group(2) @binding(3) var skyview_sampler: sampler;
@group(2) @binding(4) var skyview_lut: texture_2d<f32>;
// The air in front of every part of the frame, sliced by distance: the
// in-scattered light and, separately, what survives of what was behind it.
@group(2) @binding(5) var aerial_scatter_lut: texture_3d<f32>;
@group(2) @binding(6) var aerial_transmit_lut: texture_3d<f32>;

@group(3) @binding(0) var<uniform> palette: Palette;
// A material id in the low sixteen bits and where inside its pixel the ground
// sits in the rest; only the id is wanted here. See `MATERIAL_MASK` in
// `src/terrain.wgsl` for what the other half is for.
@group(3) @binding(1) var material: texture_2d<u32>;
@group(3) @binding(3) var depth: texture_2d<f32>;
@group(3) @binding(4) var normal: texture_2d<f32>;
// What the cloud march left, at half this pass's resolution: scattered light
// and, in `w`, how much of what is behind it survives. Beside it, how far along
// the ray that cloud was, in metres.
//
// Loaded rather than sampled, and declared unfilterable so that nothing can
// quietly start sampling them: the upsample below is bilateral, and a hardware
// bilinear across a mountain's silhouette blends cloud that was marched past the
// mountain into pixels the mountain covers. That is the halo this pass exists to
// avoid.
@group(3) @binding(2) var cloud_colour: texture_2d<f32>;
@group(3) @binding(5) var cloud_along: texture_2d<f32>;
// How much of the sun reaches each point of a coarse world grid, and how much
// of the sky does, a channel apiece. The same volume the cloud march lights
// itself from, read at the ground instead: a cloud's shadow on a mountain and a
// cloud's shadow on another cloud are the same fact, looked up in the same
// place.
//
// Sampled rather than loaded, through the tables' own clamped sampler -- these
// clamp too, and what is wanted between their texels is exactly what a linear
// fetch gives. They are not maxima; see `cloud_colour` above for the buffers
// that are.
@group(3) @binding(6) var light_volume: texture_3d<f32>;
// Which channel holds which walk. Must match `SUN_CHANNEL` and `SKY_CHANNEL` in
// `src/cloud_march.wgsl`, which says why they are masks and not indices.
const SUN_CHANNEL: vec4<f32> = vec4<f32>(1.0, 0.0, 0.0, 0.0);
// Where each walk's frames start in `Light::frames`. Must match `SUN_FRAMES`
// and `SKY_FRAMES` in `src/cloud_march.wgsl`.
const SUN_FRAMES: u32 = 0u;
const SKY_FRAMES: u32 = 3u;
const SKY_CHANNEL: vec4<f32> = vec4<f32>(0.0, 1.0, 0.0, 0.0);
@group(3) @binding(8) var<uniform> light: Light;

// The terrain's shadow, as the height above which each point of the ground's
// lattice sees the sun, and where its cascades stand. Swept by
// `cs_shadow_sweep` in `src/terrain.wgsl`, which says what the field is; the
// struct and the two constants must match the ones there.
const SHADOW_ACROSS: u32 = 1024u;
const SHADOW_CASCADES: u32 = 6u;
struct Shadow {
    cascade: array<vec4<f32>, SHADOW_CASCADES>,
    sun: vec4<f32>,
    bounds: vec4<f32>,
};
// Loaded rather than sampled: each of four taps is compared with the point
// before any blending, and a blend of shadow heights across a ridge's edge would
// be a height nothing stands at. Integers because a texel is an `f32` and two
// half floats; see `shadow_texel`.
@group(3) @binding(9) var shadow_field: texture_2d_array<u32>;
@group(3) @binding(10) var<uniform> shadow: Shadow;

// The sun's light the aerial volume counted and the terrain or a cloud kept out
// of the air, per two-by-two block of the frame. See `src/shafts.wgsl`: what
// the block's ray lost out to its farthest pixel and out to its nearest, each
// with in `a` how far that pixel is in kilometres, and the share of the far
// one that was lost in front of the cloud.
@group(3) @binding(11) var shafts_far: texture_2d<f32>;
@group(3) @binding(12) var shafts_near: texture_2d<f32>;
@group(3) @binding(13) var shafts_share: texture_2d<f32>;

// The irradiance probes: the light the world sends to itself, in cascades on
// the eye. See `src/probe.wgsl` for what they hold and `probe_irradiance` for
// how they are read. The constants and the struct must match the ones there.
const PROBE_ACROSS: u32 = 32u;
const PROBE_CASCADES: u32 = 8u;
const SH_0: f32 = 0.282095;
const SH_1: f32 = 0.488603;
struct Probes {
    // Per cascade: the corner of the lattice in the world, and the metres
    // between probes. Probe `i` stands at `corner + (i + 0.5) * spacing`.
    cascade: array<vec4<f32>, PROBE_CASCADES>,
    // The same, as last frame had it.
    was: array<vec4<f32>, PROBE_CASCADES>,
};
@group(3) @binding(14) var probe_field: texture_3d<f32>;
@group(3) @binding(15) var<uniform> probes: Probes;


// The ray through a point on the screen, before it is normalised.
//
// Must match `ndc_of` and `ray_raw_at` in `src/terrain.wgsl`, character for
// character. There is no preprocessor here and no `#include`, so this is a
// second copy of the march's own arithmetic, and it has to be the same
// arithmetic: a last-bit difference would put the air on a slightly different
// ray from the ground it is colouring, which stands still in a still frame and
// shimmers in a moving one. There is a test comparing the two as text.
//
// The march takes the viewport from its own uniform where this takes it from
// the depth buffer's dimensions. They are always the same number -- both come
// from the one viewport `Scene::resize` hands out -- but that is an invariant
// rather than a proof, so it is written down here rather than assumed.
//
// Unnormalised, with its component along the view axis exactly one, which is
// what makes `distance_at` a multiply rather than a divide by a length.
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

// Where a world point sits in one cascade of a light volume. Must match
// `light_uvw` in `src/cloud_march.wgsl`; there is a test comparing these
// functions here as text against that file, because a shadow the ground draws in
// a different place from the one the cloud draws it is a cloud floating free of
// its own shadow.
fn light_uvw(p: vec3<f32>, frame: u32) -> vec3<f32> {
    let q = vec4<f32>(p, 1.0);
    let forward = light.frames[frame].to_texture;
    return vec3<f32>(dot(forward[0], q), dot(forward[1], q), dot(forward[2], q));
}

// A cascade's own vertical coordinate in the stacked volume. Must match
// `stacked_w` in `src/cloud_march.wgsl`.
fn stacked_w(w: f32, cascade: u32) -> f32 {
    let slices = f32(LIGHT_SLICES);
    let inside = clamp(w * slices, 0.5, slices - 0.5);
    return (f32(cascade) * slices + inside) / (slices * f32(LIGHT_CASCADES));
}

// How far outside a cascade a coordinate has strayed, as a share of it. Must
// match `beyond_light` in `src/cloud_march.wgsl`.
fn beyond_light(uvw: vec3<f32>, fade: vec3<f32>) -> f32 {
    let out = abs(uvw - vec3<f32>(0.5)) * 2.0 * fade;
    return max(max(out.x, out.y), out.z);
}

// How much of a light volume reaches a point, over all its cascades. Must match
// `reaching` in `src/cloud_march.wgsl` -- which is what the texture and the
// sampler are parameters for: the two callers hold different ones and everything
// else about the walk has to be the same.
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

// Must match `sun_reaching` in `src/cloud_march.wgsl`, but for the sampler:
// there the fields tile and are read through a repeating one, here everything
// else in the group is a table and the clamped one is already to hand.
fn sun_reaching(p: vec3<f32>) -> f32 {
    return reaching(light_volume, lut_sampler, p, SUN_FRAMES, SUN_CHANNEL);
}

// Must match `sky_reaching` in `src/cloud_march.wgsl`, likewise.
fn sky_reaching(p: vec3<f32>) -> f32 {
    return reaching(light_volume, lut_sampler, p, SKY_FRAMES, SKY_CHANNEL);
}

// Must match `sky_share` in `src/cloud_march.wgsl`.
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

// The zenith angle of the horizon. Must match `horizon_zenith` in
// `src/sky.wgsl`.
fn horizon_zenith(r: f32) -> f32 {
    return PI - asin(clamp(GROUND_RADIUS / max(r, GROUND_RADIUS), -1.0, 1.0));
}

// Must match `skyview_v` in `src/sky.wgsl`.
fn skyview_v(r: f32, zenith: f32) -> f32 {
    let horizon = horizon_zenith(r);
    if zenith < horizon {
        return 0.5 * (1.0 - sqrt(max(1.0 - zenith / horizon, 0.0)));
    }
    return 0.5 + 0.5 * sqrt(max((zenith - horizon) / (PI - horizon), 0.0));
}

// The sky in a direction, from the table. Must match `sample_skyview` in
// `src/sky.wgsl`, including that it reads below the horizon as well as above:
// the table marches air and never adds what the ground reflects, so its lower
// half is the haze a downward ray crosses, which is what a world whose terrain
// simply stops should show.
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

// Where a view-axis distance sits down the aerial-perspective volume. Must
// match `aerial_w` and `aerial_z` in `src/sky.wgsl`.
fn aerial_depth(along: f32) -> f32 {
    let w = sqrt(clamp(along, 0.0, AERIAL_FAR) / AERIAL_FAR);
    return w - 0.5 / f32(AERIAL_SLICES);
}

// Half the sun's apparent width, in radians: 0.5334 degrees across, which is
// what it is from this planet. Must match `SUN_ANGULAR_RADIUS` in `src/sky.rs`.
const SUN_ANGULAR_RADIUS: f32 = 0.004654;

// One over the disc's solid angle, `2 pi (1 - cos r)` = 6.805e-5 steradians.
//
// Everything else here is a radiance and the sun is given as an irradiance, so
// this is what turns one into the other: spreading the whole of the sun's light
// over the small patch of sky it actually occupies. Must match
// `SUN_DISC_RADIANCE` in `src/sky.rs`.
const SUN_DISC_RADIANCE: f32 = 14696.0;

// The sun itself, where the ray points at it.
//
// Only ever called on a pixel whose ray found no ground, which is what makes
// terrain occlude the disc: there is no separate visibility test and none is
// wanted, because the march has already answered that question exactly.
fn sun_disc(direction: vec3<f32>) -> vec3<f32> {
    let angle = acos(clamp(dot(direction, sky.sun.xyz), -1.0, 1.0));
    // One pixel of feather, so the edge is a curve rather than a staircase.
    // The disc is about six pixels across at 720 rows and a 60-degree field,
    // which is small enough that a hard cut is plainly stepped.
    let feather = sky.up.w;
    let edge = 1.0 - smoothstep(
        SUN_ANGULAR_RADIUS - feather,
        SUN_ANGULAR_RADIUS + feather,
        angle,
    );
    if edge <= 0.0 {
        return vec3<f32>(0.0);
    }

    // Limb darkening: the disc is a sphere of gas seen through more of its own
    // atmosphere at the edges than at the middle, so it is not flat. Invisible
    // while the sun is high -- everything here is thousands of times over the
    // white point and clips alike -- and the whole of the shape of it once the
    // air has taken the disc down near that point, which is the only time
    // anyone looks straight at one.
    let across = clamp(angle / SUN_ANGULAR_RADIUS, 0.0, 1.0);
    let limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - across * across, 0.0)));

    // Through the same air the sky in front of it went through, which is what
    // reddens the disc as it sets and what makes it vanish on its own once it
    // is down. No branch for "the sun has set": the table already answers that.
    let mu = dot(sky.up.xyz, direction);
    return vec3<f32>(SUN_DISC_RADIANCE * edge * limb) * sample_transmittance(sky.eye.w, mu);
}

// Radiance to a displayable colour: expose, then roll the top off.
//
// Extended Reinhard, per channel, and chosen over a fitted curve like ACES for
// one reason -- it inverts in closed form, so a test can predict a byte from a
// radiance instead of restating the shader. It is the identity to first order
// near zero, so dark ground still scales linearly with the light falling on it;
// it passes exactly through one at `WHITE`, so the white point is a number
// rather than a fitted constant; and it is monotone, so nothing ever crosses
// over anything else.
//
// In linear space, because the render target is sRGB and re-encodes on write --
// the same arrangement the palette has always relied on.
fn tonemap(radiance: vec3<f32>) -> vec3<f32> {
    let x = radiance * EXPOSURE;
    return saturate(x * (1.0 + x / (WHITE * WHITE)) / (1.0 + x));
}

// One step of the eight-bit sRGB target, in the linear units this shader writes.
//
// The frame is the only place in the renderer that is not floating point: the
// cloud is carried in `Rgba16Float` and the sky tables likewise, and then the
// whole thing is written to `Rgba8UnormSrgb`. Two hundred and fifty-six levels
// is plenty for a picture with texture in it and nowhere near enough for a
// smooth one, and a translucent cloud is the smoothest thing in this program --
// a veil of it over ground crosses sixteen levels in five hundred pixels, so the
// steps between them are flat bands thirty pixels wide.
//
// The target re-encodes on write, so a step is not a constant: it is the inverse
// slope of the sRGB transfer at this value. Below the toe the encode is linear
// and the step is constant; above it the encode is a power and the step grows
// with the value, which is the whole point of the curve and the reason a fixed
// dither would be far too coarse in the shadows and too fine in the highlights.
fn srgb_step(v: f32) -> f32 {
    if v <= 0.0031308 {
        return 1.0 / (255.0 * 12.92);
    }
    return pow(v, 1.0 - 1.0 / 2.4) * 2.4 / (255.0 * 1.055);
}

// A dither of half a step either way, so the bands become a boundary that moves
// pixel by pixel instead of one the eye can follow across the screen.
//
// Jorge Jimenez's interleaved gradient noise, which is a hash cheap enough to be
// free here and spread evenly enough that no pattern shows in it. A function of
// the pixel and of nothing else -- deliberately, and this is the one thing about
// it that is not free to change. Rolling it with the frame would look better
// still, being noise the eye averages away over a few frames rather than a fixed
// grain, but it would mean a settled still camera no longer draws the same bytes
// twice. That property is not decoration here: comparing consecutive frames of a
// motionless world byte for byte is how flicker is found in this renderer, and
// it is how the three faults before this one were located. A dither that moved
// would put noise into every one of those comparisons and hide the next fault
// under it. Static grain is the price and it is worth paying.
//
// Applied to all three channels alike so the noise it leaves is a wobble in
// brightness rather than in colour, which is far less visible at the same
// amplitude.
fn dithered(colour: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    let noise = fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
    let step = vec3<f32>(srgb_step(colour.r), srgb_step(colour.g), srgb_step(colour.b));
    return saturate(colour + (noise - 0.5) * step);
}

// How much of the sky's own distance a block and a pixel may disagree by before
// the block stops speaking for the pixel.
//
// A hundredth of `AERIAL_FAR`, so a kilometre of view-axis distance. Below that
// two taps are treated as describing the same place, which is what keeps a
// smooth slope from being upsampled through one tap at a time; above it a tap
// falls away as the reciprocal of how far out it is. A sky tap beside a ground
// pixel two kilometres off is down by a hundred, which is the case this is set
// for.
const BILATERAL_BIAS: f32 = 0.01;

// The cloud in front of one pixel, and where it was.
struct Cloud {
    // Scattered light, and what survives of everything behind it.
    lit: vec4<f32>,
    // The view-axis distance to put that light at, or zero where there is none.
    along: f32,
};

// How far the cloud march let a block of four pixels run, along the view axis.
//
// The same rule the march itself applies -- the farthest of the four, and
// unbounded if any of them found no ground -- so this reproduces the reach that
// produced the tap rather than guessing at it. In view-axis units where the
// march works in metres, which costs nothing: this is only ever compared
// against another number in the same units, never used to rebuild a position.
fn block_reach(block: vec2<i32>) -> f32 {
    let full = vec2<i32>(textureDimensions(depth));
    var reach = 0.0;
    for (var j = 0; j < 2; j += 1) {
        for (var i = 0; i < 2; i += 1) {
            let d = textureLoad(depth, min(block + vec2<i32>(i, j), full - 1), 0).r;
            if d == 0.0 {
                reach = AERIAL_FAR;
            } else {
                reach = max(reach, min(distance_at(d), AERIAL_FAR));
            }
        }
    }
    return reach;
}

// The half-resolution cloud buffer, read at a full-resolution pixel.
//
// A bilinear weight per tap, divided by how far that tap's block reached from
// what this pixel is showing. Across the open sky and across a continuous slope
// every tap agrees and this is an ordinary bilinear; at a silhouette the taps
// that marched past the ridge fall away by two orders of magnitude and the pixel
// is reconstructed from the ones that stopped where it did.
//
// The distance comes from the single best tap and is never blended, which is not
// fussiness: a blended distance addresses the aerial-perspective volume at a
// place no cloud is, and between a cloud at two kilometres and one at forty the
// average is a haze belonging to neither.
fn cloud_at(pixel: vec2<i32>, along: f32) -> Cloud {
    // Half-resolution texel `h` stands for the point where its block's four
    // pixels meet, which is full-resolution coordinate `2h + 1`; see
    // `cs_cloud_march`. So a pixel centre at `p + 0.5` sits at `(p - 0.5) / 2`
    // in the half-resolution grid.
    let at = (vec2<f32>(pixel) - 0.5) * 0.5;
    let corner = floor(at);
    let f = at - corner;
    let last = vec2<i32>(textureDimensions(cloud_colour)) - 1;

    var lit = vec4<f32>(0.0);
    var total = 0.0;
    var best = -1.0;
    var found = 0.0;
    var empty = true;
    for (var j = 0; j < 2; j += 1) {
        for (var i = 0; i < 2; i += 1) {
            let tap = clamp(vec2<i32>(corner) + vec2<i32>(i, j), vec2<i32>(0), last);
            let share = mix(1.0 - f.x, f.x, f32(i)) * mix(1.0 - f.y, f.y, f32(j));
            let apart = abs(block_reach(tap * 2) - along) / AERIAL_FAR;
            let weight = share / (apart + BILATERAL_BIAS);
            let marched = textureLoad(cloud_colour, tap, 0);
            empty = empty && marched.w >= 1.0;
            lit = lit + marched * weight;
            total = total + weight;
            if weight > best {
                best = weight;
                found = textureLoad(cloud_along, tap, 0).r;
            }
        }
    }

    var out: Cloud;
    // Read off the taps rather than off the average of them. Four taps that all
    // say "nothing here" average to a transmittance of one in exact
    // arithmetic and to one and a hair either side of it in this one: the
    // weights are summed twice, once into `total` and once through the alpha
    // channel, and nothing obliges a compiler to associate the two sums the same
    // way. Dividing then leaves a last-bit error, which is enough to take a
    // pixel down the compositing path and shift its colour by a count. It shows
    // as a fine magenta static over the whole frame when the two paths are
    // painted apart -- which is how this was found, since the difference from a
    // clear sky is otherwise a handful of counts nobody would see.
    if empty {
        out.lit = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        out.lit = lit / max(total, 1e-6);
    }
    // The buffer holds metres and the tables are addressed along the view axis,
    // which is more than a metre per unit anywhere but the middle of the frame.
    out.along = found / max(length(ray_raw_at(vec2<f32>(pixel) + 0.5)), 1e-6);
    return out;
}

// The light a pixel's air never had, and how much of it was in front of the
// cloud on its ray.
//
// One block's answer, since a block is two pixels across and a shaft is
// hundreds. What varies inside a block is how far each pixel reaches: across a
// silhouette one pixel is ridge and the next is sky forty kilometres beyond it,
// and the air in front of the two is very different air. So the march kept the
// light lost out to the block's nearest pixel as well as its farthest, and a
// pixel takes the share between them its own distance says. Along the view
// axis, as `along` is everywhere in this pass.
fn shafts_at(pixel: vec2<i32>, along: f32) -> vec4<f32> {
    let block = pixel / 2;
    let far = textureLoad(shafts_far, block, 0);
    let near = textureLoad(shafts_near, block, 0);
    let share = textureLoad(shafts_share, block, 0).r;
    let span = max((far.a - near.a) * 1000.0, 1e-3);
    let between = saturate((along - near.a * 1000.0) / span);
    return vec4<f32>(mix(near.rgb, far.rgb, between), share);
}

// The cloud put in front of what was already there.
//
// Everything the frame had is radiance measured at the eye, so it already
// carries the air between the eye and it. Putting something halfway along that
// path means undoing the near half of the air, inserting the cloud, and putting
// the near half back:
//
//   beyond = (background - haze(t)) / through(t)
//   final  = haze(t) + through(t) * (scattered + transmitted * beyond)
//
// which collapses to `background` exactly when the cloud is absent, and which
// is why a cloud reddens at sunset and dims the sun's disc for nothing extra --
// the transmittance multiplies the whole background, disc included.
//
// The clamp at zero is not decoration. The sky-view table and the aerial volume
// are built by different marches over the same air and disagree by a hair near
// the horizon, where `background` can come out a shade under the haze that is
// supposed to be part of it.
//
// `lost` is the light the shafts took out of the air in front of the cloud,
// which the haze read here still counts.
fn composite(
    background: vec3<f32>,
    screen: vec2<f32>,
    cloud: Cloud,
    lost: vec3<f32>,
) -> vec3<f32> {
    // A pixel with nothing in front of it takes the path it always took, to the
    // last bit. That is what lets a clear sky be checked against the frames this
    // pass drew before there were clouds in it.
    if cloud.lit.w >= 1.0 {
        return background;
    }
    let uvw = vec3<f32>(
        screen / vec2<f32>(textureDimensions(depth)),
        aerial_depth(cloud.along),
    );
    let haze = max(
        textureSampleLevel(aerial_scatter_lut, lut_sampler, uvw, 0.0).rgb - lost,
        vec3<f32>(0.0),
    );
    let through = max(
        textureSampleLevel(aerial_transmit_lut, lut_sampler, uvw, 0.0).rgb,
        vec3<f32>(1e-4),
    );
    let beyond = max((background - haze) / through, vec3<f32>(0.0));
    return haze + through * (cloud.lit.rgb + cloud.lit.w * beyond);
}

@vertex
fn vs_shade(@builtin(vertex_index)index: u32) -> @builtin(position) vec4<f32> {
    // The same oversized triangle the geometry pass draws.
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 1.0, 1.0);
}

@fragment
fn fs_shade(@builtin(position)clip: vec4<f32>) -> @location(0) vec4<f32> {
    // The fragment coordinate is the pixel centre, so truncation is the index.
    let pixel = vec2<i32>(clip.xy);

    // The march writes zero depth where its ray found no ground, and the
    // reversed-infinite projection cannot write zero for any finite hit, so
    // this test is exact: nothing is there and it is sky.
    //
    // Recomputed here every frame rather than stored, which is what keeps a
    // moving sun honest: the reprojection carries sky pixels between frames as
    // a fact about a *direction*, and a direction is all this needs.
    if textureLoad(depth, pixel, 0).r == 0.0 {
        let towards = normalize(ray_raw_at(clip.xy));
        // ... less the sunlight the sky-view table counted in air that a
        // mountain or a cloud keeps in shadow, which is a god ray seen from its
        // dark side.
        let shafts = shafts_at(pixel, AERIAL_FAR);
        let sky = max(sample_skyview(towards) - shafts.rgb, vec3<f32>(0.0)) + sun_disc(towards);
        // A sky ray reaches nothing, so as far as the cloud upsample is
        // concerned it is showing whatever is at the end of the table.
        let cloud = cloud_at(pixel, AERIAL_FAR);
        let shaded = composite(sky, clip.xy, cloud, shafts.rgb * shafts.a);
        return vec4<f32>(dithered(tonemap(shaded), clip.xy), 1.0);
    }

    let id = textureLoad(material, pixel, 0).r & MATERIAL_MASK;
    var albedo = MAGENTA.rgb;
    if id < PALETTE_SIZE {
        albedo = palette.colours[id].rgb;
    }

    // Where this pixel's ground actually is, rebuilt from the depth and the
    // same ray the march walked. The G-buffer stores no world position on
    // purpose -- a depth says the same thing in four bytes rather than sixteen
    // -- and this is the rebuild `src/deferred.rs` has always said a haze would
    // want. The sub-pixel offset is not applied: it moves the point by under
    // half a pixel, and the air at that scale is the same air.
    let raw = ray_raw_at(clip.xy);
    let ground = camera.position.xyz + raw * distance_at(textureLoad(depth, pixel, 0).r);
    // Into the planet's frame, whose centre sits one ground radius below the
    // world's origin. See the module doc of `src/sky.rs`.
    let centred = ground + vec3<f32>(0.0, GROUND_RADIUS, 0.0);
    let radius = max(length(centred), GROUND_RADIUS);
    let up = centred / radius;
    let sun_mu = dot(up, sky.sun.xyz);

    let surface = textureLoad(normal, pixel, 0).xyz;

    // Direct sun: what a surface at this angle collects, times what survives
    // the air between this patch of ground and space. That second factor is the
    // whole of what the old `SUNLIGHT` constant was standing in for, and it is
    // where a low sun turns orange -- the same table that reddens the sky
    // reddens what the sky is lighting.
    //
    // ... and by whatever cloud stands between this patch of ground and the
    // sun, which is the same volume the cloud march shades itself from, read at
    // the ground instead. There is no shadow map and there is no second
    // traversal: a cloud's shadow on a mountain and a cloud's shadow on another
    // cloud are one fact, and it was integrated once.
    //
    // ... and by the terrain itself, which is the shadow a mountain casts into
    // the valley behind it. Read off the same field the cloud and the air are,
    // so a ridge's shadow on the valley floor, on the fog pooled in it and in
    // the haze above it is one shadow -- asked as the ground, which is lit by
    // where the field's own ground stands rather than where this pixel's does.
    // See `shadow_tap`.
    let direct = sample_transmittance(radius, sun_mu)
        * max(dot(surface, sky.sun.xyz), 0.0)
        * sun_reaching(ground)
        * sun_clear(ground, true);

    // ... and everything else that arrives: the sky the terrain lets this patch
    // see, and the light the terrain around it sends back -- the warm fill a
    // sunlit slope throws into the valley in its shadow, and the dark of a
    // gully that sees a strip of sky. That is what the irradiance probes hold,
    // read off the cascades around this point. See `src/probe.wgsl`.
    //
    // Past the widest cascade, sixty-five kilometres out, the old stand-in
    // takes over: the multiple-scattering table, which *is* the sky's isotropic
    // radiance, times `pi` for a surface facing straight up and a wrap term for
    // one that faces the ground rather than the dome. Out there the haze in
    // front of the ground has the last word anyway.
    //
    // Dimmed by the cloud overhead as well, and that half is not optional:
    // taking the sun away and leaving the sky at full strength gives an
    // overcast landscape that is bright and flat rather than dim, which looks
    // less like weather than like the sun having been switched off. The probes
    // hold the light with that left out, so it is applied here, by the cloud
    // over this point.
    let probed = probe_irradiance(probe_field, lut_sampler, ground, surface);
    let open = sample_multiscatter(radius, sun_mu) * PI * (0.5 + 0.5 * dot(surface, up));
    let ambient = (probed.rgb + (1.0 - probed.a) * open) * sky_share(sky_reaching(ground));

    // A Lambertian surface scatters what lands on it over a hemisphere, which
    // is the `1 / pi`. The palette is stored linearised and is an albedo now
    // rather than a finished colour -- what fraction of each wavelength the
    // ground sends back -- and everything above is in units of the sun's own
    // irradiance, so this line is where radiance is made.
    let leaving = albedo * (direct + ambient) / PI;

    // ... and this is what the air between here and there does to it: takes
    // some of it out, and puts its own scattered light in front. Addressed by
    // the same view-axis distance the depth already encodes, at this pixel's
    // own place in the frame, so there is no normalise and no length -- the
    // froxel column standing over this pixel marched the very ray the march
    // did.
    //
    // The viewport comes from the depth buffer's dimensions, which is the same
    // number the froxel volume's own basis was built from.
    let along = distance_at(textureLoad(depth, pixel, 0).r);
    let uvw = vec3<f32>(
        clip.xy / vec2<f32>(textureDimensions(depth)),
        aerial_depth(along),
    );
    let haze = textureSampleLevel(aerial_scatter_lut, lut_sampler, uvw, 0.0).rgb;
    let through = textureSampleLevel(aerial_transmit_lut, lut_sampler, uvw, 0.0).rgb;
    // ... and then whatever cloud stands between the eye and all of that. The
    // ground, the air in front of it and the cloud in front of that are three
    // layers of one integral, composited in that order.
    // ... less what the aerial volume counted of the sun in air the terrain or
    // a cloud keeps it out of. See `src/shafts.wgsl`.
    let shafts = shafts_at(pixel, along);
    let background = max(leaving * through + haze - shafts.rgb, vec3<f32>(0.0));
    let cloud = cloud_at(pixel, along);
    return vec4<f32>(
        dithered(tonemap(composite(background, clip.xy, cloud, shafts.rgb * shafts.a)), clip.xy),
        1.0,
    );
}
