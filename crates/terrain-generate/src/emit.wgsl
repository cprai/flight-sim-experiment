// The per-texel half of the generator, as a compute shader.
//
// Emitting is 283 s of a 457 s run and almost all of it is this: one pure
// function of position evaluated 129 M times, which is the shape a GPU is for.
// The simulation half stays on the CPU -- see the commit that measured a
// compute-shader flood against the heap and lost.
//
// Everything here is a transcription of the Rust it shares a crate with, and
// the two are checked against each other rather than trusted:
// `noise.rs`, `fields.rs` and `detail.rs` respectively. Where a constant
// appears twice it is because WGSL has no way to read the other one, and each
// such pair is named in the test that pins them together.
//
// Nothing in here allocates, branches on data in a way that diverges across a
// texel block, or reads anything but the five erosion channels. That is
// deliberate: the same code is meant to run inside the renderer at startup one
// day, against the same channels uploaded once.

struct Params {
    // Cells across and down the erosion grid.
    width: u32,
    rows: u32,
    // Ground one cell of that grid covers, in metres.
    cell_metres: f32,
    // Ground one texel of the level being written covers, in metres.
    texel_metres: f32,
    // Raster metres of this tile's first texel centre.
    origin: vec2<f32>,
    // Texels across the tile, and the seed the landscape was built from.
    tile_size: u32,
    seed: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> height: array<f32>;
@group(0) @binding(2) var<storage, read> hardness: array<f32>;
@group(0) @binding(3) var<storage, read> flow: array<f32>;
@group(0) @binding(4) var<storage, read> deposit: array<f32>;
@group(0) @binding(5) var<storage, read> filled: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_height: array<f32>;

// ---------------------------------------------------------------- noise.rs

const GRADIENT_COUNT: u32 = 16u;
const DIAGONAL: f32 = 0.70710678;
const NEAR_AXIS: f32 = 0.9238795;
const OFF_AXIS: f32 = 0.3826834;
const GRADIENT_SCALE: f32 = 1.4142136;
const MAX_OCTAVES: u32 = 12u;

// Unit vectors at every 22.5 degrees. Sixteen rather than eight: eight leaves
// gradient noise biased along the axes, which on a mountain reads as ridges
// that all run the same four ways.
fn noise_gradient_at(index: u32) -> vec2<f32> {
    switch index {
        case 0u: { return vec2<f32>(1.0, 0.0); }
        case 1u: { return vec2<f32>(NEAR_AXIS, OFF_AXIS); }
        case 2u: { return vec2<f32>(DIAGONAL, DIAGONAL); }
        case 3u: { return vec2<f32>(OFF_AXIS, NEAR_AXIS); }
        case 4u: { return vec2<f32>(0.0, 1.0); }
        case 5u: { return vec2<f32>(-OFF_AXIS, NEAR_AXIS); }
        case 6u: { return vec2<f32>(-DIAGONAL, DIAGONAL); }
        case 7u: { return vec2<f32>(-NEAR_AXIS, OFF_AXIS); }
        case 8u: { return vec2<f32>(-1.0, 0.0); }
        case 9u: { return vec2<f32>(-NEAR_AXIS, -OFF_AXIS); }
        case 10u: { return vec2<f32>(-DIAGONAL, -DIAGONAL); }
        case 11u: { return vec2<f32>(-OFF_AXIS, -NEAR_AXIS); }
        case 12u: { return vec2<f32>(0.0, -1.0); }
        case 13u: { return vec2<f32>(OFF_AXIS, -NEAR_AXIS); }
        case 14u: { return vec2<f32>(DIAGONAL, -DIAGONAL); }
        default: { return vec2<f32>(NEAR_AXIS, -OFF_AXIS); }
    }
}

fn noise_mix(bits: u32) -> u32 {
    var value = bits;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

// The two coordinates are folded in one at a time, each through its own mixer.
// Combining them first is cheaper and grows a herringbone, because `x` and `y`
// then meet only through a single xor and whole diagonals of the lattice
// collide.
fn noise_hash(x: i32, y: i32, seed: u32) -> u32 {
    var bits = seed * 0x9e3779b1u;
    bits = noise_mix(bits ^ (u32(x) * 0x3504f333u));
    bits = noise_mix(bits ^ (u32(y) * 0xf1bbcdcbu));
    return bits;
}

// Perlin's quintic interpolant, which has zero first *and* second derivative at
// both ends. The cubic leaves a second-derivative jump at every lattice line,
// invisible in the height and very visible in the shading.
fn noise_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn noise_lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

// Written out rather than using the builtin, so the CPU and this agree bit for
// bit on the ramps every threshold in the generator goes through.
fn noise_smoothstep(low: f32, high: f32, at: f32) -> f32 {
    let t = clamp((at - low) / (high - low), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn noise_corner(x: i32, y: i32, offset: vec2<f32>, seed: u32) -> f32 {
    let gradient = noise_gradient_at(noise_hash(x, y, seed) % GRADIENT_COUNT);
    return gradient.x * offset.x + gradient.y * offset.y;
}

// Gradient ("Perlin") noise, in -1..=1, zero at every lattice point.
fn noise_gradient(x: f32, y: f32, seed: u32) -> f32 {
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = x - x0;
    let fy = y - y0;
    let ix = i32(x0);
    let iy = i32(y0);

    let ux = noise_fade(fx);
    let uy = noise_fade(fy);
    let bottom = noise_lerp(
        noise_corner(ix, iy, vec2<f32>(fx, fy), seed),
        noise_corner(ix + 1, iy, vec2<f32>(fx - 1.0, fy), seed),
        ux,
    );
    let top = noise_lerp(
        noise_corner(ix, iy + 1, vec2<f32>(fx, fy - 1.0), seed),
        noise_corner(ix + 1, iy + 1, vec2<f32>(fx - 1.0, fy - 1.0), seed),
        ux,
    );
    return noise_lerp(bottom, top, uy) * GRADIENT_SCALE;
}

// A fractal, flattened out of the Rust struct: the lacunarity and gain are the
// only ones this crate ever uses, so they are constants here rather than fields
// nobody varies.
const LACUNARITY: f32 = 2.017;
const GAIN: f32 = 0.5;

// How many octaves survive a band limit at `finest`.
//
// Only the count of *summed* octaves moves; the normalisation below stays that
// of the whole fractal. That is the difference between a coarse level being the
// same surface with its finest octaves removed and it being a differently
// scaled one -- renormalising would make every coarse level about a seventh
// louder than the one under it, which draws as the ground breathing as a ring
// crosses it.
fn octaves_summed(wavelength: f32, octaves: u32, finest: f32) -> u32 {
    var summed = 0u;
    var at = wavelength;
    loop {
        if summed >= octaves || at < finest {
            break;
        }
        summed += 1u;
        at /= LACUNARITY;
    }
    return summed;
}

fn total_amplitude(octaves: u32) -> f32 {
    var total = 0.0;
    var amplitude = 1.0;
    for (var i = 0u; i < octaves; i += 1u) {
        total += amplitude;
        amplitude *= GAIN;
    }
    return total;
}

fn fbm(x: f32, y: f32, seed: u32, wavelength: f32, octaves: u32, finest: f32) -> f32 {
    let summed = min(octaves_summed(wavelength, octaves, finest), MAX_OCTAVES);
    if summed == 0u {
        return 0.0;
    }
    var frequency = 1.0 / wavelength;
    var amplitude = 1.0;
    var sum = 0.0;
    for (var octave = 0u; octave < summed; octave += 1u) {
        sum += noise_gradient(x * frequency, y * frequency, seed ^ (octave * 0x51ed270bu))
            * amplitude;
        frequency *= LACUNARITY;
        amplitude *= GAIN;
    }
    return sum / total_amplitude(octaves);
}

// Ridged multifractal, in 0..=1. Each octave is weighted by the one above it,
// so detail collects on the ridges and leaves the slopes between them smooth --
// that feedback is the whole difference between this and `1 - |fbm|`.
fn ridged(x: f32, y: f32, seed: u32, wavelength: f32, octaves: u32, finest: f32) -> f32 {
    let summed = min(octaves_summed(wavelength, octaves, finest), MAX_OCTAVES);
    if summed == 0u {
        return 0.0;
    }
    var frequency = 1.0 / wavelength;
    var amplitude = 1.0;
    var weight = 1.0;
    var sum = 0.0;
    for (var octave = 0u; octave < summed; octave += 1u) {
        let raw = noise_gradient(x * frequency, y * frequency, seed ^ (octave * 0x68e31da4u));
        var signal = 1.0 - abs(raw);
        signal *= signal;
        signal *= weight;
        weight = clamp(signal * 2.0, 0.0, 1.0);
        sum += signal * amplitude;
        frequency *= LACUNARITY;
        amplitude *= GAIN;
    }
    return clamp(sum / total_amplitude(octaves), 0.0, 1.0);
}

// Billowy noise, in 0..=1: the counterpart to `ridged`, and what deposited
// ground wants -- moraine, fans, anything dropped rather than cut.
fn billow(x: f32, y: f32, seed: u32, wavelength: f32, octaves: u32, finest: f32) -> f32 {
    let summed = min(octaves_summed(wavelength, octaves, finest), MAX_OCTAVES);
    if summed == 0u {
        return 0.0;
    }
    var frequency = 1.0 / wavelength;
    var amplitude = 1.0;
    var sum = 0.0;
    for (var octave = 0u; octave < summed; octave += 1u) {
        sum += abs(noise_gradient(x * frequency, y * frequency, seed ^ (octave * 0x1b56c4e9u)))
            * amplitude;
        frequency *= LACUNARITY;
        amplitude *= GAIN;
    }
    return clamp(sum / total_amplitude(octaves), 0.0, 1.0);
}

// --------------------------------------------------------------- fields.rs

// The Catmull-Rom cubic: the curve passes through the two middle nodes, and its
// slope at each is the central difference of that node's own neighbours. The
// four weights always add to one, so a flat field stays flat and a lake stays
// exactly at its own level.
fn catmull_rom_weights(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

// Clamped at the edges, so the world ends at the raster's edge and a read past
// it behaves like a mirror wall that neither drains nor floods.
fn channel_at(channel: u32, column: i32, row: i32) -> f32 {
    let c = u32(clamp(column, 0, i32(params.width) - 1));
    let r = u32(clamp(row, 0, i32(params.rows) - 1));
    let index = r * params.width + c;
    switch channel {
        case 0u: { return height[index]; }
        case 1u: { return hardness[index]; }
        case 2u: { return flow[index]; }
        case 3u: { return deposit[index]; }
        default: { return filled[index]; }
    }
}

// Catmull-Rom rather than bilinear, and the reason is the renderer's normals: a
// bilinear surface is continuous but its gradient is not, so every cell line
// draws as a crease and a sixteen-metre grid of them covers the world in
// axis-aligned facets.
fn sample_smooth(channel: u32, column: f32, row: f32) -> f32 {
    let c0 = floor(column);
    let r0 = floor(row);
    let across = catmull_rom_weights(column - c0);
    let down = catmull_rom_weights(row - r0);
    let ic = i32(c0);
    let ir = i32(r0);

    var total = 0.0;
    for (var dy = 0; dy < 4; dy += 1) {
        var line = 0.0;
        for (var dx = 0; dx < 4; dx += 1) {
            line += across[dx] * channel_at(channel, ic + dx - 1, ir + dy - 1);
        }
        total += down[dy] * line;
    }
    return total;
}

struct Sample {
    height: f32,
    hardness: f32,
    flow: f32,
    deposit: f32,
    filled: f32,
    slope: f32,
    aspect: vec2<f32>,
}

// The slope and aspect are central differences one cell either side, so they
// describe the *landform* -- whether this is a cliff or a valley floor --
// rather than the roughness the detail pass is about to add on top.
fn sample_fields(x: f32, y: f32) -> Sample {
    let column = x / params.cell_metres;
    let row = y / params.cell_metres;

    let east = sample_smooth(0u, column + 1.0, row);
    let west = sample_smooth(0u, column - 1.0, row);
    let south = sample_smooth(0u, column, row + 1.0);
    let north = sample_smooth(0u, column, row - 1.0);
    let span = 2.0 * params.cell_metres;
    let fall_east = (west - east) / span;
    let fall_south = (north - south) / span;
    let slope = sqrt(fall_east * fall_east + fall_south * fall_south);
    var aspect = vec2<f32>(0.0, 0.0);
    if slope > 1e-6 {
        aspect = vec2<f32>(fall_east / slope, fall_south / slope);
    }

    var sample: Sample;
    sample.height = sample_smooth(0u, column, row);
    sample.hardness = sample_smooth(1u, column, row);
    sample.flow = sample_smooth(2u, column, row);
    sample.deposit = sample_smooth(3u, column, row);
    sample.filled = sample_smooth(4u, column, row);
    sample.slope = slope;
    sample.aspect = aspect;
    return sample;
}

fn water_depth(sample: Sample) -> f32 {
    return max(sample.filled - sample.height, 0.0);
}

// --------------------------------------------------------------- detail.rs

const TEXTURE_WAVELENGTH: f32 = 512.0;
const TEXTURE_OCTAVES: u32 = 10u;
const TEXTURE_METRES: f32 = 5.0;
const RIB_WAVELENGTH: f32 = 380.0;
const RIB_OCTAVES: u32 = 6u;
const RIB_METRES: f32 = 15.0;
const MORAINE_WAVELENGTH: f32 = 220.0;
const MORAINE_OCTAVES: u32 = 4u;
const MORAINE_METRES: f32 = 6.0;
const GENTLE_SLOPE: f32 = 0.25;
const STEEP_SLOPE: f32 = 0.90;
const FILLED_METRES: f32 = 6.0;
const SHORE_METRES: f32 = 0.4;
const LAKE_METRES: f32 = 2.5;
const CHANNEL_FLOW: f32 = 11.5;
const RIVER_FLOW: f32 = 15.5;
const CHANNEL_METRES: f32 = 4.0;
const CHANNEL_WIDTH_CELLS: f32 = 2.5;

// How much of a feature survives at a given texel size, 0..=1. The same rule the
// fractals band-limit themselves by, applied to a feature that comes out of the
// simulation rather than out of noise.
fn resolvable(feature_metres: f32, texel_metres: f32) -> f32 {
    return noise_smoothstep(0.5, 1.5, feature_metres / (2.0 * texel_metres));
}

struct Ground {
    steepness: f32,
    rockiness: f32,
    filling: f32,
    lake: f32,
    channel: f32,
}

fn ground_of(sample: Sample) -> Ground {
    let steepness = noise_smoothstep(GENTLE_SLOPE, STEEP_SLOPE, sample.slope);
    let lake = noise_smoothstep(SHORE_METRES, LAKE_METRES, water_depth(sample));
    let channel = noise_smoothstep(CHANNEL_FLOW, RIVER_FLOW, sample.flow)
        * (1.0 - lake)
        * resolvable(CHANNEL_WIDTH_CELLS * params.cell_metres, params.texel_metres);

    var ground: Ground;
    ground.steepness = steepness;
    ground.rockiness = steepness * (0.35 + 0.65 * sample.hardness);
    ground.filling = noise_smoothstep(0.0, FILLED_METRES, sample.deposit);
    ground.lake = lake;
    ground.channel = channel;
    return ground;
}

// The bare earth at a texel: the coarse surface with fractal detail on it,
// band-limited to what this level can hold. An octave whose features are
// narrower than two texels cannot be represented here, and summing it anyway is
// how a pyramid starts to shimmer.
fn bare_height(sample: Sample, ground: Ground, x: f32, y: f32) -> f32 {
    let finest = 2.0 * params.texel_metres;
    let seed = params.seed;

    let texture = fbm(x, y, seed ^ 0x7c1a3f55u, TEXTURE_WAVELENGTH, TEXTURE_OCTAVES, finest);
    // Centred, so ribs raise and gully alike rather than lifting the whole face.
    let ribs = ridged(x, y, seed ^ 0x2b90d417u, RIB_WAVELENGTH, RIB_OCTAVES, finest) - 0.5;
    // Centred as well, so hummocks sit either side of the floor the water left.
    let moraine = billow(
        x, y, seed ^ 0x9d5486e1u, MORAINE_WAVELENGTH, MORAINE_OCTAVES, finest,
    ) - 0.5;

    // Rough where it is steep, smooth where the water dropped its load.
    let texture_metres =
        TEXTURE_METRES * (0.25 + 1.5 * ground.steepness) * (1.0 - 0.7 * ground.filling);
    let rib_metres = RIB_METRES * ground.rockiness * (1.0 - ground.filling);
    let moraine_metres = MORAINE_METRES * ground.filling * (1.0 - ground.steepness);

    let land = sample.height
        + texture * texture_metres
        + ribs * rib_metres
        + moraine * moraine_metres
        - ground.channel * CHANNEL_METRES;

    // Standing water is flat, and it is the surface the renderer draws: there is
    // no separate water pass, so a lake is terrain at the lake's level.
    return noise_lerp(land, sample.filled, ground.lake);
}

@compute @workgroup_size(8, 8)
fn cs_bare(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.tile_size || id.y >= params.tile_size {
        return;
    }
    let x = params.origin.x + f32(id.x) * params.texel_metres;
    let y = params.origin.y + f32(id.y) * params.texel_metres;
    let sample = sample_fields(x, y);
    let ground = ground_of(sample);
    out_height[id.y * params.tile_size + id.x] = bare_height(sample, ground, x, y);
}
