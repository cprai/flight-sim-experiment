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
    // The relief the landscape spans. Every band the classifier draws is a
    // share of this rather than a fixed number of metres.
    valley_metres: f32,
    peak_metres: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> height: array<f32>;
@group(0) @binding(2) var<storage, read> hardness: array<f32>;
@group(0) @binding(3) var<storage, read> flow: array<f32>;
@group(0) @binding(4) var<storage, read> deposit: array<f32>;
@group(0) @binding(5) var<storage, read> filled: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_height: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_cover: array<u32>;

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
    let texture_metres = TEXTURE_METRES * (0.25 + 1.5 * ground.steepness) * (1.0 - 0.7 * ground.filling);
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

// ------------------------------------------------------------- classify.rs

// Material discriminants, from `terrain-materials`. Restated because WGSL
// cannot read the enum; `the_shader_knows_the_same_material_ids` pins them.
const MAT_NULL: u32 = 0u;
const MAT_LAKE: u32 = 0x0101u;
const MAT_RIVER: u32 = 0x0103u;
const MAT_STREAM: u32 = 0x0104u;
const MAT_MARSH: u32 = 0x0200u;
const MAT_BOG: u32 = 0x0202u;
const MAT_FOREST_NEEDLELEAVED: u32 = 0x0300u;
const MAT_FOREST_BROADLEAVED: u32 = 0x0301u;
const MAT_FOREST_MIXED: u32 = 0x0302u;
const MAT_SCRUB: u32 = 0x0400u;
const MAT_HEATH: u32 = 0x0402u;
const MAT_GRASSLAND: u32 = 0x0403u;
const MAT_MEADOW: u32 = 0x0405u;
const MAT_FELL: u32 = 0x0407u;
const MAT_BARE_ROCK: u32 = 0x0500u;
const MAT_SCREE: u32 = 0x0501u;
const MAT_SHINGLE: u32 = 0x0502u;
const MAT_SAND: u32 = 0x0503u;
const MAT_GLACIER: u32 = 0x0505u;
const MAT_BARE_EARTH: u32 = 0x0506u;

const TREELINE_SHARE: f32 = 0.52;
const SNOWLINE_SHARE: f32 = 0.76;
const LINE_WAVELENGTH: f32 = 2600.0;
const LINE_SHARE: f32 = 0.05;
const ASPECT_SHARE: f32 = 0.063;
const KRUMMHOLZ_SHARE: f32 = 0.137;
const ROCK_BAND_SHARE: f32 = 0.079;
const MOTTLE_WAVELENGTH: f32 = 140.0;
const MOTTLE_OCTAVES: u32 = 4u;
const ICE_STEEPNESS: f32 = 0.45;
const ROCK_THRESHOLD_WOODED: f32 = 0.86;
const ROCK_THRESHOLD_ALPINE: f32 = 0.42;
const SCREE_STEEPNESS: f32 = 0.30;
const SCREE_FILLING: f32 = 0.22;
const TIMBER_SHARE: f32 = 0.25;

// Every band is a share of the relief rather than a fixed number of metres: a
// fixed one is right for the range this crate generates by default and wrong
// for every other.
struct Lines {
    treeline: f32,
    snowline: f32,
    span: f32,
    band: f32,
    mottle: f32,
}

fn lines_of(sample: Sample, ground: Ground, x: f32, y: f32) -> Lines {
    // Not band-limited, unlike the mottle: the wobble is 2.6 km across and
    // survives every level the pyramid stores.
    let wobble = fbm(x, y, params.seed ^ 0x4b19c2e7u, LINE_WAVELENGTH, 3u, 0.0);
    // The southward component of downhill is how much the slope faces the sun.
    let sun = sample.aspect.y * ground.steepness;
    let base = params.valley_metres;
    let span = params.peak_metres - params.valley_metres;

    var lines: Lines;
    lines.span = span;
    lines.treeline = base + span * (TREELINE_SHARE + wobble * LINE_SHARE + sun * ASPECT_SHARE);
    lines.snowline = base + span * (SNOWLINE_SHARE + wobble * LINE_SHARE * 0.6 + sun * ASPECT_SHARE * 1.2);
    lines.band = clamp((sample.height - base) / max(span, 1.0), 0.0, 1.0);
    lines.mottle = fbm(
        x,
        y,
        params.seed ^ 0xa71f63b9u,
        MOTTLE_WAVELENGTH,
        MOTTLE_OCTAVES,
        2.0 * params.texel_metres,
    );
    return lines;
}

fn alpine(sample: Sample, ground: Ground, lines: Lines) -> u32 {
    let bare = noise_smoothstep(
        lines.treeline,
        lines.snowline,
        sample.height + lines.mottle * lines.span * LINE_SHARE,
    );
    if bare > 0.55 {
        return MAT_FELL;
    }
    if ground.steepness > 0.25 || sample.hardness > 0.6 {
        return MAT_HEATH;
    }
    return MAT_GRASSLAND;
}

fn wooded(sample: Sample, ground: Ground, lines: Lines) -> u32 {
    let krummholz_band = noise_smoothstep(
        lines.treeline - lines.span * KRUMMHOLZ_SHARE,
        lines.treeline,
        sample.height + lines.mottle * lines.span * LINE_SHARE * 0.75,
    );
    if krummholz_band > 0.6 {
        return MAT_SCRUB;
    }
    // Both qualified by the mottling as well as by height, so they arrive as
    // stands rather than as an altitude band -- a mixed belt ringing every
    // mountain at one height is the tell of a rule that keyed off elevation.
    let low = lines.band < 0.30 + lines.mottle * 0.08;
    let wet = sample.flow > CHANNEL_FLOW - 3.5;
    let sunny = sample.aspect.y > 0.25;
    if low && wet && ground.steepness < 0.3 {
        return MAT_FOREST_BROADLEAVED;
    }
    if low && (sunny || lines.mottle > 0.25) {
        return MAT_FOREST_MIXED;
    }
    return MAT_FOREST_NEEDLELEAVED;
}

fn valley_floor(sample: Sample, ground: Ground, lines: Lines) -> u32 {
    if ground.channel > 0.04 && ground.filling > 0.45 {
        return MAT_SHINGLE;
    }
    if ground.filling > 0.7 && lines.mottle > 0.45 {
        return MAT_SAND;
    }
    if ground.filling < 0.1 && sample.slope > 0.12 && lines.mottle < -0.5 {
        return MAT_BARE_EARTH;
    }
    if lines.mottle > 0.15 {
        return MAT_MEADOW;
    }
    return MAT_GRASSLAND;
}

fn cover_of(sample: Sample, ground: Ground, lines: Lines) -> u32 {
    // Water first: it covers whatever is underneath it.
    if ground.lake > 0.5 {
        return MAT_LAKE;
    }
    if ground.channel > 0.62 {
        return MAT_RIVER;
    }
    if ground.channel > 0.28 {
        return MAT_STREAM;
    }
    let boggy = sample.slope < 0.035 && sample.flow > CHANNEL_FLOW - 2.5;
    if ground.lake > 0.12 || (boggy && ground.filling > 0.25) {
        if sample.height > lines.treeline {
            return MAT_BOG;
        }
        return MAT_MARSH;
    }

    if sample.height > lines.snowline + lines.mottle * lines.span * LINE_SHARE
        && ground.steepness < ICE_STEEPNESS {
        return MAT_GLACIER;
    }

    // Rock and the talus under it are about the ground rather than the climate,
    // so they come before the treeline is consulted: a cliff is bare at any
    // altitude.
    let band = lines.span * ROCK_BAND_SHARE;
    let above_the_trees = noise_smoothstep(lines.treeline - band, lines.treeline + band, sample.height);
    let bare = noise_lerp(ROCK_THRESHOLD_WOODED, ROCK_THRESHOLD_ALPINE, above_the_trees);
    if ground.rockiness > bare + lines.mottle * 0.10 {
        return MAT_BARE_ROCK;
    }
    if ground.steepness > SCREE_STEEPNESS && ground.filling > SCREE_FILLING {
        return MAT_SCREE;
    }

    if sample.height > lines.treeline {
        return alpine(sample, ground, lines);
    }
    if sample.slope < 0.09 && lines.band < 0.35 {
        return valley_floor(sample, ground, lines);
    }
    return wooded(sample, ground, lines);
}

// The density is what share of the crown lattice holds a tree; the health
// scales the crown heights that lattice grows.
struct Trees {
    density: f32,
    health: f32,
}

fn timber(sample: Sample, ground: Ground, lines: Lines) -> Trees {
    let exposure = noise_smoothstep(
        lines.treeline - lines.span * TIMBER_SHARE,
        lines.treeline,
        sample.height,
    );
    // Steep ground holds less soil and fewer trees, and what it holds is
    // smaller. It does not stop the forest -- conifers root on slopes nobody
    // would walk up.
    let footing = 1.0 - noise_smoothstep(0.30, 0.85, ground.steepness);
    let damp = noise_smoothstep(CHANNEL_FLOW - 7.0, CHANNEL_FLOW - 2.0, sample.flow);

    var trees: Trees;
    trees.health = clamp(
        noise_lerp(1.0, 0.5, exposure) * noise_lerp(0.78, 1.0, footing)
            + 0.10 * damp
            + 0.06 * lines.mottle,
        0.0,
        1.0,
    );
    trees.density = clamp(
        noise_lerp(1.0, 0.55, exposure) * noise_lerp(0.78, 1.0, footing) + 0.16 * lines.mottle,
        0.0,
        1.0,
    );
    return trees;
}

fn krummholz(lines: Lines) -> Trees {
    var trees: Trees;
    trees.density = clamp(0.30 + 0.16 * lines.mottle, 0.0, 1.0);
    trees.health = clamp(0.17 + 0.05 * lines.mottle, 0.0, 1.0);
    return trees;
}

fn trees_of(sample: Sample, ground: Ground, lines: Lines, cover: u32) -> Trees {
    if cover == MAT_FOREST_NEEDLELEAVED
        || cover == MAT_FOREST_BROADLEAVED
        || cover == MAT_FOREST_MIXED {
        return timber(sample, ground, lines);
    }
    if cover == MAT_SCRUB {
        return krummholz(lines);
    }
    var none: Trees;
    none.density = 0.0;
    none.health = 0.0;
    return none;
}

// Two densities and a stature: the densities are the shares of each lattice
// holding a stone, the stature scales how tall they stand without touching how
// wide they are.
struct Rocks {
    boulders: f32,
    rubble: f32,
    stature: f32,
}

fn rocks_none() -> Rocks {
    var none: Rocks;
    none.boulders = 0.0;
    none.rubble = 0.0;
    none.stature = 0.0;
    return none;
}

fn rocks_of(sample: Sample, ground: Ground, lines: Lines, cover: u32) -> Rocks {
    var stone: Rocks;
    if cover == MAT_SCREE {
        // Talus: the rubble follows the deposit channel, which is the pass that
        // recorded where material actually piled; the blocks follow the
        // rockiness, because hard beds shed hard corners.
        stone.boulders = clamp(0.25 + 0.30 * ground.rockiness + 0.20 * lines.mottle, 0.0, 1.0);
        stone.rubble = clamp(0.55 + 0.35 * ground.filling + 0.15 * lines.mottle, 0.0, 1.0);
        stone.stature = 0.7;
        return stone;
    }
    if cover == MAT_BARE_ROCK {
        // A face steep enough to read as bare rock has already shed anything
        // loose to the talus below it, so the rubble fades out with steepness
        // rather than following it up.
        stone.boulders = clamp(0.12 + 0.20 * lines.mottle, 0.0, 1.0);
        stone.rubble = clamp(0.15 * (1.0 - ground.steepness), 0.0, 1.0);
        stone.stature = 0.9;
        return stone;
    }
    if cover == MAT_SHINGLE {
        // A river that could roll a five-metre block would not have left it
        // here, so the stature is low rather than the densities.
        stone.boulders = 0.05;
        stone.rubble = clamp(0.55 + 0.20 * lines.mottle, 0.0, 1.0);
        stone.stature = 0.35;
        return stone;
    }
    let frosted = cover == MAT_FELL || cover == MAT_HEATH;
    let floor_kind = cover == MAT_GRASSLAND
        || cover == MAT_MEADOW
        || cover == MAT_SAND
        || cover == MAT_BARE_EARTH;
    // `Grassland` comes out of both `alpine` and `floor`, and which one it was
    // decides what is lying on it: frost-shattered plateau above the treeline,
    // ground the ice left below it.
    if frosted || (floor_kind && sample.height > lines.treeline) {
        // Frost splits the bed it stands on rather than carrying anything, so
        // hardness is most of the answer and slope is almost none of it.
        stone.boulders = clamp(
            0.18 + 0.25 * sample.hardness * (1.0 - ground.steepness) + 0.15 * lines.mottle,
            0.0,
            1.0,
        );
        stone.rubble = clamp(0.35 + 0.30 * sample.hardness + 0.15 * lines.mottle, 0.0, 1.0);
        stone.stature = 0.75;
        return stone;
    }
    if floor_kind {
        // Erratics: keyed on the filling so the blocks land on moraine and
        // outwash, where the ice actually dropped them, rather than on any flat
        // ground at all.
        stone.boulders = clamp(0.05 + 0.25 * ground.filling + 0.12 * lines.mottle, 0.0, 1.0);
        stone.rubble = clamp(0.10 * ground.filling, 0.0, 1.0);
        stone.stature = 1.0;
        return stone;
    }
    return rocks_none();
}

// ------------------------------------- terrain-canopy and terrain-rocks

const CANOPY_SPACING: f32 = 7.0;
const CANOPY_SHORTEST: f32 = 15.0;
const CANOPY_TALLEST: f32 = 28.0;
const CANOPY_RADIUS: f32 = 3.5;
const CANOPY_ROUNDNESS: f32 = 0.15;
const CANOPY_FLOOR: f32 = 0.35;
const CANOPY_EDGE: f32 = 0.30;
const CANOPY_SEED: u32 = 0x54726565u;
const CANOPY_NOISE_WAVELENGTH: f32 = 34.0;
const CLUMP_THINNEST: f32 = 0.6;
const CLUMP_THICKEST: f32 = 1.9;
const CANOPY_SILHOUETTE: f32 = 0.15;
const CANOPY_PAINTED: f32 = 0.25;

const BOULDER_SPACING: f32 = 24.0;
const RUBBLE_SPACING: f32 = 3.0;
const BOULDER_RADIUS: f32 = 5.0;
const RUBBLE_RADIUS: f32 = 1.2;
const BOULDER_SHORTEST: f32 = 2.5;
const BOULDER_TALLEST: f32 = 9.0;
const RUBBLE_SHORTEST: f32 = 0.4;
const RUBBLE_TALLEST: f32 = 1.6;
const STONE_ROUNDNESS: f32 = 0.9;
const STONE_EDGE: f32 = 0.30;
const BOULDER_SEED: u32 = 0x526f636bu;
const RUBBLE_SEED: u32 = 0x53637265u;
const FIELD_WAVELENGTH: f32 = 200.0;
const FIELD_EDGE: f32 = 0.42;
const FIELD_FULL: f32 = 0.78;
const FIELD_THICKEST: f32 = 2.2;
const STREW_WAVELENGTH: f32 = 60.0;
const STREW_THINNEST: f32 = 0.55;
const STREW_THICKEST: f32 = 1.7;
const STONE_SILHOUETTE: f32 = 0.08;
const BOULDERED: f32 = 0.07;
const STREWN: f32 = 0.30;
const MAT_CANOPY: u32 = 0x0304u;
const MAT_BOULDER: u32 = 0x0508u;
const MAT_RUBBLE: u32 = 0x0509u;

// Three independent smooth fields, each 0 to 1, out of one lattice. Four fields
// out of one word: splitting a hash beats taking four of them, and they are
// independent because the mixer's avalanche already made every output bit
// depend on every input bit.
fn lattice_fields(x: f32, y: f32, wavelength: f32, seed: u32) -> vec3<f32> {
    let u = x / wavelength;
    let v = y / wavelength;
    let cell_x = floor(u);
    let cell_y = floor(v);
    let fx = stand_fade(u - cell_x);
    let fy = stand_fade(v - cell_y);
    let ix = i32(cell_x);
    let iy = i32(cell_y);

    let c0 = noise_hash(ix, iy, seed);
    let c1 = noise_hash(ix + 1, iy, seed);
    let c2 = noise_hash(ix, iy + 1, seed);
    let c3 = noise_hash(ix + 1, iy + 1, seed);

    var out = vec3<f32>(0.0);
    for (var field = 0u; field < 3u; field += 1u) {
        let shift = 10u * field;
        let t0 = f32((c0 >> shift) & 0x3ffu) * (1.0 / 1023.0);
        let t1 = f32((c1 >> shift) & 0x3ffu) * (1.0 / 1023.0);
        let t2 = f32((c2 >> shift) & 0x3ffu) * (1.0 / 1023.0);
        let t3 = f32((c3 >> shift) & 0x3ffu) * (1.0 / 1023.0);
        let value = noise_lerp(noise_lerp(t0, t1, fx), noise_lerp(t2, t3, fx), fy);
        if field == 0u {
            out.x = value;
        } else if field == 1u {
            out.y = value;
        } else {
            out.z = value;
        }
    }
    return out;
}

// The canopy and rock crates ease with the cubic Hermite, not with the quintic
// `noise::fade` the gradient noise uses. They are different curves and the
// difference is not cosmetic: easing the crown lattice with the quintic instead
// made a closed stand bake 6.2 m short of what the crate bakes, because `grow`
// governs both how tall a crown is and how wide.
fn stand_fade(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn field_ramp(edge0: f32, edge1: f32, t: f32) -> f32 {
    return stand_fade(clamp((t - edge0) / (edge1 - edge0), 0.0, 1.0));
}

// A stand thins and thickens inside itself rather than being uniform out to its
// edge. Reaches well above one, which is harmless: every cell holds a tree and
// there is nothing further to add.
fn canopy_clump(x: f32, y: f32) -> f32 {
    let f = lattice_fields(x, y, CANOPY_NOISE_WAVELENGTH, CANOPY_SEED);
    return noise_lerp(CLUMP_THINNEST, CLUMP_THICKEST, f.z);
}

// A gate rather than a multiplier, which is what separates a boulder field from
// an even sprinkle of stones over a whole mountainside.
fn stone_field(x: f32, y: f32) -> f32 {
    let f = lattice_fields(x, y, FIELD_WAVELENGTH, BOULDER_SEED);
    return FIELD_THICKEST * field_ramp(FIELD_EDGE, FIELD_FULL, f.x);
}

// A multiplier and not a gate: rubble covers a talus slope everywhere, and what
// varies is how thickly.
fn stone_strew(x: f32, y: f32) -> f32 {
    let f = lattice_fields(x, y, STREW_WAVELENGTH, RUBBLE_SEED);
    return noise_lerp(STREW_THINNEST, STREW_THICKEST, f.y);
}

// A floor under the crowns, closing the gaps so a stand does not read as spikes
// standing on bare ground. It is *not* canopy: a sample standing on it is forest
// floor and wants the floor's own colour.
fn understorey(density: f32, health: f32) -> f32 {
    return CANOPY_FLOOR * CANOPY_SHORTEST * health * min(density, 1.0);
}

// One lattice walked over the nine cells around a point, for either a crown or a
// stone: they are the same shape with different numbers. Nine rather than one
// because they overlap -- a tree wide enough to close a canopy reaches out of
// its own cell -- and those nine hashes are the whole cost of this half.
fn scattered_at(
    x: f32,
    y: f32,
    density: f32,
    stature: f32,
    spacing: f32,
    radius: f32,
    shortest: f32,
    tallest: f32,
    roundness: f32,
    edge: f32,
    seed: u32,
) -> f32 {
    if density <= 0.0 || stature <= 0.0 {
        return 0.0;
    }
    let cell_x = i32(floor(x / spacing));
    let cell_y = i32(floor(y / spacing));
    var found = 0.0;

    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            let cx = cell_x + dx;
            let cy = cell_y + dy;
            let bits = noise_hash(cx, cy, seed);

            let jitter_x = f32(bits & 0x3ffu) * (1.0 / 1024.0);
            let jitter_y = f32((bits >> 10u) & 0x3ffu) * (1.0 / 1024.0);
            let grade = f32((bits >> 20u) & 0x3fu) * (1.0 / 64.0);
            let wants = f32((bits >> 26u) & 0x3fu) * (1.0 / 64.0);

            let grow = stand_fade(clamp((density - wants) / edge, 0.0, 1.0));
            if grow <= 0.0 {
                continue;
            }

            // Anywhere in its own cell, which is what stops the field drawing
            // as a grid.
            let middle_x = (f32(cx) + jitter_x) * spacing;
            let middle_y = (f32(cy) + jitter_y) * spacing;
            // A short one is a narrow one. Tying the two together stops the
            // field looking like one shape scaled up and down, and keeps every
            // radius under `radius`, which the nine-cell search relies on.
            let scale = grow * (0.72 + 0.28 * grade);
            let reach = max(radius * scale, 1.0 / 1024.0);
            let height = stature * noise_lerp(shortest, tallest, grade) * grow;

            let offset = vec2<f32>(x - middle_x, y - middle_y);
            let u = length(offset) / reach;
            if u < 1.0 {
                let cone = 1.0 - u;
                let dome = sqrt(max(1.0 - u * u, 0.0));
                found = max(found, height * noise_lerp(cone, dome, roundness));
            }
        }
    }
    return found;
}

fn canopy_samples(texel: f32) -> u32 {
    return clamp(u32(ceil(texel / (0.25 * CANOPY_RADIUS))), 4u, 32u);
}

fn stone_samples(texel: f32) -> u32 {
    return clamp(u32(ceil(texel / (0.25 * RUBBLE_RADIUS))), 4u, 32u);
}

// Buckets the order statistic below counts its samples into.
const BUCKETS: u32 = 16u;

// The mean of the tallest `share` of a block, from cumulative counts and sums.
//
// The answer has to be an *average* or it does not survive a change of texel
// size, and it has to be an average of the *tall* part or a distant forest draws
// as a green hillside twenty metres short of its own treetops. The crates get
// that by sorting the block and taking a prefix, and a shader cannot: a thousand
// samples is a thousand registers, and an array indexed by a running position
// lands in scratch memory.
//
// So the samples are counted into sixteen buckets on the way past --
// cumulatively, `counts[k]` being how many reached the k'th edge, which makes
// the accumulation sixteen fixed adds rather than one indexed one -- and the
// quantile is read back off them, interpolating inside whichever bucket the
// boundary falls in. That approximates the exact order statistic by at most the
// spread within one bucket, and approximates it the same way at every level,
// which is the property that actually matters: what must not move between two
// levels is the bias. This is the one place in the port where the shader does
// not compute what the crate computes, and `the_shader_and_the_crate_agree_
// about_what_stands` measures the gap rather than assuming it.
fn tallest_mean(
    counts: array<f32, 17>,
    sums: array<f32, 17>,
    samples: f32,
    share: f32,
) -> f32 {
    let taken = max(1.0, ceil(samples * share));
    var mean = 0.0;
    var found = false;
    // Downwards, so the first bucket holding enough samples is the tightest one
    // that does. The seventeenth entry is never written and is therefore zero,
    // which is what makes `k + 1` safe at the top.
    for (var k = i32(BUCKETS) - 1; k >= 0; k -= 1) {
        if !found && counts[k] >= taken {
            let above = counts[k + 1];
            let above_sum = sums[k + 1];
            let inside = max(counts[k] - above, 1e-6);
            mean = (above_sum + (sums[k] - above_sum) * (taken - above) / inside) / taken;
            found = true;
        }
    }
    return mean;
}

// What one texel carries once the crowns and stones on it have been looked at.
struct Standing {
    // How high what stands here reaches, in metres above the earth under it.
    lift: f32,
    // The id this texel should be painted with, or zero to keep the ground's
    // own. Taken from the same walk as the height, because a texel drawn as a
    // tree and painted as a meadow is worse than either.
    id: u32,
}

fn canopy_baked(centre: vec2<f32>, texel: f32, trees: Trees) -> Standing {
    let density = trees.density * canopy_clump(centre.x, centre.y);
    let health = trees.health;
    let across = canopy_samples(texel);
    let step = texel / f32(across);
    // Sample centres, so the block is symmetric about the texel and a texel
    // twice the size of its neighbour covers the same ground its four children
    // did between them.
    let first = 0.5 * step - 0.5 * texel;
    let floor_height = understorey(density, health);
    let top = max(health * CANOPY_TALLEST, 1e-6);

    var counts = array<f32, 17>();
    var sums = array<f32, 17>();
    var under = 0.0;
    var samples = 0.0;
    for (var row = 0u; row < across; row += 1u) {
        for (var column = 0u; column < across; column += 1u) {
            let at = centre + first + vec2<f32>(f32(column), f32(row)) * step;
            var here = scattered_at(
                at.x, at.y, density, health, CANOPY_SPACING, CANOPY_RADIUS,
                CANOPY_SHORTEST, CANOPY_TALLEST, CANOPY_ROUNDNESS, CANOPY_EDGE,
                CANOPY_SEED,
            );
            // The floor stands under every crown, and under the gaps too.
            if density > 0.0 && health > 0.0 {
                here = max(here, floor_height);
            }
            let rung = here * (f32(BUCKETS) / top);
            for (var k = 0; k < i32(BUCKETS); k += 1) {
                let hit = select(0.0, 1.0, f32(k) <= rung);
                counts[k] += hit;
                sums[k] += here * hit;
            }
            under += select(0.0, 1.0, here > floor_height);
            samples += 1.0;
        }
    }

    var out: Standing;
    out.lift = tallest_mean(counts, sums, samples, CANOPY_SILHOUETTE);
    // One rule at every level, and it means two different things without
    // needing to be told which: close up the block is inside a single crown or
    // the gap beside it, so this asks "is this texel a treetop"; far out it
    // spans a stand, so it asks "is this mostly wood". Both are the question the
    // pixel wants answered, and the two stone rules read the same way at both
    // ends.
    out.id = select(0u, MAT_CANOPY, under / samples >= CANOPY_PAINTED);
    return out;
}

fn stone_baked(centre: vec2<f32>, texel: f32, stone: Rocks) -> Standing {
    let boulders = stone.boulders * stone_field(centre.x, centre.y);
    let rubble = stone.rubble * stone_strew(centre.x, centre.y);
    let stature = stone.stature;
    let across = stone_samples(texel);
    let step = texel / f32(across);
    let first = 0.5 * step - 0.5 * texel;
    let top = max(stature * BOULDER_TALLEST, 1e-6);

    var counts = array<f32, 17>();
    var sums = array<f32, 17>();
    var under_boulder = 0.0;
    var under_stone = 0.0;
    var samples = 0.0;
    for (var row = 0u; row < across; row += 1u) {
        for (var column = 0u; column < across; column += 1u) {
            let at = centre + first + vec2<f32>(f32(column), f32(row)) * step;
            let block = scattered_at(
                at.x, at.y, boulders, stature, BOULDER_SPACING, BOULDER_RADIUS,
                BOULDER_SHORTEST, BOULDER_TALLEST, STONE_ROUNDNESS, STONE_EDGE,
                BOULDER_SEED,
            );
            let fine = scattered_at(
                at.x, at.y, rubble, stature, RUBBLE_SPACING, RUBBLE_RADIUS,
                RUBBLE_SHORTEST, RUBBLE_TALLEST, STONE_ROUNDNESS, STONE_EDGE,
                RUBBLE_SEED,
            );
            // Whichever class is higher here is the surface a ray meets.
            let here = max(block, fine);
            let rung = here * (f32(BUCKETS) / top);
            for (var k = 0; k < i32(BUCKETS); k += 1) {
                let hit = select(0.0, 1.0, f32(k) <= rung);
                counts[k] += hit;
                sums[k] += here * hit;
            }
            // There is no floor to compare against: the ground between the
            // stones is the ground, so anything above it is a stone.
            under_boulder += select(0.0, 1.0, block > 0.0);
            under_stone += select(0.0, 1.0, here > 0.0);
            samples += 1.0;
        }
    }

    var out: Standing;
    out.lift = tallest_mean(counts, sums, samples, STONE_SILHOUETTE);
    out.id = 0u;
    // Boulders before rubble, so a block lying in talus paints as the block: it
    // is the coarser of the two answers and the only one a texel at any distance
    // can actually resolve.
    if under_boulder / samples >= BOULDERED {
        out.id = MAT_BOULDER;
    } else if under_stone / samples >= STREWN {
        out.id = MAT_RUBBLE;
    }
    return out;
}

@compute @workgroup_size(8, 8)
fn cs_bare(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= params.tile_size || id.y >= params.tile_size {
        return;
    }
    let x = params.origin.x + f32(id.x) * params.texel_metres;
    let y = params.origin.y + f32(id.y) * params.texel_metres;
    let sample = sample_fields(x, y);
    let ground = ground_of(sample);
    out_height[id.y * params.tile_size + id.x] = bare_height(sample, ground, x, y);
}

// The classifier's three answers, which all come off one set of lines: the
// ground cover, what grows on it, and what is lying on it. One walk rather than
// three, because a texel drawn as a tree and painted as a meadow is worse than
// either -- and because the lines cost two fractals to build.
//
// Packed rather than returned as three products: the densities are shares in
// 0..=1, so eight bits each is finer than anything downstream distinguishes,
// and one buffer keeps the readback to one copy.
@compute @workgroup_size(8, 8)
fn cs_cover(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= params.tile_size || id.y >= params.tile_size {
        return;
    }
    let at = id.y * params.tile_size + id.x;
    let x = params.origin.x + f32(id.x) * params.texel_metres;
    let y = params.origin.y + f32(id.y) * params.texel_metres;
    let sample = sample_fields(x, y);
    let ground = ground_of(sample);
    let lines = lines_of(sample, ground, x, y);
    let cover = cover_of(sample, ground, lines);
    let trees = trees_of(sample, ground, lines, cover);
    let stone = rocks_of(sample, ground, lines, cover);

    out_cover[at] = cover;
    out_height[at * 5u] = trees.density;
    out_height[at * 5u + 1u] = trees.health;
    out_height[at * 5u + 2u] = stone.boulders;
    out_height[at * 5u + 3u] = stone.rubble;
    out_height[at * 5u + 4u] = stone.stature;
}

// A whole texel of both products: the height a ray meets and the id a pixel is
// painted with, from one walk of the ground under it.
//
// This is what `emit` writes, and the two answers have to come from the same
// walk for the reason the crates take them from one: a texel raised as a tree
// and painted as a meadow is worse than either of those on its own.
@compute @workgroup_size(8, 8)
fn cs_texel(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= params.tile_size || id.y >= params.tile_size {
        return;
    }
    let at = id.y * params.tile_size + id.x;
    let x = params.origin.x + f32(id.x) * params.texel_metres;
    let y = params.origin.y + f32(id.y) * params.texel_metres;

    let sample = sample_fields(x, y);
    let ground = ground_of(sample);
    let lines = lines_of(sample, ground, x, y);
    let cover = cover_of(sample, ground, lines);
    let bare = bare_height(sample, ground, x, y);

    let centre = vec2<f32>(x, y);
    let crowns = canopy_baked(centre, params.texel_metres, trees_of(sample, ground, lines, cover));
    let stones = stone_baked(centre, params.texel_metres, rocks_of(sample, ground, lines, cover));

    // The higher of the two rather than the sum. Both are surfaces standing on
    // the same ground and a ray meets whichever is above the other; adding them
    // would raise a boulder by the height of the trees beside it.
    out_height[at] = bare + max(crowns.lift, stones.lift);
    // Crowns first, because a closed stand hides whatever is under it from
    // above; then the stones, then the ground's own cover.
    if crowns.id != 0u {
        out_cover[at] = crowns.id;
    } else if stones.id != 0u {
        out_cover[at] = stones.id;
    } else {
        out_cover[at] = cover;
    }
}
