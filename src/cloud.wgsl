// The 3D noise a cloud is carved out of, built once at load.
//
// Two volumes. The shape is what decides where cloud is at all -- billows a few
// kilometres across, which a weather map then thresholds into cover. The detail
// is what decides what its edges look like close up -- wisps a couple of hundred
// metres across, which erode the shape rather than adding to it.
//
// Both tile. A cloud field spans a hundred kilometres and these are 128 and 32
// texels square, so they are read wrapped, over and over, and a lattice that did
// not wrap would draw a seam every few kilometres across the whole sky. Tiling
// is not a nicety here; it is the only reason a volume this small can cover a
// world this large. Every lattice cell index is therefore folded back into the
// period before it is hashed, which is what makes the field periodic in one
// unit exactly, and what the tiling test checks.
//
// The recipe is Schneider's, from "The Real-time Volumetric Cloudscapes of
// Horizon Zero Dawn" (SIGGRAPH 2015 Advances in Real-Time Rendering): a
// Perlin-Worley base for the billows, and Worley at rising frequencies to eat
// into them.

// Sides of the two volumes, in texels. Must match `SHAPE_SIZE` and
// `DETAIL_SIZE` in `src/cloud.rs`, which is where the textures are made.
const SHAPE_SIZE: u32 = 128u;
const DETAIL_SIZE: u32 = 32u;

// Lattice cells across the unit cube, per channel.
//
// Every one is a power of two and every one divides the volume it is built
// into, so a cell is a whole number of texels and the periods of the channels
// agree with each other as well as with the texture. The coarsest is four:
// fewer and the billows are so large that the volume reads as one blob when it
// is tiled, more and the tile repeats visibly.
const SHAPE_CELLS: i32 = 4;
const DETAIL_CELLS: i32 = 2;

// Octaves summed into the shape's two terms.
//
// Two each, which is fewer than the recipe calls for and was arrived at by
// looking. The base shape is thresholded by a coverage map and then eroded by
// the three single-frequency Worley channels beside it and by the detail
// volume, so every octave summed in here is one the erosion will go over again
// -- and at four the sum reads as static rather than as billows. What this
// channel is for is the large structure; the small structure has three other
// places to come from.
const PERLIN_OCTAVES: u32 = 2u;
const WORLEY_OCTAVES: u32 = 2u;

// Seeds. Separate per field, so that two channels built from the same lattice
// size are not the same field twice.
const PERLIN_SEED: u32 = 0x50726c6eu;
const WORLEY_SEED: u32 = 0x576f726cu;
const DETAIL_SEED: u32 = 0x44746c73u;

// Side of the weather map, in texels, and how many decks it describes. Must
// match `WEATHER_SIZE` and `DECKS` in `src/cloud.rs`.
//
// Two hundred and fifty-six over sixty kilometres puts a texel at about two
// hundred and thirty metres, which is far coarser than a cloud and exactly
// right for what this holds: not cloud, but where cloud is *allowed* -- the
// shape of the weather rather than the shape of a cumulus. The volumes above
// supply everything finer.
const WEATHER_SIZE: u32 = 256u;
const DECKS: u32 = 3u;

// Lattice cells across one tile of the weather map.
//
// Three, so a tile holds three or four weather systems across sixty kilometres
// -- about the scale real cover varies on. It is also the period of the time
// axis, in the sense that the field returns to itself after `clock.y` seconds.
const WEATHER_CELLS: i32 = 3;

// Octaves of the cover field, and of the three fields beside it.
//
// Three for cover: the first is the front, the second the breaks in it, the
// third the ragged edges of those. One for the rest, because what they say --
// which way a patch of cloud leans, how dense it is, how high its base sits --
// varies over a whole weather system and not within one. Giving them three
// apiece doubled what this pass cost and changed nothing anybody could see.
const WEATHER_OCTAVES: u32 = 3u;
const WEATHER_SLOW_OCTAVES: u32 = 1u;

// What one deck of one preset looks like.
struct Deck {
    // The field values that map to no cloud and to solid cloud, then which way
    // the deck leans -- nothing for flat stratus, one for heaped cumulus -- and
    // how dense it is where it is solid.
    look: vec4<f32>,
    // The seed this deck's field is drawn from, and three spare.
    seed: vec4<u32>,
};

struct Weather {
    decks: array<Deck, 3>,
    // Seconds since the world started, then how long the weather takes to come
    // back round to where it was. The rest is spare.
    clock: vec4<f32>,
};

@group(1) @binding(0) var<uniform> weather: Weather;

@group(3) @binding(0) var out_shape: texture_storage_3d<rgba8unorm, write>;
@group(3) @binding(1) var out_detail: texture_storage_3d<rgba8unorm, write>;
@group(3) @binding(2) var out_weather: texture_storage_2d_array<rgba8unorm, write>;

// Wellons' `lowbias32`. Must match `noise_mix` in `src/terrain.wgsl` character
// for character; there is no preprocessor here and a test compares the two as
// text. Two copies of a mixer is not a risk in itself -- nothing would break if
// they differed -- but a reader who found two spellings would have to work out
// whether the difference meant something, and it does not.
fn noise_mix(bits: u32) -> u32 {
    var b = bits;
    b ^= b >> 16u;
    b = b * 0x7feb352du;
    b ^= b >> 15u;
    b = b * 0x846ca68bu;
    b ^= b >> 16u;
    return b;
}

// Three coordinates folded in one at a time, each through its own mixer.
//
// The same rule `noise_hash` in `src/terrain.wgsl` follows and for the same
// reason: combining the coordinates before mixing lets whole diagonals of the
// lattice collide, and the noise grows a herringbone. A third axis only makes
// that worse -- the collisions become whole planes.
fn noise_hash(at: vec3<i32>, seed: u32) -> u32 {
    var bits = seed * 0x9e3779b1u;
    bits = noise_mix(bits ^ (u32(at.x) * 0x3504f333u));
    bits = noise_mix(bits ^ (u32(at.y) * 0xf1bbcdcbu));
    bits = noise_mix(bits ^ (u32(at.z) * 0x2545f491u));
    return bits;
}

// A lattice index folded back into one period.
//
// The whole of what makes these volumes tile. `%` in WGSL takes the sign of the
// dividend, so a negative index needs the second add to land in range -- and
// negative indices do arise, because the walk below visits the cell before the
// one it is standing in.
fn wrapped(at: vec3<i32>, cells: i32) -> vec3<i32> {
    return ((at % cells) + cells) % cells;
}

// Twelve gradients: the midpoints of a cube's edges.
//
// Ken Perlin's own set from "Improving Noise" (SIGGRAPH 2002), and the reason
// is the one he gives -- the obvious alternative, points on a sphere, needs a
// normalise per lattice corner and produces directional clumping unless the
// distribution is chosen carefully. These are uniform by construction, and each
// dot product is an add and a subtract rather than three multiplies.
const GRADIENTS = array<vec3<f32>, 12>(
    vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(-1.0, 1.0, 0.0),
    vec3<f32>(1.0, -1.0, 0.0),
    vec3<f32>(-1.0, -1.0, 0.0),
    vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(-1.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, -1.0),
    vec3<f32>(-1.0, 0.0, -1.0),
    vec3<f32>(0.0, 1.0, 1.0),
    vec3<f32>(0.0, -1.0, 1.0),
    vec3<f32>(0.0, 1.0, -1.0),
    vec3<f32>(0.0, -1.0, -1.0),
);

// Perlin's quintic fade. Must match `noise_fade` in `src/terrain.wgsl`.
//
// The cubic would do for a field nobody differentiates, and this is not that:
// the cloud's lighting comes from how fast its density changes, so a jump in
// the second derivative at every lattice plane would show as faceting in the
// shading exactly the way it does on terrain normals.
fn noise_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn noise_corner(at: vec3<i32>, offset: vec3<f32>, cells: i32, seed: u32) -> f32 {
    let index = noise_hash(wrapped(at, cells), seed) % 12u;
    return dot(GRADIENTS[index], offset);
}

// Gradient noise in `-1..=1`, periodic in `cells` lattice steps.
fn perlin(p: vec3<f32>, cells: i32, seed: u32) -> f32 {
    let scaled = p * f32(cells);
    let base = floor(scaled);
    let f = scaled - base;
    let at = vec3<i32>(base);
    let u = vec3<f32>(noise_fade(f.x), noise_fade(f.y), noise_fade(f.z));

    var edges: array<f32, 4>;
    for (var i = 0u; i < 4u; i += 1u) {
        let dy = i32(i & 1u);
        let dz = i32(i >> 1u);
        let corner = at + vec3<i32>(0, dy, dz);
        let offset = f - vec3<f32>(0.0, f32(dy), f32(dz));
        edges[i] = mix(
            noise_corner(corner, offset, cells, seed),
            noise_corner(corner + vec3<i32>(1, 0, 0), offset - vec3<f32>(1.0, 0.0, 0.0), cells, seed),
            u.x,
        );
    }
    let front = mix(edges[0], edges[1], u.y);
    let back = mix(edges[2], edges[3], u.y);
    // Root three over two, which is the largest a three-dimensional gradient
    // noise built from unit-edge gradients can reach; without it the field
    // never uses the top of its range and every threshold below is off.
    return mix(front, back, u.z) * 1.1547005;
}

// One over `n`, as a billow: one where a cell's point is, falling away from it.
//
// Inverted from the usual distance field because cloud is made of lumps rather
// than of the gaps between them. The walk is the 3x3x3 around the cell the
// sample stands in, which is what `crown_at` in `src/terrain.wgsl` does in two
// dimensions -- and the jitter is read out of one hash in three fields for the
// reason given there: the mixer's avalanche has already made every output bit
// depend on every input bit, so splitting one word beats taking three.
fn worley(p: vec3<f32>, cells: i32, seed: u32) -> f32 {
    let scaled = p * f32(cells);
    let home = vec3<i32>(floor(scaled));
    var nearest = 2.0;
    for (var dz = -1; dz <= 1; dz += 1) {
        for (var dy = -1; dy <= 1; dy += 1) {
            for (var dx = -1; dx <= 1; dx += 1) {
                let cell = home + vec3<i32>(dx, dy, dz);
                // Hashed wrapped so the field tiles; measured unwrapped so the
                // distance across the seam is the distance it looks like.
                let bits = noise_hash(wrapped(cell, cells), seed);
                let jitter = vec3<f32>(
                    f32(bits & 0x3ffu),
                    f32((bits >> 10u) & 0x3ffu),
                    f32((bits >> 20u) & 0x3ffu),
                ) * (1.0 / 1024.0);
                nearest = min(nearest, distance(scaled, vec3<f32>(cell) + jitter));
            }
        }
    }
    return 1.0 - clamp(nearest, 0.0, 1.0);
}

// Worley summed over octaves, each twice the frequency and half the weight.
fn worley_fractal(p: vec3<f32>, cells: i32, seed: u32) -> f32 {
    var sum = 0.0;
    var weight = 0.0;
    var frequency = cells;
    var amplitude = 1.0;
    for (var octave = 0u; octave < WORLEY_OCTAVES; octave += 1u) {
        sum += worley(p, frequency, seed + octave * 0x9e3779b9u) * amplitude;
        weight += amplitude;
        frequency *= 2;
        amplitude *= 0.5;
    }
    return sum / weight;
}

fn perlin_fractal(p: vec3<f32>, cells: i32, seed: u32) -> f32 {
    var sum = 0.0;
    var weight = 0.0;
    var frequency = cells;
    var amplitude = 1.0;
    for (var octave = 0u; octave < PERLIN_OCTAVES; octave += 1u) {
        sum += perlin(p, frequency, seed + octave * 0x9e3779b9u) * amplitude;
        weight += amplitude;
        frequency *= 2;
        amplitude *= 0.5;
    }
    // Into `0..1`, which is what the storage format holds and what the remap
    // below assumes of it.
    return clamp(sum / weight * 0.5 + 0.5, 0.0, 1.0);
}

// How far the Perlin field may push a billow, either way.
//
// Chosen by looking, then measured: it brings the channel to a standard
// deviation of 0.150, against 0.17 to 0.18 for the single-frequency Worley
// channels beside it. Higher opens the gaps further and starts to read as
// static; lower and the billows stop joining up at all.
const WISP_WEIGHT: f32 = 0.6;

// Where the cloud is, before anything has eroded it.
//
// Worley alone gives billows with hard gaps between them; Perlin alone gives
// connected wisps with no body. Together they give lumps joined by strands,
// which is what a cumulus field looks like from a distance.
//
// Schneider's own combination is `remap(perlin, 0, 1, worleyFBM, 1)`, which
// lifts the billows *towards solid* wherever the Perlin field is high. That was
// written here first and measured: because a Perlin fractal averages a half,
// the remap lands almost everything between 0.6 and 0.86, a standard deviation
// of 0.078 against the 0.18 a single Worley octave gets. In a floating-point
// field that is merely a scale; in the eight bits this is stored in it is
// twenty usable levels out of 255, and a cloud edge drawn through twenty levels
// bands. So the Perlin term *displaces* the billows instead of lerping them,
// which keeps the mean near a half and the spread near the Worley it is made
// from, and still opens the gaps where the wisps run and closes them where they
// do not.
fn perlin_worley(p: vec3<f32>) -> f32 {
    let billows = worley_fractal(p, SHAPE_CELLS, WORLEY_SEED);
    let wisps = perlin_fractal(p, SHAPE_CELLS, PERLIN_SEED);
    return clamp(billows + (wisps - 0.5) * WISP_WEIGHT, 0.0, 1.0);
}

// Where a texel's centre sits in the unit cube.
//
// The half-texel is the same correction `to_texture` in `src/sky.wgsl` makes,
// and for the same reason -- a texel stands for the middle of what it covers,
// not for its near corner -- but unlike there, nothing here can tell if it is
// wrong. A table is addressed by a quantity that means something, so an offset
// table is read at the wrong altitude; this is noise, and noise shifted by half
// a texel is the same noise. The convention is followed because a volume that
// disagreed with every other sampled thing in the program would be a trap for
// whoever next wrote a reader, not because a test can catch it. One cannot: it
// was tried, and dropping this changes no property worth asserting.
fn texel_centre(id: vec3<u32>, size: u32) -> vec3<f32> {
    return (vec3<f32>(id) + 0.5) / f32(size);
}

@compute @workgroup_size(4, 4, 4)
fn cs_cloud_shape(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= vec3<u32>(SHAPE_SIZE)) {
        return;
    }
    let p = texel_centre(id, SHAPE_SIZE);
    // Three single frequencies rather than three fractals: the march wants to
    // choose how hard to erode and at what scale, and three octaves handed over
    // separately can be summed at any weights, where three fractals have
    // already had their weights chosen here.
    textureStore(
        out_shape,
        vec3<i32>(id),
        vec4<f32>(
            perlin_worley(p),
            worley(p, SHAPE_CELLS, WORLEY_SEED),
            worley(p, SHAPE_CELLS * 2, WORLEY_SEED + 1u),
            worley(p, SHAPE_CELLS * 4, WORLEY_SEED + 2u),
        ),
    );
}

// The weather field: a fractal over the map, with time as its third axis.
//
// Time being an axis of the *same* noise, rather than a second field blended
// over the first, is what makes the weather evolve rather than cross-fade. A
// front does not appear and disappear where it stands; it drifts in from
// somewhere, because moving along the third axis of a solid noise slides the
// features about in the other two. It costs nothing over a static field -- the
// noise was already three-dimensional.
//
// Periodic in all three, so the map tiles over the world and the weather comes
// back round after `clock.y` seconds. Nothing depends on the loop, but a field
// that returned to itself was free and one that ran forever would eventually
// walk out of the range `f32` holds a lattice index in exactly.
fn weather_field(at: vec2<f32>, seed: u32, octaves: u32) -> f32 {
    let t = weather.clock.x / max(weather.clock.y, 1.0);
    let p = vec3<f32>(at, fract(t));
    var sum = 0.0;
    var weight = 0.0;
    var cells = WEATHER_CELLS;
    var amplitude = 1.0;
    for (var octave = 0u; octave < octaves; octave += 1u) {
        sum += perlin(p, cells, seed + octave * 0x9e3779b9u) * amplitude;
        weight += amplitude;
        cells *= 2;
        amplitude *= 0.5;
    }
    return clamp(sum / weight * 0.5 + 0.5, 0.0, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn cs_cloud_weather(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= WEATHER_SIZE || id.y >= WEATHER_SIZE || id.z >= DECKS {
        return;
    }
    let deck = weather.decks[id.z];
    let at = (vec2<f32>(id.xy) + 0.5) / f32(WEATHER_SIZE);

    // Cover: the field stretched between the two values the preset called the
    // edges of its cloud. A preset that wants none of a deck puts both above
    // anything the field can reach, which is what makes `clear` exactly clear
    // rather than nearly clear -- and `clamp` of a negative is a hard zero, so
    // no eight-bit rounding can leave a speck of cloud behind.
    let field = weather_field(at, deck.seed.x, WEATHER_OCTAVES);
    let cover = clamp((field - deck.look.x) / max(deck.look.y - deck.look.x, 1e-3), 0.0, 1.0);

    // Which way this patch leans, and how dense it is. Both wander with fields
    // of their own so that a deck is not one kind of cloud everywhere: the
    // heaped part of a broken sky is heaped, and the part filling in behind it
    // is flatter. Half the preset's figure, half the field.
    let slow = WEATHER_SLOW_OCTAVES;
    let kind = clamp(deck.look.z * (0.5 + weather_field(at, deck.seed.x + 1u, slow)), 0.0, 1.0);
    let density = clamp(deck.look.w * (0.5 + weather_field(at, deck.seed.x + 2u, slow)), 0.0, 1.0);

    // Where in its slab this deck's base sits, as a fraction. A deck whose base
    // is flat everywhere reads as a ceiling; real cloud bases sag and lift by
    // hundreds of metres across a sky, and this is the field that says so.
    let base = weather_field(at, deck.seed.x + 3u, slow);

    textureStore(
        out_weather,
        vec2<i32>(id.xy),
        i32(id.z),
        vec4<f32>(cover, kind, density, base),
    );
}

@compute @workgroup_size(4, 4, 4)
fn cs_cloud_detail(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= vec3<u32>(DETAIL_SIZE)) {
        return;
    }
    let p = texel_centre(id, DETAIL_SIZE);
    let low = worley(p, DETAIL_CELLS, DETAIL_SEED);
    let mid = worley(p, DETAIL_CELLS * 2, DETAIL_SEED + 1u);
    let high = worley(p, DETAIL_CELLS * 4, DETAIL_SEED + 2u);
    // The fourth channel is the fractal of the other three, which is what an
    // edge wants nine times out of ten. It costs nothing to store and saves the
    // march three fetches and two multiplies every time it does.
    textureStore(
        out_detail,
        vec3<i32>(id),
        vec4<f32>(low, mid, high, low * 0.625 + mid * 0.25 + high * 0.125),
    );
}
