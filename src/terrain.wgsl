// Terrain drawn entirely by raymarching a max pyramid.
//
// There is no mesh. Every pixel of ground is found by walking a ray through a
// quadtree of maximum heights: skip a texel whenever the ray stays above the
// ceiling it holds, climb a level after every skip, descend a level when a
// texel might be hit, and at the finest level resident there solve against the
// surface itself. Tevs, Ihrke and Seidel, "Maximum Mipmaps for Fast, Accurate,
// and Scalable Dynamic Height Field Rendering", I3D 2008.
//
// The mip chain *is* the quadtree. Level `l` is mip `l - resident_base`, and
// holds one ceiling per level-`l` texel bounding every surface the renderer
// might draw across the closed square that texel covers, so climbing the
// quadtree is reading the next mip out. See
// `crates/terrain-tiles/src/maxima.rs` for what a cell means and why the bound
// has to hold for coarse levels as well as fine.
//
// The whole raster is resident from `resident_base` upwards and nothing
// streams, so a texel index *is* a texture coordinate: every level's mask is
// all ones and `slot` is the identity. The mask stays because it is what a
// generated level below the base will need, and because paying for it is one
// AND against a word `resident` has already loaded.
//
// Which level a point can be read at is decided by residency, and below the
// base there is none. That is the level of detail, and it needs no rule of its
// own: a ray simply finds nothing finer resident.

struct Camera {
    view_proj: mat4x4<f32>,
    // The projection that drew the previous frame, which is what a point's
    // motion is measured against.
    was_view_proj: mat4x4<f32>,
    position: vec4<f32>,
    // The ray basis: forward plus the screen axes, already scaled for the field
    // of view, so a pixel's direction is one mad each.
    ray_right: vec4<f32>,
    ray_up: vec4<f32>,
    ray_forward: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

// Must match `MAX_LEVELS` in `src/terrain/gpu.rs`.
const MAX_LEVELS: u32 = 16u;

// Threads per workgroup of the compacted march, which `cs_args` divides the
// hole count by to size the dispatch.
//
// Must match `MARCH_GROUP` in `src/reproject.rs` and `@workgroup_size` on
// `cs_march` below, which WGSL will not let this constant stand in for.
const MARCH_GROUP: u32 = 64u;

// Workgroups per row of the march's dispatch.
//
// An adapter caps how many workgroups one dimension of a dispatch may hold --
// 65535 on the baseline this asks for -- and an *indirect* dispatch is not
// validated against it, so exceeding it does not fail loudly. On this hardware
// it does not fail at all: the dispatch is dropped, the march writes nothing,
// and because nothing clears the G-buffer every pixel keeps whatever last
// reached it. The frame goes on looking like a frame.
//
// A screen's worth of rays passes that cap easily. At 3840x2160 a frame with no
// history is 129600 workgroups, twice over it, and it takes only one such frame
// to stick: the march writes nothing, so the G-buffer holds nothing to
// reproject from, so the next frame is another whole screen of rays.
//
// So lay the list out in rows instead. A thousand and twenty-four is arbitrary
// beyond being a power of two well under the cap; two dimensions of it reach
// four billion pixels, and one row is not dispatched until it is needed.
//
// Must match `MARCH_ROW` in `src/reproject.rs`.
const MARCH_ROW: u32 = 1024u;

struct Level {
    // The texels readable at this level, as a half-open range in this level's
    // own texels from the raster's origin. The whole level, while the whole
    // raster is resident: outside it there is no coarser level to fall to, only
    // the edge of the world. `slot` clamps into this range rather than holding
    // a texel back from it.
    valid_low: vec2<i32>,
    valid_high: vec2<i32>,
    // The highest ground anywhere resident at this level. A ray above it and
    // climbing has nothing to find here.
    ceiling: f32,
    padding: f32,
    // What wraps a texel index onto its texture coordinate at this level. All
    // ones for a resident level, whose index is its coordinate; a power-of-two
    // square below one is a mask rather than a modulo, and it is per level
    // because a resident chain and a window under it are different widths.
    mask: vec2<i32>,
};

struct Terrain {
    levels: array<Level, MAX_LEVELS>,
    // World XZ of the raster's texel (0, 0), and the size of one level-0 texel.
    // Together these turn a world position into a texel index at any level: the
    // level-`l` index is the level-0 one divided by `2^l`, exactly, which is
    // what lets a ray hand over between levels with nothing to correct.
    origin: vec2<f32>,
    metres_per_texel: vec2<f32>,
    // World XZ of the outermost samples the raster actually holds. Squares
    // reach past this, and everything out there is invented, so it is cut away
    // rather than drawn.
    data_min: vec2<f32>,
    data_max: vec2<f32>,
    level_count: u32,
    // The finest level worth descending to. Below it nothing is resident, or
    // its texels would be smaller than the pixels they land in.
    base_level: u32,
    // The level mip zero holds. Level `l` is mip `l - resident_base`.
    resident_base: u32,
    // How many texels a ray may cross before the march gives up on it.
    march_steps: u32,
    // The highest ground anywhere resident, across every level being marched.
    // A ray above it and climbing is sky.
    ceiling: f32,
    // How far past a cell wall a ray is put so the next step lands in the next
    // cell, in level-0 texels. Sized to the raster rather than fixed: see
    // `wall_nudge` in `src/terrain/gpu.rs`.
    wall_nudge: f32,
    // The target's size in pixels, which is what turns a pixel coordinate into
    // the ray through it. Carried in the uniform rather than interpolated from
    // the vertex stage so that every reader of a pixel's ray -- however it came
    // to be looking at that pixel -- derives it by exactly the same arithmetic.
    viewport: vec2<u32>,
};

@group(1) @binding(0) var<uniform> terrain: Terrain;
// Non-filterable on purpose: heights and ceilings are only ever fetched at
// exact texel centres, never sampled, so no float-filtering support is needed.
// Materials could not be filtered even in principle: ids are labels, and a
// blend of two labels is a third, wrong, label.
@group(1) @binding(1) var heights: texture_2d<f32>;
@group(1) @binding(2) var materials: texture_2d<u32>;
@group(1) @binding(3) var maxima: texture_2d<f32>;
// Levels below the base, which are generated rather than measured. A window of
// whole tiles around the camera per level, wrapped onto its slots the way the
// streamed clipmap was -- there is no measured ground under the base to hold
// whole, so these are the one thing left that moves.
@group(1) @binding(4) var detail_heights: texture_2d_array<f32>;
@group(1) @binding(5) var detail_maxima: texture_2d_array<f32>;
// How much of the chain's surface is what stands on it rather than what the
// survey measured, in metres. A mip per level, and read by nothing that draws:
// the coarse mips are how a level's lift reaches the level above it, and mip
// zero is what `cs_detail` subtracts to recover the bare earth -- a generated
// level grows its own crowns at its own texel size, and interpolating the base's
// would have it grow them twice. Half precision, which a lift of tens of metres
// is far more than served by.
@group(1) @binding(6) var lift: texture_2d<f32>;
// What a generated texel is painted as: the survey's ground cover upscaled onto
// this level's texels, or what stands on it where a crown or a stone hides the
// ground. One id per texel, complete in itself -- the resident chain is not
// consulted for a texel this covers.
@group(1) @binding(7) var detail_materials: texture_2d_array<u32>;

// Elevations below this are the raster's nodata rather than ground.
//
// HRDEM writes -32767. The exact value is not worth passing in: the deepest
// ground on Earth is a small fraction of this, so anything below it is a hole
// however the producer chose to spell it. Kept in step with `NODATA_BELOW` in
// `crates/terrain-tiles/src/texel.rs`, which is where the filter that drops
// these texels when it builds a coarse level reads the same threshold.
const NODATA_BELOW: f32 = -30000.0;

// Halvings used to place the intercept once the texel holding it is known.
// Eight takes a texel to a two-hundred-and-fiftieth of its width, far finer
// than the pixel that asked.
const REFINE_STEPS: u32 = 8u;

// Stand-in for an infinite distance, in a form arithmetic can still be done on.
const NEVER: f32 = 1e30;

// The lowest of four corners.
//
// Interpolating first would bury a hole: three real metres averaged with one
// -32767 comes out around -7800, which is far below any ground but nowhere near
// the nodata value, so a test on the result alone would let it through.
fn lowest(corner: vec4<f32>) -> f32 {
    return min(min(corner.x, corner.y), min(corner.z, corner.w));
}

// Whether a texel of a level is loaded and safe to read.
fn resident(level: u32, cell: vec2<i32>) -> bool {
    let info = terrain.levels[level];
    return all(cell >= info.valid_low) && all(cell < info.valid_high);
}

// The texture coordinate a texel index lands at, in this level.
//
// The identity for a resident level, whose mask is all ones. For a window under
// one it is a mask rather than a modulo, because such a window is a power-of-two
// square of tiles -- and it depends on nothing but the index, so a tile's
// address does not move when the window does. Negative indices wrap correctly:
// a two's complement AND is exactly the non-negative remainder.
//
// Then clamped to the level, which is what lets the last texel of a level be
// read at all. The bilinear patch there reaches one sample past itself, and
// past the last one there is nothing: `textureLoad` out of bounds answers zero,
// which draws as sea level rather than as absent. Clamping repeats the border
// instead, which is exactly what the tile store does past the edge of a survey
// -- so the two agree about ground that is off the end of the data.
//
// This is what the resident square used to hold back a texel for. It could not
// clamp, because past its edge sat a real texel of somewhere else entirely; a
// chain has no somewhere else. Holding a texel back here instead would make the
// coarsest levels empty -- the top of this chain is one texel across, and one
// less than that is none.
fn slot(level: u32, cell: vec2<i32>) -> vec2<i32> {
    let info = terrain.levels[level];
    if (level >= terrain.resident_base) {
        return clamp(cell, info.valid_low, info.valid_high - vec2<i32>(1));
    }
    // A window wraps and does not clamp. The cells being generated or derived
    // are deliberately outside what it currently advertises -- that is what a
    // step in flight means -- and clamping them would fold a whole tile onto
    // its own edge. Nothing reads past the edge either: a window advertises
    // itself one texel short, so the neighbour a bilinear patch reaches for is
    // still a texel of the same square.
    return cell & info.mask;
}

// Which mip of the resident chain a level is.
fn mip(level: u32) -> i32 {
    return i32(level - terrain.resident_base);
}

// A height off the resident chain, which is every level from the base up.
//
// Kept apart from `height_at` because `cs_detail` may only touch this one: it
// writes a layer of the generated heights, and a shader that so much as names
// that texture elsewhere has it bound both ways in the same dispatch, which
// wgpu refuses. The branch below would never be taken in that pass, and naga
// counts the global as used all the same.
fn resident_height_at(level: u32, cell: vec2<i32>) -> f32 {
    return textureLoad(heights, slot(level, cell), mip(level)).r;
}

fn height_at(level: u32, cell: vec2<i32>) -> f32 {
    if (level >= terrain.resident_base) {
        return resident_height_at(level, cell);
    }
    return textureLoad(detail_heights, slot(level, cell), i32(level), 0).r;
}

// The ceiling one cell of one level carries, from whichever pyramid holds it.
fn ceiling_at(level: u32, cell: vec2<i32>) -> f32 {
    if (level >= terrain.resident_base) {
        return textureLoad(maxima, slot(level, cell), mip(level)).r;
    }
    return textureLoad(detail_maxima, slot(level, cell), i32(level), 0).r;
}

// The four corner heights of one texel, in the order [`surface`] expects: the
// texel itself, then east, south, and south-east of it.
fn corners(level: u32, cell: vec2<i32>) -> vec4<f32> {
    return vec4<f32>(
        height_at(level, cell),
        height_at(level, cell + vec2<i32>(1, 0)),
        height_at(level, cell + vec2<i32>(0, 1)),
        height_at(level, cell + vec2<i32>(1, 1)),
    );
}

// The bilinear patch through four corners, at a fractional position inside the
// texel they belong to.
//
// Separated from the fetch because the march evaluates one texel's patch many
// times over -- at both ends of a segment and at every halving between them --
// and the four heights do not change while the ray is inside the texel. It is
// the same arithmetic a combined fetch-and-interpolate did, with the fetches
// hoisted out.
//
// Clamped rather than taken as read. The far end of a segment sits exactly on
// the texel wall, where rounding can put the fraction a hair outside; a bilinear
// is continuous across that edge, so clamping there gives the value the
// neighbouring texel's patch would -- without reading a texel this level may not
// have loaded.
fn surface(corner: vec4<f32>, at: vec2<f32>) -> f32 {
    let f = clamp(at, vec2<f32>(0.0), vec2<f32>(1.0));
    return mix(mix(corner.x, corner.y, f.x), mix(corner.z, corner.w, f.x), f.y);
}

// This file knows nothing about trees, and that is recent. The march used to
// grow the crowns itself, carrying a hand transcription of the whole of
// `crates/terrain-canopy` -- the crown lattice, the hashes, the distance field
// and a sphere trace per leaf texel -- and painting a pixel that met one with a
// canopy material id of its own. All of it is gone. `terrain-generate` writes
// the trees into the elevation and the ground cover instead, so a ray meets a
// tree by meeting the ground and a pixel is forest because the material under
// it says so.
//
// That was worth about three quarters of the march. The cost was never the walk
// -- shortening it barely helped -- it was *entering* the wooded path at all, a
// cover lookup and nine hashes at every leaf texel a ray crossed, which at level
// 0 is once a metre through a stand. It also took a whole class of failure with
// it: two spellings of one function, pinned by a GPU test, where a disagreement
// would put crowns above the ceiling meant to bound them and let rays through
// the forest with nothing reporting it.

// Normalized device coordinates of a pixel's centre, which is what the camera's
// ray basis wants.
//
// Derived from the pixel index rather than handed down from a vertex stage, as
// it was while this was a fullscreen triangle. Interpolating the corners and
// computing this agree to within rounding, but only to within rounding, and a
// last-bit difference in a ray direction can send the march down the far side
// of a cell wall thousands of texels away. Deriving it means anything that
// wants the ray through a pixel gets the identical ray, whether it is covering
// the screen or working from a list of pixel indices.
//
// `y` flips: the framebuffer counts rows downwards from the top and clip space
// counts upwards from the middle.
fn ndc_of(pixel: vec2<f32>) -> vec2<f32> {
    let size = vec2<f32>(terrain.viewport);
    return vec2<f32>(
        pixel.x / size.x * 2.0 - 1.0,
        1.0 - pixel.y / size.y * 2.0,
    );
}

// The ray through the centre of a pixel, before it is normalised.
//
// The centre is where the ray goes through it, and is what the rasterizer used
// to hand the march as the fragment coordinate. One function so that nothing
// asking about a pixel's ray can end up with a different one.
//
// Unnormalised, its component along the view axis is exactly one, because the
// basis is the near plane at unit distance. That is what makes [`distance_at`]
// a multiply rather than a divide by a length.
fn ray_raw_at(screen: vec2<f32>) -> vec3<f32> {
    let ndc = ndc_of(screen);
    return camera.ray_right.xyz * ndc.x + camera.ray_up.xyz * ndc.y + camera.ray_forward.xyz;
}

fn ray_raw(pixel: vec2<u32>) -> vec3<f32> {
    return ray_raw_at(vec2<f32>(pixel) + 0.5);
}

fn ray_through(pixel: vec2<u32>) -> vec3<f32> {
    return normalize(ray_raw(pixel));
}

// How far along [`ray_raw`] a reversed-Z depth puts a point.
//
// The projection writes `z_near / d`, where `d` is the distance along the view
// axis, so this inverts it. Multiplying the raw ray by the result lands on the
// point exactly, with no normalise and no length: the raw ray advances one unit
// along the view axis per unit of `s`.
fn distance_at(depth: f32) -> f32 {
    return camera.ray_forward.w / depth;
}

struct Hit {
    found: bool,
    // Whether the ray was abandoned rather than answered.
    //
    // A ray can fail to find ground for two quite different reasons. It can
    // establish that there is none -- it climbed over everything, or it left
    // the raster -- which is a fact about the world and stays true next frame.
    // Or it can give up: the eye was under the surface, or the ground it wanted
    // was not resident. That is a fact about this frame's residency and this
    // ray's budget, and it must not be carried into the next frame as though it
    // were sky. See `unknown`.
    abandoned: bool,
    // Where the ray met the ground, in world space.
    position: vec3<f32>,
    // And which level it was found at, and where in that level's texels, so the
    // material is looked up from the same place the height came from.
    level: u32,
    w: vec2<f32>,
};

// Walks `dir` from the eye until it meets the ground.
//
// The march works throughout in *level-0 texel* coordinates, so a level is
// nothing but a cell size: level `l`'s texels are `2^l` of them across. Nothing
// has to be rebased when a ray changes level, which a chain gives for free --
// mip `m` of a texture is exactly half mip `m - 1`.
fn march(eye: vec3<f32>, dir: vec3<f32>) -> Hit {
    var out: Hit;
    out.found = false;
    out.abandoned = false;
    ray_abandoned = false;
    ray_spent = false;

    let coarsest = terrain.level_count - 1u;
    let p0 = (eye.xz - terrain.origin) / terrain.metres_per_texel;
    let d0 = dir.xz / terrain.metres_per_texel;
    let speed = max(max(abs(d0.x), abs(d0.y)), 1.0 / NEVER);
    // Past a wall far enough to be past it, so the next step lands in the next
    // texel rather than back in this one. `wall_nudge` is a distance along the
    // dominant axis; dividing by `speed` turns it into a distance along the ray.
    let nudge = terrain.wall_nudge / speed;
    // A ray that has crossed the raster twice has left it, or is going straight
    // up or down and never will. Taken from the data bounds rather than from
    // the texture, which since the whole raster is resident is the same figure
    // and the more honest of the two: what bounds a ray is the ground, not the
    // memory holding it.
    let extent = (terrain.data_max - terrain.data_min) / terrain.metres_per_texel;
    let limit = 2.0 * max(extent.x, extent.y) / speed;

    var t = 0.0;
    // Start at the coarsest level and descend, which is the shape a maximum
    // mipmap traversal is usually written in. Starting at the finest and
    // climbing measured the same on this raster, so this is the conventional
    // form rather than a measured improvement over it.
    var level = coarsest;
    // Whether the ray has advanced at all. Being under the ground before it has
    // means something quite different from being under it after.
    var moved = false;
    // The finest level still worth trying where the ray is standing *now*.
    //
    // Climbing without advancing `t` -- because nothing is loaded here, or
    // because the ray is over the whole level -- rules out everything below as
    // well, a finer level's square being contained in a coarser one's. Without
    // that recorded, the descent below would take the ray straight back to the
    // level it just left and the two would trade places at one `t` until the
    // step budget ran out.
    var finest = terrain.base_level;
    // Whether the last thing the ray crossed was ground nothing is known about.
    // A ray that drops through a hole comes up against the underside of the
    // ground beside it, and has to be told apart from one that merely grazed a
    // surface it was above the whole way.
    var hole = false;

    for (var step = 0u; step < terrain.march_steps; step += 1u) {
        if (t >= limit) {
            return out;
        }
        let p = p0 + d0 * t;
        let size = f32(1u << level);
        let cell = vec2<i32>(floor(p / size));

        // Off the end of this level: the ray has left the raster at this
        // resolution, so hand it out to the level beyond, which covers twice
        // the ground per texel and so reaches one texel further.
        if (!resident(level, cell)) {
            if (level >= coarsest) {
                // Past the coarsest level is past the raster, and there is
                // nothing out there to draw. This used to be reported as a ray
                // that gave up, because it was genuinely ambiguous: a square
                // still filling looked exactly like the edge of the world, and
                // calling either one sky would have drawn a hole while a level
                // loaded. Nothing fills now -- the whole raster is resident
                // before the first frame -- so a ray that leaves the chain has
                // left the world, and sky is the honest answer rather than a
                // diagnostic. It also lets the reprojection carry the pixel,
                // where an abandoned one was re-marched every frame forever.
                return out;
            }
            level += 1u;
            finest = level;
            continue;
        }

        // A whole level cleared in one test, before any texel of it is looked
        // at. Worth asking on arrival: at any horizon view most of the frame is
        // sky, and every one of those rays would otherwise walk the length of
        // each square it passes through.
        if (dir.y >= 0.0 && eye.y + dir.y * t > terrain.levels[level].ceiling) {
            if (level >= coarsest) {
                return out;
            }
            level += 1u;
            finest = level;
            continue;
        }

        // Where the ray leaves the texel it is standing in, at this level.
        let low = vec2<f32>(cell) * size;
        let wall = low + select(vec2<f32>(0.0), vec2<f32>(size), d0 > vec2<f32>(0.0));
        let crossing = (wall - p) / d0;
        let span = min(
            select(NEVER, crossing.x, abs(d0.x) > 0.0),
            select(NEVER, crossing.y, abs(d0.y) > 0.0),
        );
        let exit = min(t + max(span, 0.0), limit);

        let ceiling = ceiling_at(level, cell);
        // The texel bounds its whole closed square, so a ray above that ceiling
        // at both ends of the segment is above it throughout.
        if (min(eye.y + dir.y * t, eye.y + dir.y * exit) > ceiling) {
            t = exit + nudge;
            moved = true;
            level = min(level + 1u, coarsest);
            finest = terrain.base_level;
            continue;
        }

        // Something could be here. Look closer, if anything finer is loaded --
        // which is to say, if this ground is near enough the camera to have it.
        if (level > finest) {
            let finer = level - 1u;
            if (resident(finer, vec2<i32>(floor(p / (size * 0.5))))) {
                level = finer;
                continue;
            }
        }

        // The finest level there is here, so this texel is the leaf and the
        // ground across it is the bilinear patch through its four corners.
        //
        // Read once. The ray is inside this texel for the whole segment, so
        // every height wanted below -- both ends and every halving between them
        // -- comes out of the same four numbers, and fetching them per
        // evaluation was forty texture reads a hit instead of four.
        //
        // It is also the only way to keep the far end inside what residency
        // promised. That end sits on the texel wall, so interpolating it from
        // where it lands reads the *next* texel's corners, one past the square
        // -- and while the height is unaffected, because the out-of-range
        // corners carry no weight there, `lowest` is a minimum over all four at
        // full weight. A tile of somewhere else sharing that slot could hold
        // nodata, and the ray would skip ground that is really there.
        let w = p / size;
        let corner = corners(level, cell);
        let deepest = lowest(corner);
        let base = vec2<f32>(cell);
        let enter = surface(corner, w - base);
        let leave = surface(corner, (p0 + d0 * exit) / size - base);

        // Ground nothing is known about is not ground. The sentinel is far below
        // anything a canopy could add to it, so ground with trees on it is still
        // ground and a hole is still a hole.
        if (deepest < NODATA_BELOW) {
            hole = true;
            t = exit + nudge;
            moved = true;
            level = min(level + 1u, coarsest);
            finest = terrain.base_level;
            continue;
        }

        if (eye.y + dir.y * t <= enter) {
            if (!moved || hole) {
                // Either the ray began below the surface -- where it went in is
                // behind the eye and cannot be found from here -- or it has just
                // dropped through a hole and this is the underside of the ground
                // beside it. Neither is a hit.
                //
                // The first of those is the eye being inside the terrain, which
                // is a fact about where the camera is and stops being true the
                // moment it climbs out; the second is a hole in the survey,
                // which is a fact about the data. Only the first is abandoned.
                out.abandoned = !moved;
                ray_abandoned = !moved;
                return out;
            }
            // Otherwise the ray has grazed the surface within a hair of a texel
            // boundary -- every earlier texel it crossed it was proven to clear,
            // so the crossing is here, not somewhere it was skipped past.
            out.found = true;
            out.position = eye + dir * t;
            out.level = level;
            out.w = w;
            return out;
        }

        // Standing above measured ground, so whatever the ray crossed earlier is
        // behind it now and cannot be what a later descent is coming up under.
        hole = false;

        if (eye.y + dir.y * exit > leave) {
            // The ceiling allowed a hit somewhere in the texel; the surface
            // itself does not reach the ray.
            t = exit + nudge;
            moved = true;
            level = min(level + 1u, coarsest);
            finest = terrain.base_level;
            continue;
        }

        // Above at one end and below at the other, so the crossing is
        // bracketed -- but not necessarily as tightly as `exit` suggests. A ray
        // too close to vertical to cross a wall inside the whole march budget
        // has no wall to stop at, so `exit` came back as the budget itself and
        // halving that eight times resolves the crossing to a fifth of it. The
        // hit would land kilometres under the ground, with a depth that
        // underflows and a texel index read from somewhere else entirely --
        // invisible in the frame, because the material and normal are taken at
        // the texel the ray is standing in and come out right, and wrong in the
        // G-buffer the reprojection and the motion field are built from.
        //
        // So bound the far end by where the ray falls past the lowest corner of
        // the patch. Below that it is below every height the patch can take, so
        // the crossing is behind it, and for a ray that is climbing or level
        // there is nothing to bound: it cannot fall to meet anything.
        let fall = (eye.y + dir.y * t - deepest) / max(-dir.y, 1.0 / NEVER);
        var above = t;
        var below = min(exit, t + max(fall, 0.0));
        for (var i = 0u; i < REFINE_STEPS; i += 1u) {
            let middle = 0.5 * (above + below);
            let ground = surface(corner, (p0 + d0 * middle) / size - base);
            if (eye.y + dir.y * middle > ground) {
                above = middle;
            } else {
                below = middle;
            }
        }

        out.found = true;
        out.position = eye + dir * below;
        out.level = level;
        out.w = (p0 + d0 * below) / size;
        return out;
    }

    // Out of steps. A ray gets here by running along a slope just above the
    // surface -- never far enough from it to skip a texel, never near enough to
    // meet one -- so where it had got to is close to the ground and its colour
    // is close to the ground's. Reporting that beats reporting sky by a wide
    // margin: the ground is genuinely there, and the alternative punches a hole
    // through a ridge that the pixels either side of it drew perfectly well.
    //
    // Only for a ray that has moved and is not standing over a hole, which are
    // the same two conditions the leaf applies for the same reasons.
    //
    // However it ends, getting here at all means the budget ran out rather than
    // the ray settling anything, which is what `ray_spent` records.
    ray_spent = true;
    if (moved && !hole) {
        out.found = true;
        out.position = eye + dir * t;
        out.level = level;
        out.w = (p0 + d0 * t) / f32(1u << level);
    } else {
        // Never advanced at all, so the eye is inside the terrain. Same case as
        // the leaf above, reached the long way round.
        out.abandoned = !moved;
        ray_abandoned = !moved;
    }
    return out;
}

// The G-buffer the shading pass reads, written a texel at a time.
//
// Storage rather than colour attachments because the march is a compute pass
// and a compute pass has no attachments. Nothing clears these, so every pixel
// has to be written by the dispatch below -- a pixel that found no ground
// writes zeroes explicitly, and depth zero is how the shading pass knows a
// pixel is sky.
//
// No world position among them. A depth and the sub-pixel offset riding in the
// spare half of the material word say the same thing in four bytes rather than
// sixteen; see `MATERIAL_MASK` for the encoding and `rebuilt` in
// `src/reproject.wgsl` for the one reader that wants a position back.

// A material id and where inside its pixel the ground sits, in one word.
@group(2) @binding(0) var out_material: texture_storage_2d<r32uint, write>;
// The unit normal of that ground, in world space, and in `w` which of the two
// reasons a pixel with no depth has none. See `SKY_HERE`.
@group(2) @binding(2) var out_normal: texture_storage_2d<rgba16float, write>;
// The distance the ground was actually found at, as reversed-Z depth.
@group(2) @binding(3) var out_depth: texture_storage_2d<r32float, write>;
// How far this pixel's ground has moved across the screen since last frame,
// as two half floats in one integer channel. See `MOTION_FORMAT`.
@group(2) @binding(4) var out_motion: texture_storage_2d<r32uint, write>;

// What the reprojection carried over from the last frame, already placed where
// this camera sees it. See `src/reproject.wgsl` for how it got here.
//
// Depth zero is the reversed-Z far plane, which no carried point can write, so
// it is exactly "nothing landed on this pixel" -- the same test, for the same
// reason, that the shading pass applies to the G-buffer.
@group(3) @binding(0) var carried_material: texture_2d<u32>;
// No carried position: it is sixteen bytes a point for something the depth
// below already determines. See `carried_at`.
@group(3) @binding(2) var carried_normal: texture_2d<f32>;
@group(3) @binding(3) var carried_depth: texture_depth_2d;

// The pixels the reprojection did not reach, packed as `x | y << 16`.
//
// This is the point of the whole arrangement. A wave costs as much as the
// longest ray in it, so a march that runs a thread per pixel and lets most of
// them return early does not get faster for having less to do -- the lanes go
// idle but the wave still waits. Compacting the misses into a list lets the
// dispatch be sized to the work rather than to the screen, so every wave that
// runs is full of rays that actually had to be cast.
//
// What it costs is locality: consecutive entries are wherever on screen the
// reprojection failed, so the rays in a wave no longer walk neighbouring cells
// of the quadtree the way a rectangle of pixels did.
@group(3) @binding(4) var<storage, read_write> holes: array<u32>;

// How many pixels `cs_compact` sent down each of its three paths.
//
// `holes` is load bearing -- `cs_args` sizes the march from it and `cs_march`
// bounds itself by it -- and the other two exist only to be read back and
// reported. Together they say what the reprojection is actually buying, which
// the hole count alone cannot: a pixel that was not marched was either carried
// over from the last frame or settled as sky for free, and those are not the
// same achievement.
//
// Every pixel of the viewport takes exactly one of the three, so they sum to
// the pixel count. That is worth asserting, and it is what makes the two
// diagnostic counters worth the atomics rather than deriving one from the
// others.
//
// The last two are not paths. They are subsets of the marched pixels, counted
// so the overlay can say *why* a march came back empty-handed: `abandoned` is
// rays that gave up (see `Hit::abandoned`), `spent` is rays that ran out of
// step budget and were painted as ground where they stopped. Both read zero on
// a healthy frame, so the atomics cost nothing there, and either one running
// away is the signature of a march that is failing rather than working.
//
// `holes` stays first so its offset is unchanged; the members after it are
// mirrored by `Coverage` in `src/reproject.rs`.
struct Tally {
    holes: atomic<u32>,
    carried: atomic<u32>,
    sky: atomic<u32>,
    abandoned: atomic<u32>,
    spent: atomic<u32>,
    // Pixels `cs_march` actually stored, and the workgroup count it was
    // dispatched with.
    //
    // `holes` is what the compaction asked for; these two are what the march
    // was told to do and what it did. They exist because those can disagree
    // without anything else noticing: nothing clears the G-buffer, so a march
    // that does not run leaves every pixel holding whatever last reached it,
    // which on a slow-moving camera looks like a picture rather than like a
    // failure. `wrote` short of `holes` means the dispatch did not cover the
    // list; `groups` says how large that dispatch was, which is the number an
    // adapter limit would cap.
    wrote: atomic<u32>,
    groups: atomic<u32>,
};

// What the march did with this thread's ray, for the two counters above.
//
// Private rather than returned, because `ground_at` reduces a `Hit` to a
// `Ground` and neither of these belongs in what the G-buffer stores. One
// `ground_at` per invocation, so there is nothing to reset between rays.
var<private> ray_abandoned: bool = false;
var<private> ray_spent: bool = false;

@group(3) @binding(5) var<storage, read_write> tally: Tally;
// `[workgroups, 1, 1]`, written by `cs_args` for the march to be dispatched by.
@group(3) @binding(6) var<storage, read_write> march_args: array<u32, 3>;

// The finished motion field, the depth beside it, and the pair of numbers per
// dither cell that `cs_risk` reduces them to. Bound only for that pass, which
// writes neither of the two above and so cannot be given the same layout as
// the march.
@group(3) @binding(7) var motion: texture_2d<u32>;
@group(3) @binding(11) var settled_depth: texture_2d<f32>;
@group(3) @binding(8) var out_risk: texture_storage_2d<rg32float, write>;

// That same cell summary read back, and the reach `cs_reach` spreads it into.
// Its own layout again, and for the same reason: one pass cannot bind the risk
// texture as writable storage and read it as a texture at the same time.
@group(3) @binding(9) var risk: texture_2d<f32>;
@group(3) @binding(10) var out_reach: texture_storage_2d<r32float, write>;

// One height, paired with whether a difference may use it.
//
// Three ways it may not, and two of them are the same mistake at different
// edges. It can be the raster's nodata, which is a hole in the survey rather
// than a measurement of flat ground. It can lie outside this level, where
// `slot` clamps and answers with the border texel repeated. Or it can lie past
// `last`, outside the survey altogether, where the store did the same thing
// when the level was read. Both of those are ground of no slope, which is the
// one wrong answer that looks like a right one.
//
// The march itself never had to ask: at the leaf it reads the four corners of
// the texel it is standing in, and the far one carries no weight at the edge,
// whereas a central difference reaches two texels past the hit.
fn sample_height(level: u32, cell: vec2<i32>, last: vec2<i32>) -> vec2<f32> {
    if (any(cell < vec2<i32>(0)) || any(cell > last) || !resident(level, cell)) {
        return vec2<f32>(0.0, 0.0);
    }
    let height = height_at(level, cell);
    return select(vec2<f32>(0.0, 0.0), vec2<f32>(height, 1.0), height >= NODATA_BELOW);
}

// The slope across one axis at a texel, from the samples either side of it.
//
// Central where both neighbours are ground, one-sided where only one is, and
// flat where the texel stands alone -- so the last row of a survey slopes the
// way its two real samples do rather than the way a repeated edge would.
fn axis_slope(here: f32, low: vec2<f32>, high: vec2<f32>, metres: f32) -> f32 {
    if (low.y > 0.0 && high.y > 0.0) {
        return (high.x - low.x) / (2.0 * metres);
    }
    if (low.y > 0.0) {
        return (here - low.x) / metres;
    }
    if (high.y > 0.0) {
        return (high.x - here) / metres;
    }
    return 0.0;
}

// The ground's unit normal at a fractional position, in world space.
//
// A normal is a derivative, and this takes it from the same heights the march
// traced, at the level the ray stopped at, so shading and silhouette describe
// one surface rather than two. What that costs is the far field: a coarse texel
// is already a smoothed surface and its slopes are the slopes of that
// smoothing, so relief flattens with distance in a way a normal averaged down
// from level 0 would not. Nothing finer is available to flatten less -- past
// the finest level's reach the fine heights are not resident and cannot be.
//
// Bilinear over the four surrounding texels rather than the nearest of them.
// Nearest is flat shading: one constant direction a texel, so the ground breaks
// into facets that the eye reads as blocks however smooth the surface under
// them is. The gradient of the ray's own bilinear patch would be worse again --
// continuous inside a texel and discontinuous at every wall between them.
// Blending four central differences is what keeps the normal continuous.
//
// Nodata is not a direction and must not be averaged into one: those corners
// are dropped and the weight redistributed over the rest, so ground beside a
// hole takes the normal of the ground that was measured. Flat is the answer
// where nothing was. Dropping a corner cannot break the continuity above,
// because a corner that is about to be left behind already carries no weight.
fn normal_at(level: u32, w: vec2<f32>) -> vec3<f32> {
    let base = vec2<i32>(floor(w));
    let f = fract(w);
    // A texel of this level, on the ground, in each of the raster's two
    // directions. The two need not be the same number, so neither difference
    // may borrow the other's.
    let metres = terrain.metres_per_texel * f32(1u << level);
    // The outermost sample the survey holds, in this level's texels. `origin`
    // is the position of texel (0, 0) and `data_max` that of the last one, so
    // the near bound is zero at every level and only the far one is worked out.
    // A coarse texel sits on the level-0 lattice, so its last index is the
    // level-0 one shifted down.
    let last = vec2<i32>(round((terrain.data_max - terrain.origin) / terrain.metres_per_texel))
        >> vec2<u32>(level);

    // Every height the four corner normals are built from: the four-by-four
    // block around them, less its own corners, which no central difference
    // reaches. Twelve reads, against the four a baked normal took -- the whole
    // price of deriving these here, paid once per marched pixel rather than
    // once per step.
    var nearby: array<vec2<f32>, 16>;
    for (var j = -1; j <= 2; j += 1) {
        for (var i = -1; i <= 2; i += 1) {
            let outside = (i < 0 || i > 1) && (j < 0 || j > 1);
            if (!outside) {
                nearby[(j + 1) * 4 + i + 1] = sample_height(level, base + vec2<i32>(i, j), last);
            }
        }
    }

    var sum = vec3<f32>(0.0);
    var total = 0.0;
    for (var j = 0; j <= 1; j += 1) {
        for (var i = 0; i <= 1; i += 1) {
            let at = (j + 1) * 4 + i + 1;
            let here = nearby[at];
            if (here.y <= 0.0) {
                continue;
            }
            let weight = select(1.0 - f.x, f.x, i == 1) * select(1.0 - f.y, f.y, j == 1);
            let east = axis_slope(here.x, nearby[at - 1], nearby[at + 1], metres.x);
            let south = axis_slope(here.x, nearby[at - 4], nearby[at + 4], metres.y);
            sum += weight * normalize(vec3<f32>(-east, 1.0, -south));
            total += weight;
        }
    }
    if (total <= 0.0) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    // A mean of directions, which is not itself one. Renormalising is what
    // makes it a direction again, and unlike the packed pair this replaces
    // there is a third component here to renormalise with.
    return normalize(sum);
}

// What the normal's fourth channel says about a pixel whose depth is zero.
//
// Three states rather than two, because the reprojection has to tell a pixel
// whose ray found nothing from a pixel nothing has ever been written to. Sky is
// worth carrying between frames -- it is most of a horizon view and none of it
// can be carried the way ground is, having no position to carry -- but a buffer
// that has only just been allocated must carry nothing at all. Zero is what
// wgpu leaves a fresh texture as, so zero is the one that means "not yet", and
// this marks the pixels that really did establish there is nothing there.
//
// It rides in the normal because a normal is what a pixel with no ground has
// none of: ground writes a unit vector and leaves `w` alone, and a zero depth
// is what says to look at `w` at all.
const SKY_HERE: f32 = 1.0;

// The material word is a material id and the sub-pixel position the point sits
// at inside its pixel, packed together.
//
// Ids reach 0x080c and so need sixteen bits of the thirty-two; the rest were
// spare, and the offset rides in them for nothing -- no extra target, no extra
// export bandwidth. Eight bits an axis puts the point within a 256th of a pixel
// of where it really is. Must match `pack_offset` in `src/reproject.wgsl`.
//
// The offset is what lets the world position go unstored: a pixel's depth says
// how far along a ray its ground is, and this says which ray. A marched pixel
// sits at the centre, because the ray was cast through the centre; a carried one
// sits wherever its point landed. Everything that wants the position rebuilds it
// from the two, which is what `carried_at` already did for the carried half.
const MATERIAL_MASK: u32 = 0xffffu;
const OFFSET_SCALE: f32 = 256.0;

fn unpack_offset(packed: u32) -> vec2<f32> {
    return vec2<f32>(
        f32((packed >> 16u) & 0xffu),
        f32((packed >> 24u) & 0xffu),
    ) / OFFSET_SCALE;
}

fn pack_offset(id: u32, offset: vec2<f32>) -> u32 {
    let axis = vec2<u32>(clamp(offset, vec2<f32>(0.0), vec2<f32>(0.99609375)) * OFFSET_SCALE);
    return (id & MATERIAL_MASK) | (axis.x << 16u) | (axis.y << 24u);
}

// What one ray found, in the form the G-buffer stores it.
//
// Material zero is `Null` and depth zero is the reversed-Z far plane, which no
// finite hit can produce, so those two are what the shading pass reads as sky.
struct Ground {
    material: u32,
    // Where the point sits inside its pixel, in pixels. Stored in the spare
    // half of the material word rather than beside the position, because the
    // position is not stored at all -- see `MATERIAL_MASK`.
    offset: vec2<f32>,
    // Where the ground is, which nothing writes: it is here so the motion field
    // can be worked out from it before it is thrown away.
    position: vec3<f32>,
    normal: vec4<f32>,
    depth: f32,
};

fn nothing() -> Ground {
    return Ground(0u, vec2<f32>(0.5), vec3<f32>(0.0), vec4<f32>(0.0, 0.0, 0.0, SKY_HERE), 0.0);
}

// A pixel whose ray was abandoned rather than answered. See `Hit::abandoned`.
//
// Draws as sky, because zero depth is what the shading pass reads as sky and
// there is nothing better to show. What makes it different from `nothing` is
// leaving the normal at all zeroes, which is what a G-buffer nobody has written
// reads as: the splat drops the point, so the pixel is marched again next frame
// instead of being answered from a frame that did not know either.
//
// Without this a camera that dips below the surface fills the G-buffer with
// sky, and because carried sky is placed far enough away to ignore where the
// eye has moved to, that sky lands back on the same pixel frame after frame.
// Climbing out does not clear it: there is no ground in the history to splat
// over it, so the frame stays holed until the dither happens to drop each
// cell, which takes up to sixteen frames.
fn unknown() -> Ground {
    return Ground(0u, vec2<f32>(0.5), vec3<f32>(0.0), vec4<f32>(0.0), 0.0);
}

// Where a world point sat on the previous frame's screen, in pixels.
//
// Undefined behind the old camera, where the perspective divide flips; those
// come back as a zero motion rather than a wild one, which reads as "nothing
// worth spending rays on" and is the safe way to be wrong.
fn was_at(position: vec3<f32>) -> vec2<f32> {
    let clip = camera.was_view_proj * vec4<f32>(position, 1.0);
    if (clip.w <= 0.0) {
        return vec2<f32>(0.0);
    }
    let ndc = clip.xy / clip.w;
    let size = vec2<f32>(terrain.viewport);
    return vec2<f32>((ndc.x * 0.5 + 0.5) * size.x, (0.5 - ndc.y * 0.5) * size.y);
}

// How far a pixel's ground has moved across the screen since the last frame.
//
// Zero for sky, which has no world position to have moved. That is not the same
// as sky being still -- it turns with the camera like everything else -- but
// sky is not what the drop pattern is trying to find.
fn motion_of(pixel: vec2<u32>, ground: Ground) -> vec2<f32> {
    if (ground.depth == 0.0) {
        return vec2<f32>(0.0);
    }
    let now = vec2<f32>(pixel) + 0.5;
    let was = was_at(ground.position);
    if (all(was == vec2<f32>(0.0))) {
        return vec2<f32>(0.0);
    }
    return now - was;
}

// The ground down the ray through one pixel, or `nothing()` for sky.
//
// Split out from the dispatch so that the answer for a pixel depends on the
// pixel alone, however the caller came to be asking about it.
fn ground_at(pixel: vec2<u32>) -> Ground {
    let eye = camera.position.xyz;
    let dir = ray_through(pixel);

    // A ray already above the highest ground anywhere resident, and still
    // climbing, is never coming back down. Worth one comparison: at any horizon
    // view most of the frame is sky.
    if (dir.y >= 0.0 && eye.y >= terrain.ceiling) {
        ray_abandoned = false;
        ray_spent = false;
        return nothing();
    }

    let hit = march(eye, dir);
    if (!hit.found) {
        if (hit.abandoned) {
            return unknown();
        }
        return nothing();
    }

    // Squares reach past the raster and reads out there wrap onto whatever
    // shares their slot, so the ground beyond the last real sample is not
    // ground. A straight ray leaves the data once and never comes back, so
    // there is nothing further along worth carrying on for.
    if (any(hit.position.xz < terrain.data_min) || any(hit.position.xz > terrain.data_max)) {
        return nothing();
    }

    var out: Ground;
    let clip = camera.view_proj * vec4<f32>(hit.position, 1.0);
    out.depth = clip.z / clip.w;
    // The nearest texel to the hit: sample centres sit at integer `w`, the
    // convention the height bilinear reads by, so the texel whose centre is
    // closest is the rounded index.
    let cell = vec2<i32>(floor(hit.w + 0.5));
    // No question about what the ray met -- which there could not be, because a
    // crown baked into the heights is indistinguishable from a hillock and the
    // march has nothing left to ask. A treetop is labelled as one already: the
    // pass that raised it wrote a canopy id at the same time, so the gaps
    // between the trees keep the floor's own colour and the trees do not.
    //
    // Two products, because there are two chains. The survey's ground cover is
    // held only on the resident one -- nothing finer than the base has a
    // material anyone measured -- and a generated level carries its own ids,
    // written by the pass that generated its heights: the survey's cover
    // upscaled onto its texels, with a crown or a stone painted over it wherever
    // that pass grew enough to hide the ground.
    //
    // A whole id either way, so there is nothing to combine here. It used to
    // read the base's cover and let a non-zero detail id override it, which was
    // the same answer while a generated level had nothing to say about the
    // ground itself; now it does, and the level that was descended to is the
    // level that answers.
    if (hit.level < terrain.resident_base) {
        out.material = textureLoad(detail_materials, slot(hit.level, cell), i32(hit.level), 0).r;
    } else {
        out.material = textureLoad(materials, slot(hit.level, cell), mip(hit.level)).r;
    }
    out.position = hit.position;
    // The ray was cast through the centre of the pixel, so that is where the
    // hit sits inside it, exactly.
    out.offset = vec2<f32>(0.5);
    // Differenced out of the same heights the ray was traced through, at the
    // level it stopped at, so the surface that is shaded is the surface that
    // was drawn. Near ground gets the finest level the clipmap is holding;
    // far ground gets whatever coarse level answered, and flattens with it.
    //
    // A crown's own slope comes out of this too, with nothing added: the trees
    // are in the heights being differenced, so the flank of a treetop is a slope
    // like any other. It used to need a second gradient of the crown field on
    // top, because the surface drawn was the ground plus a canopy the heights
    // knew nothing about.
    out.normal = vec4<f32>(normal_at(hit.level, hit.w), 0.0);
    return out;
}

// What a pixel the reprojection reached should be written as.
//
// The splat carries a material and a normal but no position, because a position
// is sixteen bytes a point of export bandwidth and the depth it also carries
// already determines one: the point is wherever this pixel's ray reaches at
// that depth. Rebuilding it here costs a multiply-add and saves the splat more
// than half of what it writes.
//
// Not quite the point the march originally found. That point sat on the ray
// through whichever pixel it was found in, and this puts it on the ray through
// the pixel it has landed in, so it slides by up to half a pixel across the
// view. The error is re-taken from the current camera every frame rather than
// accumulated, and it is the same sub-pixel resampling the material and the
// normal already suffer.
//
// Sky is told from ground by the normal: the march writes a unit vector for
// ground and zeroes for sky, and no ground normal can be short.
fn carried_at(pixel: vec2<u32>, depth: f32) -> Ground {
    let normal = textureLoad(carried_normal, vec2<i32>(pixel), 0);
    if (dot(normal.xyz, normal.xyz) < 0.5) {
        return nothing();
    }
    var out: Ground;
    let packed = textureLoad(carried_material, vec2<i32>(pixel), 0).r;
    out.material = packed & MATERIAL_MASK;
    // Rebuilt where the point actually landed inside the pixel, not at the
    // pixel's centre. The centre is up to half a pixel from the truth, and half
    // a pixel at ten kilometres is metres of ground -- which would not matter if
    // it were the same half pixel every frame, but the point lands somewhere
    // different each time the camera moves, so it would be re-snapped in a
    // different direction every frame and the ground would visibly crawl. This
    // is what keeps a carried point standing still.
    out.offset = unpack_offset(packed);
    let landed = vec2<f32>(pixel) + out.offset;
    out.position = camera.position.xyz + ray_raw_at(landed) * distance_at(depth);
    out.normal = normal;
    out.depth = depth;
    return out;
}

// Writes one pixel's worth of ground into the G-buffer.
fn store(pixel: vec2<u32>, ground: Ground) {
    let at = vec2<i32>(pixel);
    textureStore(
        out_material,
        at,
        vec4<u32>(pack_offset(ground.material, ground.offset), 0u, 0u, 0u),
    );
    textureStore(out_normal, at, ground.normal);
    textureStore(out_depth, at, vec4<f32>(ground.depth, 0.0, 0.0, 0.0));
    let motion = motion_of(pixel, ground);
    textureStore(out_motion, at, vec4<u32>(pack2x16float(motion), 0u, 0u, 0u));
}

// One thread per pixel: settle what can be settled, and list the rest.
//
// Eight by eight rather than a flat sixty-four because this one *is* indexed by
// screen position, and neighbouring pixels take the same branch far more often
// than distant ones do.
//
// Three outcomes, and every pixel has exactly one of them, which is what lets
// the G-buffer go uncleared: it is either written from what the reprojection
// carried, written as sky, or handed to the march, which writes it.
@compute @workgroup_size(8, 8)
fn cs_compact(@builtin(global_invocation_id) id: vec3<u32>) {
    // The dispatch is rounded up to whole workgroups, so the last row and
    // column of them run past the target.
    if (any(id.xy >= terrain.viewport)) {
        return;
    }
    let pixel = id.xy;
    let at = vec2<i32>(pixel);

    // Already answered by a ray cast on an earlier frame.
    let depth = textureLoad(carried_depth, at, 0);
    if (depth != 0.0) {
        store(pixel, carried_at(pixel, depth));
        atomicAdd(&tally.carried, 1u);
        return;
    }

    // Sky the ceiling test can settle without walking anything. Only fires with
    // the camera above every resident peak; below them a ray heading for the
    // horizon has to be walked to the end of its budget before it can be called
    // sky, which is why carrying sky across frames is worth as much as it is.
    //
    // The height comparison first, and not for tidiness: `&&` short-circuits,
    // and the comparison is against a uniform every pixel answers the same way.
    // Below the peaks -- which is most cameras, and every one that reports no
    // sky at all -- it rejects the whole test before the normalize behind
    // `ray_through` is ever run.
    let eye = camera.position.xyz;
    if (eye.y >= terrain.ceiling && ray_through(pixel).y >= 0.0) {
        store(pixel, nothing());
        atomicAdd(&tally.sky, 1u);
        return;
    }

    holes[atomicAdd(&tally.holes, 1u)] = pixel.x | (pixel.y << 16u);
}

// Turns the hole count into a dispatch size for the march.
//
// A whole pass for three integers, because the count is not known until every
// workgroup of `cs_compact` has finished and the CPU never sees it at all.
@compute @workgroup_size(1)
fn cs_args() {
    let groups = (atomicLoad(&tally.holes) + MARCH_GROUP - 1u) / MARCH_GROUP;
    // One row while the work fits in one, so a short list still dispatches
    // exactly what it needs; full rows after that, which is what lets
    // `cs_march` recover the index from the two dimensions without being told
    // how wide the grid is.
    march_args[0] = min(groups, MARCH_ROW);
    march_args[1] = (groups + MARCH_ROW - 1u) / MARCH_ROW;
    march_args[2] = 1u;
    atomicStore(&tally.groups, groups);
}

// What one dither cell of this frame held, in the two numbers the next frame
// asks about it.
//
// The first is the largest screen-space motion any pixel of the cell has, which
// is the signal the drop pattern spends its extra rays on: ground that is
// sweeping across the view is ground whose carried material and normal go
// stale soonest, and under forward flight it is also the near ground that
// magnifies fastest and so leaves the most gaps between splats.
//
// The second is the nearest ground in the cell, as reversed-Z depth, where
// larger is nearer and zero is sky. Motion alone cannot say whether what swept
// across a pixel had any business covering what was there: only something
// *nearer* occludes. See `cs_reach`, which pairs them.
//
// The maximum of each rather than the mean, because a cell is dropped or kept
// whole and the worst pixel in it is what decides whether keeping it shows.
//
// The two maxima are taken *independently*, so the pair reported for a cell
// need not come from any one pixel of it: a cell holding fast far ground and
// still near ground is described as fast and near. That is deliberate. It
// errs towards saying something could have arrived, which costs march time and
// never leaves a wrong pixel standing, where the other way round would. Pairing
// them -- carrying the depth of the fastest pixel -- was tried and is worse:
// flying a low camera into a hillside at 200 m/s, it leaves 101 pixels showing
// ground more than half again too far away against 94 for this, because the
// nearest thing in a cell is often not the fastest thing in it.
//
// One workgroup per cell, which is why the workgroup *is* the cell: the
// reduction is over exactly the pixels whose fate the answer decides.
var<workgroup> worst: array<vec2<f32>, 64>;

@compute @workgroup_size(8, 8)
fn cs_risk(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) cell: vec3<u32>,
    @builtin(local_invocation_index) slot: u32,
) {
    // Pixels past the edge take the identity of the reduction rather than
    // dropping out, so every lane has something defined to contribute.
    var here = vec2<f32>(0.0);
    if (all(id.xy < terrain.viewport)) {
        let motion = unpack2x16float(textureLoad(motion, vec2<i32>(id.xy), 0).r);
        here = vec2<f32>(
            max(abs(motion.x), abs(motion.y)),
            textureLoad(settled_depth, vec2<i32>(id.xy), 0).r,
        );
    }
    worst[slot] = here;
    workgroupBarrier();

    for (var stride = 32u; stride > 0u; stride >>= 1u) {
        if (slot < stride) {
            worst[slot] = max(worst[slot], worst[slot + stride]);
        }
        workgroupBarrier();
    }

    if (slot == 0u) {
        textureStore(out_risk, vec2<i32>(cell.xy), vec4<f32>(worst[0], 0.0, 0.0));
    }
}

// Side of one cell of the drop pattern, in pixels. Must match `DITHER_BLOCK` in
// `src/reproject.rs` and the workgroup size of `cs_risk` above.
const RISK_CELL: f32 = 8.0;

// How far, in cells, ground is looked for around a cell of sky.
//
// A clamp on the search, not on the answer: ground moving further than this
// reaches further than this says, and the sky beyond is carried on when it
// should not be. Eight was measured rather than picked. Flying a low camera at
// a hillside at 200 m/s -- the worst case the installed raster offers, the eye
// a few metres over the slope -- carrying sky wrongly settles 18,365 pixels of
// a 1280x720 frame; a reach of two cells leaves 5,766 of them, five leaves
// 1,584, and eight leaves none. The cost is the search, which is quadratic in
// this and paid once per cell rather than once per pixel.
const REACH_CELLS: i32 = 8;

// The nearest ground that can have swept across each cell since the last frame.
//
// The risk field says how fast the ground *in* a cell was moving and how near
// it was. Neither alone answers what a carried point needs to know. Motion
// alone is nothing at all for a cell holding only sky -- and a cell of sky is
// exactly the one about to be overrun by a ridge coming up from below it --
// while depth alone says nothing about whether the near thing has had time to
// get here. Paired, they do: a cell's ground reaches this one if it moved
// further than the gap between them, and what arrives is worth worrying about
// only if it is nearer than whatever it lands on.
//
// So this takes the nearest ground among the cells that can reach, and leaves
// the comparison against what is already here to `swept`, which is where the
// depth of the point being judged is known. Zero means nothing within reach
// moved far enough to arrive.
//
// One thread per cell rather than one per pixel, and a pass of its own because
// a cell needs its neighbours' finished risk. See `swept` in
// `src/reproject.wgsl` for what reads it.
@compute @workgroup_size(8, 8)
fn cs_reach(@builtin(global_invocation_id) id: vec3<u32>) {
    let cells = vec2<i32>((terrain.viewport + u32(RISK_CELL) - 1u) / u32(RISK_CELL));
    let cell = vec2<i32>(id.xy);
    if (any(cell >= cells)) {
        return;
    }
    var nearest = 0.0;
    for (var dy = -REACH_CELLS; dy <= REACH_CELLS; dy++) {
        for (var dx = -REACH_CELLS; dx <= REACH_CELLS; dx++) {
            let at = cell + vec2<i32>(dx, dy);
            // Out of bounds reads zero, which neither moves nor is near, so the
            // edges of the screen need no special case.
            let there = textureLoad(risk, at, 0).rg;
            // The narrowest gap between the two cells, in pixels, which is zero
            // for a cell against itself: ground moving at all in a cell puts
            // its own sky in doubt, and a neighbour's has to cross the space
            // between before it counts.
            let gap = RISK_CELL * f32(max(abs(dx), abs(dy)) - 1);
            if (there.r > max(gap, 0.0)) {
                nearest = max(nearest, there.g);
            }
        }
    }
    textureStore(out_reach, cell, vec4<f32>(nearest, 0.0, 0.0, 0.0));
}

// Deriving the max pyramid from the heights already resident, rather than
// streaming it as a product of its own.
//
// `terrain-process` builds the pyramid offline over the whole raster and the
// clipmap carries it in as a third set of tiles. Everything the recurrence
// needs is already in texture memory, though: this level's heights, and the
// level below's cells. So
//
//     M[l] = max(quad_max(level-l heights), reduce_max(M[l - 1]))
//
// which is the recurrence in `crates/terrain-tiles/src/maxima.rs` with the
// second term taken only over children a ray could actually descend into.
//
// Dropping the rest is not an approximation. The march descends into a finer
// level only where that level is resident -- see the descent in `march` -- so a
// child no ray can reach has nothing this cell must bound. A derived cell is
// therefore at or below the offline product's and never above it: every term it
// keeps is a term the offline chain has too, so the march cannot come out
// slower for reading these instead.
//
// What this cannot see on its own is a finer tile arriving after the coarse
// cells above it were written. Re-deriving those is the host's job; see
// `derive_maxima` in `src/terrain/gpu.rs`.

// One rectangle of one level to derive.
//
// Rectangles rather than whole levels because almost nothing changes between
// frames: a tile arrives, and what it invalidates is its own ground at every
// level from its own upwards.
struct MaximaJob {
    // North-west cell of the rectangle, in this level's own texels from the
    // raster origin -- an index, not a slot. `slot` wraps it, exactly as the
    // march does, so a rectangle is written where a ray will look for it.
    origin: vec2<i32>,
    // How many cells across and down.
    size: vec2<u32>,
    // The layer being written.
    level: u32,
    // Whether there is a level below to carry from. Zero at the finest level
    // resident, which has nothing under it.
    carry: u32,
    // The level below's resident range, in its own texels, exactly as the march
    // tests it -- so a child is carried if and only if a ray could descend to
    // it. Read only when `carry` is set.
    below_low: vec2<i32>,
    below_high: vec2<i32>,
    // `u32`s per row of `out_halves`, which is the byte stride the copy out of
    // it uses divided by four.
    stride: u32,
    // `f32`s per row of `out_surface`, likewise. A second stride rather than one
    // scaled by two: a row is padded up to what a buffer-to-texture copy demands
    // *after* it is sized, so the wider cells round differently.
    wide_stride: u32,
};

@group(3) @binding(12) var<uniform> job: MaximaJob;
// Cells go out through a buffer rather than a storage texture because `r16float`
// is not a storage format in WebGPU. A buffer costs one copy afterwards and
// needs no format widened, no feature asked for and no render pipeline in a
// file that has none.
//
// Two passes write through this: the pyramid's ceilings and the cover's lift.
// Both are half-precision cells of a level being rebuilt, so both want the same
// buffer, the same packing and the same copy.
@group(3) @binding(13) var<storage, read_write> out_halves: array<u32>;
// Full-precision cells, for the pass that also raises the ground. Elevations
// reach four figures and a half float is two metres coarse up there, so the
// surface a ray meets cannot go out packed the way the lift above it does.
@group(3) @binding(16) var<storage, read_write> out_surface: array<f32>;

// Generating the levels below the base, which no survey holds.
//
// The base is the finest ground anyone measured. Everything under it used to be
// streamed at a metre and is now invented: a level is a pure function of its
// position, so tiles are seamless with no overlap to handle and a window can be
// refilled a tile at a time as it moves.
//
// This pass writes the *surface*, once per texel. That distinction is the whole
// reason it is affordable -- see the archaeology above the march, where growing
// crowns per ray-texel cost three quarters of the frame and baking them per
// texel cost nothing measurable.
//
// A generated texel is three things added: the base read smoothly, the fractal
// octaves the base is too coarse to hold, and the crowns and stones standing on
// it at this level's own texel size. The first two are the survey's own ground
// put back at a resolution nobody stored; the third is a surface that was never
// ground at all, and it is why this pass writes an id as well as a height.
struct DetailJob {
    // North-west cell of the rectangle, in this level's own texels.
    origin: vec2<i32>,
    size: vec2<u32>,
    // The level being generated. Always below `resident_base`.
    level: u32,
    // Octaves of the fractal this level may carry: the ones whose features are
    // larger than two of its texels. Everything finer would alias into a
    // shimmer that moves with the camera rather than a detail that stays put.
    octaves: u32,
    // Feature size of the first octave, in metres.
    wavelength: f32,
    // Metres of relief the whole fractal adds where the ground is steepest.
    relief: f32,
};

@group(3) @binding(14) var<uniform> detail: DetailJob;
// Written as storage rather than through a buffer and a copy, which is what the
// max pyramid has to do: `r32float` is a storage format and `r16float` is not.
@group(3) @binding(15) var out_detail: texture_storage_2d_array<r32float, write>;
// Thirty-two bits for an id that fits in twelve, because `r16uint` is not a
// storage format either and a generated level is written by a dispatch rather
// than copied in. Three levels of a four-thousand-texel window is 192 MiB, which
// is the price of a near field that paints what it draws -- and it buys the
// ground cover as well as the crowns, since a level whose ids are its own is a
// level whose boundaries can sit somewhere the base grid could not put them.
@group(3) @binding(17) var out_detail_ids: texture_storage_2d_array<r32uint, write>;

// The fractal detail the survey does not hold, transcribed from
// `crates/terrain-generate/src/noise.rs`.
//
// That module was written to be transcribed -- `f32` arithmetic, `u32` bit
// twiddling, fixed loop bounds, a gradient table small enough to be a `const`
// array rather than a buffer to bind -- and this is the transcription it was
// waiting for. The values must not drift: a landscape generated offline and one
// generated here are meant to be the same landscape.

const GRADIENT_COUNT: u32 = 16u;
const DIAGONAL: f32 = 0.70710678;
const NEAR_AXIS: f32 = 0.9238795;
const OFF_AXIS: f32 = 0.3826834;

// Unit vectors at every 22.5 degrees. Sixteen rather than the eight a
// three-bit selector would give: eight leaves gradient noise with a visible
// bias along the axes and diagonals, which on a mountain reads as ridges that
// all run the same four ways.
const GRADIENTS = array<vec2<f32>, 16>(
    vec2<f32>(1.0, 0.0),
    vec2<f32>(NEAR_AXIS, OFF_AXIS),
    vec2<f32>(DIAGONAL, DIAGONAL),
    vec2<f32>(OFF_AXIS, NEAR_AXIS),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(-OFF_AXIS, NEAR_AXIS),
    vec2<f32>(-DIAGONAL, DIAGONAL),
    vec2<f32>(-NEAR_AXIS, OFF_AXIS),
    vec2<f32>(-1.0, 0.0),
    vec2<f32>(-NEAR_AXIS, -OFF_AXIS),
    vec2<f32>(-DIAGONAL, -DIAGONAL),
    vec2<f32>(-OFF_AXIS, -NEAR_AXIS),
    vec2<f32>(0.0, -1.0),
    vec2<f32>(OFF_AXIS, -NEAR_AXIS),
    vec2<f32>(DIAGONAL, -DIAGONAL),
    vec2<f32>(NEAR_AXIS, -OFF_AXIS),
);

// What unit gradient noise has to be multiplied by to reach `-1..=1`.
const GRADIENT_SCALE: f32 = 1.4142136;

// Wellons' `lowbias32`, whose avalanche is measured rather than assumed: one
// input bit flips about half the output bits, which is what stops neighbouring
// lattice points drawing correlated gradients and putting a grain in the
// terrain.
fn noise_mix(bits: u32) -> u32 {
    var b = bits;
    b ^= b >> 16u;
    b = b * 0x7feb352du;
    b ^= b >> 15u;
    b = b * 0x846ca68bu;
    b ^= b >> 16u;
    return b;
}

// The two coordinates are folded in one at a time, each through its own mixer,
// rather than combined and mixed once. Combining first is cheaper and wrong in
// a way that shows: they would meet only through a single xor, so whole
// diagonals of the lattice collide and the noise grows a herringbone.
fn noise_hash(x: i32, y: i32, seed: u32) -> u32 {
    var bits = seed * 0x9e3779b1u;
    bits = noise_mix(bits ^ (u32(x) * 0x3504f333u));
    bits = noise_mix(bits ^ (u32(y) * 0xf1bbcdcbu));
    return bits;
}

// Perlin's quintic interpolant, which has zero first *and* second derivative at
// both ends. The cubic is cheaper and leaves a second-derivative jump at every
// lattice line -- invisible in the height and very visible in the shading,
// because the normal comes from the heights.
fn noise_fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn noise_corner(at: vec2<i32>, offset: vec2<f32>, seed: u32) -> f32 {
    return dot(GRADIENTS[noise_hash(at.x, at.y, seed) % GRADIENT_COUNT], offset);
}

// Gradient noise in `-1..=1`, zero at every lattice point.
fn gradient_noise(p: vec2<f32>, seed: u32) -> f32 {
    let cell = floor(p);
    let f = p - cell;
    let at = vec2<i32>(cell);
    let u = vec2<f32>(noise_fade(f.x), noise_fade(f.y));
    let bottom = mix(
        noise_corner(at, f, seed),
        noise_corner(at + vec2<i32>(1, 0), f - vec2<f32>(1.0, 0.0), seed),
        u.x,
    );
    let top = mix(
        noise_corner(at + vec2<i32>(0, 1), f - vec2<f32>(0.0, 1.0), seed),
        noise_corner(at + vec2<i32>(1, 1), f - vec2<f32>(1.0, 1.0), seed),
        u.x,
    );
    return mix(bottom, top, u.y) * GRADIENT_SCALE;
}

// The most octaves any generated level sums, which is one per level below the
// base. A bound rather than a choice: WGSL wants a loop it can unroll.
const DETAIL_OCTAVES: u32 = 8u;

// Constants, because nothing that crosses a pipeline boundary depends on them.
// The heights get one field and the cover's two-component warp gets two more,
// all independent: one field used twice would put the warp on the diagonal, and
// a warp sharing the height's field would tie a boundary to a ridge.
const DETAIL_SEED: u32 = 0x4465746cu;
const COVER_SEED_X: u32 = 0x43767278u;
const COVER_SEED_Y: u32 = 0x43767279u;

// The fractal, in `-1..=1`, stopped before any octave this level cannot hold.
//
// Doubling exactly, rather than the 2.017 the offline generator uses to stop
// octaves reinforcing on the lattice lines they share. Here the octaves are
// *meant* to line up with something: a level's texels are a power of two, so an
// octave at twice a texel is the finest thing it can represent, and a lacunarity
// that drifted would put the band limits between levels instead of on them.
//
// The sum is normalised by the whole fractal's amplitude rather than by the
// octaves that survive the limit. That is the difference between a coarse level
// being *the same surface with its finest octaves removed* and it being a
// differently scaled one -- renormalising would make every coarse level louder
// than the one under it, and the handover would draw as the ground breathing.
fn fractal(p: vec2<f32>, wavelength: f32, octaves: u32, whole: u32, seed: u32) -> f32 {
    var frequency = 1.0 / wavelength;
    var amplitude = 1.0;
    var sum = 0.0;
    var total = 0.0;
    for (var octave = 0u; octave < DETAIL_OCTAVES; octave++) {
        if (octave >= whole) {
            break;
        }
        total += amplitude;
        if (octave < octaves) {
            sum += gradient_noise(p * frequency, seed ^ (octave * 0x51ed270bu)) * amplitude;
        }
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    if (total <= 0.0) {
        return 0.0;
    }
    return sum / total;
}

// The high byte of a ground-cover id, which is its category. Water is 0x01xx;
// see `crates/terrain-materials/src/lib.rs`.
const WATER_COVER: u32 = 1u;

// The last category nature draws the edges of. See `drawn_by_nature`.
const BARE_COVER: u32 = 5u;

// How far the cover boundary may wander from where the rasterizer left it, as a
// fraction of a base texel.
//
// A ceiling, not a typical displacement, and the difference is the whole reason
// this number is as large as it is. Gradient noise reaches its bound at one
// point in a lattice cell and is a small fraction of it nearly everywhere else,
// so an amplitude picked to keep the *worst* case inside half a base texel
// leaves the boundary sitting on the staircase it was meant to leave. At a
// quarter of a texel the checkerboard in
// `the_cover_wanders_across_the_blocks_it_was_stored_in` moved two texels of
// nine hundred and sixty-one; the blocks were still the blocks.
//
// What decides the number is where the warp starts folding. `centre +
// warp(centre)` is a displacement of the plane only while the warp's gradient
// stays under one; past that the map stops being one-to-one and a boundary
// stops being a boundary, fraying into detached specks of one cover stranded in
// the other -- which reads as dither rather than as a coastline. The gradient
// cannot be argued from the amplitude alone, because every octave contributes
// about the same slope and they add. So it was measured, on that checkerboard
// and by eye over a glacial lake in the installed survey:
//
// | `COVER_WARP` | texels moved | stranded | the shoreline            |
// | ---          | ---          | ---      | ---                      |
// | 0.25         | 2 of 961     | 0        | the 8 m staircase        |
// | 0.5          | 45           | 2        | the staircase, corners   |
// |              |              |          | bitten off               |
// | 1.0          | 137          | 3        | ragged and coherent      |
// | 1.5          | 196          | 7        | starting to fray         |
// | 2.0          | 231          | 14       | frayed                   |
// | 3.0          | 286          | 15       | dithered                 |
//
// One base texel, then. The stranded fraction of what moved is at its lowest
// there -- 2% against 4% below it, where nothing much moves at all, and 6%
// above it -- which is the folding and the wandering being weighed against each
// other rather than a number chosen by taste.
const COVER_WARP: f32 = 1.0;

// The survey's ground cover under one texel of any level.
//
// The base texel a texel falls in, exactly: the grids are aligned, so a level-`l`
// texel covers level-0 texels `[cell << l, (cell + 1) << l)` and every one of
// them sits in the same base texel.
fn cover_at(cell: vec2<i32>, level: u32) -> u32 {
    let base = terrain.resident_base;
    return textureLoad(materials, slot(base, cell >> vec2<u32>(base - level)), mip(base)).r;
}

// That cover, upscaled to one texel of a generated level.
//
// The same trade the heights make, for a quantity that cannot take it the same
// way. A height is a number and a level below the base gets the base
// interpolated smoothly and fractal relief added to it; an id is a label, and
// there is no interpolating between two labels -- the mean of a lake and a
// meadow is not a third kind of ground, it is a wrong one. So the fractal moves
// the *place the label is read from* instead of the label: the id at a texel is
// the survey's id at that texel's centre displaced by a band-limited fractal
// vector. Nothing is invented -- every id that comes out is an id the survey
// wrote within a base texel of here -- and what changes is only where one gives
// way to the next, which is exactly the thing the base grid quantised.
//
// Band-limited by the level, as the relief is and for the same reason: the warp
// carries one octave per level below the base, so a coarse generated level is
// the same displacement with its finest octaves dropped rather than a different
// one, and the wander shrinks as the level approaches the base the survey is
// stored at. That is what bounds the ring where the coarsest generated level
// hands over to the unwarped base: it carries one octave of the normalisation's
// 1.75, so its boundary can jog by at most `COVER_WARP / 1.75` of a base texel,
// and the ring is by definition the distance at which a base texel is about a
// pixel. Ceiling 0.57 of a pixel, then, against the half pixel the crates budget
// a handover -- and the ceiling is not what happens: the coarsest generated
// level of the checkerboard test reached 2 m of its 8 m base texel, a quarter of
// a pixel.
//
// Anchored in metres from the raster origin rather than in texels, so the same
// ground gets the same cover at every level and after every regeneration.
fn cover_upscaled(cell: vec2<i32>, level: u32, octaves: u32) -> u32 {
    let base = terrain.resident_base;
    let metres = terrain.metres_per_texel.x * f32(1u << base);
    let centre = cover_centre(cell, level);
    let warp = COVER_WARP * metres * vec2<f32>(
        fractal(centre, metres, octaves, base, COVER_SEED_X),
        fractal(centre, metres, octaves, base, COVER_SEED_Y),
    );
    // Which base texel the displaced point landed in. With no displacement this
    // is `cover_at`'s own shift exactly -- `(cell + 0.5) / 2^(base - level)`
    // floors to `cell >> (base - level)` -- so the warp is the only thing that
    // moves an id, and turning it off gives the blocks back rather than
    // something a half texel off them.
    let node = vec2<i32>(floor((centre + warp) / metres));
    let there = textureLoad(materials, slot(base, node), mip(base)).r;
    let here = cover_at(cell, level);
    if (there == here || !drawn_by_nature(there) || !drawn_by_nature(here)) {
        return here;
    }
    return there;
}

// Whether a boundary this cover takes part in is one nature drew.
//
// The categories are ordered, and the order is not an accident: water, wetland,
// forest, scrub, bare ground -- then agriculture, developed ground, and
// maintained leisure ground. Everything up to bare ground has an edge that some
// process put where it is and the base grid then squared off, which is exactly
// what the warp is for. Everything after it has an edge somebody *drew*: a
// field boundary, a road, a fairway. Those were straight before the survey
// quantised them, and roughening a straight line invents the opposite of what
// is true -- rendered over a town at the amplitude below, the road network
// dissolved into grey blotches while the forest beside it improved. So a
// boundary with a drawn cover on either side of it stays exactly where the
// survey left it, staircase and all, and only nature's boundaries move.
//
// `Null` counts as nature's. It is the id for ground no mapped area covers,
// which is wilderness rather than a decision about a boundary.
fn drawn_by_nature(cover: u32) -> bool {
    return (cover >> 8u) <= BARE_COVER;
}

// Slopes between which the relief comes in, as a rise over run.
//
// Flat ground gets none. A survey's flat ground is genuinely flat -- a field, a
// lake, a road -- and roughening it invents texture where the measurement says
// there is none, which reads as noise rather than as terrain. A hillside is
// where the box filter that made the base threw detail away, and it is where
// putting some back is honest.
const RELIEF_FLAT: f32 = 0.08;
const RELIEF_STEEP: f32 = 0.60;

// What the four nodes around a position contribute to it, for a fraction of the
// way from the second to the third.
//
// The Catmull-Rom cubic: the curve passes through the two middle nodes, and its
// slope at each is the central difference of that node's own neighbours, which
// is what makes the pieces meet with a matching gradient. The four weights
// always add to one, so a flat field stays flat and a lake stays exactly at its
// own level.
fn catmull_rom(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

// The base surface at a point given in the base level's own texels.
//
// Sixteen taps rather than a bilinear four, and the reason is the same one
// `fade` exists for one level down: a bilinear surface is continuous but its
// gradient is not, so every cell line draws as a crease. That is invisible in
// the height and very visible in the shading, because the normal is derived
// from the heights. Measured in the offline generator, the second difference
// along a row was 11.1 times larger on the lattice than off it, and Catmull-Rom
// took that to 1.33.
//
// A quintic fade on the bilinear weights is cheaper and also smooth, and it was
// rejected there for a reason that holds here: it forces the gradient to zero
// at every node, so a hillside comes out as a quilt of level patches with steps
// between them. Catmull-Rom passes through the nodes with the slope the data
// implies and reproduces any linear surface exactly, so a ramp gains no ripples
// and a lake stays flat.
// The survey's own ground at one texel of the base, with whatever stands on it
// taken back off.
//
// Only mip zero carries a surface rather than a survey, so this is the only
// level the question arises at. Subtracting is exact -- the lift went in as the
// half float that comes back out -- and it is what keeps a generated level from
// growing the base's crowns a second time on top of its own.
fn bare_above(level: u32, cell: vec2<i32>) -> f32 {
    return resident_height_at(level, cell) - lift_at(level, cell);
}

fn lift_at(level: u32, cell: vec2<i32>) -> f32 {
    return textureLoad(lift, slot(level, cell), mip(level)).r;
}

fn bare_height_at(cell: vec2<i32>) -> f32 {
    let base = terrain.resident_base;
    let height = resident_height_at(base, cell);
    if (height < NODATA_BELOW) {
        return height;
    }
    return height - lift_at(base, cell);
}

fn base_height(at: vec2<f32>) -> f32 {
    let node = floor(at);
    let across = catmull_rom(at.x - node.x);
    let down = catmull_rom(at.y - node.y);
    let corner = vec2<i32>(node) - vec2<i32>(1);

    var total = 0.0;
    var deepest = NEVER;
    for (var row = 0; row < 4; row++) {
        var line = 0.0;
        for (var column = 0; column < 4; column++) {
            let sample = bare_height_at(corner + vec2<i32>(column, row));
            line += across[column] * sample;
            deepest = min(deepest, sample);
        }
        total += down[row] * line;
    }
    // A hole may not be interpolated. Three real metres blended with one
    // sentinel comes out around -8000, which is far below any ground and
    // nowhere near the sentinel, so the march's own test would read it as a
    // cliff dropping eight kilometres rather than as a hole. Spreading the hole
    // by the width of the kernel instead draws slightly more of the survey's
    // edge as nothing, which is the safe direction.
    if (deepest < NODATA_BELOW) {
        return deepest;
    }
    return total;
}

// One thread per texel of a generated tile.
@compute @workgroup_size(8, 8)
fn cs_detail(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= detail.size.x || id.y >= detail.size.y) {
        return;
    }
    let level = detail.level;
    let cell = detail.origin + vec2<i32>(id.xy);
    // This texel's position in the base level's own texels. The grids line up
    // exactly -- level `l`'s sample `i` sits at level-0 texel `i * 2^l` -- so a
    // generated sample at an even index falls on a base node, and Catmull-Rom
    // passes through it. That is what stops the handover between a generated
    // level and the base from showing.
    let base = terrain.resident_base;
    let at = vec2<f32>(cell) / f32(1u << (base - level));
    var height = base_height(at);
    // The survey's ground cover, upscaled to this level's texels. Everything
    // below reads this one answer rather than the base texel's own id: the
    // water it holds flat, the stand it grows, and the id it paints are all the
    // same question about the same texel, and asking it twice at two
    // resolutions is how a crown ends up standing on ground painted as meadow.
    var painted = cover_upscaled(cell, level, detail.octaves);

    // Nothing is added to a hole. The sentinel is what says the survey measured
    // nothing here, and a hole with texture on it is ground. The cover still
    // stands: a hole is ground nobody measured the height of, not ground nobody
    // mapped, and the id is what the shading has to paint it with either way.
    if (height >= NODATA_BELOW) {
        // How steep the base is under this point, as a rise over run, from the
        // four samples around the node. The relief comes in with it: a box
        // filter throws away most on a hillside and nothing at all on a flat,
        // so that is where putting some back is honest.
        let node = vec2<i32>(floor(at));
        let metres = terrain.metres_per_texel * f32(1u << base);
        let fall = vec2<f32>(
            bare_height_at(node + vec2<i32>(1, 0)) - bare_height_at(node - vec2<i32>(1, 0)),
            bare_height_at(node + vec2<i32>(0, 1)) - bare_height_at(node - vec2<i32>(0, 1)),
        );
        let slope = length(fall / (2.0 * metres));
        var relief = detail.relief * smoothstep(RELIEF_FLAT, RELIEF_STEEP, slope);

        // A lake with waves in it is worse than a lake with none, and the
        // survey is right about water being flat. Asked of the upscaled cover,
        // so the flat ends exactly where the blue does.
        if ((painted >> 8u) == WATER_COVER) {
            relief = 0.0;
        }

        // In metres from the raster origin, so the detail is anchored to the
        // world rather than to the window: a tile regenerated after the camera
        // has been away comes back exactly as it was.
        let world = vec2<f32>(cell) * f32(1u << level) * terrain.metres_per_texel;
        height += relief * fractal(world, detail.wavelength, detail.octaves, base, DETAIL_SEED);

        // And then what stands on it, at this level's own texel size. The base
        // carries the same stand sampled over eight metres; here it is sampled
        // over one, so a crown is a crown rather than the eighth of a hillside
        // it averages into up there.
        let grown = standing_at(painted, cell, level);
        height += grown.lift;
        // Only where something stands: a crown hides the ground and takes the
        // texel's id with it, and everything else leaves the ground's own.
        if (grown.id != 0u) {
            painted = grown.id;
        }
    }

    textureStore(
        out_detail,
        cell & terrain.levels[level].mask,
        i32(level),
        vec4<f32>(height, 0.0, 0.0, 0.0),
    );
    // The whole id, ground and stand together, because both were decided here
    // and neither can be recovered from the survey afterwards: the ground's own
    // id is the upscaled one rather than the base texel's, and what stands on it
    // is a walk of this texel. A crown drawn and a meadow painted is the failure
    // both shares exist to prevent, and it can only be prevented by writing them
    // off the same walk.
    textureStore(out_detail_ids, cell & terrain.levels[level].mask, i32(level), vec4<u32>(painted));
}

// What stands on the ground, rather than what the ground is.
//
// `crates/terrain-canopy` and `crates/terrain-rocks` transcribed, and the two
// crates' own docs carry the arguments this does not repeat: why a stand needs
// three sources of irregularity to look unplanted, why a crown is nearly a cone
// and a boulder nearly a dome, why the rubble is a second lattice rather than a
// wider spread of radii on the first, and why every threshold sits where it
// does.
//
// The whole of that ran offline until now, once per texel of every level of the
// stored products, and `a5b0704` is why: growing crowns *in the march* cost
// three quarters of the frame, because entering the wooded path was paid once
// per ray-texel and a ray crosses a forest a texel at a time. Nothing about that
// has changed. What has changed is that the products are no longer stored at the
// resolution a crown lives at, so the same per-texel walk has to happen here --
// and the pass it happens in writes a texture, so it is still paid once per
// texel. The inversion `a5b0704` identified is the one this keeps.
//
// It is also the second reason that commit gives for baking, turned around. The
// crown field had to exist twice, once in Rust for the offline pyramid and once
// in WGSL for the march, and a GPU test existed only to keep the two spellings
// honest. Here there is one spelling again, because the pyramid is derived from
// the surface this raises rather than reduced from a product beside it.

// The crown lattice. See `crates/terrain-canopy/src/lib.rs`, whose constants
// these are; the names carry the crate's prefix so the two scatters can share
// this file.
const CANOPY_SPACING: f32 = 7.0;
const CANOPY_SHORTEST: f32 = 15.0;
const CANOPY_TALLEST: f32 = 28.0;
const CANOPY_RADIUS: f32 = 3.5;
const CANOPY_ROUNDNESS: f32 = 0.15;
const CANOPY_FLOOR: f32 = 0.35;
const CANOPY_SEED: u32 = 0x54726565u;
const CANOPY_WAVELENGTH: f32 = 34.0;
const CLUMP_THINNEST: f32 = 0.6;
const CLUMP_THICKEST: f32 = 1.9;
const CANOPY_SILHOUETTE: f32 = 0.15;
const PAINTED: f32 = 0.25;

// The two stone lattices. See `crates/terrain-rocks/src/lib.rs`.
const BOULDER_SPACING: f32 = 24.0;
const RUBBLE_SPACING: f32 = 3.0;
const BOULDER_RADIUS: f32 = 5.0;
const RUBBLE_RADIUS: f32 = 1.2;
const BOULDER_SHORTEST: f32 = 2.5;
const BOULDER_TALLEST: f32 = 9.0;
const RUBBLE_SHORTEST: f32 = 0.4;
const RUBBLE_TALLEST: f32 = 1.6;
const STONE_ROUNDNESS: f32 = 0.9;
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

// How wide a band of density a crown or a stone grows in over. Both crates set
// it to the same three tenths and for the same reason: the cover is evaluated
// once per texel and handed to every sample inside it, so two neighbouring
// texels disagree slightly about a lattice cell that straddles them. A hard
// threshold turns that into a tree sliced vertically down the middle; a band
// turns it into a tree a few centimetres shorter on one side.
const COVER_EDGE: f32 = 0.30;

// The coarsest texel the rubble is scattered on, in metres.
//
// Not in either crate: they scatter both classes at every size. It is here
// because of what the finer class costs to *find*. `stone_samples` spaces its
// samples by a quarter of the smaller radius, so a rubble-bearing texel of eight
// metres wants twenty-seven samples across where a boulder-bearing one wants
// seven -- fifteen times the walk, over the whole resident base, to resolve a
// stone 2.4 m wide and at most 1.6 m tall.
//
// What that costs is a step at the ring where the last level fine enough to
// carry rubble hands over to the first that is not. The rubble's own order
// statistic is a few tenths of a metre, and a two-metre texel is a pixel at
// 3.7 km, so the step is under a fifth of a pixel: the same slack the crates
// spend their `SILHOUETTE` on, measured the same way.
const RUBBLE_TEXEL: f32 = 2.0;

// Hermite ease, flat at both ends -- the crates' `fade`, which is *not*
// `noise_fade`. The gradient noise above wants a quintic because a
// second-derivative jump shows through the shading; a density field feeding a
// threshold does not care, and both crates use the cubic.
fn cover_fade(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn cover_ramp(edge0: f32, edge1: f32, t: f32) -> f32 {
    return cover_fade(clamp((t - edge0) / (edge1 - edge0), 0.0, 1.0));
}

// One smooth field in `0..=1` off a lattice of the given size.
//
// The crates take three fields out of one set of four hashes by splitting each
// word into three ten-bit slices, because their callers want all three. Every
// caller here wants exactly one, so the slice is a parameter and the saving is
// the other two fields rather than the other three hashes. The slice numbers
// still have to match theirs: `clump` is the third, the boulder field the
// first, the strewing the second.
fn cover_field(p: vec2<f32>, wavelength: f32, seed: u32, slice: u32) -> f32 {
    let u = p / wavelength;
    let cell = floor(u);
    let f = vec2<f32>(cover_fade(u.x - cell.x), cover_fade(u.y - cell.y));
    let at = vec2<i32>(cell);
    let shift = 10u * slice;
    let take = 1.0 / 1023.0;
    let a = f32((noise_hash(at.x, at.y, seed) >> shift) & 0x3ffu) * take;
    let b = f32((noise_hash(at.x + 1, at.y, seed) >> shift) & 0x3ffu) * take;
    let c = f32((noise_hash(at.x, at.y + 1, seed) >> shift) & 0x3ffu) * take;
    let d = f32((noise_hash(at.x + 1, at.y + 1, seed) >> shift) & 0x3ffu) * take;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// A stand thins and thickens inside itself rather than being uniform to its
// edge; a boulder field is there or not there at all; rubble covers a slope
// everywhere and only varies in how thickly. Those three shapes are why the
// canopy multiplies, the boulders gate, and the rubble multiplies again.
fn canopy_clump(p: vec2<f32>) -> f32 {
    return mix(CLUMP_THINNEST, CLUMP_THICKEST, cover_field(p, CANOPY_WAVELENGTH, CANOPY_SEED, 2u));
}

fn stone_field(p: vec2<f32>) -> f32 {
    let noise = cover_field(p, FIELD_WAVELENGTH, BOULDER_SEED, 0u);
    return FIELD_THICKEST * cover_ramp(FIELD_EDGE, FIELD_FULL, noise);
}

fn stone_strew(p: vec2<f32>) -> f32 {
    let noise = cover_field(p, STREW_WAVELENGTH, BOULDER_SEED, 1u);
    return mix(STREW_THINNEST, STREW_THICKEST, noise);
}

// How high the crowns reach over one point, in metres above the ground.
//
// Nine cells rather than one, because a crown wide enough to close a canopy
// reaches out of its own cell and has to be findable from the cells beside it.
// Those nine hashes are the whole cost of this file.
fn crown_at(p: vec2<f32>, density: f32, health: f32) -> f32 {
    if (health <= 0.0 || density <= 0.0) {
        return 0.0;
    }
    // The understorey: a floor under the crowns for the ground between them,
    // faded out by density so a clearing does not draw as a raised plate. It is
    // not canopy, which is what the share below counts.
    var found = CANOPY_FLOOR * CANOPY_SHORTEST * health * min(density, 1.0);
    let home = vec2<i32>(floor(p / CANOPY_SPACING));

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let cell = home + vec2<i32>(dx, dy);
            let bits = noise_hash(cell.x, cell.y, CANOPY_SEED);
            // Four fields out of one word. Splitting a hash beats taking four:
            // a crown costs two mixers rather than eight, and the fields are
            // independent because the mixer's avalanche already made every
            // output bit depend on every input bit.
            let jitter = vec2<f32>(f32(bits & 0x3ffu), f32((bits >> 10u) & 0x3ffu)) * (1.0 / 1024.0);
            let size = f32((bits >> 20u) & 0x3fu) * (1.0 / 64.0);
            let wants = f32((bits >> 26u) & 0x3fu) * (1.0 / 64.0);

            let grow = cover_fade(clamp((density - wants) / COVER_EDGE, 0.0, 1.0));
            if (grow <= 0.0) {
                continue;
            }
            // Anywhere in its own cell, which is what stops the stand drawing
            // as a grid, and a short tree is a narrow one.
            let trunk = (vec2<f32>(cell) + jitter) * CANOPY_SPACING;
            let scale = grow * (0.72 + 0.28 * size);
            let radius = max(CANOPY_RADIUS * scale, 1.0 / 1024.0);
            let height = health * mix(CANOPY_SHORTEST, CANOPY_TALLEST, size) * grow;

            let u = length(p - trunk) / radius;
            if (u < 1.0) {
                let cone = 1.0 - u;
                let dome = sqrt(max(1.0 - u * u, 0.0));
                found = max(found, height * mix(cone, dome, CANOPY_ROUNDNESS));
            }
        }
    }
    return found;
}

// How high the stones of one class stand over one point, in metres.
//
// The crown's walk with no understorey under it: a boulder field is stones lying
// on ground you can see between, so the gaps are the ground at the ground's own
// height, and a floor here would draw every talus slope as a raised plate with
// lumps on it.
fn stone_class(
    p: vec2<f32>,
    density: f32,
    stature: f32,
    spacing: f32,
    radius: f32,
    shortest: f32,
    tallest: f32,
    seed: u32,
) -> f32 {
    if (density <= 0.0 || stature <= 0.0) {
        return 0.0;
    }
    var found = 0.0;
    let home = vec2<i32>(floor(p / spacing));

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let cell = home + vec2<i32>(dx, dy);
            let bits = noise_hash(cell.x, cell.y, seed);
            let jitter = vec2<f32>(f32(bits & 0x3ffu), f32((bits >> 10u) & 0x3ffu)) * (1.0 / 1024.0);
            let grade = f32((bits >> 20u) & 0x3fu) * (1.0 / 64.0);
            let wants = f32((bits >> 26u) & 0x3fu) * (1.0 / 64.0);

            let grow = cover_fade(clamp((density - wants) / COVER_EDGE, 0.0, 1.0));
            if (grow <= 0.0) {
                continue;
            }
            let middle = (vec2<f32>(cell) + jitter) * spacing;
            let scale = grow * (0.72 + 0.28 * grade);
            let reach = max(radius * scale, 1.0 / 1024.0);
            let height = stature * mix(shortest, tallest, grade) * grow;

            let u = length(p - middle) / reach;
            if (u < 1.0) {
                let cone = 1.0 - u;
                let dome = sqrt(max(1.0 - u * u, 0.0));
                found = max(found, height * mix(cone, dome, STONE_ROUNDNESS));
            }
        }
    }
    return found;
}

// How many samples across a texel each walk takes.
//
// A quarter of the finest radius in play, floored at four however fine the texel
// is and capped at thirty-two however coarse. Both halves serve one end: a crown
// rises eight metres for every metre of ground, so a sample that misses the apex
// misses the height by an amount that depends on the texel size -- and a
// clipping that varies by level is a stand that changes height at every ring,
// which is the pop the crates were rebuilt to remove.
fn canopy_samples(texel: f32) -> u32 {
    return clamp(u32(ceil(texel / (0.25 * CANOPY_RADIUS))), 4u, 32u);
}

fn stone_samples(texel: f32) -> u32 {
    let radius = select(BOULDER_RADIUS, RUBBLE_RADIUS, texel <= RUBBLE_TEXEL);
    return clamp(u32(ceil(texel / (0.25 * radius))), 4u, 32u);
}

// Buckets the order statistic below counts its samples into.
const COVER_BUCKETS: u32 = 16u;

// The mean of the tallest `share` of a block, from cumulative counts and sums.
//
// The answer has to be an *average*, or it does not survive a change of texel
// size: averaging is scale-invariant, so a stand keeps its height however coarse
// the texel holding it gets, while a maximum climbs -- a wider block has more
// chances to land on an apex, and `8c928a9` measured a closed stand growing
// eleven metres from a one-metre texel to a sixteen-metre one, a step at every
// ring and forest that grows as you fly away from it. It also has to be an
// average of the *tall* part, or a distant forest draws as a green-painted
// hillside twenty metres short of its own treetops, because the honest mean of
// this canopy is a quarter of its own height.
//
// The crates get that by sorting the block and taking a prefix. A shader cannot:
// a hundred samples is a hundred registers, and an array indexed by a running
// position lands in scratch memory. So the samples are counted into sixteen
// buckets on the way past -- cumulatively, `counts[k]` being how many reached
// the k'th edge, which is what makes the accumulation sixteen fixed adds rather
// than one indexed one -- and the quantile is read back off them, interpolating
// inside whichever bucket the boundary falls in. That approximates the exact
// order statistic by at most the spread within one bucket, and it approximates
// it the same way at every level, which is the property that actually matters
// here: what must not move between two levels is the *bias*.
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
    for (var k = i32(COVER_BUCKETS) - 1; k >= 0; k--) {
        if (!found && counts[k] >= taken) {
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
    // own. Taken from the same walk as the height, because a texel that is drawn
    // as a tree and painted as a meadow is worse than either.
    id: u32,
}

// The trees on a texel, given how closed the stand is and how well it grows.
fn canopy_baked(centre: vec2<f32>, texel: f32, density: f32, health: f32) -> Standing {
    let across = canopy_samples(texel);
    let step = texel / f32(across);
    // Sample centres, so the block is symmetric about the texel and a texel
    // twice the size of its neighbour covers the same ground its four children
    // did between them.
    let first = 0.5 * step - 0.5 * texel;
    let floor_height = CANOPY_FLOOR * CANOPY_SHORTEST * health * min(density, 1.0);
    let top = max(health * CANOPY_TALLEST, 1e-6);

    var counts = array<f32, 17>();
    var sums = array<f32, 17>();
    var under = 0.0;
    var samples = 0.0;
    for (var row = 0u; row < across; row++) {
        for (var column = 0u; column < across; column++) {
            let at = centre + first + vec2<f32>(f32(column), f32(row)) * step;
            let here = crown_at(at, density, health);
            let rung = here * (f32(COVER_BUCKETS) / top);
            for (var k = 0; k < i32(COVER_BUCKETS); k++) {
                let hit = select(0.0, 1.0, f32(k) <= rung);
                counts[k] += hit;
                sums[k] += here * hit;
            }
            // A sample is under a crown when it stands on something taller than
            // the forest floor.
            under += select(0.0, 1.0, here > floor_height);
            samples += 1.0;
        }
    }

    var out: Standing;
    out.lift = tallest_mean(counts, sums, samples, CANOPY_SILHOUETTE);
    out.id = select(0u, CANOPY_ID, under / samples >= PAINTED);
    return out;
}

// The stones on a texel, given what is scattered on it.
fn stone_baked(
    centre: vec2<f32>,
    texel: f32,
    boulders: f32,
    rubble: f32,
    stature: f32,
) -> Standing {
    let across = stone_samples(texel);
    let step = texel / f32(across);
    let first = 0.5 * step - 0.5 * texel;
    let strewn = texel <= RUBBLE_TEXEL;
    let top = max(stature * BOULDER_TALLEST, 1e-6);

    var counts = array<f32, 17>();
    var sums = array<f32, 17>();
    var under_boulder = 0.0;
    var under_stone = 0.0;
    var samples = 0.0;
    for (var row = 0u; row < across; row++) {
        for (var column = 0u; column < across; column++) {
            let at = centre + first + vec2<f32>(f32(column), f32(row)) * step;
            let block = stone_class(
                at,
                boulders,
                stature,
                BOULDER_SPACING,
                BOULDER_RADIUS,
                BOULDER_SHORTEST,
                BOULDER_TALLEST,
                BOULDER_SEED,
            );
            var here = block;
            if (strewn) {
                here = max(
                    here,
                    stone_class(
                        at,
                        rubble,
                        stature,
                        RUBBLE_SPACING,
                        RUBBLE_RADIUS,
                        RUBBLE_SHORTEST,
                        RUBBLE_TALLEST,
                        RUBBLE_SEED,
                    ),
                );
            }
            let rung = here * (f32(COVER_BUCKETS) / top);
            for (var k = 0; k < i32(COVER_BUCKETS); k++) {
                let hit = select(0.0, 1.0, f32(k) <= rung);
                counts[k] += hit;
                sums[k] += here * hit;
            }
            // No floor to compare against: the ground between the stones is the
            // ground, so anything above it is a stone.
            under_boulder += select(0.0, 1.0, block > 0.0);
            under_stone += select(0.0, 1.0, here > 0.0);
            samples += 1.0;
        }
    }

    var out: Standing;
    out.lift = tallest_mean(counts, sums, samples, STONE_SILHOUETTE);
    // Boulder first, so a texel with both gets the coarser answer, which is the
    // one a viewer can actually resolve.
    out.id = select(
        select(0u, RUBBLE_ID, under_stone / samples >= STREWN),
        BOULDER_ID,
        under_boulder / samples >= BOULDERED,
    );
    return out;
}

// Ground-cover ids this file reads or writes; see
// `crates/terrain-materials/src/lib.rs`.
const CANOPY_ID: u32 = 0x0304u;
const BOULDER_ID: u32 = 0x0508u;
const RUBBLE_ID: u32 = 0x0509u;
const SCRUB_ID: u32 = 0x0400u;
const SHRUBBERY_ID: u32 = 0x0401u;
const HEATH_ID: u32 = 0x0402u;
const GRASSLAND_ID: u32 = 0x0403u;
const MEADOW_ID: u32 = 0x0405u;
const FELL_ID: u32 = 0x0407u;
const BARE_ROCK_ID: u32 = 0x0500u;
const SCREE_ID: u32 = 0x0501u;
const SHINGLE_ID: u32 = 0x0502u;
const SAND_ID: u32 = 0x0503u;
const BARE_EARTH_ID: u32 = 0x0506u;
const CLEARCUT_ID: u32 = 0x0303u;
const FOREST_COVER: u32 = 3u;

// What the survey's ground cover and the slope under it grow or shed.
//
// `terrain-generate` has a classifier for this and it is a much better one: it
// asks the fields it built the landscape out of -- the treeline, the flow, the
// hardness of the bed, where the ice dropped its load -- about the very point it
// is writing. None of that exists for a survey. What exists is one ground-cover
// id per base texel, rasterized from OpenStreetMap, and the slope of the
// measured ground under it, so this is what those two can say.
//
// That is less of a loss than it sounds. The mapped cover already knows where
// the treeline is, because the polygons stop there -- a fact about *this*
// landscape rather than a model of one -- so the terms `timber` drops are the
// ones the survey has already answered. The terms it keeps are the ones the
// survey cannot: how a stand thins on a slope, and the mottling that stops one
// density per polygon drawing as one flat density, which the noise fields above
// put back.
//
// The five kinds of strewn ground come straight from `classify::rocks`, with the
// generator's `hardness` and `filling` -- fields no survey carries -- taken at
// their midpoint, and its `mottle` dropped because `stone_field` and
// `stone_strew` already vary the same numbers over the same distances.
fn cover_slope_footing(slope: f32) -> f32 {
    // Steep ground holds less soil and fewer trees, and what it holds is
    // smaller. It does not stop the forest -- conifers root on slopes nobody
    // would walk up, which is why the cover says wooded here at all.
    return 1.0 - smoothstep(0.30, 0.85, slope);
}

fn standing_on(cover: u32, centre: vec2<f32>, texel: f32, slope: f32) -> Standing {
    let clump = canopy_clump(centre);
    if (cover >> 8u) == FOREST_COVER && cover != CLEARCUT_ID && cover != CANOPY_ID {
        let footing = mix(0.78, 1.0, cover_slope_footing(slope));
        return canopy_baked(centre, texel, 0.92 * footing * clump, 0.95 * footing);
    }
    // Krummholz: the same trees, beaten down to head height and scattered. Flat
    // numbers, because nothing about this belt varies in a way that reads from
    // the air -- what makes it look right is that it is sparse and short.
    if (cover == SCRUB_ID || cover == SHRUBBERY_ID) {
        return canopy_baked(centre, texel, 0.30 * clump, 0.17);
    }

    let steep = smoothstep(0.20, 1.00, slope);
    let field = stone_field(centre);
    let strew = stone_strew(centre);
    var scatter = vec3<f32>(0.0);
    if (cover == SCREE_ID) {
        // Talus: the rubble a cliff sheds, with the blocks that did not break up
        // lying in it.
        scatter = vec3<f32>(0.40, 0.72, 0.7);
    } else if (cover == BARE_ROCK_ID) {
        // A cliff: swept, with what the ledges managed to hold. A face steep
        // enough to read as bare rock has already sent anything loose to the
        // talus below it, so the rubble fades out with the steepness.
        scatter = vec3<f32>(0.22, 0.15 * (1.0 - steep), 0.9);
    } else if (cover == FELL_ID || cover == HEATH_ID) {
        // Felsenmeer: the alpine plateau, shattered in place. The blocks want
        // flat ground, because anything steep enough to shed them has.
        scatter = vec3<f32>(0.18 + 0.12 * (1.0 - steep), 0.50, 0.75);
    } else if (cover == SHINGLE_ID) {
        // A gravel bar: cobbles and not much else. The stature is what is low
        // rather than the densities -- a river that could roll a five-metre
        // block would not have left it here.
        scatter = vec3<f32>(0.05, 0.65, 0.35);
    } else if (
        cover == GRASSLAND_ID
        || cover == MEADOW_ID
        || cover == SAND_ID
        || cover == BARE_EARTH_ID
    ) {
        // Erratics: single blocks standing on ground that has none of their
        // kind. The sparsest and the most visible, which is not a contradiction
        // -- a valley floor is flat, open and looked at from low down, so one
        // block on it reads from a long way off, and it is the case
        // `Material::Boulder` exists for because the ground under it is grass.
        scatter = vec3<f32>(0.18, 0.05, 1.0);
    } else {
        var bare: Standing;
        bare.lift = 0.0;
        bare.id = 0u;
        return bare;
    }
    return stone_baked(centre, texel, scatter.x * field, scatter.y * strew, scatter.z);
}

// The middle of a texel of some level, in metres from the raster origin.
//
// The centre rather than the corner the height is indexed at, because a walk
// covers the *block* a texel stands for: a level-`l` sample is a mean over the
// square of level-0 texels starting at `cell << l`, so centring the block there
// is what makes a coarse texel cover exactly the ground its four children do.
fn cover_centre(cell: vec2<i32>, level: u32) -> vec2<f32> {
    let texels = f32(1u << level);
    return (vec2<f32>(cell) + 0.5) * texels * terrain.metres_per_texel.x;
}

// One thread per pair of cells across, as `cs_maxima` is and for the same
// reason: the cells go out through a storage buffer as packed halves.
@compute @workgroup_size(8, 8)
fn cs_cover(@builtin(global_invocation_id) id: vec3<u32>) {
    let pairs = (job.size.x + 1u) / 2u;
    if (id.x >= pairs || id.y >= job.size.y) {
        return;
    }
    let at = job.origin + vec2<i32>(i32(id.x) * 2, i32(id.y));
    let low = level_lift(at);
    var high = low;
    if (id.x * 2u + 1u < job.size.x) {
        high = level_lift(at + vec2<i32>(1, 0));
    }
    out_halves[id.y * job.stride + id.x] = ceiling_half(low) | (ceiling_half(high) << 16u);
    // The raised surface goes out beside the lift, off the same walk, because
    // the walk is the expensive half of this file and doing it twice to get two
    // numbers out of it would double the load.
    //
    // Both are wanted afterwards and neither can be recovered from the other
    // cheaply: the march reads a surface and must not pay two fetches a step to
    // get one, and `cs_detail` reads the survey underneath -- a generated level
    // grows its own crowns at its own texel size, so interpolating the base's
    // would count them twice.
}

// One thread per pair of cells across, over the same rectangles once the whole
// lift chain is written.
//
// A second sweep rather than a second output of the first, because the walk
// above reads the survey and this writes over it. Adding as it went would leave
// the pass reading rectangles it had already raised -- and what it reads them
// for is the slope, which is a question about the ground rather than about what
// is standing on it.
@compute @workgroup_size(8, 8)
fn cs_raise(@builtin(global_invocation_id) id: vec3<u32>) {
    let pairs = (job.size.x + 1u) / 2u;
    if (id.x >= pairs || id.y >= job.size.y) {
        return;
    }
    let at = job.origin + vec2<i32>(i32(id.x) * 2, i32(id.y));
    let wide = id.y * job.wide_stride + id.x * 2u;
    out_surface[wide] = level_surface(at);
    if (id.x * 2u + 1u < job.size.x) {
        out_surface[wide + 1u] = level_surface(at + vec2<i32>(1, 0));
    }
}

fn level_surface(cell: vec2<i32>) -> f32 {
    let height = resident_height_at(job.level, cell);
    // Nothing stands on a hole. The sentinel is what says the survey measured
    // nothing here, and a hole with trees on it is ground.
    if (height < NODATA_BELOW) {
        return height;
    }
    return height + lift_at(job.level, cell);
}

// How much of one resident texel's surface is what stands on it.
//
// Walked at the base and *averaged* at every level above it, and that split is
// the whole design of this pass rather than a saving in it.
//
// Walking every level is what the offline generator did, and it is the answer
// each level would give for itself: the mean of the tallest fraction of its own
// block, which climbs a little as the block widens because a wider block reaches
// further into the tall tail. Averaging instead gives a level the mean of its
// four children, which is not quite that number -- a coarse level comes out a
// couple of metres short of what it would have found on its own -- and in
// exchange the chain has *no step anywhere in it*, because a parent is the mean
// of its children by construction rather than by an argument about how nearly
// scale-invariant an order statistic is.
//
// The cost decides it either way. A thirty-two metre texel wants a thousand
// crown samples where an eight-metre one wants a hundred, so walking the chain
// honestly is three times the base pass again, and it buys a step at every ring
// where averaging leaves none.
fn level_lift(cell: vec2<i32>) -> f32 {
    if (resident_height_at(job.level, cell) < NODATA_BELOW) {
        return 0.0;
    }
    if (job.level == terrain.resident_base) {
        // The survey's own id, unwarped: this texel *is* the resolution the
        // cover was stored at, so there is nothing here to upscale.
        return standing_at(cover_at(cell, job.level), cell, job.level).lift;
    }
    // Holes are dropped from the mean rather than averaged in, exactly as the
    // heights under them were when the tools built this level: a texel half
    // outside the survey carries what the measured half of it carries, not half
    // of it.
    var sum = 0.0;
    var found = 0.0;
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let child = cell * 2 + vec2<i32>(dx, dy);
            if (resident_height_at(job.level - 1u, child) >= NODATA_BELOW) {
                sum += lift_at(job.level - 1u, child);
                found += 1.0;
            }
        }
    }
    if (found <= 0.0) {
        return 0.0;
    }
    return sum / found;
}

// What stands on one texel of any level, generated or resident.
//
// One function for both, which is what stops the near field and the far field
// growing two different forests. The stand is a function of the cover and of
// the slope of the hillside, and both are answered at the base or above it, so
// a run of texels under one cover is handed one stand -- the same density, the
// same health, and therefore the same trunks off the same lattice. What differs
// between the levels is only how finely that stand is sampled, which is the
// difference the order statistic is built to be indifferent to.
//
// The cover comes in rather than being read here, because a generated level
// does not read it where a resident one does: it asks [`cover_upscaled`] and
// gets a boundary that wanders inside the base texel. The trees have to follow
// that boundary and not the block -- what is grown and what is painted are
// written off one walk precisely so a crown is never drawn on ground painted as
// meadow -- and handing the cover in is what keeps the two answering to the
// same id.
//
// The slope comes from the mip above the base rather than from a texel's own
// neighbours. `footing` asks how steep the *hillside* is, which is a question
// about a couple of hundred metres of ground, and a single pair of eight-metre
// texels answers it more noisily than the thing being asked about. Taking it
// from a level nothing writes also keeps this pass off ground it is raising.
fn standing_at(cover: u32, cell: vec2<i32>, level: u32) -> Standing {
    let base = terrain.resident_base;
    let parent = cell >> vec2<u32>(base - level);
    let above = min(base + 1u, terrain.level_count - 1u);
    let coarse = parent >> vec2<u32>(above - base);
    let metres = f32(1u << above) * terrain.metres_per_texel.x;
    // The survey's own ground, which is what a hillside's steepness is a fact
    // about: a stand thins on a slope because the soil does, and the slope of
    // the canopy over it says nothing about that.
    //
    // Written as a subtraction rather than as a read of an unraised level,
    // because there is no longer such a level -- and it is the same subtraction
    // whether this runs while the chain is being built or while a tile is being
    // generated. The pass that walks this reaches the base before the level
    // above it, so the lift it reads there is still the zero a texture starts
    // at, and the heights are still the survey's; afterwards both have been
    // written and the difference is the same number.
    let fall = vec2<f32>(
        bare_above(above, coarse + vec2<i32>(1, 0)) - bare_above(above, coarse - vec2<i32>(1, 0)),
        bare_above(above, coarse + vec2<i32>(0, 1)) - bare_above(above, coarse - vec2<i32>(0, 1)),
    );
    let slope = length(fall / (2.0 * metres));
    let texel = f32(1u << level) * terrain.metres_per_texel.x;
    return standing_on(cover, cover_centre(cell, level), texel, slope);
}

// The smallest half float that is not below `height`, as bits.
//
// A transcription of `ceiling_half` in `src/terrain/maxima.rs`, and it has to
// stay one: the pyramid is stored at half precision, and a cell rounded the
// wrong way is a ceiling below the ground it bounds -- a ridge with a hole
// through it, which is the one failure this whole structure exists to prevent.
//
// Written as convert-then-correct rather than as a conversion trusted to round
// the right way. `pack2x16float` is round-to-nearest on most backends and
// towards zero on some, and neither is towards positive infinity. No rounding
// mode is out by more than one representable step, so one step always closes it.
fn ceiling_half(height: f32) -> u32 {
    let bits = pack2x16float(vec2<f32>(height, 0.0)) & 0xffffu;
    if (unpack2x16float(bits).x >= height) {
        return bits;
    }
    // One step away from zero within a sign, which is one step towards positive
    // infinity. Negative zero cannot arrive here: anything that rounds to it
    // came from a value at or below zero, which it is therefore not below.
    if ((bits & 0x8000u) != 0u) {
        return bits - 1u;
    }
    return bits + 1u;
}

// The ceiling one cell of `job.level` holds.
fn derived_ceiling(cell: vec2<i32>) -> f32 {
    let level = job.level;
    // This level's own samples over the cell's *closed* square: the four around
    // it, reaching one sample past it, because the bilinear patch the march
    // solves against here is fed by the neighbour. Nodata needs no case of its
    // own -- the sentinel is far below any ground, so a maximum ignores a hole
    // beside real ground and a cell of nothing but holes stays at the sentinel.
    var top = height_at(level, cell);
    top = max(top, height_at(level, cell + vec2<i32>(1, 0)));
    top = max(top, height_at(level, cell + vec2<i32>(0, 1)));
    top = max(top, height_at(level, cell + vec2<i32>(1, 1)));
    if (job.carry == 0u) {
        return top;
    }

    // Everything finer, carried up. Two adjacent closed squares share their
    // boundary, so the four cells under this one cover its whole square between
    // them and nothing has to be widened.
    //
    // Tested per child rather than for the group, because the square's edge can
    // fall between them: a ray may descend into one child of a cell and not its
    // neighbour, and the one it can reach is the one that has to be bounded.
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let child = cell * 2 + vec2<i32>(dx, dy);
            if (all(child >= job.below_low) && all(child < job.below_high)) {
                top = max(top, ceiling_at(level - 1u, child));
            }
        }
    }
    return top;
}

// One thread per pair of cells across, because the smallest thing a storage
// buffer can be written in is four bytes and a cell is two.
@compute @workgroup_size(8, 8)
fn cs_maxima(@builtin(global_invocation_id) id: vec3<u32>) {
    let pairs = (job.size.x + 1u) / 2u;
    if (id.x >= pairs || id.y >= job.size.y) {
        return;
    }
    let at = job.origin + vec2<i32>(i32(id.x) * 2, i32(id.y));
    let low = ceiling_half(derived_ceiling(at));
    // An odd-width rectangle leaves the last pair half empty. Those two bytes
    // are in the buffer but not in the copy out of it, so they are filled by
    // repeating rather than by reading a cell the rectangle does not cover.
    var high = low;
    if (id.x * 2u + 1u < job.size.x) {
        high = ceiling_half(derived_ceiling(at + vec2<i32>(1, 0)));
    }
    out_halves[id.y * job.stride + id.x] = low | (high << 16u);
}

// One thread per pixel the reprojection could not answer.
@compute @workgroup_size(64)
fn cs_march(@builtin(global_invocation_id) id: vec3<u32>) {
    // The list is laid out in rows of `MARCH_ROW` workgroups; see the constant
    // for why it is not one long line. A grid one row deep has `y` of zero
    // throughout and this is `id.x`, so the short case costs nothing.
    let at = id.y * MARCH_ROW * MARCH_GROUP + id.x;
    // The dispatch is whole workgroups and whole rows, so the end runs past the
    // list.
    if (at >= atomicLoad(&tally.holes)) {
        return;
    }
    let packed = holes[at];
    let pixel = vec2<u32>(packed & 0xffffu, packed >> 16u);
    store(pixel, ground_at(pixel));
    // Diagnostics, and only paid for by the rays that failed: on a healthy
    // frame almost nothing reaches either of these.
    if (ray_abandoned) {
        atomicAdd(&tally.abandoned, 1u);
    }
    if (ray_spent) {
        atomicAdd(&tally.spent, 1u);
    }
    atomicAdd(&tally.wrote, 1u);
}
