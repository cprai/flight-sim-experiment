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
    return clamp(cell & info.mask, info.valid_low, info.valid_high - vec2<i32>(1));
}

// Which mip of the resident chain a level is.
fn mip(level: u32) -> i32 {
    return i32(level - terrain.resident_base);
}

fn height_at(level: u32, cell: vec2<i32>) -> f32 {
    return textureLoad(heights, slot(level, cell), mip(level)).r;
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

        let ceiling = textureLoad(maxima, slot(level, cell), mip(level)).r;
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
    // One load, and no question about what the ray met -- which there could not
    // be, because a crown baked into the heights is indistinguishable from a
    // hillock and the march has nothing left to ask. A treetop is labelled as
    // one already: `terrain-generate` writes a canopy id into this product
    // wherever the crowns cover enough of a texel, so the gaps between the
    // trees keep the floor's own colour and the trees do not.
    out.material = textureLoad(materials, slot(hit.level, cell), mip(hit.level)).r;
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
    // `u32`s per row of `out_maxima`, which is the byte stride the copy out of
    // it uses divided by four.
    stride: u32,
    padding: u32,
};

@group(3) @binding(12) var<uniform> job: MaximaJob;
// Cells go out through a buffer rather than a storage texture because `r16float`
// is not a storage format in WebGPU. A buffer costs one copy afterwards and
// needs no format widened, no feature asked for and no render pipeline in a
// file that has none.
@group(3) @binding(13) var<storage, read_write> out_maxima: array<u32>;

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
    let below = mip(level - 1u);
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let child = cell * 2 + vec2<i32>(dx, dy);
            if (all(child >= job.below_low) && all(child < job.below_high)) {
                top = max(top, textureLoad(maxima, slot(level - 1u, child), below).r);
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
    out_maxima[id.y * job.stride + id.x] = low | (high << 16u);
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
