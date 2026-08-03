// Terrain drawn entirely by raymarching a max pyramid.
//
// There is no mesh. Every pixel of ground is found by walking a ray through a
// quadtree of maximum heights: skip a texel whenever the ray stays above the
// ceiling it holds, climb a level after every skip, descend a level when a
// texel might be hit, and at the finest level resident there solve against the
// surface itself. Tevs, Ihrke and Seidel, "Maximum Mipmaps for Fast, Accurate,
// and Scalable Dynamic Height Field Rendering", I3D 2008.
//
// The level array *is* the quadtree. Level `l` holds one ceiling per level-`l`
// texel, bounding every surface the renderer might draw across the closed
// square that texel covers, so climbing the quadtree is reading the next level
// out. Nothing carries a mip chain of its own. See
// `crates/terrain-tiles/src/maxima.rs` for what a cell means and why the bound
// has to hold for coarse levels as well as fine.
//
// Which level a point can be read at is decided by residency alone, and
// residency is decided by distance from the camera -- a level holds a square of
// whole tiles around it. That is the level of detail, and it needs no rule of
// its own: a ray far from the camera simply finds nothing finer resident.

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

struct Level {
    // The texels resident at this level, as a half-open range measured in this
    // level's own texels from the raster's origin. A point outside it has to be
    // read at a coarser level, which covers twice the ground from the same
    // number of tiles and is therefore resident wherever this one is.
    //
    // One texel short of the tiles actually loaded on the high side, because
    // the bilinear patch at the last texel reads its neighbour.
    valid_low: vec2<i32>,
    valid_high: vec2<i32>,
    // The highest ground anywhere resident at this level, taken across the
    // tiles themselves. A ray above it and climbing has nothing to find here.
    ceiling: f32,
    padding: f32,
    more_padding: vec2<f32>,
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
    // The finest level being kept. Below it nothing is loaded, because its
    // texels would be smaller than the pixels they land in.
    base_level: u32,
    // A level's texture is a power of two square, so wrapping a texel index
    // onto its slot is an AND with this.
    texel_mask: u32,
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
@group(1) @binding(1) var heights: texture_2d_array<f32>;
@group(1) @binding(2) var materials: texture_2d_array<u32>;
@group(1) @binding(3) var maxima: texture_2d_array<f32>;
// The east and south components of the ground's unit normal, signed and
// normalised by the texture format. See `Normal` in
// `crates/terrain-tiles/src/texel.rs` for why the third one is not stored.
@group(1) @binding(4) var normals: texture_2d_array<f32>;

// Elevations below this are the raster's nodata rather than ground.
//
// HRDEM writes -32767. The exact value is not worth passing in: the deepest
// ground on Earth is a small fraction of this, so anything below it is a hole
// however the producer chose to spell it. Kept in step with `NODATA_BELOW` in
// `crates/terrain-tiles/src/texel.rs`, which is where the filter that drops
// these texels when it builds a coarse level reads the same threshold.
const NODATA_BELOW: f32 = -30000.0;

// The finest level the normals are stored at.
//
// Requests for anything finer are served by repeating texels, so this is where
// they stop being distinct and interpolating below it would only ramp between
// copies of one value. Kept in step with `NORMAL_BASE_LEVEL` in
// `crates/terrain-tiles/src/manifest.rs`.
const NORMAL_BASE_LEVEL: u32 = 3u;

// Halvings used to place the intercept once the texel holding it is known.
// Eight takes a texel to a two-hundred-and-fiftieth of its width, far finer
// than the pixel that asked.
const REFINE_STEPS: u32 = 8u;

// Stand-in for an infinite distance, in a form arithmetic can still be done on.
const NEVER: f32 = 1e30;

// A height and the worst of the texels that went into it.
struct Sample {
    height: f32,
    // The lowest texel sampled. Interpolating first would bury a hole: three
    // real metres averaged with one -32767 comes out around -7800, which is far
    // below any ground but nowhere near the nodata value, so a test on the
    // result alone would let it through.
    lowest: f32,
}

// Whether a texel of a level is loaded and safe to read.
fn resident(level: u32, cell: vec2<i32>) -> bool {
    let info = terrain.levels[level];
    return all(cell >= info.valid_low) && all(cell < info.valid_high);
}

// The slot a texel index lands in.
//
// A level's square is a power-of-two number of tiles and a tile is a power of
// two texels, so this is a mask rather than a modulo -- and it depends on
// nothing but the index, because a tile's slot does not move when the square
// does. Negative indices wrap correctly: a two's complement AND is exactly the
// non-negative remainder.
fn slot(cell: vec2<i32>) -> vec2<i32> {
    return cell & vec2<i32>(i32(terrain.texel_mask));
}

fn height_at(level: u32, cell: vec2<i32>) -> f32 {
    return textureLoad(heights, slot(cell), i32(level), 0).r;
}

// Height at a fractional position in a level's own texels.
fn height_bilinear(level: u32, w: vec2<f32>) -> Sample {
    let base = vec2<i32>(floor(w));
    let f = fract(w);
    let a = height_at(level, base);
    let b = height_at(level, base + vec2<i32>(1, 0));
    let c = height_at(level, base + vec2<i32>(0, 1));
    let d = height_at(level, base + vec2<i32>(1, 1));
    let top = mix(a, b, f.x);
    let bottom = mix(c, d, f.x);

    var sample: Sample;
    sample.height = mix(top, bottom, f.y);
    sample.lowest = min(min(a, b), min(c, d));
    return sample;
}

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
// has to be rebased when a ray changes level, which is the whole reason the
// resident squares are anchored to the raster rather than to the camera.
fn march(eye: vec3<f32>, dir: vec3<f32>) -> Hit {
    var out: Hit;
    out.found = false;

    let coarsest = terrain.level_count - 1u;
    let p0 = (eye.xz - terrain.origin) / terrain.metres_per_texel;
    let d0 = dir.xz / terrain.metres_per_texel;
    let speed = max(max(abs(d0.x), abs(d0.y)), 1.0 / NEVER);
    // Past a wall far enough to be past it, so the next step lands in the next
    // texel rather than back in this one. `wall_nudge` is a distance along the
    // dominant axis; dividing by `speed` turns it into a distance along the ray.
    let nudge = terrain.wall_nudge / speed;
    // A ray that has crossed the coarsest square twice has left it, or is going
    // straight up or down and never will.
    let limit = 2.0 * f32(terrain.texel_mask + 1u) * f32(1u << coarsest) / speed;

    var t = 0.0;
    // Start at the coarsest level and descend, which is the shape a maximum
    // mipmap traversal is usually written in. Starting at the finest and
    // climbing measured the same on this raster, so this is the conventional
    // form rather than a measured improvement over it.
    var level = coarsest;
    // Whether the ray has advanced at all. Being under the ground before it has
    // means something quite different from being under it after.
    var moved = false;
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

        // Nothing loaded here at this level: the ray has left this square, so
        // hand it out to the level beyond, which covers twice the ground.
        if (!resident(level, cell)) {
            if (level >= coarsest) {
                return out;
            }
            level += 1u;
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

        let ceiling = textureLoad(maxima, slot(cell), i32(level), 0).r;
        // The texel bounds its whole closed square, so a ray above that ceiling
        // at both ends of the segment is above it throughout.
        if (min(eye.y + dir.y * t, eye.y + dir.y * exit) > ceiling) {
            t = exit + nudge;
            moved = true;
            level = min(level + 1u, coarsest);
            continue;
        }

        // Something could be here. Look closer, if anything finer is loaded --
        // which is to say, if this ground is near enough the camera to have it.
        if (level > terrain.base_level) {
            let finer = level - 1u;
            if (resident(finer, vec2<i32>(floor(p / (size * 0.5))))) {
                level = finer;
                continue;
            }
        }

        // The finest level there is here, so this texel is the leaf and the
        // ground across it is the bilinear patch through its four corners.
        let w = p / size;
        let enter = height_bilinear(level, w);
        let leave = height_bilinear(level, (p0 + d0 * exit) / size);

        // Ground nothing is known about is not ground.
        if (min(enter.lowest, leave.lowest) < NODATA_BELOW) {
            hole = true;
            t = exit + nudge;
            moved = true;
            level = min(level + 1u, coarsest);
            continue;
        }

        if (eye.y + dir.y * t <= enter.height) {
            if (!moved || hole) {
                // Either the ray began below the surface -- where it went in is
                // behind the eye and cannot be found from here -- or it has just
                // dropped through a hole and this is the underside of the ground
                // beside it. Neither is a hit.
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

        if (eye.y + dir.y * exit > leave.height) {
            // The ceiling allowed a hit somewhere in the texel; the surface
            // itself does not reach the ray.
            t = exit + nudge;
            moved = true;
            level = min(level + 1u, coarsest);
            continue;
        }

        // Above at one end and below at the other, so the crossing is bracketed.
        var above = t;
        var below = exit;
        for (var i = 0u; i < REFINE_STEPS; i += 1u) {
            let middle = 0.5 * (above + below);
            let ground = height_bilinear(level, (p0 + d0 * middle) / size).height;
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
    if (moved && !hole) {
        out.found = true;
        out.position = eye + dir * t;
        out.level = level;
        out.w = (p0 + d0 * t) / f32(1u << level);
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
@group(2) @binding(0) var out_material: texture_storage_2d<r32uint, write>;
// Where the ray met the ground, in world space; `w` is 1 to say so.
@group(2) @binding(1) var out_position: texture_storage_2d<rgba32float, write>;
// The unit normal of that ground, in world space.
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
// `holes` stays first so its offset is unchanged; the members after it are
// mirrored by `Coverage` in `src/reproject.rs`.
struct Tally {
    holes: atomic<u32>,
    carried: atomic<u32>,
    sky: atomic<u32>,
};

@group(3) @binding(5) var<storage, read_write> tally: Tally;
// `[workgroups, 1, 1]`, written by `cs_args` for the march to be dispatched by.
@group(3) @binding(6) var<storage, read_write> march_args: array<u32, 3>;

// The finished motion field, and the one number per dither cell that `cs_risk`
// reduces it to. Bound only for that pass, which writes neither of the two
// above and so cannot be given the same layout as the march.
@group(3) @binding(7) var motion: texture_2d<u32>;
@group(3) @binding(8) var out_risk: texture_storage_2d<r32float, write>;

// Whether a stored pair is a direction at all.
//
// Both components of a unit normal fit inside the unit disc, so the sentinel
// the tools write for unmeasured ground -- the most negative pair the format
// holds -- is the one value that cannot be mistaken for one.
fn measured(stored: vec2<f32>) -> f32 {
    return select(0.0, 1.0, dot(stored, stored) <= 1.0);
}

// The ground's unit normal at a fractional position, in world space.
//
// Bilinear over the four surrounding texels rather than the nearest of them.
// Nearest is flat shading: every texel is one constant normal, so the ground
// breaks into facets that the eye reads as blocks however smooth the surface
// under them is. Interpolating turns the same data into a normal that varies
// continuously across a texel, which is the smooth-shading half of what a
// stored normal is for.
//
// The two stored components are what gets mixed, and the third is rebuilt from
// them afterwards. A height field's normal always points upwards, so the
// vertical component is whatever is left of unit length -- and a mean of pairs
// inside the unit disc is inside it too, so the result comes out unit without
// a normalize. Mixing three components and renormalising would be the same
// thing with a step added.
//
// Nodata is not a direction and must not be averaged into one: those corners
// are dropped and the weight redistributed over the rest, so ground beside a
// hole takes the normal of the ground that was measured. Flat is the answer
// where nothing was.
fn normal_bilinear(level: u32, w: vec2<f32>) -> vec3<f32> {
    let base = vec2<i32>(floor(w));
    let f = fract(w);
    let a = textureLoad(normals, slot(base), i32(level), 0).rg;
    let b = textureLoad(normals, slot(base + vec2<i32>(1, 0)), i32(level), 0).rg;
    let c = textureLoad(normals, slot(base + vec2<i32>(0, 1)), i32(level), 0).rg;
    let d = textureLoad(normals, slot(base + vec2<i32>(1, 1)), i32(level), 0).rg;

    let corner = vec4<f32>(
        (1.0 - f.x) * (1.0 - f.y) * measured(a),
        f.x * (1.0 - f.y) * measured(b),
        (1.0 - f.x) * f.y * measured(c),
        f.x * f.y * measured(d),
    );
    let total = corner.x + corner.y + corner.z + corner.w;
    if (total <= 0.0) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }

    let mean = (a * corner.x + b * corner.y + c * corner.z + d * corner.w) / total;
    return vec3<f32>(mean.r, sqrt(max(1.0 - dot(mean, mean), 0.0)), mean.g);
}

// What `position.w` says about a pixel of the G-buffer.
//
// Three states rather than two, because the reprojection has to tell a pixel
// whose ray found nothing from a pixel nothing has ever been written to. Sky is
// worth carrying between frames -- it is most of a horizon view and none of it
// can be carried by world position, because it has none -- but a buffer that
// has only just been allocated must carry nothing at all. Zero is what wgpu
// leaves a fresh texture as, so zero is the one that means "not yet".
const GROUND_HERE: f32 = 1.0;
const SKY_HERE: f32 = -1.0;

// The carried material word is a material id and the sub-pixel position the
// point landed at, packed together.
//
// Ids reach 0x080c and so need sixteen bits of the thirty-two; the rest were
// spare, and the offset rides in them for nothing -- no extra target, no extra
// export bandwidth. Eight bits an axis puts the point within a 256th of a pixel
// of where it really is. Must match `pack_offset` in `src/reproject.wgsl`.
const MATERIAL_MASK: u32 = 0xffffu;
const OFFSET_SCALE: f32 = 256.0;

fn unpack_offset(packed: u32) -> vec2<f32> {
    return vec2<f32>(
        f32((packed >> 16u) & 0xffu),
        f32((packed >> 24u) & 0xffu),
    ) / OFFSET_SCALE;
}

// What one ray found, in the form the G-buffer stores it.
//
// Material zero is `Null` and depth zero is the reversed-Z far plane, which no
// finite hit can produce, so those two are what the shading pass reads as sky.
struct Ground {
    material: u32,
    position: vec4<f32>,
    normal: vec4<f32>,
    depth: f32,
};

fn nothing() -> Ground {
    return Ground(0u, vec4<f32>(0.0, 0.0, 0.0, SKY_HERE), vec4<f32>(0.0), 0.0);
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
    if (ground.position.w != GROUND_HERE) {
        return vec2<f32>(0.0);
    }
    let now = vec2<f32>(pixel) + 0.5;
    let was = was_at(ground.position.xyz);
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
        return nothing();
    }

    let hit = march(eye, dir);
    if (!hit.found) {
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
    out.material = textureLoad(materials, slot(cell), i32(hit.level), 0).r;
    out.position = vec4<f32>(hit.position, GROUND_HERE);
    // Read the normals where they are still distinct rather than at the hit's
    // own level: they are stored no finer than level 3 and the store serves
    // finer requests by repeating texels, so interpolating below it would ramp
    // between copies of one value and leave the eight-metre grid on show.
    //
    // Never a square the clipmap does not hold. The level asked for is at least
    // the hit's, which was resident, and a coarser level's window covers twice
    // the ground of the next finer one from the same camera. The min is for a
    // raster with no level 3 at all, which only a test builds.
    let normal_level = min(max(hit.level, NORMAL_BASE_LEVEL), terrain.level_count - 1u);
    let normal_w = hit.w * exp2(f32(hit.level) - f32(normal_level));
    // Still not the gradient of the bilinear patch the ray actually
    // intersected: that patch is a smoothing of the ground, and it flattens as
    // the level coarsens, where this carries the mean of the finest normals
    // there are. The far field keeps its relief at the cost of shading and
    // silhouette parting company a little.
    out.normal = vec4<f32>(normal_bilinear(normal_level, normal_w), 0.0);
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
    let landed = vec2<f32>(pixel) + unpack_offset(packed);
    out.position = vec4<f32>(
        camera.position.xyz + ray_raw_at(landed) * distance_at(depth),
        GROUND_HERE,
    );
    out.normal = normal;
    out.depth = depth;
    return out;
}

// Writes one pixel's worth of ground into the G-buffer.
fn store(pixel: vec2<u32>, ground: Ground) {
    let at = vec2<i32>(pixel);
    textureStore(out_material, at, vec4<u32>(ground.material, 0u, 0u, 0u));
    textureStore(out_position, at, ground.position);
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
    let eye = camera.position.xyz;
    if (ray_through(pixel).y >= 0.0 && eye.y >= terrain.ceiling) {
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
    march_args[0] = (atomicLoad(&tally.holes) + MARCH_GROUP - 1u) / MARCH_GROUP;
    march_args[1] = 1u;
    march_args[2] = 1u;
}

// How fast the picture is moving across one dither cell.
//
// The largest screen-space motion any pixel of the cell has, which is the
// signal the drop pattern spends its extra rays on: ground that is sweeping
// across the view is ground whose carried material and normal go stale
// soonest, and under forward flight it is also the near ground that magnifies
// fastest and so leaves the most gaps between splats.
//
// The maximum rather than the mean, because a cell is dropped or kept whole and
// the worst pixel in it is what decides whether keeping it shows.
//
// One workgroup per cell, which is why the workgroup *is* the cell: the
// reduction is over exactly the pixels whose fate the answer decides.
var<workgroup> worst: array<f32, 64>;

@compute @workgroup_size(8, 8)
fn cs_risk(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) cell: vec3<u32>,
    @builtin(local_invocation_index) slot: u32,
) {
    // Pixels past the edge take the identity of the reduction rather than
    // dropping out, so every lane has something defined to contribute.
    var here = 0.0;
    if (all(id.xy < terrain.viewport)) {
        let motion = unpack2x16float(textureLoad(motion, vec2<i32>(id.xy), 0).r);
        here = max(abs(motion.x), abs(motion.y));
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
        textureStore(out_risk, vec2<i32>(cell.xy), vec4<f32>(worst[0], 0.0, 0.0, 0.0));
    }
}

// One thread per pixel the reprojection could not answer.
@compute @workgroup_size(64)
fn cs_march(@builtin(global_invocation_id) id: vec3<u32>) {
    // The dispatch is whole workgroups, so the last one runs past the list.
    if (id.x >= atomicLoad(&tally.holes)) {
        return;
    }
    let packed = holes[id.x];
    let pixel = vec2<u32>(packed & 0xffffu, packed >> 16u);
    store(pixel, ground_at(pixel));
}
