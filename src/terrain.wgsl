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
    more_padding: vec2<f32>,
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

struct ScreenOut {
    @builtin(position) clip: vec4<f32>,
    // Normalized device coordinates, which is what the camera's ray basis wants.
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_terrain(@builtin(vertex_index) index: u32) -> ScreenOut {
    // One oversized triangle rather than two: no diagonal seam down the middle
    // of the screen, and the quads either side of it are not rasterized twice.
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let ndc = corner * 2.0 - 1.0;

    var out: ScreenOut;
    out.clip = vec4<f32>(ndc, 1.0, 1.0);
    out.ndc = ndc;
    return out;
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

// One pixel of the G-buffer the shading pass reads. A pixel this shader
// discards keeps the cleared values -- material zero, depth zero -- and depth
// zero is how the shading pass knows a pixel is sky.
struct GBufferOut {
    @location(0) material: u32,
    // Where the ray met the ground, in world space; `w` is 1 to say so.
    @location(1) position: vec4<f32>,
    // The unit normal of that ground, in world space.
    @location(2) normal: vec4<f32>,
    // Written rather than interpolated, so the ground takes its place in the
    // depth buffer at the distance it was actually found at.
    @builtin(frag_depth) depth: f32,
};

// The stored normal of a texel, rebuilt into a world-space unit vector.
//
// Only two components are stored, and a height field's normal always points
// upwards, so the third is what is left of unit length. The pair that no real
// normal can reach means no elevation was measured here; flat is the answer
// that cannot mislead, and the ray did not hit anything to shade in any case.
fn normal_at(level: u32, cell: vec2<i32>) -> vec3<f32> {
    let stored = textureLoad(normals, slot(cell), i32(level), 0).rg;
    let flat = dot(stored, stored);
    if (flat > 1.0) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec3<f32>(stored.r, sqrt(1.0 - flat), stored.g);
}

@fragment
fn fs_terrain(in: ScreenOut) -> GBufferOut {
    var out: GBufferOut;
    out.material = 0u;
    out.position = vec4<f32>(0.0);
    out.normal = vec4<f32>(0.0);
    out.depth = 0.0;

    let eye = camera.position.xyz;
    let dir = normalize(
        camera.ray_right.xyz * in.ndc.x + camera.ray_up.xyz * in.ndc.y + camera.ray_forward.xyz,
    );

    // A ray already above the highest ground anywhere resident, and still
    // climbing, is never coming back down. Worth one comparison: at any horizon
    // view most of the frame is sky.
    if (dir.y >= 0.0 && eye.y >= terrain.ceiling) {
        discard;
        return out;
    }

    let hit = march(eye, dir);
    if (!hit.found) {
        discard;
        return out;
    }

    // Squares reach past the raster and reads out there wrap onto whatever
    // shares their slot, so the ground beyond the last real sample is not
    // ground. A straight ray leaves the data once and never comes back, so
    // there is nothing further along worth carrying on for.
    if (any(hit.position.xz < terrain.data_min) || any(hit.position.xz > terrain.data_max)) {
        discard;
        return out;
    }

    let clip = camera.view_proj * vec4<f32>(hit.position, 1.0);
    out.depth = clip.z / clip.w;
    // The nearest texel to the hit: sample centres sit at integer `w`, the
    // convention the height bilinear reads by, so the texel whose centre is
    // closest is the rounded index.
    let cell = vec2<i32>(floor(hit.w + 0.5));
    out.material = textureLoad(materials, slot(cell), i32(hit.level), 0).r;
    out.position = vec4<f32>(hit.position, 1.0);
    // From the same texel the material came from. It is not the gradient of the
    // bilinear patch the ray actually intersected: that patch is a smoothing of
    // the ground, and it flattens as the level coarsens, where this carries the
    // mean of the finest normals there are. The far field keeps its relief at
    // the cost of shading and silhouette parting company a little.
    out.normal = vec4<f32>(normal_at(hit.level, cell), 0.0);
    return out;
}
