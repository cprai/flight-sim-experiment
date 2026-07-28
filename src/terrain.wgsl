// Geometry clipmap terrain.
//
// Every vertex arrives knowing only its integer position within a patch of a
// regular grid. Where that lands in the world, and how high it sits, comes from
// the level's transform and the clipmap's height texture, so one small vertex
// buffer serves every patch of every level.

struct Camera {
    view_proj: mat4x4<f32>,
    // World-space eye position. `w` is padding.
    position: vec4<f32>,
    // A pixel at normalized device coordinates (x, y) looks along
    // `x * ray_right + y * ray_up + ray_forward`. `w` is padding on each.
    ray_right: vec4<f32>,
    ray_up: vec4<f32>,
    ray_forward: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

// Sized generously; only `level_count` entries are ever read. The whole block
// is a few hundred bytes, far inside any uniform buffer limit.
const MAX_LEVELS: u32 = 16u;

struct Level {
    // World XZ of the vertex at the window's origin.
    origin: vec2<f32>,
    // World metres per texel, which differs per axis when a geographic raster
    // has been squared up into metres.
    spacing: vec2<f32>,
    // Where the window's origin currently sits in the texture. Windows are
    // addressed toroidally so that moving the camera costs an edge strip of
    // uploads rather than a full recopy.
    torus: vec2<u32>,
    // Where this window's origin lands in the next coarser window, so a vertex
    // can find its own position on the level it blends towards.
    coarse_offset: vec2<f32>,
};

struct Terrain {
    levels: array<Level, MAX_LEVELS>,
    level_count: u32,
    // Window size is a power of two, so wrapping is an AND with this.
    window_mask: u32,
    morph_band: f32,
    // Side length of a level's grid in quads, as a float for the morph maths.
    grid_quads: f32,
    // World XZ of the outermost samples the raster actually holds. Rings reach
    // past this, and everything they cover out there is invented by clamping to
    // the border, so it is cut away rather than drawn.
    data_min: vec2<f32>,
    data_max: vec2<f32>,
    // The finest level being drawn. Levels below it are dropped as the camera
    // climbs away from the ground, so this is the innermost one and the one
    // carrying the solid centre.
    base_level: u32,
    // How far the base level has been blended into the level outside it, on top
    // of whatever its own position in the ring asks for. Rises to one as the
    // camera climbs towards the altitude at which the base level is dropped, so
    // that by the time it goes it is already drawing the coarser surface and its
    // disappearance cannot be seen.
    base_morph: f32,
    // How far out geometry is rasterized, in metres. Ground further away than
    // this is raymarched instead, so the raster stage cuts its fragments here
    // and the far stage starts its rays here. The two must not overlap: they
    // describe the same surface a little differently, and both drawing it would
    // fight over the depth buffer.
    near_radius: f32,
    // Coarsest mip of the max pyramid, where one texel covers a whole window.
    max_mip: u32,
};

@group(1) @binding(0) var<uniform> terrain: Terrain;
// Non-filterable on purpose: heights are only ever fetched at exact texel
// centres, never sampled, so no float-filtering support is required.
@group(1) @binding(1) var heights: texture_2d_array<f32>;
@group(1) @binding(2) var colours: texture_2d_array<f32>;
@group(1) @binding(3) var colour_sampler: sampler;

struct VertexIn {
    // Position within the shared grid, in quads.
    @location(0) grid: vec2<u32>,
};

struct InstanceIn {
    // Where this patch starts within its level's grid.
    @location(1) origin: vec2<u32>,
    // Clipmap level, which is also the texture array layer.
    @location(2) level: u32,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) colour_uv: vec2<f32>,
    @location(1) coarse_uv: vec2<f32>,
    @location(2) @interpolate(flat) level: u32,
    @location(3) @interpolate(flat) coarse_level: u32,
    // How far this vertex has been blended into the coarser level.
    @location(4) morph: f32,
    // World position, carried so the fragment stage can tell whether it is
    // standing on real data and how far from the eye it is.
    @location(5) world: vec3<f32>,
    // Non-zero if any texel this vertex was built from is a hole.
    //
    // Interpolated rather than flat, so it is non-zero across the whole of any
    // triangle touching a hole and the fragment stage cuts the lot. Cutting the
    // whole triangle is the conservative reading -- part of it does cover ground
    // nothing is known about -- and it matches how the data's outer edge is
    // already handled.
    @location(6) nodata: f32,
};

// Elevations below this are the raster's nodata rather than ground.
//
// HRDEM writes -32767. The exact value is not worth passing in: the deepest
// ground on Earth is a small fraction of this, so anything below it is a hole
// however the producer chose to spell it. Kept in step with `NODATA_BELOW` in
// `crates/terrain-tiles/src/texel.rs`, which is where the filter that drops
// these texels when it builds a coarse level reads the same threshold.
const NODATA_BELOW: f32 = -30000.0;

// A height and the worst of the texels that went into it.
struct Sample {
    height: f32,
    // The lowest texel sampled. Interpolating first would bury a hole: three
    // real metres averaged with one -32767 comes out around -7800, which is far
    // below any ground but nowhere near the nodata value, so a test on the
    // result alone would let it through.
    lowest: f32,
}

// Wraps a window coordinate onto the texel that currently holds it.
fn window_texel(level: u32, w: vec2<i32>) -> vec2<i32> {
    let mask = vec2<i32>(i32(terrain.window_mask));
    return (vec2<i32>(terrain.levels[level].torus) + w) & mask;
}

fn height_at(level: u32, w: vec2<i32>) -> f32 {
    return textureLoad(heights, window_texel(level, w), i32(level), 0).r;
}

// Height at a fractional window position.
//
// Every vertex that morphs lands on a multiple of half a coarse texel, so the
// interpolation weights are only ever 0 or 1/2 and this is exact rather than an
// approximation. On the outer boundary -- where the blend reaches the coarser
// level completely -- one weight is always 0, so the result lies on the coarse
// level's own edge and the two surfaces meet with no crack.
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

// How far into the coarser level a vertex has blended: 0 across most of a ring,
// rising to 1 exactly at its outer edge.
//
// Measured with a square distance from the window's centre rather than a radial
// one so that the band follows the ring's own shape.
fn morph_factor(w: vec2<f32>) -> f32 {
    let centre = terrain.grid_quads * 0.5;
    let from_centre = abs(w - centre) / centre;
    let edge = max(from_centre.x, from_centre.y);
    return clamp(
        (edge - (1.0 - terrain.morph_band)) / terrain.morph_band,
        0.0,
        1.0,
    );
}

// Texture coordinates for a window position. The sampler repeats, so the half
// of the window past the seam wraps around on its own and bilinear taps stay
// correct across it.
fn colour_uv(level: u32, w: vec2<f32>) -> vec2<f32> {
    let size = f32(terrain.window_mask + 1u);
    return (vec2<f32>(terrain.levels[level].torus) + w + 0.5) / size;
}

@vertex
fn vs_main(vertex: VertexIn, instance: InstanceIn) -> VertexOut {
    let level = instance.level;
    let info = terrain.levels[level];

    let w = vec2<i32>(vertex.grid + instance.origin);
    let wf = vec2<f32>(w);
    let ground = info.origin + wf * info.spacing;

    // Blend the outer band of every ring into the level outside it. This is
    // what removes both the cracks -- a ring's edge vertices end up exactly on
    // the coarser surface, so the two meet -- and the pop that would otherwise
    // happen as a vertex crossed from one level to the next.
    let coarse_level = min(level + 1u, terrain.level_count - 1u);
    var morph = morph_factor(wf);
    if (level == terrain.base_level) {
        // Whichever is further along: the ring's own outward blend, or the
        // altitude blend that is retiring this level altogether.
        morph = max(morph, terrain.base_morph);
    }
    if (coarse_level == level) {
        // The outermost ring has nothing to blend towards.
        morph = 0.0;
    }

    var height = height_at(level, w);
    var lowest = height;
    var coarse_uv = vec2<f32>(0.0);
    if (morph > 0.0) {
        let coarse_w = info.coarse_offset + wf * 0.5;
        let coarse = height_bilinear(coarse_level, coarse_w);
        height = mix(height, coarse.height, morph);
        lowest = min(lowest, coarse.lowest);
        coarse_uv = colour_uv(coarse_level, coarse_w);
    }

    // A hole's vertex is flattened to sea level rather than left at -32767.
    // Every fragment of its triangles is discarded either way, but a vertex
    // thirty kilometres underground would stretch them across the whole scene
    // and cost a great deal of rasterization to throw away.
    let hole = lowest < NODATA_BELOW;
    if (hole) {
        height = 0.0;
    }

    var out: VertexOut;
    out.clip = camera.view_proj * vec4<f32>(ground.x, height, ground.y, 1.0);
    out.colour_uv = colour_uv(level, wf);
    out.coarse_uv = coarse_uv;
    out.level = level;
    out.coarse_level = coarse_level;
    out.morph = morph;
    out.world = vec3<f32>(ground.x, height, ground.y);
    out.nodata = select(0.0, 1.0, hole);
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Rings are sized to reach past the raster so that the camera always has a
    // level coarse enough to cover the horizon, which means their outer parts
    // hang over ground the dataset says nothing about. Reads out there clamp to
    // the border texel, so drawing them would smear the edge row outwards as a
    // plateau that looks like terrain but is not. Cut it at the last real
    // sample instead. Discarding rather than culling whole triangles keeps the
    // boundary exactly on the data's edge instead of a quad away from it, which
    // at the coarsest level is hundreds of metres.
    if (any(in.world.xz < terrain.data_min) || any(in.world.xz > terrain.data_max)) {
        discard;
    }

    // Past the near radius the raymarched pass owns this ground. Cutting here
    // rather than only culling whole patches on the CPU is what lets the radius
    // be any distance at all instead of snapping to a ring boundary.
    if (distance(in.world, camera.position.xyz) > terrain.near_radius) {
        discard;
    }

    // Holes inside the data are as real as the edge of it. Tiles with nothing
    // under them are never written, so a survey's ragged boundary and the water
    // it stops at both arrive as nodata rather than as ground.
    if (in.nodata > 0.0) {
        discard;
    }

    // The colour raster is imagery whose own lighting and shadows are already
    // in the pixels, so it is shown as-is rather than lit a second time.
    let fine = textureSampleLevel(colours, colour_sampler, in.colour_uv, in.level, 0.0);
    if (in.morph <= 0.0) {
        return vec4<f32>(fine.rgb, 1.0);
    }

    // Cross-fade the colour on the same schedule as the geometry, so the
    // texture does not visibly swim across a ring boundary.
    let coarse = textureSampleLevel(colours, colour_sampler, in.coarse_uv, in.coarse_level, 0.0);
    return vec4<f32>(mix(fine.rgb, coarse.rgb, in.morph), 1.0);
}

// ---------------------------------------------------------------------------
// The max pyramid
//
// A quadtree over each level's height window, built fresh every frame, that the
// far field is raymarched through. Layer `l` mip `m` texel (i, j) is an upper
// bound on level `l`'s surface across the closed square
// `[i*2^m, (i+1)*2^m] x [j*2^m, (j+1)*2^m]` of its window, so a ray that stays
// above that value cannot hit anything inside the square and skips the lot in
// one step.
//
// Held in window space rather than the toroidal layout the height texture uses.
// Window origins are only ever snapped to *even* texels, so a 2x2 reduction of
// the torus lines up at the first mip and at no mip above it -- from the second
// up it would be taking the maximum of texels either side of the seam, which are
// kilometres apart on the ground. Copying into window space costs one pass and
// keeps the torus out of the raymarching path entirely.
// ---------------------------------------------------------------------------

@group(2) @binding(0) var maxima: texture_2d_array<f32>;

// Only read by `cs_reduce`; `cs_cell_max` sources the height texture instead, so
// its layout does not carry this binding.
@group(3) @binding(0) var pyramid_source: texture_2d_array<f32>;
@group(3) @binding(1) var pyramid_target: texture_storage_2d_array<r32float, write>;

// The finest mip: for each quad of the window, the highest of its four corners.
//
// Bounding the *quad* rather than the sample is the whole point. A cell holding
// only the height at its own corner would say nothing about the surface between
// that corner and the next, and a ray skipping such a cell can pass clean
// through a ridge rising between two samples. That failure shows up as pinholes
// of sky scattered across the far field, which is the worst way this can go
// wrong -- so the base of the reduction is per-quad, and every mip above it
// inherits the property by induction.
@compute @workgroup_size(8, 8, 1)
fn cs_cell_max(@builtin(global_invocation_id) id: vec3<u32>) {
    // Levels below the base are not being uploaded, so their windows hold
    // whatever ground they were abandoned over. The dispatch starts at the base
    // and the raymarcher never descends below it.
    let layer = id.z + terrain.base_level;
    if (layer >= terrain.level_count || any(id.xy > vec2<u32>(terrain.window_mask))) {
        return;
    }

    // The window's last row and column have no quad beyond them. Clamping keeps
    // those cells bounding a degenerate square of their own rather than wrapping
    // round the torus to unrelated ground.
    let near = vec2<i32>(id.xy);
    let far = vec2<i32>(min(id.xy + vec2<u32>(1u), vec2<u32>(terrain.window_mask)));
    let a = height_at(layer, near);
    let b = height_at(layer, vec2<i32>(far.x, near.y));
    let c = height_at(layer, vec2<i32>(near.x, far.y));
    let d = height_at(layer, far);

    textureStore(
        pyramid_target,
        id.xy,
        layer,
        vec4<f32>(max(max(a, b), max(c, d)), 0.0, 0.0, 0.0),
    );
}

// Every mip above the first: the maximum of the four cells beneath it.
//
// Self-describing from the target's own size, so the mip being written never has
// to be passed in and the pipeline needs no immediates.
@compute @workgroup_size(8, 8, 1)
fn cs_reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    let layer = id.z + terrain.base_level;
    let size = textureDimensions(pyramid_target);
    if (layer >= terrain.level_count || any(id.xy >= size)) {
        return;
    }

    let s = vec2<i32>(id.xy * 2u);
    let a = textureLoad(pyramid_source, s, layer, 0).r;
    let b = textureLoad(pyramid_source, s + vec2<i32>(1, 0), layer, 0).r;
    let c = textureLoad(pyramid_source, s + vec2<i32>(0, 1), layer, 0).r;
    let d = textureLoad(pyramid_source, s + vec2<i32>(1, 1), layer, 0).r;

    textureStore(
        pyramid_target,
        id.xy,
        layer,
        vec4<f32>(max(max(a, b), max(c, d)), 0.0, 0.0, 0.0),
    );
}

// ---------------------------------------------------------------------------
// The raymarched far field
//
// Everything past `near_radius` is found per pixel instead of per vertex, by
// walking a ray through the max pyramid: skip a cell whenever the ray stays
// above the ceiling it holds, climb a mip after every skip, descend a mip when
// a cell might be hit, and at the finest mip solve against the surface itself.
// Tevs, Ihrke and Seidel, "Maximum Mipmaps for Fast, Accurate, and Scalable
// Dynamic Height Field Rendering", I3D 2008.
//
// The ray starts at `near_radius` rather than at the eye. The raster stage cuts
// its fragments at the same distance, so the two cover complementary shells and
// never draw the same ground twice -- which matters because they do not agree
// about it to the last centimetre, and an overlap would z-fight the length of
// the near field.
// ---------------------------------------------------------------------------

// How many cells one ray may visit before giving up and showing sky. A grazing
// ray along a ridgeline is the case that runs long; letting it show sky is a far
// gentler failure than letting it cost whatever it likes.
const MAX_STEPS: u32 = 192u;

// Halvings used to place the intercept once the quad holding it is known. Eight
// takes a quad to a two-hundred-and-fiftieth of its width, which is far finer
// than the pixel that asked.
const REFINE_STEPS: u32 = 8u;

// Stand-in for an infinite distance, in a form arithmetic can still be done on.
const NEVER: f32 = 1e30;

struct FarOut {
    @builtin(position) clip: vec4<f32>,
    // Normalized device coordinates, which is what the camera's ray basis wants.
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_far(@builtin(vertex_index) index: u32) -> FarOut {
    // One oversized triangle rather than two: no diagonal seam down the middle
    // of the screen, and the quads either side of it are not rasterized twice.
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    let ndc = corner * 2.0 - 1.0;

    var out: FarOut;
    out.clip = vec4<f32>(ndc, 1.0, 1.0);
    out.ndc = ndc;
    return out;
}

struct Hit {
    found: bool,
    // Where the ray met the ground, in world space.
    position: vec3<f32>,
    // And which level's window it was found in, and where in that window, so the
    // colour can be looked up from the same place the height came from.
    level: u32,
    w: vec2<f32>,
};

// Walks `dir` from `start` metres out until it meets the ground.
//
// Level selection is the clipmap's own rule rather than an approximation of it:
// a point belongs to the finest level whose grid still contains it, which is
// exactly the level whose ring the rasterizer would have drawn it with. The
// march therefore reads the same data the mesh would have, and the radius the
// two halves meet at costs no detail.
//
// The level only ever coarsens along a ray. Distance from the eye grows with
// `t`, every window is centred on the eye, so a ray that has left one grid
// cannot re-enter it.
fn march(eye: vec3<f32>, dir: vec3<f32>, start: f32) -> Hit {
    var out: Hit;
    out.found = false;

    let coarsest = terrain.level_count - 1u;
    // A ray that has run further than the coarsest grid's diagonal has left it
    // -- or is going straight up or down, in which case it never will and has to
    // be stopped some other way. Keeping the bound finite is also what makes an
    // infinite `near_radius`, meaning "rasterize everything", fall straight out
    // as a miss on the first step.
    let limit = start + length(terrain.grid_quads * terrain.levels[coarsest].spacing);

    var level = terrain.base_level;
    // The eye's position in the current level's window, and how many of its
    // texels a metre along the ray crosses. Both halve on handing over to the
    // level outside, which is precisely what `coarse_offset` describes, so the
    // ray needs no world-space round trip per step.
    var w0 = (eye.xz - terrain.levels[level].origin) / terrain.levels[level].spacing;
    var wd = dir.xz / terrain.levels[level].spacing;
    var mip = terrain.max_mip;
    var t = start;
    // Whether the ray has advanced at all. Being under the ground before it has
    // means something quite different from being under it after.
    var moved = false;

    for (var step = 0u; step < MAX_STEPS; step += 1u) {
        if (t >= limit) {
            return out;
        }
        let w = w0 + wd * t;

        if (any(w < vec2<f32>(0.0)) || any(w > vec2<f32>(terrain.grid_quads))) {
            if (level >= coarsest) {
                return out;
            }
            w0 = w0 * 0.5 + terrain.levels[level].coarse_offset;
            wd = wd * 0.5;
            level += 1u;
            mip = terrain.max_mip;
            continue;
        }

        // Where the ray leaves the cell it is standing in, at this mip.
        let size = f32(1u << mip);
        let cell = floor(w / size);
        let wall = (cell + select(vec2<f32>(0.0), vec2<f32>(1.0), wd > vec2<f32>(0.0))) * size;
        let cross = (wall - w) / wd;
        let span = min(
            select(NEVER, cross.x, abs(wd.x) > 0.0),
            select(NEVER, cross.y, abs(wd.y) > 0.0),
        );
        let exit = min(t + max(span, 0.0), limit);
        // A thousandth of a texel past the wall, so the next step lands in the
        // next cell rather than back in this one. Expressed in texels because
        // the metres they are worth vary by a factor of two per level.
        let nudge = 0.001 / max(max(abs(wd.x), abs(wd.y)), 1.0 / NEVER);

        let ceiling = textureLoad(maxima, vec2<i32>(cell), i32(level), i32(mip)).r;
        // The cell bounds its whole closed square, so a ray above that ceiling
        // at both ends of the segment is above it throughout.
        if (min(eye.y + dir.y * t, eye.y + dir.y * exit) > ceiling) {
            t = exit + nudge;
            moved = true;
            mip = min(mip + 1u, terrain.max_mip);
            continue;
        }

        if (mip > 0u) {
            mip -= 1u;
            continue;
        }

        // At the finest mip the cell is a single quad, and the ground across it
        // is the bilinear patch through its four corners.
        let enter = height_bilinear(level, w);
        let leave = height_bilinear(level, w0 + wd * exit);

        // Ground nothing is known about is not ground. Cutting the whole quad
        // matches how the raster stage cuts any triangle touching a hole.
        if (min(enter.lowest, leave.lowest) < NODATA_BELOW) {
            t = exit + nudge;
            moved = true;
            mip = min(mip + 1u, terrain.max_mip);
            continue;
        }

        if (eye.y + dir.y * t <= enter.height) {
            if (!moved) {
                // The ray began below the surface. Where it went in is behind
                // the eye and cannot be found from here, and the ground over it
                // is inside the near radius, so it belongs to the raster stage.
                return out;
            }
            // Otherwise the ray has grazed the surface within a hair of a cell
            // boundary -- every earlier cell it crossed it was proven to clear,
            // so the crossing is here, not somewhere it was skipped past.
            // Reporting the hit at the boundary rather than giving up is what
            // stops a one-pixel crack appearing wherever the ground happens to
            // meet a ray exactly where two cells join.
            out.found = true;
            out.position = eye + dir * t;
            out.level = level;
            out.w = w;
            return out;
        }
        if (eye.y + dir.y * exit > leave.height) {
            // The ceiling allowed a hit somewhere in the cell; the surface
            // itself does not reach the ray.
            t = exit + nudge;
            moved = true;
            mip = min(mip + 1u, terrain.max_mip);
            continue;
        }

        // Above at one end and below at the other, so the crossing is bracketed.
        var above = t;
        var below = exit;
        for (var i = 0u; i < REFINE_STEPS; i += 1u) {
            let middle = 0.5 * (above + below);
            let ground = height_bilinear(level, w0 + wd * middle).height;
            if (eye.y + dir.y * middle > ground) {
                above = middle;
            } else {
                below = middle;
            }
        }

        out.found = true;
        out.position = eye + dir * below;
        out.level = level;
        out.w = w0 + wd * below;
        return out;
    }

    return out;
}

struct FarHit {
    @location(0) colour: vec4<f32>,
    // Written rather than interpolated, so the far field takes its place in the
    // depth buffer at the distance it was actually found at. That costs early
    // depth rejection -- a shader writing its own depth cannot be skipped before
    // it runs -- and buys correct occlusion against the near field, and a depth
    // buffer that later passes can read.
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_far(in: FarOut) -> FarHit {
    var out: FarHit;
    out.colour = vec4<f32>(0.0);
    out.depth = 0.0;

    let eye = camera.position.xyz;
    let dir = normalize(
        camera.ray_right.xyz * in.ndc.x + camera.ray_up.xyz * in.ndc.y + camera.ray_forward.xyz,
    );
    let start = max(terrain.near_radius, 0.0);

    // The coarsest level's top mip is one texel covering every window, so it is
    // the highest ground anywhere the clipmap can currently see. A ray already
    // above it and still climbing is never coming back down.
    let everything = textureLoad(
        maxima,
        vec2<i32>(0),
        i32(terrain.level_count - 1u),
        i32(terrain.max_mip),
    ).r;
    if (dir.y >= 0.0 && eye.y + dir.y * start >= everything) {
        discard;
        return out;
    }

    let hit = march(eye, dir, start);
    if (!hit.found) {
        discard;
        return out;
    }

    // Rings reach past the raster and reads out there clamp to the border, so
    // the ground beyond the last real sample is invented. The raster stage cuts
    // it and so does this one. A straight ray leaves the data once and never
    // comes back, so there is nothing further along worth carrying on for.
    if (any(hit.position.xz < terrain.data_min) || any(hit.position.xz > terrain.data_max)) {
        discard;
        return out;
    }

    let clip = camera.view_proj * vec4<f32>(hit.position, 1.0);
    out.depth = clip.z / clip.w;
    out.colour = vec4<f32>(
        textureSampleLevel(
            colours,
            colour_sampler,
            colour_uv(hit.level, hit.w),
            hit.level,
            0.0,
        ).rgb,
        1.0,
    );
    return out;
}
