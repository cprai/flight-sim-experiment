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
};

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
fn height_bilinear(level: u32, w: vec2<f32>) -> f32 {
    let base = vec2<i32>(floor(w));
    let f = fract(w);
    let top = mix(
        height_at(level, base),
        height_at(level, base + vec2<i32>(1, 0)),
        f.x,
    );
    let bottom = mix(
        height_at(level, base + vec2<i32>(0, 1)),
        height_at(level, base + vec2<i32>(1, 1)),
        f.x,
    );
    return mix(top, bottom, f.y);
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
    if (coarse_level == level) {
        // The outermost ring has nothing to blend towards.
        morph = 0.0;
    }

    var height = height_at(level, w);
    var coarse_uv = vec2<f32>(0.0);
    if (morph > 0.0) {
        let coarse_w = info.coarse_offset + wf * 0.5;
        height = mix(height, height_bilinear(coarse_level, coarse_w), morph);
        coarse_uv = colour_uv(coarse_level, coarse_w);
    }

    var out: VertexOut;
    out.clip = camera.view_proj * vec4<f32>(ground.x, height, ground.y, 1.0);
    out.colour_uv = colour_uv(level, wf);
    out.coarse_uv = coarse_uv;
    out.level = level;
    out.coarse_level = coarse_level;
    out.morph = morph;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
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
