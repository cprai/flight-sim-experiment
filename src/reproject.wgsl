// Last frame's ground, carried into this frame's camera.
//
// Every pixel of the previous G-buffer holds the world position the march found
// there. Where the camera has moved, that point is still the same point: it
// only lands somewhere else on screen. So this draws one *point primitive* per
// pixel of the last frame, projects the stored world position through the new
// camera, and lets the rasterizer put it where it now belongs. The hardware
// depth test sorts the overlaps -- when two carried points land on one pixel,
// the nearer is the one that occludes -- which is the whole reason for
// scattering rather than gathering. A gather would need to invert a motion
// field it has no way to build.
//
// What comes out is not the frame. It is a set of buffers the geometry pass
// consults before it marches: where a point landed, that pixel is already
// answered; where none did, the march runs as it always has. Nothing here can
// produce a *wrong* pixel by failing -- a pixel this leaves empty is simply
// marched.
//
// Carrying the world position rather than the depth is what makes this exact
// and what makes the previous camera unnecessary. The position is absolute, so
// a pixel that survives many frames never drifts: it stays the surface point
// the march found. What goes stale is the material and the normal stored with
// it, not the geometry.

// Must match `Camera` in `src/terrain.wgsl` and `CameraUniform` in
// `src/scene.rs`. WGSL has no includes, so the block is spelled out again.
struct Camera {
    view_proj: mat4x4<f32>,
    position: vec4<f32>,
    ray_right: vec4<f32>,
    ray_up: vec4<f32>,
    ray_forward: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var history_material: texture_2d<u32>;
@group(1) @binding(1) var history_position: texture_2d<f32>;
@group(1) @binding(2) var history_normal: texture_2d<f32>;
// Binding 3 is the history depth. It is in the layout so that one description
// of a G-buffer serves everything that reads one, and it is deliberately not
// declared here: the depth a carried point takes is the one *this* camera sees
// it at, which the rasterizer derives from the projection below. Carrying the
// old depth across would put the point at the distance the old camera was from
// it.

struct Splatting {
    // Which frame this is. Wraps freely; only the low six bits are read.
    frame: u32,
    // Spelled out as scalars rather than a `vec3<u32>`: a three-component
    // vector is aligned to sixteen bytes in a uniform block, which would round
    // this struct up and leave the Rust mirror the wrong size.
    padding_a: u32,
    padding_b: u32,
    padding_c: u32,
    // The ray basis of the camera that drew the history, which is what turns a
    // sky pixel of it back into the direction it was looking along. Ground does
    // not need this -- its world position is absolute -- but sky has no
    // position, only a direction.
    was_ray_right: vec4<f32>,
    was_ray_up: vec4<f32>,
    was_ray_forward: vec4<f32>,
};

@group(1) @binding(4) var<uniform> splatting: Splatting;

// How many of the sixty-four ranks of the dither are dropped, which is how many
// pixels are handed back to the march however well the reprojection did.
//
// Must match `DROP_RANKS` in `src/reproject.rs`, which is where the fraction it
// comes from is written down and where the refresh interval it implies is
// checked.
const DROP_RANKS: u32 = 19u;

// Side of one cell of the drop pattern, in pixels.
//
// Must match `DITHER_BLOCK` in `src/reproject.rs`.
const DITHER_BLOCK: u32 = 8u;

// The ordered 8x8 Bayer matrix, in closed form.
//
// Bayer's recursion is `M(2n) = [[4M, 4M+2], [4M+3, 4M+1]]`, so each bit of the
// coordinates contributes one base-four digit `2*(x^y) + y` -- and the *least*
// significant bit drives the *most* significant digit, because the innermost
// two-by-two is the one that splits the finest. That reversal is what makes any
// contiguous run of ranks spread evenly over the tile instead of clumping,
// which is the property the drop pattern is chosen for.
fn bayer8(cell: vec2<u32>) -> u32 {
    let y = cell.y & 7u;
    let m = (cell.x & 7u) ^ y;
    // Digit `k` of the base-four expansion is `2*(x^y) + y` taken at bit `k`,
    // and bit 0 of the coordinates drives the *most* significant digit.
    return ((m & 1u) << 5u) | ((y & 1u) << 4u)
         | ((m & 2u) << 2u) | ((y & 2u) << 1u)
         | ((m & 4u) >> 1u) | ((y & 4u) >> 2u);
}

// Whether this pixel is handed back to the march this frame.
//
// The pattern is translated by one cell a frame, sweeping the whole eight-by-
// eight torus in sixty-four frames, so every pixel takes every rank exactly
// once per cycle and is dropped exactly `DROP_RANKS` times in it.
//
// Translating rather than adding a phase to the rank: a translate of the
// pattern is still a translate of the prefix set `{p : bayer8(p) < DROP_RANKS}`,
// which is the set Bayer is built to spread evenly. Rotating the rank instead
// would make each frame's dropped set a contiguous *window* of ranks, which is
// a difference of two prefixes and clumps visibly more.
fn dropped(pixel: vec2<u32>) -> bool {
    let cell = pixel / DITHER_BLOCK
        + vec2<u32>(splatting.frame & 7u, (splatting.frame >> 3u) & 7u);
    return bayer8(cell) < DROP_RANKS;
}

struct Splat {
    @builtin(position) clip: vec4<f32>,
    // Flat throughout: a point covers one pixel, so there is nothing between
    // two vertices to interpolate, and a material id could not be interpolated
    // in any case.
    @location(0) @interpolate(flat) material: u32,
    // Zero for sky and a unit vector for ground, which is how the compaction
    // tells the two apart once they have landed.
    @location(1) @interpolate(flat) normal: vec4<f32>,
};

// Where a point goes to not be drawn: `z` outside the zero-to-one range wgpu
// clips against, so it is thrown away before rasterization. Cheaper than a
// fragment that discards, and it costs a dropped point nothing beyond the
// vertex invocation itself.
const CULLED = vec4<f32>(0.0, 0.0, -1.0, 1.0);

// What `position.w` of the history says was found there. Must match
// `GROUND_HERE` and `SKY_HERE` in `src/terrain.wgsl`; zero is a buffer nothing
// has been written to yet, which is the first frame and the first after a
// resize.
const GROUND_HERE: f32 = 1.0;

// How far away a carried sky pixel is placed.
//
// Sky belongs at infinity, and a point at infinity would be the honest way to
// say so -- but it projects to exactly the reversed-Z far plane, which is the
// value the carried buffer clears to, so it would be indistinguishable from a
// pixel nothing reached. Putting it merely very far away instead gives it a
// depth that is greater than the clear and smaller than any real ground, so it
// registers as carried and loses to any ground landing on the same pixel.
//
// Far enough that the camera's own position is lost in the rounding, which is
// what makes this ignore translation the way sky should: a thousand kilometres
// of eye movement would shift it by a ten-thousandth of a radian. Still fifty
// times nearer than the reversed-Z far plane's resolution runs out.
const SKY_DISTANCE: f32 = 1.0e9;

@vertex
fn vs_reproject(@builtin(vertex_index) index: u32) -> Splat {
    var out: Splat;
    out.clip = CULLED;
    out.material = 0u;
    out.normal = vec4<f32>(0.0);

    // One point per pixel of the history, in row-major order.
    let size = textureDimensions(history_position);
    let pixel = vec2<u32>(index % size.x, index / size.x);

    // Tested before anything is read, so a dropped point pays for no memory it
    // will not use.
    if (dropped(pixel)) {
        return out;
    }

    let stored = textureLoad(history_position, vec2<i32>(pixel), 0);

    // Zero is a buffer nothing has been written to: the first frame, and the
    // first after a resize. There is no history to carry, so every point is
    // dropped and the march does the whole frame, exactly as it did before any
    // of this existed.
    if (stored.w == 0.0) {
        return out;
    }

    if (stored.w != GROUND_HERE) {
        // Sky, which has no world position -- only the direction the old camera
        // was looking along through this pixel. Rebuilt from that camera's ray
        // basis and placed far enough away that where the eye has moved to
        // stops mattering, which is right: sky turns with the camera and
        // ignores its translation.
        let ndc = vec2<f32>(
            (f32(pixel.x) + 0.5) / f32(size.x) * 2.0 - 1.0,
            1.0 - (f32(pixel.y) + 0.5) / f32(size.y) * 2.0,
        );
        let was = normalize(
            splatting.was_ray_right.xyz * ndc.x
                + splatting.was_ray_up.xyz * ndc.y
                + splatting.was_ray_forward.xyz,
        );
        out.clip = camera.view_proj
            * vec4<f32>(camera.position.xyz + was * SKY_DISTANCE, 1.0);
        // Normal left at zero, which is what tells the compaction this is sky
        // rather than ground once it has landed.
        return out;
    }

    // Points behind the eye come out with a negative `w` and are clipped.
    out.clip = camera.view_proj * vec4<f32>(stored.xyz, 1.0);
    out.material = textureLoad(history_material, vec2<i32>(pixel), 0).r;
    out.normal = textureLoad(history_normal, vec2<i32>(pixel), 0);
    return out;
}

struct Carried {
    @location(0) material: u32,
    @location(1) normal: vec4<f32>,
};

// No `frag_depth`: the depth being written is the one the rasterizer already
// derived from the clip position above, and it is the one the depth test just
// used to decide this point won the pixel. Writing it by hand would say the
// same thing and cost the early depth test.
@fragment
fn fs_reproject(in: Splat) -> Carried {
    var out: Carried;
    out.material = in.material;
    out.normal = in.normal;
    return out;
}
