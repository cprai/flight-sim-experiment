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
    was_view_proj: mat4x4<f32>,
    position: vec4<f32>,
    ray_right: vec4<f32>,
    ray_up: vec4<f32>,
    ray_forward: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

// The material id of the ground and where inside its pixel it sits, in one
// word. The offset is half of what replaced the stored world position: see
// `rebuilt`.
@group(1) @binding(0) var history_material: texture_2d<u32>;
// The unit normal of that ground, and in `w` -- where the depth is zero --
// which of the two reasons there is no ground. See `SKY_HERE` in
// `src/terrain.wgsl`.
@group(1) @binding(2) var history_normal: texture_2d<f32>;
// How far the history's own camera found the ground down each of its rays, as
// reversed-Z depth. Two things want it, and neither of them is the depth this
// frame's points are placed at: that one is derived by the rasterizer from the
// projection below, and carrying the old one across would put every point at
// the distance the *old* camera was from it.
//
// `rebuilt` wants it as a distance, to recover the world position the G-buffer
// no longer stores, against the camera that measured it. `swept` wants it as a
// comparison, to ask whether what has swept across a pixel is nearer than what
// was standing there.
@group(1) @binding(3) var history_depth: texture_2d<f32>;

struct Splatting {
    // Which frame this is. Wraps freely; only the low six bits are read.
    frame: u32,
    // How far the eye has moved since the frame that drew the history, in
    // metres. Zero is the whole of what carried sky needs to be exact -- see
    // `swept` -- so this is tested against zero rather than scaled by.
    moved: f32,
    // Spelled out as scalars rather than a `vec3<u32>`: a three-component
    // vector is aligned to sixteen bytes in a uniform block, which would round
    // this struct up and leave the Rust mirror the wrong size.
    padding_a: u32,
    padding_b: u32,
    // The ray basis of the camera that drew the history, and in `was_eye` the
    // point it was drawn from with its near plane in `w`.
    //
    // Sky needs the basis, having only a direction and no position of its own.
    // Ground needs all of it too, now that its position is not stored: a depth
    // is a distance along a ray, and this is what says which ray, from where,
    // and at what scale. See `rebuilt`.
    was_ray_right: vec4<f32>,
    was_ray_up: vec4<f32>,
    was_ray_forward: vec4<f32>,
    was_eye: vec4<f32>,
};

@group(1) @binding(4) var<uniform> splatting: Splatting;
// What each dither cell held on the last frame: how fast the picture was
// moving across it in pixels, and how near its nearest ground was. One texel
// per cell. Only the first is wanted here; `cs_reach` in `src/terrain.wgsl`,
// which writes both, is what the second is for.
@group(1) @binding(5) var risk: texture_2d<f32>;
// The nearest ground that can have swept across each cell since the last frame,
// as reversed-Z depth. One texel per cell again, and zero where nothing within
// reach moved far enough to arrive. See `cs_reach` in `src/terrain.wgsl`.
@group(1) @binding(6) var reach: texture_2d<f32>;

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

// The screen-space motion, in pixels per frame, at which a cell is treated as
// moving as fast as this knows how to describe. Anything faster is the same as
// this.
//
// Must match `RISK_FULL` in `src/reproject.rs`.
const RISK_FULL: f32 = 8.0;

// How much of the way to dropping everything a fully risky cell is taken.
//
// One would re-march such a cell entirely, every frame, which is what the
// reprojection exists to avoid; the point is to spend more rays where the
// picture is coming apart, not to give up on it. Must match `RISK_GAIN` in
// `src/reproject.rs`.
const RISK_GAIN: f32 = 0.6;

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
// How many of the sixty-four ranks this cell drops.
//
// `DROP_RANKS` is the floor, so every cell keeps the refresh guarantee that
// constant describes however still it stands, and a cell the picture is
// sweeping across is given more on top. Adding rather than redistributing: a
// fixed budget would have had to take rays away from somewhere, and this costs
// march time only while the camera is moving.
fn ranks_for(cell: vec2<u32>) -> u32 {
    let speed = textureLoad(risk, vec2<i32>(cell), 0).r;
    let extra = RISK_GAIN * clamp(speed / RISK_FULL, 0.0, 1.0) * f32(64u - DROP_RANKS);
    return DROP_RANKS + u32(extra);
}

fn dropped(pixel: vec2<u32>) -> bool {
    let cell = pixel / DITHER_BLOCK;
    let turned = cell + vec2<u32>(splatting.frame & 7u, (splatting.frame >> 3u) & 7u);
    return bayer8(turned) < ranks_for(cell);
}

// Whether something nearer could have swept across this pixel since the frame
// that answered it.
//
// Two ways a carried answer goes stale, and they turn out to be one. Sky is
// carried as a fact about a *direction*, which is what putting it at
// `SKY_DISTANCE` says: exactly right under rotation, and wrong under
// translation, because "no ground down this ray" was established from where
// the eye was standing and the parallel ray through the same pixel from where
// it is standing now is a different one. Ground is carried as a fact about a
// *point*, which stays true -- but whether that point is still the nearest
// thing along its ray does not. A ridge sweeping across it should hide it, and
// will not if the ridge's own points were dropped by the dither or spread
// apart by magnification, so the background shows through the foreground in
// eight-by-eight speckles along every skyline.
//
// Both are the same question: has something nearer arrived here? `cs_reach`
// has already worked out what could have arrived -- the nearest ground among
// the cells whose motion carries them this far -- so this compares it against
// what is standing here. Sky is the limiting case and needs no case of its
// own: its depth is zero, the reversed-Z far plane, so anything at all that
// arrives is nearer.
//
// `OCCLUDER_NEARER` is what keeps this from firing on ground that is merely
// sloping. A surface seen at a grazing angle changes distance quickly across
// the screen without ever occluding itself, so the nearest thing within reach
// of a pixel is routinely a little nearer than the pixel; what marks a real
// occlusion boundary is a jump. Two was measured rather than picked -- see the
// constant.
//
// Only whether the eye moved is asked of `moved`, not how far. A rotation
// cannot bring ground into a ray or one surface in front of another, so a
// camera that only turns keeps the carry exactly as it was and pays nothing
// for this. Any translation at all can, and how much is what the reach already
// says.
fn swept(pixel: vec2<u32>, mine: f32) -> bool {
    if (splatting.moved == 0.0) {
        return false;
    }
    let intruder = textureLoad(reach, vec2<i32>(pixel / DITHER_BLOCK), 0).r;
    return intruder > mine * OCCLUDER_NEARER;
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

// The material word's mask, so an id can be told from the sub-pixel offset
// riding in the same word. Must match `MATERIAL_MASK` in `src/terrain.wgsl`.
const MATERIAL_MASK: u32 = 0xffffu;
const OFFSET_SCALE: f32 = 256.0;

// The direction the history's camera looked through a point of its screen,
// before it is normalised. The same arithmetic as `ray_raw_at` in
// `src/terrain.wgsl`, against the basis of the camera that drew the history
// rather than the one drawing now.
fn was_ray_raw(screen: vec2<f32>, size: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(
        screen.x / size.x * 2.0 - 1.0,
        1.0 - screen.y / size.y * 2.0,
    );
    return splatting.was_ray_right.xyz * ndc.x
        + splatting.was_ray_up.xyz * ndc.y
        + splatting.was_ray_forward.xyz;
}

// Where a pixel of the history put its ground, in world space.
//
// The G-buffer does not store it. It stores a reversed-Z depth and, in the spare
// half of the material word, where inside the pixel the point sits -- and those
// two with the camera that wrote them are the same information in twelve fewer
// bytes a pixel. The projection writes `z_near / d` for a distance `d` along the
// view axis, and the unnormalised ray advances exactly one unit along that axis,
// so the multiplication below lands on the point with no normalise and no
// length.
//
// Exact, not an approximation of what used to be stored. A marched pixel's ray
// was cast through the centre of the pixel and its offset says so; a carried
// one was rebuilt from its own depth and offset by `carried_at` in
// `src/terrain.wgsl`, by this same arithmetic against what was then the current
// camera. Either way this recovers the position that was written, bit for bit.
fn rebuilt(pixel: vec2<u32>, packed: u32, depth: f32, size: vec2<f32>) -> vec3<f32> {
    let offset = vec2<f32>(
        f32((packed >> 16u) & 0xffu),
        f32((packed >> 24u) & 0xffu),
    ) / OFFSET_SCALE;
    let at = vec2<f32>(pixel) + offset;
    return splatting.was_eye.xyz + was_ray_raw(at, size) * (splatting.was_eye.w / depth);
}

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

// How much nearer than a carried point something has to be before it counts as
// able to occlude it, as a ratio of reversed-Z depths -- which is a ratio of
// distances the other way up, so two means half as far.
//
// A threshold is needed at all because the reach is taken over a whole dither
// cell and its neighbours, and ground seen at a grazing angle changes distance
// fast across eight pixels without any occlusion being involved. Without one,
// every pixel that is not the nearest in its own cell is handed back and the
// reprojection stops carrying anything while the camera moves: measured on a
// horizon view, 66.8% of the frame carried falls to 27.6%.
//
// Two, because that is where the two populations separate on the installed
// raster. What this is meant to catch is a skyline, where the ground behind a
// ridge is several times further off than the ridge; what it must not catch is
// a slope. See the commit that added it for the sweep.
const OCCLUDER_NEARER: f32 = 2.0;

@vertex
fn vs_reproject(@builtin(vertex_index) index: u32) -> Splat {
    var out: Splat;
    out.clip = CULLED;
    out.material = 0u;
    out.normal = vec4<f32>(0.0);

    // One point per pixel of the history, in row-major order.
    let size = textureDimensions(history_depth);
    let pixel = vec2<u32>(index % size.x, index / size.x);

    // Tested before anything is read, so a dropped point pays for no memory it
    // will not use.
    if (dropped(pixel)) {
        return out;
    }

    let normal = textureLoad(history_normal, vec2<i32>(pixel), 0);
    let depth = textureLoad(history_depth, vec2<i32>(pixel), 0).r;

    if (depth == 0.0) {
        // No ground down this pixel's ray -- or nothing known about it at all.
        // The two are told apart by the normal's fourth channel, which is where
        // a marched sky pixel leaves a mark and an abandoned ray, like a buffer
        // nobody has written, leaves zeroes. There is no history to carry in the
        // second case, so the point is dropped and the march does the pixel over
        // -- which on the first frame, and the first after a resize, is the
        // whole screen, exactly as it was before any of this existed.
        if (normal.w == 0.0) {
            return out;
        }
        // Sky the eye may since have moved behind something is not sky any
        // more, so hand the pixel back rather than answer it from a ray that
        // was cast from somewhere else.
        if (swept(pixel, 0.0)) {
            return out;
        }
        // Sky, which has no world position -- only the direction the old camera
        // was looking along through this pixel. Rebuilt from that camera's ray
        // basis and placed far enough away that where the eye has moved to
        // stops mattering, which is right: sky turns with the camera and
        // ignores its translation.
        let was = normalize(was_ray_raw(vec2<f32>(pixel) + 0.5, vec2<f32>(size)));
        out.clip = camera.view_proj
            * vec4<f32>(camera.position.xyz + was * SKY_DISTANCE, 1.0);
        // Normal left at zero, which is what tells the compaction this is sky
        // rather than ground once it has landed.
        return out;
    }

    // Ground the carry is still entitled to place, but not necessarily still
    // entitled to show: something nearer may have swept in front of it.
    if (swept(pixel, depth)) {
        return out;
    }

    let id = textureLoad(history_material, vec2<i32>(pixel), 0).r;
    let stored = rebuilt(pixel, id, depth, vec2<f32>(size));

    // Points behind the eye come out with a negative `w` and are clipped.
    out.clip = camera.view_proj * vec4<f32>(stored, 1.0);
    // Exactly where inside its pixel this point is about to land. The
    // rasterizer will round it to a pixel and the fragment stage will only ever
    // see that pixel's centre, so if it is not recorded here it is gone -- and
    // the compaction would have to assume the centre, which is what makes
    // carried ground crawl once the camera moves.
    let ndc = out.clip.xy / out.clip.w;
    let landed = vec2<f32>(
        (ndc.x * 0.5 + 0.5) * f32(size.x),
        (0.5 - ndc.y * 0.5) * f32(size.y),
    );
    out.material = pack_offset(id, landed);
    out.normal = normal;
    return out;
}

// The material id and the sub-pixel position the point landed at, packed into
// one word. See `MATERIAL_MASK` in `src/terrain.wgsl` for why they share.
fn pack_offset(id: u32, screen: vec2<f32>) -> u32 {
    let frac = clamp(fract(screen), vec2<f32>(0.0), vec2<f32>(0.99609375));
    let axis = vec2<u32>(frac * OFFSET_SCALE);
    return (id & MATERIAL_MASK) | (axis.x << 16u) | (axis.y << 24u);
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
