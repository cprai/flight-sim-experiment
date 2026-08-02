// Shading: the G-buffer in, the image out.
//
// The geometry pass has already done the expensive part -- every pixel holds
// the material id and world position of the ground its ray met, or depth zero
// where it met nothing. This pass is one fetch and one table lookup per
// pixel. For now the table is a flat colour per material; lighting and real
// material texture belong here later, which is the point of the split -- they
// will not touch the march.

// Must match `PALETTE_SIZE` in `src/palette.rs`: one slot per id up to the
// last assigned category block.
const PALETTE_SIZE: u32 = 2304u;

// Must match `CLEAR_COLOR` in `src/scene.rs`. The sky is drawn here rather
// than left to the clear because this pass writes every pixel.
const SKY: vec4<f32> = vec4<f32>(0.30, 0.55, 0.85, 1.0);

// Missing data. Ids at or past the table's end can only come from a corrupt
// tile -- in-range unassigned ids are magenta in the table itself.
const MAGENTA: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

struct Palette {
    colours: array<vec4<f32>, 2304>,
};

@group(0) @binding(0) var<uniform> palette: Palette;
@group(0) @binding(1) var material: texture_2d<u32>;
// Bound but not yet read: the world-space position is the input every later
// shading feature starts from.
@group(0) @binding(2) var position: texture_2d<f32>;
@group(0) @binding(3) var depth: texture_depth_2d;

@vertex
fn vs_shade(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // The same oversized triangle the geometry pass draws.
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 1.0, 1.0);
}

@fragment
fn fs_shade(@builtin(position) clip: vec4<f32>) -> @location(0) vec4<f32> {
    // The fragment coordinate is the pixel centre, so truncation is the index.
    let pixel = vec2<i32>(clip.xy);

    // Depth clears to zero and the reversed-infinite projection cannot write
    // zero for any finite hit, so this test is exact: the march left this
    // pixel alone and it is sky.
    if (textureLoad(depth, pixel, 0) == 0.0) {
        return SKY;
    }

    let id = textureLoad(material, pixel, 0).r;
    if (id >= PALETTE_SIZE) {
        return MAGENTA;
    }
    return palette.colours[id];
}
