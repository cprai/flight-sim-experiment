// Shading: the G-buffer in, the image out.
//
// The geometry pass has already done the expensive part -- every pixel holds
// the material id, world position and surface normal of the ground its ray
// met, or depth zero where it met nothing. This pass is one fetch, one table
// lookup and one dot product per pixel: the table gives the material's colour
// and the normal decides how much of the sun reaches it. Real material texture
// belongs here later, which is the point of the split -- it will not touch the
// march.

// Must match `PALETTE_SIZE` in `src/palette.rs`: one slot per id up to the
// last assigned category block.
const PALETTE_SIZE: u32 = 2304u;

// Must match `CLEAR_COLOR` in `src/scene.rs`. The sky is drawn here rather
// than left to the clear because this pass writes every pixel.
const SKY: vec4<f32> = vec4<f32>(0.30, 0.55, 0.85, 1.0);

// Missing data. Ids at or past the table's end can only come from a corrupt
// tile -- in-range unassigned ids are magenta in the table itself.
const MAGENTA: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

// How the light splits between the sun and everything else.
//
// The stand-in for a sky: no shadows are traced, so ground facing away from
// the sun is lit by this alone and would otherwise be black. A third of the
// light is enough to keep the material readable in the shade while leaving the
// slope that faces the sun clearly brighter than the one that does not.
//
// The two sum to one, which fixes what the palette means: a material's colour
// in `src/palette.rs` is what it looks like square-on to the sun, and no pixel
// is ever brighter than the colour it was given.
const AMBIENT: f32 = 0.35;
const SUNLIGHT: f32 = 0.65;

// Must match `MATERIAL_MASK` in `src/terrain.wgsl`.
const MATERIAL_MASK: u32 = 0xffffu;

struct Palette {
    colours: array<vec4<f32>, 2304>,
};

// What the world is lit by. Mirrors `SkyUniform` in `src/sky.rs`.
//
// `sun` is the unit vector pointing at the sun from the ground; `w` is unused
// padding.
struct Sky {
    sun: vec4<f32>,
};

// Group 1, because group 0 is the camera for every pipeline in this program
// and this pass will want it as soon as the shading needs a world position.
// Group 2 is left for the scattering tables and group 3 is this pass's own,
// which is the arrangement the atmosphere is being built into rather than one
// this commit needs on its own.
@group(1) @binding(0) var<uniform> sky: Sky;

@group(3) @binding(0) var<uniform> palette: Palette;
// A material id in the low sixteen bits and where inside its pixel the ground
// sits in the rest; only the id is wanted here. See `MATERIAL_MASK` in
// `src/terrain.wgsl` for what the other half is for.
@group(3) @binding(1) var material: texture_2d<u32>;
@group(3) @binding(3) var depth: texture_2d<f32>;
@group(3) @binding(4) var normal: texture_2d<f32>;

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

    // The march writes zero depth where its ray found no ground, and the
    // reversed-infinite projection cannot write zero for any finite hit, so
    // this test is exact: nothing is there and it is sky.
    if (textureLoad(depth, pixel, 0).r == 0.0) {
        return SKY;
    }

    let id = textureLoad(material, pixel, 0).r & MATERIAL_MASK;
    if (id >= PALETTE_SIZE) {
        return MAGENTA;
    }

    // Lambert, and nothing more: the fraction of the sun a surface at this
    // angle collects, floored at zero where it has turned away entirely. No
    // shadows, so a slope facing the sun is bright whatever stands between it
    // and the sun -- the relief this brings out is local, and a mountain does
    // not yet darken the valley behind it.
    let surface = textureLoad(normal, pixel, 0).xyz;
    let light = AMBIENT + SUNLIGHT * max(dot(surface, sky.sun.xyz), 0.0);

    // The palette is stored linearised and the target re-encodes on write, so
    // the light scales the colour in the space where scaling light is what it
    // means. Alpha is the palette's, which is opaque.
    let colour = palette.colours[id];
    return vec4<f32>(colour.rgb * light, colour.a);
}
