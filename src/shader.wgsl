struct Camera {
    view_proj: mat4x4<f32>,
    // World-space eye position. `w` is padding.
    position: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(position, 1.0);
    out.world_position = position;
    return out;
}

const GRASS_DARK = vec3<f32>(0.16, 0.34, 0.12);
const GRASS_LIGHT = vec3<f32>(0.24, 0.45, 0.16);
// Side length of one checker cell, in world units.
const CELL_SIZE = 50.0;
// Distance over which the checker blends away into flat grass.
const FADE_DISTANCE = 3000.0;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Two shades of green in a checkerboard: a flat plane gives the eye nothing
    // to judge the perspective by, and the cells receding to the horizon do.
    let cell = floor(in.world_position.xz / CELL_SIZE);
    // `fract` is `x - floor(x)`, so this is 0.0 or 0.5 even for negative cells.
    let checker = fract((cell.x + cell.y) * 0.5) * 2.0;

    // Cells shrink below a pixel near the horizon, which aliases into moire.
    // Fading to the average color with distance costs one mix and avoids it.
    let distance = length(in.world_position - camera.position.xyz);
    let fade = clamp(distance / FADE_DISTANCE, 0.0, 1.0);
    let average = (GRASS_DARK + GRASS_LIGHT) * 0.5;

    let color = mix(mix(GRASS_DARK, GRASS_LIGHT, checker), average, fade);
    return vec4<f32>(color, 1.0);
}
