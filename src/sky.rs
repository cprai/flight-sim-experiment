//! Where the sun is, and the uniform that tells the shaders about it.
//!
//! The sun used to be a constant in `src/shading.wgsl` -- `vec3(0.5,
//! 0.70710678, 0.5)`, mid-morning in the south-east -- with a note saying it
//! would become a uniform once there was a time of day to drive it from. This
//! is that uniform. Nothing drives it round the clock yet, but it is a value
//! rather than a constant now, which is what the atmosphere needs: every
//! scattering table below is a function of where the sun is, so a sun the CPU
//! cannot name is a sun the tables cannot be built for.
//!
//! The default reproduces the old constant, so a frame drawn without asking for
//! a sun is the frame that was drawn before. Not bit for bit -- a sine and a
//! cosine of 45 degrees in `f32` land on 0.49999997 rather than a half -- but
//! the difference is a ten-millionth of a Lambert term that ends up in eight
//! bits, and the rendered frame really is byte for byte the one from before.
//! That is deliberate and it is tested both ways: it lets the plumbing land
//! without changing a pixel.

use glam::Vec3;

/// Where the sun is, as the unit vector pointing at it from the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sun {
    pub direction: Vec3,
}

impl Sun {
    /// Elevation above the horizon and compass azimuth, both in degrees.
    ///
    /// Azimuth is a bearing, like the camera's yaw: zero is north, ninety is
    /// east. World space is Y-up with +X east and -Z north, so north is the
    /// negative Z axis and the horizontal part of the bearing turns from it
    /// towards +X.
    ///
    /// An elevation below zero is allowed and means exactly what it says: the
    /// sun is under the horizon. Nothing clamps it, because the whole point of
    /// a scattering model is that dusk is a place on the same curve as noon.
    pub fn from_angles(elevation_degrees: f32, azimuth_degrees: f32) -> Self {
        let (elevation, azimuth) = (elevation_degrees.to_radians(), azimuth_degrees.to_radians());
        let (flat, up) = (elevation.cos(), elevation.sin());
        Self {
            direction: Vec3::new(azimuth.sin() * flat, up, -azimuth.cos() * flat),
        }
    }

    /// Elevation of the sun the shader used to hold as a constant.
    ///
    /// Forty-five degrees up and forty-five round from north towards the east
    /// gives the exact halves and the exact root-half the constant was written
    /// as. High enough that nothing faces away from it outright, off-axis
    /// enough in both horizontal axes that no slope facing a cardinal direction
    /// comes out the same as its neighbours.
    pub const DEFAULT_ELEVATION: f32 = 45.0;
    /// Azimuth of the same, as a bearing: south-east.
    pub const DEFAULT_AZIMUTH: f32 = 135.0;
}

impl Default for Sun {
    fn default() -> Self {
        Self::from_angles(Self::DEFAULT_ELEVATION, Self::DEFAULT_AZIMUTH)
    }
}

/// Mirrors the `Sky` uniform block in `src/shading.wgsl`.
///
/// One member so far. It grows as the scattering does -- the tables want the
/// eye's radius and the local up as well -- and it is a block of its own rather
/// than three more words on the camera because the camera is where the eye is
/// and this is what the world is lit by. Two different things, changed by two
/// different parts of the frame.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniform {
    /// The unit vector pointing at the sun. `w` is unused padding; uniform
    /// members are aligned to sixteen bytes anyway.
    sun: [f32; 4],
}

impl SkyUniform {
    fn new(sun: Sun) -> Self {
        Self {
            sun: sun.direction.extend(0.0).to_array(),
        }
    }
}

/// The sky uniform, its layout, and the bind group tying them together.
///
/// Group 1 wherever it is bound, which is the shading pass and, later, the
/// passes that build the scattering tables. Group 0 stays the camera, as it is
/// for every other pipeline in the program.
pub struct Sky {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl Sky {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky uniform"),
            size: std::mem::size_of::<SkyUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                // The shading reads it in the fragment stage. The table builds
                // will read it in compute, which is why the visibility is
                // already both.
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            layout,
            bind_group,
        }
    }

    /// Uploads where the sun is for the frame about to be drawn.
    pub fn set_frame(&self, queue: &wgpu::Queue, sun: Sun) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&SkyUniform::new(sun)));
    }

    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The default sun is the constant `src/shading.wgsl` used to hold.
    ///
    /// This is what lets the uniform land without changing a frame, and it is
    /// worth pinning rather than trusting: the bearing convention has two ways
    /// to be wrong -- the sign of the north axis and which way the azimuth
    /// turns -- and both produce a plausible-looking sun somewhere else in the
    /// sky.
    #[test]
    fn the_default_sun_is_the_constant_the_shader_used_to_hold() {
        let was = Vec3::new(0.5, std::f32::consts::FRAC_1_SQRT_2, 0.5);
        let now = Sun::default().direction;
        assert!(
            (now - was).length() < 1e-6,
            "the default sun is {now}, not the {was} the shader held"
        );
    }

    /// The bearing means what a compass means by it.
    ///
    /// Four cardinal directions on the horizon, where the answer is an axis and
    /// there is nothing to round.
    #[test]
    fn azimuth_is_a_compass_bearing_from_north_through_east() {
        let cases = [
            (0.0, Vec3::NEG_Z, "north"),
            (90.0, Vec3::X, "east"),
            (180.0, Vec3::Z, "south"),
            (270.0, Vec3::NEG_X, "west"),
        ];
        for (azimuth, want, name) in cases {
            let got = Sun::from_angles(0.0, azimuth).direction;
            assert!(
                (got - want).length() < 1e-6,
                "azimuth {azimuth} should point {name} at {want}, got {got}"
            );
        }
    }

    /// Elevation raises it, and goes on working past the horizon.
    #[test]
    fn elevation_lifts_the_sun_and_may_take_it_below_the_horizon() {
        assert!((Sun::from_angles(90.0, 0.0).direction - Vec3::Y).length() < 1e-6);
        assert!(Sun::from_angles(-10.0, 135.0).direction.y < 0.0);
        for elevation in [-30.0, -5.0, 0.0, 12.0, 60.0, 89.0] {
            let length = Sun::from_angles(elevation, 40.0).direction.length();
            assert!(
                (length - 1.0).abs() < 1e-6,
                "elevation {elevation} gave a direction of length {length}"
            );
        }
    }

    /// The shader reads the sun out of the uniform this writes.
    ///
    /// There is no preprocessor and no reflection, so the only thing keeping
    /// the two in step is that both are written by hand. Checking the text is
    /// the cheapest guard there is against the constant creeping back in, and
    /// it is the same trick `src/reproject.rs` uses on its own constants.
    #[test]
    fn the_shading_shader_takes_its_sun_from_the_uniform() {
        let source = include_str!("shading.wgsl");
        assert!(
            source.contains("var<uniform> sky: Sky"),
            "the shading shader has no sky uniform"
        );
        assert!(
            source.contains("sky.sun.xyz"),
            "the shading shader does not read the sun out of the uniform"
        );
        // The colon matters: `const SUNLIGHT` starts with `const SUN`, and
        // that constant is still there and still wanted.
        assert!(
            !source.contains("const SUN:"),
            "the shading shader still holds a hard-coded sun"
        );
    }
}
