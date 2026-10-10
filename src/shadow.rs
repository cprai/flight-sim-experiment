//! The terrain's shadow, as a height above the ground: cascades of it centred on
//! the eye, swept along the sun every frame.
//!
//! What the field is, and why it is a height rather than a shadow map, is said
//! over `cs_shadow_sweep` in `src/terrain.wgsl`; this is where its cascades
//! stand and how the sweep is told to walk them. Everything here is a function
//! of where the eye is and where the sun is, never of which way the eye looks.

use glam::{Vec2, Vec3};

use crate::terrain::gpu::Terrain;

/// Texels across one cascade. Must match `SHADOW_ACROSS` in `src/terrain.wgsl`.
///
/// A thousand and twenty-four on the terrain's own resident lattice is eight
/// kilometres of 8 m texels for the innermost cascade, which is the resolution
/// the ground under the camera is drawn at before the generated levels take
/// over. The sweep is one workgroup per line, so this is also the length of a
/// line: [`SWEEP_GROUP`] threads of `SHADOW_ACROSS / SWEEP_GROUP` texels apiece.
pub const SHADOW_ACROSS: u32 = 1024;

/// Cascades, each on the lattice of the terrain level one coarser than the one
/// inside it, so each spans twice as far. Must match `SHADOW_CASCADES` in
/// `src/terrain.wgsl`.
///
/// Six from an 8 m base reaches 256 m texels over 262 km, which is more than
/// twice the width of any raster this has been given. That is what lets the
/// outermost cascade need nothing outside it: from anywhere on the raster, all
/// of the raster is inside it, so every occluder that can cast a shadow onto
/// anything the camera can see has been swept.
pub const SHADOW_CASCADES: u32 = 6;

/// Threads in one sweep workgroup. Must match `SWEEP_GROUP` in
/// `src/terrain.wgsl`.
const SWEEP_GROUP: u32 = 256;
const _: () = assert!(SHADOW_ACROSS % SWEEP_GROUP == 0, "a line is whole runs");

/// The steepest the sun is taken to stand, as the tangent of its elevation.
///
/// The sweep adds `j * drop` to every height it scans, and a sun at the zenith
/// makes `drop` infinite. Eighty-four degrees keeps the largest of those within
/// a couple of hundred kilometres, which an `f32` still holds to a centimetre,
/// and a shadow that short is shorter than a texel anyway.
const STEEPEST: f32 = 10.0;

/// The format the field is held in: the shadow height as the bits of an `f32`,
/// and the distance to the occluder and the ground's own clearance as two half
/// floats. See `out_shadow` in `src/terrain.wgsl`.
///
/// Thirty-two bits for the height because it is compared against points in the
/// air: a half float has two-metre steps at three thousand metres, and a shadow
/// surface stepping by two metres across a slope it grazes is acne. Integers
/// rather than floats so the halves can share a word, and because nothing reads
/// the field through a filter anyway: every reader compares four taps of its own
/// before blending.
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

/// Mirrors `Shadow` in `src/terrain.wgsl` and every other shader that reads the
/// field.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniform {
    /// Per cascade: world XZ of texel (0, 0), and the metres between texels
    /// along X and Z.
    cascade: [[f32; 4]; SHADOW_CASCADES as usize],
    /// The horizontal unit vector towards the sun, the tangent of its
    /// elevation, and the penumbra's growth per metre from the occluder.
    sun: [f32; 4],
    /// The highest ground anywhere, then padding.
    bounds: [f32; 4],
}

/// Mirrors `Sweep` in `src/terrain.wgsl`: one cascade's lines.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct SweepUniform {
    cascade: u32,
    level: u32,
    major: u32,
    first: i32,
    forward: i32,
    slope: f32,
    step: f32,
    drop: f32,
    lines: i32,
    padding: [u32; 3],
}

/// Where the terrain's texels stand, as [`Terrain::lattice`] gives it.
#[derive(Clone, Copy, Debug)]
pub struct Lattice {
    /// World XZ of level 0's texel (0, 0).
    pub origin: Vec2,
    /// Metres per level-0 texel along X and Z.
    pub metres: Vec2,
    /// The level mip zero of the resident chain holds.
    pub resident_base: u32,
    /// How many levels the resident chain has.
    pub mips: u32,
}

impl Lattice {
    fn of(terrain: &Terrain) -> Self {
        let (origin, metres, resident_base, mips) = terrain.lattice();
        Self {
            origin,
            metres,
            resident_base,
            mips,
        }
    }

    /// The metres between texels of cascade `k`, along X and Z.
    fn texel(&self, k: u32) -> Vec2 {
        self.metres * (1u32 << (self.resident_base + k)) as f32
    }

    /// The level cascade `k` reads its heights from: its own, or the coarsest
    /// there is when the chain stops short of it.
    fn level(&self, k: u32) -> u32 {
        (self.resident_base + k).min(self.resident_base + self.mips.max(1) - 1)
    }
}

impl ShadowUniform {
    /// Where the cascades go for an eye and a sun, over the terrain's lattice,
    /// with nothing standing above `ceiling`.
    pub fn at(eye: Vec3, sun: Vec3, lattice: Lattice, ceiling: f32) -> Self {
        // Each cascade centred on the eye and snapped to a whole texel of its own,
        // counted from the terrain's origin -- so its texels are the terrain's own
        // height samples at that level, and the lattice stands still in the world
        // while the camera moves across it. See `LightUniform::at` in
        // `src/cloud.rs`, which does the same for the same reason: a shadow that
        // slid with the eye would crawl.
        let cascade = std::array::from_fn(|k| {
            let texel = lattice.texel(k as u32);
            let eye = Vec2::new(eye.x, eye.z);
            let centre = lattice.origin + ((eye - lattice.origin) / texel).round() * texel;
            let corner = centre - texel * (SHADOW_ACROSS / 2) as f32;
            [corner.x, corner.y, texel.x, texel.y]
        });
        let flat = Vec2::new(sun.x, sun.z);
        let across = flat.length();
        let (toward, tan) = if across > 1e-6 {
            (flat / across, (sun.y / across).clamp(0.0, STEEPEST))
        } else {
            (Vec2::X, STEEPEST)
        };
        // The sun is a disc and not a point, so an edge of shadow is a band whose
        // width grows with the distance back to what casts it: the heights the
        // disc's upper and lower limbs clear are `d tan(e + r)` and `d tan(e - r)`
        // apart, which for a small radius is `2 r d / cos^2 e`.
        let penumbra = 2.0 * crate::sky::SUN_ANGULAR_RADIUS * (1.0 + tan * tan);
        Self {
            cascade,
            sun: [toward.x, toward.y, tan, penumbra],
            // An infinite ceiling, which is what the terrain holds until it has
            // read its chain, leaves nothing above it -- so nothing is let off.
            bounds: [ceiling.min(f32::MAX), 0.0, 0.0, 0.0],
        }
    }

    /// How each cascade is swept, and how many lines that takes.
    fn sweeps(&self, lattice: Lattice) -> [(SweepUniform, u32); SHADOW_CASCADES as usize] {
        let toward = Vec2::new(self.sun[0], self.sun[1]);
        let tan = self.sun[2];
        std::array::from_fn(|k| {
            let k = k as u32;
            let texel = lattice.texel(k);
            // Away from the sun, in texels per metre of travel along each axis.
            let away = -toward / texel;
            let major = usize::from(away.y.abs() > away.x.abs());
            let minor = 1 - major;
            // One texel along the major axis, and what that is along the line.
            let step = texel[major] / toward[major].abs().max(1e-12);
            let slope = away[minor] / away[major].abs().max(1e-12);
            // Every line starts at an integer minor intercept on the upstream
            // edge and stands at `line + i * slope` at step `i`. A texel is
            // resolved from the two lines either side of it, so the intercepts
            // have to run over every pair that brackets some texel at some step:
            // from the line below the lowest intercept a texel can have to the
            // line above the highest. See `cs_shadow_resolve`.
            let reach = ((SHADOW_ACROSS - 1) as f32 * slope.abs()).ceil() as i32;
            let first = if slope >= 0.0 { -reach - 1 } else { -1 };
            let lines = SHADOW_ACROSS + reach as u32 + 2;
            let sweep = SweepUniform {
                cascade: k,
                level: lattice.level(k),
                major: major as u32,
                first,
                forward: if away[major] >= 0.0 { 1 } else { -1 },
                slope,
                step,
                drop: step * tan,
                lines: lines as i32,
                padding: [0; 3],
            };
            (sweep, lines)
        })
    }
}

/// The swept field and everything that sweeps it.
pub struct Shadow {
    #[allow(dead_code, reason = "read by the tests that check what was swept")]
    texture: wgpu::Texture,
    /// All the cascades, for the passes that read the field.
    view: wgpu::TextureView,
    uniform: wgpu::Buffer,
    /// One per cascade: which lines it walks.
    sweep_uniforms: Vec<wgpu::Buffer>,
    /// One per cascade: its own layer to write and the one outside it to read.
    bind_groups: Vec<wgpu::BindGroup>,
    /// Walks every line of a cascade into the scratch buffer below.
    sweep: wgpu::ComputePipeline,
    /// Gathers those lines into the cascade's texels.
    resolve: wgpu::ComputePipeline,
    /// One cascade's lines at a time; see `swept_lines` in `src/terrain.wgsl`.
    #[allow(dead_code, reason = "kept alive by the bind groups; owned here")]
    scratch: wgpu::Buffer,
    /// How many lines each cascade's sweep is this frame.
    lines: [u32; SHADOW_CASCADES as usize],
    lattice: Lattice,
}

impl Shadow {
    pub fn new(device: &wgpu::Device, terrain: &Terrain) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("terrain shadow"),
            size: wgpu::Extent3d {
                width: SHADOW_ACROSS,
                height: SHADOW_ACROSS,
                depth_or_array_layers: SHADOW_CASCADES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("terrain shadow cascades"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer = |k: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("terrain shadow cascade"),
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: k,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        // What the outermost cascade reads as the one outside it, which it never
        // does: a binding has to name something, and it may not name the layer
        // the same dispatch writes.
        let nothing = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("terrain shadow beyond"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SHADOW_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain shadow uniform"),
            size: std::mem::size_of::<ShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sweep_uniforms: Vec<_> = (0..SHADOW_CASCADES)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("terrain shadow sweep"),
                    size: std::mem::size_of::<SweepUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        // The most lines a sweep can have: a sun on the diagonal, whose lines
        // lean a whole texel a step and so start from both edges.
        let scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain shadow lines"),
            size: u64::from((2 * SHADOW_ACROSS + 2) * SHADOW_ACROSS) * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let uniform_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain shadow sweep layout"),
            entries: &[
                uniform_entry(12),
                uniform_entry(13),
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: SHADOW_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_groups = (0..SHADOW_CASCADES)
            .map(|k| {
                let outer = if k + 1 < SHADOW_CASCADES {
                    layer(k + 1)
                } else {
                    nothing.clone()
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("terrain shadow sweep bind group"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 12,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 13,
                            resource: sweep_uniforms[k as usize].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 14,
                            resource: wgpu::BindingResource::TextureView(&layer(k)),
                        },
                        wgpu::BindGroupEntry {
                            binding: 15,
                            resource: wgpu::BindingResource::TextureView(&outer),
                        },
                        wgpu::BindGroupEntry {
                            binding: 16,
                            resource: scratch.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect();
        let sweep = terrain.pipeline(
            device,
            "terrain shadow sweep pipeline",
            "cs_shadow_sweep",
            [None, None, Some(&layout)],
        );
        let resolve = terrain.pipeline(
            device,
            "terrain shadow resolve pipeline",
            "cs_shadow_resolve",
            [None, None, Some(&layout)],
        );
        Self {
            texture,
            view,
            uniform,
            sweep_uniforms,
            bind_groups,
            sweep,
            resolve,
            scratch,
            lines: [0; SHADOW_CASCADES as usize],
            lattice: Lattice::of(terrain),
        }
    }

    /// Places the cascades on the eye and lays the sweeps along the sun.
    pub fn set_frame(&mut self, queue: &wgpu::Queue, eye: Vec3, sun: Vec3, ceiling: f32) {
        let uniform = ShadowUniform::at(eye, sun, self.lattice, ceiling);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        for (k, (sweep, lines)) in uniform.sweeps(self.lattice).into_iter().enumerate() {
            queue.write_buffer(&self.sweep_uniforms[k], 0, bytemuck::bytes_of(&sweep));
            self.lines[k] = lines;
        }
    }

    /// Records the sweep of cascade `k` along its lines, into a pass of its own.
    ///
    /// Outermost first, each in its own pass: a cascade reads the one outside
    /// it, and a pass boundary is what makes that one's writes visible.
    pub fn draw_sweep(&self, pass: &mut wgpu::ComputePass<'_>, terrain: &Terrain, k: u32) {
        pass.set_pipeline(&self.sweep);
        pass.set_bind_group(1, terrain.bind_group(), &[]);
        pass.set_bind_group(3, &self.bind_groups[k as usize], &[]);
        pass.dispatch_workgroups(self.lines[k as usize], 1, 1);
    }

    /// Records the gather of cascade `k`'s lines into its texels, in the pass
    /// after its sweep and before the next cascade's, which overwrites them.
    pub fn draw_resolve(&self, pass: &mut wgpu::ComputePass<'_>, terrain: &Terrain, k: u32) {
        pass.set_pipeline(&self.resolve);
        pass.set_bind_group(1, terrain.bind_group(), &[]);
        pass.set_bind_group(3, &self.bind_groups[k as usize], &[]);
        let groups = SHADOW_ACROSS.div_ceil(16);
        pass.dispatch_workgroups(groups, groups, 1);
    }

    /// Every cascade of the field, as a two-dimensional array.
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Where the cascades stand and what the sun is doing, as the readers of
    /// [`Shadow::view`] need it.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    #[cfg(test)]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lattice() -> Lattice {
        Lattice {
            origin: Vec2::new(-49152.0, -57344.0),
            metres: Vec2::splat(1.0),
            resident_base: 3,
            mips: 12,
        }
    }

    fn shader_says(name: &str) -> String {
        let source = include_str!("terrain.wgsl");
        let line = source
            .lines()
            .find(|line| line.starts_with(&format!("const {name}:")))
            .unwrap_or_else(|| panic!("no {name} in src/terrain.wgsl"));
        line.split('=').nth(1).unwrap().trim().trim_end_matches(';').to_string()
    }

    #[test]
    fn the_shader_agrees_about_the_cascades() {
        assert_eq!(shader_says("SHADOW_ACROSS"), format!("{SHADOW_ACROSS}u"));
        assert_eq!(shader_says("SHADOW_CASCADES"), format!("{SHADOW_CASCADES}u"));
        assert_eq!(shader_says("SWEEP_GROUP"), format!("{SWEEP_GROUP}u"));
        assert_eq!(
            shader_says("SWEEP_RUN"),
            format!("{}u", SHADOW_ACROSS / SWEEP_GROUP)
        );
    }

    #[test]
    fn every_shadow_cascade_stands_still_while_the_camera_moves() {
        let sun = Vec3::new(0.3, 0.2, -0.9).normalize();
        let here = ShadowUniform::at(Vec3::new(0.0, 500.0, 0.0), sun, lattice(), 3000.0);
        for k in 0..SHADOW_CASCADES as usize {
            let texel = here.cascade[k][2];
            // Anywhere within half a texel of here leaves every cascade where it
            // was; a whole texel moves it by exactly a texel.
            let near = ShadowUniform::at(
                Vec3::new(0.3 * texel, 900.0, -0.3 * texel),
                sun,
                lattice(),
                3000.0,
            );
            assert_eq!(near.cascade[k], here.cascade[k], "cascade {k}");
            // And it stands on the terrain's own lattice.
            let from = (here.cascade[k][0] - lattice().origin.x) / texel;
            assert_eq!(from, from.round(), "cascade {k} is off the lattice");
        }
    }

    #[test]
    fn every_texel_stands_between_two_swept_lines() {
        let n = SHADOW_ACROSS as i32;
        for degrees in (0..360).step_by(7).chain([0, 45, 90, 135, 180, 225, 270, 315]) {
            let azimuth = (degrees as f32).to_radians();
            let sun = Vec3::new(azimuth.sin(), 0.3, -azimuth.cos()).normalize();
            let uniform = ShadowUniform::at(Vec3::ZERO, sun, lattice(), 3000.0);
            for (sweep, lines) in uniform.sweeps(lattice()) {
                assert!(sweep.slope.abs() <= 1.0, "a line skips texels at {degrees}");
                let last = sweep.first + lines as i32 - 1;
                // The extremes of where a texel can stand across the lines are
                // at the corners of the cascade, at its first and last steps.
                for minor in [0, n - 1] {
                    for i in [0, n - 1] {
                        let across = minor as f32 - i as f32 * sweep.slope;
                        let below = across.floor() as i32;
                        assert!(
                            below >= sweep.first && below + 1 <= last,
                            "at {degrees} degrees a texel at {across} falls outside \
                             lines {}..={last}",
                            sweep.first
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_sweep_runs_away_from_the_sun() {
        // A sun in the north-east: lines start at the north-east edge and the
        // shadow falls towards the south-west.
        let sun = Vec3::new(0.6, 0.3, -0.5).normalize();
        let uniform = ShadowUniform::at(Vec3::ZERO, sun, lattice(), 3000.0);
        let (sweep, _) = uniform.sweeps(lattice())[0];
        assert_eq!(sweep.major, 0, "east-west is the longer lean");
        assert_eq!(sweep.forward, -1, "away from an eastern sun is westwards");
        assert!(sweep.slope > 0.0, "and away from a northern one is southwards");
        let tan = sun.y / Vec2::new(sun.x, sun.z).length();
        assert!((sweep.drop - sweep.step * tan).abs() < 1e-4);
    }
}
