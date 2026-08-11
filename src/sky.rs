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

/// The planet, as the pair of spheres the scattering is integrated between.
///
/// Must match `src/sky.wgsl`. The world this draws is flat and this model is
/// not, and the mapping between them is the one real approximation in the whole
/// arrangement: the planet's centre is pinned at `(0, -GROUND_RADIUS, 0)` in
/// world space, so a world point's radius is its distance from there.
///
/// Three consequences, all small and all worth knowing:
///
/// - The local up tilts away from world `+Y` by `d / R` -- half a degree at the
///   edge of the installed raster, nine tenths of one at a hundred kilometres.
/// - Flat ground at horizontal distance `d` stands `d^2 / 2R` above the sphere:
///   265 m at the raster's edge, 785 m at a hundred kilometres. Against an
///   eight-kilometre scale height that is at most a tenth of the density, and
///   it errs towards less haze rather than more.
/// - The sphere's horizon is `sqrt(2 R h)` away: 138 km from 1500 m up, 390 km
///   from twelve. Always past the raster, so terrain never reaches it -- which
///   is what lets the sky be drawn as though the ground below the horizon were
///   simply not there. There is a test for it.
///
/// Rejected: taking the radius as `GROUND_RADIUS + y`, a stack of flat slabs.
/// It is simpler and it destroys the curved horizon, which is the feature the
/// sky-view table's whole parameterisation is built around, and it never lets
/// the sun set properly.
pub const GROUND_RADIUS: f32 = 6_360_000.0;
/// The top of the atmosphere, a hundred kilometres up.
pub const TOP_RADIUS: f32 = 6_460_000.0;

/// Format of every scattering table.
///
/// `Rgba16Float` is storage-writable *and* filterable in core WebGPU, and it
/// has to be both: a compute pass fills these and a bilinear fetch reads them.
/// `R32Float` would be the tempting choice for the transmittance, being one
/// channel of more precision, and it is a trap -- sampling a 32-bit float
/// texture with a filtering sampler needs the `FLOAT32_FILTERABLE` feature, so
/// it would turn a table into a failed device request. Half floats are ample
/// here: the values are transmittances between one and about `1e-9` and
/// radiances between `0.005` and `0.5`, and no exposure is ever baked in.
pub const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Size of the transmittance table: how much sun survives the air above a
/// point, by altitude and sun angle.
pub const TRANSMITTANCE_SIZE: glam::UVec2 = glam::UVec2::new(256, 64);
/// Size of the multiple-scattering table, by altitude and sun angle.
///
/// Far smaller than the transmittance because it is far smoother: light that
/// has bounced more than once has been averaged over every direction and has no
/// sharp horizon feature left to resolve.
pub const MULTISCATTER_SIZE: glam::UVec2 = glam::UVec2::new(32, 32);

/// Steps the shader integrates the optical depth in. Must match `src/sky.wgsl`.
const TRANSMITTANCE_STEPS: u32 = 40;

/// The sky uniform, the scattering tables, and everything that fills them.
///
/// Group 1 is the uniform and group 2 is the tables, wherever either is bound.
/// Group 0 stays the camera, as it is for every other pipeline in the program.
pub struct Sky {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    /// The two tables themselves, kept for the readback tests.
    transmittance: wgpu::Texture,
    multiscatter: wgpu::Texture,
    /// Group 2 in full: the sampler and both tables.
    tables_layout: wgpu::BindGroupLayout,
    tables_bind_group: wgpu::BindGroup,
    /// The pipelines and bindings that fill them, kept only until they have.
    build: Option<Build>,
}

/// What the one-off table build needs, and nothing a frame does.
///
/// Separated so it can be dropped once the tables are filled: two pipelines and
/// three bind groups that are used exactly once, at load.
struct Build {
    transmittance: wgpu::ComputePipeline,
    multiscatter: wgpu::ComputePipeline,
    /// Group 2 for the multiple-scattering build: the sampler and the
    /// transmittance table, and deliberately *not* the table being written.
    ///
    /// wgpu tracks a resource's usage across a whole pass, so a group holding a
    /// texture as sampled while another group has it bound for writing is a
    /// validation error -- and one raised at dispatch rather than at pipeline
    /// creation, which makes it look like a mystery. A layout that stops short
    /// of the table being written is what avoids it, and it costs nothing: the
    /// build has no use for the table it is producing.
    read_transmittance: wgpu::BindGroup,
    write_transmittance: wgpu::BindGroup,
    write_multiscatter: wgpu::BindGroup,
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
        let table = |label, size: glam::UVec2| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: LUT_FORMAT,
                // `COPY_SRC` for the reason the G-buffer's targets carry it: a
                // frame says almost nothing about what is in a table, so
                // reading one back is the only way to check it.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let transmittance = table("transmittance table", TRANSMITTANCE_SIZE);
        let multiscatter = table("multiple scattering table", MULTISCATTER_SIZE);
        let transmittance_view = transmittance.create_view(&Default::default());
        let multiscatter_view = multiscatter.create_view(&Default::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scattering table sampler"),
            // Clamped in both axes: a table runs to the ends of its range and
            // stops there. The horizon is the horizon, and the top of the
            // atmosphere is the top of it.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sampled = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sampler_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let tables_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scattering tables layout"),
            entries: &[sampler_entry, sampled(1), sampled(2)],
        });
        let sampler_binding = wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Sampler(&sampler),
        };
        let tables_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scattering tables bind group"),
            layout: &tables_layout,
            entries: &[
                sampler_binding.clone(),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&multiscatter_view),
                },
            ],
        });

        // Group 2 without the table being written; see [`Build`].
        let read_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transmittance read layout"),
            entries: &[sampler_entry, sampled(1)],
        });
        let read_transmittance = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transmittance read bind group"),
            layout: &read_layout,
            entries: &[
                sampler_binding,
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&transmittance_view),
                },
            ],
        });

        let written = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: LUT_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let write_layout = |label, binding| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[written(binding)],
            })
        };
        let write_bind = |label, layout, binding, view| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(view),
                }],
            })
        };
        let write_transmittance_layout = write_layout("transmittance write layout", 0);
        let write_multiscatter_layout = write_layout("multiscatter write layout", 1);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sky shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sky.wgsl").into()),
        });
        let stage = |label: &str, entry_point: &str, layouts: &[Option<&wgpu::BindGroupLayout>]| {
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            buffer,
            layout,
            bind_group,
            tables_layout,
            tables_bind_group,
            build: Some(Build {
                // Neither build binds the sky uniform: both tables are
                // functions of the medium's own constants and of nothing else,
                // which is exactly why they can be built once and kept.
                transmittance: stage(
                    "transmittance build",
                    "cs_transmittance",
                    &[None, None, None, Some(&write_transmittance_layout)],
                ),
                multiscatter: stage(
                    "multiscatter build",
                    "cs_multiscatter",
                    &[
                        None,
                        None,
                        Some(&read_layout),
                        Some(&write_multiscatter_layout),
                    ],
                ),
                read_transmittance,
                write_transmittance: write_bind(
                    "transmittance write bind group",
                    &write_transmittance_layout,
                    0,
                    &transmittance_view,
                ),
                write_multiscatter: write_bind(
                    "multiscatter write bind group",
                    &write_multiscatter_layout,
                    1,
                    &multiscatter_view,
                ),
            }),
            transmittance,
            multiscatter,
        }
    }

    /// Fills the transmittance and multiple-scattering tables, once.
    ///
    /// On an encoder of its own, submitted here rather than folded into a
    /// frame, the way [`crate::terrain::gpu::Terrain::build_pyramid`] raises the
    /// max pyramid at load. Nothing about it is incremental and nothing under
    /// it ever changes, so no frame should be the first to pay for it -- and
    /// the profiler is deliberately not involved, because a row that is zero on
    /// every measured frame is noise rather than a measurement.
    ///
    /// Called from `update` rather than `new` for the reason `build_pyramid` is:
    /// a constructor has no queue to submit on.
    pub fn ensure_built(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(build) = self.build.take() else {
            return;
        };
        let started = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transmittance"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&build.transmittance);
            pass.set_bind_group(3, &build.write_transmittance, &[]);
            pass.dispatch_workgroups(
                TRANSMITTANCE_SIZE.x.div_ceil(8),
                TRANSMITTANCE_SIZE.y.div_ceil(8),
                1,
            );
        }
        {
            // Its own pass, not another dispatch in the one above: the second
            // build samples the first's output, and a pass boundary is what
            // makes those writes visible. The same reason `args` has a pass to
            // itself in `src/scene.rs`.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("multiple scattering"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&build.multiscatter);
            pass.set_bind_group(2, &build.read_transmittance, &[]);
            pass.set_bind_group(3, &build.write_multiscatter, &[]);
            // One workgroup per texel: the sixty-four threads of a workgroup
            // are the sixty-four directions it integrates over.
            pass.dispatch_workgroups(MULTISCATTER_SIZE.x, MULTISCATTER_SIZE.y, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        // Waited on rather than left in flight, so the cost lands at load where
        // it belongs and the first frame finds the tables ready.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        log::info!(
            "sky: built the {} x {} transmittance and {} x {} scattering tables in {:.2?}",
            TRANSMITTANCE_SIZE.x,
            TRANSMITTANCE_SIZE.y,
            MULTISCATTER_SIZE.x,
            MULTISCATTER_SIZE.y,
            started.elapsed()
        );
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

    /// Group 2: the sampler and both tables, for whatever reads them.
    #[allow(dead_code, reason = "bound by the shading pass from the next commit")]
    pub fn tables_layout(&self) -> &wgpu::BindGroupLayout {
        &self.tables_layout
    }

    #[allow(dead_code, reason = "bound by the shading pass from the next commit")]
    pub fn tables_bind_group(&self) -> &wgpu::BindGroup {
        &self.tables_bind_group
    }

    /// The tables themselves, for the readback tests.
    #[cfg(test)]
    pub fn tables(&self) -> (&wgpu::Texture, &wgpu::Texture) {
        (&self.transmittance, &self.multiscatter)
    }
}

/// The half-texel correction, and its inverse. Mirrors `src/sky.wgsl`.
///
/// A table of `n` texels has its first centre at `0.5/n` and its last at
/// `1 - 0.5/n`, so a parameter running the full range has to be squeezed into
/// that span. Skipping it is the classic failure of this technique and it fails
/// quietly -- the picture looks nearly right, with the horizon about a degree
/// out. Written here as well as in the shader so a test can round-trip it.
pub fn to_texture(x: f32, n: f32) -> f32 {
    0.5 / n + x * (1.0 - 1.0 / n)
}

pub fn to_unit(u: f32, n: f32) -> f32 {
    (u - 0.5 / n) / (1.0 - 1.0 / n)
}

/// Distance to the top of the atmosphere. Mirrors `top_distance` in the shader.
pub fn top_distance(r: f32, mu: f32) -> f32 {
    let discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
    (-r * mu + discriminant.max(0.0).sqrt()).max(0.0)
}

/// Where `(r, mu)` sits in the transmittance table. Mirrors the shader.
pub fn transmittance_uv(r: f32, mu: f32) -> glam::Vec2 {
    let horizon = (r * r - GROUND_RADIUS * GROUND_RADIUS).max(0.0).sqrt();
    let atmosphere = (TOP_RADIUS * TOP_RADIUS - GROUND_RADIUS * GROUND_RADIUS)
        .max(0.0)
        .sqrt();
    let distance = top_distance(r, mu);
    let shortest = TOP_RADIUS - r;
    let longest = horizon + atmosphere;
    let span = (longest - shortest).max(1e-6);
    glam::Vec2::new(
        to_texture(
            ((distance - shortest) / span).clamp(0.0, 1.0),
            TRANSMITTANCE_SIZE.x as f32,
        ),
        to_texture(
            (horizon / atmosphere).clamp(0.0, 1.0),
            TRANSMITTANCE_SIZE.y as f32,
        ),
    )
}

/// The same mapping backwards, giving `(r, mu)`. Mirrors the shader.
pub fn transmittance_params(uv: glam::Vec2) -> (f32, f32) {
    let atmosphere = (TOP_RADIUS * TOP_RADIUS - GROUND_RADIUS * GROUND_RADIUS)
        .max(0.0)
        .sqrt();
    let horizon = atmosphere * to_unit(uv.y, TRANSMITTANCE_SIZE.y as f32);
    let r = (horizon * horizon + GROUND_RADIUS * GROUND_RADIUS)
        .max(0.0)
        .sqrt();
    let shortest = TOP_RADIUS - r;
    let longest = horizon + atmosphere;
    let distance = shortest + to_unit(uv.x, TRANSMITTANCE_SIZE.x as f32) * (longest - shortest);
    let mu = if distance > 0.0 {
        ((atmosphere * atmosphere - horizon * horizon - distance * distance) / (2.0 * r * distance))
            .clamp(-1.0, 1.0)
    } else {
        1.0
    };
    (r, mu)
}

/// What the air at height `h` takes out of a beam, per metre. Mirrors the
/// shader's `medium().extinction`.
pub fn extinction(height: f32) -> Vec3 {
    let h = height.max(0.0);
    let rayleigh = Vec3::new(5.802e-6, 13.558e-6, 33.100e-6) * (-h / 8000.0).exp();
    let mie = 4.400e-6 * (-h / 1200.0).exp();
    let ozone =
        Vec3::new(0.650e-6, 1.881e-6, 0.085e-6) * (1.0 - (h - 25000.0).abs() / 15000.0).max(0.0);
    rayleigh + Vec3::splat(mie) + ozone
}

/// The optical depth from `(r, mu)` to the top of the atmosphere, integrated
/// here rather than on the GPU.
///
/// Not a copy of the shader for its own sake: the readback test needs something
/// to check the table against that is not the table, and running the same
/// integral at four times the step count is the cheapest independent answer
/// there is. If the two agree, the shader's forty steps are converged and its
/// parameterisation inverts to the place the forward mapping would have put it.
pub fn optical_depth(r: f32, mu: f32, steps: u32) -> Vec3 {
    let end = top_distance(r, mu);
    let step = end / steps as f32;
    let mut depth = Vec3::ZERO;
    for i in 0..steps {
        let t = (i as f32 + 0.5) * step;
        let radius = (r * r + t * t + 2.0 * r * mu * t).max(0.0).sqrt();
        depth += extinction(radius - GROUND_RADIUS) * step;
    }
    depth
}

/// Steps the reference integral takes: four times the shader's, so agreeing
/// with it says the shader has converged rather than that both are wrong alike.
#[cfg(test)]
const REFERENCE_STEPS: u32 = TRANSMITTANCE_STEPS * 4;

#[cfg(test)]
mod tests {
    use super::*;

    /// A half float, as a whole one.
    ///
    /// The tables are `Rgba16Float` and Rust has no stable `f16`, so a readback
    /// has to decode them. Written arithmetically rather than by shifting an
    /// exponent into an `f32`'s: the three cases are then the three the format
    /// has, and each reads as its own definition.
    fn from_half(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let mantissa = f32::from(bits & 0x3ff);
        if exponent == 0 {
            // Subnormal: no implied leading one, and a fixed exponent.
            sign * mantissa * 2f32.powi(-24)
        } else if exponent == 0x1f {
            sign * if mantissa == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        } else {
            sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
        }
    }

    /// Builds the tables on the headless device and reads both back.
    ///
    /// The same device the screenshot mode and every other GPU test runs on, so
    /// what this measures is what the application would have produced.
    fn built_tables() -> (Vec<Vec3>, Vec<Vec3>) {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let mut sky = Sky::new(&device);
        sky.ensure_built(&device, &queue);
        let (transmittance, multiscatter) = sky.tables();
        (
            read_table(&device, &queue, transmittance, TRANSMITTANCE_SIZE),
            read_table(&device, &queue, multiscatter, MULTISCATTER_SIZE),
        )
    }

    /// Copies one table off the GPU as RGB triples, row-major.
    fn read_table(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        size: glam::UVec2,
    ) -> Vec<Vec3> {
        // Four half-float channels is eight bytes a texel, and both tables are
        // at least 32 wide, so every row is already a multiple of the 256-byte
        // copy alignment.
        let bytes_per_row = size.x * 8;
        assert_eq!(bytes_per_row % 256, 0, "a row of {size} would need padding");
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("table readback"),
            size: u64::from(bytes_per_row * size.y),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(size.y),
                },
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let halves: Vec<u16> =
            bytemuck::cast_slice::<u8, u16>(&readback.get_mapped_range(..).expect("mapped"))
                .to_vec();
        readback.unmap();
        halves
            .chunks_exact(4)
            .map(|texel| {
                Vec3::new(
                    from_half(texel[0]),
                    from_half(texel[1]),
                    from_half(texel[2]),
                )
            })
            .collect()
    }

    /// The centre of texel `(x, y)` of a table `size` across, in uv.
    fn texel_uv(x: u32, y: u32, size: glam::UVec2) -> glam::Vec2 {
        glam::Vec2::new(
            (x as f32 + 0.5) / size.x as f32,
            (y as f32 + 0.5) / size.y as f32,
        )
    }

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

    /// Rust and the shader agree on every constant they both spell out.
    ///
    /// There is no preprocessor and no `#include`, so each of these is written
    /// twice by hand. A pair that drifted would not fail loudly: the table
    /// would be built for one atmosphere and read for another, and the picture
    /// would simply be wrong. Compared as text because the text is already in
    /// the binary, which is the trick `src/reproject.rs` uses on its own.
    #[test]
    fn the_shader_and_rust_agree_on_the_atmosphere() {
        let source = include_str!("sky.wgsl");
        let pairs = [
            ("GROUND_RADIUS", format!("{GROUND_RADIUS:.1}"), "f32"),
            ("TOP_RADIUS", format!("{TOP_RADIUS:.1}"), "f32"),
            (
                "TRANSMITTANCE_WIDTH",
                format!("{}u", TRANSMITTANCE_SIZE.x),
                "u32",
            ),
            (
                "TRANSMITTANCE_HEIGHT",
                format!("{}u", TRANSMITTANCE_SIZE.y),
                "u32",
            ),
            (
                "MULTISCATTER_SIZE",
                format!("{}u", MULTISCATTER_SIZE.x),
                "u32",
            ),
            (
                "TRANSMITTANCE_STEPS",
                format!("{TRANSMITTANCE_STEPS}u"),
                "u32",
            ),
        ];
        for (name, value, kind) in pairs {
            let declaration = format!("const {name}: {kind}");
            let line = source
                .lines()
                .find(|line| line.trim_start().starts_with(&declaration))
                .unwrap_or_else(|| panic!("src/sky.wgsl declares no {name}"));
            assert!(
                line.contains(&value),
                "src/sky.wgsl says {line:?}, but src/sky.rs says {name} is {value}"
            );
        }
        // The multiple-scattering table is square, which the shader's single
        // constant assumes and this is the only thing that says so.
        assert_eq!(MULTISCATTER_SIZE.x, MULTISCATTER_SIZE.y);
    }

    /// Rust and the shader compute the half-texel correction the same way.
    ///
    /// The two are separate copies of a three-term formula, and the test that
    /// round-trips the Rust one against itself cannot see them disagree: a
    /// shader that dropped the correction entirely would still invert its own
    /// mapping perfectly and still write a table, just a table addressed a
    /// half-texel out at both ends. Since the arithmetic cannot be run from
    /// here, the text is what there is to compare.
    #[test]
    fn the_shader_corrects_for_the_half_texel_the_way_rust_does() {
        let source = include_str!("sky.wgsl");
        let body = |name: &str| {
            let start = source
                .find(&format!("fn {name}(x: f32, n: f32)"))
                .or_else(|| source.find(&format!("fn {name}(u: f32, n: f32)")))
                .unwrap_or_else(|| panic!("src/sky.wgsl declares no {name}"));
            let end = source[start..].find('}').expect("unterminated function");
            source[start..start + end].to_owned()
        };
        assert!(
            body("to_texture").contains("0.5 / n + x * (1.0 - 1.0 / n)"),
            "src/sky.wgsl's to_texture is not the one src/sky.rs mirrors: {}",
            body("to_texture")
        );
        assert!(
            body("to_unit").contains("(u - 0.5 / n) / (1.0 - 1.0 / n)"),
            "src/sky.wgsl's to_unit is not the one src/sky.rs mirrors: {}",
            body("to_unit")
        );
    }

    /// The transmittance parameterisation inverts to where it came from.
    #[test]
    fn the_transmittance_mapping_round_trips() {
        let mut worst: f32 = 0.0;
        for y in 0..TRANSMITTANCE_SIZE.y {
            for x in 0..TRANSMITTANCE_SIZE.x {
                let uv = texel_uv(x, y, TRANSMITTANCE_SIZE);
                let (r, mu) = transmittance_params(uv);
                let back = transmittance_uv(r, mu);
                worst = worst.max((back - uv).abs().max_element());
            }
        }
        assert!(
            worst < 1e-4,
            "the mapping and its inverse disagree by up to {worst} in uv"
        );
    }

    /// ... and its ends are the ends of the range, not somewhere inside them.
    ///
    /// The half a round trip cannot see: a mapping that squeezed the parameter
    /// into the wrong span would still invert itself perfectly.
    #[test]
    fn the_transmittance_table_reaches_the_ground_and_the_top() {
        let (bottom, up) = transmittance_params(texel_uv(0, 0, TRANSMITTANCE_SIZE));
        assert!(
            (bottom - GROUND_RADIUS).abs() < 1.0,
            "the first row is at radius {bottom}, not the ground at {GROUND_RADIUS}"
        );
        // The first column at any row is the ray that leaves soonest, which is
        // the one pointing straight up.
        assert!(
            up > 0.999,
            "the first column is at mu {up}, not straight up"
        );

        let (top, _) =
            transmittance_params(texel_uv(0, TRANSMITTANCE_SIZE.y - 1, TRANSMITTANCE_SIZE));
        assert!(
            (top - TOP_RADIUS).abs() < 1.0,
            "the last row is at radius {top}, not the top at {TOP_RADIUS}"
        );
    }

    /// The transmittance table holds what an independent integral says it does.
    ///
    /// Checked against `optical_depth` run at four times the shader's step
    /// count -- a different implementation in a different language, so agreeing
    /// says the shader's forty steps have converged *and* that its inverse
    /// mapping lands where the forward one would have put it. A test that
    /// re-derived the table from the shader's own arithmetic would say neither.
    #[test]
    fn the_transmittance_table_is_the_one_physics_says_it_is() {
        let (table, _) = built_tables();
        let mut worst: f32 = 0.0;
        let mut worst_at = (0, 0);
        for y in 0..TRANSMITTANCE_SIZE.y {
            for x in 0..TRANSMITTANCE_SIZE.x {
                let uv = texel_uv(x, y, TRANSMITTANCE_SIZE);
                let (r, mu) = transmittance_params(uv);
                let want = (-optical_depth(r, mu, REFERENCE_STEPS)).exp();
                let got = table[(y * TRANSMITTANCE_SIZE.x + x) as usize];
                let difference = (got - want).abs().max_element();
                if difference > worst {
                    worst = difference;
                    worst_at = (x, y);
                }
            }
        }
        assert!(
            worst < 0.01,
            "the table is up to {worst} away from the integral, worst at {worst_at:?}"
        );
    }

    /// Three things about the table that no one bug breaks together.
    #[test]
    fn the_transmittance_table_reddens_thickens_and_clears() {
        let (table, _) = built_tables();
        let at = |x: u32, y: u32| table[(y * TRANSMITTANCE_SIZE.x + x) as usize];

        // Blue is scattered out harder than red, which is the whole reason a
        // low sun is red. Never the other way round anywhere in the table --
        // and equality is allowed, because near the top of the atmosphere there
        // is so little air left that all three channels round to the same half
        // float. That is the format doing its job rather than the physics
        // failing: 0.9995117 is what sixteen bits has just below one.
        for y in 0..TRANSMITTANCE_SIZE.y {
            for x in 0..TRANSMITTANCE_SIZE.x {
                let here = at(x, y);
                assert!(
                    here.x >= here.z,
                    "at ({x}, {y}) the air takes more red than blue: {here}"
                );
            }
        }

        // ... and where there is enough air for the difference to survive the
        // format, it is not a rounding's worth but a landscape's. A ray leaving
        // the ground along the horizon is the longest path the table holds.
        let grazing = at(TRANSMITTANCE_SIZE.x - 1, 0);
        assert!(
            grazing.x > 20.0 * grazing.z.max(1e-9),
            "along the horizon at sea level the air passes {grazing}, which is \
             not the reddening a sunset is made of"
        );

        // At the top of the atmosphere looking up there is nothing left to
        // pass through.
        let top = at(0, TRANSMITTANCE_SIZE.y - 1);
        assert!(
            (top - Vec3::ONE).abs().max_element() < 1e-2,
            "looking up from the top of the atmosphere gives {top}, not 1"
        );

        // ... and tilting a ray towards the horizon only ever puts more air in
        // front of it. The columns run from the shortest path out of the
        // atmosphere to the longest, so every row must fall.
        for y in 0..TRANSMITTANCE_SIZE.y {
            for x in 1..TRANSMITTANCE_SIZE.x {
                let (before, after) = (at(x - 1, y), at(x, y));
                assert!(
                    after.z <= before.z + 1e-3,
                    "row {y} brightens from {before} to {after} between columns {} and {x}",
                    x - 1
                );
            }
        }
    }

    /// The multiple-scattering table is light, and more of it with the sun up.
    ///
    /// This table is the whole of the ambient term the ground is about to be
    /// lit by, so the things worth pinning are that it is not zero -- which
    /// would make every shaded slope black -- and that it tracks the sun, which
    /// is what says the sun angle reached the integral at all.
    #[test]
    fn the_multiple_scattering_table_follows_the_sun() {
        let (_, table) = built_tables();
        let at = |x: u32, y: u32| table[(y * MULTISCATTER_SIZE.x + x) as usize];

        for (index, value) in table.iter().enumerate() {
            assert!(
                value.min_element() >= 0.0 && value.max_element().is_finite(),
                "texel {index} of the scattering table is {value}"
            );
        }

        // Sea level, sun overhead against sun below the horizon. The columns
        // are the sun's cosine, running from -1 to 1.
        let overhead = at(MULTISCATTER_SIZE.x - 1, 0);
        let under = at(0, 0);
        assert!(
            overhead.length() > 0.05,
            "with the sun overhead the sky returns only {overhead}, which \
             would leave every shaded slope black"
        );
        assert!(
            under.length() < 1e-3,
            "with the sun below the horizon the sky still returns {under}"
        );

        // Multiply-scattered light is Rayleigh light that has bounced, so it
        // keeps the blue bias that put it there.
        assert!(
            overhead.z > overhead.x,
            "scattered light at sea level is {overhead}, which is not blue"
        );

        // And it rises with the sun, monotonically, along the whole row.
        for x in 1..MULTISCATTER_SIZE.x {
            let (before, after) = (at(x - 1, 0), at(x, 0));
            assert!(
                after.length() >= before.length() - 1e-4,
                "raising the sun from column {} to {x} dimmed the sky, {before} to {after}",
                x - 1
            );
        }
    }

    /// The raster never reaches the planet's horizon.
    ///
    /// The assumption the sky rests on. Terrain is drawn on a flat world and
    /// the air is integrated on a round one, and the two only stay consistent
    /// while no ground is far enough away to fall below the sphere's horizon --
    /// past that the model would be putting sky where there is ground.
    #[test]
    fn the_raster_never_reaches_the_planet_horizon() {
        // Half the diagonal of the installed survey, which is the furthest any
        // ground can be from a camera over the middle of it.
        let furthest = (98_304.0f32.powi(2) + 114_688.0f32.powi(2)).sqrt() * 0.5;
        for altitude in [500.0f32, 1500.0, 3000.0, 12_000.0] {
            let horizon = (2.0 * GROUND_RADIUS * altitude).sqrt();
            assert!(
                horizon > furthest,
                "from {altitude} m the horizon is {horizon:.0} m away, inside \
                 the {furthest:.0} m corner of the raster"
            );
        }
    }
}
