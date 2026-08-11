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

/// Mirrors the `Sky` uniform block in `src/sky.wgsl` and `src/shading.wgsl`.
///
/// A block of its own rather than more words on the camera, because the camera
/// is where the eye is and this is what the world is lit by: two different
/// things, written by two different parts of the frame.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniform {
    /// The unit vector pointing at the sun. `w` is unused padding; uniform
    /// members are aligned to sixteen bytes anyway.
    sun: [f32; 4],
    /// The eye in planet space, with its radius in `w`.
    ///
    /// Planet space is world space with the planet's centre at the origin, so
    /// this is the camera plus one ground radius of `y`. The radius is carried
    /// rather than recomputed because every table lookup wants it and a length
    /// of a six-million-metre vector is worth doing once, on the CPU, in double
    /// precision -- an `f32` holding 6.36e6 has metre-scale steps left.
    eye: [f32; 4],
    /// The local up at the eye, with the angle one pixel subtends in `w`.
    ///
    /// That angle is what feathers the sun's edge. It belongs here rather than
    /// being derived in the shader because it needs the field of view and the
    /// viewport height, and the fragment stage has neither.
    up: [f32; 4],
    /// The sun projected into the eye's tangent plane, normalised: where the
    /// sky-view table's azimuth is measured from.
    ///
    /// Built here rather than in the shader so the degenerate case is decided
    /// once. With the sun exactly overhead the projection vanishes and any
    /// direction in the tangent plane will do -- the sky is then circularly
    /// symmetric, so the choice cannot be seen.
    sun_tangent: [f32; 4],
}

impl SkyUniform {
    fn new(sun: Sun, eye: Vec3, pixel_angle: f32) -> Self {
        // Doubles for the one subtraction that needs them: the eye is metres
        // from a centre six thousand kilometres away, and an `f32` there has
        // steps of about half a metre.
        let centred = glam::DVec3::new(
            f64::from(eye.x),
            f64::from(eye.y) + f64::from(GROUND_RADIUS),
            f64::from(eye.z),
        );
        let radius = centred.length().max(f64::from(GROUND_RADIUS));
        let up = (centred / radius).as_vec3();
        let flat = sun.direction - up * sun.direction.dot(up);
        // Any tangent direction when the sun is overhead; see the field's doc.
        let tangent = if flat.length_squared() > 1e-12 {
            flat.normalize()
        } else {
            up.cross(Vec3::X)
                .try_normalize()
                .unwrap_or(up.cross(Vec3::Z).normalize())
        };
        Self {
            sun: sun.direction.extend(0.0).to_array(),
            eye: centred.as_vec3().extend(radius as f32).to_array(),
            up: up.extend(pixel_angle).to_array(),
            sun_tangent: tangent.extend(0.0).to_array(),
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
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub const GROUND_RADIUS: f32 = 6_360_000.0;
/// The top of the atmosphere, a hundred kilometres up.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
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

/// Size of the sky-view table: the sky in every direction, for this frame's
/// eye altitude and sun.
///
/// The first table that cannot be precomputed. It is a raymarch from a
/// particular altitude with the sun in a particular place, and baking those two
/// in as extra axes is what would make it four-dimensional -- 3.4 GB at a
/// resolution that keeps the horizon crowding, and a shippable 16 MB only by
/// throwing that crowding away and banding every sunset. Marching 192 x 108 x 30
/// samples once a frame is cheaper than either.
///
/// 192 across is 1.875 degrees a texel of azimuth; 108 down is the same aspect
/// as the window, which is not required but keeps the two comparable.
pub const SKYVIEW_SIZE: glam::UVec2 = glam::UVec2::new(192, 108);

/// Size of the aerial-perspective volume: the air in front of every part of the
/// frame, sliced by distance.
///
/// Two horizontal axes that *are* the camera's frustum, so this one could not
/// be precomputed even in principle. It is less a lookup table than a cache of
/// this frame at a fraction of its resolution: 32 x 32 x 64 marches standing in
/// for a per-pixel raymarch at 1280 x 720, which is a hundredfold saving and
/// the reason the shading pass can ask what a hundred kilometres of air did
/// with two filtered fetches.
pub const AERIAL_SIZE: glam::UVec3 = glam::UVec3::new(32, 32, 64);

/// How far that volume reaches along the view axis, in metres.
///
/// Hillaire's is 32 km. This survey is 115 km across and a ridge at eighty of
/// them is exactly what the haze is for, so the volume has to reach past it.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub const AERIAL_FAR: f32 = 100_000.0;

/// Half the sun's apparent width, in radians: 0.5334 degrees across, which is
/// what it is from this planet. Must match `src/shading.wgsl`.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub const SUN_ANGULAR_RADIUS: f32 = 0.004654;

/// One over the disc's solid angle, `2 pi (1 - cos r)`.
///
/// The sun is given as an irradiance and everything else here is a radiance, so
/// this is what turns one into the other: the whole of its light spread over
/// the small patch of sky it occupies. Must match `src/shading.wgsl`.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub const SUN_DISC_RADIANCE: f32 = 14696.0;

/// The angle one pixel subtends at the centre of the frame, in radians.
///
/// What feathers the sun's edge, and the same quantity the clipmap sizes its
/// windows by -- see `pixel_angle` in `crate::terrain::residency`, which is the
/// small-angle form of this. The tangent is used here because the sun is
/// measured in fractions of a degree and the two forms differ by a tenth at
/// sixty degrees of field.
pub fn pixel_angle(fov_y: f32, height: u32) -> f32 {
    2.0 * (fov_y * 0.5).tan() / height.max(1) as f32
}

/// Steps the shader integrates the optical depth in. Must match `src/sky.wgsl`.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
const TRANSMITTANCE_STEPS: u32 = 40;

/// How much display brightness a unit of radiance is worth.
///
/// Everything the tables hold is in units of the sun's irradiance at the top of
/// the atmosphere, taken as one per channel. That fixes the scale of the model
/// but says nothing about the screen, and this is the number that connects the
/// two. Must match `EXPOSURE` in `src/shading.wgsl`.
///
/// Derived rather than dialled in. Level ground of albedo `a` under the
/// reference 45-degree sun at sea level comes out at
///
///   `L = a/pi * (T(45) * cos 45 + pi * psi_ms) ~= a/pi * (0.76 * 0.707 + 0.12)`
///
/// which is about `0.209 a`. The light this replaces was `0.35 + 0.65 cos 45`,
/// or `0.81 a`, and solving `tonemap(E * 0.209 a) = 0.81 a` gives five. So the
/// change of lighting model is a change of *behaviour* -- where the light comes
/// from and what colour it is -- rather than a change of overall brightness,
/// which is what makes the two frames comparable at all.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub const EXPOSURE: f32 = 5.0;

/// The radiance that maps to white. Must match `WHITE` in `src/shading.wgsl`.
///
/// 1.6 in unexposed units. Sunlit snow reaches about 0.3 and the sun's own disc
/// is four orders of magnitude above that, so the disc clips to white and
/// nothing else in the frame does -- which is the only thing a white point has
/// to get right.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub const WHITE: f32 = 8.0;

/// The shader's tonemap, in Rust.
///
/// Extended Reinhard, per channel. Not used by the shader -- there is no
/// preprocessor -- but pinned to it by a test, and here so that a test can
/// predict a byte from a radiance instead of restating the shader's arithmetic
/// as its own expectation. That is why the curve is this one and not a fitted
/// ACES: it inverts in closed form.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub fn tonemap(radiance: Vec3) -> Vec3 {
    let x = radiance * EXPOSURE;
    (x * (Vec3::ONE + x / (WHITE * WHITE)) / (Vec3::ONE + x)).clamp(Vec3::ZERO, Vec3::ONE)
}

/// [`tonemap`] backwards: the radiance a displayed value came from.
///
/// This is what the closed form buys, and it is what a test uses to measure the
/// light in a rendered frame rather than assume it. Solving
/// `y = x (1 + x/W^2) / (1 + x)` for `x` is one quadratic:
///
///   `x^2 / W^2 + x (1 - y) - y = 0`
///
/// whose positive root is the line below. Exact at both ends -- `y = 1` gives
/// `W`, and near zero it is the identity, which is what makes the curve usable
/// for dark ground in the first place.
///
/// Undefined above one, where the curve has clipped and the radiance that made
/// a pixel is no longer recoverable from it.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub fn untonemap(displayed: Vec3) -> Vec3 {
    let white = WHITE * WHITE;
    let root = ((Vec3::ONE - displayed) * (Vec3::ONE - displayed) + displayed * (4.0 / white))
        .max(Vec3::ZERO);
    let exposed = (displayed - Vec3::ONE + root.powf(0.5)) * (white * 0.5);
    exposed / EXPOSURE
}

/// The sky uniform, the scattering tables, and everything that fills them.
///
/// Group 1 is the uniform and group 2 is the tables, wherever either is bound.
/// Group 0 stays the camera, as it is for every other pipeline in the program.
pub struct Sky {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    /// The two tables themselves, rather than views of them.
    ///
    /// A frame says almost nothing about what is in a table -- the shading
    /// reduces both to a single colour -- so the only way to check either is to
    /// copy it back and read the values, and a copy needs the texture. The same
    /// reasoning as `Targets` on the G-buffer.
    #[allow(dead_code, reason = "read only by the table readback tests")]
    transmittance: wgpu::Texture,
    #[allow(dead_code, reason = "read only by the table readback tests")]
    multiscatter: wgpu::Texture,
    #[allow(dead_code, reason = "read only by the table readback tests")]
    skyview: wgpu::Texture,
    #[allow(dead_code, reason = "read only by the table readback tests")]
    aerial_scatter: wgpu::Texture,
    #[allow(dead_code, reason = "read only by the table readback tests")]
    aerial_transmit: wgpu::Texture,
    aerial_build: wgpu::ComputePipeline,
    write_aerial: wgpu::BindGroup,
    /// The per-frame build: rebuilt every frame, so it is kept rather than
    /// dropped the way the two one-off builds are.
    skyview_build: wgpu::ComputePipeline,
    /// Group 2 for that build: the sampler and the two finished tables, and
    /// deliberately not the sky-view table it is writing. See [`Build`].
    read_tables: wgpu::BindGroup,
    write_skyview: wgpu::BindGroup,
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
    /// `camera_layout` is the scene's, and only the aerial-perspective build
    /// wants it: its volume is the camera's own frustum. Every other pass here
    /// leaves group 0 empty.
    pub fn new(device: &wgpu::Device, camera_layout: &wgpu::BindGroupLayout) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky uniform"),
            size: std::mem::size_of::<SkyUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            layout: &sky_layout,
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
        let skyview = table("sky view table", SKYVIEW_SIZE);
        let transmittance_view = transmittance.create_view(&Default::default());
        let multiscatter_view = multiscatter.create_view(&Default::default());
        let skyview_view = skyview.create_view(&Default::default());

        let volume = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: AERIAL_SIZE.x,
                    height: AERIAL_SIZE.y,
                    depth_or_array_layers: AERIAL_SIZE.z,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: LUT_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let aerial_scatter = volume("aerial scattering volume");
        let aerial_transmit = volume("aerial transmittance volume");
        let aerial_scatter_view = aerial_scatter.create_view(&Default::default());
        let aerial_transmit_view = aerial_transmit.create_view(&Default::default());

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
        // The sky-view table's azimuth runs the whole way round, so it has to
        // wrap or the seam behind the sun draws as a line. Its own sampler
        // rather than wrapping the shared one, because every other table has an
        // axis that genuinely stops and would then read the far end of itself.
        let wrapping = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky view sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sampler_at = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let volume_at = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };
        let tables_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scattering tables layout"),
            entries: &[
                sampler_entry,
                sampled(1),
                sampled(2),
                sampler_at(3),
                sampled(4),
                volume_at(5),
                volume_at(6),
            ],
        });
        let sampler_binding = wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Sampler(&sampler),
        };
        let transmittance_binding = wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&transmittance_view),
        };
        let multiscatter_binding = wgpu::BindGroupEntry {
            binding: 2,
            resource: wgpu::BindingResource::TextureView(&multiscatter_view),
        };
        let tables_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scattering tables bind group"),
            layout: &tables_layout,
            entries: &[
                sampler_binding.clone(),
                transmittance_binding.clone(),
                multiscatter_binding.clone(),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&wrapping),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&skyview_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&aerial_scatter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&aerial_transmit_view),
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
            entries: &[sampler_binding.clone(), transmittance_binding.clone()],
        });

        // The same, for the sky-view build: both finished tables and neither of
        // the table it writes.
        let read_tables_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("finished tables read layout"),
                entries: &[sampler_entry, sampled(1), sampled(2)],
            });
        let read_tables = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("finished tables read bind group"),
            layout: &read_tables_layout,
            entries: &[sampler_binding, transmittance_binding, multiscatter_binding],
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
        let write_skyview_layout = write_layout("sky view write layout", 2);
        let write_volume = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: LUT_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };
        let write_aerial_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aerial write layout"),
                entries: &[write_volume(3), write_volume(4)],
            });
        let write_aerial = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("aerial write bind group"),
            layout: &write_aerial_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&aerial_scatter_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&aerial_transmit_view),
                },
            ],
        });

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

        // Built before the struct literal, because the literal moves
        // `sky_layout` into the `layout` field and this borrows it.
        let aerial_build = stage(
            "aerial perspective build",
            "cs_aerial",
            &[
                Some(camera_layout),
                Some(&sky_layout),
                Some(&read_tables_layout),
                Some(&write_aerial_layout),
            ],
        );
        let skyview_build = stage(
            "sky view build",
            "cs_skyview",
            &[
                None,
                Some(&sky_layout),
                Some(&read_tables_layout),
                Some(&write_skyview_layout),
            ],
        );

        Self {
            buffer,
            layout: sky_layout,
            bind_group,
            tables_layout,
            tables_bind_group,
            skyview_build,
            aerial_build,
            write_aerial,
            read_tables,
            write_skyview: write_bind(
                "sky view write bind group",
                &write_skyview_layout,
                2,
                &skyview_view,
            ),
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
            skyview,
            aerial_scatter,
            aerial_transmit,
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

    /// Uploads where the sun and the eye are, for the frame about to be drawn.
    pub fn set_frame(&self, queue: &wgpu::Queue, sun: Sun, eye: Vec3, pixel_angle: f32) {
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::bytes_of(&SkyUniform::new(sun, eye, pixel_angle)),
        );
    }

    /// Records the per-frame table builds into an already-started pass.
    ///
    /// The sky-view table depends on this frame's eye altitude and sun and on
    /// nothing the frame produces, so it is built before anything else in
    /// `Scene::draw` rather than fitted around the march.
    pub fn draw_sky_view(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.skyview_build);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_bind_group(2, &self.read_tables, &[]);
        pass.set_bind_group(3, &self.write_skyview, &[]);
        pass.dispatch_workgroups(SKYVIEW_SIZE.x.div_ceil(8), SKYVIEW_SIZE.y.div_ceil(8), 1);
    }

    /// The aerial-perspective volume. The caller has set group 0 to the camera,
    /// which this one needs and the sky-view build does not.
    ///
    /// One thread per froxel column rather than per froxel: a thread per froxel
    /// would march from the eye again for every slice.
    pub fn draw_aerial(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.aerial_build);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_bind_group(2, &self.read_tables, &[]);
        pass.set_bind_group(3, &self.write_aerial, &[]);
        pass.dispatch_workgroups(AERIAL_SIZE.x.div_ceil(8), AERIAL_SIZE.y.div_ceil(8), 1);
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

    #[cfg(test)]
    pub fn sky_view(&self) -> &wgpu::Texture {
        &self.skyview
    }
}

/// The half-texel correction, and its inverse. Mirrors `src/sky.wgsl`.
///
/// A table of `n` texels has its first centre at `0.5/n` and its last at
/// `1 - 0.5/n`, so a parameter running the full range has to be squeezed into
/// that span. Skipping it is the classic failure of this technique and it fails
/// quietly -- the picture looks nearly right, with the horizon about a degree
/// out. Written here as well as in the shader so a test can round-trip it.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub fn to_texture(x: f32, n: f32) -> f32 {
    0.5 / n + x * (1.0 - 1.0 / n)
}

#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub fn to_unit(u: f32, n: f32) -> f32 {
    (u - 0.5 / n) / (1.0 - 1.0 / n)
}

/// Distance to the top of the atmosphere. Mirrors `top_distance` in the shader.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
pub fn top_distance(r: f32, mu: f32) -> f32 {
    let discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
    (-r * mu + discriminant.max(0.0).sqrt()).max(0.0)
}

/// Where `(r, mu)` sits in the transmittance table. Mirrors the shader.
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
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
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
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
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
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
#[allow(
    dead_code,
    reason = "the shader's mirror; the tests are the only Rust caller"
)]
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
        let mut sky = Sky::new(&device, &crate::scene::test_camera_layout(&device));
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

    /// The two shaders map the sky the same way.
    ///
    /// `skyview_v` is written twice by hand -- once in `src/sky.wgsl`, which
    /// builds the table, and once in `src/shading.wgsl`, which reads it. If the
    /// two parted, every direction would be looked up in the wrong row and the
    /// result would still be a smooth, plausible gradient: the horizon would
    /// simply be somewhere else.
    ///
    /// Nothing numeric catches that, and this was checked rather than assumed.
    /// Making the build's mapping linear where the fetch's crowds towards the
    /// horizon leaves every test in the suite passing, because the table is
    /// still monotone, its horizon is still at the row the mapping puts it, and
    /// the frame is still darker overhead than low down. So the text is what
    /// there is to compare, the same defence the half-texel correction gets.
    #[test]
    fn both_shaders_map_the_sky_the_same_way() {
        let body = |source: &str, name: &str| {
            let start = source
                .find(&format!("fn {name}("))
                .unwrap_or_else(|| panic!("no {name}"));
            let end = source[start..].find("\n}").expect("unterminated");
            source[start..start + end]
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        };
        let build = include_str!("sky.wgsl");
        let shade = include_str!("shading.wgsl");
        for name in ["horizon_zenith", "skyview_v"] {
            assert_eq!(
                body(build, name),
                body(shade, name),
                "{name} differs between src/sky.wgsl and src/shading.wgsl"
            );
        }
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

    /// The tonemap and its inverse are inverses.
    ///
    /// Worth pinning because a test measures the light in a rendered frame by
    /// running the curve backwards, and a wrong root would make that
    /// measurement quietly plausible rather than obviously broken.
    #[test]
    fn the_tonemap_inverts() {
        for radiance in [0.0f32, 1e-4, 0.01, 0.1, 0.209, 0.5, 1.0, 1.5] {
            let there = tonemap(Vec3::splat(radiance));
            let back = untonemap(there);
            assert!(
                (back.x - radiance).abs() < 1e-3 * radiance.max(1e-3),
                "{radiance} tonemaps to {there} and back to {back}"
            );
        }
        // The white point is the number it claims to be, and nothing below it
        // has clipped.
        assert!((tonemap(Vec3::splat(WHITE / EXPOSURE)).x - 1.0).abs() < 1e-5);
        assert!(tonemap(Vec3::splat(WHITE / EXPOSURE * 0.99)).x < 1.0);
    }

    /// The shader tonemaps the way Rust says it does.
    ///
    /// Two copies of one curve, and a test measures a frame through the Rust
    /// one. If they parted the measurement would be of the wrong curve.
    #[test]
    fn the_shading_shader_tonemaps_the_way_rust_does() {
        let source = include_str!("shading.wgsl");
        for (name, value) in [("EXPOSURE", EXPOSURE), ("WHITE", WHITE)] {
            let declaration = format!("const {name}: f32 = {value:.1};");
            assert!(
                source.contains(&declaration),
                "src/shading.wgsl does not declare {declaration}"
            );
        }
        assert!(
            source.contains("saturate(x * (1.0 + x / (WHITE * WHITE)) / (1.0 + x))"),
            "src/shading.wgsl's tonemap is not the curve src/sky.rs mirrors"
        );
    }

    /// The sky-view table puts the horizon exactly down its middle.
    ///
    /// The whole point of the vertical mapping, and the thing that fails
    /// quietly if it is wrong: a table whose horizon sat a few rows off would
    /// still look like a sky, with the pale band in slightly the wrong place.
    /// Found by looking for the row of steepest change rather than by trusting
    /// the formula, which is the same formula the shader used to build it.
    #[test]
    fn the_sky_view_table_puts_the_horizon_down_the_middle() {
        let (_, _, table) = built_sky_view(3000.0);
        let at = |x: u32, y: u32| table[(y * SKYVIEW_SIZE.x + x) as usize];

        // Straight away from the sun, where nothing but the horizon is
        // happening: the aureole would otherwise be the steepest thing in the
        // column.
        let column = 0;
        let (mut steepest, mut steepest_row) = (0.0f32, 0u32);
        for y in 1..SKYVIEW_SIZE.y {
            let change = (at(column, y) - at(column, y - 1)).length();
            if change > steepest {
                steepest = change;
                steepest_row = y;
            }
        }
        let middle = SKYVIEW_SIZE.y / 2;
        assert!(
            steepest_row.abs_diff(middle) <= 1,
            "the sky changes fastest at row {steepest_row} of {}, not at the \
             horizon in the middle at {middle}",
            SKYVIEW_SIZE.y
        );

        // And the sky is blue overhead and paler at the horizon, which is the
        // one fact about a daytime sky everyone can check by looking up.
        let zenith = at(column, 0);
        let horizon = at(column, middle - 2);
        assert!(
            zenith.z / zenith.x > horizon.z / horizon.x,
            "the zenith is {zenith} and the horizon {horizon}, which is not a \
             sky that gets paler as it comes down"
        );
    }

    /// Climbing lowers the horizon, and by how much.
    ///
    /// The geometry the sky-view table's vertical mapping is built around, and
    /// the thing a stack of flat slabs could not produce at all: on a flat
    /// world the horizon is level from every altitude, so the sky would meet
    /// the ground in the same place from a hilltop as from orbit.
    ///
    /// The table cannot show this and it is worth saying why: the mapping puts
    /// the horizon exactly half way down at every altitude, which is the point
    /// of it. What moves is the *angle* that row stands for, which is what this
    /// measures.
    #[test]
    fn climbing_lowers_the_horizon() {
        // `horizon_zenith` in `src/sky.wgsl`, mirrored: the angle from straight
        // up to the horizon, a right angle on the ground and more above it.
        let dip_degrees = |altitude: f32| {
            let r = GROUND_RADIUS + altitude;
            let zenith = std::f32::consts::PI
                - (GROUND_RADIUS / r.max(GROUND_RADIUS))
                    .clamp(-1.0, 1.0)
                    .asin();
            (zenith - std::f32::consts::FRAC_PI_2).to_degrees()
        };
        assert!(
            dip_degrees(0.0).abs() < 1e-3,
            "on the ground the horizon should be level, and it dips {}",
            dip_degrees(0.0)
        );
        // Against the small-angle form, `sqrt(2h/R)` radians, which is a
        // genuinely separate derivation rather than the arcsine restated: it
        // comes from the tangent-line right triangle instead of from the
        // sphere's own geometry, and the two only agree if both are right.
        for altitude in [500.0f32, 1500.0, 3000.0, 12_000.0] {
            let got = dip_degrees(altitude);
            let want = (2.0 * altitude / GROUND_RADIUS).sqrt().to_degrees();
            assert!(
                (got - want).abs() < 0.01,
                "from {altitude} m the horizon dips {got} degrees where the \
                 small-angle form says {want}"
            );
            assert!(
                got > dip_degrees(altitude * 0.5),
                "climbing has to lower the horizon, not raise it"
            );
        }
        // And the magnitude, so this cannot all agree on nothing: a degree and
        // a quarter from 1500 m, three and a half from twelve kilometres.
        assert!((dip_degrees(1500.0) - 1.244).abs() < 0.01);
        assert!((dip_degrees(12_000.0) - 3.520).abs() < 0.01);
    }

    /// Builds the sky-view table for an eye at `altitude`, and reads it back.
    ///
    /// Returns the eye's radius alongside, because half of what the table means
    /// is which radius it was built for.
    fn built_sky_view(altitude: f32) -> (f32, f32, Vec<Vec3>) {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let mut sky = Sky::new(&device, &crate::scene::test_camera_layout(&device));
        sky.ensure_built(&device, &queue);
        let eye = Vec3::new(0.0, altitude, 0.0);
        sky.set_frame(
            &queue,
            Sun::default(),
            eye,
            pixel_angle(60f32.to_radians(), 720),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sky view"),
                timestamp_writes: None,
            });
            sky.draw_sky_view(&mut pass);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let radius = f64::from(altitude) + f64::from(GROUND_RADIUS);
        (
            radius as f32,
            altitude,
            read_table(&device, &queue, sky.sky_view(), SKYVIEW_SIZE),
        )
    }

    /// The aerial-perspective volume really does reach a hundred kilometres.
    ///
    /// The test for the one place this departs from the paper on range rather
    /// than on method. Left at Hillaire's 32 km the last slice would hold three
    /// times less air, and the blue channel -- which is what the distance does
    /// most to -- would come out several times too bright. So the figures here
    /// are absolute, not merely ordered.
    #[test]
    fn the_aerial_volume_reaches_a_hundred_kilometres() {
        let volume = built_aerial();
        // The middle column of the volume, which is the middle of the frame:
        // the camera below looks level, so this is a horizontal ray through the
        // thickest air the model has.
        let at = |slice: u32| {
            let (x, y) = (AERIAL_SIZE.x / 2, AERIAL_SIZE.y / 2);
            volume[((slice * AERIAL_SIZE.y + y) * AERIAL_SIZE.x + x) as usize]
        };

        // Nothing has happened in the first slice: it ends 24 m out.
        let first = at(0);
        assert!(
            (first.transmittance - Vec3::ONE).abs().max_element() < 1e-2,
            "the first slice already passes only {}, and it is 24 m deep",
            first.transmittance
        );
        assert!(
            first.scattered.max_element() < 1e-3,
            "the first slice has already scattered {}",
            first.scattered
        );

        // Transmittance only ever falls, and in-scattering only ever rises: the
        // volume stores a running integral, so anything else is a bug in the
        // accumulation rather than in the physics.
        for slice in 1..AERIAL_SIZE.z {
            let (before, now) = (at(slice - 1), at(slice));
            assert!(
                now.transmittance.max_element() <= before.transmittance.max_element() + 1e-3,
                "slice {slice} passes more light than slice {}: {} then {}",
                slice - 1,
                before.transmittance,
                now.transmittance
            );
            assert!(
                now.scattered.length() >= before.scattered.length() - 1e-3,
                "slice {slice} scattered less than slice {}",
                slice - 1
            );
        }

        // And at the far end, a hundred kilometres of air has taken the blue
        // out far harder than the red. Both figures matter: the ratio says the
        // colour is right, and the absolute says the distance is.
        let far = at(AERIAL_SIZE.z - 1);
        println!(
            "100 km: transmittance {}, scattered {}",
            far.transmittance, far.scattered
        );
        // Both bounds are set from measurement rather than guessed, and both
        // separate this volume from the paper's 32 km one. Along the same
        // level ray from 2000 m the last slice passes
        //
        //   100 km: red 0.529, blue 0.053 -- a ratio of 10.0
        //    32 km: red 0.833, blue 0.412 -- a ratio of  2.0
        //
        // so either check alone fails if the range is cut back, and the pair
        // says the colour and the distance are both right.
        assert!(
            far.transmittance.x > 4.0 * far.transmittance.z,
            "over a hundred kilometres the air passes {}, a red-to-blue ratio \
             of {:.1} where the distance should give ten",
            far.transmittance,
            far.transmittance.x / far.transmittance.z
        );
        assert!(
            far.transmittance.z < 0.15,
            "a hundred kilometres of air still passes {} of the blue, where it \
             should pass a twentieth; 32 km would pass 0.41",
            far.transmittance.z
        );
    }

    /// One froxel: the light the air put in, and what it let through.
    #[derive(Clone, Copy)]
    struct Froxel {
        scattered: Vec3,
        transmittance: Vec3,
    }

    /// Builds the aerial volume for a level camera at 2 km and reads it back.
    fn built_aerial() -> Vec<Froxel> {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let mut sky = Sky::new(&device, &crate::scene::test_camera_layout(&device));
        sky.ensure_built(&device, &queue);

        // A camera looking level, so the volume's middle column is the longest
        // path through the atmosphere the model holds.
        let camera = crate::camera::Camera::new(
            Vec3::new(0.0, 2000.0, 0.0),
            crate::camera::Camera::from_yaw_pitch_roll(0.0, 0.0, 0.0),
            16.0 / 9.0,
        );
        let (_, group) = crate::scene::test_camera(&device, &queue, &camera);
        sky.set_frame(
            &queue,
            Sun::default(),
            camera.position,
            pixel_angle(camera.fov_y, 720),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("aerial"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &group, &[]);
            sky.draw_aerial(&mut pass);
        }
        queue.submit(std::iter::once(encoder.finish()));

        let scattered = read_volume(&device, &queue, &sky.aerial_scatter);
        let transmittance = read_volume(&device, &queue, &sky.aerial_transmit);
        scattered
            .into_iter()
            .zip(transmittance)
            .map(|(scattered, transmittance)| Froxel {
                scattered,
                transmittance,
            })
            .collect()
    }

    /// Copies a 3D table off the GPU as RGB triples, slice by slice.
    fn read_volume(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Vec<Vec3> {
        let bytes_per_row = AERIAL_SIZE.x * 8;
        assert_eq!(bytes_per_row % 256, 0, "a row of the volume needs padding");
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("volume readback"),
            size: u64::from(bytes_per_row * AERIAL_SIZE.y * AERIAL_SIZE.z),
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
                    rows_per_image: Some(AERIAL_SIZE.y),
                },
            },
            wgpu::Extent3d {
                width: AERIAL_SIZE.x,
                height: AERIAL_SIZE.y,
                depth_or_array_layers: AERIAL_SIZE.z,
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

    /// The slice distances are the ones the comments claim.
    ///
    /// Cheap, and it is what says the quadratic distribution is a distribution
    /// rather than a formula that happens to end in the right place: uniform
    /// slices over the same range would put slice 15 at 25 km, not 6.25.
    #[test]
    fn the_aerial_slices_crowd_towards_the_eye() {
        let ends = |slice: u32| {
            AERIAL_FAR * {
                let w = (slice + 1) as f32 / AERIAL_SIZE.z as f32;
                w * w
            }
        };
        for (slice, want) in [
            (0u32, 24.4f32),
            (15, 6250.0),
            (31, 25_000.0),
            (63, 100_000.0),
        ] {
            let got = ends(slice);
            assert!(
                (got - want).abs() < want * 0.01,
                "slice {slice} ends at {got} m, not {want}"
            );
        }
        // The near slices are far finer than uniform slicing would give, which
        // is the whole reason for the mapping.
        let uniform = AERIAL_FAR / AERIAL_SIZE.z as f32;
        assert!(
            ends(0) < uniform / 50.0,
            "the first slice is {} m against {uniform} m if they were even",
            ends(0)
        );
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
