//! The wind, solved around the mountains once and then read for the rest of the
//! run.
//!
//! A coarse box over the whole raster holds one velocity per cell. The terrain
//! is an obstacle in it, so what comes out is not the uniform drift a constant
//! would give: air piles against a windward slope, accelerates over a ridge and
//! sinks into the lee behind it. Cloud is then displaced by that field rather
//! than by a direction and a speed, which is what makes a cap sit on a peak and
//! a valley run clear.
//!
//! # Why it is baked
//!
//! The alternative is to step the solve every frame, and it was considered and
//! rejected. A relaxation-and-projection solve reaches a steady state and stays
//! there: the free stream is already divergence free, the only divergence
//! generated is the terrain blocking the flow, and once that has propagated the
//! field stops changing. Paying for it every frame would buy a static answer
//! sixty times a second. Worse, it would have to ping-pong two velocity
//! textures inside [`crate::scene::Scene::draw`], which takes `&self` and
//! therefore cannot flip an index, and it would make the frame depend on how
//! many steps had run -- so `render` would stop being reproducible and
//! `Scene::settle` would silently age the weather.
//!
//! Baking answers all of it. The solve runs at load, long enough to converge
//! properly, and the frame reads two textures. What the frame loses is any
//! response to the weather changing, which nothing yet asks for; a change of
//! `--wind` re-bakes.
//!
//! # What it is not
//!
//! Not a forecast and not an aerodynamic model. There is no temperature, no
//! humidity, no Coriolis force and no time. It is an incompressible steady flow
//! around a fixed obstacle, which is the smallest thing that produces the three
//! effects cloud needs: blocking, channelling, and lift over a ridge.

use glam::{UVec2, Vec3};
use wgpu::util::DeviceExt;

use crate::terrain::gpu::Terrain;

/// Cells across the grid, in x, up, and z. Must match `CELLS_X`, `CELLS_Y` and
/// `CELLS_Z` in `src/air.wgsl`.
///
/// A hundred and sixty across the raster puts a cell at six hundred metres on
/// the survey and three hundred on the generated one, which is coarse against a
/// mountain and fine against a cloud. Twenty layers to seven kilometres is the
/// other half of that trade; see `TOP_METRES` in the shader for why they are
/// level rather than distributed.
pub const CELLS: [u32; 3] = [160, 20, 160];

/// How high the solved air reaches, in metres. Must match `TOP_METRES` in
/// `src/air.wgsl` and `AIR_TOP` in `src/cloud_march.wgsl`.
#[allow(dead_code, reason = "read by the tests and mirrored into two shaders")]
pub const TOP_METRES: f32 = 7000.0;

/// Metres per layer. Must match `CELL_Y` in `src/air.wgsl`, which derives it
/// the same way rather than spelling the quotient out.
#[allow(dead_code, reason = "read by the tests and by the cloud march to come")]
pub const CELL_Y: f32 = TOP_METRES / CELLS[1] as f32;

/// How many steps a parcel is followed back in. Must match `DRIFT_STEPS` in
/// `src/air.wgsl`, where it is a loop bound and so cannot be a uniform.
#[allow(dead_code, reason = "read only by the test comparing the two copies")]
const DRIFT_STEPS: u32 = 20;

/// Roughness length of the ground, in metres.
///
/// One metre is open country with hedges and scattered trees. It and the three
/// below are handed to the shader in the uniform rather than written out again
/// beside it: a knob that only the shader reads has no reason to exist twice,
/// where the grid's *shape* does because it decides how big a texture is.
const ROUGHNESS: f32 = 1.0;

/// The height by which the wind has reached its free-stream speed, in metres.
///
/// A logarithmic boundary layer: nothing at the ground, full strength by here.
/// There is no Ekman turning -- the real boundary layer also swings direction
/// with height, by up to thirty degrees, and leaving it out costs a rotation
/// nobody looking at cloud can see.
const BOUNDARY_METRES: f32 = 1500.0;

/// How long the flow takes to be pulled back towards the wind aloft, in
/// seconds.
///
/// Long against the half-minute step and short against the two and a half hours
/// the bake runs for, so the interior settles to the free stream wherever the
/// ground is not in the way and to something else wherever it is.
const RELAX_SECONDS: f32 = 300.0;

/// How far back along its own streamline a parcel is followed, in seconds.
///
/// Long enough for a ridge to have visibly stretched what crossed it, short
/// enough that the offset stays a perturbation rather than growing without
/// bound.
pub const DRIFT_TAU: f32 = 90.0;

/// Steps of the solve, and how long each covers in seconds.
///
/// Three hundred half-minute steps is two and a half hours of weather, which is
/// what it takes for air entering one side of a hundred-kilometre raster at
/// twenty metres a second to cross it and for the wake behind every ridge to
/// settle. Semi-Lagrangian advection is unconditionally stable, so the step is
/// chosen by how fast the answer should settle rather than by the cell size --
/// a Courant number well over ten is not a problem here, it is the point.
const STEPS: u32 = 300;
const STEP_SECONDS: f32 = 30.0;

/// Red-and-black sweeps of the Poisson solve per step.
///
/// Four, which is far short of convergence for one step and ample across three
/// hundred: the pressure buffer is never cleared, so each step starts from the
/// last step's answer and the solve is continued rather than restarted. Four
/// sweeps propagate information eight cells, which is the scale of the thing
/// generating the divergence -- one mountain. The large-scale divergence is
/// never removed and does not need to be: the free stream this relaxes towards
/// is already divergence free.
///
/// Measured over the mesa `the_baked_wind_never_leaves_the_grid_divergent`
/// solves, as the residual divergence left in air, in reciprocal seconds:
///
/// | sweeps | worst    | mean     |
/// |--------|----------|----------|
/// | 4      | 1.071e-2 | 6.65e-5  |
/// | 8      | 1.067e-2 | 6.57e-5  |
/// | 16     | 1.061e-2 | 6.56e-5  |
/// | 32     | 1.052e-2 | 6.56e-5  |
///
/// Eight times the sweeps buys 1.4% off the mean, so the floor is not
/// under-relaxation: it is the collocated grid's odd-even decoupling. A
/// gradient taken across two cells cannot see a checkerboard, so the projection
/// cannot remove a divergence that lives at the grid scale, and no number of
/// sweeps will change that. Four is where this sits.
const SWEEPS: u32 = 4;

/// Steps recorded into one command buffer before it is handed to the queue.
///
/// A recorded compute pass costs host memory until it is submitted, and the
/// solve records eleven a step. Thirty-two keeps a chunk to a few hundred
/// passes. See the note at the submit itself for what the unchunked version
/// cost.
const SUBMIT_EVERY: u32 = 32;

/// Threads per workgroup, in each axis. Must match `@workgroup_size` on every
/// entry point in `src/air.wgsl`.
const GROUP: u32 = 4;

/// The format the two solved fields are held in.
///
/// `Rgba16Float` because it is the only core format that is both
/// storage-writable and filterable, and the solve needs both: the advection
/// samples the velocity between cells, and the frame will too. Three of the
/// four channels carry a vector and the fourth carries the rise; nothing is
/// wasted. See `LUT_FORMAT` in `src/sky.rs`, which lands on the same format
/// after the same reasoning.
const FIELD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Where the wind comes from and how hard it blows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wind {
    /// Metres per second, in the free stream above the boundary layer.
    pub speed: f32,
    /// The compass bearing the wind blows *from*, in degrees.
    ///
    /// Meteorology's convention, not geometry's: a westerly is `270` and blows
    /// towards the east. It reads backwards to anyone who has not met it and is
    /// what every forecast, windsock and runway number in the world uses, so a
    /// flight simulator that reversed it would be the thing that was wrong.
    pub from_degrees: f32,
}

impl Wind {
    /// The velocity vector this names, in world space.
    ///
    /// A bearing is measured from north through east, and world space is Y-up
    /// with +X east and -Z north -- the same frame [`crate::sky::Sun`] reads
    /// its azimuth in. Blowing *from* the bearing is what turns the sine and
    /// cosine round from the sun's.
    pub fn velocity(self) -> Vec3 {
        let from = self.from_degrees.to_radians();
        Vec3::new(-from.sin(), 0.0, from.cos()) * self.speed
    }
}

impl std::str::FromStr for Wind {
    type Err = anyhow::Error;

    /// Reads `SPEED,BEARING`, the way a forecast says it.
    ///
    /// Parsed by the type rather than by a closure at the flag, for the reason
    /// [`crate::headless::SunAngles`] is: the reading is where the message that
    /// tells someone what they typed wrong lives.
    fn from_str(text: &str) -> anyhow::Result<Self> {
        use anyhow::{Context, bail};
        let numbers: Vec<f32> = text
            .split(',')
            .map(|part| {
                part.trim()
                    .parse()
                    .with_context(|| format!("{part:?} is not a number"))
            })
            .collect::<anyhow::Result<_>>()?;
        let [speed, from_degrees] = numbers[..] else {
            bail!(
                "expected SPEED,BEARING -- two numbers, got {}",
                numbers.len()
            );
        };
        if speed < 0.0 {
            bail!("a wind speed of {speed} is not a direction, it is a mistake");
        }
        Ok(Self {
            speed,
            from_degrees,
        })
    }
}

impl Default for Wind {
    /// A moderate breeze from the west.
    ///
    /// Ten metres a second is force five, which moves cloud visibly across a
    /// frame without tearing it apart, and west is the prevailing direction in
    /// the mid-latitudes this survey was flown over.
    fn default() -> Self {
        Self {
            speed: 10.0,
            from_degrees: 270.0,
        }
    }
}

/// Mirrors the `Air` uniform block in `src/air.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AirUniform {
    /// World x and z of the low corner, then metres per cell across each.
    bounds: [f32; 4],
    /// The wind aloft, and the seconds this dispatch covers.
    aloft: [f32; 4],
    /// [`ROUGHNESS`], [`BOUNDARY_METRES`], [`RELAX_SECONDS`], [`DRIFT_TAU`].
    knobs: [f32; 4],
}

/// The solved wind, and the machinery that solved it until it has.
pub struct Air {
    /// One velocity per cell, in metres per second.
    ///
    /// Not read by anything outside the solve. What the cloud wants is where
    /// the air has *been*, which is the field below; a velocity would have to
    /// be integrated to say that, and it was integrated once here rather than
    /// per sample per frame. It is kept because the solve's own tests read it
    /// and because it is the field a future glider or a windsock would want.
    #[allow(dead_code, reason = "read only by the solve's own tests")]
    wind: wgpu::Texture,
    /// How far the air arriving at each cell has strayed from the bulk drift,
    /// in metres, and in `w` how far it has climbed to get there.
    #[allow(dead_code, reason = "read through its view")]
    drift: wgpu::Texture,
    wind_view: wgpu::TextureView,
    drift_view: wgpu::TextureView,
    /// Dropped once the bake has run; see [`Build`].
    build: Option<Build>,
    /// What the field was solved for, so a change of wind can be noticed.
    baked: Option<Wind>,
    /// How much world the grid was laid over, in metres. Zero until it has
    /// been: see [`Air::bounds`].
    extent: glam::Vec2,
}

/// Everything the bake needs and the frame does not.
///
/// Held in an [`Option`] and dropped the moment the solve finishes, which
/// returns the second velocity texture and both Poisson buffers -- twelve
/// megabytes that would otherwise sit untouched for the rest of the run. The
/// same arrangement `Build` in `src/sky.rs` uses, and for the same reason.
struct Build {
    /// The other half of the ping-pong. The answer always lands in `wind`,
    /// because each step advects into this one and projects back out of it.
    ///
    /// Held rather than only viewed, for the reason the terrain holds its own
    /// textures beside their views: owning it here is what says this is where
    /// the memory lives, and dropping `Build` is what gives it back.
    #[allow(dead_code, reason = "owned here, reached through the bind groups")]
    scratch: wgpu::Texture,
    ground: wgpu::Buffer,
    uniform: wgpu::Buffer,
    air_group: wgpu::BindGroup,
    /// Group 2 and group 3 for the two directions the ping-pong runs in.
    reading: [wgpu::BindGroup; 2],
    writing: [wgpu::BindGroup; 2],
    advect: wgpu::ComputePipeline,
    divergence: wgpu::ComputePipeline,
    red: wgpu::ComputePipeline,
    black: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
    drift: wgpu::ComputePipeline,
}

/// Cells in the grid.
fn cell_count() -> u32 {
    CELLS[0] * CELLS[1] * CELLS[2]
}

/// Columns in the grid.
fn column_count() -> u32 {
    CELLS[0] * CELLS[2]
}

/// Workgroups needed to cover the grid.
fn groups() -> [u32; 3] {
    [
        CELLS[0].div_ceil(GROUP),
        CELLS[1].div_ceil(GROUP),
        CELLS[2].div_ceil(GROUP),
    ]
}

impl Air {
    pub fn new(device: &wgpu::Device) -> Self {
        let extent = wgpu::Extent3d {
            width: CELLS[0],
            height: CELLS[1],
            depth_or_array_layers: CELLS[2],
        };
        let field = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: FIELD_FORMAT,
                // `COPY_SRC` so a test can read the field back and say what it
                // is, which is the only way to check a solve: the frame reduces
                // a velocity to a displacement and says nothing about it. The
                // same reasoning as the `COPY_SRC` on the G-buffer.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let wind = field("air wind");
        let drift = field("air drift");
        let scratch = field("air scratch");
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let (wind_view, drift_view, scratch_view) = (view(&wind), view(&drift), view(&scratch));

        let buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let cells = u64::from(cell_count()) * 4;
        let ground = buffer("air ground", u64::from(column_count()) * 4);
        let pressure = buffer("air pressure", cells);
        let divergence = buffer("air divergence", cells);

        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("air uniform"),
            contents: bytemuck::bytes_of(&AirUniform {
                bounds: [0.0; 4],
                aloft: [0.0; 4],
                knobs: [0.0; 4],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // The advection samples between cells, so this is a filtering sampler;
        // `FIELD_FORMAT` is filterable, which is half of why it was chosen.
        // Clamped rather than repeating: past the edge of the grid the nearest
        // cell is the best answer there is, and wrapping would fetch the
        // opposite side of the raster.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("air sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let air_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("air uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let read_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("air read layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let written = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: FIELD_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };
        let write_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("air write layout"),
            entries: &[
                storage(0, true),
                storage(1, false),
                storage(2, false),
                written(3),
                written(4),
            ],
        });

        let air_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("air uniform group"),
            layout: &air_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let reads = |label, from: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &read_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(from),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            })
        };
        let writes = |label, into: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &write_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ground.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pressure.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: divergence.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(into),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(&drift_view),
                    },
                ],
            })
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("air shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("air.wgsl").into()),
        });
        // Group 0 is left empty. Every other pipeline in the program has the
        // camera there, and this one runs before there is a frame to have a
        // camera for -- the same shape the sky's own build pipelines have.
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("air pipeline layout"),
            bind_group_layouts: &[
                None,
                Some(&air_layout),
                Some(&read_layout),
                Some(&write_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = |label, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            wind_view: wind_view.clone(),
            drift_view: drift_view.clone(),
            build: Some(Build {
                reading: [
                    reads("air reads wind", &wind_view),
                    reads("air reads scratch", &scratch_view),
                ],
                writing: [
                    writes("air writes scratch", &scratch_view),
                    writes("air writes wind", &wind_view),
                ],
                scratch,
                ground,
                uniform,
                air_group,
                advect: pipeline("air advect", "cs_air_advect"),
                divergence: pipeline("air divergence", "cs_air_divergence"),
                red: pipeline("air red", "cs_air_red"),
                black: pipeline("air black", "cs_air_black"),
                project: pipeline("air project", "cs_air_project"),
                drift: pipeline("air drift", "cs_air_drift"),
            }),
            wind,
            drift,
            baked: None,
            extent: glam::Vec2::ZERO,
        }
    }

    /// Solves the field, once, against the terrain that has just been read in.
    ///
    /// Called from [`crate::scene::Scene::update`] after the terrain's own
    /// update, because the coarse height mirror this reads is filled by the
    /// terrain's first load and is empty before it. Waits for the GPU rather
    /// than letting the solve trail into the first frame: it is a load cost and
    /// belongs where the tile reads and the scattering tables are, not smeared
    /// across the first few frames of a flight where it would look like the
    /// frame being slow.
    pub fn ensure_baked(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrain: &Terrain,
        wind: Wind,
    ) {
        if self.baked == Some(wind) || self.build.is_none() {
            return;
        }
        let extent = terrain.world_extent();
        // Nothing to solve around until the chain has been read in. Not an
        // error: the first update loads it, and this runs on every update until
        // it succeeds.
        if extent.x <= 0.0 || extent.y <= 0.0 || !terrain.has_ground() {
            return;
        }

        let cell = glam::Vec2::new(extent.x / CELLS[0] as f32, extent.y / CELLS[2] as f32);
        let origin = -extent * 0.5;

        // The ground under every column, from the mirror the terrain keeps for
        // its own level choice. No GPU pass and no terrain bind group: the
        // answer is already on this side, and it is the same answer the march
        // would give at this scale.
        let mut heights = vec![0.0f32; column_count() as usize];
        for z in 0..CELLS[2] {
            for x in 0..CELLS[0] {
                let at =
                    origin + glam::Vec2::new((x as f32 + 0.5) * cell.x, (z as f32 + 0.5) * cell.y);
                heights[(z * CELLS[0] + x) as usize] = terrain.ground_at(at.x, at.y);
            }
        }
        self.solve(device, queue, extent, &heights, wind);
    }

    /// Runs the solve over a ground field somebody else has laid out.
    ///
    /// Split from [`Air::ensure_baked`] so that a test can hand over a ridge it
    /// drew itself rather than having to build a whole terrain to get one, and
    /// so that where the ground comes from is one decision and what is done
    /// with it is another. `heights` is one metre figure per column, x fastest.
    fn solve(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        extent: glam::Vec2,
        heights: &[f32],
        wind: Wind,
    ) {
        assert_eq!(heights.len(), column_count() as usize);
        self.extent = extent;
        let started = std::time::Instant::now();
        let cell = glam::Vec2::new(extent.x / CELLS[0] as f32, extent.y / CELLS[2] as f32);
        let origin = -extent * 0.5;

        let build = self.build.as_ref().expect("checked by the only caller");
        queue.write_buffer(&build.ground, 0, bytemuck::cast_slice(heights));
        let aloft = wind.velocity();
        queue.write_buffer(
            &build.uniform,
            0,
            bytemuck::bytes_of(&AirUniform {
                bounds: [origin.x, origin.y, cell.x, cell.y],
                aloft: [aloft.x, aloft.y, aloft.z, STEP_SECONDS],
                knobs: [ROUGHNESS, BOUNDARY_METRES, RELAX_SECONDS, DRIFT_TAU],
            }),
        );

        // Nothing clears the velocity: the first advection reads whatever the
        // texture holds, and a fresh texture is zeroed. Starting from still air
        // rather than from the free stream is deliberate -- the relaxation
        // fills it in over the first few steps, and it does so *around* the
        // terrain, where seeding the free stream everywhere would start with
        // air already inside every mountain.
        let [gx, gy, gz] = groups();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("air bake"),
        });
        for step in 0..STEPS {
            // Each of these is its own pass, because each reads what the one
            // before it wrote. A pass boundary is what says every workgroup of
            // the previous dispatch has finished; two dispatches in one pass
            // have no such guarantee.
            build.step(&mut encoder, &build.advect, 0, [gx, gy, gz]);
            build.step(&mut encoder, &build.divergence, 1, [gx, gy, gz]);
            for _ in 0..SWEEPS {
                build.step(&mut encoder, &build.red, 1, [gx, gy, gz]);
                build.step(&mut encoder, &build.black, 1, [gx, gy, gz]);
            }
            build.step(&mut encoder, &build.project, 1, [gx, gy, gz]);

            // Handed over every so often rather than as one command buffer at
            // the end. The whole solve is eleven passes a step, and a recorded
            // pass is not free to hold: the encoder for all three hundred of
            // them came to 170 MB of host memory, which is nothing on its own
            // and is a great deal when a test suite is solving twenty-four of
            // them at once. Submitting in chunks bounds that, and lets the GPU
            // start on the first chunk while the rest is still being recorded.
            if (step + 1) % SUBMIT_EVERY == 0 {
                queue.submit(std::iter::once(encoder.finish()));
                encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("air bake"),
                });
            }
        }
        // Reads the settled field, which is back in `wind` after the last
        // projection, and writes the displacement beside it.
        build.step(&mut encoder, &build.drift, 0, [gx, gy, gz]);
        queue.submit(std::iter::once(encoder.finish()));

        // Blocking here is what puts the cost at load. Without it the work
        // would still be in flight when the first frame was recorded.
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        log::info!(
            "air: solved {} x {} x {} cells of wind at {:.0} x {:.0} m in {:.2?}",
            CELLS[0],
            CELLS[1],
            CELLS[2],
            cell.x,
            cell.y,
            started.elapsed(),
        );

        self.baked = Some(wind);
        self.build = None;
    }

    /// The solved velocity and displacement fields.
    #[allow(dead_code, reason = "only the drift is read; the velocity is not")]
    pub fn views(&self) -> (&wgpu::TextureView, &wgpu::TextureView) {
        (&self.wind_view, &self.drift_view)
    }

    /// Where the grid stands: its near corner in x and z, then how much world
    /// it covers in each.
    ///
    /// Zero-sized until the bake has run, which is what a reader outside this
    /// module tests to know whether there is a field to read at all -- and what
    /// stops it dividing by an extent of nothing on the frames before there is.
    /// The grid is centred on the origin, as the raster is.
    pub fn bounds(&self) -> [f32; 4] {
        [
            -self.extent.x * 0.5,
            -self.extent.y * 0.5,
            self.extent.x,
            self.extent.y,
        ]
    }

    /// What the field was solved for, or [`None`] if it has not been.
    #[cfg(test)]
    pub fn baked_for(&self) -> Option<Wind> {
        self.baked
    }

    /// Says the field is solved without solving it, leaving it zeroed.
    ///
    /// For the offscreen tests in `src/scene.rs`, which build a scene apiece to
    /// assert something about the march or the shading and would otherwise each
    /// pay for a wind solve that no assertion of theirs can see. Measured over
    /// the whole suite: 6 s before this module existed, 58 s with every scene
    /// solving, 12 s with this.
    ///
    /// The solve itself is covered by this module's own tests, which drive
    /// [`Air::solve`] directly over ground they draw themselves, and the wiring
    /// from a real terrain is covered by one scene test that deliberately does
    /// not call this. Dropping `build` is what makes a later `ensure_baked` a
    /// no-op, so a scene that skipped the wind cannot quietly solve it later.
    #[cfg(test)]
    pub fn assume_baked(&mut self, wind: Wind) {
        self.baked = Some(wind);
        self.build = None;
    }
}

impl Build {
    /// Records one dispatch of one kernel, in a pass of its own.
    ///
    /// `direction` picks which way the ping-pong runs: zero reads the answer
    /// and writes the scratch, one reads the scratch and writes the answer. A
    /// step advects one way and projects back the other, so the settled field
    /// is always in `wind` at the end of a step whatever the step count is --
    /// there is no parity to keep straight and no final copy to remember.
    fn step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        direction: usize,
        groups: [u32; 3],
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("air"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(1, &self.air_group, &[]);
        pass.set_bind_group(2, &self.reading[direction], &[]);
        pass.set_bind_group(3, &self.writing[direction], &[]);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }
}

/// The extent of the grid in world metres, for whoever has to place it.
#[allow(dead_code, reason = "read by the tests and by the cloud march")]
pub fn grid_size() -> UVec2 {
    UVec2::new(CELLS[0], CELLS[2])
}

#[cfg(test)]
mod tests {
    use glam::Vec2;

    use super::*;

    /// How wide the test grid's world is, in metres.
    ///
    /// Twenty-five kilometres, so a cell is 156 m across against 350 m tall.
    /// Small enough that the free stream crosses it several times over in the
    /// three hundred steps the solve runs for, which is what makes these tests
    /// about the settled field rather than about a transient.
    const WORLD: f32 = 25_000.0;

    /// A half float, as a whole one.
    ///
    /// The fields are `Rgba16Float` and Rust has no stable `f16`, so a readback
    /// has to decode them. The same three cases `from_half` in `src/sky.rs`
    /// spells out, and for the same reason -- there is no other way to say what
    /// a solve produced.
    fn from_half(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exponent = i32::from((bits >> 10) & 0x1f);
        let mantissa = f32::from(bits & 0x3ff);
        if exponent == 0 {
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

    /// A solved field, read back off the GPU as one vector and a scalar per
    /// cell, indexed the way the shader indexes it.
    struct Field {
        cells: Vec<[f32; 4]>,
    }

    impl Field {
        fn at(&self, x: u32, y: u32, z: u32) -> Vec3 {
            let [vx, vy, vz, _] = self.cells[(x + CELLS[0] * (y + CELLS[1] * z)) as usize];
            Vec3::new(vx, vy, vz)
        }

        /// The fourth channel, which the drift field uses for the rise.
        fn spare(&self, x: u32, y: u32, z: u32) -> f32 {
            self.cells[(x + CELLS[0] * (y + CELLS[1] * z)) as usize][3]
        }
    }

    /// Copies a solved field off the GPU.
    ///
    /// A 3D texture copies row by row with the same 256-byte row alignment a 2D
    /// one has, and 160 cells of eight bytes is 1280, which is already a
    /// multiple of it -- so there is no padding to drop here, unlike
    /// `crate::headless::capture`. Asserted rather than assumed.
    fn read_field(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Field {
        let bytes_per_row = CELLS[0] * 8;
        assert_eq!(bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("air readback"),
            size: u64::from(bytes_per_row * CELLS[1] * CELLS[2]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(CELLS[1]),
                },
            },
            wgpu::Extent3d {
                width: CELLS[0],
                height: CELLS[1],
                depth_or_array_layers: CELLS[2],
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let (sender, receiver) = std::sync::mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();
        let mapped = readback.get_mapped_range(..).unwrap();
        let halves: &[u16] = bytemuck::cast_slice(&mapped);
        let cells = halves
            .chunks_exact(4)
            .map(|c| {
                [
                    from_half(c[0]),
                    from_half(c[1]),
                    from_half(c[2]),
                    from_half(c[3]),
                ]
            })
            .collect();
        drop(mapped);
        readback.unmap();
        Field { cells }
    }

    /// Solves over a ground field of the caller's drawing and reads both
    /// results back.
    fn solved(heights: &[f32], wind: Wind) -> (Field, Field) {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let mut air = Air::new(&device);
        air.solve(&device, &queue, Vec2::splat(WORLD), heights, wind);
        (
            read_field(&device, &queue, &air.wind),
            read_field(&device, &queue, &air.drift),
        )
    }

    /// The world position a cell's centre stands at. Must agree with
    /// `cell_centre` in `src/air.wgsl`, which is what the solve used.
    fn centre(x: u32, y: u32, z: u32) -> Vec3 {
        let cell = WORLD / CELLS[0] as f32;
        Vec3::new(
            -WORLD * 0.5 + (x as f32 + 0.5) * cell,
            (y as f32 + 0.5) * CELL_Y,
            -WORLD * 0.5 + (z as f32 + 0.5) * cell,
        )
    }

    /// Flat ground at sea level: an obstacle-free box.
    fn flat() -> Vec<f32> {
        vec![0.0; column_count() as usize]
    }

    /// A ridge running north-south, so a westerly meets it broadside.
    ///
    /// A raised cosine rather than a step, because a step would put the whole
    /// of the deflection into one column of cells and the question being asked
    /// is whether the flow goes *over* a slope, not whether it stops at a wall.
    fn ridge(peak: f32) -> Vec<f32> {
        let mut heights = flat();
        for z in 0..CELLS[2] {
            for x in 0..CELLS[0] {
                let across = ((x as f32 + 0.5) / CELLS[0] as f32 - 0.5) / RIDGE_HALF_WIDTH;
                let shape = if across.abs() < 1.0 {
                    0.5 * (1.0 + (across * std::f32::consts::PI).cos())
                } else {
                    0.0
                };
                heights[(z * CELLS[0] + x) as usize] = peak * shape;
            }
        }
        heights
    }

    /// Half the ridge's width, as a fraction of the world.
    const RIDGE_HALF_WIDTH: f32 = 0.15;

    /// A flat-topped mesa with sides as steep as the grid can express.
    ///
    /// The ridge is the shape the lift is worth measuring over -- a real slope,
    /// which air climbs. This is the shape the *projection* is worth measuring
    /// over: a wall the free stream runs straight into, so that the divergence
    /// the solve has to remove is as large as this grid can produce. Over the
    /// gentle ridge the flow is barely divergent either way and the measurement
    /// cannot tell a working projection from an absent one.
    fn mesa(top: f32) -> Vec<f32> {
        let mut heights = flat();
        for z in 0..CELLS[2] {
            for x in 0..CELLS[0] {
                let across = (x as f32 + 0.5) / CELLS[0] as f32;
                if (0.4..0.6).contains(&across) {
                    heights[(z * CELLS[0] + x) as usize] = top;
                }
            }
        }
        heights
    }

    /// Whether the solve treated a cell as ground.
    ///
    /// Exactly what `is_solid` decided, not an approximation of it. The shader
    /// tests a cell's *centre*, and `ground_at` interpolates on a lattice whose
    /// nodes are the cell centres -- so at a centre the weights are one and
    /// zero and the bilinear fetch returns that column's height unchanged.
    /// There is nothing to replicate.
    fn solid(heights: &[f32], x: u32, y: u32, z: u32) -> bool {
        centre(x, y, z).y < heights[(z * CELLS[0] + x) as usize]
    }

    /// The two copies of the grid say the same thing.
    ///
    /// There is no preprocessor and no `#include`, so every constant the shader
    /// and this module both need is written twice. A grid that disagreed would
    /// not fail to compile or to run -- it would index a texture with one
    /// shape as though it had another, and the field would come out sheared.
    /// The same guard `the_shader_and_rust_agree_on_the_atmosphere` puts on the
    /// sky.
    #[test]
    fn the_shader_and_rust_agree_on_the_air_grid() {
        let source = include_str!("air.wgsl");
        let pairs = [
            ("CELLS_X", format!("{}u", CELLS[0]), "u32"),
            ("CELLS_Y", format!("{}u", CELLS[1]), "u32"),
            ("CELLS_Z", format!("{}u", CELLS[2]), "u32"),
            ("TOP_METRES", format!("{TOP_METRES:.1}"), "f32"),
            ("DRIFT_STEPS", format!("{DRIFT_STEPS}u"), "u32"),
        ];
        for (name, value, kind) in pairs {
            let declaration = format!("const {name}: {kind}");
            let line = source
                .lines()
                .find(|line| line.trim_start().starts_with(&declaration))
                .unwrap_or_else(|| panic!("src/air.wgsl declares no {name}"));
            assert!(
                line.contains(&value),
                "src/air.wgsl says {line:?}, but src/air.rs says {name} is {value}"
            );
        }
        // The shader derives the layer height rather than spelling it out, so
        // the text comparison cannot reach it. This is what says the division
        // is the same division.
        assert!(
            source.contains("const CELL_Y: f32 = TOP_METRES / f32(CELLS_Y);"),
            "src/air.wgsl no longer derives CELL_Y from the grid"
        );
        assert_eq!(CELL_Y, 350.0);
        // Every kernel is dispatched with the same workgroup shape.
        assert_eq!(
            source.matches("@compute @workgroup_size(4, 4, 4)").count(),
            6,
            "src/air.wgsl has kernels this module would dispatch wrongly"
        );
        assert_eq!(GROUP, 4);
    }

    /// A bearing is where the wind comes *from*.
    #[test]
    fn a_westerly_blows_towards_the_east() {
        let west = Wind {
            speed: 10.0,
            from_degrees: 270.0,
        };
        let velocity = west.velocity();
        assert!(velocity.x > 9.99, "{velocity:?}");
        assert!(velocity.z.abs() < 1e-3, "{velocity:?}");
        assert_eq!(velocity.y, 0.0);

        // A northerly blows south, which is +Z in a world whose north is -Z.
        let north = Wind {
            speed: 4.0,
            from_degrees: 0.0,
        };
        assert!(north.velocity().z > 3.99, "{:?}", north.velocity());

        // And the speed is the length of it, whatever the bearing.
        for from_degrees in [0.0, 37.0, 90.0, 180.0, 271.0, 359.0] {
            let wind = Wind {
                speed: 7.5,
                from_degrees,
            };
            assert!(
                (wind.velocity().length() - 7.5).abs() < 1e-4,
                "{from_degrees}"
            );
        }
    }

    /// A westerly over flat ground is a westerly.
    ///
    /// The solve has to leave a field it was given alone: the free stream is
    /// already divergence free and already satisfies every boundary condition,
    /// so an obstacle-free box is the one case where the right answer is known
    /// in closed form. Anything the projection does here is something it is
    /// doing wrong, and it would do the same thing over terrain where there is
    /// nothing to compare against.
    #[test]
    fn flat_ground_leaves_the_wind_where_it_found_it() {
        let wind = Wind {
            speed: 10.0,
            from_degrees: 270.0,
        };
        let (solved, _) = solved(&flat(), wind);

        // Well above the boundary layer, where the target is the full speed,
        // and away from the open edges, where air genuinely enters and leaves.
        let y = CELLS[1] - 4;
        for z in 8..CELLS[2] - 8 {
            for x in 8..CELLS[0] - 8 {
                let at = solved.at(x, y, z);
                assert!(
                    (at.x - 10.0).abs() < 0.35 && at.y.abs() < 0.35 && at.z.abs() < 0.35,
                    "flat ground bent the free stream at {x},{y},{z} to {at:?}"
                );
            }
        }
    }

    /// Nothing moves inside a mountain.
    ///
    /// The first half of the obstacle boundary condition, and the half that can
    /// be stated exactly: whatever else the solve does, a cell the ground fills
    /// holds still. Without it air would flow through rock and the lift over a
    /// ridge would be whatever leaked past rather than what went over.
    #[test]
    fn a_solid_cell_lets_no_wind_through_it() {
        let heights = ridge(1400.0);
        let (solved, drift) = solved(
            &heights,
            Wind {
                speed: 10.0,
                from_degrees: 270.0,
            },
        );

        let mut found = 0;
        for z in 0..CELLS[2] {
            for y in 0..CELLS[1] {
                for x in 0..CELLS[0] {
                    if !solid(&heights, x, y, z) {
                        continue;
                    }
                    found += 1;
                    let at = solved.at(x, y, z);
                    assert_eq!(at, Vec3::ZERO, "rock is moving at {x},{y},{z}");
                    assert!(drift.at(x, y, z).is_finite(), "at {x},{y},{z}");
                }
            }
        }
        // A test that found no rock would pass by saying nothing. The ridge is
        // 1400 m over a 25 km box, so four of the twenty layers are underground
        // at the crest and none at the edges.
        assert!(found > 10_000, "only {found} cells of the ridge were solid");
    }

    /// What the projection is for: the settled flow has nowhere piling up.
    ///
    /// Measured only where the shader's stencil reduces to a plain central
    /// difference -- a fluid cell whose six neighbours are all fluid and all
    /// inside the grid -- so this is the property itself and not a second copy
    /// of the boundary rules that could agree with a wrong one.
    #[test]
    fn the_baked_wind_never_leaves_the_grid_divergent() {
        let heights = mesa(2800.0);
        let (solved, _) = solved(
            &heights,
            Wind {
                speed: 25.0,
                from_degrees: 270.0,
            },
        );

        let cell = WORLD / CELLS[0] as f32;
        // No flux across a wall: a neighbour that is rock, or off the edge,
        // answers with the cell doing the asking, so the difference across that
        // face is zero. One statement of the boundary condition, written here
        // as what is being asserted rather than copied from the shader -- a
        // solve that used a different rule would show up as flux this could
        // see.
        let flux = |x: u32, y: u32, z: u32, d: (i32, i32, i32)| {
            let (a, b, c) = (x as i32 + d.0, y as i32 + d.1, z as i32 + d.2);
            let inside = a >= 0
                && b >= 0
                && c >= 0
                && (a as u32) < CELLS[0]
                && (b as u32) < CELLS[1]
                && (c as u32) < CELLS[2];
            if !inside || solid(&heights, a as u32, b as u32, c as u32) {
                solved.at(x, y, z)
            } else {
                solved.at(a as u32, b as u32, c as u32)
            }
        };

        let mut worst = 0.0f32;
        let mut total = 0.0f32;
        let mut counted = 0;
        for z in 0..CELLS[2] {
            for y in 0..CELLS[1] {
                for x in 0..CELLS[0] {
                    if solid(&heights, x, y, z) {
                        continue;
                    }
                    let divergence = (flux(x, y, z, (1, 0, 0)).x - flux(x, y, z, (-1, 0, 0)).x)
                        / (2.0 * cell)
                        + (flux(x, y, z, (0, 1, 0)).y - flux(x, y, z, (0, -1, 0)).y)
                            / (2.0 * CELL_Y)
                        + (flux(x, y, z, (0, 0, 1)).z - flux(x, y, z, (0, 0, -1)).z) / (2.0 * cell);
                    worst = worst.max(divergence.abs());
                    total += divergence.abs();
                    counted += 1;
                }
            }
        }

        assert!(counted > 100_000, "only {counted} cells were open air");
        let mean = total / counted as f32;

        // Against the scale the field works at: twenty-five metres a second
        // across one 156 m cell is 0.16 per second.
        //
        // The *mean* is what carries this test. Measured at 6.65e-5 with the
        // projection and 3.21e-4 without it -- a factor of 4.8 -- so a bound at
        // a thousandth of the scale sits 2.4 times above the working figure and
        // 2 times below the broken one.
        //
        // The worst case is not asserted tightly and cannot be: it is 1.07e-2
        // either way. That residual lives in single cells against the wall of
        // the mesa and is the odd-even decoupling described at [`SWEEPS`],
        // which the projection has no power to remove and whose size therefore
        // says nothing about whether the projection ran. It is bounded loosely
        // here only to catch a field that has actually diverged.
        let scale = 25.0 / cell;
        assert!(
            mean < scale * 1e-3,
            "the mean divergence left was {mean:.3e} per second, against {scale:.3e}"
        );
        assert!(
            worst < scale * 0.5,
            "the worst divergence left was {worst:.3e} per second, against {scale:.3e}"
        );
    }

    /// Air goes over a ridge rather than through it, and comes down behind.
    ///
    /// The property the whole grid exists for. A constant wind would give a
    /// vertical velocity of exactly zero everywhere; what makes cloud sit on a
    /// windward slope and clear in the lee is that this one does not.
    #[test]
    fn the_wind_lifts_over_a_ridge_and_sinks_behind_it() {
        let heights = ridge(1400.0);
        let (solved, drift) = solved(
            &heights,
            Wind {
                speed: 10.0,
                from_degrees: 270.0,
            },
        );

        // A westerly blows towards +x, so the windward flank is the low half of
        // the ridge and the lee is the high half. The crest tops out inside
        // layer 4, which is the first layer of air over the whole of it.
        let layer = 4;
        assert!(centre(0, layer, 0).y > 1400.0);
        let flank = |from: f32, to: f32| {
            let span = |f: f32| ((0.5 + f * RIDGE_HALF_WIDTH) * CELLS[0] as f32) as u32;
            let (low, high) = (span(from), span(to));
            let mut rise = 0.0;
            let mut lifted = 0.0;
            let mut count = 0.0;
            for z in 8..CELLS[2] - 8 {
                for x in low..high {
                    rise += solved.at(x, layer, z).y;
                    lifted += drift.spare(x, layer, z);
                    count += 1.0;
                }
            }
            (rise / count, lifted / count)
        };

        let (windward, climbed) = flank(-0.9, -0.2);
        let (lee, dropped) = flank(0.2, 0.9);
        assert!(
            windward > 0.05,
            "air over the windward slope rose at {windward:.4} m/s"
        );
        assert!(lee < -0.05, "air over the lee rose at {lee:.4} m/s");

        // And the displacement field agrees with the velocity that produced it:
        // a parcel arriving over the windward slope has just climbed, and one
        // over the lee has just fallen. This is the channel the cloud coverage
        // will read, so it has to carry the same story.
        assert!(
            climbed > 1.0,
            "parcels over the windward slope rose {climbed:.2} m"
        );
        assert!(dropped < -1.0, "parcels over the lee rose {dropped:.2} m");
    }

    /// The displacement stays a perturbation.
    ///
    /// It is an integral, and an integral over a steady field is the one thing
    /// here that could grow without bound -- which would stretch cloud across
    /// the whole sky rather than deforming it. The window is fixed at
    /// [`DRIFT_TAU`], so the bound is what the fastest air could cover in that
    /// long, and a field that broke the fixed window would sail past it.
    #[test]
    fn the_drift_never_strays_further_than_the_window_allows() {
        let heights = ridge(1400.0);
        let speed = 10.0;
        let (_, drift) = solved(
            &heights,
            Wind {
                speed,
                from_degrees: 270.0,
            },
        );

        // Three times the free stream, because air accelerating over a ridge
        // genuinely outruns it and the deviation is measured against it.
        let bound = 3.0 * speed * DRIFT_TAU;
        let mut worst = 0.0f32;
        let mut strayed = 0.0f32;
        for z in 0..CELLS[2] {
            for y in 0..CELLS[1] {
                for x in 0..CELLS[0] {
                    let at = drift.at(x, y, z);
                    assert!(at.is_finite(), "the drift is not a number at {x},{y},{z}");
                    worst = worst.max(at.length());
                    if !solid(&heights, x, y, z) {
                        strayed = strayed.max(at.length());
                    }
                }
            }
        }
        assert!(
            worst < bound,
            "a parcel strayed {worst:.0} m against {bound:.0}"
        );
        // And it is not simply zero, which would pass the bound and mean the
        // terrain never touched the flow at all.
        assert!(
            strayed > 10.0,
            "the ridge displaced nothing: {strayed:.2} m"
        );
    }
}
