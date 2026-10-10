//! Irradiance probes: the light the world sends to itself, in cascades of points
//! centred on the eye.
//!
//! The rays are traced in `cs_probe_rays` (`src/terrain.wgsl`), lit and
//! projected in `cs_probe_gather` (`src/probe.wgsl`), and carried as the
//! lattice steps under a moving eye by `cs_probe_shift`. This is where the
//! cascades stand, which probes a frame updates, and the textures between the
//! passes and their readers.
//!
//! Every frame updates a sixty-fourth of every cascade -- one residue of the
//! world's own probe index modulo four on each axis -- so every probe is traced
//! once in sixty-four frames wherever the camera looks. What is updated is chosen by where
//! a probe is in the world and by nothing else, which is what keeps a still
//! camera's light still.

use glam::Vec3;

use crate::terrain::gpu::Terrain;

/// Probes along each side of a cascade. Must match `PROBE_ACROSS` in
/// `src/terrain.wgsl` and `src/probe.wgsl`.
pub const PROBE_ACROSS: u32 = 32;

/// Cascades, each twice the spacing and twice the span of the one inside it.
/// Must match `PROBE_CASCADES`.
///
/// Eight from sixteen metres puts the innermost at half a kilometre across and
/// the outermost at sixty-five, two kilometres between its probes: from sixteen
/// metres, which is a gully's width, to the width of a whole valley, and out to
/// where the haze in front of the ground has taken over from the ground.
pub const PROBE_CASCADES: u32 = 8;

/// Metres between the innermost cascade's probes.
const PROBE_SPACING: f32 = 16.0;

/// Rays per probe per update. Must match `PROBE_RAYS`.
const PROBE_RAYS: u32 = 128;

/// Frames to update every probe once. Must match `PROBE_PHASES`.
///
/// Sixty-four, and a hundred and twenty-eight rays apiece, rather than sixteen
/// and thirty-two for the same rays a frame. A probe's rays are drawn once and
/// kept -- see `probe_direction` in `src/probe.wgsl` -- so what thirty-two of
/// them get wrong about the ground around it stays wrong, and next to a
/// neighbour that got it wrong the other way it reads as a faint blotch the size
/// of the spacing on everything the two light. It shows on open ground in a
/// valley's shade, where the ground the rays meet is a patchwork of sunlit and
/// shadowed slopes, and a hundred and twenty-eight take it out. What it costs is
/// how long a probe takes to notice a change, a second at sixty frames, which
/// for light that follows the sun and the weather is no cost at all; and ground
/// the eye has just reached is lit by the coarser cascade until its own probes
/// come round, which they would have done in a quarter of a second.
const PROBE_PHASES: u32 = 64;

/// How far the eye may move in one frame before the history is thrown away and
/// every probe traced afresh, in metres: half the innermost cascade.
const TELEPORT: f32 = 0.5 * PROBE_SPACING * PROBE_ACROSS as f32;

/// The format the probes are held in: four channels a texel, three texels a
/// probe. See `probe_field` in `src/probe.wgsl`.
const PROBE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Mirrors `Probes` in `src/terrain.wgsl` and `src/probe.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ProbeUniform {
    /// Per cascade: the lattice's corner in the world, and the metres between
    /// probes.
    cascade: [[f32; 4]; PROBE_CASCADES as usize],
    /// The same, as last frame had it.
    was: [[f32; 4]; PROBE_CASCADES as usize],
}

impl ProbeUniform {
    /// Where the cascades stand for an eye.
    ///
    /// Centred on it in all three axes and snapped to each cascade's own
    /// spacing, so the lattice stands still in the world while the camera moves
    /// within a spacing and steps by exactly one probe when it crosses one --
    /// the same arrangement as the cloud's light volume and the terrain's
    /// shadow, and for the same reason.
    pub(crate) fn cascades(eye: Vec3) -> [[f32; 4]; PROBE_CASCADES as usize] {
        std::array::from_fn(|level| {
            let spacing = PROBE_SPACING * (1u32 << level) as f32;
            let centre = (eye / spacing).round() * spacing;
            let corner = centre - spacing * (PROBE_ACROSS / 2) as f32;
            [corner.x, corner.y, corner.z, spacing]
        })
    }
}

/// The residue of the world lattice a phase updates: the world index modulo
/// four on each axis, x fastest.
fn residue(phase: u32) -> [u32; 4] {
    [phase % 4, (phase / 4) % 4, phase / 16, 0]
}

/// The probes and everything that traces, lights and carries them.
pub struct Probes {
    /// Every probe as readers see it: what the last frame left.
    current: wgpu::Texture,
    current_view: wgpu::TextureView,
    /// What this frame's passes write, copied over `current` once they are done.
    next: wgpu::Texture,
    uniform: wgpu::Buffer,
    /// What each ray of a frame's probes met.
    #[allow(dead_code, reason = "kept alive by the bind groups; owned here")]
    hits: wgpu::Buffer,
    rays: wgpu::ComputePipeline,
    gather: wgpu::ComputePipeline,
    shift: wgpu::ComputePipeline,
    /// One per phase: the uniform, which residue, and the hits.
    ray_groups: Vec<wgpu::BindGroup>,
    /// One per phase: the same and everything the light needs.
    gather_groups: Vec<wgpu::BindGroup>,
    shift_group: wgpu::BindGroup,
    /// Where the cascades stood last frame, if there was one.
    was: Option<(Vec3, [[f32; 4]; PROBE_CASCADES as usize])>,
    /// Frames of updates so far.
    count: u32,
    /// Whether the next frame drawn traces every phase rather than one.
    ///
    /// Owed until a frame is drawn, not decided per update: a scene is settled
    /// by updating it until its terrain is in, without drawing, and a priming
    /// decided by the first of those updates and forgotten by the next would
    /// leave the first frame anybody sees lit by probes nobody had traced. A
    /// cell because drawing is done through a shared reference.
    priming: std::cell::Cell<bool>,
}

/// Everything the gather reads that someone else owns.
pub struct Lit<'a> {
    pub sky: &'a crate::sky::Sky,
    pub palette: &'a wgpu::Buffer,
    pub clouds: crate::deferred::Clouds<'a>,
    pub shadow: crate::shadow::Field<'a>,
}

/// The probes themselves and where they stand, made before anything that reads
/// or writes them -- the cloud march reads them, and the passes that write them
/// read the cloud's light volume, so neither can be made first if this belongs
/// to either.
pub struct Store {
    current: wgpu::Texture,
    current_view: wgpu::TextureView,
    next: wgpu::Texture,
    next_view: wgpu::TextureView,
    uniform: wgpu::Buffer,
}

impl Store {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 3 * PROBE_ACROSS,
                    height: PROBE_ACROSS,
                    depth_or_array_layers: PROBE_ACROSS * PROBE_CASCADES,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: PROBE_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let current = texture("probes");
        let next = texture("probes next");
        let current_view = current.create_view(&Default::default());
        let next_view = next.create_view(&Default::default());
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe uniform"),
            size: std::mem::size_of::<ProbeUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            current,
            current_view,
            next,
            next_view,
            uniform,
        }
    }

    /// What the readers of the probes bind: the probes, and where they stand.
    pub fn field(&self) -> Field<'_> {
        Field {
            view: &self.current_view,
            uniform: &self.uniform,
        }
    }
}

impl Probes {
    pub fn new(device: &wgpu::Device, terrain: &Terrain, store: Store, lit: Lit<'_>) -> Self {
        let Store {
            current,
            current_view,
            next,
            next_view,
            uniform,
        } = store;
        let phases: Vec<wgpu::Buffer> = (0..PROBE_PHASES)
            .map(|phase| {
                use wgpu::util::DeviceExt;
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("probe phase"),
                    contents: bytemuck::bytes_of(&residue(phase)),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            })
            .collect();
        let per_frame = PROBE_CASCADES * PROBE_ACROSS.pow(3) / PROBE_PHASES;
        let hits = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe hits"),
            size: u64::from(per_frame * PROBE_RAYS) * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let uniform_ty = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let sampled = |sample_type, view_dimension| wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        };
        let filtered = wgpu::TextureSampleType::Float { filterable: true };
        let written = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: PROBE_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D3,
        };

        let ray_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe ray layout"),
            entries: &[
                entry(20, uniform_ty),
                entry(21, uniform_ty),
                entry(22, storage(false)),
            ],
        });
        let gather_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe gather layout"),
            entries: &[
                entry(0, uniform_ty),
                entry(1, uniform_ty),
                entry(2, storage(true)),
                entry(4, written),
                entry(5, uniform_ty),
                entry(6, sampled(filtered, wgpu::TextureViewDimension::D3)),
                entry(7, uniform_ty),
                entry(
                    8,
                    sampled(wgpu::TextureSampleType::Uint, wgpu::TextureViewDimension::D2Array),
                ),
                entry(9, uniform_ty),
            ],
        });
        // The shift's own, which reads the probes and writes their next frame and
        // has no business with anything else.
        let shift_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("probe shift layout"),
            entries: &[
                entry(0, uniform_ty),
                entry(3, sampled(filtered, wgpu::TextureViewDimension::D3)),
                entry(4, written),
            ],
        });

        let rays = terrain.pipeline(
            device,
            "probe ray pipeline",
            "cs_probe_rays",
            [None, None, Some(&ray_layout)],
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("probe.wgsl").into()),
        });
        let pipeline = |label, entry_point, layouts: &[Option<&wgpu::BindGroupLayout>]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let gather = pipeline(
            "probe gather pipeline",
            "cs_probe_gather",
            &[
                Some(lit.sky.layout()),
                None,
                Some(lit.sky.tables_layout()),
                Some(&gather_layout),
            ],
        );
        let shift = pipeline(
            "probe shift pipeline",
            "cs_probe_shift",
            &[None, None, Some(lit.sky.tables_layout()), Some(&shift_layout)],
        );

        let view = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        fn buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let ray_groups = phases
            .iter()
            .map(|phase| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe ray group"),
                    layout: &ray_layout,
                    entries: &[buffer(20, &uniform), buffer(21, phase), buffer(22, &hits)],
                })
            })
            .collect();
        let gather_groups = phases
            .iter()
            .map(|phase| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("probe gather group"),
                    layout: &gather_layout,
                    entries: &[
                        buffer(0, &uniform),
                        buffer(1, phase),
                        buffer(2, &hits),
                        view(4, &next_view),
                        buffer(5, lit.palette),
                        view(6, lit.clouds.volume),
                        buffer(7, lit.clouds.light),
                        view(8, lit.shadow.view),
                        buffer(9, lit.shadow.uniform),
                    ],
                })
            })
            .collect();
        let shift_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("probe shift group"),
            layout: &shift_layout,
            entries: &[buffer(0, &uniform), view(3, &current_view), view(4, &next_view)],
        });

        Self {
            current,
            current_view,
            next,
            uniform,
            hits,
            rays,
            gather,
            shift,
            ray_groups,
            gather_groups,
            shift_group,
            was: None,
            count: 0,
            priming: std::cell::Cell::new(true),
        }
    }

    /// Places the cascades on the eye and says which probes this frame traces.
    ///
    /// Every probe is traced afresh on the first frame and after the eye jumps
    /// further than half the innermost cascade, so a frame drawn once -- a
    /// test's, or a headless render's -- is lit by probes that have all been
    /// traced rather than by whatever a sixty-fourth of them found.
    pub fn set_frame(&mut self, queue: &wgpu::Queue, eye: Vec3) {
        let cascade = ProbeUniform::cascades(eye);
        let (was_eye, was) = self.was.unwrap_or((eye, cascade));
        if self.was.is_none() || was_eye.distance(eye) > TELEPORT {
            self.priming.set(true);
        }
        self.count = self.count.wrapping_add(1);
        let uniform = ProbeUniform { cascade, was };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        self.was = Some((eye, cascade));
    }

    /// Records this frame's passes into `scope`: the shift, then the rays and
    /// the light of each phase this frame traces, then the copy that makes what
    /// they wrote what everything else reads.
    pub fn draw(
        &self,
        scope: &mut wgpu_profiler::Scope<'_, wgpu::CommandEncoder>,
        terrain: &Terrain,
        sky: &crate::sky::Sky,
    ) {
        let per_frame = PROBE_ACROSS.pow(3) / PROBE_PHASES;
        {
            let mut pass = scope.scoped_compute_pass("probe-shift");
            pass.set_pipeline(&self.shift);
            pass.set_bind_group(2, sky.tables_bind_group(), &[]);
            pass.set_bind_group(3, &self.shift_group, &[]);
            pass.dispatch_workgroups(
                PROBE_ACROSS / 4,
                PROBE_ACROSS / 4,
                PROBE_ACROSS * PROBE_CASCADES / 4,
            );
        }
        let phases = if self.priming.replace(false) {
            0..PROBE_PHASES
        } else {
            let phase = self.count % PROBE_PHASES;
            phase..phase + 1
        };
        for phase in phases {
            {
                let mut pass = scope.scoped_compute_pass("probe-rays");
                pass.set_pipeline(&self.rays);
                pass.set_bind_group(1, terrain.bind_group(), &[]);
                pass.set_bind_group(3, &self.ray_groups[phase as usize], &[]);
                pass.dispatch_workgroups(per_frame, PROBE_CASCADES, 1);
            }
            let mut pass = scope.scoped_compute_pass("probe-gather");
            pass.set_pipeline(&self.gather);
            pass.set_bind_group(0, sky.bind_group(), &[]);
            pass.set_bind_group(2, sky.tables_bind_group(), &[]);
            pass.set_bind_group(3, &self.gather_groups[phase as usize], &[]);
            pass.dispatch_workgroups(per_frame, PROBE_CASCADES, 1);
        }
        scope.copy_texture_to_texture(
            self.next.as_image_copy(),
            self.current.as_image_copy(),
            self.current.size(),
        );
    }

    /// Has the next frame trace every probe, as though the camera had stood
    /// where it is for the sixty-four frames it takes them to come round.
    ///
    /// Only the tests ask, for the reason `Scene::finish_the_cloud` exists: a
    /// harness draws a frame once, and a frame drawn once after a flight has
    /// probes that entered the lattice on the way and have not yet been traced.
    /// A running renderer has traced them by the time anybody looks.
    #[cfg(test)]
    pub fn prime(&mut self) {
        self.was = None;
    }

    /// What the readers of the probes bind: the probes, and where they stand.
    pub fn field(&self) -> Field<'_> {
        Field {
            view: &self.current_view,
            uniform: &self.uniform,
        }
    }

    #[cfg(test)]
    pub fn texture(&self) -> &wgpu::Texture {
        &self.current
    }
}

/// Probes that answer for nowhere, for the tests of passes that read them but
/// have no terrain to trace them over: every cascade stands a world away, so
/// no point falls in one and every reader takes its stand-in.
#[cfg(test)]
pub fn unlit(device: &wgpu::Device) -> (wgpu::TextureView, wgpu::Buffer) {
    use wgpu::util::DeviceExt;
    let view = device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("no probes"),
            size: wgpu::Extent3d {
                width: 3,
                height: 1,
                depth_or_array_layers: PROBE_CASCADES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: PROBE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default());
    let far = [[1.0e9, 1.0e9, 1.0e9, 1.0]; PROBE_CASCADES as usize];
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("no probe uniform"),
        contents: bytemuck::bytes_of(&ProbeUniform {
            cascade: far,
            was: far,
        }),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    (view, buffer)
}

/// What a pass reads the probes through.
#[derive(Clone, Copy)]
pub struct Field<'a> {
    pub view: &'a wgpu::TextureView,
    pub uniform: &'a wgpu::Buffer,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader_says(source: &str, name: &str) -> String {
        let line = source
            .lines()
            .find(|line| line.starts_with(&format!("const {name}:")))
            .unwrap_or_else(|| panic!("no {name}"));
        line.split('=').nth(1).unwrap().trim().trim_end_matches(';').to_string()
    }

    /// Source from `from` up to the next line that starts with a closing brace,
    /// comments dropped and whitespace normalised.
    fn block(source: &str, from: &str) -> String {
        let start = source.find(from).unwrap_or_else(|| panic!("no `{from}`"));
        let end = source[start..].find("\n}").expect("unterminated");
        source[start..start + end]
            .lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .flat_map(str::split_whitespace)
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn constant(source: &str, name: &str) -> Option<String> {
        source
            .lines()
            .find(|line| line.starts_with(&format!("const {name}:")))
            .map(str::to_string)
    }

    /// The probes are traced in one shader, lit in another and read in a third,
    /// and those only agree about which probe is which, where it stands, which
    /// way its rays go and how it is read between its neighbours if they share
    /// the arithmetic. A probe lit for one place and read for another is light
    /// in the wrong valley.
    ///
    /// Bodies compared with whitespace normalised and comments stripped.
    #[test]
    fn the_probes_read_the_same_everywhere() {
        let terrain = include_str!("terrain.wgsl");
        let probe = include_str!("probe.wgsl");
        let shading = include_str!("shading.wgsl");
        for function in [
            "fn noise_mix(",
            "fn noise_hash(",
            "fn probe_of(",
            "fn probe_centre(",
            "fn probe_world(",
            "fn probe_direction(",
            "struct Probes {",
            "struct Phase {",
        ] {
            assert_eq!(
                block(probe, function),
                block(terrain, function),
                "{function} differs between src/probe.wgsl and src/terrain.wgsl"
            );
        }
        assert_eq!(block(shading, "struct Probes {"), block(terrain, "struct Probes {"));
        for function in [
            "fn probe_grid(",
            "fn probe_uvw(",
            "fn probe_share(",
            "fn probe_irradiance(",
            "fn probe_upwelling(",
        ] {
            assert_eq!(
                block(probe, function),
                block(shading, function),
                "{function} differs between src/probe.wgsl and src/shading.wgsl"
            );
        }
        for name in ["PROBE_EDGE", "PROBE_FADE", "SH_0", "SH_1", "PROBE_ACROSS", "PROBE_CASCADES"] {
            assert_eq!(constant(probe, name), constant(shading, name), "{name}");
        }
        // ... and the cloud march, which reads band zero of them.
        let march = include_str!("cloud_march.wgsl");
        assert_eq!(block(march, "struct Probes {"), block(terrain, "struct Probes {"));
        for function in ["fn probe_grid(", "fn probe_uvw(", "fn probe_share(", "fn probe_upwelling("] {
            assert_eq!(
                block(march, function),
                block(shading, function),
                "{function} differs between src/cloud_march.wgsl and src/shading.wgsl"
            );
        }
        for name in ["PROBE_EDGE", "PROBE_FADE", "SH_0", "SH_1", "PROBE_ACROSS", "PROBE_CASCADES"] {
            assert_eq!(constant(march, name), constant(shading, name), "{name}");
        }
    }

    /// What the light pass reads of the sky, the cloud and the terrain's
    /// shadow, it reads the way the shading does: the ground a probe's ray meets
    /// is lit as the ground the eye sees is.
    #[test]
    fn the_probes_light_the_ground_the_way_the_shading_does() {
        let probe = include_str!("probe.wgsl");
        let shading = include_str!("shading.wgsl");
        for function in [
            "fn to_texture(",
            "fn top_distance(",
            "fn transmittance_uv(",
            "fn sample_transmittance(",
            "fn sample_multiscatter(",
            "fn horizon_zenith(",
            "fn skyview_v(",
            "fn sample_skyview(",
            "fn light_uvw(",
            "fn stacked_w(",
            "fn beyond_light(",
            "fn reaching(",
            "fn sun_reaching(",
            "fn sky_reaching(",
            "fn sky_share(",
            "struct Light {",
            "struct Frame {",
        ] {
            assert_eq!(
                block(probe, function),
                block(shading, function),
                "{function} differs between src/probe.wgsl and src/shading.wgsl"
            );
        }
        for name in [
            "GROUND_RADIUS",
            "TOP_RADIUS",
            "TRANSMITTANCE_WIDTH",
            "TRANSMITTANCE_HEIGHT",
            "MULTISCATTER_SIZE",
            "SKYVIEW_HEIGHT",
            "LIGHT_ACROSS",
            "LIGHT_SLICES",
            "LIGHT_CASCADES",
            "CASCADE_EDGE",
            "CASCADE_FADE",
            "BOUNCED",
            "SUN_FRAMES",
            "SKY_FRAMES",
            "PALETTE_SIZE",
            "MATERIAL_MASK",
        ] {
            assert_eq!(constant(probe, name), constant(shading, name), "{name}");
        }
    }

    #[test]
    fn the_shaders_agree_about_the_probes() {
        // The highest anything reads a probe is the top of the light volume.
        assert_eq!(
            constant(include_str!("terrain.wgsl"), "PROBE_TOP"),
            Some(format!("const PROBE_TOP: f32 = {:.1};", crate::cloud::CEILING_TOP))
        );
        for source in [include_str!("terrain.wgsl"), include_str!("probe.wgsl")] {
            assert_eq!(shader_says(source, "PROBE_ACROSS"), format!("{PROBE_ACROSS}u"));
            assert_eq!(shader_says(source, "PROBE_CASCADES"), format!("{PROBE_CASCADES}u"));
            assert_eq!(shader_says(source, "PROBE_RAYS"), format!("{PROBE_RAYS}u"));
            assert_eq!(shader_says(source, "PROBE_PHASES"), format!("{PROBE_PHASES}u"));
        }
    }

    #[test]
    fn every_probe_cascade_stands_still_while_the_camera_moves() {
        let eye = Vec3::new(3.0, 1003.0, -5.0);
        let here = ProbeUniform::cascades(eye);
        let near = ProbeUniform::cascades(eye + Vec3::new(2.0, -3.0, 4.0));
        for level in 0..PROBE_CASCADES as usize {
            let spacing = here[level][3];
            assert_eq!(here[level], near[level], "cascade {level} slid");
            let moved = ProbeUniform::cascades(eye + Vec3::X * spacing)[level];
            assert_eq!(moved[0] - here[level][0], spacing, "cascade {level} jumped");
            for axis in 0..3 {
                let rungs = here[level][axis] / spacing;
                assert_eq!(rungs, rungs.round(), "cascade {level} is off its own lattice");
            }
        }
    }

    /// A full round of phases updates every probe of a cascade exactly once,
    /// wherever the cascade stands -- the CPU twin of `probe_of`.
    #[test]
    fn every_phase_together_visits_every_probe_once() {
        let n = PROBE_ACROSS as usize;
        for first in [[0i32, 0, 0], [3, -7, 5], [-1, 2, -3]] {
            let mut seen = vec![0u8; n * n * n];
            for phase in 0..PROBE_PHASES {
                let r = residue(phase);
                for slot in 0..PROBE_ACROSS.pow(3) / PROBE_PHASES {
                    let reduced = [slot % 8, (slot / 8) % 8, slot / 64];
                    let shift: [u32; 3] =
                        std::array::from_fn(|a| (r[a] as i32 - first[a]).rem_euclid(4) as u32);
                    let index: [u32; 3] = std::array::from_fn(|a| reduced[a] * 4 + shift[a]);
                    // The residue it was chosen for is the residue it has.
                    assert_eq!((index[0] as i32 + first[0]).rem_euclid(4) as u32, r[0]);
                    seen[(index[2] as usize * n + index[1] as usize) * n + index[0] as usize] += 1;
                }
            }
            assert!(seen.iter().all(|&count| count == 1));
        }
    }
}
