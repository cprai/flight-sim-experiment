//! The filled surface, computed by relaxation on the GPU.
//!
//! See `flood.wgsl` for what the relaxation is and why it has to run downwards
//! from an upper bound. This is the part that owns the two buffers it
//! ping-pongs between, decides when it has converged, and reports how many
//! iterations that took -- which is the number the whole GPU port's speed rests
//! on, so it is logged rather than left to be inferred from a total.

use crate::gpu::{Extent, Gpu};

/// Iterations run between convergence checks.
///
/// Every check is a readback, and a readback drains the queue: at one check per
/// iteration the stalls cost more than the arithmetic they are measuring. The
/// price is the batch that reports nothing moved, plus whatever was left of the
/// one before it -- at most thirty-one iterations spent proving the fill had
/// already finished, against a stall saved fifteen times out of sixteen.
///
/// It is also the coarser convergence test the determinism rules ask for: a
/// fill that would have stopped at iteration 100 on one card and 101 on another
/// stops at the same batch boundary on both.
const CHECK_EVERY: u32 = 16;

/// Where a fill starts from.
#[derive(Clone, Copy, Debug)]
pub enum Seed {
    /// A surface far above the landscape. Correct from any state, and slow:
    /// information travels one cell an iteration, so this costs about as many
    /// iterations as the grid is wide.
    Cold,
    /// The previous fill, raised by the largest rise anywhere since it was
    /// computed. Only valid if [`Flood::result`] still holds that fill.
    Warm { lift: f32 },
}

/// How far along each direction one iteration looks.
///
/// The plain one-cell stencil is correct and far too slow: a surface settles at
/// one cell an iteration and this grid is 3073 cells across, which measured as
/// 2736 iterations and 1.4 s for a cold fill. Warm-starting from the previous
/// round does not help -- three quarters of the cells fall in every round of
/// incision, so the fill genuinely changes almost everywhere and a tighter seed
/// saves nothing (1952 iterations either way, measured). The only lever left is
/// how far news travels per iteration, which is this.
pub const REACH: u32 = 16;

/// Cells a tiled workgroup writes back, and iterations it runs before doing so.
///
/// Must match `TILE` and `HALO` in `flood.wgsl`; the dispatch is sized in tiles
/// here and the patch is laid out in workgroup memory there, and the two
/// disagreeing would leave stripes of the grid never written.
const TILE: u32 = 48;
const PATCH_ITERATIONS: u32 = 8;

/// The most iterations a fill will run before giving up and saying so.
///
/// A relaxation from above cannot fail to converge, so hitting this means the
/// seed was not above the answer after all -- which is not a slow fill but a
/// wrong one, and it should be loud rather than silent. It also stops an
/// oscillating fill from hanging a run: seeded from below, the iteration can
/// rise as well as fall and need never settle at all.
const MOST_ITERATIONS: u32 = 8192;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    rows: u32,
    lift: f32,
    reach: u32,
}

pub struct Flood {
    extent: Extent,
    params: wgpu::Buffer,
    seed_cold: wgpu::ComputePipeline,
    seed_warm: wgpu::ComputePipeline,
    fill: wgpu::ComputePipeline,
    fill_tiled: wgpu::ComputePipeline,
    /// Whether to relax a patch at a time in workgroup memory. Off only for the
    /// measurement that compares the two.
    tiled: bool,
    /// The two surfaces the relaxation alternates between.
    surfaces: [wgpu::Buffer; 2],
    /// `groups[i]` reads `surfaces[i]` and writes `surfaces[i ^ 1]`.
    groups: [wgpu::BindGroup; 2],
    /// Which surface holds the live answer.
    live: usize,
    changed: wgpu::Buffer,
    /// How far one iteration looks; [`REACH`] outside the measurements.
    reach: u32,
}

impl Flood {
    pub fn new(gpu: &Gpu, extent: Extent, height: &wgpu::Buffer) -> Self {
        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("flood"),
                source: wgpu::ShaderSource::Wgsl(include_str!("flood.wgsl").into()),
            });

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        };
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("flood layout"),
                entries: &[
                    entry(
                        0,
                        wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    ),
                    entry(1, storage(true)),
                    entry(2, storage(true)),
                    entry(3, storage(false)),
                    entry(4, storage(false)),
                ],
            });
        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("flood pipeline layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
        let pipeline = |label, entry_point| {
            gpu.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        let params = gpu.uniform(
            "flood params",
            &Params {
                width: extent.width,
                rows: extent.rows,
                lift: 0.0,
                reach: REACH,
            },
        );
        let surfaces = [
            gpu.storage("flood surface a", extent.channel_bytes()),
            gpu.storage("flood surface b", extent.channel_bytes()),
        ];
        let changed = gpu.storage("flood changed", size_of::<u32>() as u64);

        let group = |previous: &wgpu::Buffer, next: &wgpu::Buffer| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flood"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: height.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: previous.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: next.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: changed.as_entire_binding(),
                    },
                ],
            })
        };
        let groups = [
            group(&surfaces[0], &surfaces[1]),
            group(&surfaces[1], &surfaces[0]),
        ];

        Self {
            extent,
            params,
            seed_cold: pipeline("flood seed cold", "cs_seed_cold"),
            seed_warm: pipeline("flood seed warm", "cs_seed_warm"),
            fill_tiled: pipeline("flood fill tiled", "cs_fill_tiled"),
            tiled: false,
            fill: pipeline("flood fill", "cs_fill"),
            surfaces,
            groups,
            live: 0,
            changed,
            reach: REACH,
        }
    }

    /// Overrides how far an iteration looks, for the measurement that chose it.
    #[cfg(test)]
    pub fn with_reach(mut self, reach: u32) -> Self {
        self.reach = reach;
        self.tiled = false;
        self
    }

    /// Relaxes a patch at a time in workgroup memory, for the measurement that
    /// found it slower than relaxing the whole grid.
    #[cfg(test)]
    pub fn tiled(mut self) -> Self {
        self.tiled = true;
        self
    }

    /// The buffer holding the surface as it stands.
    pub fn result(&self) -> &wgpu::Buffer {
        &self.surfaces[self.live]
    }

    /// Relaxes to the filled surface, and says how many iterations it took.
    ///
    /// The count is the measurement the schedule depends on: a warm start that
    /// needs as many iterations as a cold one means the eighty rounds of
    /// incision cost eighty cold fills, and something else has to change.
    pub fn fill(&mut self, gpu: &Gpu, seed: Seed) -> u32 {
        let lift = match seed {
            Seed::Cold => 0.0,
            // Never negative: the seed has to be an upper bound, and a
            // landscape that only fell this round still gets its previous fill
            // unchanged rather than one lowered by guesswork.
            Seed::Warm { lift } => lift.max(0.0),
        };
        gpu.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                width: self.extent.width,
                rows: self.extent.rows,
                lift,
                reach: self.reach,
            }),
        );

        let (across, down) = self.extent.workgroups();
        let encode = |pipeline: &wgpu::ComputePipeline, group: &wgpu::BindGroup, label| {
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(across, down, 1);
            }
            encoder
        };

        let seeding = match seed {
            Seed::Cold => &self.seed_cold,
            Seed::Warm { .. } => &self.seed_warm,
        };
        let encoder = encode(seeding, &self.groups[self.live], "flood seed");
        gpu.queue.submit(std::iter::once(encoder.finish()));
        self.live ^= 1;

        // A tiled dispatch settles a patch `PATCH_ITERATIONS` times over before
        // it writes anything back, so it needs proportionally fewer of them
        // between convergence checks -- and each one covers a whole tile of
        // cells rather than a workgroup's worth.
        let (pipeline, tiles, per_dispatch) = if self.tiled {
            let tiles = (
                self.extent.width.div_ceil(TILE),
                self.extent.rows.div_ceil(TILE),
            );
            (&self.fill_tiled, tiles, PATCH_ITERATIONS)
        } else {
            (&self.fill, self.extent.workgroups(), 1)
        };
        let dispatches = (CHECK_EVERY / per_dispatch).max(1);

        let mut iterations = 0;
        loop {
            gpu.queue.write_buffer(&self.changed, 0, &0u32.to_le_bytes());
            let mut encoder = gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flood batch"),
                });
            for _ in 0..dispatches {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("flood fill"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &self.groups[self.live], &[]);
                pass.dispatch_workgroups(tiles.0, tiles.1, 1);
                drop(pass);
                self.live ^= 1;
                iterations += per_dispatch;
            }
            gpu.queue.submit(std::iter::once(encoder.finish()));

            let flag: Vec<u32> = gpu.download(&self.changed, 1);
            if flag[0] == 0 {
                break iterations;
            }
            if iterations >= MOST_ITERATIONS {
                // Loud, because this is not a fill that took a long time: it is
                // a fill that was seeded below its own answer and is not
                // descending towards anything. The surface handed back is
                // whatever the last iteration happened to leave.
                log::error!(
                    "the filled surface did not settle in {iterations} iterations; \
                     the seed cannot have been above it"
                );
                break iterations;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::fields::Fields;
    use crate::gpu::test_gpu;

    /// A landscape with hollows in it: a bowl in the middle of a plain that
    /// falls away east, and a scatter of pits to make sure the fill is not
    /// merely getting one big feature right.
    fn hollowed(side: usize) -> Fields {
        let mut fields = Fields::new([(side - 1) as f32 * 10.0, (side - 1) as f32 * 10.0], 10.0);
        let middle = (side - 1) as f32 * 0.5;
        for row in 0..side {
            for column in 0..side {
                let index = fields.height.index(column, row);
                let (dx, dy) = (column as f32 - middle, row as f32 - middle);
                let bowl = (400.0 - (dx * dx + dy * dy)).max(0.0) * 0.1;
                // A plain tilted east, so there is somewhere for the water to
                // leave, and a pit every seventh cell so the fill has many
                // small basins to settle as well as one large one.
                let pit = if column % 7 == 3 && row % 7 == 5 { 12.0 } else { 0.0 };
                fields.height.values[index] = 100.0 - column as f32 * 0.5 - bowl - pit;
            }
        }
        fields
    }

    fn extent_of(fields: &Fields) -> Extent {
        Extent {
            width: fields.width() as u32,
            rows: fields.rows() as u32,
        }
    }

    /// The relaxation and the heap have to agree exactly, not approximately.
    ///
    /// Exactly is a fair thing to demand of these two: every value in a filled
    /// surface is some cell's own height carried across the map by `min` and
    /// `max`, and neither of those rounds. A tolerance here would hide the one
    /// failure that matters -- a basin left unfilled because the relaxation
    /// stopped early -- behind a number small enough to look like arithmetic.
    #[test]
    fn the_relaxation_fills_exactly_what_the_flood_fills() {
        let fields = hollowed(97);
        let expected = crate::flow::drainage(&fields).filled;

        let gpu = test_gpu();
        let heights = gpu.uploaded("height", &fields.height.values);
        let mut flood = Flood::new(&gpu, extent_of(&fields), &heights);
        flood.fill(&gpu, Seed::Cold);
        let filled: Vec<f32> = gpu.download(flood.result(), fields.height.values.len());

        let wrong = filled
            .iter()
            .zip(&expected)
            .enumerate()
            .filter(|(_, (got, want))| got != want)
            .count();
        assert_eq!(wrong, 0, "{wrong} cells differ from the flood's surface");
        // And it has to have done something: a fill that returned the ground
        // untouched would also pass a comparison against a broken oracle.
        let raised = filled
            .iter()
            .zip(&fields.height.values)
            .filter(|(filled, ground)| filled > ground)
            .count();
        assert!(raised > 100, "only {raised} cells were filled at all");
    }

    /// A warm start has to land where a cold one does, whichever way ground moved.
    ///
    /// Two mutations, because the two directions fail differently and only one
    /// of them is obvious. **Lowering** ground lowers the fill, so the previous
    /// surface is still above the answer and the relaxation walks down to it.
    /// **Raising** a basin's rim raises the fill *above* the previous surface,
    /// and a seed that trusted the previous surface unadjusted would start below
    /// the answer -- where this iteration does not converge to it at all, but
    /// creeps up a lattice step per sweep and stops wherever the sweeps ran out.
    /// `lift` is what buys the bound back, and the raised rim below is what
    /// makes its absence show: without it this test passes the ordinary case and
    /// says nothing about the one that matters.
    #[test]
    fn a_warm_start_lands_where_a_cold_one_does() {
        let mut fields = hollowed(97);
        let gpu = test_gpu();
        let heights = gpu.uploaded("height", &fields.height.values);
        let mut flood = Flood::new(&gpu, extent_of(&fields), &heights);
        flood.fill(&gpu, Seed::Cold);

        let width = fields.width();
        let middle = (fields.width() - 1) as f32 * 0.5;
        let radius_squared = |column: usize, row: usize| {
            let (dx, dy) = (column as f32 - middle, row as f32 - middle);
            dx * dx + dy * dy
        };

        // A closed wall around the bowl, which lifts the level it spills at and
        // so lifts the water inside it. A ridge that did not enclose anything
        // would leave every fill exactly where it was and test nothing.
        for row in 0..fields.rows() {
            for column in 0..width {
                let r2 = radius_squared(column, row);
                if (400.0..=520.0).contains(&r2) {
                    fields.height.values[row * width + column] += 8.0;
                }
            }
        }
        gpu.queue
            .write_buffer(&heights, 0, bytemuck::cast_slice(&fields.height.values));
        flood.fill(&gpu, Seed::Warm { lift: 8.0 });
        let raised: Vec<f32> = gpu.download(flood.result(), fields.height.values.len());
        let expected = crate::flow::drainage(&fields).filled;
        let wrong = raised
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got != want)
            .count();
        assert_eq!(wrong, 0, "{wrong} cells differ after the rim was raised");

        // And the other direction: breach the wall, so the water that was just
        // dammed drains away and the fill drops well below the previous surface.
        for row in 44..52 {
            for column in 0..width {
                if radius_squared(column, row) >= 380.0 {
                    fields.height.values[row * width + column] -= 40.0;
                }
            }
        }
        gpu.queue
            .write_buffer(&heights, 0, bytemuck::cast_slice(&fields.height.values));
        flood.fill(&gpu, Seed::Warm { lift: 0.0 });
        let breached: Vec<f32> = gpu.download(flood.result(), fields.height.values.len());
        let expected = crate::flow::drainage(&fields).filled;
        let wrong = breached
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got != want)
            .count();
        assert_eq!(wrong, 0, "{wrong} cells differ after the wall was breached");
        // The breach has to have actually drained something, or the second half
        // of this test is comparing two identical surfaces and proving nothing.
        let drained = breached
            .iter()
            .zip(&raised)
            .filter(|(after, before)| after < before)
            .count();
        assert!(drained > 100, "only {drained} cells drained after the breach");
    }

    /// How far an iteration should look, measured rather than guessed.
    ///
    /// Looking `reach` cells along each of eight directions costs up to eight
    /// times the reads of the plain stencil and saves however many iterations
    /// the extra distance buys. Neither side of that is worth predicting: the
    /// rays quit early wherever they climb above the answer, so the cost per
    /// iteration is nothing like eight times, and the saving depends on how
    /// straight the paths out of this landscape are. Run it with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore = "a measurement on the full grid, not a check"]
    fn measure_how_far_an_iteration_should_look() {
        let mut fields = Fields::new([49152.0, 57344.0], 16.0);
        crate::shape::raise(
            &mut fields,
            crate::shape::Relief {
                valley_metres: 700.0,
                peak_metres: 2600.0,
            },
            0,
        );
        crate::thermal::relax(&mut fields, crate::thermal::Settling::Bedrock);

        let gpu = test_gpu();
        let heights = gpu.uploaded("height", &fields.height.values);
        let mut reference: Option<Vec<f32>> = None;
        for reach in [1u32, 2, 4, 8, 16, 32, 64] {
            let mut flood = Flood::new(&gpu, extent_of(&fields), &heights).with_reach(reach);
            let at = std::time::Instant::now();
            let iterations = flood.fill(&gpu, Seed::Cold);
            let elapsed = at.elapsed();
            // Every reach has to land on the same surface. A wider ray that
            // took a shortcut it was not entitled to would look like a
            // wonderful speed-up and be a different landscape.
            let filled: Vec<f32> = gpu.download(flood.result(), fields.height.values.len());
            let differs = reference
                .as_ref()
                .map(|first| filled.iter().zip(first).filter(|(a, b)| a != b).count())
                .unwrap_or(0);
            println!(
                "reach {reach:2}: {iterations:5} iterations in {elapsed:.1?} \
                 ({:.2?} each), {differs} cells differ from reach 1",
                elapsed / iterations.max(1),
            );
            assert_eq!(differs, 0, "reach {reach} settled on a different surface");
            reference.get_or_insert(filled);
        }

        // And the tiled relaxation, which buys its distance the other way:
        // instead of one iteration reading further, eight iterations run inside
        // workgroup memory for one trip through the grid.
        let mut flood = Flood::new(&gpu, extent_of(&fields), &heights).tiled();
        let at = std::time::Instant::now();
        let iterations = flood.fill(&gpu, Seed::Cold);
        let elapsed = at.elapsed();
        let filled: Vec<f32> = gpu.download(flood.result(), fields.height.values.len());
        let differs = reference
            .as_ref()
            .map(|first| filled.iter().zip(first).filter(|(a, b)| a != b).count())
            .unwrap_or(0);
        println!(
            "tiled:      {iterations:5} iterations in {elapsed:.1?}, {differs} cells differ"
        );
        assert_eq!(differs, 0, "the tiled fill settled on a different surface");
    }

    /// What share of a round of incision the flood it replaces actually is.
    ///
    /// Decides how much is left to win if the filled surface stays on the CPU:
    /// everything else in a round -- the receivers, the shared drainage area,
    /// the implicit sweep, the creep -- is either already parallel or a scan
    /// that ports cleanly, so this number is the floor the rest would run into.
    #[test]
    #[ignore = "a measurement on the full grid, not a check"]
    fn measure_what_share_of_a_round_the_flood_is() {
        // The phase timings inside `flow::drainage` are logged rather than
        // returned, and a test has no logger unless it builds one.
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(false)
            .try_init();
        let mut fields = Fields::new([49152.0, 57344.0], 16.0);
        crate::shape::raise(
            &mut fields,
            crate::shape::Relief {
                valley_metres: 700.0,
                peak_metres: 2600.0,
            },
            0,
        );
        crate::thermal::relax(&mut fields, crate::thermal::Settling::Bedrock);

        let at = std::time::Instant::now();
        let drainage = crate::flow::drainage(&fields);
        let whole = at.elapsed();
        println!(
            "flow::drainage over {} cells took {whole:.1?}, reaching {} of them",
            fields.height.values.len(),
            drainage.order.len()
        );

        let at = std::time::Instant::now();
        crate::incise::rivers(&mut fields, 1);
        println!(
            "a whole round of incision took {:.1?}, of which the drainage above was {:.0}%",
            at.elapsed(),
            whole.as_secs_f64() * 100.0 / at.elapsed().as_secs_f64()
        );
    }

    /// What the eighty rounds of incision will actually cost.
    ///
    /// Not an assertion: the number this prints is the one the whole GPU port
    /// was planned around -- if a warm start needs as many iterations as a cold
    /// one, the rounds cost eighty cold fills and the plan needs rewriting. Run
    /// it with `--ignored --nocapture`; it drives the real grid through real
    /// rounds of the CPU incision, which is minutes rather than seconds.
    #[test]
    #[ignore = "a measurement on the full grid, not a check"]
    fn measure_what_a_warm_start_saves() {
        const ROUNDS: usize = 8;

        let mut fields = Fields::new([49152.0, 57344.0], 16.0);
        crate::shape::raise(
            &mut fields,
            crate::shape::Relief {
                valley_metres: 700.0,
                peak_metres: 2600.0,
            },
            0,
        );
        crate::thermal::relax(&mut fields, crate::thermal::Settling::Bedrock);
        println!(
            "grid {} x {} cells, {} in all",
            fields.width(),
            fields.rows(),
            fields.width() * fields.rows()
        );

        let gpu = test_gpu();
        let heights = gpu.uploaded("height", &fields.height.values);
        let mut flood = Flood::new(&gpu, extent_of(&fields), &heights);
        // A second relaxation over the same heights, seeded without the lift.
        // That seed is *unsound* -- it can start below the answer where ground
        // rose, which is the whole reason `lift` exists -- so the surface it
        // lands on is not to be believed. Its iteration count is, and that is
        // the diagnostic: if dropping the lift collapses the count then the
        // global lift is what costs the warm start its advantage, and a tighter
        // per-cell bound is the fix. If it does not, the map genuinely changes
        // everywhere each round and no bound will help.
        let mut unsound = Flood::new(&gpu, extent_of(&fields), &heights);
        unsound.fill(&gpu, Seed::Cold);

        let at = std::time::Instant::now();
        let cold = flood.fill(&gpu, Seed::Cold);
        println!("cold start: {cold} iterations in {:.1?}", at.elapsed());

        for round in 0..ROUNDS {
            let before = fields.height.values.clone();
            let at = std::time::Instant::now();
            crate::incise::rivers(&mut fields, 1);
            let cpu = at.elapsed();
            let lift = before
                .iter()
                .zip(&fields.height.values)
                .fold(0.0f32, |most, (was, now)| most.max(now - was));
            let fell = before
                .iter()
                .zip(&fields.height.values)
                .filter(|(was, now)| now < was)
                .count();
            gpu.queue
                .write_buffer(&heights, 0, bytemuck::cast_slice(&fields.height.values));

            let at = std::time::Instant::now();
            let warm = flood.fill(&gpu, Seed::Warm { lift });
            let sound = at.elapsed();
            let bare = unsound.fill(&gpu, Seed::Warm { lift: 0.0 });
            println!(
                "round {round}: warm {warm} iterations in {sound:.1?}, {bare} without the \
                 {lift:.3} m lift; {:.1}% of cells fell (the CPU round took {cpu:.1?})",
                fell as f64 * 100.0 / before.len() as f64,
            );
        }
    }
}
