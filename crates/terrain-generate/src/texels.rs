//! Evaluating the per-texel half of the generator on the GPU.
//!
//! One dispatch per tile, reading the erosion channels straight out of the
//! buffers they were uploaded into once. See `emit.wgsl` for the transcription
//! and why this half is the one worth moving.
//!
//! This is the first of three pieces. It computes the bare earth -- the coarse
//! surface with its band-limited fractal detail -- and nothing that stands on
//! it. The crowns and stones are the expensive part and the classifier decides
//! where they go, so neither can land until the classifier does; what this
//! pins down first is the foundation both of them read: the field sampling, the
//! noise, and the band limit. Every one of those is checked against the Rust it
//! was transcribed from, on the same landscape, texel for texel.

use crate::fields::Fields;
use crate::gpu::{Extent, GROUP, Gpu};
use crate::shape::Relief;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    width: u32,
    rows: u32,
    cell_metres: f32,
    texel_metres: f32,
    origin: [f32; 2],
    tile_size: u32,
    seed: u32,
    valley_metres: f32,
    peak_metres: f32,
}

/// The channels, resident, and the pipeline that reads them.
pub struct Texels {
    gpu_extent: Extent,
    params: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bare: wgpu::ComputePipeline,
    cover: wgpu::ComputePipeline,
    /// The five erosion channels, uploaded once and never read back.
    channels: Vec<wgpu::Buffer>,
    /// Wide enough for the five per-texel shares `cs_cover` writes as well as
    /// the one height `cs_bare` does.
    out_height: wgpu::Buffer,
    out_cover: wgpu::Buffer,
    tile_size: u32,
    seed: u32,
    cell_metres: f32,
    relief: Relief,
}

impl Texels {
    /// Uploads the channels and builds the pipeline.
    ///
    /// The upload happens once for a whole run: the fields do not change while
    /// tiles are being written, and at 42 MB a channel there is no reason to
    /// send them again per tile.
    pub fn new(gpu: &Gpu, fields: &Fields, tile_size: u32, seed: u32, relief: Relief) -> Self {
        let gpu_extent = Extent {
            width: fields.width() as u32,
            rows: fields.rows() as u32,
        };
        let channels = vec![
            gpu.uploaded("height", &fields.height.values),
            gpu.uploaded("hardness", &fields.hardness.values),
            gpu.uploaded("flow", &fields.flow.values),
            gpu.uploaded("deposit", &fields.deposit.values),
            gpu.uploaded("filled", &fields.filled.values),
        ];
        let texels = u64::from(tile_size) * u64::from(tile_size);
        let out_height = gpu.storage("tile heights", texels * 5 * size_of::<f32>() as u64);
        let out_cover = gpu.storage("tile cover", texels * size_of::<u32>() as u64);

        let module = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("emit"),
                source: wgpu::ShaderSource::Wgsl(include_str!("emit.wgsl").into()),
            });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let mut entries = vec![entry(
            0,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        )];
        for binding in 1..=5 {
            entries.push(entry(binding, storage(true)));
        }
        entries.push(entry(6, storage(false)));
        entries.push(entry(7, storage(false)));
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("emit layout"),
                entries: &entries,
            });
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("emit pipeline layout"),
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

        Self {
            gpu_extent,
            params: gpu.uniform(
                "emit params",
                &Params {
                    width: gpu_extent.width,
                    rows: gpu_extent.rows,
                    cell_metres: fields.metres_per_cell,
                    texel_metres: 1.0,
                    origin: [0.0, 0.0],
                    tile_size,
                    seed,
                    valley_metres: relief.valley_metres,
                    peak_metres: relief.peak_metres,
                },
            ),
            layout,
            bare: pipeline("emit bare", "cs_bare"),
            cover: pipeline("emit cover", "cs_cover"),
            channels,
            out_height,
            out_cover,
            tile_size,
            seed,
            cell_metres: fields.metres_per_cell,
            relief,
        }
    }

    /// The bare earth over one tile, in raster metres from `origin`.
    ///
    /// `origin` is the centre of the tile's first texel, which is what
    /// `emit::texel_metres` hands the CPU functions -- half a texel in, because
    /// the format's rasters are `PixelIsArea` and a texel is a square of ground
    /// rather than a point.
    pub fn bare_tile(&self, gpu: &Gpu, origin: [f32; 2], texel_metres: f32) -> Vec<f32> {
        self.run(gpu, &self.bare, origin, texel_metres);
        gpu.download(&self.out_height, (self.tile_size * self.tile_size) as usize)
    }

    /// The classifier's three answers over one tile: the ground cover, and the
    /// five shares that say what grows and what lies on it.
    pub fn cover_tile(
        &self,
        gpu: &Gpu,
        origin: [f32; 2],
        texel_metres: f32,
    ) -> (Vec<u32>, Vec<f32>) {
        self.run(gpu, &self.cover, origin, texel_metres);
        let texels = (self.tile_size * self.tile_size) as usize;
        (
            gpu.download(&self.out_cover, texels),
            gpu.download(&self.out_height, texels * 5),
        )
    }

    fn run(
        &self,
        gpu: &Gpu,
        pipeline: &wgpu::ComputePipeline,
        origin: [f32; 2],
        texel_metres: f32,
    ) {
        gpu.queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                width: self.gpu_extent.width,
                rows: self.gpu_extent.rows,
                cell_metres: self.cell_metres,
                texel_metres,
                origin,
                tile_size: self.tile_size,
                seed: self.seed,
                valley_metres: self.relief.valley_metres,
                peak_metres: self.relief.peak_metres,
            }),
        );

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: self.params.as_entire_binding(),
        }];
        for (index, channel) in self.channels.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: index as u32 + 1,
                resource: channel.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: self.out_height.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 7,
            resource: self.out_cover.as_entire_binding(),
        });
        let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("emit"),
            layout: &self.layout,
            entries: &entries,
        });

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("emit tile"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("emit bare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            let across = self.tile_size.div_ceil(GROUP);
            pass.dispatch_workgroups(across, across, 1);
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::detail;
    use crate::gpu::test_gpu;

    fn relief() -> Relief {
        Relief {
            valley_metres: 700.0,
            peak_metres: 2600.0,
        }
    }

    /// A landscape with every channel filled, so nothing being compared is
    /// reading a field that happens to be zero everywhere.
    fn landscape(cells: f32) -> Fields {
        let mut fields = Fields::new([cells, cells], 16.0);
        crate::shape::raise(&mut fields, relief(), 0);
        crate::thermal::relax(&mut fields, crate::thermal::Settling::Bedrock);
        // Enough cutting to leave ground steep enough to classify as rock.
        // Two rounds left a landscape of gentle wooded hillside, where the
        // bare-rock threshold is never approached and the mottling that nudges
        // it could be deleted without any test noticing.
        crate::incise::rivers(&mut fields, 30);
        crate::hydraulic::erode(&mut fields, 0, 3);
        // As the real run does, and it is not cosmetic here: the classifier's
        // bands are shares of the relief, so a landscape left at whatever range
        // the erosion happened to leave never reaches its own treeline and half
        // the branches -- every alpine one, the rock, the ice -- go untested.
        crate::shape::rescale(&mut fields.height, relief());
        crate::flow::route(&mut fields);
        fields
    }

    /// The shader and the Rust have to agree about the bare earth.
    ///
    /// Not exactly: the two evaluate the same arithmetic in a different order on
    /// different hardware, and WGSL is free to contract a multiply and an add
    /// where rustc is not. What they must not do is disagree *structurally* --
    /// a band limit counted differently, a channel sampled from the wrong
    /// lattice, a hash that diverges after the first octave -- and all of those
    /// show up as metres rather than as millimetres on ground that spans 1900 m.
    ///
    /// Every level, because the band limit is what changes with level and it is
    /// the part most easily got wrong in a way a single level would hide.
    #[test]
    fn the_shader_and_the_crate_agree_about_bare_earth() {
        const TILE: u32 = 64;

        let fields = landscape(8192.0);
        let gpu = test_gpu();
        let texels = Texels::new(&gpu, &fields, TILE, 0, relief());
        let cell = fields.metres_per_cell;

        for level in 3..=8u32 {
            let texel = (1u32 << level) as f32;
            // Well inside the raster, so no sample is answered by the edge
            // clamp -- which both sides do, but agreeing about clamped ground
            // would not be evidence about the interpolation.
            let origin = [2048.0 + 0.5 * texel, 2048.0 + 0.5 * texel];
            let got = texels.bare_tile(&gpu, origin, texel);

            let mut worst = 0.0f32;
            for row in 0..TILE {
                for column in 0..TILE {
                    let x = origin[0] + column as f32 * texel;
                    let y = origin[1] + row as f32 * texel;
                    let sample = fields.sample(x, y);
                    let ground = detail::ground(&sample, texel, cell);
                    let want = detail::height(&sample, &ground, x, y, texel, 0);
                    let mine = got[(row * TILE + column) as usize];
                    worst = worst.max((want - mine).abs());
                }
            }
            assert!(
                worst < 0.002,
                "level {level} differs by up to {worst:.4} m from the crate",
            );
        }
    }

    /// The classifier has to reach the same verdict as the crate, texel for
    /// texel, and it is the one layer where "close" is not a defence.
    ///
    /// A material id is a label. Two ids either match or they do not, and a
    /// texel that comes out `Scree` on one side and `BareRock` on the other is
    /// a visible patch of the wrong colour however small the number that
    /// decided it was.
    ///
    /// Measured, they agree on **every texel of every level** -- not nearly
    /// all, all of them. The allowance below is not slack being taken up; it is
    /// there because the branches are stacked thresholds fed by values that
    /// differ in the last bit or two between the two implementations, so a
    /// texel sitting exactly on a threshold could fall either way on a card
    /// that rounds differently. Four texels in four thousand is far too few to
    /// hide a rule transcribed wrongly, which shows up as a region rather than
    /// as a speckle.
    ///
    /// The stones and trees are compared as numbers, since those are shares
    /// rather than labels, but only where both sides picked the same cover:
    /// a texel that fell differently at a threshold is expected to carry
    /// different things, and counting it twice would say nothing new.
    ///
    /// # What this does not catch
    ///
    /// A term that only moves a threshold slightly, where the landscape rarely
    /// sits. Deleting the `mottle * 0.10` nudge from the bare-rock threshold
    /// passes this test, and it was checked: `rockiness` is bimodal, because
    /// `steepness` is a smoothstep that has saturated at one end or the other
    /// on nearly all ground, so almost nothing lands in the tenth-wide band
    /// where that term decides anything. Widening the fixture until some texel
    /// does would cost more than the term is worth -- it breaks up the edge of
    /// a rock face so it does not read as a contour line, which is a thing to
    /// judge in a render rather than in a count. What this test is for is a
    /// rule transcribed wrongly, which moves regions: the same check catches
    /// swapping one `&&` for a `||` in the scree rule on 196 texels.
    #[test]
    fn the_shader_and_the_crate_agree_about_cover() {
        const TILE: u32 = 64;

        let fields = landscape(8192.0);
        let gpu = test_gpu();
        let texels = Texels::new(&gpu, &fields, TILE, 0, relief());
        let cell = fields.metres_per_cell;

        let mut seen = std::collections::BTreeSet::new();
        for level in 3..=8u32 {
            let texel = (1u32 << level) as f32;
            let (mut differing, mut worst_share) = (0usize, 0.0f32);
            let mut counted = 0usize;
            // Several windows rather than one. A single 64-texel window is a
            // few hundred metres of one hillside, and a rule that only bites
            // near a threshold needs enough ground for some texel to sit near
            // it: with one window, dropping the mottling from the bare-rock
            // threshold changed nothing at all and the test passed.
            for corner in [1024.0f32, 2560.0, 4096.0, 5632.0] {
                let origin = [corner + 0.5 * texel, corner + 0.5 * texel];
                let (ids, shares) = texels.cover_tile(&gpu, origin, texel);

                for row in 0..TILE {
                    for column in 0..TILE {
                        let at = (row * TILE + column) as usize;
                        let x = origin[0] + column as f32 * texel;
                        let y = origin[1] + row as f32 * texel;
                        let sample = fields.sample(x, y);
                        let ground = detail::ground(&sample, texel, cell);
                        let want =
                            crate::classify::material(&sample, &ground, x, y, texel, 0, relief());
                        seen.insert(want.id());
                        counted += 1;
                        if want.id() != ids[at] {
                            differing += 1;
                            continue;
                        }
                        let grown =
                            crate::classify::trees(&sample, &ground, x, y, texel, 0, relief());
                        let strewn =
                            crate::classify::rocks(&sample, &ground, x, y, texel, 0, relief());
                        for (want, got) in [
                            (grown.density, shares[at * 5]),
                            (grown.health, shares[at * 5 + 1]),
                            (strewn.boulders, shares[at * 5 + 2]),
                            (strewn.rubble, shares[at * 5 + 3]),
                            (strewn.stature, shares[at * 5 + 4]),
                        ] {
                            worst_share = worst_share.max((want - got).abs());
                        }
                    }
                }
            }
            let texels_here = counted as f64;
            let share = differing as f64 / texels_here;
            assert!(
                share < 0.001,
                "level {level}: {differing} of {texels_here} texels were classified \
                 differently ({:.2}%)",
                share * 100.0,
            );
            assert!(
                worst_share < 0.001,
                "level {level}: a tree or stone share differs by {worst_share:.5}",
            );
            // A landscape that came back as one material would agree with
            // itself perfectly and prove nothing about any of the branches.
        }
        // A landscape that came back as one material would agree with itself
        // perfectly and prove nothing about any of the branches. Ten is what
        // the fixture actually reaches, across the levels between them.
        assert!(
            seen.len() >= 10,
            "only {} materials were exercised: {:?}",
            seen.len(),
            seen.iter()
                .map(|id| format!("{id:#06x}"))
                .collect::<Vec<_>>(),
        );
    }
}
