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
}

/// The channels, resident, and the pipeline that reads them.
pub struct Texels {
    gpu_extent: Extent,
    params: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bare: wgpu::ComputePipeline,
    /// The five erosion channels, uploaded once and never read back.
    channels: Vec<wgpu::Buffer>,
    out_height: wgpu::Buffer,
    tile_size: u32,
    seed: u32,
    cell_metres: f32,
}

impl Texels {
    /// Uploads the channels and builds the pipeline.
    ///
    /// The upload happens once for a whole run: the fields do not change while
    /// tiles are being written, and at 42 MB a channel there is no reason to
    /// send them again per tile.
    pub fn new(gpu: &Gpu, fields: &Fields, tile_size: u32, seed: u32) -> Self {
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
        let out_height = gpu.storage(
            "tile heights",
            u64::from(tile_size) * u64::from(tile_size) * size_of::<f32>() as u64,
        );

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
        let layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("emit layout"),
                entries: &entries,
            });
        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("emit pipeline layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                });
        let bare = gpu
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("emit bare"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("cs_bare"),
                compilation_options: Default::default(),
                cache: None,
            });

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
                },
            ),
            layout,
            bare,
            channels,
            out_height,
            tile_size,
            seed,
            cell_metres: fields.metres_per_cell,
        }
    }

    /// The bare earth over one tile, in raster metres from `origin`.
    ///
    /// `origin` is the centre of the tile's first texel, which is what
    /// `emit::texel_metres` hands the CPU functions -- half a texel in, because
    /// the format's rasters are `PixelIsArea` and a texel is a square of ground
    /// rather than a point.
    pub fn bare_tile(&self, gpu: &Gpu, origin: [f32; 2], texel_metres: f32) -> Vec<f32> {
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
            pass.set_pipeline(&self.bare);
            pass.set_bind_group(0, &group, &[]);
            let across = self.tile_size.div_ceil(GROUP);
            pass.dispatch_workgroups(across, across, 1);
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));

        gpu.download(&self.out_height, (self.tile_size * self.tile_size) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::detail;
    use crate::gpu::test_gpu;
    use crate::shape::Relief;

    /// A landscape with every channel filled, so nothing being compared is
    /// reading a field that happens to be zero everywhere.
    fn landscape(cells: f32) -> Fields {
        let mut fields = Fields::new([cells, cells], 16.0);
        crate::shape::raise(
            &mut fields,
            Relief {
                valley_metres: 700.0,
                peak_metres: 2600.0,
            },
            0,
        );
        crate::thermal::relax(&mut fields, crate::thermal::Settling::Bedrock);
        crate::incise::rivers(&mut fields, 2);
        crate::hydraulic::erode(&mut fields, 0, 1);
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
        let texels = Texels::new(&gpu, &fields, TILE, 0);
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
}
