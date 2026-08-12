//! The noise a cloud is carved out of.
//!
//! Two 3D volumes, built once at load and read for the rest of the run: a
//! shape, which says where cloud is at all, and a detail, which says what its
//! edges look like. Both tile, which is what lets eight megabytes cover a
//! hundred kilometres of sky; see `src/cloud.wgsl` for how the lattice is
//! folded to make that true.
//!
//! Generated rather than shipped. A 128-cubed volume is 8 MB of file that would
//! have to live somewhere, be found at run time, and be kept in step with the
//! shader that reads it -- against a third of a second of compute that produces
//! the same bytes on every machine, every run. The same trade the scattering
//! tables in [`crate::sky`] make, and for the same reasons.

/// Side of the shape volume, in texels. Must match `SHAPE_SIZE` in
/// `src/cloud.wgsl`.
///
/// A hundred and twenty-eight, tiled over four kilometres, puts a texel at
/// about thirty metres -- fine enough that the billows have edges and coarse
/// enough that the whole volume is eight megabytes.
pub const SHAPE_SIZE: u32 = 128;

/// Side of the detail volume, in texels. Must match `DETAIL_SIZE` in
/// `src/cloud.wgsl`.
///
/// Tiled over a couple of hundred metres, so a texel is a few metres. It is
/// only ever read where the shape has already put cloud, so it does not need
/// the resolution the shape does -- what it needs is to be small enough to sit
/// in cache while the march is stepping through it.
pub const DETAIL_SIZE: u32 = 32;

/// Threads per workgroup, in each axis. Must match `@workgroup_size` on both
/// entry points in `src/cloud.wgsl`.
const GROUP: u32 = 4;

/// The format both volumes are held in.
///
/// `Rgba8Unorm` is storage-writable and filterable in core WebGPU, which is the
/// pair of properties that rules almost everything else out -- see
/// `FIELD_FORMAT` in `src/air.rs` and `LUT_FORMAT` in `src/sky.rs`, which hit
/// the same wall from different sides. Eight bits a channel is ample: this is
/// noise being thresholded, not a measurement, and the quantisation is far
/// below the softest edge the march can draw.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The two volumes, and the pipelines that fill them until they have been.
pub struct Cloud {
    #[allow(dead_code, reason = "read by the cloud march, which lands later")]
    shape: wgpu::Texture,
    #[allow(dead_code, reason = "read by the cloud march, which lands later")]
    detail: wgpu::Texture,
    shape_view: wgpu::TextureView,
    detail_view: wgpu::TextureView,
    /// Dropped once the volumes are filled; see [`Build`].
    build: Option<Build>,
}

/// Everything the build needs and the frame does not.
///
/// Dropped the moment the two dispatches have run, exactly as `Build` in
/// `src/sky.rs` and `src/air.rs` are. There is nothing large in it -- the
/// volumes themselves outlive it -- but a pipeline that can never run again is
/// a pipeline someone can be tempted to run.
struct Build {
    bind_group: wgpu::BindGroup,
    shape: wgpu::ComputePipeline,
    detail: wgpu::ComputePipeline,
}

impl Cloud {
    pub fn new(device: &wgpu::Device) -> Self {
        let volume = |label, size| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: size,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: FORMAT,
                // `COPY_SRC` so a test can read the volume back and say whether
                // it tiles, which is the one property that cannot be seen by
                // looking at a frame: a seam in a cloud looks like a cloud.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let shape = volume("cloud shape noise", SHAPE_SIZE);
        let detail = volume("cloud detail noise", DETAIL_SIZE);
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let (shape_view, detail_view) = (view(&shape), view(&detail));

        let written = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: FORMAT,
                view_dimension: wgpu::TextureViewDimension::D3,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud noise layout"),
            entries: &[written(0), written(1)],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud noise group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&shape_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&detail_view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cloud.wgsl").into()),
        });
        // Group 3, and nothing in the three below it. The convention is that
        // group 3 is where a pass's private bindings go; these two kernels have
        // no camera, no domain uniform and nothing shared to read, because they
        // are functions of position and a seed and of nothing else at all.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud noise pipeline layout"),
            bind_group_layouts: &[None, None, None, Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label, entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        Self {
            shape_view,
            detail_view,
            build: Some(Build {
                bind_group,
                shape: pipeline("cloud shape", "cs_cloud_shape"),
                detail: pipeline("cloud detail", "cs_cloud_detail"),
            }),
            shape,
            detail,
        }
    }

    /// Fills both volumes, once.
    ///
    /// Called from [`crate::scene::Scene::update`] on the first update, for the
    /// reason `Sky::ensure_built` is: filling them needs a queue, and a
    /// constructor has none. Waits for the GPU, so the cost lands at load with
    /// the tile reads and the scattering tables rather than being smeared
    /// across the opening frames of a flight, where it would read as the frame
    /// being slow.
    pub fn ensure_built(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(build) = self.build.take() else {
            return;
        };
        let started = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cloud noise"),
        });
        {
            // One pass for both: they write different textures and neither
            // reads what the other wrote, so there is nothing for a pass
            // boundary to order.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cloud noise"),
                timestamp_writes: None,
            });
            pass.set_bind_group(3, &build.bind_group, &[]);
            let groups = |size: u32| size.div_ceil(GROUP);
            pass.set_pipeline(&build.shape);
            pass.dispatch_workgroups(groups(SHAPE_SIZE), groups(SHAPE_SIZE), groups(SHAPE_SIZE));
            pass.set_pipeline(&build.detail);
            pass.dispatch_workgroups(
                groups(DETAIL_SIZE),
                groups(DETAIL_SIZE),
                groups(DETAIL_SIZE),
            );
        }
        queue.submit(std::iter::once(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        log::info!(
            "cloud: built {}^3 shape and {}^3 detail noise, {:.1} MiB, in {:.2?}",
            SHAPE_SIZE,
            DETAIL_SIZE,
            bytes() as f64 / (1024.0 * 1024.0),
            started.elapsed(),
        );
    }

    /// The two volumes.
    #[allow(dead_code, reason = "read by the cloud march, which lands later")]
    pub fn views(&self) -> (&wgpu::TextureView, &wgpu::TextureView) {
        (&self.shape_view, &self.detail_view)
    }
}

/// What the two volumes cost in video memory.
fn bytes() -> u64 {
    let cube = |size: u64| size * size * size * 4;
    cube(u64::from(SHAPE_SIZE)) + cube(u64::from(DETAIL_SIZE))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One volume read back off the GPU, one byte per channel.
    struct Volume {
        size: u32,
        texels: Vec<[u8; 4]>,
    }

    impl Volume {
        fn at(&self, x: u32, y: u32, z: u32) -> [u8; 4] {
            let size = self.size;
            self.texels[((z * size + y) * size + x) as usize]
        }

        /// The same, wrapping, so a caller can ask for the texel one past the
        /// end and be given the one the tiling says is there.
        fn wrapped(&self, x: i32, y: i32, z: i32) -> [u8; 4] {
            let fold = |v: i32| (v.rem_euclid(self.size as i32)) as u32;
            self.at(fold(x), fold(y), fold(z))
        }
    }

    fn read_volume(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        size: u32,
    ) -> Volume {
        // A texture-to-buffer copy wants its rows on a 256-byte stride. The
        // shape's rows are 512 bytes and need no padding; the detail's are 128
        // and do, so the slack is dropped on the way out -- the same
        // arrangement `crate::headless::capture` makes for a narrow frame.
        let packed = size * 4;
        let bytes_per_row = packed.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud readback"),
            size: u64::from(bytes_per_row * size * size),
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
                    rows_per_image: Some(size),
                },
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: size,
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

        let mut texels = Vec::with_capacity((size * size * size) as usize);
        for row in 0..size * size {
            let start = (row * bytes_per_row) as usize;
            for x in 0..size as usize {
                let at = start + x * 4;
                texels.push([mapped[at], mapped[at + 1], mapped[at + 2], mapped[at + 3]]);
            }
        }
        drop(mapped);
        readback.unmap();
        Volume { size, texels }
    }

    fn built() -> (Volume, Volume) {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let mut cloud = Cloud::new(&device);
        cloud.ensure_built(&device, &queue);
        (
            read_volume(&device, &queue, &cloud.shape, SHAPE_SIZE),
            read_volume(&device, &queue, &cloud.detail, DETAIL_SIZE),
        )
    }

    /// The largest step between neighbouring texels along one axis, and the
    /// largest step across the seam where the volume meets itself.
    ///
    /// Both measured over the same channel and the same volume, so they are
    /// directly comparable: if the field tiles, the seam is not a special place
    /// and its steps sit inside the range the interior's do.
    fn steps(volume: &Volume, channel: usize, axis: usize) -> (u32, u32) {
        let mut interior = 0;
        let mut seam = 0;
        let size = volume.size;
        for a in 0..size {
            for b in 0..size {
                for c in 0..size {
                    let at = |offset: i32| {
                        let mut p = [c as i32, a as i32, b as i32];
                        p[axis] += offset;
                        i32::from(volume.wrapped(p[0], p[1], p[2])[channel])
                    };
                    let step = (at(1) - at(0)).unsigned_abs();
                    // The step out of the last texel is the one that crosses
                    // the seam; every other is interior.
                    if c == size - 1 {
                        seam = seam.max(step);
                    } else {
                        interior = interior.max(step);
                    }
                }
            }
        }
        (interior, seam)
    }

    /// The volumes meet themselves without a seam, in all three axes.
    ///
    /// This is the property the whole build is arranged around, and it is
    /// invisible in a frame: a discontinuity in a cloud looks like a cloud. So
    /// it is measured rather than looked at -- the largest jump across the wrap
    /// must sit inside the largest jump the field makes anywhere else. A
    /// lattice that failed to fold would put a fresh, uncorrelated set of
    /// gradients on the far side of the boundary and the seam step would run to
    /// most of the range.
    #[test]
    fn the_noise_volumes_tile_seamlessly_in_every_axis() {
        let (shape, detail) = built();
        for (name, volume) in [("shape", &shape), ("detail", &detail)] {
            for channel in 0..4 {
                for axis in 0..3 {
                    let (interior, seam) = steps(volume, channel, axis);
                    assert!(
                        seam <= interior,
                        "{name} channel {channel} axis {axis}: the seam steps by \
                         {seam} where the interior never steps by more than {interior}"
                    );
                }
            }
        }
    }

    /// Every channel spreads across the range it is stored in.
    ///
    /// Two things at once. It says the dispatch ran -- an unwritten storage
    /// texture reads as zeroes, which would tile perfectly and pass the test
    /// above without a murmur. And it says the field is worth eight bits:
    /// what is stored here is thresholded to decide where cloud is, so a
    /// channel crowded into a narrow band spends its whole signal on a few of
    /// the 255 levels it has, and a cloud edge drawn through a few levels
    /// bands.
    ///
    /// The bound is on the standard deviation rather than on the extremes,
    /// because extremes are what a handful of outlying texels can supply on
    /// their own. It is set at 0.10, against a measured 0.15 for the shape's
    /// Perlin-Worley and 0.17 to 0.18 for every single-frequency Worley
    /// channel -- and against 0.078 for the first version of `perlin_worley`,
    /// which lerped towards solid the way Schneider's own does and which this
    /// bound is set to reject. The one channel that is deliberately narrower
    /// is the detail's fractal, at 0.12, because averaging three octaves is
    /// what it is for.
    #[test]
    fn every_channel_of_the_noise_uses_the_range_it_is_stored_in() {
        let (shape, detail) = built();
        for (name, volume) in [("shape", &shape), ("detail", &detail)] {
            for channel in 0..4 {
                let count = volume.texels.len() as f64;
                let value = |texel: &[u8; 4]| f64::from(texel[channel]) / 255.0;
                let mean = volume.texels.iter().map(value).sum::<f64>() / count;
                let spread = (volume
                    .texels
                    .iter()
                    .map(|t| (value(t) - mean).powi(2))
                    .sum::<f64>()
                    / count)
                    .sqrt();
                assert!(
                    spread > 0.10,
                    "{name} channel {channel} has a spread of {spread:.3}, which is \
                     too narrow for eight bits to hold without banding"
                );
                // And it sits in the middle of the range rather than against an
                // end, so a threshold sweeping the field meets it gradually
                // rather than all at once.
                assert!(
                    (0.3..0.7).contains(&mean),
                    "{name} channel {channel} averages {mean:.3}, which is off centre"
                );
            }
        }
    }

    /// Writes a slice of each volume out to look at.
    ///
    /// Ignored because it asserts nothing: it is the check the tests above
    /// cannot make. They say the field tiles and that it uses its range, and a
    /// field that did both and still looked like static rather than like cloud
    /// would pass them all. Run it with
    /// `cargo test --release -- --ignored dump_noise --nocapture` and open the
    /// PNGs it names.
    ///
    /// Two slices of the shape a quarter of the volume apart, so the tiling can
    /// be judged as well: the right edge of each is what meets its own left.
    #[test]
    #[ignore = "writes an image to look at rather than asserting anything"]
    fn dump_noise() {
        let (shape, detail) = built();
        let out = std::env::temp_dir();

        let write = |name: &str, volume: &Volume, slice: u32| {
            let size = volume.size;
            // One channel per row of the image, so all four are seen at once:
            // the shape's Perlin-Worley over its three Worley octaves, or the
            // detail's three frequencies over their fractal.
            let mut pixels = vec![0u8; (size * size * 4 * 4) as usize];
            for channel in 0..4usize {
                for y in 0..size {
                    for x in 0..size {
                        let value = volume.at(x, y, slice)[channel];
                        let row = channel as u32 * size + y;
                        let at = ((row * size + x) * 4) as usize;
                        pixels[at] = value;
                        pixels[at + 1] = value;
                        pixels[at + 2] = value;
                        pixels[at + 3] = 255;
                    }
                }
            }
            let path = out.join(name);
            crate::headless::write_png(&path, glam::UVec2::new(size, size * 4), &pixels)
                .expect("failed to write");
            eprintln!("wrote {}", path.display());
        };

        for (name, volume) in [("shape", &shape), ("detail", &detail)] {
            for channel in 0..4usize {
                let values: Vec<f64> = volume
                    .texels
                    .iter()
                    .map(|t| f64::from(t[channel]) / 255.0)
                    .collect();
                let n = values.len() as f64;
                let mean = values.iter().sum::<f64>() / n;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let at = |q: f64| sorted[((n - 1.0) * q) as usize];
                eprintln!(
                    "{name}[{channel}] mean={mean:.3} sd={:.3} p05={:.3} p50={:.3} p95={:.3}",
                    variance.sqrt(),
                    at(0.05),
                    at(0.50),
                    at(0.95)
                );
            }
        }

        write("cloud-shape-0.png", &shape, 0);
        write("cloud-shape-32.png", &shape, SHAPE_SIZE / 4);
        write("cloud-detail-0.png", &detail, 0);
    }

    /// The two copies of the mixer are the same mixer.
    ///
    /// Nothing would break if they differed -- they seed different fields -- but
    /// a reader finding two spellings has to work out whether the difference is
    /// deliberate, and it is not. The same text comparison
    /// `both_shaders_map_the_sky_the_same_way` makes between the sky's two
    /// copies of its own arithmetic.
    #[test]
    fn both_shaders_mix_their_noise_the_same_way() {
        let body = |source: &str| {
            let start = source
                .find("fn noise_mix(bits: u32) -> u32 {")
                .expect("no noise_mix");
            let end = source[start..].find("\n}").expect("unterminated");
            source[start..start + end].to_owned()
        };
        assert_eq!(
            body(include_str!("cloud.wgsl")),
            body(include_str!("terrain.wgsl")),
            "src/cloud.wgsl and src/terrain.wgsl spell `noise_mix` differently"
        );
    }

    /// Rust and the shader agree about how big the volumes are.
    ///
    /// The sizes decide the textures on this side and the bounds test and the
    /// texel centres on that one. A disagreement would not fail to compile: it
    /// would write a corner of the volume and leave the rest as zeroes, or
    /// address it half a texel out.
    #[test]
    fn the_shader_and_rust_agree_on_the_noise_volumes() {
        let source = include_str!("cloud.wgsl");
        for (name, value) in [("SHAPE_SIZE", SHAPE_SIZE), ("DETAIL_SIZE", DETAIL_SIZE)] {
            let declaration = format!("const {name}: u32");
            let line = source
                .lines()
                .find(|line| line.trim_start().starts_with(&declaration))
                .unwrap_or_else(|| panic!("src/cloud.wgsl declares no {name}"));
            assert!(
                line.contains(&format!("{value}u")),
                "src/cloud.wgsl says {line:?}, but src/cloud.rs says {name} is {value}"
            );
        }
        assert_eq!(
            source
                .matches(&format!(
                    "@compute @workgroup_size({GROUP}, {GROUP}, {GROUP})"
                ))
                .count(),
            2,
            "src/cloud.wgsl has kernels this module would dispatch wrongly"
        );
    }
}
