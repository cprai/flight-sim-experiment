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

/// Side of the weather map, in texels. Must match `WEATHER_SIZE` in
/// `src/cloud.wgsl`.
///
/// Two hundred and fifty-six over sixty kilometres is a texel every 234 m,
/// which is far coarser than a cloud and exactly right for what this holds:
/// not cloud, but where cloud is *allowed*. The finest thing in it is the
/// third octave, about five kilometres across, so a feature still spans twenty
/// texels. It was 512 first, and that cost 0.62 ms a frame to say the same
/// thing at four times the resolution of anything in it.
pub const WEATHER_SIZE: u32 = 256;

/// How many decks the weather describes. Must match `DECKS` in
/// `src/cloud.wgsl`.
///
/// Three, at the three heights cloud actually forms at: a low deck that
/// mountains reach into, a middle one, and cirrus. They are layers of one array
/// texture rather than three textures, so the march reads them with one binding
/// and an index.
pub const DECKS: usize = 3;

/// How long the weather takes to come back round to where it was, in seconds.
///
/// Ten minutes. Long enough that a front does not visibly cycle within a
/// flight, short enough that a sky left alone is not the same sky an hour
/// later.
const WEATHER_PERIOD: f32 = 600.0;

/// Threads per workgroup, in each axis. Must match `@workgroup_size` on the two
/// noise entry points in `src/cloud.wgsl`.
const GROUP: u32 = 4;

/// The same, for the weather, which is a flat map and takes a flat group. Must
/// match `@workgroup_size` on `cs_cloud_weather`.
const WEATHER_GROUP: u32 = 8;

/// The format both volumes are held in.
///
/// `Rgba8Unorm` is storage-writable and filterable in core WebGPU, which is the
/// pair of properties that rules almost everything else out -- see
/// `FIELD_FORMAT` in `src/air.rs` and `LUT_FORMAT` in `src/sky.rs`, which hit
/// the same wall from different sides. Eight bits a channel is ample: this is
/// noise being thresholded, not a measurement, and the quantisation is far
/// below the softest edge the march can draw.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// What kind of day it is.
///
/// A name rather than a set of numbers, because the numbers are not
/// independent: a sky with solid low cloud does not also have crisp cirrus over
/// it, and a scattered fair-weather sky has cumulus that heap rather than
/// stratus that lie flat. What each one means to each deck is [`Preset::decks`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Preset {
    /// Nothing at all, at any height. Exactly nothing; see [`Deck::look`].
    Clear,
    /// Scattered fair-weather cumulus with a little cirrus over them.
    #[default]
    Fair,
    /// Cumulus grown together into a broken sheet, with more above.
    Broken,
    /// A solid grey lid, low and flat.
    Overcast,
    /// The lid, lower and much thicker, heaped up underneath.
    Storm,
}

/// What one preset asks of one deck.
///
/// Mirrors `Deck` in `src/cloud.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DeckUniform {
    /// The field values that map to no cloud and to solid cloud, then the lean
    /// -- nothing for flat stratus, one for heaped cumulus -- and the density.
    look: [f32; 4],
    /// Where this deck's fields are drawn from, and three spare.
    seed: [u32; 4],
}

/// Mirrors the `Weather` uniform block in `src/cloud.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WeatherUniform {
    decks: [DeckUniform; DECKS],
    /// Seconds since the world started, then [`WEATHER_PERIOD`].
    clock: [f32; 4],
}

/// Where each deck's fields are drawn from.
///
/// Far apart, so that the three fields one deck takes from consecutive seeds --
/// cover, lean, density, base -- cannot collide with another deck's.
const DECK_SEEDS: [u32; DECKS] = [0x4c6f7700, 0x4d696400, 0x48696700];

impl Preset {
    /// What this preset asks of the low, middle and high decks.
    ///
    /// The four numbers are the two ends of the cover ramp, the lean and the
    /// density. A ramp whose ends both sit above one is a deck with no cloud in
    /// it at all: the field cannot reach them, so the cover is a hard zero
    /// rather than a small number, which is what lets `clear` be clear.
    ///
    /// The ramps narrow as the sky fills. Fair weather wants cumulus with sky
    /// between them, so its ramp is high and short -- only the peaks of the
    /// field become cloud. Overcast wants the opposite: a ramp so low that
    /// almost the whole field clears it, leaving breaks only where the field
    /// dips hardest.
    fn decks(self) -> [[f32; 4]; DECKS] {
        const NONE: [f32; 4] = [2.0, 3.0, 0.0, 0.0];
        match self {
            Self::Clear => [NONE, NONE, NONE],
            Self::Fair => [[0.50, 0.72, 1.0, 0.75], NONE, [0.60, 0.85, 0.0, 0.35]],
            Self::Broken => [
                [0.42, 0.66, 0.8, 0.9],
                [0.58, 0.82, 0.4, 0.5],
                [0.52, 0.80, 0.0, 0.4],
            ],
            Self::Overcast => [[0.16, 0.44, 0.2, 1.0], [0.40, 0.70, 0.2, 0.7], NONE],
            Self::Storm => [
                [0.04, 0.34, 0.9, 1.0],
                [0.24, 0.56, 0.6, 1.0],
                [0.45, 0.75, 0.0, 0.5],
            ],
        }
    }

    fn uniform(self, elapsed: std::time::Duration) -> WeatherUniform {
        let looks = self.decks();
        WeatherUniform {
            decks: std::array::from_fn(|deck| DeckUniform {
                look: looks[deck],
                seed: [DECK_SEEDS[deck], 0, 0, 0],
            }),
            clock: [elapsed.as_secs_f32(), WEATHER_PERIOD, 0.0, 0.0],
        }
    }
}

/// The two volumes, the weather over them, and the pipelines that fill them.
pub struct Cloud {
    #[allow(dead_code, reason = "read by the cloud march, which lands later")]
    shape: wgpu::Texture,
    #[allow(dead_code, reason = "read by the cloud march, which lands later")]
    detail: wgpu::Texture,
    /// One texel per patch of sky per deck: how much cloud it may hold, which
    /// way that cloud leans, how dense it is and where its base sits.
    ///
    /// Rewritten every frame rather than built once, because it moves. It is
    /// the cheapest thing in the frame by a wide margin -- see the `weather`
    /// row -- and evolving it is what stops a sky being the same sky for the
    /// length of a flight.
    #[allow(dead_code, reason = "read by the cloud march, which lands later")]
    weather: wgpu::Texture,
    shape_view: wgpu::TextureView,
    detail_view: wgpu::TextureView,
    weather_view: wgpu::TextureView,
    uniform: wgpu::Buffer,
    weather_group: wgpu::BindGroup,
    storage_group: wgpu::BindGroup,
    /// Fills the weather map. Kept, unlike the two below: it runs every frame.
    weather_pipeline: wgpu::ComputePipeline,
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
        let weather = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud weather"),
            size: wgpu::Extent3d {
                width: WEATHER_SIZE,
                height: WEATHER_SIZE,
                depth_or_array_layers: DECKS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            // Three layers of one 2D texture, not a volume: the decks sit at
            // different heights and nothing ever interpolates between them.
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = |texture: &wgpu::Texture| texture.create_view(&Default::default());
        let (shape_view, detail_view) = (view(&shape), view(&detail));
        let weather_view = weather.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud weather uniform"),
            size: std::mem::size_of::<WeatherUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let weather_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud weather layout"),
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
        let weather_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud weather group"),
            layout: &weather_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });

        let written = |binding, dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: FORMAT,
                view_dimension: dimension,
            },
            count: None,
        };
        // All three written things in one layout, and one bind group for all
        // three kernels. A kernel need only be given the bindings it uses, so
        // the two noise builds could have had a narrower group of their own --
        // but three descriptions of the same three textures is three things to
        // keep in step, and none of them is ever bound for reading here.
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud storage layout"),
            entries: &[
                written(0, wgpu::TextureViewDimension::D3),
                written(1, wgpu::TextureViewDimension::D3),
                written(2, wgpu::TextureViewDimension::D2Array),
            ],
        });
        let storage_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud storage group"),
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&weather_view),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cloud shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("cloud.wgsl").into()),
        });
        // Group 1 is the domain uniform and group 3 is what is written, which
        // is the convention everywhere else here. There is no camera and
        // nothing shared to read: the noise is a function of position and a
        // seed, and the weather of position, a seed and the clock.
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cloud pipeline layout"),
            bind_group_layouts: &[None, Some(&weather_layout), None, Some(&layout)],
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
            weather_view,
            uniform,
            weather_group,
            storage_group,
            weather_pipeline: pipeline("cloud weather", "cs_cloud_weather"),
            build: Some(Build {
                shape: pipeline("cloud shape", "cs_cloud_shape"),
                detail: pipeline("cloud detail", "cs_cloud_detail"),
            }),
            shape,
            detail,
            weather,
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
            pass.set_bind_group(1, &self.weather_group, &[]);
            pass.set_bind_group(3, &self.storage_group, &[]);
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

    /// Says what kind of day this frame is and how far into it the world is.
    ///
    /// Uploaded every frame rather than when it changes, for the reason the sky
    /// uploads its sun every frame: the clock moves whatever else does, and a
    /// branch to avoid rewriting eighty bytes costs more than the write.
    pub fn set_frame(&self, queue: &wgpu::Queue, preset: Preset, elapsed: std::time::Duration) {
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::bytes_of(&preset.uniform(elapsed)),
        );
    }

    /// Records the weather into an already-started compute pass.
    ///
    /// Every frame, because the weather moves. Its own dispatch rather than
    /// part of the march's, so the readout says what it costs -- a quarter of a
    /// million texels of three-octave noise is not nothing, and the only way to
    /// find out whether it matters is for it to have a row.
    pub fn draw_weather(&self, pass: &mut wgpu::ComputePass<'_>) {
        pass.set_pipeline(&self.weather_pipeline);
        pass.set_bind_group(1, &self.weather_group, &[]);
        pass.set_bind_group(3, &self.storage_group, &[]);
        let across = WEATHER_SIZE.div_ceil(WEATHER_GROUP);
        pass.dispatch_workgroups(across, across, DECKS as u32);
    }

    /// The two volumes and the weather over them.
    #[allow(dead_code, reason = "read by the cloud march, which lands later")]
    pub fn views(&self) -> (&wgpu::TextureView, &wgpu::TextureView, &wgpu::TextureView) {
        (&self.shape_view, &self.detail_view, &self.weather_view)
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

    /// The weather map for one preset at one moment, read back per deck.
    fn forecast(preset: Preset, elapsed: std::time::Duration) -> Vec<Volume> {
        let (device, queue) = crate::headless::device().expect("no headless device");
        let mut cloud = Cloud::new(&device);
        cloud.ensure_built(&device, &queue);
        cloud.set_frame(&queue, preset, elapsed);

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            cloud.draw_weather(&mut pass);
        }
        queue.submit(std::iter::once(encoder.finish()));

        (0..DECKS)
            .map(|deck| read_layer(&device, &queue, &cloud.weather, deck as u32))
            .collect()
    }

    /// One layer of the weather array, as a square of texels.
    fn read_layer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
        layer: u32,
    ) -> Volume {
        let size = WEATHER_SIZE;
        // 512 texels of four bytes is 2048, already a multiple of the 256-byte
        // copy alignment, so there is no padding to drop.
        let bytes_per_row = size * 4;
        assert_eq!(bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("weather readback"),
            size: u64::from(bytes_per_row * size),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
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
                depth_or_array_layers: 1,
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
        let texels = mapped
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        drop(mapped);
        readback.unmap();
        Volume { size, texels }
    }

    /// The share of the sky one deck covers, from nothing to all of it.
    fn cover(deck: &Volume) -> f64 {
        deck.texels
            .iter()
            .map(|texel| f64::from(texel[0]) / 255.0)
            .sum::<f64>()
            / deck.texels.len() as f64
    }

    /// A clear sky has no cloud in it anywhere, at any height.
    ///
    /// Exactly none, not nearly none. It is the setting every other test in the
    /// tree will reach for to say "and this changes nothing", so a stray texel
    /// of cover would turn up later as an unexplained difference in a frame
    /// that was supposed to be untouched. The ramp a clear preset asks for sits
    /// above anything the field can reach, so the subtraction is negative
    /// everywhere and `clamp` makes it a hard zero.
    #[test]
    fn a_clear_sky_has_no_cloud_at_any_height() {
        for deck in forecast(Preset::Clear, std::time::Duration::ZERO) {
            let worst = deck.texels.iter().map(|t| t[0]).max().unwrap_or(0);
            assert_eq!(worst, 0, "a clear sky covered part of a deck");
        }
    }

    /// The presets are an order, and the sky fills as they go.
    ///
    /// Each name is a claim about how much sky is covered, and the numbers that
    /// back the claim are four per deck in a table that is easy to edit and
    /// easy to get subtly wrong. This is what says the table still means what
    /// the names say -- measured over the low deck, which is the one every
    /// preset is mostly about.
    #[test]
    fn each_preset_covers_more_sky_than_the_one_before_it() {
        let order = [
            Preset::Clear,
            Preset::Fair,
            Preset::Broken,
            Preset::Overcast,
            Preset::Storm,
        ];
        let covers: Vec<f64> = order
            .iter()
            .map(|preset| cover(&forecast(*preset, std::time::Duration::ZERO)[0]))
            .collect();
        for pair in order.iter().zip(&covers).collect::<Vec<_>>().windows(2) {
            let [(before, less), (after, more)] = pair else {
                unreachable!()
            };
            assert!(
                more > less,
                "{after:?} covers {more:.3} of the sky, which is no more than {before:?} at {less:.3}"
            );
        }
        // And the ends mean what they say: nothing at all, and very nearly
        // everything. Without this the order could hold across five presets
        // that were all the same drizzle.
        assert_eq!(covers[0], 0.0);
        assert!(covers[1] < 0.35, "fair weather covered {:.3}", covers[1]);
        assert!(
            covers[4] > 0.85,
            "a storm covered only {:.3} of the sky",
            covers[4]
        );
    }

    /// The weather moves, and moves slowly.
    ///
    /// Both halves matter. A field that did not move at all would be a static
    /// sky dressed up as a changing one; a field that moved fast would flicker
    /// between frames rather than evolve over minutes. So: a frame apart is
    /// almost the same sky, and a third of the period apart is a different one.
    #[test]
    fn the_weather_evolves_over_minutes_rather_than_frames() {
        let at =
            |seconds: f32| forecast(Preset::Broken, std::time::Duration::from_secs_f32(seconds));
        let difference = |a: &Volume, b: &Volume| {
            a.texels
                .iter()
                .zip(&b.texels)
                .map(|(x, y)| (f64::from(x[0]) - f64::from(y[0])).abs() / 255.0)
                .sum::<f64>()
                / a.texels.len() as f64
        };

        let start = at(0.0);
        let frame_later = at(1.0 / 60.0);
        let much_later = at(WEATHER_PERIOD / 3.0);

        let flicker = difference(&start[0], &frame_later[0]);
        let drift = difference(&start[0], &much_later[0]);
        assert!(
            flicker < 0.002,
            "the sky changed by {flicker:.4} in a single frame"
        );
        assert!(
            drift > 0.05,
            "the sky changed by only {drift:.4} in {:.0} seconds",
            WEATHER_PERIOD / 3.0
        );
    }

    /// Writes each preset's low deck out to look at.
    ///
    /// Ignored, like `dump_noise`, and for the same reason: the tests say the
    /// presets are an order and that a clear sky is clear, and a table that
    /// satisfied both could still produce cover shaped like nothing in the sky.
    /// Run with `cargo test --release -- --ignored dump_weather --nocapture`.
    ///
    /// Cover on the left, lean in the middle, base on the right, so a preset
    /// can be judged on all three at once.
    #[test]
    #[ignore = "writes an image to look at rather than asserting anything"]
    fn dump_weather() {
        let out = std::env::temp_dir();
        for preset in [
            Preset::Clear,
            Preset::Fair,
            Preset::Broken,
            Preset::Overcast,
            Preset::Storm,
        ] {
            let decks = forecast(preset, std::time::Duration::ZERO);
            let size = WEATHER_SIZE;
            let wide = size * 3;
            let mut pixels = vec![0u8; (wide * size * DECKS as u32 * 4) as usize];
            for (index, deck) in decks.iter().enumerate() {
                for y in 0..size {
                    for x in 0..size {
                        let texel = deck.at(x, y, 0);
                        // Cover, lean, base -- density is the one channel that
                        // says nothing about the shape of the sky.
                        for (column, channel) in [texel[0], texel[1], texel[3]].iter().enumerate() {
                            let row = index as u32 * size + y;
                            let at = ((row * wide + column as u32 * size + x) * 4) as usize;
                            pixels[at] = *channel;
                            pixels[at + 1] = *channel;
                            pixels[at + 2] = *channel;
                            pixels[at + 3] = 255;
                        }
                    }
                }
            }
            let path = out.join(format!("weather-{preset:?}.png").to_lowercase());
            crate::headless::write_png(&path, glam::UVec2::new(wide, size * DECKS as u32), &pixels)
                .expect("failed to write");
            eprintln!("wrote {} ({:.3} cover)", path.display(), cover(&decks[0]));
        }
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
