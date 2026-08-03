//! Carrying last frame's ground into this frame, so the march does less.
//!
//! Consecutive frames of a flight look almost identical. A surface point the
//! march found last frame is still the same point in world space this frame;
//! only the camera moved. Walking the whole quadtree again to rediscover it is
//! work already done.
//!
//! So before the march runs, the previous G-buffer is *scattered* into the new
//! camera's view: one point primitive per pixel of it, placed by projecting the
//! world position stored there, with the hardware depth test settling which of
//! several points landing on one pixel is the nearest. See `reproject.wgsl` for
//! why that has to be a scatter and not a gather.
//!
//! What this produces is not a frame. It is a set of buffers `cs_compact` in
//! `src/terrain.wgsl` consults: a pixel a point landed on is already answered,
//! and a pixel none landed on goes into a list for the march to work through.
//! Nothing here can make a pixel *wrong* by failing -- an empty pixel is simply
//! marched, which is what the code did for every pixel before this existed.
//!
//! A fixed share of the points is dropped every frame regardless, so that no
//! pixel can go on being answered from older and older data. See
//! [`DROP_RANKS`].

use glam::UVec2;

/// Depth format of the buffer the splat depth-tests against.
///
/// A real depth format, unlike the G-buffer's own depth
/// ([`crate::deferred::DEPTH_FORMAT`], which had to give that up to be
/// storage-writable): this one is a render attachment and exists precisely so
/// the fixed-function depth test can resolve overlapping points for free.
pub const CARRIED_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// How fast the picture is moving across one dither cell.
///
/// One texel per cell, so a sixty-fourth of the pixels -- a hundred and sixty
/// by ninety at 720p, which is why the width of the channel is not worth
/// economising on. Full floats because the half-float formats are not
/// storage-writable without a feature and `R32Float` is guaranteed, the same
/// reason [`crate::deferred::MOTION_FORMAT`] is packed into an integer.
pub const RISK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Depth cleared into the carried buffer, and the value that means "nothing
/// landed here".
///
/// Reversed depth puts the far distance at 0, so a carried point always writes
/// something greater and a pixel still reading 0 is one no point reached.
pub const NOTHING_CARRIED: f32 = 0.0;

/// How many of the sixty-four ranks of the dither are dropped each frame.
///
/// Nineteen is 29.7% -- near enough the three in ten that keeps stale ground
/// from accumulating without handing most of the frame back to the march.
///
/// Turning this up costs march time and buys freshness. It also sets how often
/// a pixel is guaranteed to be redrawn from scratch: the pattern is translated
/// one cell a frame and sweeps the whole 8x8 torus in 64, so every pixel takes
/// every rank exactly once per cycle and is therefore dropped exactly this many
/// times in any 64 consecutive frames. The gaps are not even, though -- the
/// per-row counts of ranks below 19 are 4, 2, 4, 0, 4, 1, 4, 0, so a pixel can
/// wait around sixteen frames in the worst case against a mean of 64/19.
///
/// Must match `DROP_RANKS` in `src/reproject.wgsl`, which is pinned by
/// [`tests::the_shader_drops_the_same_ranks_the_rust_side_counts`].
#[allow(
    dead_code,
    reason = "the shader is what uses these; Rust mirrors them so the tests can check what it was compiled with"
)]
pub const DROP_RANKS: u32 = 19;

/// The share of rays dropped per frame, which [`DROP_RANKS`] quantises.
#[allow(
    dead_code,
    reason = "the shader is what uses these; Rust mirrors them so the tests can check what it was compiled with"
)]
pub const DROP_FRACTION: f32 = 0.30;

/// Side of one cell of the drop pattern, in pixels.
///
/// Not one. A GPU runs pixels in waves and a wave costs as much as the longest
/// ray in it, so dropping one pixel in three *scattered* leaves a marching ray
/// in nearly every wave and saves almost nothing: measured on a horizon view,
/// a per-pixel pattern took the march from 1.81 ms to 1.74 ms, where dropping
/// whole eight-by-eight cells took it to 1.51 ms. Eight squared is also the
/// shape of the compaction's own workgroup, so a dropped cell is exactly one
/// workgroup that does not have to run.
///
/// The cost is that staleness arrives in blocks rather than as noise. Turning
/// this down makes it finer and the saving smaller.
///
/// Must match `DITHER_BLOCK` in `src/reproject.wgsl`.
#[allow(
    dead_code,
    reason = "the shader is what uses these; Rust mirrors them so the tests can check what it was compiled with"
)]
pub const DITHER_BLOCK: u32 = 8;

/// The screen-space motion, in pixels a frame, at which a cell counts as moving
/// as fast as the drop pattern knows how to respond to.
///
/// Must match `RISK_FULL` in `src/reproject.wgsl`.
#[allow(
    dead_code,
    reason = "the shader is what uses these; Rust mirrors them so the tests can check what it was compiled with"
)]
pub const RISK_FULL: f32 = 8.0;

/// How far towards dropping everything a fully risky cell is taken.
///
/// Must match `RISK_GAIN` in `src/reproject.wgsl`.
#[allow(
    dead_code,
    reason = "the shader is what uses these; Rust mirrors them so the tests can check what it was compiled with"
)]
pub const RISK_GAIN: f32 = 0.6;

/// Threads per workgroup of the compacted march.
///
/// Flat rather than square, because the pixels it works through are a list
/// rather than a rectangle: consecutive threads take consecutive entries of the
/// hole list, which are wherever on screen the reprojection failed.
///
/// Must match `@workgroup_size` on `cs_march` in `src/terrain.wgsl`.
#[allow(
    dead_code,
    reason = "the shader is what uses these; Rust mirrors them so the tests can check what it was compiled with"
)]
pub const MARCH_GROUP: u32 = 64;

/// The ordered 8x8 Bayer rank of a cell, 0 to 63.
///
/// The Rust mirror of `bayer8` in `src/reproject.wgsl`, kept so the pattern and
/// the guarantees claimed for it can be checked without a GPU.
#[allow(
    dead_code,
    reason = "the shader is what uses these; Rust mirrors them so the tests can check what it was compiled with"
)]
pub fn bayer8(x: u32, y: u32) -> u32 {
    let y = y & 7;
    let m = (x & 7) ^ y;
    ((m & 1) << 5)
        | ((y & 1) << 4)
        | ((m & 2) << 2)
        | ((y & 2) << 1)
        | ((m & 4) >> 1)
        | ((y & 4) >> 2)
}

/// Whether a pixel is handed back to the march on a given frame.
///
/// The Rust mirror of `dropped` in `src/reproject.wgsl`.
#[allow(
    dead_code,
    reason = "the shader is what uses these; Rust mirrors them so the tests can check what it was compiled with"
)]
pub fn dropped(x: u32, y: u32, frame: u32) -> bool {
    bayer8(
        x / DITHER_BLOCK + (frame & 7),
        y / DITHER_BLOCK + ((frame >> 3) & 7),
    ) < DROP_RANKS
}

/// Mirrors the `Splatting` uniform block in `reproject.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SplattingUniform {
    frame: u32,
    padding: [u32; 3],
    /// [`crate::camera::Camera::ray_basis`] of the camera that drew the
    /// history, one vector per row, `w` unused on each.
    was_ray_right: [f32; 4],
    was_ray_up: [f32; 4],
    was_ray_forward: [f32; 4],
}

/// How a frame's pixels were settled, read back from [`Carried::tally`].
///
/// One number per path through `cs_compact`, and every pixel of the viewport
/// takes exactly one of them, so they sum to the pixel count. `reprojected` is
/// the one the reprojection is judged on -- what the previous frame answered
/// for this one. Sky is kept apart from it rather than counted with it because
/// the ceiling test settles sky for nothing whether there is a history or not,
/// so folding the two together would credit the reprojection with pixels it
/// never had to carry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Coverage {
    /// Pixels no carried point reached that the ceiling test could not settle,
    /// so a ray was cast for them.
    pub marched: u32,
    /// Pixels a splat landed on, carried over from the previous frame.
    pub reprojected: u32,
    /// Pixels the ceiling test called sky without casting a ray or consulting
    /// any history.
    pub sky: u32,
}

impl Coverage {
    /// Size of the buffer this is read out of.
    pub const BYTES: u64 = 12;

    /// Decodes the three counters in the order the `Tally` struct in
    /// `src/terrain.wgsl` declares them.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let at = |index: usize| {
            u32::from_le_bytes(
                bytes[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("four bytes"),
            )
        };
        Self {
            marched: at(0),
            reprojected: at(1),
            sky: at(2),
        }
    }

    /// The pixels accounted for, which should be all of them.
    pub fn total(self) -> u32 {
        self.marched + self.reprojected + self.sky
    }
}

/// Reads [`Coverage`] back frame after frame without the frame waiting on it.
///
/// A buffer map does not complete until the work that wrote the buffer has run,
/// so asking a frame for its own coverage means blocking on the GPU -- which in
/// a windowed run is most of the frame, and would make the overlay cost far
/// more than the thing it reports on. `crate::headless::profile` can afford to
/// block because it does the read once, outside the measured run; a window
/// cannot.
///
/// So one read is in flight at a time and the answer arrives when it arrives,
/// two or three frames late. That is the same trade [`wgpu_profiler`] makes for
/// the timestamps shown beside it, and for the same reason the overlay already
/// shows the previous frame's rows.
pub struct CoverageReader {
    readback: wgpu::Buffer,
    /// Cloned into each map callback. Held here as well so the receiver can
    /// never see the channel as disconnected.
    sender: std::sync::mpsc::Sender<Result<(), wgpu::BufferAsyncError>>,
    arrivals: std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    /// A copy has been recorded and is waiting for its submit, after which it
    /// can be mapped.
    copied: bool,
    /// A map is outstanding. What stops a second copy being recorded over a
    /// buffer that is still being read out of.
    mapping: bool,
    latest: Option<Coverage>,
}

impl CoverageReader {
    pub fn new(device: &wgpu::Device) -> Self {
        let (sender, arrivals) = std::sync::mpsc::channel();
        Self {
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("coverage readback"),
                size: Coverage::BYTES,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            sender,
            arrivals,
            copied: false,
            mapping: false,
            latest: None,
        }
    }

    /// Copies `tally` out, if the last read has finished with the buffer.
    ///
    /// Rides on the frame's own encoder rather than taking one of its own: a
    /// twelve-byte copy is not worth a second submit, and the submit is a row
    /// the overlay reports.
    ///
    /// Call after the passes are recorded and before the encoder is finished;
    /// [`Self::collect`] then goes after the submit.
    pub fn record(&mut self, encoder: &mut wgpu::CommandEncoder, tally: &wgpu::Buffer) {
        if self.copied || self.mapping {
            return;
        }
        encoder.copy_buffer_to_buffer(tally, 0, &self.readback, 0, Coverage::BYTES);
        self.copied = true;
    }

    /// Takes whatever has come back, and starts the next read.
    ///
    /// Returns the most recent coverage read rather than only a fresh one, so a
    /// caller can show it every frame without holding a copy of its own. [`None`]
    /// until the first read lands, which is a few frames into a run.
    ///
    /// Nothing here polls the device, and nothing needs to: a map callback runs
    /// from inside a poll, and `Queue::submit` ends with
    /// `device.maintain(PollType::Poll)` followed by `callbacks.fire()`
    /// (`wgpu-core-30.0.0/src/device/queue.rs:1541`). So a frame loop that
    /// submits is polling whether it says so or not, which is already what
    /// delivers [`wgpu_profiler`]'s timestamps to the same overlay. A blocking
    /// poll here would defeat the entire arrangement.
    pub fn collect(&mut self) -> Option<Coverage> {
        if self.mapping {
            match self.arrivals.try_recv() {
                Ok(Ok(())) => {
                    match self.readback.get_mapped_range(..) {
                        Ok(mapped) => self.latest = Some(Coverage::from_bytes(&mapped)),
                        // Cannot happen -- the map has just succeeded -- but
                        // losing a debug readout is not worth a panic in a
                        // frame loop.
                        Err(err) => log::warn!("the coverage readback would not open: {err}"),
                    }
                    self.readback.unmap();
                    self.mapping = false;
                }
                Ok(Err(err)) => {
                    log::warn!("failed to read the coverage back: {err}");
                    self.mapping = false;
                }
                // Still in flight. The next frame asks again.
                Err(_) => {}
            }
        }

        // Only now, because a map is only legal once the copy feeding it has
        // been submitted, which is what the caller has just done.
        if self.copied && !self.mapping {
            let sender = self.sender.clone();
            self.readback
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = sender.send(result);
                });
            self.copied = false;
            self.mapping = true;
        }

        self.latest
    }
}

/// Where the reprojection puts what it carried, and the work list it leaves.
///
/// Screen-sized, so it is rebuilt with the G-buffer whenever the target
/// changes size.
pub struct Carried {
    pub material: wgpu::TextureView,
    /// Zero where the carried point is sky, a unit vector where it is ground.
    ///
    /// No position beside it: sixteen bytes a point of export bandwidth for
    /// something the depth already fixes. `carried_at` in `src/terrain.wgsl`
    /// rebuilds it.
    pub normal: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    /// Pixels no carried point reached, packed as `x | y << 16`.
    ///
    /// Written by `cs_compact` and read by `cs_march`. Sized for every pixel,
    /// because a frame with no history at all -- the first one, and the first
    /// after a resize -- leaves every pixel of ground in it.
    pub holes: wgpu::Buffer,
    /// How many pixels `cs_compact` sent down each of its three paths.
    ///
    /// Cleared per frame and counted up with atomics. The first member is how
    /// many entries of `holes` are live, which the march is sized from; the
    /// other two are the coverage measurement. See [`Coverage`], which decodes
    /// it, and the `Tally` struct in `src/terrain.wgsl`, which writes it.
    pub tally: wgpu::Buffer,
    /// `[workgroups, 1, 1]` for the march's indirect dispatch.
    pub march_args: wgpu::Buffer,
    /// One number per dither cell: the fastest screen-space motion in it, which
    /// is what the drop pattern spends its extra rays on.
    pub risk: wgpu::TextureView,
}

impl Carried {
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        let size = size.max(UVec2::ONE);
        let pixels = u64::from(size.x) * u64::from(size.y);
        let target = |label, format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size.x,
                        height: size.y,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        // Drawn into by the splat, read by `cs_compact`.
        let attachment =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        Self {
            material: target(
                "carried material",
                crate::deferred::MATERIAL_FORMAT,
                attachment,
            ),
            normal: target("carried normal", crate::deferred::NORMAL_FORMAT, attachment),
            depth: target("carried depth", CARRIED_DEPTH_FORMAT, attachment),
            holes: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("hole list"),
                size: pixels * 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            tally: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("compaction tally"),
                size: Coverage::BYTES,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            march_args: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("march dispatch"),
                size: 12,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
            risk: device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("cell risk"),
                    size: wgpu::Extent3d {
                        width: size.x.div_ceil(DITHER_BLOCK),
                        height: size.y.div_ceil(DITHER_BLOCK),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: RISK_FORMAT,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default()),
        }
    }
}

/// The layout `cs_compact` and `cs_march` reach the carried buffers and the
/// work list through.
///
/// One layout for both, even though the march never reads the carried textures:
/// a pipeline may leave entries of its layout untouched, and one description
/// that both dispatches share cannot drift between them.
pub fn work_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let buffer = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("reprojection work layout"),
        entries: &[
            texture(0, wgpu::TextureSampleType::Uint),
            texture(2, wgpu::TextureSampleType::Float { filterable: false }),
            texture(3, wgpu::TextureSampleType::Depth),
            buffer(4),
            buffer(5),
        ],
    })
}

/// The layout `cs_args` reaches the count and the dispatch size through.
///
/// Separate from [`work_layout`] for one reason: the march *dispatches* from
/// `march_args`, and wgpu will not let a buffer be bound as writable storage in
/// the same dispatch that reads it as the indirect argument. Keeping it out of
/// the layout the march binds is what keeps those two uses apart. Both sit at
/// group 3, so the shader declares them alongside the rest of the work.
pub fn args_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let buffer = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("reprojection args layout"),
        entries: &[buffer(5), buffer(6)],
    })
}

/// The layout `cs_risk` reduces the motion field through.
///
/// Its own, like [`args_layout`], and for a related reason: it reads the motion
/// target that the march *writes*, and a pipeline may not have one texture
/// bound both ways at once.
pub fn risk_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("reprojection risk layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: RISK_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ],
    })
}

/// Points `cs_risk` at the motion field and the risk it reduces it to.
pub fn bind_risk(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    gbuffer: &crate::deferred::GBuffer,
    carried: &Carried,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reprojection risk bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&gbuffer.motion),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&carried.risk),
            },
        ],
    })
}

/// Points `cs_args` at the count it reads and the dispatch size it writes.
pub fn bind_args(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    carried: &Carried,
) -> wgpu::BindGroup {
    let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reprojection args bind group"),
        layout,
        entries: &[
            entry(5, carried.tally.as_entire_binding()),
            entry(6, carried.march_args.as_entire_binding()),
        ],
    })
}

/// Points the two dispatches at this frame's carried buffers and work list.
pub fn bind_work(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    carried: &Carried,
) -> wgpu::BindGroup {
    let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reprojection work bind group"),
        layout,
        entries: &[
            entry(0, wgpu::BindingResource::TextureView(&carried.material)),
            entry(2, wgpu::BindingResource::TextureView(&carried.normal)),
            entry(3, wgpu::BindingResource::TextureView(&carried.depth)),
            entry(4, carried.holes.as_entire_binding()),
            entry(5, carried.tally.as_entire_binding()),
        ],
    })
}

/// The splat that scatters the previous G-buffer into the new camera's view.
pub struct Reprojection {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    dither: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl Reprojection {
    pub fn new(
        device: &wgpu::Device,
        camera_layout: &wgpu::BindGroupLayout,
        history: &crate::deferred::GBuffer,
        carried: &Carried,
    ) -> Self {
        let dither = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splatting uniform"),
            size: size_of::<SplattingUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Read in the *vertex* stage, which is the one thing about this layout
        // that is not like the shading pass's: the position decides where the
        // point goes, so it has to be known before rasterization.
        let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("reprojection history layout"),
            entries: &[
                texture(0, wgpu::TextureSampleType::Uint),
                texture(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture(2, wgpu::TextureSampleType::Float { filterable: false }),
                texture(3, wgpu::TextureSampleType::Float { filterable: false }),
                texture(5, wgpu::TextureSampleType::Float { filterable: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = Self::bind(device, &layout, &dither, history, carried);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("reprojection shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("reproject.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("reprojection pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&layout)],
            immediate_size: 0,
        });
        let colour = |format| {
            Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("reprojection pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_reproject"),
                compilation_options: Default::default(),
                // One point per pixel of the history, from the vertex index.
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_reproject"),
                compilation_options: Default::default(),
                targets: &[
                    colour(crate::deferred::MATERIAL_FORMAT),
                    colour(crate::deferred::NORMAL_FORMAT),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                // One pixel each, and wgpu gives no way to ask for more: WGSL
                // has no `point_size` builtin, and naga writes the constant 1
                // into every vertex entry point on Vulkan.
                topology: wgpu::PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: CARRIED_DEPTH_FORMAT,
                // The whole reason this is rasterized rather than computed: the
                // fixed-function test decides which of several points landing
                // on one pixel is the nearest, at the same reversed-Z
                // comparison and the same precision the march itself uses.
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            layout,
            pipeline,
            dither,
            bind_group,
        }
    }

    /// Points the splat at a rebuilt G-buffer, after a resize.
    pub fn rebind(
        &mut self,
        device: &wgpu::Device,
        history: &crate::deferred::GBuffer,
        carried: &Carried,
    ) {
        self.bind_group = Self::bind(device, &self.layout, &self.dither, history, carried);
    }

    /// Tells the splat which frame this is and where the history was looking.
    ///
    /// The frame number moves the dither's pattern on, so a different share of
    /// the screen is dropped each time. The ray basis is the previous camera's,
    /// which is what a carried sky pixel is rebuilt from -- ground needs no such
    /// thing, its world position being absolute.
    pub fn set_frame(&self, queue: &wgpu::Queue, frame: u32, was: [glam::Vec3; 3]) {
        queue.write_buffer(
            &self.dither,
            0,
            bytemuck::bytes_of(&SplattingUniform {
                frame,
                padding: [0; 3],
                was_ray_right: was[0].extend(0.0).to_array(),
                was_ray_up: was[1].extend(0.0).to_array(),
                was_ray_forward: was[2].extend(0.0).to_array(),
            }),
        );
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        dither: &wgpu::Buffer,
        history: &crate::deferred::GBuffer,
        carried: &Carried,
    ) -> wgpu::BindGroup {
        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reprojection history bind group"),
            layout,
            entries: &[
                entry(0, wgpu::BindingResource::TextureView(&history.material)),
                entry(1, wgpu::BindingResource::TextureView(&history.position)),
                entry(2, wgpu::BindingResource::TextureView(&history.normal)),
                entry(3, wgpu::BindingResource::TextureView(&history.depth)),
                entry(4, dither.as_entire_binding()),
                entry(5, wgpu::BindingResource::TextureView(&carried.risk)),
            ],
        })
    }

    /// Records the splat into an already-started render pass.
    ///
    /// `pixels` is the whole of the previous frame -- one point each, dropped
    /// ones included. A dropped point is thrown away in the vertex stage rather
    /// than never issued, because the drop is decided per pixel and nothing on
    /// the CPU knows which those are.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, pixels: u32) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..pixels, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The matrix the closed form has to reproduce, written out.
    const BAYER: [[u32; 8]; 8] = [
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21],
    ];

    /// Nothing else would catch a shift in the wrong direction: a bit-reversed
    /// interleave is still a permutation of nought to sixty-three, so it still
    /// drops the right *number* of pixels and still looks like a dither. It
    /// just stops being the pattern that spreads them evenly.
    #[test]
    fn the_dither_is_the_ordered_bayer_matrix() {
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(bayer8(x, y), BAYER[y as usize][x as usize], "at ({x}, {y})");
            }
        }
    }

    #[test]
    fn the_dither_drops_the_fraction_it_claims_to() {
        assert_eq!(DROP_RANKS, (DROP_FRACTION * 64.0).round() as u32);
        let dropped = (0..8)
            .flat_map(|y| (0..8).map(move |x| bayer8(x, y)))
            .filter(|rank| *rank < DROP_RANKS)
            .count();
        assert_eq!(dropped as u32, DROP_RANKS);
    }

    /// The claim [`DROP_RANKS`] makes about staleness, as an assertion.
    ///
    /// Translating the pattern one cell a frame sweeps the whole torus in 64,
    /// so this is exact rather than statistical -- every pixel is redrawn from
    /// scratch the same number of times, and the worst wait is bounded.
    #[test]
    fn every_pixel_is_marched_again_within_one_turn_of_the_dither() {
        for y in 0..8 {
            for x in 0..8 {
                let frames: Vec<u32> = (0..64).filter(|f| dropped(x, y, *f)).collect();
                assert_eq!(
                    frames.len() as u32,
                    DROP_RANKS,
                    "pixel ({x}, {y}) is dropped {} times in 64 frames",
                    frames.len()
                );
                // The longest run of frames between two drops, wrapping round
                // the cycle, which is how long the pixel can hold stale ground.
                let gap = frames
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .chain(std::iter::once(64 - frames[frames.len() - 1] + frames[0]))
                    .max()
                    .unwrap();
                assert!(gap <= 16, "pixel ({x}, {y}) waits {gap} frames");
            }
        }
    }

    /// The paired `// Must match` comments say the constants agree; this checks
    /// it. Possible here and not for most such pairs because the shader is
    /// already a `&'static str` in the binary.
    #[test]
    fn the_shader_drops_the_same_ranks_the_rust_side_counts() {
        let source = include_str!("reproject.wgsl");
        assert!(
            source.contains(&format!("const DROP_RANKS: u32 = {DROP_RANKS}u;")),
            "reproject.wgsl does not declare DROP_RANKS as {DROP_RANKS}"
        );
        assert!(
            source.contains(&format!("const DITHER_BLOCK: u32 = {DITHER_BLOCK}u;")),
            "reproject.wgsl does not declare DITHER_BLOCK as {DITHER_BLOCK}"
        );
        let march = include_str!("terrain.wgsl");
        assert!(
            march.contains(&format!("const MARCH_GROUP: u32 = {MARCH_GROUP}u;")),
            "terrain.wgsl does not declare MARCH_GROUP as {MARCH_GROUP}"
        );
        assert!(
            march.contains(&format!("@workgroup_size({MARCH_GROUP})")),
            "cs_march is not {MARCH_GROUP} threads wide"
        );
    }

    /// [`Coverage`] reads three integers by position, so a pair swapped in the
    /// shader would report one path's pixels as another's and nothing else
    /// would notice: the counts would still sum to the pixel count, and the
    /// march would still be sized from offset zero.
    #[test]
    fn the_tally_is_laid_out_the_way_the_coverage_reads_it() {
        let march = include_str!("terrain.wgsl");
        let declared = march
            .split_once("struct Tally {")
            .expect("terrain.wgsl declares no Tally")
            .1
            .split_once('}')
            .expect("the Tally declaration is not closed")
            .0;
        let members: Vec<&str> = declared
            .split(':')
            .map(|member| member.rsplit(',').next().unwrap_or_default().trim())
            .filter(|member| !member.is_empty())
            .collect();
        assert_eq!(members, ["holes", "carried", "sky"], "in {declared:?}");

        // And that the three of them are the whole of the buffer, so a member
        // added without widening it would truncate rather than go unreported.
        assert_eq!(Coverage::BYTES, members.len() as u64 * 4);
    }

    #[test]
    fn the_coverage_decodes_the_tally_in_that_order() {
        let bytes: Vec<u8> = [7u32, 11, 13]
            .iter()
            .flat_map(|n| n.to_le_bytes())
            .collect();
        let coverage = Coverage::from_bytes(&bytes);
        assert_eq!(
            coverage,
            Coverage {
                marched: 7,
                reprojected: 11,
                sky: 13,
            }
        );
        assert_eq!(coverage.total(), 31);
    }
}
