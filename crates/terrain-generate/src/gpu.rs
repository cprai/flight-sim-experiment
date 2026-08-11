//! The device the erosion runs on, and the buffers it runs over.
//!
//! Every pass here takes a [`Gpu`] rather than making one, and none of them
//! knows where the landscape came from or where it is going. That is
//! deliberate: the intention is that this half eventually runs inside the
//! renderer at startup, on the device it already has, writing into the textures
//! it already holds -- at which point the only thing that has to change is who
//! calls `Gpu::new`.
//!
//! # Why buffers and not textures
//!
//! The channels are read by neighbour offsets and written back in place, never
//! sampled with a filter, so a texture would buy interpolation nothing here
//! wants and cost a storage format that has to exist for every channel. A
//! `f32` storage buffer is the plainest thing that does the job, and it is also
//! what makes the ping-pong cheap: two buffers and a swap, rather than two
//! textures and a bind group rebuild.
//!
//! One buffer per channel rather than one packed buffer for all five. At
//! 3073 x 3585 cells a channel is 42 MB, comfortably inside WebGPU's default
//! 128 MiB `max_storage_buffer_binding_size`; the 210 MB the five would come to
//! packed together is not, and asking for a raised limit would make the code
//! refuse to run on hardware that could otherwise manage it perfectly well.

use anyhow::{Context, Result};

/// Side of the square of cells one workgroup covers.
///
/// Must match `@workgroup_size` in every kernel in this crate. Eight squared is
/// 64 invocations, which is a whole wavefront on the hardware this was written
/// against and two on Nvidia's -- and a 2D group is what the stencils want,
/// since a row-shaped one reads each neighbour row eight times over.
pub const GROUP: u32 = 8;

/// The device and queue the passes run on.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// Opens a device, with no window and no surface.
    ///
    /// The power preference matters more than it looks, and for the same reason
    /// it does in the renderer's `headless::device`: with the default, wgpu
    /// takes whichever adapter enumerates first and does no sorting, so a
    /// machine with a discrete GPU, an integrated one and a software fallback
    /// picks one by coin toss. A run that lands on llvmpipe would still produce
    /// a landscape, just hours later, which is the worst way for this to fail.
    pub fn new() -> Result<Self> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::from_env()
                        .unwrap_or(wgpu::PowerPreference::HighPerformance),
                    ..Default::default()
                }),
            )
            .context("no wgpu adapter available")?;
        // Which device this ran on decides what the timings mean and whether a
        // landscape that came out different is the change or the driver, so say
        // it rather than leave it to be assumed.
        log::info!("using adapter: {:?}", adapter.get_info());

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("terrain-generate device"),
            required_features: wgpu::Features::empty(),
            required_limits: Self::limits(&adapter.limits()),
            ..Default::default()
        }))
        .context("failed to create a device")?;
        Ok(Self { device, queue })
    }

    /// A device that can hold a whole patch of the grid in workgroup memory.
    ///
    /// Stock limits everywhere except one: WebGPU allows a workgroup only
    /// 16 KB of shared storage, and the tiled relaxation wants a 48 x 48 patch
    /// double-buffered with the ground beside it, which is 27 KB. That is the
    /// whole reason the tiled pass exists -- a patch that fits in 16 KB is small
    /// enough that the halo costs more than the iterations it buys -- so the
    /// limit is asked for from the adapter rather than assumed. Nothing here
    /// targets the web, where the stock figure would be the ceiling.
    fn limits(adapter: &wgpu::Limits) -> wgpu::Limits {
        wgpu::Limits {
            max_compute_workgroup_storage_size: adapter.max_compute_workgroup_storage_size,
            ..wgpu::Limits::default()
        }
    }

    /// An empty buffer a pass can read, write, and have read back.
    pub fn storage(&self, label: &str, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// The same, with `values` already in it.
    pub fn uploaded<T: bytemuck::Pod>(&self, label: &str, values: &[T]) -> wgpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(values);
        let buffer = self.storage(label, bytes.len() as u64);
        self.queue.write_buffer(&buffer, 0, bytes);
        buffer
    }

    /// A uniform buffer holding one value.
    pub fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&buffer, 0, bytemuck::bytes_of(value));
        buffer
    }

    /// Reads `count` values back off the device, blocking until they arrive.
    ///
    /// Every call is a full pipeline stall: the queue has to drain before the
    /// copy can be mapped. That is fine for the once-per-run readbacks and
    /// ruinous inside a loop, which is why the convergence tests in this crate
    /// check every few iterations rather than every one.
    pub fn download<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
        let bytes = (count * size_of::<T>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        staging.map_async(wgpu::MapMode::Read, .., |result| {
            result.expect("readback failed")
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let mapped = staging.get_mapped_range(..).expect("buffer not mapped");
        let values: Vec<T> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        values
    }
}

/// How many cells across and down the grid a pass dispatches over is.
///
/// Carried as its own type because every kernel needs it twice -- once to size
/// the dispatch and once inside the shader to bound the reads -- and the two
/// disagreeing is a class of bug that shows up as a stripe of untouched cells
/// down one edge rather than as an error.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Extent {
    pub width: u32,
    pub rows: u32,
}

impl Extent {
    pub fn cells(self) -> usize {
        self.width as usize * self.rows as usize
    }

    /// Bytes one `f32` channel over this grid occupies.
    pub fn channel_bytes(self) -> u64 {
        self.cells() as u64 * size_of::<f32>() as u64
    }

    /// The workgroup counts covering it.
    ///
    /// The grid is 3073 x 3585 at the shipped `--sim-metres`, which is neither
    /// a multiple of the group nor even -- there is one more node than there
    /// are cells on each axis, see `Fields::new` -- so this rounds up and every
    /// kernel has to drop the invocations that land past the edge.
    pub fn workgroups(self) -> (u32, u32) {
        (self.width.div_ceil(GROUP), self.rows.div_ceil(GROUP))
    }
}

/// A device for the tests, or a skip if the machine has none.
///
/// Panicking rather than returning: a test with no GPU has nothing to say, and
/// every caller would only unwrap. The same device the tool itself opens, so a
/// passing test is evidence about what a real run does rather than about some
/// other configuration.
#[cfg(test)]
pub fn test_gpu() -> Gpu {
    Gpu::new().expect("no device for the tests")
}
