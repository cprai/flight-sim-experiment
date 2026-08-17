use anyhow::Result;
use glam::UVec2;
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::deferred::{GBuffer, Shading};
use crate::terrain::gpu::Terrain;
use crate::terrain::residency::Residency;

/// What an interrupted frame shows.
///
/// It was the sky once, kept in step by hand with a constant of the same value
/// in `src/shading.wgsl`. There is no such constant now: the sky is a gradient
/// read out of the sky-view table, different in every direction and different
/// again when the sun moves, and no single colour stands for it.
///
/// So this is only the clear, and the clear never survives -- the shading pass
/// writes every pixel. It is kept because a frame interrupted between the
/// passes showing a plausible daylit blue beats one showing whatever was in the
/// buffer.
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.30,
    g: 0.55,
    b: 0.85,
    a: 1.0,
};

/// The longest step the world's clock takes in one frame.
///
/// The same guard, and for the same reason, as `MAX_STEP` in
/// `src/controls.rs`: a stalled frame -- a resize, a shader compile, the window
/// being dragged -- would otherwise hand its whole wall-clock gap to
/// [`Scene::update`] at once, and the weather would jump. Kept here rather than
/// shared with the controller's, because the two are answering different
/// questions and would not have to move together: that one bounds how far the
/// camera is flung, this one bounds how far the sky is wound on.
const MAX_STEP: std::time::Duration = std::time::Duration::from_millis(100);

/// Mirrors the `Camera` uniform block in `terrain.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
    /// The same, for the camera that drew the previous frame.
    ///
    /// What a motion vector is measured against: a point's screen position now,
    /// less where this matrix says it was. The march knows a point's world
    /// position, so one matrix is the whole of what motion needs.
    was_view_proj: [[f32; 4]; 4],
    /// `w` is unused padding; uniform members are aligned to 16 bytes anyway.
    position: [f32; 4],
    /// [`Camera::ray_basis`], one vector per row.
    ///
    /// Carried alongside the matrix rather than derived from it because the
    /// raymarched far field needs a ray per pixel and inverting `view_proj` on
    /// the GPU to get one would cost far more than three vectors of uniform.
    ///
    /// `w` is unused on the first two. On the third it carries the near plane,
    /// which is what turns a depth back into a distance -- see `distance_at` in
    /// `src/terrain.wgsl`.
    ray_right: [f32; 4],
    ray_up: [f32; 4],
    ray_forward: [f32; 4],
    /// Where *this* frame's eye lands in the previous frame's clip space, from
    /// [`Camera::clip_of`].
    ///
    /// The fixed half of a reprojection, handed over already worked out because
    /// working it out on the GPU is what loses the precision. A pass that
    /// reprojects a point `o` metres from this eye wants
    /// `was_view_proj * (position + o, 1)`, and that is `was_clip` plus
    /// `was_view_proj * (o, 0)` exactly -- the second term never forms a
    /// world-scale coordinate, and the first was computed where the subtraction
    /// could be done first. See [`Camera::clip_of`] for what that is worth.
    was_clip: [f32; 4],
}

impl CameraUniform {
    fn new(camera: &Camera, was: &Camera) -> Self {
        let [right, up, forward] = camera.ray_basis();
        Self {
            view_proj: camera.view_projection().to_cols_array_2d(),
            was_view_proj: was.view_projection().to_cols_array_2d(),
            was_clip: was.clip_of(camera.position).to_array(),
            position: camera.position.extend(1.0).to_array(),
            ray_right: right.extend(0.0).to_array(),
            ray_up: up.extend(0.0).to_array(),
            ray_forward: forward.extend(camera.z_near).to_array(),
        }
    }
}

/// The terrain plus the camera looking at it, and the GPU state to draw them.
pub struct Scene {
    pub camera: Camera,
    /// Where the sun is. Public because it is a property of the scene the way
    /// the camera is, set by whoever set the camera up -- see `--sun` in
    /// `src/main.rs`. Nothing moves it with the clock yet.
    pub sun: crate::sky::Sun,
    /// Which way the wind blows. Public for the reason [`Scene::sun`] is:
    /// a property of the scene, set by whoever set the camera up.
    ///
    /// Read only by the bake, and only until it has run. Changing it after
    /// that has no effect until something asks for a re-bake, which nothing
    /// does; it is a flag at startup, like the sun was before it moved.
    pub wind: crate::air::Wind,
    /// What the world is lit by, on the GPU: [`Scene::sun`] as a uniform.
    sky: crate::sky::Sky,
    /// The wind solved around the mountains, once, at load.
    air: crate::air::Air,
    /// What kind of day it is. Public for the reason [`Scene::sun`] is.
    ///
    /// Unlike the wind, this is read every frame rather than once, so changing
    /// it mid-flight would work -- there is simply nothing yet that would.
    pub weather: crate::cloud::Preset,
    /// The noise a cloud is carved out of, and the weather over it.
    cloud: crate::cloud::Cloud,
    /// The coarse bound on where that cloud can be, and the half-resolution
    /// march that draws it. Screen-sized, so rebuilt by [`Scene::resize`].
    march: crate::cloud::March,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    terrain: Terrain,
    /// Screen-sized; rebuilt by [`Scene::resize`], never mid-frame.
    gbuffer: GBuffer,
    /// How the march reaches the G-buffer, and the layout it was built against.
    ///
    /// The layout outlives any one G-buffer -- it is what the march pipeline was
    /// compiled with -- so it is kept to rebind against on resize.
    storage_layout: wgpu::BindGroupLayout,
    storage_bind_group: wgpu::BindGroup,
    /// Where last frame's ground is scattered to, and the work list the march
    /// is dispatched over. Screen-sized, so rebuilt with the G-buffer.
    carried: crate::reproject::Carried,
    work_layout: wgpu::BindGroupLayout,
    work_bind_group: wgpu::BindGroup,
    args_layout: wgpu::BindGroupLayout,
    args_bind_group: wgpu::BindGroup,
    risk_layout: wgpu::BindGroupLayout,
    risk_bind_group: wgpu::BindGroup,
    reach_layout: wgpu::BindGroupLayout,
    reach_bind_group: wgpu::BindGroup,
    reproject: crate::reproject::Reprojection,
    /// Which frame this is, which is all the dither needs to move its pattern
    /// on. Wraps freely: only the low six bits are ever read.
    frame: u32,
    /// How long the world has been running, in the world's own time.
    ///
    /// Not wall-clock and not the process's age: it is the sum of the steps
    /// [`Scene::update`] was handed, so a headless flight of sixty frames has
    /// advanced exactly one nominal second whatever the machine took to draw
    /// them. That is what makes two runs of the same flight the same flight --
    /// the same reasoning that puts a nominal step in [`crate::headless`]
    /// rather than a measured one.
    ///
    /// [`Scene::settle`] advances it by nothing, for the reason it puts the
    /// frame counter back: settling is not time passing.
    elapsed: std::time::Duration,
    /// The ray basis of the camera that drew what is now the history.
    ///
    /// Only carried sky needs it -- ground reprojects from its own world
    /// position, which is absolute and needs no camera at all -- but sky has
    /// only a direction, and the direction it had is the one the previous
    /// camera gave it.
    was_basis: [glam::Vec3; 3],
    /// Where the eye stood when it drew what is now the history.
    ///
    /// Only carried sky needs it, and only to know whether the eye has moved at
    /// all: sky is carried as a fact about a direction, which a rotation leaves
    /// true and a translation does not. See `swept` in `src/reproject.wgsl`.
    was_eye: glam::Vec3,
    /// The projection that drew what is now the history, for motion vectors.
    was_camera: Camera,
    shading: Shading,
    /// What the last camera upload cost, zero unless a run asked to be timed.
    camera_span: std::time::Duration,
    /// Inert unless [`Scene::profile`] turned it on; see [`crate::profile`].
    ///
    /// Owned here rather than by the caller because the scene submits work of
    /// its own: the terrain generates and derives on encoders of their own,
    /// before the frame's, and those submissions have to be scoped by the same
    /// profiler and land in the same frame as the passes -- or they show up
    /// nowhere, which is what they did.
    profiler: wgpu_profiler::GpuProfiler,
}

impl Scene {
    /// Opens the terrain tile pyramid and frames the camera on it.
    ///
    /// `viewport` is the target's size in pixels, not merely its aspect: how
    /// many levels are worth keeping is decided by whether their texels are
    /// still smaller than the pixels they land in, and that needs the pixel
    /// count.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        viewport: UVec2,
        terrain_root: &std::path::Path,
    ) -> Result<Self> {
        let residency = Residency {
            pixel_angle: crate::terrain::residency::pixel_angle(
                viewport.y,
                f64::from(crate::camera::FOV_Y_DEGREES).to_radians(),
            ),
            ..Residency::default()
        };
        Self::with_residency(device, format, viewport, terrain_root, residency)
    }

    /// As [`Scene::new`], but over a residency configured by the caller.
    ///
    /// Only [`dump_installed_terrain`] uses this, to time one view against
    /// several shapes of square; the application takes the default.
    pub fn with_residency(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        viewport: UVec2,
        terrain_root: &std::path::Path,
        residency: Residency,
    ) -> Result<Self> {
        let (camera_buffer, camera_layout, camera_bind_group) = camera_binding(device);
        let storage_layout = crate::deferred::storage_layout(device);
        let work_layout = crate::reproject::work_layout(device);
        let args_layout = crate::reproject::args_layout(device);
        let risk_layout = crate::reproject::risk_layout(device);
        let reach_layout = crate::reproject::reach_layout(device);
        let terrain = Terrain::from_tiles(
            device,
            &camera_layout,
            &storage_layout,
            &work_layout,
            &args_layout,
            &risk_layout,
            &reach_layout,
            residency,
            viewport,
            terrain_root,
        )?;
        Ok(Self::assemble(
            device,
            format,
            viewport,
            camera_buffer,
            camera_bind_group,
            storage_layout,
            work_layout,
            args_layout,
            risk_layout,
            reach_layout,
            &camera_layout,
            terrain,
        ))
    }

    /// Frames the camera on an already-built terrain.
    ///
    /// Kept separate from [`Scene::new`] so tests can supply rasters directly
    /// instead of depending on files that are not in version control.
    #[cfg(test)]
    pub fn from_terrain(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        viewport: UVec2,
        terrain: impl FnOnce(
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
        ) -> Terrain,
    ) -> Self {
        let (camera_buffer, camera_layout, camera_bind_group) = camera_binding(device);
        let storage_layout = crate::deferred::storage_layout(device);
        let work_layout = crate::reproject::work_layout(device);
        let args_layout = crate::reproject::args_layout(device);
        let risk_layout = crate::reproject::risk_layout(device);
        let reach_layout = crate::reproject::reach_layout(device);
        let terrain = terrain(
            &camera_layout,
            &storage_layout,
            &work_layout,
            &args_layout,
            &risk_layout,
            &reach_layout,
        );
        Self::assemble(
            device,
            format,
            viewport,
            camera_buffer,
            camera_bind_group,
            storage_layout,
            work_layout,
            args_layout,
            risk_layout,
            reach_layout,
            &camera_layout,
            terrain,
        )
    }

    fn assemble(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        viewport: UVec2,
        camera_buffer: wgpu::Buffer,
        camera_bind_group: wgpu::BindGroup,
        storage_layout: wgpu::BindGroupLayout,
        work_layout: wgpu::BindGroupLayout,
        args_layout: wgpu::BindGroupLayout,
        risk_layout: wgpu::BindGroupLayout,
        reach_layout: wgpu::BindGroupLayout,
        camera_layout: &wgpu::BindGroupLayout,
        terrain: Terrain,
    ) -> Self {
        let aspect = viewport.x as f32 / viewport.y.max(1) as f32;
        let camera = Camera::overlooking(terrain.world_extent(), terrain.height_range().1, aspect);
        let gbuffer = GBuffer::new(device, viewport);
        let storage_bind_group = crate::deferred::bind_storage(device, &storage_layout, &gbuffer);
        let carried = crate::reproject::Carried::new(device, viewport);
        let work_bind_group = crate::reproject::bind_work(device, &work_layout, &carried);
        let args_bind_group = crate::reproject::bind_args(device, &args_layout, &carried);
        let risk_bind_group = crate::reproject::bind_risk(device, &risk_layout, &gbuffer, &carried);
        let reach_bind_group = crate::reproject::bind_reach(device, &reach_layout, &carried);
        let reproject =
            crate::reproject::Reprojection::new(device, camera_layout, &gbuffer, &carried);
        let sky = crate::sky::Sky::new(device, camera_layout);
        let cloud = crate::cloud::Cloud::new(device);
        let air = crate::air::Air::new(device);
        // Before the shading, which now reads what it leaves.
        let march = crate::cloud::March::new(
            device,
            camera_layout,
            sky.layout(),
            sky.sun_tables_layout(),
            &cloud,
            &gbuffer,
            &air,
        );
        let shading = Shading::new(
            device,
            format,
            &gbuffer,
            camera_layout,
            sky.layout(),
            sky.tables_layout(),
            march.views(),
        );
        Self {
            camera,
            sun: crate::sky::Sun::default(),
            wind: crate::air::Wind::default(),
            sky,
            air,
            weather: crate::cloud::Preset::default(),
            cloud,
            march,
            camera_buffer,
            camera_bind_group,
            terrain,
            gbuffer,
            storage_layout,
            storage_bind_group,
            carried,
            work_layout,
            work_bind_group,
            args_layout,
            args_bind_group,
            risk_layout,
            risk_bind_group,
            reach_layout,
            reach_bind_group,
            reproject,
            frame: 0,
            elapsed: std::time::Duration::ZERO,
            was_basis: camera.ray_basis(),
            was_eye: camera.position,
            was_camera: camera,
            shading,
            camera_span: std::time::Duration::ZERO,
            profiler: crate::profile::profiler(device, false),
        }
    }

    /// Follows the render target to a new size.
    ///
    /// The G-buffer has to match the target pixel for pixel, and the shading
    /// pass reads the G-buffer, so both are rebuilt together. The camera's
    /// aspect follows too -- without that the projection would stretch the
    /// scene to fit rather than widen the view.
    pub fn resize(&mut self, device: &wgpu::Device, viewport: UVec2) {
        self.gbuffer = GBuffer::new(device, viewport);
        self.storage_bind_group =
            crate::deferred::bind_storage(device, &self.storage_layout, &self.gbuffer);
        self.carried = crate::reproject::Carried::new(device, viewport);
        self.work_bind_group =
            crate::reproject::bind_work(device, &self.work_layout, &self.carried);
        self.args_bind_group =
            crate::reproject::bind_args(device, &self.args_layout, &self.carried);
        self.risk_bind_group =
            crate::reproject::bind_risk(device, &self.risk_layout, &self.gbuffer, &self.carried);
        self.reach_bind_group =
            crate::reproject::bind_reach(device, &self.reach_layout, &self.carried);
        self.reproject.rebind(device, &self.gbuffer, &self.carried);
        // The half-resolution cloud buffers follow the frame, and the march
        // reads the depth of the G-buffer just rebuilt above. Before the
        // shading, which reads the buffers this throws away and remakes.
        self.march
            .resize(device, &self.cloud, &self.gbuffer, &self.air);
        self.shading
            .rebind(device, &self.gbuffer, self.march.views());
        self.terrain.resize(viewport);
        self.camera.aspect = viewport.x as f32 / viewport.y.max(1) as f32;
    }

    /// Uploads the current camera and brings residency up to date with it.
    ///
    /// Call once per frame before [`Scene::draw`]. Bounded: a frame generates
    /// at most a few tiles, so crossing a tile boundary costs a known amount
    /// rather than a stall, and a level that falls behind is drawn coarser at
    /// its outer edge rather than wrongly. What that costs is the `detail` and
    /// `maxima` rows of the readout -- 0.72 ms and 0.18 ms of GPU on average
    /// at 4 km/s, against 0 on a frame that crossed nothing.
    ///
    /// `dt` is how much of the world's time this frame covers, clamped to
    /// [`MAX_STEP`]. It is a parameter rather than a clock read here because
    /// the caller is the only one who knows what a frame is worth: a window
    /// hands over the wall-clock gap since the last redraw, where a headless
    /// flight hands over a nominal step so that two runs of it are the same
    /// run. What reads the total is the weather; see [`Scene::elapsed`].
    ///
    /// [`Scene::elapsed`]: Scene#structfield.elapsed
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: std::time::Duration) {
        let clock = crate::profile::Clock::start(self.terrain.spans().is_some());
        self.elapsed += dt.min(MAX_STEP);
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::new(&self.camera, &self.was_camera)),
        );
        self.camera_span = clock.elapsed();
        // Wrapping is fine and deliberate: the dither reads the low six bits,
        // and its pattern is periodic in sixty-four frames anyway.
        self.frame = self.frame.wrapping_add(1);
        self.reproject.set_frame(
            queue,
            self.frame,
            self.was_basis,
            self.was_eye,
            self.camera.z_near,
            self.camera.position.distance(self.was_eye),
        );
        // Once, on the first update: the scattering tables are functions of the
        // medium alone, so nothing about a frame can change them. Here rather
        // than in the constructor because filling them needs a queue.
        self.sky.ensure_built(device, queue);
        // Beside the scattering tables and for the same reason: functions of
        // nothing a frame can change, so once is all they are ever built.
        self.cloud.ensure_built(device, queue);
        self.cloud.set_frame(queue, self.weather, self.elapsed);
        // Once, after the buffers are made and before the resolve reads them:
        // a texel of the history nothing has written yet is opaque black cloud.
        self.march.ensure_cleared(queue);
        // The light volumes are placed on the camera and leaned along the sun,
        // so this needs both.
        self.march.set_frame(
            queue,
            self.camera.position,
            self.sun.direction,
            &self.air,
            self.wind,
            self.elapsed,
        );
        self.march.set_rotation(queue, self.frame);
        // Uploaded every frame rather than only when it changes. Nothing moves
        // the sun yet, so this rewrites the same sixteen bytes each time --
        // which is cheaper than the branch that would avoid it, and is what
        // will already be right the day something does move it.
        self.sky.set_frame(
            queue,
            self.sun,
            self.camera.position,
            crate::sky::pixel_angle(self.camera.fov_y, self.gbuffer.size.y),
        );
        // What this frame draws becomes the next one's history, so the basis it
        // is drawn with is the basis that history will have to be read back
        // through.
        self.was_basis = self.camera.ray_basis();
        self.was_eye = self.camera.position;
        self.was_camera = self.camera;
        // Scoped by this scene's own profiler, on the terrain's own encoders.
        // They are submitted before the frame's, so the timestamps are already
        // written by the time the frame encoder resolves the query set.
        self.terrain
            .update(device, queue, self.camera.position, &self.profiler);
        // After the terrain, because the first of those updates is the one that
        // reads the chain in, and the wind is solved around the ground that
        // reading produces. Once, on its own encoder, and unprofiled -- the
        // same arrangement `Sky::ensure_built` has, for the same reason: it is
        // a load cost and not a frame's.
        self.air
            .ensure_baked(device, queue, &self.terrain, self.wind);
    }

    /// Leaves the wind unsolved, and says so, so nothing solves it later.
    ///
    /// See [`crate::air::Air::assume_baked`] for why the offscreen tests want
    /// this: a scene apiece, each paying six hundred milliseconds for a field
    /// none of them reads.
    #[cfg(test)]
    pub fn skip_the_wind(&mut self) {
        self.air.assume_baked(self.wind);
    }

    /// How the last frame's pixels were settled: carried over, sky, or marched.
    ///
    /// The whole of the coverage measurement, three `u32`s laid out for
    /// [`crate::reproject::Coverage`] to decode. Lives on the GPU and is only
    /// worth reading back outside a measured frame -- see
    /// [`crate::headless::profile`].
    pub fn tally(&self) -> &wgpu::Buffer {
        &self.carried.tally
    }

    /// The highest ground anywhere resident; see [`Terrain::ceiling`].
    ///
    /// [`Terrain::ceiling`]: crate::terrain::gpu::Terrain::ceiling
    pub fn ceiling(&self) -> f32 {
        self.terrain.ceiling()
    }

    /// Starts or stops accounting for where a frame's time goes.
    ///
    /// Off by default, and off costs nothing: see [`crate::profile`]. Both
    /// clocks are switched together, because a run that wants one wants the
    /// other -- the CPU rows only mean anything beside the GPU rows they
    /// overlap.
    pub fn profile(&mut self, device: &wgpu::Device, on: bool) {
        self.terrain.profile(on);
        self.profiler = crate::profile::profiler(device, on);
    }

    /// The profiler every scope of this scene's work is opened on.
    ///
    /// Handed out so a caller can time work of its own -- the overlay, which is
    /// drawn over the frame and is not the scene's -- against the same clock
    /// and in the same frame.
    pub fn profiler(&self) -> &wgpu_profiler::GpuProfiler {
        &self.profiler
    }

    /// The same profiler, for the frame bookkeeping only the caller can do.
    ///
    /// [`GpuProfiler::resolve_queries`] and [`GpuProfiler::end_frame`] both want
    /// `&mut`, and both belong to whoever owns the frame's encoder rather than
    /// to the scene, which does not know when the frame is finished.
    ///
    /// [`GpuProfiler::resolve_queries`]: wgpu_profiler::GpuProfiler::resolve_queries
    /// [`GpuProfiler::end_frame`]: wgpu_profiler::GpuProfiler::end_frame
    pub fn profiler_mut(&mut self) -> &mut wgpu_profiler::GpuProfiler {
        &mut self.profiler
    }

    /// Fills in the CPU side of `frame` from the update just run.
    pub fn record(&self, frame: &mut crate::profile::Frame) {
        frame.cpu.camera = self.camera_span;
        frame.cpu.terrain = self.terrain.spans().unwrap_or_default();
    }

    /// Updates until every level holds all the tiles it wants.
    ///
    /// For anything that draws one frame and stops -- a screenshot, a test --
    /// where streaming in over the next second is no use to anybody.
    pub fn settle(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        // Settling is not time passing. It runs an unpredictable number of
        // updates -- however many the tiles happen to need -- and the dither's
        // phase belongs to frames that are actually drawn, so it is put back
        // afterwards. Without this the pattern at the first drawn frame would
        // depend on how much of the pyramid was on disk.
        //
        // The world's clock needs no putting back, because it is never wound
        // on: a zero step is the same statement about time that restoring the
        // counter is about the dither, made where it can be made directly.
        let frame = self.frame;
        let still = std::time::Duration::ZERO;
        self.update(device, queue, still);
        while self.terrain.pending() {
            self.update(device, queue, still);
        }
        self.frame = frame;
        self.reproject.set_frame(
            queue,
            self.frame,
            self.was_basis,
            self.was_eye,
            self.camera.z_near,
            self.camera.position.distance(self.was_eye),
        );
        // And the cloud march's own rotation with it, which is counted off the
        // same number: settling must not decide which quarter of the buffer the
        // first drawn frame marches, any more than it decides the dither's
        // phase.
        self.march.set_rotation(queue, self.frame);
    }

    /// Records the two passes that make a frame into `view`.
    ///
    /// The geometry pass is a compute dispatch that raymarches every pixel of
    /// ground into the G-buffer; the shading pass draws the image from it.
    /// `view` must match the size the scene was built or last
    /// [`Scene::resize`]d to, because the G-buffer is looked up by pixel
    /// coordinate and the dispatch is sized to the viewport the terrain was
    /// last told about.
    ///
    /// Every pass is opened through a `passes` scope on [`Scene::profiler`], so
    /// each is timed at its boundaries and the whole block is timed around
    /// them. It costs an unprofiled run nothing: a disabled profiler writes no
    /// timestamps and the scopes fall away.
    ///
    /// `passes` is not the whole of the GPU's frame and is not named `gpu` for
    /// that reason. The terrain generates tiles and raises its pyramid on
    /// submissions of its own, from [`Scene::update`], and those are scoped
    /// separately -- the readout adds them up.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut scope = self.profiler.scope(crate::profile::PASSES, encoder);
        let gpu = &mut scope;
        let target = |view, clear| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })
        };

        // Counted up from nothing every frame by `cs_compact`.
        gpu.clear_buffer(&self.carried.tally, 0, None);

        {
            // First, because it depends on nothing this frame produces -- only
            // on the camera and the sun, both already uploaded -- and because
            // every pass after it may read what it writes.
            //
            // A scope of its own with the build named inside it, so the
            // overlay and the profile table both show what the atmosphere
            // costs as one row with its parts under it. Note what that row is
            // and is not: it is the cost of *building* the tables. The cost of
            // *reading* them is the growth in `shading`, which cannot be
            // separated out without a toggle to turn the sky off.
            let mut atmosphere = gpu.scope(crate::profile::ATMOSPHERE);
            {
                let mut pass = atmosphere.scoped_compute_pass("sky-view");
                self.sky.draw_sky_view(&mut pass);
            }
            {
                // Its own pass so it gets its own row, and because it is the
                // only one of the two that wants the camera: the volume it
                // fills is the frustum itself.
                let mut pass = atmosphere.scoped_compute_pass("aerial");
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                self.sky.draw_aerial(&mut pass);
            }
        }

        {
            // Reads the G-buffer -- still holding last frame, because nothing
            // has written this one yet -- and scatters it into buffers of its
            // own. The two sets swap roles rather than contents, so there is no
            // ping-pong to keep straight, and wgpu puts the barrier in at the
            // pass boundary.
            //
            // Cleared to zero depth, which is the reversed-Z far plane and
            // therefore "no point reached this pixel": a carried point always
            // writes something greater.
            let mut pass = gpu.scoped_render_pass(
                "reproject",
                wgpu::RenderPassDescriptor {
                    label: Some("reprojection pass"),
                    color_attachments: &[
                        target(&self.carried.material, wgpu::Color::TRANSPARENT),
                        target(&self.carried.normal, wgpu::Color::TRANSPARENT),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.carried.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(crate::reproject::NOTHING_CARRIED),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                },
            );
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            let size = self.gbuffer.size;
            self.reproject.draw(&mut pass, size.x * size.y);
        }

        {
            // Nothing clears the G-buffer, because nothing needs to: between
            // them these two dispatches write every pixel exactly once. A pixel
            // whose ray found no ground writes zeroes itself, and depth zero is
            // the reversed-Z far plane, which the shading pass reads as sky.
            let mut pass = gpu.scoped_compute_pass("compact");
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(2, &self.storage_bind_group, &[]);
            pass.set_bind_group(3, &self.work_bind_group, &[]);
            self.terrain.compact(&mut pass);
        }

        {
            // Its own pass, not another dispatch in the one above: the count is
            // only final once every workgroup of the compaction has run, and a
            // pass boundary is what guarantees that.
            let mut pass = gpu.scoped_compute_pass("args");
            pass.set_bind_group(3, &self.args_bind_group, &[]);
            self.terrain.args(&mut pass);
        }

        {
            let mut pass = gpu.scoped_compute_pass("march");
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(2, &self.storage_bind_group, &[]);
            pass.set_bind_group(3, &self.work_bind_group, &[]);
            self.terrain.march(&mut pass, &self.carried.march_args);
        }

        {
            // Last, because the motion field is only whole once both the
            // compaction and the march have written their share of it. What it
            // leaves is read by the next frame's splat.
            let mut pass = gpu.scoped_compute_pass("risk");
            pass.set_bind_group(3, &self.risk_bind_group, &[]);
            self.terrain.risk(&mut pass);
        }

        {
            // Its own pass again, and for the reason `args` has one: a cell
            // needs its neighbours' finished risk, and a pass boundary is what
            // says every workgroup above has run.
            let mut pass = gpu.scoped_compute_pass("reach");
            pass.set_bind_group(3, &self.reach_bind_group, &[]);
            self.terrain.reach(&mut pass);
        }

        {
            // After the march, because the last of these needs the depth it
            // wrote: the ground is what stops a cloud ray, and reading the
            // G-buffer is what makes that occlusion exact rather than a guess.
            // The first two depend on nothing this frame produces and could sit
            // anywhere; they are here so the readout shows what a sky costs as
            // one row with its parts under it.
            //
            // Three passes rather than one dispatch after another, because each
            // reads what the one before it wrote and a pass boundary is what
            // makes those writes visible.
            let mut cloud = gpu.scope(crate::profile::CLOUD);
            {
                let mut pass = cloud.scoped_compute_pass("weather");
                self.cloud.draw_weather(&mut pass);
            }
            {
                let mut pass = cloud.scoped_compute_pass("cloud-ceiling");
                self.march.draw_ceiling(&mut pass);
            }
            {
                let mut pass = cloud.scoped_compute_pass("cloud-light");
                self.march.draw_light(&mut pass);
            }
            {
                let mut pass = cloud.scoped_compute_pass("cloud-march");
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                self.march.draw(&mut pass, &self.sky);
            }
            {
                // The march fills one texel in four; this fills the rest from
                // the buffer the last frame left, carried through the camera's
                // own motion. Its own pass because it reads what the march just
                // wrote.
                let mut pass = cloud.scoped_compute_pass("cloud-resolve");
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                self.march.draw_resolve(&mut pass);
            }
        }

        // The clear never survives -- the shading pass writes every pixel --
        // but an interrupted frame showing sky beats one showing garbage.
        let mut pass = gpu.scoped_render_pass(
            "shading",
            wgpu::RenderPassDescriptor {
                label: Some("shading pass"),
                color_attachments: &[target(view, CLEAR_COLOR)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            },
        );
        pass.set_bind_group(0, &self.camera_bind_group, &[]);
        self.shading.draw(&mut pass, &self.sky, self.march.parity());
    }

    /// Marches out the rest of the cloud rotation over the frame just drawn,
    /// and draws the image again from it.
    ///
    /// The march fills one texel of every two-by-two block and carries the
    /// other three from the frame before, so the buffer is whole only after a
    /// full rotation. Every frame a running renderer shows has one, because it
    /// has been drawing for longer than four frames; a harness that draws once
    /// has a quarter of one, and a texel that no marched neighbour is looking
    /// at -- which is what a ridge line makes -- has nothing at all.
    ///
    /// So this gives it one, and gives it the *same* one: the G-buffer, the
    /// reprojection and the tally are left exactly as the single draw left
    /// them, because they are not what is being warmed. Only the cloud passes
    /// run again, over the depth already written, and then the shading reads
    /// the buffer they filled.
    #[cfg(test)]
    fn finish_the_cloud(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        // These rounds are the same frame over again, so the camera they are
        // carried back through is this one. Said rather than left to whatever
        // the last update wrote: a harness aims its camera and then draws once,
        // so the frame before this one was taken from somewhere else entirely,
        // and reprojecting through that would scatter the history across the
        // screen. A running renderer's history came from the frame before, and
        // this one's comes from itself.
        queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::bytes_of(&CameraUniform::new(&self.camera, &self.camera)),
        );
        for step in 1..crate::cloud::ROTATION.len() as u32 {
            self.march
                .set_rotation(queue, self.frame.wrapping_add(step));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.cloud.draw_weather(&mut pass);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.march.draw_ceiling(&mut pass);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.march.draw_light(&mut pass);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                self.march.draw(&mut pass, &self.sky);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_bind_group(0, &self.camera_bind_group, &[]);
                self.march.draw_resolve(&mut pass);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shading pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            self.shading.draw(&mut pass, &self.sky, self.march.parity());
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}

/// The camera uniform, its layout, and the bind group tying them together.
///
/// The layout is handed to the terrain pipeline as well, so both agree on what
/// group 0 holds.
fn camera_binding(device: &wgpu::Device) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera uniform"),
        contents: bytemuck::bytes_of(&CameraUniform {
            view_proj: [[0.0; 4]; 4],
            was_view_proj: [[0.0; 4]; 4],
            position: [0.0; 4],
            ray_right: [0.0; 4],
            ray_up: [0.0; 4],
            ray_forward: [0.0; 4],
            was_clip: [0.0; 4],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera bind group layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            // The march reads it in compute; the reprojection reads it in the
            // vertex stage, where it decides which pixel a carried point lands
            // on; and the shading reads it in the fragment stage, where it
            // rebuilds the world position behind a pixel to ask how much air
            // stands in front of it.
            visibility: wgpu::ShaderStages::COMPUTE
                | wgpu::ShaderStages::VERTEX
                | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera bind group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (buffer, layout, bind_group)
}

/// A headless device and queue for the offscreen tests.
///
/// The same one the screenshot mode runs on, so a test passing here is evidence
/// about what the application does rather than about some other configuration.
/// Panicking rather than returning, because a test with no GPU has nothing to
/// say and every caller would only unwrap.
#[cfg(test)]
pub fn test_device() -> (wgpu::Device, wgpu::Queue) {
    crate::headless::device().expect("no headless device")
}

/// The camera bind group layout alone, for tests that build a terrain without a
/// whole scene around it. The real one, so group 0 cannot drift out of step.
#[cfg(test)]
pub fn test_camera_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    camera_binding(device).1
}

/// The camera binding with `camera` already uploaded into it.
///
/// For tests that need a pass to see a particular camera without a whole scene
/// to hold it -- the aerial-perspective volume is the frustum, so its readback
/// has to say which frustum. The real `CameraUniform`, written the way
/// [`Scene::update`] writes it, so a test cannot be looking at a layout the
/// application never uses.
#[cfg(test)]
pub fn test_camera(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera: &Camera,
) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    let (_, layout, bind_group) = test_camera_buffer(device, queue, camera);
    (layout, bind_group)
}

/// The same, keeping the buffer, for a test that moves the camera between
/// frames.
///
/// What that needs and the above does not is the second matrix: a pass that
/// reads where a point *was* -- which the cloud resolve does -- says nothing
/// until two cameras have been written, and they have to go into the one
/// buffer the one bind group names. Rewrite it with
/// [`CameraUniform::new(camera, was)`].
#[cfg(test)]
pub fn test_camera_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    camera: &Camera,
) -> (wgpu::Buffer, wgpu::BindGroupLayout, wgpu::BindGroup) {
    let (buffer, layout, bind_group) = camera_binding(device);
    queue.write_buffer(
        &buffer,
        0,
        bytemuck::bytes_of(&CameraUniform::new(camera, camera)),
    );
    (buffer, layout, bind_group)
}

/// The camera uniform for a camera that has just come from `was`.
#[cfg(test)]
pub fn test_camera_moved(camera: &Camera, was: &Camera) -> impl bytemuck::Pod {
    CameraUniform::new(camera, was)
}

/// Where a world point lands on screen, in pixels, with (0, 0) at the top left.
///
/// Only used by tests, but it belongs next to the projection it inverts.
#[cfg(test)]
fn to_pixels(view_proj: glam::Mat4, point: glam::Vec3, width: u32, height: u32) -> (f32, f32) {
    let clip = view_proj * point.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    (
        (ndc.x + 1.0) * 0.5 * width as f32,
        (1.0 - ndc.y) * 0.5 * height as f32,
    )
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use glam::{IVec2, UVec2, Vec2, Vec3};
    use terrain_materials::Material;

    use super::*;
    use crate::terrain::geotiff::Georeferencing;
    use crate::terrain::gpu::Sources;
    use crate::terrain::pyramid::{Level, Pyramid, RasterSource};
    use crate::terrain::tiles::MaterialId;

    /// Side of the offscreen render target.
    const SIZE: u32 = 256;
    /// Side of the synthetic rasters, in texels.
    const RASTER: u32 = 128;
    const METRES_PER_TEXEL: f64 = 30.0;

    /// A residency holding the whole test raster at full resolution.
    ///
    /// A raster a test can afford to build is smaller than one real tile, so
    /// there is no coarsening to do: base level zero is the whole 128 texels
    /// square and the chain over it is eight mips.
    fn test_residency() -> Residency {
        Residency {
            resident_base: 0,
            detail_tiles: 8,
            detail_tile_texels: 8,
            // Whole windows per update, so no test has to drain a queue to see
            // a settled frame.
            detail_per_update: 4096,
            // A far coarser pixel than any real viewport, because the rule for
            // giving up a level compares its texels to one. This raster's are
            // thirty metres, which a 256-pixel frame still resolves from
            // thirteen kilometres up -- four times further than the raster is
            // wide, so no camera that can see it would ever drop a level and
            // the tests for dropping one would have nothing to say. Three and a
            // half degrees a pixel puts the handover at a kilometre instead,
            // which the altitudes below fly through.
            pixel_angle: 0.06,
            ..Default::default()
        }
    }

    fn placement() -> Georeferencing {
        Georeferencing::square(RASTER, RASTER, METRES_PER_TEXEL)
    }

    /// Must match `MATERIAL_MASK` in `src/terrain.wgsl`: the id is the low
    /// sixteen bits of the material word, the rest being the sub-pixel offset.
    const MATERIAL_MASK: u32 = 0xffff;

    const GRASS: MaterialId = MaterialId(Material::Grass.id() as u16);
    const SAND: MaterialId = MaterialId(Material::Sand.id() as u16);
    const LAKE: MaterialId = MaterialId(Material::Lake.id() as u16);
    const ROCK: MaterialId = MaterialId(Material::BareRock.id() as u16);
    /// An id inside the water block that no version of the enum has assigned.
    const UNASSIGNED: MaterialId = MaterialId(0x0109);

    /// Missing data, as the shading pass paints it: magenta, at whatever
    /// brightness the light left it. Loose on purpose, matching anything with
    /// strong red and blue and little green, which no material's flat colour
    /// and no sky is allowed to have.
    fn is_magenta([r, g, b, _]: [u8; 4]) -> bool {
        r > 100 && b > 100 && g < 80
    }

    fn flat_ground() -> Vec<MaterialId> {
        vec![GRASS; (RASTER * RASTER) as usize]
    }

    /// Builds terrain from raw texels and renders one frame of it.
    fn render(
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
        aim: impl FnOnce(&mut Camera),
    ) -> Frame {
        render_after(heights, materials, aim, &[])
    }

    /// As [`render`], but stepping the camera through `path` first so residency
    /// has to swap tiles in and out before the frame that is captured.
    fn render_after(
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
        aim: impl FnOnce(&mut Camera),
        path: &[Vec3],
    ) -> Frame {
        render_config(test_residency(), heights, materials, aim, path)
    }

    /// The same shape holding the raster one level coarser.
    ///
    /// The only knob that changes how much ground truth the march has: the
    /// finest level held is the finest a ray can ever descend to.
    fn coarse_residency() -> Residency {
        Residency {
            resident_base: 1,
            ..test_residency()
        }
    }

    /// A scene over synthetic rasters, with the normals derived from the
    /// heights the way `terrain-process` derives them.
    fn test_scene(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        residency: Residency,
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
    ) -> Scene {
        test_scene_over(
            device,
            format,
            residency,
            heights,
            materials,
            placement(),
            None,
        )
    }

    /// The same, over a georeferencing of the caller's choosing.
    ///
    /// Only the aerial-perspective test wants one: a hundred kilometres of air
    /// cannot be measured on a raster four across.
    ///
    /// Every scene built here has its wind marked solved without solving it --
    /// see [`Scene::skip_the_wind`]. Nothing any of these tests asserts can see
    /// the field, and solving one apiece takes the suite from twelve seconds to
    /// fifty-eight. The one test that does want the wiring exercised,
    /// `a_scene_solves_the_wind_over_its_own_terrain`, builds its scene without
    /// going through here.
    #[allow(clippy::too_many_arguments, reason = "one test wants every knob")]
    fn test_scene_over(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        residency: Residency,
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
        placement: Georeferencing,
        wind: Option<crate::air::Wind>,
    ) -> Scene {
        let mut scene = Scene::from_terrain(
            device,
            format,
            UVec2::splat(SIZE),
            |camera_layout, storage_layout, work_layout, args_layout, risk_layout, reach_layout| {
                Terrain::new(
                    device,
                    camera_layout,
                    storage_layout,
                    work_layout,
                    args_layout,
                    risk_layout,
                    reach_layout,
                    residency,
                    UVec2::splat(SIZE),
                    placement,
                    Sources {
                        heights: Box::new(Pyramid::build(Level::new(
                            RASTER,
                            RASTER,
                            heights.clone(),
                        ))),
                        materials: Box::new(Pyramid::build(Level::new(RASTER, RASTER, materials))),
                    },
                )
            },
        );
        // Solved only for the one test that is about the wind; see
        // [`Scene::skip_the_wind`] for what a solve apiece costs the suite.
        match wind {
            Some(blowing) => scene.wind = blowing,
            None => scene.skip_the_wind(),
        }
        // Under a clear sky unless the caller says otherwise, for the reason
        // given on [`render_sunlit`]: none of these tests is about cloud, and
        // cloud in front of what one of them is measuring is noise in the
        // measurement. `render_over` sets this again from its own argument.
        scene.weather = crate::cloud::Preset::Clear;
        scene
    }

    /// The same, but solving the wind the way the application does.
    fn test_scene_with_wind(
        device: &wgpu::Device,
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
    ) -> Scene {
        Scene::from_terrain(
            device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            UVec2::splat(SIZE),
            |camera_layout, storage_layout, work_layout, args_layout, risk_layout, reach_layout| {
                Terrain::new(
                    device,
                    camera_layout,
                    storage_layout,
                    work_layout,
                    args_layout,
                    risk_layout,
                    reach_layout,
                    test_residency(),
                    UVec2::splat(SIZE),
                    placement(),
                    Sources {
                        heights: Box::new(Pyramid::build(Level::new(RASTER, RASTER, heights))),
                        materials: Box::new(Pyramid::build(Level::new(RASTER, RASTER, materials))),
                    },
                )
            },
        )
    }

    /// As [`render_after`], but over a residency configured by the caller.
    fn render_config(
        residency: Residency,
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
        aim: impl FnOnce(&mut Camera),
        path: &[Vec3],
    ) -> Frame {
        render_sunlit(
            residency,
            heights,
            materials,
            aim,
            path,
            crate::sky::Sun::default(),
        )
    }

    /// The same, under a sun of the caller's choosing.
    ///
    /// Separate rather than a sixth argument on every call site, because only
    /// the tests that are *about* the sun have any business naming one. The
    /// rest want the default, which is the sun every frame in this file has
    /// always been lit by.
    ///
    /// Under a clear sky, and that is not the application's default. Cloud in
    /// front of the thing a test is measuring is noise in the measurement, and
    /// the day the composite landed it stopped being hypothetical: a
    /// fair-weather cumulus over the setting sun took the disc from white to
    /// [143, 69, 32], and the weather moving between two frames of a still
    /// camera changed ninety-five pixels the reprojection was being held to
    /// account for. Both are the renderer working. Neither is what those tests
    /// are about.
    fn render_sunlit(
        residency: Residency,
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
        aim: impl FnOnce(&mut Camera),
        path: &[Vec3],
        sun: crate::sky::Sun,
    ) -> Frame {
        render_over(
            residency,
            heights,
            materials,
            aim,
            path,
            sun,
            placement(),
            crate::cloud::Preset::Clear,
            None,
        )
    }

    /// As [`render_sunlit`], but under weather of the caller's choosing.
    ///
    /// Separate for the reason the sun is separate: only a test that is *about*
    /// the cloud has any business naming a preset, and the rest want the clear
    /// sky every frame in this file has always been drawn under.
    fn render_under(
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
        aim: impl FnOnce(&mut Camera),
        weather: crate::cloud::Preset,
    ) -> Frame {
        render_over(
            test_residency(),
            heights,
            materials,
            aim,
            &[],
            crate::sky::Sun::default(),
            placement(),
            weather,
            None,
        )
    }

    /// The same, over a georeferencing of the caller's choosing. See
    /// [`test_scene_over`].
    #[allow(clippy::too_many_arguments, reason = "one test wants every knob")]
    fn render_over(
        residency: Residency,
        heights: Vec<f32>,
        materials: Vec<MaterialId>,
        aim: impl FnOnce(&mut Camera),
        path: &[Vec3],
        sun: crate::sky::Sun,
        placement: Georeferencing,
        weather: crate::cloud::Preset,
        wind: Option<crate::air::Wind>,
    ) -> Frame {
        let (device, queue) = test_device();

        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut scene = test_scene_over(
            &device, format, residency, heights, materials, placement, wind,
        );
        scene.sun = sun;
        scene.weather = weather;
        aim(&mut scene.camera);

        // A wind is solved around the ground, and there is no ground until the
        // resident chain has been read in -- which is what settling does. Only
        // when one was asked for: settling is a second of updates, and the
        // frames that do not care about the wind are the ones that walk a path
        // below and would have it walked from somewhere else.
        if wind.is_some() {
            scene.settle(&device, &queue);
            assert!(
                scene.air.baked_for().is_some(),
                "settling did not reach the bake, so this frame has no wind in it"
            );
        }

        // Walk the requested path first, so the windows arrive at the captured
        // frame through a series of incremental updates.
        let destination = scene.camera.position;
        for step in path {
            scene.camera.position = *step;
            scene.update(&device, &queue, crate::headless::STEP);
        }
        scene.camera.position = destination;
        scene.update(&device, &queue, crate::headless::STEP);

        // `SIZE * 4` is already a multiple of the 256-byte copy alignment, and
        // the depth channel is four bytes wide too, so one figure serves both.
        let bytes_per_row = SIZE * 4;
        let staging = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: u64::from(bytes_per_row * SIZE),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let readback = staging("readback");
        let depth_readback = staging("depth readback");
        let material_readback = staging("material readback");

        let profiler = crate::profile::profiler(&device, false);
        {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut gpu = profiler.scope("gpu", &mut encoder);
                scene.draw(&mut gpu, &view);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }
        // The cloud buffer is a quarter marched after one frame; see
        // [`Scene::finish_the_cloud`]. Every frame the renderer shows has a
        // whole one, so this frame gets one too.
        scene.finish_the_cloud(&device, &queue, &view);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let mut copy = |source: wgpu::TexelCopyTextureInfo, into: &wgpu::Buffer| {
            encoder.copy_texture_to_buffer(
                source,
                wgpu::TexelCopyBufferInfo {
                    buffer: into,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bytes_per_row),
                        rows_per_image: Some(SIZE),
                    },
                },
                wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
            );
        };
        copy(texture.as_image_copy(), &readback);
        copy(scene.gbuffer.targets.depth.as_image_copy(), &depth_readback);
        copy(
            scene.gbuffer.targets.material.as_image_copy(),
            &material_readback,
        );

        // The cloud buffer is half the frame and four half floats a texel, so
        // it takes a copy of its own rather than riding on the one above. At
        // this viewport its rows are 1024 bytes, already a multiple of the copy
        // alignment, and the assertion says so rather than the reader having to
        // work it out.
        let (cloud_colour, _, _, _) = scene.march.buffers_for_test();
        let cloud_size = scene.march.size();
        let cloud_row = cloud_size.x * 8;
        assert_eq!(cloud_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        let cloud_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cloud readback"),
            size: u64::from(cloud_row * cloud_size.y),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            cloud_colour.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &cloud_readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(cloud_row),
                    rows_per_image: Some(cloud_size.y),
                },
            },
            wgpu::Extent3d {
                width: cloud_size.x,
                height: cloud_size.y,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        for buffer in [
            &readback,
            &depth_readback,
            &material_readback,
            &cloud_readback,
        ] {
            buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        }
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");

        let pixels = readback
            .get_mapped_range(..)
            .expect("buffer not mapped")
            .to_vec();
        readback.unmap();
        let depths = bytemuck::cast_slice::<u8, f32>(
            &depth_readback
                .get_mapped_range(..)
                .expect("buffer not mapped"),
        )
        .to_vec();
        depth_readback.unmap();
        let materials = bytemuck::cast_slice::<u8, u32>(
            &material_readback
                .get_mapped_range(..)
                .expect("buffer not mapped"),
        )
        .to_vec();
        material_readback.unmap();
        // Only the transmittance: what the cloud scattered is a matter for the
        // composite, but whether a ray met cloud at all is what the depth clip
        // decides, and that is one number.
        let clouds = cloud_readback
            .get_mapped_range(..)
            .expect("buffer not mapped")
            .chunks_exact(8)
            .map(|texel| half::f16::from_le_bytes([texel[6], texel[7]]).to_f32())
            .collect();
        cloud_readback.unmap();

        Frame {
            pixels,
            depths,
            materials,
            clouds,
            cloud_size,
            base_level: scene.terrain.base_level(),
        }
    }

    /// One rendered frame: the picture, and the G-buffer behind it.
    ///
    /// The picture alone used to be enough to say what a pixel *was* -- sky was
    /// one flat colour and a material was another -- so the helpers here asked
    /// the bytes. That was always a little false and it is about to be plainly
    /// so: [`untouched`], the predicate this replaces, had to be stricter than
    /// its neighbour because "bluer than it is red or green" finds every lake
    /// in the frame as well as the sky.
    ///
    /// So the buffers the march wrote answer what was drawn, and the picture is
    /// asked only how it was lit. The march covers every pixel exactly once and
    /// writes zero depth where its ray found no ground -- which the reversed
    /// infinite projection cannot produce for any finite hit -- so [`Frame::sky`]
    /// is the same exact test `fs_shade` itself makes, rather than a guess at
    /// what it produced. `targets` on the G-buffer exists for this; see the
    /// note on [`crate::deferred::GBuffer::targets`].
    struct Frame {
        pixels: Vec<u8>,
        /// Reversed-Z depth per pixel, straight from the G-buffer.
        depths: Vec<f32>,
        /// The material word per pixel: the id in the low sixteen bits and
        /// where inside the pixel the ground sits in the rest. See
        /// `MATERIAL_MASK` in `src/terrain.wgsl`.
        materials: Vec<u32>,
        /// How much of the background survives the cloud, per half-resolution
        /// texel. One is a ray that met no cloud at all.
        clouds: Vec<f32>,
        cloud_size: UVec2,
        /// How much detail the camera's height above the ground bought:
        /// everything below this level was dropped. A test that means to look
        /// at more than one level has to say so, because a camera high enough
        /// leaves only the coarsest and the test would pass on an empty promise.
        base_level: u32,
    }

    impl Frame {
        fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
            pixel(&self.pixels, x, y)
        }

        /// Whether the march found no ground at this pixel.
        fn sky(&self, x: u32, y: u32) -> bool {
            self.depths[(y * SIZE + x) as usize] == 0.0
        }

        /// Which material the march found here, or nothing where it found no
        /// ground.
        ///
        /// The id the march wrote, not the colour the shading turned it into.
        /// Two materials whose flat colours sit within a few counts of each
        /// other -- and the palette has plenty, being hues by category -- are
        /// the same pixel to a colour test and different ids to this one.
        fn material(&self, x: u32, y: u32) -> Option<MaterialId> {
            if self.sky(x, y) {
                return None;
            }
            let packed = self.materials[(y * SIZE + x) as usize];
            Some(MaterialId((packed & MATERIAL_MASK) as u16))
        }

        /// How many pixels of the frame are sky.
        fn count_sky(&self) -> usize {
            (0..SIZE)
                .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
                .filter(|&(x, y)| self.sky(x, y))
                .count()
        }

        /// Sky pixels that have ground both above and below them.
        ///
        /// Sky above a ridge is honest; sky enclosed by ground is a ray that
        /// should have found something and did not.
        /// How much cloud the ray through this pixel met, from none to all.
        ///
        /// One number per two-by-two block of pixels, because that is what the
        /// march produces; a pixel is told what its block found.
        fn cloud(&self, x: u32, y: u32) -> f32 {
            let at = (y / 2) * self.cloud_size.x + x / 2;
            1.0 - self.clouds[at as usize]
        }

        fn holes(&self) -> Vec<(u32, u32)> {
            (0..SIZE)
                .flat_map(|x| {
                    let drawn: Vec<bool> = (0..SIZE).map(|y| !self.sky(x, y)).collect();
                    (0..SIZE)
                        .filter(|&y| {
                            !drawn[y as usize]
                                && drawn[..y as usize].iter().any(|hit| *hit)
                                && drawn[y as usize..].iter().any(|hit| *hit)
                        })
                        .map(move |y| (x, y))
                        .collect::<Vec<_>>()
                })
                .collect()
        }
    }

    fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i..i + 4].try_into().unwrap()
    }

    /// Looks straight down from high enough to see most of the raster.
    fn straight_down(camera: &mut Camera) {
        camera.position = Vec3::new(0.0, 3000.0, 0.0);
        camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
    }

    /// World position of the centre of a raster texel, on the ground.
    fn world_of(col: f64, row: f64) -> Vec3 {
        let (x, z) = placement().world_of_texel(0, col, row);
        Vec3::new(x as f32, 0.0, z as f32)
    }

    /// Mean absolute difference between two frames, per colour byte.
    fn mean_difference(a: &[u8], b: &[u8]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(a, b)| f64::from(a.abs_diff(*b)))
            .sum::<f64>()
            / a.len() as f64
    }

    /// An oblique view from low enough that the mesh is not blending vertically.
    ///
    /// `detail_base` starts blending the finest level into the one outside it as
    /// soon as a pixel covers more than one of its texels -- 500 m above the
    /// ground for this raster and this test's deliberately coarse pixel. That
    /// blend is a coin thrown per texel now rather than a step, which makes it
    /// no easier to compare against: a frame drawn half at one level and half at
    /// the next matches nothing exactly. Six hundred metres over ground standing
    /// at about 180 leaves the weight at zero, so a comparison from here is
    /// measuring the traversal rather than measuring the dissolve.
    fn low_and_looking_out(camera: &mut Camera) {
        camera.position = Vec3::new(70.0, 600.0, -110.0);
        camera.orientation = Camera::from_yaw_pitch_roll(0.0, -20f32.to_radians(), 0.0);
    }

    /// A ray that runs out of steps must not leave a hole in a ridge.
    ///
    /// The expensive ray in any maximum-mipmap traversal is the one running
    /// along a slope just above the surface: too close to skip a cell, too far
    /// to hit one. A grazing view is made of them, and a whole column of pixels
    /// can be doing it at once, so when the budget was too small for the window
    /// the failure was not a scattering of pinholes but vertical bands of sky
    /// through solid ground.
    ///
    /// Two things stop it and this checks the pair: a budget that scales with
    /// the traversal it bounds, and a march that reports where it had got to
    /// rather than reporting sky when the budget does run out.
    #[test]
    fn a_grazing_ray_never_leaves_a_hole_in_the_ground() {
        let (heights, materials) = rugged();
        // Aimed *into* the raster, from its north-west corner towards the far
        // one. A grazing ray that leaves the data is cut by the bounds test
        // however much budget it had -- ground outside the survey is invented
        // and does not get drawn -- so a view pointed outwards measures that
        // cut rather than what happens when a ray runs out, which is the thing
        // here.
        let grazing = |camera: &mut Camera| {
            camera.position = Vec3::new(-1500.0, 400.0, -1500.0);
            camera.orientation =
                Camera::from_yaw_pitch_roll(135f32.to_radians(), -1.5f32.to_radians(), 0.0);
        };
        let frame = |texels: u32| {
            render_config(
                Residency {
                    march_texels: texels,
                    ..test_residency()
                },
                heights.clone(),
                materials.clone(),
                grazing,
                &[],
            )
        };

        // A budget far below what the traversal needs, so that rays really do
        // run out and what is being looked at is what happens when they do.
        // This raster is too small to exhaust the shipped budget from any
        // camera, which is why the starving is deliberate rather than hoped
        // for: without it the test would pass on an empty promise. A twentieth
        // of the budget is where rays start running out and the fallback still
        // covers every one of them.
        let starved = frame(24);
        let holes = starved.holes();

        assert!(
            holes.is_empty(),
            "{} pixels of ground came out as sky, first at {:?}",
            holes.len(),
            holes.first()
        );

        // ... and where it had got to is close enough to where it was going
        // that the picture barely notices.
        let whole = frame(Residency::default().march_texels);
        let difference = mean_difference(&starved.pixels, &whole.pixels);
        assert!(
            difference < 3.0,
            "giving up early moved the frame by {difference:.2} of 255"
        );
    }

    /// The base level is the only thing that decides how much detail there is.
    ///
    /// The whole arrangement rests on this and nothing else measures it. It
    /// used to be the *width* of the resident window, because a level held only
    /// a square around the camera and detail fell away with distance. Nothing
    /// falls away now -- every level covers the whole raster -- so what is left
    /// to trade is the finest level held at all, which is what this change
    /// exists to trade.
    ///
    /// The ground is flat and painted in a one-texel check of two materials, so
    /// a ray's hit position is identical either way and the only thing that can
    /// differ is which level's ids it reads there. Ids do not blend the way
    /// colours did: the mode fold makes every coarse level of a two-way check
    /// *uniform* -- each two-by-two holds two of each and the tie always goes
    /// the same way -- so the check is visible exactly where level zero is held
    /// and vanishes when it is not.
    ///
    /// Measured as the number of horizontally adjacent pixel pairs showing
    /// the two different materials, which only level zero can produce.
    #[test]
    fn a_finer_base_reads_more_detail_at_the_same_distance() {
        let check: Vec<MaterialId> = (0..RASTER * RASTER)
            .map(|index| {
                let (x, y) = (index % RASTER, index / RASTER);
                if (x + y) % 2 == 0 { SAND } else { GRASS }
            })
            .collect();

        let transitions = |residency: Residency| {
            let frame = render_config(
                residency,
                vec![0.0; (RASTER * RASTER) as usize],
                check.clone(),
                low_and_looking_out,
                &[],
            );
            let mut changes = 0u64;
            let mut ground = 0u64;
            for y in 0..SIZE {
                for x in 0..SIZE - 1 {
                    let (here, next) = (frame.material(x, y), frame.material(x + 1, y));
                    if here.is_none() || next.is_none() {
                        continue;
                    }
                    ground += 1;
                    let flipped = (here == Some(SAND) && next == Some(GRASS))
                        || (here == Some(GRASS) && next == Some(SAND));
                    if flipped {
                        changes += 1;
                    }
                }
            }
            assert!(ground > 10_000, "only {ground} pixels of ground to measure");
            changes
        };

        let coarse = transitions(coarse_residency());
        let fine = transitions(test_residency());
        assert!(
            fine > coarse * 2,
            "holding the finer base showed {fine} material transitions against {coarse}, \
             which is not the detail it was supposed to buy"
        );
    }

    /// Looking straight down at rough ground, a ray must find it.
    ///
    /// The failure this is here for is a max pyramid whose cells bound only the
    /// samples at their corners rather than the ground between them: rays then
    /// slip through ridges and the frame fills with pinholes of sky.
    #[test]
    fn the_far_field_does_not_let_pinholes_of_sky_through() {
        let (heights, _) = rugged();
        let rastered = render_config(
            test_residency(),
            heights.clone(),
            flat_ground(),
            straight_down,
            &[],
        );
        let marched = render_config(test_residency(), heights, flat_ground(), straight_down, &[]);

        // Not zero either way: the frame's corners reach past the raster, and
        // that ground is cut by both halves alike. What matters is that marching
        // does not add to it.
        let (sky, marched_sky) = (rastered.count_sky(), marched.count_sky());
        assert!(
            marched_sky < sky + 200,
            "marching showed {marched_sky} sky pixels where the mesh showed {sky}"
        );
    }

    #[test]
    fn the_opening_view_looks_out_over_terrain_under_sky() {
        let frame = render(vec![0.0; (RASTER * RASTER) as usize], flat_ground(), |_| {});

        let sky = frame.pixel(SIZE / 2, 4);
        assert_eq!(sky[3], 255, "sky should be opaque");
        assert!(frame.sky(SIZE / 2, 4), "top of frame should be sky");

        assert!(
            !frame.sky(SIZE / 2, SIZE - 4),
            "bottom of frame should be ground"
        );
        assert_eq!(
            frame.material(SIZE / 2, SIZE - 4),
            Some(GRASS),
            "bottom of frame should be the material it is painted"
        );
    }

    /// Every material reaches the frame as the colour the palette gives it.
    ///
    /// This used to be checked sideways, by the helper the tests above used to
    /// find *where* a material was drawn: it compared a pixel against the
    /// palette entry it expected, so it happened to prove the id-to-colour
    /// mapping as a side effect of proving the registration. Those tests read
    /// the id out of the G-buffer now, which is exact where a colour was
    /// approximate -- and leaves nothing checking the mapping at all.
    ///
    /// So check it properly, which the old arrangement never did. It only ever
    /// touched grass, sand and lake, and it accepted anything within eight
    /// counts per channel -- a tolerance several pairs in the table fall
    /// inside, the palette being hues by category. This paints every material
    /// in the book at once and holds each pixel of ground to the entry for the
    /// id the march actually wrote there, whichever level supplied it.
    ///
    /// Eight-texel blocks rather than a fine check because the levels fold ids
    /// by mode: a block has to be wider than the fold to survive to a coarse
    /// level intact. The frame is flat and seen from straight above, so every
    /// normal is straight up and the light is the same everywhere -- which is
    /// what lets one expected colour per material stand for the whole block.
    /// High enough that the whole raster is in shot, so every block is on
    /// screen rather than only the middle ones. The raster is 3840 m across and
    /// a 60-degree frame spans 4157 m from here, which fits it with a margin of
    /// sky at the corners. From 2200 m -- where the frame spans 2540 m -- only
    /// 60 of the 80 materials made it in.
    fn over_the_checkerboard(camera: &mut Camera) {
        camera.position = Vec3::new(0.0, 3600.0, 0.0);
        camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
    }

    #[test]
    fn every_material_reaches_the_frame_as_its_own_colour() {
        const BLOCK: u32 = 8;
        let across = RASTER / BLOCK;
        let materials: Vec<MaterialId> = (0..RASTER * RASTER)
            .map(|index| {
                let (col, row) = (index % RASTER, index / RASTER);
                let block = (row / BLOCK) * across + col / BLOCK;
                MaterialId(Material::ALL[block as usize % Material::ALL.len()].id() as u16)
            })
            .collect();

        let frame = render(
            vec![0.0; (RASTER * RASTER) as usize],
            materials,
            over_the_checkerboard,
        );

        // What the palette holds for an id, linearised: an albedo, which is
        // the fraction of each wavelength the ground sends back.
        let albedo = |id: u32| {
            let bytes = Material::try_from_u32(id)
                .map_or(crate::palette::MAGENTA, crate::palette::flat_colour);
            Vec3::from(bytes.map(terrain_tiles::srgb_to_linear))
        };
        let shown = |pixel: [u8; 4]| {
            crate::sky::untonemap(Vec3::new(
                terrain_tiles::srgb_to_linear(pixel[0]),
                terrain_tiles::srgb_to_linear(pixel[1]),
                terrain_tiles::srgb_to_linear(pixel[2]),
            ))
        };

        // What the air does between the ground and the eye, solved for rather
        // than recomputed here.
        //
        // Every pixel of this frame is `albedo * gain + haze`, where the gain
        // carries the light on the ground and what survived the air, and the
        // haze is what the air put in front. Both vary across the frame -- the
        // path to a corner is longer than to the middle and points a different
        // way at the sun -- so one reference pixel can no longer stand for the
        // whole picture the way it could before there was any haze.
        //
        // Two unknowns per channel, so two knowns: the same camera over ground
        // painted entirely one material and then entirely another. That gives
        // an exact solve at every pixel, and leaves this test measuring what it
        // is named for -- that an id becomes its own colour -- rather than
        // measuring a second copy of the scattering.
        let uniform = |material: Material| {
            render(
                vec![0.0; (RASTER * RASTER) as usize],
                vec![MaterialId(material.id() as u16); (RASTER * RASTER) as usize],
                over_the_checkerboard,
            )
        };
        // The palette's two extremes, and chosen for that: the solve divides by
        // the gap between them, so a narrow gap multiplies up the eighth bit of
        // the reference pixels. Grass against Ocean was the first attempt and
        // its blue channels differ by only 0.034 in linear terms, which turned
        // a rounding in the reference into four counts of error in the answer.
        // Glacier against Canopy differ by more than 0.8 in every channel.
        let (first, second) = (uniform(Material::Glacier), uniform(Material::Canopy));
        let (first_albedo, second_albedo) = (
            albedo(Material::Glacier.id()),
            albedo(Material::Canopy.id()),
        );
        let spread = (first_albedo - second_albedo).abs();
        assert!(
            spread.min_element() > 0.5,
            "the two reference materials are {first_albedo} and {second_albedo}, \
             too close in some channel to solve the air from"
        );

        let mut seen = std::collections::BTreeSet::new();
        for y in 0..SIZE {
            for x in 0..SIZE {
                let Some(id) = frame.material(x, y) else {
                    continue;
                };
                // The references have to be showing ground at this pixel too,
                // or there is nothing to solve from.
                if first.material(x, y).is_none() || second.material(x, y).is_none() {
                    continue;
                }
                seen.insert(id.0);
                let gain = (shown(first.pixel(x, y)) - shown(second.pixel(x, y)))
                    / (first_albedo - second_albedo);
                let haze = shown(first.pixel(x, y)) - first_albedo * gain;
                let want = crate::sky::tonemap(albedo(u32::from(id.0)) * gain + haze)
                    .to_array()
                    .map(terrain_tiles::linear_to_srgb);
                let got = frame.pixel(x, y);
                assert!(
                    got[..3]
                        .iter()
                        .zip(want)
                        .all(|(&got, want)| got.abs_diff(want) <= 3),
                    "material {:#06x} at ({x}, {y}) drew as {got:?}, not {want:?}",
                    id.0
                );
            }
        }
        // Or the frame showed one block and the sweep proved nothing.
        assert!(
            seen.len() >= Material::ALL.len(),
            "only {} of {} materials reached the frame",
            seen.len(),
            Material::ALL.len()
        );
    }

    /// Run against a window that just fits the grid and one with room to spare.
    ///
    /// Registration is what a margin could break: the vertex stage offsets grid
    /// coordinates into window coordinates before reading either texture, so a
    /// margin applied to the heights and not to the materials -- or to either
    /// and not to the world position -- would slide the ground cover off the
    /// ground it belongs to. The wide window puts thirty-two texels between
    /// the grid and the window's edge, so any such slip is far larger than a
    /// pixel. This is also what pins the shader's nearest-texel lookup to the
    /// same texel-centre convention the heights read by.
    #[test]
    fn materials_land_where_the_georeferencing_puts_them() {
        for residency in [test_residency(), coarse_residency()] {
            // A patch of a distinct material, well away from the raster's
            // centre so that getting the axes or the origin wrong would move
            // it visibly.
            let (patch_col, patch_row) = (32u32, 96u32);
            let half = 8u32;
            let mut materials = flat_ground();
            for row in patch_row - half..patch_row + half {
                for col in patch_col - half..patch_col + half {
                    materials[(row * RASTER + col) as usize] = SAND;
                }
            }

            let mut camera = None;
            let frame = render_config(
                residency,
                vec![0.0; (RASTER * RASTER) as usize],
                materials,
                |c| {
                    straight_down(c);
                    camera = Some(*c);
                },
                &[],
            );
            let camera = camera.expect("camera captured");
            let base = residency.resident_base;

            let centre = world_of(f64::from(patch_col), f64::from(patch_row));
            let (x, y) = to_pixels(camera.view_projection(), centre, SIZE, SIZE);
            let found = frame.material(x.round() as u32, y.round() as u32);

            assert_eq!(
                found,
                Some(SAND),
                "base {base}: expected the sand patch at ({x:.0}, {y:.0})"
            );

            // ... and the rest of the ground is still the background material,
            // so the patch has not simply been smeared over everything.
            let elsewhere = world_of(f64::from(patch_col), f64::from(RASTER - patch_row));
            let (x, y) = to_pixels(camera.view_projection(), elsewhere, SIZE, SIZE);
            let found = frame.material(x.round() as u32, y.round() as u32);
            assert_eq!(
                found,
                Some(GRASS),
                "base {base}: expected background at ({x:.0}, {y:.0})"
            );
        }
    }

    /// A camera high enough over the dissolve to see it, and low enough that
    /// nothing else is going on.
    ///
    /// [`coarse_residency`] generates one level under a base of 60 m texels and
    /// reaches 24 of the fine ones -- 720 m -- so the band runs from 360 m to
    /// there. From 450 m up, straight down, the whole frame lies inside it: the
    /// middle of the picture is 450 m from the eye and the corners about 555,
    /// so the share falls from three quarters to under a half across the frame.
    /// Any lower and the band is off the top of the picture; any higher and
    /// `detail_base` starts blending on its own account, which is a second
    /// thing to explain in one measurement.
    const DISSOLVE_ALTITUDE: f32 = 450.0;

    /// Flat ground, no relief, and a cover that changes every other texel.
    ///
    /// The plane is what makes the picture a projection with nothing in it but
    /// the cover: a crown or a stone would stand off it, and the fractal would
    /// crumple it. Grass and water because nothing grows on either -- see
    /// `standing_on` in the shader, which walks past both.
    ///
    /// The checkerboard is what makes the two levels *disagree*. A generated
    /// level upscales the survey's cover with a fractal warp and a resident one
    /// reads it square, so they answer differently only near a boundary; two
    /// texels to a square puts a boundary within reach of everywhere.
    fn dissolve_ground() -> (Vec<f32>, Vec<MaterialId>) {
        let heights = vec![0.0f32; (RASTER * RASTER) as usize];
        let materials = (0..RASTER * RASTER)
            .map(|index| {
                let (col, row) = (index % RASTER, index / RASTER);
                if (col / 2 + row / 2).is_multiple_of(2) {
                    GRASS
                } else {
                    LAKE
                }
            })
            .collect();
        (heights, materials)
    }

    fn looking_down_from(altitude: f32, east: f32) -> impl FnOnce(&mut Camera) {
        move |camera: &mut Camera| {
            camera.position = Vec3::new(east, altitude, 0.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
        }
    }

    /// The dissolve has to be fixed to the ground rather than to the frame, and
    /// this is the test that says so.
    ///
    /// A camera looking straight down at a plane projects it linearly -- every
    /// point of the plane is the same distance along the view axis -- so sliding
    /// the camera sideways slides the whole picture by exactly that much and
    /// changes nothing else. Slide it by a whole number of pixels and the two
    /// frames must overlap exactly. Whatever the dissolve did to the first, it
    /// must have done to the same ground in the second.
    ///
    /// This is what a coin thrown off the pixel, off the frame counter, or off
    /// anything screen-space would fail: the pattern would stand still while
    /// the ground moved under it, which reads as the ground crawling. It is
    /// also why the coin is thrown off the texel index and not off a world
    /// position with the height in it.
    #[test]
    fn the_dissolve_is_fixed_to_the_ground_and_not_to_the_frame() {
        let (heights, materials) = dissolve_ground();
        let residency = Residency {
            // A plane, exactly: the fractal would put a decimetre of crumple on
            // it and the projection would stop being a slide.
            detail_relief: 0.0,
            ..coarse_residency()
        };
        // One pixel, on the ground. The frame is square, so the horizontal
        // field of view is the vertical one.
        let half = f64::from(crate::camera::FOV_Y_DEGREES).to_radians() * 0.5;
        let metres_per_pixel = 2.0 * f64::from(DISSOLVE_ALTITUDE) * half.tan() / f64::from(SIZE);
        let across = 8u32;
        let frame = |east: f32| {
            render_config(
                residency,
                heights.clone(),
                materials.clone(),
                looking_down_from(DISSOLVE_ALTITUDE, east),
                &[],
            )
        };
        let still = frame(0.0);
        let slid = frame((f64::from(across) * metres_per_pixel) as f32);

        // Away from an edge, because the last place a boundary falls is decided
        // by arithmetic the slide does not leave alone: the two eyes are a
        // hundred metres apart, and the divide that turns a world position into
        // a texel index rounds where it lands. That moves a boundary by up to a
        // pixel, which over a checkerboard thirty pixels to a square is a fringe
        // of a few percent and says nothing about the coin. So a pixel counts
        // only where both frames are flat around it -- which is most of every
        // square, and all of what a coin thrown per texel decides.
        // Compared by material id rather than by colour. The dissolve decides
        // which level's ids a texel reads, so the id is what it changes and the
        // id is exact; the colour also carries the aerial perspective, which is
        // a screen-space cache 32 froxels wide and so genuinely does move when
        // the camera slides. That is honest behaviour of the haze and nothing
        // to do with the dissolve, and comparing ids keeps the two apart.
        let flat = |frame: &Frame, x: u32, y: u32| {
            let here = frame.material(x, y);
            (-1..=1).all(|dy: i32| {
                (-1..=1).all(|dx: i32| {
                    frame.material(x.wrapping_add_signed(dx), y.wrapping_add_signed(dy)) == here
                })
            })
        };
        // Moving the eye east moves the ground west in the picture, so the
        // second frame's column `x` is the first's `x + across`.
        let (mut same, mut differ) = (0u32, 0u32);
        for y in 1..SIZE - 1 {
            for x in 1..SIZE - across - 1 {
                if !flat(&still, x + across, y) || !flat(&slid, x, y) {
                    continue;
                }
                if still.material(x + across, y) == slid.material(x, y) {
                    same += 1;
                } else {
                    differ += 1;
                }
            }
        }
        assert!(
            same + differ > 20_000,
            "only {} pixels sit away from an edge, which is too few to judge",
            same + differ
        );
        let share = f64::from(same) / f64::from(same + differ);
        println!(
            "{share:.4} of {} compared pixels slid with the ground",
            same + differ
        );
        // Eight pixels, which is sixteen metres of a three-hundred-and-sixty
        // metre band. The share itself is measured from the camera and so moves
        // with it -- that is what makes the seam creep instead of jumping --
        // and a slide long enough to matter carries a percent or two of texels
        // across their own coin, which is the dissolve working rather than a
        // pattern coming loose. A short slide does not, so this can ask for
        // very nearly all of it: a coin thrown anywhere but the ground would
        // re-roll every texel in the band and lose a third of them.
        assert!(
            share > 0.999,
            "only {share:.4} of the overlap survived a slide of {across} pixels, \
             so the dissolve is not fixed to the ground"
        );
    }

    /// And it has to be a dissolve: a share that falls off with distance, not
    /// an edge that has moved inwards.
    ///
    /// The same plane, rendered with the band open and with it shut, and the
    /// texels that changed counted in the middle of the frame against the
    /// corners. The middle is 450 m from the eye and the corners about 555, so
    /// with the band running 360 m to 720 the share of ground still drawn at
    /// the fine level falls from three quarters to under a half between them --
    /// and what changed is one minus that. An edge would have given the same
    /// answer everywhere inside it and nothing outside.
    #[test]
    fn the_handover_is_a_share_that_falls_with_distance() {
        let (heights, materials) = dissolve_ground();
        let shape = |fade: f32| Residency {
            detail_relief: 0.0,
            detail_fade: fade,
            ..coarse_residency()
        };
        let frame = |fade: f32| {
            render_config(
                shape(fade),
                heights.clone(),
                materials.clone(),
                looking_down_from(DISSOLVE_ALTITUDE, 0.0),
                &[],
            )
        };
        // One is the band shut: the level is used out to its reach and then
        // stops, which from here means the whole frame.
        let shut = frame(1.0);
        let open = frame(Residency::default().detail_fade);

        let middle = SIZE / 2;
        let (mut near, mut near_seen) = (0u32, 0u32);
        let (mut far, mut far_seen) = (0u32, 0u32);
        for y in 0..SIZE {
            for x in 0..SIZE {
                let radius = Vec2::new(x as f32 - middle as f32, y as f32 - middle as f32).length();
                let changed = u32::from(shut.pixel(x, y) != open.pixel(x, y));
                if radius < 0.25 * SIZE as f32 {
                    near_seen += 1;
                    near += changed;
                } else if radius > 0.45 * SIZE as f32 {
                    far_seen += 1;
                    far += changed;
                }
            }
        }
        let (near, far) = (
            f64::from(near) / f64::from(near_seen),
            f64::from(far) / f64::from(far_seen),
        );
        println!("changed: {near:.3} in the middle of the frame, {far:.3} at the corners");
        assert!(
            near > 0.01,
            "nothing gave way in the middle of the band, which is {near:.3} changed"
        );
        assert!(
            far > 3.0 * near,
            "the corners are {far:.3} changed against {near:.3} in the middle, \
             which is an edge rather than a share falling off with distance"
        );
    }

    /// Tiles with nothing under them are never written, so a survey's ragged
    /// edge arrives as nodata in the middle of the raster rather than only at
    /// its border. Without the shader's test those texels would draw as a pit
    /// thirty kilometres deep; with it the sky shows through instead.
    #[test]
    fn a_hole_in_the_middle_of_the_data_is_cut_out_rather_than_drawn() {
        const NODATA: f32 = -32767.0;

        let with_hole = |hole: bool| {
            let mut heights = vec![0.0f32; (RASTER * RASTER) as usize];
            if hole {
                for row in 56..72 {
                    for col in 56..72 {
                        heights[(row * RASTER + col) as usize] = NODATA;
                    }
                }
            }
            heights
        };
        let solid = render(with_hole(false), flat_ground(), straight_down).count_sky();
        assert_eq!(
            solid, 0,
            "looking straight down at unbroken ground should show no sky"
        );

        let punched = render(with_hole(true), flat_ground(), straight_down).count_sky();
        assert!(
            punched > 200,
            "the hole should show sky through it, got {punched} pixels"
        );
    }

    /// Two plateaus, the near one standing in front of the far one.
    ///
    /// Exercises occlusion inside the traversal itself: nothing else is drawn,
    /// so the only thing that can hide the far plateau is the march stopping at
    /// the near ridge first.
    #[test]
    fn a_near_ridge_hides_what_is_behind_it() {
        let ridges = |near: bool| {
            let mut heights = vec![0.0f32; (RASTER * RASTER) as usize];
            let mut materials = flat_ground();
            for row in 0..RASTER {
                let (height, material) = match row {
                    66..=73 if near => (900.0, ROCK),
                    46..=53 => (250.0, LAKE),
                    _ => continue,
                };
                for col in 0..RASTER {
                    heights[(row * RASTER + col) as usize] = height;
                    materials[(row * RASTER + col) as usize] = material;
                }
            }
            (heights, materials)
        };
        let aim = |camera: &mut Camera| {
            camera.position = Vec3::new(0.0, 400.0, world_of(64.0, 76.0).z + 400.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -10f32.to_radians(), 0.0);
        };
        let count_far = |frame: &Frame| {
            (0..SIZE)
                .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
                .filter(|&(x, y)| frame.material(x, y) == Some(LAKE))
                .count()
        };

        let (heights, materials) = ridges(false);
        let alone = count_far(&render_config(
            test_residency(),
            heights,
            materials,
            aim,
            &[],
        ));
        assert!(
            alone > 500,
            "the far plateau should be plainly in shot on its own, got {alone} pixels"
        );

        let (heights, materials) = ridges(true);
        let occluded = count_far(&render_config(
            test_residency(),
            heights,
            materials,
            aim,
            &[],
        ));
        assert_eq!(
            occluded, 0,
            "every ray should have stopped at the near ridge"
        );
    }

    /// Nodata and the edge of the raster are holes to a ray as much as to a
    /// triangle, and for the same reason: there is no ground there to draw.
    #[test]
    fn the_far_field_cuts_holes_and_the_data_edge_out_too() {
        const NODATA: f32 = -32767.0;

        let with_hole = |hole: bool| {
            let mut heights = vec![0.0f32; (RASTER * RASTER) as usize];
            if hole {
                for row in 56..72 {
                    for col in 56..72 {
                        heights[(row * RASTER + col) as usize] = NODATA;
                    }
                }
            }
            heights
        };

        let solid = render_config(
            test_residency(),
            with_hole(false),
            flat_ground(),
            straight_down,
            &[],
        )
        .count_sky();
        assert_eq!(solid, 0, "unbroken ground should show no sky");

        let punched = render_config(
            test_residency(),
            with_hole(true),
            flat_ground(),
            straight_down,
            &[],
        )
        .count_sky();
        // Sized, not merely present. The hole is sixteen texels of 30 m, so
        // 480 m across; from 3000 m up, over a frame spanning 3464 m in 256
        // pixels, it projects to about 35 pixels a side and so 1250 of them. A
        // ray refuses the whole quad it is standing in whenever any corner is
        // nodata, which at the level this is marched at widens the cut by one
        // 120 m quad on each side, to about 44 pixels a side.
        //
        // The bound matters because both ways of getting this wrong land inside
        // a loose one. Cutting nothing but the exact quads leaves the ground
        // closing back over the hole, and cutting whatever the ray met after
        // dropping through it -- which is what a hit reported from under the
        // surface amounts to -- shrank this to 49 pixels.
        assert!(
            (1200..2600).contains(&punched),
            "the hole should show sky through it, got {punched} pixels"
        );

        // Climbing until the raster no longer fills the frame puts its edge in
        // shot. Rings reach past it and reads out there repeat the border texel,
        // so a march that did not cut at the data bounds would draw a plateau of
        // invented ground rather than sky.
        let beyond = render_config(
            test_residency(),
            with_hole(false),
            flat_ground(),
            |camera| {
                camera.position = Vec3::new(0.0, 6000.0, 0.0);
                camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
            },
            &[],
        )
        .count_sky();
        assert!(
            beyond > 2000,
            "the ground should stop at the raster's edge, got {beyond} sky pixels"
        );
    }

    /// Ground the materials product says nothing about is missing data, and
    /// missing data is magenta -- not sky. A hole in the *heights* is the
    /// opposite: no ground at all, honestly sky. The pair proves the depth
    /// buffer is what separates the two, because the material id is Null in
    /// both cases and only the depth differs.
    #[test]
    fn unmapped_ground_is_magenta_where_a_height_hole_is_sky() {
        let null = vec![MaterialId(0); (RASTER * RASTER) as usize];

        let flat = vec![0.0f32; (RASTER * RASTER) as usize];
        let frame = render(flat.clone(), null.clone(), straight_down);
        let centre = frame.pixel(SIZE / 2, SIZE / 2);
        // The hue rather than the bytes. What magenta shades to is a fact about
        // the light on it, which the atmosphere decides and
        // `every_material_reaches_the_frame_as_its_own_colour` is the test for;
        // what this one is about is that ground with no material is drawn at
        // all, in the colour of missing data, rather than left as sky.
        assert!(
            is_magenta(centre),
            "unmapped ground should be magenta, got {centre:?}"
        );
        assert!(
            !frame.sky(SIZE / 2, SIZE / 2),
            "a ray that met unmapped ground still met ground"
        );

        let mut holed = flat;
        for row in 56..72 {
            for col in 56..72 {
                holed[(row * RASTER + col) as usize] = -32767.0;
            }
        }
        let frame = render(holed, null, straight_down);
        assert!(
            frame.sky(SIZE / 2, SIZE / 2),
            "a hole in the heights should read as sky, got {:?}",
            frame.pixel(SIZE / 2, SIZE / 2)
        );
    }

    /// An id this binary has never heard of -- a tile painted by a newer
    /// material enum, or a corrupt texel -- draws as missing data rather than
    /// as whatever colour a neighbouring table slot happens to hold.
    #[test]
    fn an_unassigned_id_draws_as_missing_data() {
        let frame = render(
            vec![0.0; (RASTER * RASTER) as usize],
            vec![UNASSIGNED; (RASTER * RASTER) as usize],
            straight_down,
        );
        let centre = frame.pixel(SIZE / 2, SIZE / 2);
        assert!(
            is_magenta(centre),
            "unassigned ids should be magenta, got {centre:?}"
        );
        assert!(
            !frame.sky(SIZE / 2, SIZE / 2),
            "an unassigned id is still ground"
        );
    }

    /// Renders one frame and reads the depth buffer back.
    ///
    /// Nothing in a frame shows a depth: the shading reads it only to tell
    /// ground from sky, so a hit at the wrong distance draws in exactly the
    /// right colour. This is the only way to see what the march wrote.
    fn render_depths(heights: Vec<f32>, aim: impl FnOnce(&mut Camera)) -> Vec<f32> {
        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let mut scene = test_scene(&device, format, test_residency(), heights, flat_ground());
        aim(&mut scene.camera);
        scene.update(&device, &queue, crate::headless::STEP);

        // One float a texel, and `SIZE * 4` is already a multiple of the
        // 256-byte copy alignment.
        let bytes_per_row = SIZE * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depth readback"),
            size: u64::from(bytes_per_row * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let profiler = crate::profile::profiler(&device, false);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut gpu = profiler.scope("gpu", &mut encoder);
            scene.draw(
                &mut gpu,
                &target.create_view(&wgpu::TextureViewDescriptor::default()),
            );
        }
        encoder.copy_texture_to_buffer(
            scene.gbuffer.targets.depth.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let bytes = readback
            .get_mapped_range(..)
            .expect("buffer not mapped")
            .to_vec();
        readback.unmap();

        bytes
            .chunks_exact(4)
            .map(|texel| f32::from_le_bytes([texel[0], texel[1], texel[2], texel[3]]))
            .collect()
    }

    /// Level ground seen from straight above is all at one depth.
    ///
    /// Reversed-Z depth is `z_near` over the distance along the *view axis*, and
    /// looking straight down at a plane that distance is the altitude, the same
    /// for every pixel however far off centre it is. So the whole frame is one
    /// number, and any pixel that is not is the march placing a hit somewhere
    /// the ground is not.
    ///
    /// The one that would is the ray too close to vertical to cross a texel
    /// wall. It has no wall to stop at, so the segment the crossing is bracketed
    /// over comes back as the whole march budget, and halving that eight times
    /// leaves the hit a long way under the ground -- invisible in the frame,
    /// because the material and the normal are read at the texel the ray is
    /// standing in and come out right, and wrong in the buffer the reprojection
    /// and the motion field are built from. Straight down over a plane is where
    /// those rays are, in a disc of a few pixels around the centre.
    #[test]
    fn ground_square_on_to_the_camera_is_all_at_one_depth() {
        let depths = render_depths(vec![0.0; (RASTER * RASTER) as usize], |camera| {
            camera.position = Vec3::new(0.0, 3000.0, 0.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
        });
        // The corners of a square frame reach past a square raster, and what is
        // out there is sky rather than ground.
        let ground: Vec<f32> = depths.iter().copied().filter(|&d| d != 0.0).collect();
        assert!(
            ground.len() > depths.len() / 2,
            "only {} of {} pixels found ground; the camera has to be over it",
            ground.len(),
            depths.len()
        );
        let want = 1.0 / 3000.0;
        let worst = ground
            .iter()
            .map(|&d| (d - want).abs() / want)
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "the furthest pixel is {:.3}% off the {want} every pixel should read",
            worst * 100.0
        );
    }

    /// Renders one frame and reads the normal buffer back as world vectors.
    ///
    /// The shading reduces a normal to one number, so this is the only way to
    /// see the vector the march actually wrote.
    fn render_normals(heights: Vec<f32>, aim: impl FnOnce(&mut Camera)) -> Vec<[f32; 4]> {
        render_normals_config(test_residency(), heights, aim)
    }

    /// As [`render_normals`], but over a residency configured by the caller.
    fn render_normals_config(
        residency: Residency,
        heights: Vec<f32>,
        aim: impl FnOnce(&mut Camera),
    ) -> Vec<[f32; 4]> {
        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let mut scene = test_scene(&device, format, residency, heights, flat_ground());
        aim(&mut scene.camera);
        scene.update(&device, &queue, crate::headless::STEP);

        // Four half floats a texel, and `SIZE * 8` is already a multiple of
        // the 256-byte copy alignment.
        let bytes_per_row = SIZE * 8;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("normal readback"),
            size: u64::from(bytes_per_row * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let profiler = crate::profile::profiler(&device, false);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut gpu = profiler.scope("gpu", &mut encoder);
            scene.draw(
                &mut gpu,
                &target.create_view(&wgpu::TextureViewDescriptor::default()),
            );
        }
        encoder.copy_texture_to_buffer(
            scene.gbuffer.targets.normal.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let bytes = readback
            .get_mapped_range(..)
            .expect("buffer not mapped")
            .to_vec();
        readback.unmap();

        bytes
            .chunks_exact(8)
            .map(|texel| {
                std::array::from_fn(|channel| {
                    let at = channel * 2;
                    half::f16::from_le_bytes([texel[at], texel[at + 1]]).to_f32()
                })
            })
            .collect()
    }

    /// A tilted plane has one normal, and the march must write that one.
    ///
    /// The shading reduces a normal to a single number, so a frame can only
    /// say that the vector was about right; this reads the buffer back and
    /// checks the direction itself, which is what catches a sign flipped
    /// between a raster row and world +Z, or a column and +X.
    #[test]
    fn the_march_writes_the_normal_of_the_ground_it_hit() {
        // Rising eastward and falling southward, at different rates, so the
        // two axes cannot be swapped or negated without this noticing.
        let (east, south) = (0.2f32, -0.35f32);
        let metres = METRES_PER_TEXEL as f32;
        let heights: Vec<f32> = (0..RASTER * RASTER)
            .map(|index| {
                let (x, y) = ((index % RASTER) as f32, (index / RASTER) as f32);
                (east * x + south * y) * metres
            })
            .collect();

        let normals = render_normals(heights, straight_down);
        let expected = Vec3::new(-east, 1.0, -south).normalize();

        let mut written = 0;
        for (index, normal) in normals.iter().enumerate() {
            let got = Vec3::new(normal[0], normal[1], normal[2]);
            // The buffer clears to zero, so a pixel the march discarded is a
            // vector of no length rather than a direction.
            if got.length() < 0.5 {
                continue;
            }
            written += 1;
            let (x, y) = (index as u32 % SIZE, index as u32 / SIZE);
            // Nothing quantises this any more -- the march differences the
            // height texture itself -- so all that is left is the half floats
            // of the G-buffer and the rounding of the difference.
            assert!(
                (got - expected).length() < 0.005,
                "pixel ({x}, {y}) holds {got:?}, not {expected:?}"
            );
        }
        assert!(
            written > 1000,
            "only {written} pixels of the frame hit the ground"
        );
    }

    /// Ground beside a hole is shaded by the ground that was measured.
    ///
    /// A nodata texel is not a height, so a difference across one has to fall
    /// back to the samples that are real rather than average the sentinel in --
    /// which would tilt the ground beside every hole by thousands of metres per
    /// texel -- or give up and call it flat, which would flatten the shoreline
    /// of every lake in the survey. On a plane both fallbacks are exact, so
    /// punching a hole must not move a single normal that is still drawn.
    ///
    /// The pass that used to bake these normals owned this rule and tested it;
    /// the march owns it now.
    #[test]
    fn a_hole_leaves_the_normals_of_the_ground_beside_it_alone() {
        const NODATA: f32 = -32767.0;
        let (east, south) = (0.2f32, -0.35f32);
        let metres = METRES_PER_TEXEL as f32;
        let plane = |hole: bool| -> Vec<f32> {
            (0..RASTER * RASTER)
                .map(|index| {
                    let (col, row) = (index % RASTER, index / RASTER);
                    if hole && (48..80).contains(&col) && (48..80).contains(&row) {
                        return NODATA;
                    }
                    (east * col as f32 + south * row as f32) * metres
                })
                .collect()
        };

        let expected = Vec3::new(-east, 1.0, -south).normalize();
        let count = |heights: Vec<f32>| {
            let mut written = 0;
            for (index, normal) in render_normals(heights, straight_down).iter().enumerate() {
                let got = Vec3::new(normal[0], normal[1], normal[2]);
                // The buffer clears to zero, so a pixel the march discarded is
                // a vector of no length rather than a direction.
                if got.length() < 0.5 {
                    continue;
                }
                written += 1;
                let (x, y) = (index as u32 % SIZE, index as u32 / SIZE);
                assert!(
                    (got - expected).length() < 0.005,
                    "pixel ({x}, {y}) holds {got:?}, not {expected:?}"
                );
            }
            written
        };

        let solid = count(plane(false));
        let punched = count(plane(true));
        // The hole has to actually reach the frame, or every assertion above
        // passed on ground that never knew there was one.
        assert!(
            solid - punched > 1000,
            "only {} pixels of the frame fell in the hole",
            solid - punched
        );
    }

    /// Curved ground must shade as a curve, not as a staircase of facets.
    ///
    /// Reading the nearest normal texel is flat shading: every texel holds one
    /// constant direction, so the ground breaks into facets and reads as
    /// blocks -- which is what it did look like, and the reason the march
    /// interpolates. What separates the two is not whether the normals are
    /// right on average but whether they are continuous, so this measures the
    /// step between neighbouring pixels rather than the value at any of them.
    #[test]
    fn a_curved_surface_gives_a_continuous_normal_not_facets() {
        // A dome over the whole raster: every direction of slope, none of it
        // flat, and the curvature gentle enough that the stored normals of
        // adjacent texels are genuinely different rather than a step apart.
        let metres = METRES_PER_TEXEL as f32;
        let radius = 0.5 * RASTER as f32 * metres;
        let heights: Vec<f32> = (0..RASTER * RASTER)
            .map(|index| {
                let (x, y) = ((index % RASTER) as f32, (index / RASTER) as f32);
                let half = 0.5 * (RASTER - 1) as f32;
                let offset = Vec2::new(x - half, y - half) * metres / radius;
                600.0 * (1.0 - offset.length_squared())
            })
            .collect();

        // The one test here that has to be told the truth about how big a pixel
        // is. Every other fixture takes [`test_residency`]'s deliberately
        // coarse `pixel_angle`, which exists so that a 256-pixel frame of a
        // 128-texel raster gives up levels at all -- but the march now
        // dissolves the handover between two levels rather than switching, and
        // it sizes that dissolve by the same angle. A pixel fifteen times its
        // true width puts the dissolve's texels twenty-four pixels across
        // instead of one, which reads as exactly the facet this is looking for.
        // With the real angle the camera sits on the finest level with nothing
        // to blend into, which is where a question about interpolation *within*
        // a level belongs.
        let honest = Residency {
            pixel_angle: crate::terrain::residency::pixel_angle(
                SIZE,
                f64::from(crate::camera::FOV_Y_DEGREES).to_radians(),
            ),
            ..test_residency()
        };
        let normals = render_normals_config(honest, heights, straight_down);
        let at = |x: u32, y: u32| {
            let [nx, ny, nz, _] = normals[(y * SIZE + x) as usize];
            Vec3::new(nx, ny, nz)
        };

        // Along the middle row, where the dome's slope sweeps from tilted one
        // way through level to tilted the other.
        let row = SIZE / 2;
        let mut step = 0.0f32;
        let mut sweep = 0.0f32;
        let mut ground = 0;
        for x in 1..SIZE {
            let (before, here) = (at(x - 1, row), at(x, row));
            // The buffer clears to zero, so a pixel the march discarded holds
            // no direction and neither it nor its neighbour is a step.
            if before.length() < 0.5 || here.length() < 0.5 {
                continue;
            }
            ground += 1;
            step = step.max((here - before).length());
            sweep = sweep.max((here - at(1, row)).length());
        }

        assert!(
            ground > 200,
            "only {ground} pixels of ground across the row"
        );
        // The dome has to be curved enough for the measurement to mean
        // something: a flat plane would pass any smoothness test there is.
        assert!(
            sweep > 0.6,
            "the row only turns through {sweep}, which is too flat to judge"
        );
        // One stored texel covers many pixels at this distance, so flat
        // shading would hold the normal still and then jump by the whole
        // difference between neighbouring texels -- a step far larger than
        // this. Interpolating spreads that difference over the texel.
        assert!(
            step < 0.02,
            "neighbouring pixels differ by up to {step}, which is a facet edge"
        );
    }

    /// A plane through the middle of the raster, tilted south-east.
    ///
    /// `fall` of -1 puts the plane square-on to the reference sun, which sits
    /// 45 degrees above the horizon in the south-east; +1 turns it as far away
    /// as a plane can be turned. The march differences these very heights, so
    /// the normal the shading dots against the sun is the gradient written
    /// here. Through the middle so it stays under the camera whichever way it
    /// tilts.
    fn tilted_plane(fall: f32) -> Vec<f32> {
        let metres = METRES_PER_TEXEL as f32;
        (0..RASTER * RASTER)
            .map(|index| {
                let (x, y) = ((index % RASTER) as f32, (index / RASTER) as f32);
                let across = x + y - (RASTER - 1) as f32;
                fall * std::f32::consts::FRAC_1_SQRT_2 * across * metres
            })
            .collect()
    }

    /// High enough to clear the corner of a plane that reaches 2.7 km, and
    /// still close enough that the finest level is the one drawn.
    fn over_the_plane(camera: &mut Camera) {
        camera.position = Vec3::new(0.0, 9000.0, 0.0);
        camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
    }

    /// Mean brightness of the ground in a frame, on a scale of zero to one.
    fn ground_brightness(frame: &Frame) -> f32 {
        let mut total = 0.0f64;
        let mut count = 0u32;
        for y in 0..SIZE {
            for x in 0..SIZE {
                if frame.sky(x, y) {
                    continue;
                }
                let pixel = frame.pixel(x, y);
                total += f64::from(pixel[0]) + f64::from(pixel[1]) + f64::from(pixel[2]);
                count += 3;
            }
        }
        assert!(count > 3000, "only {} channels of ground to average", count);
        (total / f64::from(count) / 255.0) as f32
    }

    /// A slope facing the sun is far brighter than one turned away, and the
    /// one turned away is still lit.
    ///
    /// Both halves used to be pinned to exact bytes, because the light was two
    /// constants and a dot product and the answer could be written down. It
    /// cannot now: the sun arrives through a transmittance table and the shade
    /// is the sky's own multiple scattering, and a test that restated either
    /// would be asserting the shader against a second copy of itself.
    ///
    /// What survives is the pair of facts that made the test worth writing --
    /// the sun does most of the lighting, and ground turned away from it is not
    /// black -- plus a third the old one could not make, because it had no
    /// exposure to get wrong: the sunlit slope does not clip. That last is what
    /// says the whole chain from irradiance to byte is in range, and it is the
    /// thing a wrong `EXPOSURE` would break while every ratio stayed perfect.
    #[test]
    fn a_slope_facing_the_sun_is_far_brighter_than_one_facing_away() {
        let brightness = |fall: f32| {
            ground_brightness(&render_config(
                test_residency(),
                tilted_plane(fall),
                flat_ground(),
                over_the_plane,
                &[],
            ))
        };
        let (toward, away) = (brightness(-1.0), brightness(1.0));
        println!("square-on {toward:.4}, turned away {away:.4}");

        // Half again, measured rather than guessed, and lower than it once was
        // on purpose: the haze in front of both slopes adds the same floor to
        // each and so compresses the ratio. Square-on against turned-away is
        // 0.521 to 0.299, a factor of 1.74; dropping the direct sun altogether
        // leaves 0.310 to 0.299, a factor of 1.04. The threshold sits between.
        assert!(
            toward > 1.4 * away,
            "square-on to the sun gives {toward:.4} against {away:.4} turned \
             away, which is not the difference a sun makes"
        );
        assert!(
            away > 0.05,
            "ground turned away from the sun came out at {away:.4}, which is \
             black -- the sky is supposed to light it"
        );
        assert!(
            toward < 0.98,
            "sunlit ground came out at {toward:.4}, which has clipped: the \
             exposure is too high for the light the model produces"
        );
    }

    /// Distant ground fades into the air in front of it.
    ///
    /// The frame-level test for the aerial perspective, and it needs a fixture
    /// of its own: the raster every other test here uses is 3.8 km corner to
    /// corner, over which a hundred kilometres of air is a two percent effect
    /// and nothing about it can be measured. The same 128 texels at 900 m
    /// apiece span 115 km, which is what the installed survey actually covers.
    ///
    /// One flat plane of one material, seen obliquely, so that near ground and
    /// far ground differ in nothing whatever except how much air stands in
    /// front of them -- same albedo, same normal, same sun. Anything that
    /// separates the two bands is the air and can be nothing else.
    #[test]
    fn distant_ground_fades_into_the_air_in_front_of_it() {
        let wide = Georeferencing::square(RASTER, RASTER, 900.0);
        let frame = render_over(
            test_residency(),
            vec![0.0; (RASTER * RASTER) as usize],
            flat_ground(),
            |camera| {
                // Low and looking out, so the bottom of the frame is a few
                // kilometres away and the top is tens.
                camera.position = Vec3::new(0.0, 3000.0, 50_000.0);
                camera.orientation = Camera::from_yaw_pitch_roll(0.0, -6f32.to_radians(), 0.0);
            },
            &[],
            crate::sky::Sun::default(),
            wide,
            crate::cloud::Preset::Clear,
            None,
        );

        // Two bands of ground, near the bottom of the frame and just under the
        // skyline, with their distances read out of the depth buffer rather
        // than assumed.
        let band = |row: u32| {
            let mut colour = Vec3::ZERO;
            let mut count = 0.0;
            for x in 0..SIZE {
                if frame.sky(x, row) {
                    continue;
                }
                let pixel = frame.pixel(x, row);
                colour += Vec3::new(
                    f32::from(pixel[0]),
                    f32::from(pixel[1]),
                    f32::from(pixel[2]),
                );
                count += 1.0;
            }
            assert!(
                count > 100.0,
                "row {row} holds only {count} pixels of ground"
            );
            colour / count
        };
        // The skyline: the lowest row that is still entirely sky. Found rather
        // than worked out, because where the horizon lands depends on the
        // pitch, the altitude and the planet's own curvature together.
        let skyline = (0..SIZE)
            .rev()
            .find(|&y| (0..SIZE).all(|x| frame.sky(x, y)))
            .expect("some row of the frame is all sky");
        let (near, far) = (band(SIZE - 8), band(skyline + 6));
        let horizon = {
            let pixel = frame.pixel(SIZE / 2, skyline.saturating_sub(2));
            Vec3::new(
                f32::from(pixel[0]),
                f32::from(pixel[1]),
                f32::from(pixel[2]),
            )
        };
        println!("near {near}, far {far}, sky above {horizon}");

        // The far band is bluer, which is the light the air put in front of it.
        // Measured: far against near is 1.31 as it stands and 0.52 with the
        // in-scattering dropped -- the distance then goes dark and drab instead
        // of pale and blue -- so the threshold has room on both sides.
        let blueness = |colour: Vec3| colour.z / colour.x.max(1.0);
        assert!(
            blueness(far) > 1.2 * blueness(near),
            "near ground is {:.2} blue-to-red and far ground {:.2}, which is \
             not the air scattering blue in front of the distance",
            blueness(near),
            blueness(far)
        );

        // ... and it has moved towards the sky it stands against.
        assert!(
            (far - horizon).length() < 0.6 * (near - horizon).length(),
            "far ground is {far} against a sky of {horizon}, no closer to it \
             than the near ground at {near}"
        );

        // ... without overshooting it, which is the transmittance's own half of
        // the job and the one the two checks above cannot see. Ground darker
        // than the sky can only ever approach the sky as the air thickens:
        // what the haze adds, the extinction has already taken out. Drop the
        // transmittance and keep the in-scattering and the far band comes out
        // at 170 green against a sky of 157 -- brighter than the air it is
        // seen through, which nothing real does.
        for channel in 0..3 {
            assert!(
                far[channel] <= horizon[channel] + 2.0,
                "far ground is {far} against a sky of {horizon}, which it has \
                 no business being brighter than"
            );
        }
    }

    /// The sky is a gradient, and it turns over as the sun sets.
    ///
    /// The sky used to be one constant, so nothing in this file had anything to
    /// say about it beyond where it was. Two things worth pinning now that it
    /// is computed: it is darker overhead than at the horizon, which is the
    /// shape of every daytime sky and would be flat if the table were being
    /// sampled at one row; and that shape survives the sun going down while the
    /// colour does not, which is what says the table is rebuilt per frame
    /// rather than baked once at load.
    #[test]
    fn the_sky_is_darker_overhead_than_at_the_horizon_and_reddens_at_dusk() {
        // Level ground, and the camera aimed level and towards the sun's own
        // bearing. Level matters: the sunset band is a couple of degrees deep
        // and an eye tilted up by ten misses it entirely, which is what the
        // first draft of this test did.
        let aim = |camera: &mut Camera| {
            camera.position = Vec3::new(0.0, 2000.0, 0.0);
            camera.orientation = Camera::from_yaw_pitch_roll(
                crate::sky::Sun::DEFAULT_AZIMUTH.to_radians(),
                0.0,
                0.0,
            );
        };
        let band = |frame: &Frame, row: u32| {
            let mut sum = Vec3::ZERO;
            let mut count = 0.0;
            for x in 0..SIZE {
                if !frame.sky(x, row) {
                    continue;
                }
                let pixel = frame.pixel(x, row);
                sum += Vec3::new(
                    f32::from(pixel[0]),
                    f32::from(pixel[1]),
                    f32::from(pixel[2]),
                );
                count += 1.0;
            }
            assert!(count > 100.0, "row {row} holds only {count} pixels of sky");
            sum / count
        };
        let sky_at = |elevation: f32| {
            let frame = render_sunlit(
                test_residency(),
                vec![0.0; (RASTER * RASTER) as usize],
                flat_ground(),
                aim,
                &[],
                crate::sky::Sun::from_angles(elevation, crate::sky::Sun::DEFAULT_AZIMUTH),
            );
            // Row four is about 29 degrees up; four rows above the middle is
            // just under a degree above level, which is where a sunset is.
            (band(&frame, 4), band(&frame, SIZE / 2 - 4))
        };

        let (high_zenith, high_horizon) = sky_at(45.0);
        println!("noon: overhead {high_zenith}, horizon {high_horizon}");
        assert!(
            high_zenith.length() < high_horizon.length(),
            "the sky overhead is {high_zenith} against {high_horizon} at the \
             horizon, and a sky is supposed to be paler as it comes down"
        );
        assert!(
            high_zenith.z / high_zenith.x > high_horizon.z / high_horizon.x,
            "the sky overhead is {high_zenith} against {high_horizon} lower \
             down, which is not bluer at the top"
        );

        // A low sun turns the horizon it is behind orange, which the zenith
        // never does: it is the long path through the air that reddens, and
        // straight up is the shortest path there is.
        let (dusk_zenith, dusk_horizon) = sky_at(2.0);
        println!("dusk: overhead {dusk_zenith}, horizon {dusk_horizon}");
        let warmth = |colour: Vec3| colour.x / colour.z.max(1.0);
        assert!(
            warmth(dusk_horizon) > 2.0 * warmth(high_horizon),
            "the horizon goes from {:.2} red-to-blue at noon to {:.2} at dusk, \
             which is not a sunset",
            warmth(high_horizon),
            warmth(dusk_horizon)
        );
        assert!(
            warmth(dusk_horizon) > 1.5 * warmth(dusk_zenith),
            "at dusk the horizon is {:.2} red-to-blue and the zenith {:.2}, so \
             the whole sky reddened together rather than the low sky alone",
            warmth(dusk_horizon),
            warmth(dusk_zenith)
        );
    }

    /// The sun draws as a disc, where the sun is, about the size the sun is.
    ///
    /// Three claims, and it takes all three: a test that only asked whether
    /// some pixel was bright would pass on a highlight stuck anywhere in the
    /// frame, and one that only checked the position would pass on a sky
    /// smeared white from edge to edge.
    ///
    /// Only sky pixels are counted, which is what makes "white" mean the sun:
    /// snow is white too, and the depth buffer is what tells them apart.
    #[test]
    fn the_sun_draws_as_a_disc_where_the_sun_is() {
        let sun = crate::sky::Sun::from_angles(25.0, crate::sky::Sun::DEFAULT_AZIMUTH);
        // Aimed at the sun, so it lands near the middle of the frame where the
        // projection is least distorted.
        let mut camera = None;
        let frame = render_sunlit(
            test_residency(),
            vec![0.0; (RASTER * RASTER) as usize],
            flat_ground(),
            |c| {
                c.position = Vec3::new(0.0, 2000.0, 0.0);
                c.orientation = Camera::from_yaw_pitch_roll(
                    crate::sky::Sun::DEFAULT_AZIMUTH.to_radians(),
                    25f32.to_radians(),
                    0.0,
                );
                camera = Some(*c);
            },
            &[],
            sun,
        );
        let camera = camera.expect("camera captured");

        // Every sky pixel that has clipped to white in all three channels. The
        // disc is thousands of times over the white point and nothing else in
        // an empty sky comes close, so this is the disc and only the disc.
        let burnt: Vec<(u32, u32)> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| frame.sky(x, y) && frame.pixel(x, y)[..3].iter().all(|&c| c >= 250))
            .collect();
        assert!(!burnt.is_empty(), "the sun is not in the frame at all");

        // Where it should be: a point a long way along the sun's direction,
        // projected. The infinite projection takes it without clipping.
        let (want_x, want_y) = to_pixels(
            camera.view_projection(),
            camera.position + sun.direction * 1.0e6,
            SIZE,
            SIZE,
        );
        let centroid = burnt.iter().fold(Vec2::ZERO, |sum, &(x, y)| {
            sum + Vec2::new(x as f32, y as f32)
        }) / burnt.len() as f32;
        let off = (centroid - Vec2::new(want_x - 0.5, want_y - 0.5)).length();
        println!(
            "{} pixels of sun, centroid ({:.1}, {:.1}) against ({want_x:.1}, {want_y:.1})",
            burnt.len(),
            centroid.x,
            centroid.y
        );
        assert!(
            off < 2.0,
            "the sun draws at ({:.1}, {:.1}) but points from ({want_x:.1}, \
             {want_y:.1}), which is {off:.1} pixels out",
            centroid.x,
            centroid.y
        );

        // And about the right size, which at this resolution means the disc
        // *plus its feather*. The sun is 0.53 degrees across, or 2.3 pixels of
        // a 60-degree frame 256 wide, so the disc alone would be about four
        // pixels -- but one pixel here subtends 0.0045 radians against the
        // sun's own radius of 0.0047, so the one-pixel feather very nearly
        // doubles the radius and twelve pixels is what actually comes out. At
        // 1024 the feather is a quarter as wide and the disc dominates again.
        //
        // Bounded above all the same, which is the half that bites: a disc
        // drawn at some convenient larger angle than the sun's own fails here
        // however well it is centred.
        assert!(
            (4..=24).contains(&burnt.len()),
            "the sun covers {} pixels, where the disc and its feather should \
             cover about twelve at this size",
            burnt.len()
        );
    }

    /// The setting sun loses its blue.
    ///
    /// The disc is carried through the same transmittance table as the sky in
    /// front of it, and this is the only test that can see that term at all.
    /// While the sun is up it cannot: the disc is nine thousand times over the
    /// white point, so anything short of the air removing all but a
    /// ten-thousandth of it still clips to white, and removing the term
    /// entirely changes no pixel. Taking the sun down to where the air really
    /// does remove that much is what makes it measurable.
    ///
    /// A degree *below* level is still above the horizon here, which is the
    /// point of choosing it: from 2000 m the horizon dips 1.44 degrees, so the
    /// sun at -1 is genuinely in the sky and genuinely reddened -- not merely
    /// clipped away by the geometry.
    #[test]
    fn the_setting_sun_loses_its_blue() {
        let disc = |elevation: f32| {
            let frame = render_sunlit(
                test_residency(),
                vec![0.0; (RASTER * RASTER) as usize],
                flat_ground(),
                |c| {
                    c.position = Vec3::new(0.0, 2000.0, 0.0);
                    c.orientation = Camera::from_yaw_pitch_roll(
                        crate::sky::Sun::DEFAULT_AZIMUTH.to_radians(),
                        elevation.to_radians(),
                        0.0,
                    );
                },
                &[],
                crate::sky::Sun::from_angles(elevation, crate::sky::Sun::DEFAULT_AZIMUTH),
            );
            assert!(
                frame.sky(SIZE / 2, SIZE / 2),
                "the middle of the frame has to be sky for the sun to be in it"
            );
            frame.pixel(SIZE / 2, SIZE / 2)
        };

        let level = disc(0.0);
        let setting = disc(-1.0);
        println!("level {level:?}, setting {setting:?}");
        assert_eq!(
            &level[..3],
            &[255, 255, 255],
            "with the sun level the disc should still be white"
        );
        assert_eq!(
            &setting[..3],
            &[255, 255, 105],
            "a degree lower the air should have taken most of the blue out of \
             the disc and left the red alone"
        );
    }

    /// ... and the ground stands in front of it.
    ///
    /// Free, and worth a test precisely because it is free: the disc is drawn
    /// only where the march found no ground, so there is no visibility test to
    /// get wrong. This is what says that remains true.
    #[test]
    fn a_ridge_between_the_eye_and_the_sun_hides_the_sun() {
        let sun = crate::sky::Sun::from_angles(8.0, crate::sky::Sun::DEFAULT_AZIMUTH);
        let aim = |camera: &mut Camera| {
            camera.position = Vec3::new(0.0, 300.0, 0.0);
            camera.orientation = Camera::from_yaw_pitch_roll(
                crate::sky::Sun::DEFAULT_AZIMUTH.to_radians(),
                8f32.to_radians(),
                0.0,
            );
        };
        // A wall across the south-east, high enough to stand over an eight
        // degree sun from 300 m up.
        let wall = |raised: bool| -> Vec<f32> {
            (0..RASTER * RASTER)
                .map(|index| {
                    let (col, row) = (index % RASTER, index / RASTER);
                    let towards = col > 88 && row > 88;
                    if raised && towards { 4000.0 } else { 0.0 }
                })
                .collect()
        };
        let burnt = |raised: bool| {
            let frame = render_sunlit(test_residency(), wall(raised), flat_ground(), aim, &[], sun);
            (0..SIZE)
                .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
                .filter(|&(x, y)| {
                    frame.sky(x, y) && frame.pixel(x, y)[..3].iter().all(|&c| c >= 250)
                })
                .count()
        };

        let open = burnt(false);
        assert!(
            open > 0,
            "the sun has to be in shot for this to mean anything"
        );
        assert_eq!(
            burnt(true),
            0,
            "a ridge in front of the sun left {open} pixels of it showing"
        );
    }

    /// Lowering the sun dims the ground and reddens it.
    ///
    /// Nothing else in this file would notice if the sun uniform were ignored
    /// and the shader had quietly kept a constant, because every other test
    /// renders under the default sun -- which was chosen to reproduce that
    /// constant. This is the test that says the sun is a value.
    ///
    /// It is also the one that says the *transmittance table* is being read
    /// rather than a plain cosine: a Lambert term alone would dim the ground
    /// without changing its colour at all, and what makes a low sun orange is
    /// the air taking the blue out of it on the way in.
    #[test]
    fn lowering_the_sun_dims_the_ground_and_reddens_it() {
        // Level ground, so the only thing changing between the three frames is
        // where the sun is. A slope would confound the dimming with a cosine.
        let warmth = |elevation: f32| {
            let frame = render_sunlit(
                test_residency(),
                vec![0.0; (RASTER * RASTER) as usize],
                flat_ground(),
                |c: &mut Camera| {
                    c.position = Vec3::new(0.0, 500.0, 0.0);
                    c.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
                },
                &[],
                crate::sky::Sun::from_angles(elevation, crate::sky::Sun::DEFAULT_AZIMUTH),
            );
            let mut warm = 0.0f64;
            let mut cool = 0.0f64;
            for y in 0..SIZE {
                for x in 0..SIZE {
                    let pixel = frame.pixel(x, y);
                    warm += f64::from(pixel[0]);
                    cool += f64::from(pixel[2]);
                }
            }
            (ground_brightness(&frame), (warm / cool.max(1.0)) as f32)
        };

        let (high, high_warmth) = warmth(60.0);
        let (low, low_warmth) = warmth(6.0);
        let (dusk, _) = warmth(-4.0);
        println!(
            "60 deg: {high:.4} at {high_warmth:.3} red:blue; \
             6 deg: {low:.4} at {low_warmth:.3}; -4 deg: {dusk:.4}"
        );

        assert!(
            high > low && low > dusk,
            "the ground should dim as the sun sets: {high:.4} at 60 degrees, \
             {low:.4} at 6, {dusk:.4} at -4"
        );
        // Measured rather than guessed. Reading the transmittance table takes
        // red against blue from 1.34 to 2.21 as the sun drops, a factor of
        // 1.65; replacing that lookup with the constant it averages leaves the
        // sky and the haze as the only things that redden, and the same
        // measurement gives 1.24 to 1.40, a factor of 1.13. The threshold sits
        // between, so this fails if the sun stops arriving through the air.
        assert!(
            low_warmth > 1.4 * high_warmth,
            "dropping the sun from 60 degrees to 6 moved red against blue only \
             from {high_warmth:.3} to {low_warmth:.3}, which is a cosine \
             dimming the light rather than the air reddening it"
        );
        assert!(
            dusk < 0.25 * low,
            "with the sun below the horizon the ground is still at {dusk:.4} \
             against {low:.4} with it up"
        );
    }

    #[test]
    fn walking_the_camera_there_looks_the_same_as_arriving_directly() {
        // The incremental toroidal update path is only correct if it agrees
        // with the trivially-correct full refresh. Walk far enough that every
        // level's window moves and the finest wraps around its texture.
        let heights: Vec<f32> = (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as f32, (i / RASTER) as f32);
                120.0 * ((x * 0.21).sin() + (y * 0.17).cos()) + 60.0 * (x * 0.05 + y * 0.03).sin()
            })
            .collect();
        // Every material in the book, tiled: any misregistered window shows
        // as one id where another belongs, and ids compare exactly.
        let materials: Vec<MaterialId> = (0..RASTER * RASTER)
            .map(|i| MaterialId(Material::ALL[i as usize % Material::ALL.len()].id() as u16))
            .collect();

        let aim = |camera: &mut Camera| {
            camera.position = Vec3::new(400.0, 900.0, 300.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -30f32.to_radians(), 0.0);
        };

        let direct = render(heights.clone(), materials.clone(), aim);

        let steps: Vec<Vec3> = (0..200)
            .map(|i| {
                let t = f32::from(i as u16);
                Vec3::new(-1400.0 + t * 9.0, 900.0, 1500.0 - t * 6.0)
            })
            .collect();
        let walked = render_after(heights, materials, aim, &steps);

        // Byte-exact, as it has always been. Reported as a count rather than
        // through `assert_eq!` only because the two sides are a quarter of a
        // megabyte each and the difference is the readable part of them.
        let differing = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| direct.pixel(x, y) != walked.pixel(x, y))
            .count();
        assert_eq!(
            differing, 0,
            "incremental clipmap updates diverged from a full refresh in {differing} pixels"
        );
    }

    /// A scene over flat ground, for the tests that are about bookkeeping
    /// rather than about pixels and never draw a frame at all.
    fn clockwork_scene(device: &wgpu::Device) -> Scene {
        test_scene(
            device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            test_residency(),
            vec![0.0; (RASTER * RASTER) as usize],
            flat_ground(),
        )
    }

    #[test]
    fn a_frame_winds_the_world_clock_on_by_the_step_it_was_handed() {
        let (device, queue) = test_device();
        let mut scene = clockwork_scene(&device);
        assert_eq!(scene.elapsed, Duration::ZERO, "a new scene has no history");

        for frames in 1..=5u32 {
            scene.update(&device, &queue, crate::headless::STEP);
            assert_eq!(
                scene.elapsed,
                crate::headless::STEP * frames,
                "after {frames} nominal frames"
            );
        }
    }

    /// The clock is the sum of the steps it was handed, not of the ones it
    /// might have wanted.
    ///
    /// Sixty nominal frames are one second exactly, whatever the machine took
    /// to draw them -- which is the whole reason [`crate::headless::STEP`] is a
    /// nominal figure. A run whose weather depended on how fast the GPU was
    /// could not be compared with another run of the same flight.
    #[test]
    fn a_nominal_second_of_frames_is_a_second() {
        let (device, queue) = test_device();
        let mut scene = clockwork_scene(&device);
        for _ in 0..60 {
            scene.update(&device, &queue, crate::headless::STEP);
        }
        // Not `assert_eq!` against one second: the step is a whole number of
        // nanoseconds and sixty of them are a nanosecond short of a second.
        // That shortfall is the point being pinned -- it is fixed, and it does
        // not accumulate differently on different machines.
        assert_eq!(scene.elapsed, Duration::from_nanos(999_999_960));
    }

    /// Settling is not time passing -- the same statement the frame counter
    /// makes by being put back, made where it can be made directly.
    #[test]
    fn settling_the_scene_does_not_wind_the_world_clock_on() {
        let (device, queue) = test_device();
        let mut scene = clockwork_scene(&device);

        scene.update(&device, &queue, crate::headless::STEP);
        let drawn = scene.elapsed;
        scene.settle(&device, &queue);

        assert_eq!(
            scene.elapsed,
            drawn,
            "settling aged the world by {:?}",
            scene.elapsed.saturating_sub(drawn),
        );
    }

    /// A scene solves its wind against the terrain it was built over, once.
    ///
    /// The one test that lets a scene do it. Every other scene in this file
    /// skips the solve -- see [`test_scene_over`] -- so this is what says the
    /// wiring works at all: that `Scene::update` reaches the bake, that the
    /// bake finds the terrain's coarse heights once the chain has been read in,
    /// and that it does not run again afterwards.
    ///
    /// It is deliberately not an assertion about the *field*, which this raster
    /// is far too small to say anything useful about. What the solve produces
    /// is `src/air.rs`'s own business and is tested there, over ground drawn
    /// for the purpose.
    #[test]
    fn a_scene_solves_the_wind_over_its_own_terrain() {
        let (device, queue) = test_device();
        let breeze = crate::air::Wind {
            speed: 8.0,
            from_degrees: 200.0,
        };
        let mut scene = test_scene_with_wind(
            &device,
            vec![0.0; (RASTER * RASTER) as usize],
            flat_ground(),
        );
        scene.wind = breeze;

        // Nothing is solved before the chain has been read in, because there is
        // no ground to solve around yet.
        assert_eq!(scene.air.baked_for(), None);

        scene.settle(&device, &queue);
        assert_eq!(
            scene.air.baked_for(),
            Some(breeze),
            "settling did not reach the bake"
        );

        // And it is spent: a second pass cannot solve again, because the bake
        // dropped everything it would need to.
        scene.update(&device, &queue, crate::headless::STEP);
        assert_eq!(scene.air.baked_for(), Some(breeze));
    }

    /// A stalled frame ages the world by a frame's worth, not by the stall.
    ///
    /// The same guard the controller puts on how far the camera is flung; see
    /// [`MAX_STEP`] and the one of that name in `src/controls.rs`. Without it a
    /// shader compile or a dragged window would jump the weather forward by
    /// however long it took.
    #[test]
    fn a_stalled_frame_does_not_wind_the_world_clock_past_the_limit() {
        let (device, queue) = test_device();
        let mut scene = clockwork_scene(&device);
        scene.update(&device, &queue, Duration::from_secs(30));
        assert_eq!(scene.elapsed, MAX_STEP);
    }

    /// Renders a real tile pyramid and writes the frame out.
    ///
    /// Ignored because no pyramid is in version control -- one covering a few
    /// kilometres is hundreds of megabytes -- and because this is a look-at-it
    /// check rather than an assertion. Run it with
    /// `FLIGHT_SIM_TERRAIN=/tmp/terrain cargo test --release -- --ignored dump_installed`
    /// and open the PNG it names.
    ///
    /// The `--screenshot` mode renders the same way; this stays because it
    /// reaches knobs the command line deliberately does not expose, all of them
    /// about measuring the clipmap rather than about looking at terrain.
    ///
    /// `FLIGHT_SIM_CAMERA` overrides the opening view, as
    /// `x,y,z,yaw,pitch` -- position in metres from the pyramid's centre, then
    /// two angles in degrees. Without it the scene's own opening camera is
    /// used, which frames the whole extent and therefore looks at whatever is
    /// most of the box. That is the wrong tool for checking one corner of it:
    /// a change confined to ground the default view does not reach renders
    /// byte-identical frames and looks like it did nothing.
    ///
    /// `FLIGHT_SIM_BASE` overrides [`Residency::resident_base`], which is how
    /// the detail a finer base buys is measured against what it costs in
    /// memory and load time. It is a knob here and nowhere else.
    #[test]
    #[ignore = "requires a tile pyramid, which is not in version control"]
    fn dump_installed_terrain() {
        const WIDE: u32 = 960;
        const TALL: u32 = 540;

        // The terrain reports what it chose and what that costs through `log`,
        // which is most of what this test exists to read. Before the device,
        // so that the adapter it picked is reported too.
        let _ = env_logger::Builder::from_env(
            env_logger::Env::default().default_filter_or("warn,flight_sim=info"),
        )
        .try_init();

        let (device, queue) = test_device();
        let size = UVec2::new(WIDE, TALL);

        let started = std::time::Instant::now();
        let root = std::path::PathBuf::from(
            std::env::var("FLIGHT_SIM_TERRAIN")
                .expect("set FLIGHT_SIM_TERRAIN to a directory terrain-process wrote"),
        );
        let mut residency = Residency {
            pixel_angle: crate::terrain::residency::pixel_angle(
                TALL,
                f64::from(crate::camera::FOV_Y_DEGREES).to_radians(),
            ),
            ..Residency::default()
        };
        if let Ok(base) = std::env::var("FLIGHT_SIM_BASE") {
            residency.resident_base = base
                .parse()
                .expect("FLIGHT_SIM_BASE must be a level of the stored pyramid");
        }
        eprintln!("resident from level {}", residency.resident_base);
        let mut scene = Scene::with_residency(
            &device,
            crate::headless::CAPTURE_FORMAT,
            size,
            &root,
            residency,
        )
        .expect("failed to open the terrain pyramid");
        eprintln!("built the scene in {:.2?}", started.elapsed());

        if let Ok(aim) = std::env::var("FLIGHT_SIM_CAMERA") {
            aim.parse::<crate::headless::Placement>()
                .expect("FLIGHT_SIM_CAMERA wants x,y,z,yaw,pitch")
                .apply(&mut scene.camera);
        }

        eprintln!(
            "camera at {} facing {:?}",
            scene.camera.position, scene.camera.orientation
        );
        // Timed separately from the frame below it, because this is where the
        // tile reads and the pyramid reductions happen: a frame that draws
        // quickly can still stall here, and the two want telling apart.
        let started = std::time::Instant::now();
        if let Ok(walk) = std::env::var("FLIGHT_SIM_WALK") {
            let steps: u32 = walk.parse().expect("FLIGHT_SIM_WALK wants a count");
            let home = scene.camera.position;
            scene.camera.position = home - Vec3::new(steps as f32, 0.0, steps as f32);
            for _ in 0..steps {
                scene.camera.position += Vec3::new(1.0, 0.0, 1.0);
                scene.update(&device, &queue, crate::headless::STEP);
            }
            scene.camera.position = home;
        }
        scene.settle(&device, &queue);
        eprintln!(
            "filled every level in {:.2?}, finest level {}",
            started.elapsed(),
            scene.terrain.base_level()
        );

        let started = std::time::Instant::now();
        let pixels = crate::headless::capture(
            &device,
            &queue,
            &mut scene,
            size,
            crate::headless::Flight {
                frames: 1,
                speed: 0.0,
            },
        )
        .expect("failed to read the frame back");
        eprintln!("rendered one frame in {:.2?}", started.elapsed());

        let path = std::env::temp_dir().join("terrain.png");
        crate::headless::write_png(&path, size, &pixels).expect("failed to write the preview");
        eprintln!("wrote {}", path.display());
    }

    /// A flat shelf rising out of flat ground, east of the middle.
    ///
    /// Deliberately lower than any cloud base: a ray that stops on it has not
    /// climbed into the deck, so what it met is nothing at all rather than a
    /// little. That is what makes the comparison below a clean one.
    fn shelf(height: f32) -> (Vec<f32>, Vec<MaterialId>) {
        let heights = (0..RASTER * RASTER)
            .map(|i| if i % RASTER >= 96 { height } else { 0.0 })
            .collect();
        (heights, flat_ground())
    }

    /// West of the shelf, below its top, looking east along the flat.
    ///
    /// Rolled, so the shelf's edge runs diagonally across the frame. Level, it
    /// projects to an exactly horizontal line that lands between two rows of
    /// blocks, and no block straddles it at all -- which is the one case both
    /// tests below are most about.
    fn against_the_shelf(camera: &mut Camera) {
        camera.position = Vec3::new(world_of(4.0, 64.0).x, 200.0, world_of(4.0, 64.0).z);
        camera.orientation =
            Camera::from_yaw_pitch_roll(90f32.to_radians(), 0.0, 8f32.to_radians());
    }

    /// Ground in front of cloud hides it, exactly.
    ///
    /// The march clips its ray at the G-buffer's own depth, which is the whole
    /// of how terrain occludes cloud: there is no cloud shadow volume, no depth
    /// test in the composite, and no second traversal -- the ray simply stops
    /// where the ground the geometry pass already found is. That makes the
    /// occlusion exact, and it makes it free, and this is what says it happens.
    ///
    /// A shelf 600 m high, which is under the lowest a deck's base can sag,
    /// against an overcast sky that covers nearly all of it. A ray that meets
    /// the shelf has climbed nowhere near the cloud, so it must come back with
    /// none; a ray that clears the skyline runs for tens of kilometres inside a
    /// solid deck and must come back with almost nothing else.
    ///
    /// Measured over whole two-by-two blocks, because that is the unit the march
    /// works in: a block straddling the skyline takes the reach of its farthest
    /// pixel, which is the ray running on past the ridge, and it is *meant* to
    /// -- erring the other way would cut a notch of missing cloud along every
    /// silhouette in the frame.
    #[test]
    fn ground_in_front_of_cloud_hides_the_cloud_behind_it() {
        let (heights, materials) = shelf(600.0);
        let frame = render_under(
            heights,
            materials,
            against_the_shelf,
            crate::cloud::Preset::Overcast,
        );

        let mut against = Vec::new();
        let mut open = Vec::new();
        let mut skyline = Vec::new();
        for block_y in 0..SIZE / 2 {
            for block_x in 0..SIZE / 2 {
                let corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .map(|(i, j)| (block_x * 2 + i, block_y * 2 + j));
                let cloud = frame.cloud(block_x * 2, block_y * 2);
                if corners.iter().all(|&(x, y)| !frame.sky(x, y)) {
                    against.push(cloud);
                } else if corners.iter().all(|&(x, y)| frame.sky(x, y)) {
                    open.push(cloud);
                } else {
                    skyline.push(cloud);
                }
            }
        }
        assert!(
            against.len() > 1000 && open.len() > 1000,
            "the shelf left {} blocks of ground and {} of sky, which is not a skyline",
            against.len(),
            open.len()
        );

        let worst = against.iter().copied().fold(0.0f32, f32::max);
        assert_eq!(
            worst, 0.0,
            "a ray stopped by ground 600 m up still found {worst} of cloud"
        );
        let least = open.iter().copied().fold(1.0f32, f32::min);
        assert!(
            least > 0.9,
            "a ray over the skyline under an overcast sky found only {least} of cloud"
        );
        // And the blocks the skyline runs through go with the sky rather than
        // with the ground, which is the other half of taking the farthest of a
        // block's four depths. Taking the nearest instead would leave these
        // empty, and a row of empty blocks along a ridge is a notch bitten out
        // of the cloud behind it.
        let straddling = skyline.iter().copied().fold(1.0f32, f32::min);
        assert!(
            !skyline.is_empty() && straddling > 0.9,
            "{} blocks straddle the skyline and the emptiest found {straddling} of cloud",
            skyline.len()
        );
    }

    /// Cloud gathers on the windward side of a ridge and thins in its lee.
    ///
    /// The föhn, and the whole reason the wind was solved around the actual
    /// mountains rather than blown across them as a constant. Air pushed up a
    /// windward slope cools and makes cloud; air coming back down the other
    /// side warms and unmakes it. The bake carries how far the parcel arriving
    /// at each cell has climbed in the last minute and a half, and this is that
    /// number reaching a pixel.
    ///
    /// The one test in the tree that solves the wind rather than skipping it,
    /// and it does so twice -- once with the wind from the west and once from
    /// the east, over the same ridge under the same sky. Comparing the two
    /// halves of one frame against each other would be comparing two different
    /// pieces of noise; comparing the *same* half of two frames leaves the
    /// wind's direction as the only thing that differs.
    #[test]
    fn cloud_gathers_on_the_windward_side_of_a_ridge() {
        // A ridge running north-south, so a wind along x crosses it square. Its
        // crest reaches into the lowest a deck's base can sag, which is what
        // gives the air something to climb.
        let crest = RASTER as f32 * 0.5;
        let heights = (0..RASTER * RASTER)
            .map(|i| {
                let across = (i % RASTER) as f32;
                1400.0 * (1.0 - ((across - crest) / 16.0).abs()).max(0.0)
            })
            .collect::<Vec<f32>>();

        let over_at = |speed, from_degrees| {
            render_over(
                test_residency(),
                heights.clone(),
                flat_ground(),
                |camera: &mut Camera| {
                    // Well above every deck, looking straight down, so a pixel
                    // is a column of sky over a known patch of ground.
                    camera.position = Vec3::new(0.0, 11_000.0, 0.0);
                    camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
                },
                &[],
                crate::sky::Sun::default(),
                // Two hundred metres a texel rather than thirty, so the raster
                // is twenty-five kilometres across instead of four. The baked
                // wind covers the raster and nothing outside it, and a camera
                // high enough to see a whole deck from above sees far more
                // ground than four kilometres -- over the small world most of
                // the frame is outside the grid, faded to no wind at all, and
                // the föhn is a detail in the middle of a still sky.
                Georeferencing::square(RASTER, RASTER, 200.0),
                crate::cloud::Preset::Broken,
                Some(crate::air::Wind {
                    speed,
                    from_degrees,
                }),
            )
        };
        // Three frames: one with the air standing still, and one with it
        // crossing the ridge each way. The still frame is the baseline the
        // other two are measured against, because the weather field has a
        // lopsidedness of its own that has nothing to do with any wind -- it is
        // noise, and it does not know where the ridge is. Solved rather than
        // skipped, so all three take the same path through the same code.
        let still = over_at(0.0, 270.0);
        let westerly = over_at(18.0, 270.0);
        let easterly = over_at(18.0, 90.0);

        // How much more cloud one side of the crest holds than the other. The
        // ridge is centred on the raster and the camera over its middle, so the
        // crest runs down the middle of the frame; which half is west does not
        // have to be worked out, because the test is that *reversing the wind
        // reverses the sign*. Nothing else in the two frames differs.
        let lopsided = |frame: &Frame| {
            let side = |columns: std::ops::Range<u32>| {
                let mut sum = 0.0f64;
                let mut count = 0.0f64;
                for y in 0..SIZE {
                    for x in columns.clone() {
                        sum += f64::from(frame.cloud(x, y));
                        count += 1.0;
                    }
                }
                sum / count
            };
            side(0..SIZE / 2) - side(SIZE / 2..SIZE)
        };
        let calm = lopsided(&still);
        let (west, east) = (lopsided(&westerly) - calm, lopsided(&easterly) - calm);
        assert!(
            west * east < 0.0,
            "reversing the wind moved the cloud the same way across the ridge: \
             {west:.4} under a westerly and {east:.4} under an easterly, \
             against {calm:.4} in still air"
        );
        // ... and by enough to be the wind rather than the noise.
        assert!(
            west.abs() > 0.05 && east.abs() > 0.05,
            "the wind moved the cloud across the ridge by {west:.4} one way and \
             {east:.4} the other, which is no föhn at all"
        );
    }

    /// An overcast sky darkens the ground it stands over.
    ///
    /// The whole of the user's ask that the design has not yet answered:
    /// clouds cast shadows on the terrain. They do it out of the same volume
    /// the cloud march lights itself from, read at the ground point instead --
    /// there is no shadow map, no second traversal, and nothing that had to be
    /// built for the ground's sake alone.
    ///
    /// Flat ground seen from above, so every pixel is the same slope under the
    /// same sun and the only thing that differs between the two frames is what
    /// is overhead. Both halves of the dimming are checked, and separately: the
    /// direct sun, which an overcast deck takes away almost entirely, and the
    /// ambient, which it must take down as well. Leaving the sky at full
    /// strength gives an overcast landscape that is bright and flat rather than
    /// dim, which reads less like weather than like the sun being switched off.
    #[test]
    fn an_overcast_sky_darkens_the_ground_it_stands_over() {
        // Below the lowest a deck's base can sag, looking down. That matters:
        // from above the deck the ray to the ground crosses the cloud, and what
        // the frame then shows is the cloud in front of the ground rather than
        // the shadow on it -- which is a different thing and would pass this
        // test for the wrong reason. From underneath there is nothing between
        // the eye and the ground but air.
        let under = |weather| {
            render_under(
                vec![0.0; (RASTER * RASTER) as usize],
                flat_ground(),
                |camera: &mut Camera| {
                    camera.position = Vec3::new(0.0, 400.0, 0.0);
                    camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
                },
                weather,
            )
        };
        let clear = under(crate::cloud::Preset::Clear);
        let cloudy = under(crate::cloud::Preset::Overcast);

        let ground = |frame: &Frame| {
            let mut lit = 0f64;
            let mut count = 0f64;
            for y in 0..SIZE {
                for x in 0..SIZE {
                    if frame.sky(x, y) {
                        continue;
                    }
                    let p = frame.pixel(x, y);
                    lit += f64::from(p[0]) + f64::from(p[1]) + f64::from(p[2]);
                    count += 1.0;
                }
            }
            (lit / count.max(1.0), count)
        };
        let (open, seen) = ground(&clear);
        let (shaded, also) = ground(&cloudy);
        assert!(
            seen > 40_000.0 && also == seen,
            "the two frames disagree about where the ground is: {seen} against {also}"
        );
        let left = shaded / open;
        // Measured at 0.34. Taking only the direct sun away and leaving the sky
        // at full strength measures 0.51; dimming the sky the whole way instead
        // of leaving `BOUNCED` of it measures 0.12; leaving the sun alone and
        // dimming only the sky measures 0.95. The band below admits none of
        // them.
        assert!(
            left < 0.45,
            "an overcast sky left the ground at {left:.2} of its clear-sky \
             brightness, which is not overcast"
        );
        assert!(
            left > 0.20,
            "an overcast sky left the ground at {left:.2}, which is not a dull \
             day but a moonless night"
        );
    }

    /// A cloud in front of the sun hides the sun.
    ///
    /// Nothing in the composite mentions the sun's disc. It is hidden because
    /// the cloud's transmittance multiplies the *whole* of what was behind it,
    /// and the disc is part of that -- which is the same line that makes a cloud
    /// redden at sunset and that will make one dim a star. Getting this for
    /// nothing is the argument for compositing over a finished background
    /// rather than adding cloud into the sky as another term.
    ///
    /// The disc is the only thing in the frame that clips to white, so it can be
    /// found rather than predicted: the brightest pixel of the clear frame is
    /// it, and the assertion is about that same pixel under cloud.
    #[test]
    fn a_cloud_in_front_of_the_sun_hides_the_sun() {
        // Flat, so the whole upper frame is sky and the sun has somewhere to
        // be. Looking east at the sun's own bearing, tilted up to meet it.
        let aim = |camera: &mut Camera| {
            camera.position = Vec3::new(0.0, 300.0, 0.0);
            camera.orientation =
                Camera::from_yaw_pitch_roll(90f32.to_radians(), 25f32.to_radians(), 0.0);
        };
        let frame = |weather| {
            render_over(
                test_residency(),
                vec![0.0; (RASTER * RASTER) as usize],
                flat_ground(),
                aim,
                &[],
                crate::sky::Sun::from_angles(25.0, 90.0),
                placement(),
                weather,
                None,
            )
        };
        let clear = frame(crate::cloud::Preset::Clear);
        let cloudy = frame(crate::cloud::Preset::Overcast);

        let brightness = |pixels: &Frame, x, y| {
            let p = pixels.pixel(x, y);
            u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])
        };
        let (at, disc) = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .map(|(x, y)| ((x, y), brightness(&clear, x, y)))
            .max_by_key(|&(_, lit)| lit)
            .expect("an empty frame");
        assert!(
            disc > 700,
            "the brightest pixel of a clear sky is only {disc}, which is no sun"
        );
        assert!(
            clear.sky(at.0, at.1),
            "the brightest pixel of the frame is ground, not the sun"
        );

        // What says the sun is hidden is not that the pixel went dark -- a
        // sunlit overcast deck is bright -- but that it stopped standing out
        // from what is around it. The disc is a fraction of a degree across and
        // the sky beside it is not, so the ratio between them is the measurement
        // that survives the cloud being bright in its own right.
        let ring = |frame: &Frame| {
            let mut lit = 0u32;
            let mut count = 0u32;
            for step in 0..16i32 {
                let angle = step as f32 * std::f32::consts::TAU / 16.0;
                let x = at.0 as f32 + 40.0 * angle.cos();
                let y = at.1 as f32 + 40.0 * angle.sin();
                if (0.0..SIZE as f32).contains(&x) && (0.0..SIZE as f32).contains(&y) {
                    lit += brightness(frame, x as u32, y as u32);
                    count += 1;
                }
            }
            f64::from(lit) / f64::from(count.max(1))
        };
        let stands_out =
            |frame: &Frame| f64::from(brightness(frame, at.0, at.1)) / ring(frame).max(1.0);
        // Measured: 2.26 against a clear sky, 1.06 against an overcast one --
        // the disc goes from twice its surroundings to indistinguishable from
        // them. It is worth saying what the second number is *not*: the pixel
        // did not go dark, it went from 255 a channel to 114, because what is
        // there now is a sunlit cloud and a sunlit cloud is bright.
        let (open, hidden) = (stands_out(&clear), stands_out(&cloudy));
        assert!(open > 1.8, "a clear sky's sun stands out by only {open:.2}");
        assert!(
            hidden < 1.25,
            "an overcast sky left the sun standing out by {hidden:.2}"
        );
    }

    /// Cloud behind a ridge does not leak across it into the ridge.
    ///
    /// The half-resolution march is one ray per two-by-two block, and a block
    /// the skyline runs through marched *past* the shelf -- that is deliberate,
    /// and it is what stops a notch of missing cloud appearing along every
    /// silhouette. The price is that a plain bilinear upsample would then paint
    /// that cloud onto the pixels the shelf covers, which is the halo every
    /// half-resolution volumetric has to answer for.
    ///
    /// The answer is the bilateral weighting in `cloud_at`: a tap is worth the
    /// reciprocal of how far its block reached from what this pixel is showing,
    /// so a tap that ran to the horizon is worth a hundredth to a pixel two
    /// kilometres away. This measures the leak.
    ///
    /// Two frames of one scene, differing only in the preset. Under a clear sky
    /// the composite is the identity, so the clear frame *is* the frame this
    /// pass drew before there were clouds in it -- which makes the comparison an
    /// exact one rather than a judgement.
    #[test]
    fn cloud_behind_a_ridge_does_not_leak_onto_the_ridge() {
        let (heights, materials) = shelf(600.0);
        let clear = render_under(
            heights.clone(),
            materials.clone(),
            against_the_shelf,
            crate::cloud::Preset::Clear,
        );
        let cloudy = render_under(
            heights,
            materials,
            against_the_shelf,
            crate::cloud::Preset::Overcast,
        );

        // How much *brighter* the cloudy frame is, which is the whole of what
        // separates a halo from a shadow. Ground under an overcast sky is
        // legitimately darker -- that is the point of the commit that put the
        // shadows there -- so an unsigned difference no longer says anything.
        // Cloud leaking across a silhouette can only add light: it is white
        // scatter composited over dark forest.
        let brighter = |x, y| {
            let (a, b) = (clear.pixel(x, y), cloudy.pixel(x, y));
            (0..3).map(|c| b[c].saturating_sub(a[c])).max().unwrap_or(0)
        };
        let apart = |x, y| {
            let (a, b) = (clear.pixel(x, y), cloudy.pixel(x, y));
            (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap_or(0)
        };
        let mut ground = (0u8, 0usize, 0usize);
        let mut sky = (255u8, 0usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                if clear.sky(x, y) {
                    sky.0 = sky.0.min(apart(x, y));
                    sky.1 += 1;
                } else {
                    let leak = brighter(x, y);
                    ground.0 = ground.0.max(leak);
                    ground.1 += usize::from(leak > 0);
                    ground.2 += 1;
                }
            }
        }

        // Ground below the skyline is showing rock that no cloud stands in
        // front of, so no cloud may be added to it. It is allowed to go darker
        // -- a shelf under an overcast sky is in shadow -- but not brighter.
        //
        // Measured: not one count, on any pixel. Dropping the weighting for a
        // plain bilinear measures 41 counts over one per cent of the ground --
        // a white fringe running the whole ridge -- so the bound below has an
        // order of magnitude of room and still nothing to spare on the far
        // side, which is what a bound on a leak should look like.
        assert!(
            ground.0 <= 15,
            "cloud behind the shelf put {} counts of light onto ground in \
             front of it",
            ground.0
        );
        // How *far* the leak reaches is not what the weighting changes, and the
        // number is here to say so rather than to discriminate: both upsamples
        // leave the same fringe, a pixel or two wide, and what the weighting
        // decides is how strong it is.
        let leaked = ground.1 as f64 / ground.2 as f64;
        assert!(
            leaked < 0.03,
            "{leaked:.3} of the ground brightened when cloud was put behind \
             it, which is a band and not a fringe"
        );
        // ... and the sky above the skyline changed everywhere, which is what
        // says the frames differ at all and that the comparison has teeth.
        assert!(
            sky.1 > 1000 && sky.0 > 10,
            "{} sky pixels, the least changed by {} counts",
            sky.1,
            sky.0
        );
    }

    /// Rough terrain, so that neighbouring clipmap levels genuinely disagree
    /// about where the surface is and any seam between them would show.
    fn rugged() -> (Vec<f32>, Vec<MaterialId>) {
        let heights = (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as f32, (i / RASTER) as f32);
                300.0 * ((x * 0.31).sin() + (y * 0.27).cos()) + 150.0 * (x * 0.11 - y * 0.09).sin()
            })
            .collect();
        (heights, flat_ground())
    }

    /// As [`rugged`], but painted rather than uniformly green.
    ///
    /// Looking straight down at flat colour, geometry is nearly invisible: the
    /// frame is the same green wherever the surface happens to be. A test that
    /// means to see which level drew a patch of ground needs the ground to look
    /// different from place to place, so that both the shape and the texel it is
    /// coloured from show up in the pixels.
    #[test]
    fn the_terrain_stops_at_the_edge_of_the_data() {
        // Resident squares deliberately reach past the raster so there is
        // always a level coarse enough to cover the horizon. Out there every read clamps
        // to the border texel, which would otherwise draw the edge row smeared
        // outwards as a plateau indistinguishable from real ground.
        //
        // Flat ground, so that a point's screen position depends only on where
        // it is: over rough terrain a tall peak inside the raster projects onto
        // the same pixel as a spot outside it, and the two cannot be told apart.
        let mut camera = None;
        let frame = render(vec![0.0; (RASTER * RASTER) as usize], flat_ground(), |c| {
            // High enough that the raster's edge sits well inside the frame.
            c.position = Vec3::new(0.0, 6000.0, 0.0);
            c.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
            camera = Some(*c);
        });
        let camera = camera.expect("camera captured");
        let at = |world: Vec3| {
            let (x, y) = to_pixels(camera.view_projection(), world, SIZE, SIZE);
            (x.round() as u32, y.round() as u32)
        };

        let ((min_x, min_z), (max_x, max_z)) = placement().data_bounds();
        let (min_x, min_z) = (min_x as f32, min_z as f32);
        let (max_x, max_z) = (max_x as f32, max_z as f32);

        // Sampled at ground level, where a point's screen position does not
        // depend on the terrain's own height.
        for corner in [
            Vec3::new(min_x, 0.0, min_z),
            Vec3::new(max_x, 0.0, min_z),
            Vec3::new(min_x, 0.0, max_z),
            Vec3::new(max_x, 0.0, max_z),
        ] {
            let outside = corner + Vec3::new(corner.x.signum(), 0.0, corner.z.signum()) * 150.0;
            let (x, y) = at(outside);
            assert!(
                frame.sky(x, y),
                "{outside} lies beyond the raster but was drawn as terrain: {:?}",
                frame.pixel(x, y)
            );
        }

        // ... and the data itself is still drawn right up to its edge, so this
        // has not simply clipped the terrain away.
        let inside = Vec3::new(max_x - 150.0, 0.0, max_z - 150.0);
        let (x, y) = at(inside);
        assert!(
            !frame.sky(x, y),
            "{inside} is inside the raster but was cut away: {:?}",
            frame.pixel(x, y)
        );
    }

    /// Asserts that no one step between consecutive frames changed the picture
    /// far more than its neighbours did, which is what a pop looks like.
    ///
    /// The frames are expected to come from a camera moving steadily, so the
    /// change between any two of them is roughly the same. A level snapping into
    /// place, or vanishing, shows up as a single outlier.
    ///
    /// `tolerance` is how many times the typical step the worst one is allowed
    /// to be. How tight it can be depends on how evenly the sweep changes the
    /// frame to begin with: a camera flying along sees the picture turn over
    /// steadily and needs room, one climbing straight up mostly zooms and can be
    /// held to much less.
    /// Looks straight down from `altitude` over the same spot every time.
    fn from_altitude(altitude: f32) -> impl FnOnce(&mut Camera) {
        move |camera: &mut Camera| {
            camera.position = Vec3::new(70.0, altitude, -110.0);
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, -90f32.to_radians(), 0.0);
        }
    }

    #[test]
    fn climbing_away_from_the_ground_gives_up_the_finest_levels() {
        // Levels are chosen by how far the ground they cover is from the camera,
        // and a camera in the air is far from the ground directly below it as
        // well as from the horizon. Drawing the finest level from high up spends
        // full-resolution triangles on ground that covers a fraction of a pixel,
        // and a fine window's worth of tile reads on fetching it.
        let (heights, materials) = rugged();

        let low = render_after(
            heights.clone(),
            materials.clone(),
            from_altitude(900.0),
            &[],
        )
        .base_level;
        let frame = render_after(heights, materials, from_altitude(4000.0), &[]);
        let high = frame.base_level;

        assert_eq!(low, 0, "close to the ground every level is worth drawing");
        assert!(
            high > low,
            "climbing should have given up at least one level, still at {high}"
        );

        // ... and what the dropped levels used to draw is still drawn, by the
        // level that took over. The middle of the frame is well inside the
        // raster at this height; its edges are not, and the sky past the data is
        // honest there.
        let holes: Vec<(u32, u32)> = (SIZE / 4..SIZE * 3 / 4)
            .flat_map(|y| (SIZE / 4..SIZE * 3 / 4).map(move |x| (x, y)))
            .filter(|&(x, y)| frame.sky(x, y))
            .collect();
        assert!(
            holes.is_empty(),
            "dropping the finest levels left {} pixels of sky, first at {:?}",
            holes.len(),
            holes.first()
        );
    }

    /// A raster source that notes which levels are read from it.
    struct Counted {
        inner: Box<dyn RasterSource>,
        levels: std::rc::Rc<std::cell::RefCell<Vec<u32>>>,
    }

    impl RasterSource for Counted {
        fn level_count(&self) -> u32 {
            self.inner.level_count()
        }

        fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]) {
            self.levels.borrow_mut().push(level);
            self.inner.read_rect(level, origin, size, out);
        }
    }

    #[test]
    fn the_chain_is_read_once_and_the_camera_never_reads_again() {
        // This is the whole of what holding the raster resident buys, and
        // nothing else measures it. A window that followed the camera read a
        // strip of every level for every tile crossed, and at altitude the
        // camera crosses them fast; the chain is read once and then the camera
        // is free. What survives of the old rule is that a level too fine to
        // resolve is still not *descended* to -- it costs march steps rather
        // than tile reads now, but it still costs.
        let (device, queue) = test_device();
        let (heights, materials) = rugged();
        let reads = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));

        let mut scene = Scene::from_terrain(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            UVec2::splat(SIZE),
            |camera_layout, storage_layout, work_layout, args_layout, risk_layout, reach_layout| {
                Terrain::new(
                    &device,
                    camera_layout,
                    storage_layout,
                    work_layout,
                    args_layout,
                    risk_layout,
                    reach_layout,
                    test_residency(),
                    UVec2::splat(SIZE),
                    placement(),
                    Sources {
                        heights: Box::new(Counted {
                            inner: Box::new(Pyramid::build(Level::new(
                                RASTER,
                                RASTER,
                                heights.clone(),
                            ))),
                            levels: reads.clone(),
                        }),
                        materials: Box::new(Pyramid::build(Level::new(RASTER, RASTER, materials))),
                    },
                )
            },
        );

        let mut read_levels = |at: Vec3| {
            reads.borrow_mut().clear();
            scene.camera.position = at;
            scene.update(&device, &queue, crate::headless::STEP);
            let seen: std::collections::HashSet<u32> = reads.borrow().iter().copied().collect();
            (seen, scene.terrain.base_level())
        };

        // The first update is the load, and it reads every level exactly once.
        let (loaded, base) = read_levels(Vec3::new(70.0, 4000.0, -110.0));
        assert!(base > 0, "the sweep needs an altitude that drops a level");
        assert_eq!(
            loaded.iter().copied().min(),
            Some(0),
            "the load has to read the finest level whatever the camera is doing"
        );

        // Every update after it reads nothing at all, however far the camera
        // has moved and whichever levels it has given up or taken back.
        for at in [
            Vec3::new(70.0, 900.0, -110.0),
            Vec3::new(-900.0, 900.0, 900.0),
            Vec3::new(900.0, 4000.0, -900.0),
        ] {
            let (read, _) = read_levels(at);
            assert!(read.is_empty(), "flying to {at} read levels {read:?}");
        }

        // ... and the descent floor still follows the altitude, which is what
        // stops a ray walking levels no pixel can resolve.
        assert_eq!(read_levels(Vec3::new(70.0, 900.0, -110.0)).1, 0);
        assert!(read_levels(Vec3::new(70.0, 4000.0, -110.0)).1 > 0);
    }

    #[test]
    fn the_camera_opens_above_the_terrain_looking_at_all_of_it() {
        let extent = Vec2::new(4000.0, 9000.0);
        let camera = Camera::overlooking(extent, 2500.0, 16.0 / 9.0);

        assert!(
            camera.position.y > 2500.0,
            "the viewpoint must clear the highest ground, got {}",
            camera.position.y
        );

        // Both far corners of the terrain fall inside the frustum.
        for corner in [
            Vec3::new(-extent.x * 0.5, 0.0, -extent.y * 0.5),
            Vec3::new(extent.x * 0.5, 0.0, -extent.y * 0.5),
        ] {
            let clip = camera.view_projection() * corner.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            assert!(
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc.z),
                "{corner} projects outside the view at {ndc}"
            );
        }

        // ... and the middle of the view lands on the terrain rather than
        // beyond it. Having the corners in shot is not enough on its own: they
        // can sit along the very bottom edge with the rest of the frame sky,
        // which is what a pitch that does not follow the extent produces.
        let forward = camera.orientation * Vec3::NEG_Z;
        assert!(forward.y < 0.0, "the view must slope downwards");
        let ground = camera.position + forward * (-camera.position.y / forward.y);
        assert!(
            (-extent.y * 0.5..=extent.y * 0.5).contains(&ground.z),
            "the centre of the view meets the ground at z {}, outside the \
             terrain's {}..{}",
            ground.z,
            -extent.y * 0.5,
            extent.y * 0.5
        );
    }

    /// The streaming spans are the only account there is of the tile reads --
    /// no GPU timestamp can reach them -- so a regression that stopped filling
    /// them would leave a profiled run reporting a frame that cost nothing.
    #[test]
    fn profiling_accounts_for_the_load_and_for_nothing_after_it() {
        use std::time::Duration;

        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let heights = vec![0.0; (RASTER * RASTER) as usize];
        let mut frame = crate::profile::Frame::default();

        // Off is the default, and off keeps nothing at all rather than keeping
        // a total nobody asked for.
        let mut quiet = test_scene(
            &device,
            format,
            test_residency(),
            heights.clone(),
            flat_ground(),
        );
        quiet.update(&device, &queue, crate::headless::STEP);
        quiet.record(&mut frame);
        assert_eq!(frame.cpu.terrain, crate::profile::Terrain::default());

        // On, and the first update is the one that reads the chain in, so it
        // has plenty to report.
        let mut watched = test_scene(&device, format, test_residency(), heights, flat_ground());
        watched.profile(&device, true);
        watched.update(&device, &queue, crate::headless::STEP);
        watched.record(&mut frame);
        let load = frame.cpu.terrain;
        assert!(load.read > Duration::ZERO, "{load:?}");
        assert!(load.write > Duration::ZERO, "{load:?}");

        // Every frame after it reads nothing, which is the point of holding the
        // chain rather than streaming it -- and the reason these rows are worth
        // keeping is that they are how anyone would notice it had come back.
        watched.camera.position = Vec3::new(400.0, 900.0, -400.0);
        watched.update(&device, &queue, crate::headless::STEP);
        watched.record(&mut frame);
        let flying = frame.cpu.terrain;
        assert_eq!(flying.read, Duration::ZERO, "{flying:?}");
        assert_eq!(flying.convert, Duration::ZERO, "{flying:?}");
    }

    /// Draws one frame of `scene` and reads back how its pixels were settled.
    ///
    /// Its own submit per call, not several passes on one encoder:
    /// `queue.write_buffer` is ordered at submit, so consecutive frames batched
    /// into one would both run against the *later* uniforms -- the same dither
    /// phase and the same previous-camera basis -- and stop being consecutive
    /// frames. The same reason `crate::headless::capture` loops this way.
    fn settled_pixels(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scene: &Scene,
        view: &wgpu::TextureView,
    ) -> crate::reproject::Coverage {
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("coverage readback"),
            size: crate::reproject::Coverage::BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let profiler = crate::profile::profiler(device, false);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut gpu = profiler.scope("gpu", &mut encoder);
            scene.draw(&mut gpu, view);
        }
        encoder.copy_buffer_to_buffer(
            scene.tally(),
            0,
            &readback,
            0,
            crate::reproject::Coverage::BYTES,
        );
        queue.submit(std::iter::once(encoder.finish()));

        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let coverage = crate::reproject::Coverage::from_bytes(
            &readback.get_mapped_range(..).expect("buffer not mapped"),
        );
        readback.unmap();
        coverage
    }

    /// The invariant that replaced clearing the G-buffer: between them
    /// `cs_compact` and `cs_march` write every pixel exactly once.
    ///
    /// Nothing clears the G-buffer any more, so a pixel that fell down no path
    /// would not be blank -- it would hold whatever the frame before left
    /// there, which on a slow-moving camera looks entirely plausible. The three
    /// counters are what make that checkable at all.
    #[test]
    fn every_pixel_is_settled_by_exactly_one_path() {
        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut scene = test_scene(
            &device,
            format,
            test_residency(),
            vec![0.0; (RASTER * RASTER) as usize],
            flat_ground(),
        );

        // Well above the ground, so the ceiling test can settle a ray that
        // heads upwards, and tilted by less than half the field of view, so
        // some rays do and the rest are left with ground to find. All three
        // paths then have pixels to count, which is what makes the sum worth
        // asserting: a counter that was never incremented would otherwise hide
        // behind a path this view happened not to take.
        scene.camera.position = Vec3::new(0.0, 3000.0, 0.0);
        scene.camera.orientation = Camera::from_yaw_pitch_roll(0.0, -20f32.to_radians(), 0.0);
        scene.settle(&device, &queue);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("coverage target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let pixels = SIZE * SIZE;

        // The first frame has no history at all -- the G-buffer it reprojects
        // from has never been written -- so every point is dropped and nothing
        // is carried.
        let first = settled_pixels(&device, &queue, &scene, &view);
        assert_eq!(first.total(), pixels, "{first:?}");
        assert_eq!(first.reprojected, 0, "{first:?}");

        scene.update(&device, &queue, crate::headless::STEP);

        // The second is the first that can take all three paths.
        let second = settled_pixels(&device, &queue, &scene, &view);
        assert_eq!(second.total(), pixels, "{second:?}");
        assert!(second.reprojected > 0, "{second:?}");
        assert!(second.sky > 0, "{second:?}");
        // The dither hands a share of the screen back however still the camera
        // is, so there is always something left to march.
        assert!(second.marched > 0, "{second:?}");
    }

    /// Sky the march never established is not carried into the next frame.
    ///
    /// A camera inside the terrain finds no ground down any ray: where each one
    /// entered the surface is behind the eye, so the march gives up rather than
    /// reporting a hit, and the frame comes out sky throughout. That is an
    /// answer about where the camera is standing, not about the world, and it
    /// stops being true the moment the camera climbs out.
    ///
    /// Carrying it across holds the frame open long after. A carried sky point
    /// is placed far enough away to ignore where the eye has moved to -- right
    /// for sky, exactly wrong for this -- so it lands back on the pixel it came
    /// from; and with no ground anywhere in the history to splat over it, the
    /// hole survives until the dither happens to drop each cell, which takes up
    /// to sixteen frames. Flying a real raster, dipping under the surface and
    /// climbing straight back out left 13% of the frame reading sky over ground
    /// plainly in view, and it took eight frames to close.
    ///
    /// Looking straight down at flat ground, every pixel is ground, so any sky
    /// in the second frame is the defect and nothing else.
    #[test]
    fn climbing_out_of_the_ground_does_not_carry_its_sky() {
        const GROUND: f32 = 500.0;
        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut scene = test_scene(
            &device,
            format,
            test_residency(),
            vec![GROUND; (RASTER * RASTER) as usize],
            flat_ground(),
        );
        straight_down(&mut scene.camera);
        scene.settle(&device, &queue);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let bytes_per_row = SIZE * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let profiler = crate::profile::profiler(&device, false);

        // Buried, a hundred metres under the surface. Its own submit, because
        // `queue.write_buffer` is ordered at submit and the two frames have to
        // run against their own cameras -- the same reason
        // `crate::headless::capture` loops that way.
        scene.camera.position.y = GROUND - 100.0;
        scene.update(&device, &queue, crate::headless::STEP);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut gpu = profiler.scope("gpu", &mut encoder);
            scene.draw(&mut gpu, &view);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // Back out above it, where every ray meets the ground again.
        scene.camera.position.y = 3000.0;
        scene.update(&device, &queue, crate::headless::STEP);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut gpu = profiler.scope("gpu", &mut encoder);
            scene.draw(&mut gpu, &view);
        }
        // The depth channel rather than the picture: "no ray came back empty"
        // is a fact about what the march wrote, and zero depth is exactly that.
        encoder.copy_texture_to_buffer(
            scene.gbuffer.targets.depth.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(SIZE),
                },
            },
            wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");
        let depths = bytemuck::cast_slice::<u8, f32>(
            &readback.get_mapped_range(..).expect("buffer not mapped"),
        )
        .to_vec();

        let sky = depths.iter().filter(|&&depth| depth == 0.0).count();
        assert_eq!(
            sky,
            0,
            "climbing out of the ground left {sky} of {} pixels showing the sky \
             it saw from under the surface",
            SIZE * SIZE
        );
    }

    /// A target to draw single frames into, and the readback behind it.
    ///
    /// Shared by the two flights below, which both need to step a camera a
    /// frame at a time and look at what came out rather than at a coverage
    /// count.
    struct Offscreen {
        target: wgpu::Texture,
        view: wgpu::TextureView,
        readback: wgpu::Buffer,
        /// The G-buffer's depth and material channels, alongside the picture.
        ///
        /// Both flights below count sky and one of them counts a material, and
        /// neither is a fact about what colour came out. See [`Frame`].
        depth_readback: wgpu::Buffer,
        material_readback: wgpu::Buffer,
        profiler: wgpu_profiler::GpuProfiler,
    }

    impl Offscreen {
        fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
            let target = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("offscreen target"),
                size: wgpu::Extent3d {
                    width: SIZE,
                    height: SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let staging = |label| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: u64::from(SIZE * 4 * SIZE),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                })
            };
            Self {
                readback: staging("readback"),
                depth_readback: staging("depth readback"),
                material_readback: staging("material readback"),
                profiler: crate::profile::profiler(device, false),
                target,
                view,
            }
        }

        /// Moves the camera and draws one frame, reading it back if asked.
        ///
        /// One frame per submit, because `queue.write_buffer` is ordered at
        /// submit and each frame has to run against its own camera -- the same
        /// reason `crate::headless::capture` loops that way.
        ///
        /// [`None`] when `read` is false: a frame drawn only to advance the
        /// history has nothing to look at, and an empty [`Frame`] would be a
        /// trap for whoever indexed it.
        fn step(
            &self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            scene: &mut Scene,
            at: Vec3,
            read: bool,
        ) -> Option<Frame> {
            scene.camera.position = at;
            scene.update(device, queue, crate::headless::STEP);
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut gpu = self.profiler.scope("gpu", &mut encoder);
                scene.draw(&mut gpu, &self.view);
            }
            if read {
                let mut copy = |source: wgpu::TexelCopyTextureInfo, into: &wgpu::Buffer| {
                    encoder.copy_texture_to_buffer(
                        source,
                        wgpu::TexelCopyBufferInfo {
                            buffer: into,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset: 0,
                                bytes_per_row: Some(SIZE * 4),
                                rows_per_image: Some(SIZE),
                            },
                        },
                        wgpu::Extent3d {
                            width: SIZE,
                            height: SIZE,
                            depth_or_array_layers: 1,
                        },
                    );
                };
                copy(self.target.as_image_copy(), &self.readback);
                copy(
                    scene.gbuffer.targets.depth.as_image_copy(),
                    &self.depth_readback,
                );
                copy(
                    scene.gbuffer.targets.material.as_image_copy(),
                    &self.material_readback,
                );
            }
            queue.submit(std::iter::once(encoder.finish()));
            if !read {
                return None;
            }
            for buffer in [
                &self.readback,
                &self.depth_readback,
                &self.material_readback,
            ] {
                buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("buffer map failed"));
            }
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .expect("poll failed");
            let pixels = self
                .readback
                .get_mapped_range(..)
                .expect("buffer not mapped")
                .to_vec();
            self.readback.unmap();
            let depths = bytemuck::cast_slice::<u8, f32>(
                &self
                    .depth_readback
                    .get_mapped_range(..)
                    .expect("buffer not mapped"),
            )
            .to_vec();
            self.depth_readback.unmap();
            let materials = bytemuck::cast_slice::<u8, u32>(
                &self
                    .material_readback
                    .get_mapped_range(..)
                    .expect("buffer not mapped"),
            )
            .to_vec();
            self.material_readback.unmap();
            Some(Frame {
                pixels,
                depths,
                materials,
                // Not read back here. This harness exists to fly a camera and
                // compare the pictures it draws, and nothing along that path
                // asks about cloud; a fourth buffer copied every frame of every
                // flight would be paid for by every one of them.
                clouds: Vec::new(),
                cloud_size: UVec2::ZERO,
                base_level: scene.terrain.base_level(),
            })
        }
    }

    /// A cone of ground rising out of a flat raster, north of the middle.
    ///
    /// Something with a silhouette against the sky, and one that climbs the
    /// screen as the camera flies at it, which is what
    /// [`flying_at_a_hill_does_not_carry_the_sky_it_rises_into`] needs.
    fn hill() -> Vec<f32> {
        const PEAK: f32 = 400.0;
        const RADIUS: f32 = 20.0;
        (0..RASTER * RASTER)
            .map(|i| {
                let (x, y) = ((i % RASTER) as f32, (i / RASTER) as f32);
                let away = ((x - 64.0).powi(2) + (y - 32.0).powi(2)).sqrt();
                PEAK * (1.0 - away / RADIUS).max(0.0)
            })
            .collect()
    }

    /// Sky the eye has since moved behind something is handed back to the march.
    ///
    /// Carried sky is put at `SKY_DISTANCE`, far enough that where the eye has
    /// moved to rounds off. That is right for sky, which has only a direction,
    /// and wrong for the claim: "no ground down this ray" was established from
    /// where the eye was standing, and the parallel ray through the same pixel
    /// from where it is standing now is a different one. Flying at a hill, the
    /// crest climbs the screen, and the pixels it climbs into were sky a frame
    /// ago. Settled from the carry they stay sky until the dither drops their
    /// cell -- up to a dozen frames, in eight-by-eight blocks along the
    /// skyline, which is what makes it look like the ridge is coming apart.
    ///
    /// Flown rather than stepped once, because how far ground can reach is
    /// measured from the motion field, which describes the frame before: a
    /// standing start has nothing moving in it to measure and the first frame
    /// after it is carried on the strength of a still camera's history. That is
    /// the same frame of lag `ranks_for` already reads the field with, and it
    /// costs a frame of skyline at the start of a movement and nothing after.
    ///
    /// A second scene marches the last camera from nothing, which is the answer
    /// the first has to match: whatever the sky is there, it is not a question
    /// the reprojection is entitled to answer.
    #[test]
    fn flying_at_a_hill_does_not_carry_the_sky_it_rises_into() {
        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        // Low, so the crest is well above the eye and sweeps a good way up the
        // screen for steps the reprojection can otherwise carry most of.
        let from = Vec3::new(0.0, 30.0, 1500.0);
        // Metres are the wrong unit to judge this in: what decides whether the
        // defect shows is how far the crest sweeps across the *screen*, and
        // this raster is under four kilometres across. Six steps of a hundred
        // and fifty move the crest about three pixels of two hundred and
        // fifty-six a frame, which is what flying a real raster at 200 m/s a
        // few metres over the ground does to a skyline at 1280 by 720.
        let step_metres = 150.0;
        let steps = 6;
        let to = from - Vec3::Z * (step_metres * steps as f32);
        let aim = |camera: &mut Camera| {
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, 0.0, 0.0);
        };
        let screen = Offscreen::new(&device, format);

        let mut flown = test_scene(&device, format, test_residency(), hill(), flat_ground());
        aim(&mut flown.camera);
        flown.camera.position = from;
        flown.settle(&device, &queue);
        let start = screen.step(&device, &queue, &mut flown, from, true);
        for i in 1..steps {
            let at = from - Vec3::Z * (step_metres * i as f32);
            screen.step(&device, &queue, &mut flown, at, false);
        }
        let carried = screen.step(&device, &queue, &mut flown, to, true);

        let mut marched = test_scene(&device, format, test_residency(), hill(), flat_ground());
        aim(&mut marched.camera);
        marched.camera.position = to;
        marched.settle(&device, &queue);
        let fresh = screen.step(&device, &queue, &mut marched, to, true);

        let read = |frame: Option<Frame>| frame.expect("frame read back").count_sky();
        let (start, carried, fresh) = (read(start), read(carried), read(fresh));
        // The flight has to be one the defect could show up in at all: the
        // crest must really have climbed, taking sky with it, or there would be
        // nothing for the carry to get wrong and this would pass on nothing.
        assert!(
            start > fresh,
            "the hill covered {} more pixels by the end of the flight; it has to \
             rise into the sky for this to be a test of anything",
            start - fresh
        );
        // One-sided on purpose. Sky standing where the march finds ground is
        // the defect; a pixel of ground over the sky at the silhouette is the
        // ordinary lag of a carried point and is not what this is about.
        assert!(
            carried <= fresh,
            "flying {} m at the hill left {carried} pixels of sky where marching \
             the same camera from nothing gives {fresh}",
            step_metres * steps as f32
        );
    }

    /// A near ridge with a far one showing over the top of it.
    ///
    /// The far ridge is four times the distance and painted a different
    /// material, so which of the two a pixel is showing can be read straight
    /// off the frame. Flying at them closes the gap: the near crest climbs the
    /// screen faster than the far one, so it eats into the band of far ridge
    /// above it, which is the sweep
    /// [`flying_at_a_ridge_does_not_carry_what_it_hides`] measures.
    fn two_ridges() -> (Vec<f32>, Vec<MaterialId>) {
        let ridge = |row: f32, at: f32, half: f32, peak: f32| {
            peak * (1.0 - (row - at).abs() / half).max(0.0)
        };
        let heights = (0..RASTER * RASTER)
            .map(|i| {
                let row = (i / RASTER) as f32;
                ridge(row, 90.0, 6.0, 100.0).max(ridge(row, 16.0, 8.0, 600.0))
            })
            .collect();
        let materials = (0..RASTER * RASTER)
            .map(|i| if i / RASTER < 32 { SAND } else { GRASS })
            .collect();
        (heights, materials)
    }

    /// Whether this pixel is showing the sandy far ridge.
    ///
    /// The id the march wrote. It used to be "warmer than it is green", which
    /// separated sand from grass but would equally have found any of the dozen
    /// other warm materials in the palette had one been in shot.
    fn is_sandy(frame: &Frame, x: u32, y: u32) -> bool {
        frame.material(x, y) == Some(SAND)
    }

    fn count_sandy(frame: &Frame) -> usize {
        (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| is_sandy(frame, x, y))
            .count()
    }

    /// Ground the carry can still place is not ground it may still show.
    ///
    /// A carried point keeps its world position, which stays true however the
    /// camera moves. Whether it is still the nearest thing along its ray does
    /// not: a ridge sweeping across it should hide it, and will not if the
    /// ridge's own points were dropped by the dither or spread apart by
    /// magnification. What comes through is the background, in eight-by-eight
    /// speckles along the skyline -- the far snow on a peak, on the raster this
    /// was reported from.
    ///
    /// The depth test cannot catch this. It settles which of the points that
    /// *did* land is nearest, and the whole problem is the ones that did not.
    ///
    /// Flown for the same reason as the sky flight above: the reach is measured
    /// from the frame before, so a standing start has nothing moving in it yet.
    #[test]
    fn flying_at_a_ridge_does_not_carry_what_it_hides() {
        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (heights, materials) = two_ridges();
        // Low enough that the near ridge stands above the eye, and far enough
        // back that four times its distance is still inside the raster.
        let from = Vec3::new(0.0, 50.0, 1500.0);
        let step_metres = 50.0;
        let steps = 6;
        let to = from - Vec3::Z * (step_metres * steps as f32);
        let aim = |camera: &mut Camera| {
            camera.orientation = Camera::from_yaw_pitch_roll(0.0, 0.0, 0.0);
        };
        let screen = Offscreen::new(&device, format);

        let mut flown = test_scene(
            &device,
            format,
            test_residency(),
            heights.clone(),
            materials.clone(),
        );
        aim(&mut flown.camera);
        flown.camera.position = from;
        flown.settle(&device, &queue);
        let start = screen.step(&device, &queue, &mut flown, from, true);
        for i in 1..steps {
            let at = from - Vec3::Z * (step_metres * i as f32);
            screen.step(&device, &queue, &mut flown, at, false);
        }
        let carried = screen.step(&device, &queue, &mut flown, to, true);

        let mut marched = test_scene(&device, format, test_residency(), heights, materials);
        aim(&mut marched.camera);
        marched.camera.position = to;
        marched.settle(&device, &queue);
        let fresh = screen.step(&device, &queue, &mut marched, to, true);

        let (start, carried, fresh) = (
            start.expect("frame read back"),
            carried.expect("frame read back"),
            fresh.expect("frame read back"),
        );
        let sandy_rows = |frame: &Frame| -> Vec<usize> {
            (0..SIZE)
                .map(|y| (0..SIZE).filter(|&x| is_sandy(frame, x, y)).count())
                .collect()
        };
        let (carried_rows, fresh_rows) = (sandy_rows(&carried), sandy_rows(&fresh));
        let (start, carried, fresh) = (
            count_sandy(&start),
            count_sandy(&carried),
            count_sandy(&fresh),
        );
        // Both ridges have to be on screen, and the near one has to be closing
        // over the far one, or there is nothing here to get wrong.
        assert!(
            fresh > 0 && start > fresh,
            "the near ridge has to eat into the far one: {start} sandy pixels at \
             the start against {fresh} at the end"
        );

        // Where the near ridge cuts the far one off, in the frame that marched
        // it rather than carried it.
        let edge = fresh_rows
            .iter()
            .rposition(|&count| count > 0)
            .expect("the far ridge is on screen");

        // The defect is far ridge coming through *inside* the near one, which
        // is what a carried point the sweeping ridge failed to invalidate looks
        // like: eight-by-eight speckles well below the skyline, because whole
        // dither cells refresh together. Counting sandy pixels cannot tell that
        // from a silhouette one row fat, which is the ordinary lag of a carried
        // point and is what this actually leaves -- so measure where they are,
        // not how many there are.
        let speckles: usize = carried_rows[(edge + 2).min(SIZE as usize)..].iter().sum();
        assert_eq!(
            speckles,
            0,
            "flying {} m left {speckles} pixels of far ridge below row {edge}, \
             where the near ridge is the nearest thing along every ray",
            step_metres * steps as f32
        );
        // ... and the lag at the silhouette itself is one row of it, no more.
        assert!(
            carried <= fresh + SIZE as usize,
            "flying {} m at the ridges left {carried} pixels of the far ridge \
             where marching the same camera from nothing gives {fresh}, which is \
             more than a row of silhouette lag",
            step_metres * steps as f32
        );
    }

    /// A camera that has stopped moving must settle, and stay settled.
    ///
    /// Standing still is the carry's exact case: every point it holds projects
    /// back to the pixel it came from, so once the dither has swept the screen
    /// the frame reaches a fixed point and every frame after it is identical.
    /// Anything still changing is a point the round trip through the G-buffer
    /// does not put back where it found it.
    ///
    /// The defect this was written for did that in one direction. The sub-pixel
    /// offset was packed against the pixel the *vertex* stage predicted the
    /// point would land in rather than the one the rasterizer chose, so a point
    /// landing on a pixel boundary was stored describing the pixel above.
    /// Rebuilt from that word next frame it landed on a boundary again, so it
    /// climbed the screen at exactly one pixel a frame -- chunks of ground
    /// sliding up through the sky and off the top over a few seconds, with the
    /// camera untouched throughout. `pack_offset` in `src/reproject.wgsl` is
    /// where the whole of it is written down.
    ///
    /// Flown a few frames first, and that is the point of the flight: from a
    /// standing start every pixel is marched and a marched pixel's offset is a
    /// pixel centre, which is stable. Only a carried point can land on a
    /// boundary, so only a camera that has moved can reach the state that
    /// fails.
    ///
    /// The aim is not arbitrary either. A boundary landing is a coincidence of
    /// one point in 256, so a small frame holds few candidates and most views
    /// hold none: at this pitch the unfixed shader leaves exactly one climbing
    /// pixel, and at six degrees down it leaves none at all. What made the
    /// defect obvious on a real flight was a 3440x1440 window -- fifty times
    /// these pixels, and chunks rather than specks.
    #[test]
    fn a_still_camera_carries_every_point_back_to_itself() {
        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let (heights, materials) = rugged();
        let mut scene = test_scene(&device, format, test_residency(), heights, materials);
        // Just below the horizon, so the ground runs from under the camera out
        // to the skyline and the frame holds every distance at once.
        scene.camera.orientation = Camera::from_yaw_pitch_roll(0.0, -2f32.to_radians(), 0.0);
        let start = Vec3::new(0.0, 1200.0, 1700.0);
        scene.camera.position = start;
        scene.settle(&device, &queue);

        // A few metres a frame, which is about what a key tapped at the fly
        // speed covers.
        const STEP: f32 = 3.5;
        const MOVING: u32 = 6;
        let screen = Offscreen::new(&device, format);
        for frame in 0..MOVING {
            let at = start - Vec3::Z * (STEP * frame as f32);
            screen.step(&device, &queue, &mut scene, at, false);
        }

        // Then held, long enough for what the movement stirred up to settle:
        // the two frames either side of this are down to single pixels of
        // difference, which is what makes exact equality the right assertion.
        let stopped = start - Vec3::Z * (STEP * (MOVING - 1) as f32);
        for _ in 0..24 {
            screen.step(&device, &queue, &mut scene, stopped, false);
        }
        let settled = screen
            .step(&device, &queue, &mut scene, stopped, true)
            .expect("frame read back");
        let after = screen
            .step(&device, &queue, &mut scene, stopped, true)
            .expect("frame read back");

        let moved: Vec<(u32, u32)> = (0..SIZE)
            .flat_map(|y| (0..SIZE).map(move |x| (x, y)))
            .filter(|&(x, y)| settled.pixel(x, y) != after.pixel(x, y))
            .collect();
        assert!(
            moved.is_empty(),
            "{} pixels changed between two frames of a camera that had not moved \
             for twenty-five of them, at {:?}",
            moved.len(),
            moved,
        );
    }

    /// The overlay's reader, driven exactly the way `Renderer::render` drives
    /// it: record on the frame's encoder, submit, collect. **No poll**, on
    /// purpose -- the windowed loop has none either, and relies on the one
    /// `Queue::submit` performs internally to fire the map callback. Adding one
    /// here would test a path the window never takes and hide the failure it is
    /// meant to catch: a reader that never delivers leaves the overlay blank
    /// for the whole run, and there is no window in a test to notice that in.
    #[test]
    fn the_overlay_reader_gets_a_number_back_without_blocking() {
        let (device, queue) = test_device();
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut scene = test_scene(
            &device,
            format,
            test_residency(),
            vec![0.0; (RASTER * RASTER) as usize],
            flat_ground(),
        );
        straight_down(&mut scene.camera);
        scene.settle(&device, &queue);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("coverage target"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let mut reader = crate::reproject::CoverageReader::new(&device);
        let profiler = crate::profile::profiler(&device, false);
        let mut arrived = None;
        // Not a latency estimate, and deliberately not tuned to one. Nothing
        // here blocks, so how many frames an answer takes is however deep the
        // queue happens to run before a poll observes the copy finishing --
        // measured between 23 and 33 on the machine this was written on, where
        // an earlier bound of 16 passed for a while and then stopped. What the
        // test is actually for is a reader that never delivers at all, which
        // would leave the overlay blank for a whole run; ten seconds of a
        // sixty-a-second one is past any argument about queue depth, and still
        // fails rather than hangs.
        for _ in 0..600 {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            {
                let mut gpu = profiler.scope("gpu", &mut encoder);
                scene.draw(&mut gpu, &view);
            }
            reader.record(&mut encoder, scene.tally());
            queue.submit(std::iter::once(encoder.finish()));

            arrived = reader.collect();
            if arrived.is_some() {
                break;
            }
            scene.update(&device, &queue, crate::headless::STEP);
        }

        let coverage = arrived.expect("the reader never delivered a coverage in 16 frames");
        assert_eq!(coverage.total(), SIZE * SIZE, "{coverage:?}");
    }
}
