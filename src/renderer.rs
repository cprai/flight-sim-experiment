use std::sync::Arc;

use winit::window::Window;

use crate::camera::Camera;
use crate::hud::Hud;
use crate::profile;
use crate::scene::Scene;

/// Owns the GPU device and swapchain for a single window, and the scene it draws.
pub struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scene: Scene,
    profiling: bool,
    /// When the frame being drawn started, for the interval to the next one.
    last_frame: Option<std::time::Instant>,
    frame: profile::Frame,
    readout: profile::Smoothed,
    /// Absent unless profiling, or when the machine has no fonts to draw with.
    hud: Option<Hud>,
    /// Absent unless profiling: an unprofiled run has nowhere to show this and
    /// no reason to spend a copy a frame producing it.
    coverage: Option<crate::reproject::CoverageReader>,
    /// Absent unless profiling, for the same reason: reading the two pools
    /// costs a handful of file reads that an unwatched run has no use for.
    memory: Option<crate::memory::Meter>,
}

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        display: winit::event_loop::OwnedDisplayHandle,
        terrain_root: &std::path::Path,
        profiling: bool,
        sun: Option<crate::headless::SunAngles>,
    ) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_with_display_handle_from_env(Box::new(display)),
        );

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                // `None`, the default, makes wgpu take whichever adapter
                // enumerates first — it does no sorting at all for that value.
                // Machines with a discrete GPU beside an integrated one and a
                // software fallback need to say which they want, so honour
                // `WGPU_POWER_PREF` and keep the old behaviour when it is unset.
                power_preference: wgpu::PowerPreference::from_env().unwrap_or_default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await?;
        log::info!("using adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                // Only the timer queries, and only those the adapter has. The
                // headless device asks for the same, so a frame measured there
                // is evidence about the frame this draws.
                required_features: profile::timer_features(&adapter),
                // The WebGPU baseline but for one raise the G-buffer needs;
                // see `deferred::limits`. Shared with the headless device for
                // the same reason the features are.
                required_limits: crate::deferred::limits(&adapter.limits()),
                ..Default::default()
            })
            .await?;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Sized for the surface as it is now. A later resize rebuilds the
        // screen-sized G-buffer but not the clipmap: its textures are
        // allocated once, and rebuilding them mid-flight would mean refilling
        // every window from disk.
        let mut scene = Scene::new(
            &device,
            format,
            glam::UVec2::new(config.width, config.height),
            terrain_root,
        )?;
        scene.profile(&device, profiling);
        // Set once here rather than steered from the controls: the tables the
        // atmosphere reads are rebuilt from `scene.sun` every frame, so moving
        // the sun later would cost nothing and need no invalidation -- but that
        // is a key binding and a HUD readout, and this is the flag that lets a
        // window be pointed at the same sky a `render` was.
        scene.sun = crate::headless::SunAngles::or_default(sun);
        // Said out loud for the same reason the headless modes say it: two runs
        // that differ only in where the sun was are otherwise indistinguishable
        // in the log, and the sun decides the colour of every pixel.
        log::info!("sun towards {}", scene.sun.direction);

        // No overlay at all on an unprofiled run: there is nothing to put in it.
        let hud =
            profiling.then(|| Hud::new(&device, &queue, format, window.scale_factor() as f32));
        let coverage = profiling.then(|| crate::reproject::CoverageReader::new(&device));
        // Built from the adapter rather than the device: what the card holds is
        // the kernel's to say, and the adapter is what names the card.
        let memory = profiling.then(|| crate::memory::Meter::new(&adapter));

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            scene,
            profiling,
            last_frame: None,
            frame: profile::Frame::default(),
            readout: profile::Smoothed::default(),
            hud: hud.flatten(),
            coverage,
            memory,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn camera(&self) -> &Camera {
        &self.scene.camera
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.scene.camera
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.config.width && height == self.config.height)
        {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        // The G-buffer has to match the swapchain's dimensions exactly, and
        // the camera's aspect follows the viewport; the scene owns both.
        self.scene
            .resize(&self.device, glam::UVec2::new(width, height));
    }

    /// Reconfigures the swapchain at its current size, e.g. after it goes stale.
    fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }

    pub fn render(&mut self) {
        let (frame, stale) = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
            // Still drawable, but the swapchain no longer matches the surface. Draw
            // it anyway and rebuild afterwards -- reconfiguring while the frame is
            // still acquired is a validation error.
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.reconfigure();
                return;
            }
            // Nothing useful to draw into this frame; the next redraw will retry.
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };

        // Timed from here rather than from the top of the call: with vsync it is
        // `get_current_texture` above that blocks, and counting that wait as
        // work would make the interval agree with itself and say nothing.
        let started = std::time::Instant::now();
        if let Some(previous) = self.last_frame.replace(started) {
            self.frame.interval = started.duration_since(previous);
        }

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        self.scene.update(&self.device, &self.queue);
        self.scene.record(&mut self.frame);
        // What the coverage is a share of, the ceiling a climbing ray has to
        // clear before it can be settled as sky for free, and the viewpoint the
        // whole frame was drawn from. All of them are of this frame; the
        // coverage they sit beside is a few frames older, which matters only
        // while the camera is moving hard.
        self.frame.pixels = self.config.width * self.config.height;
        self.frame.ceiling = self.scene.ceiling();
        self.frame.position = self.scene.camera.position;
        self.frame.orientation = self.scene.camera.orientation;
        if let Some(meter) = self.memory.as_mut() {
            self.frame.memory = meter.sample(&self.device);
        }

        let clock = profile::Clock::start(self.profiling);
        self.scene.draw(&mut encoder, &view);

        // Over the top of the shaded frame, and only here: the overlay reports
        // on the renderer rather than being part of what it renders, so the
        // screenshot path in `crate::headless`, which shares `Scene::draw`,
        // stays free of it. It also wants `&mut` to lay the text out, which
        // `Scene::draw` does not take.
        //
        // In a scope of its own, beside the scene's rather than inside it: the
        // overlay is not one of the passes it reports on, and drawing several
        // hundred glyphs is not free enough to hide inside another row.
        if let Some(hud) = self.hud.as_mut() {
            let mut gpu = self.scene.profiler().scope("hud", &mut encoder);
            // The rows are last frame's: the frame being drawn cannot report
            // times it has not finished taking, and the GPU ones come back
            // later still.
            hud.draw(
                &self.device,
                &self.queue,
                &mut gpu,
                crate::hud::Target {
                    view: &view,
                    resolution: glam::UVec2::new(self.config.width, self.config.height),
                    scale_factor: self.window.scale_factor() as f32,
                },
                &self.readout.text(),
            );
        }
        self.frame.cpu.encode = clock.elapsed();

        // Outside every scope closed above, so the twelve-byte copy is not
        // charged to the passes it reports on.
        if let Some(coverage) = self.coverage.as_mut() {
            coverage.record(&mut encoder, self.scene.tally());
        }

        // Has to follow every scope on this encoder and precede its `finish`:
        // this is the copy that moves the query set into a readable buffer. It
        // covers the terrain's scopes too -- those were submitted from
        // `Scene::update` above, so their timestamps are already written by the
        // time this resolve runs.
        self.scene.profiler_mut().resolve_queries(&mut encoder);

        let clock = profile::Clock::start(self.profiling);
        self.queue.submit(std::iter::once(encoder.finish()));
        self.frame.cpu.submit = clock.elapsed();

        self.queue.present(frame);
        self.collect();

        // `present` consumed the frame, so the surface can be reconfigured now.
        if stale {
            self.reconfigure();
        }
    }

    /// Closes the profiler frame and folds whatever came back into the readout.
    ///
    /// The GPU results lag: timestamps are read back through a buffer mapping,
    /// so a frame's own numbers are not available while it is being drawn and
    /// [`GpuProfiler::process_finished_frame`] returns the oldest one that has
    /// finished, or nothing yet. That is why the overlay shows the previous
    /// frame's rows -- the alternative is showing none for the first few
    /// frames and then always being a frame behind anyway.
    ///
    /// [`GpuProfiler::process_finished_frame`]: wgpu_profiler::GpuProfiler::process_finished_frame
    fn collect(&mut self) {
        if !self.profiling {
            return;
        }
        if let Err(err) = self.scene.profiler_mut().end_frame() {
            log::warn!("the profiler dropped a frame: {err}");
            return;
        }
        let period = self.queue.get_timestamp_period();
        if let Some(results) = self.scene.profiler_mut().process_finished_frame(period) {
            self.frame.take_gpu(&results);
        }
        // Lags further than the timestamps do -- one read is in flight at a
        // time, so a fresh number lands every few frames -- and the reader
        // holds the last one, so this keeps showing it in between rather than
        // blinking out.
        if let Some(coverage) = self.coverage.as_mut() {
            self.frame.coverage = coverage.collect();
        }
        self.readout.update(&self.frame);
    }
}
