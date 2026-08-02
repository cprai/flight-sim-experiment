use std::sync::Arc;

use winit::window::Window;

use crate::camera::Camera;
use crate::hud::{FrameTimer, Hud};
use crate::scene::Scene;

/// Owns the GPU device and swapchain for a single window, and the scene it draws.
pub struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    scene: Scene,
    timer: FrameTimer,
    /// Absent when the machine has no fonts to draw the readout with.
    hud: Option<Hud>,
}

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        display: winit::event_loop::OwnedDisplayHandle,
        terrain_root: &std::path::Path,
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
                required_features: wgpu::Features::empty(),
                // The WebGPU baseline, which every GPU on the target platforms clears.
                // Nothing here needs an optional feature yet, so none are requested.
                required_limits: wgpu::Limits::default(),
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
        let scene = Scene::new(
            &device,
            format,
            glam::UVec2::new(config.width, config.height),
            terrain_root,
        )?;

        let hud = Hud::new(&device, &queue, format, window.scale_factor() as f32);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            scene,
            timer: FrameTimer::default(),
            hud,
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
        // work would make this number agree with the frame interval and say
        // nothing the frame interval does not already say.
        let started = std::time::Instant::now();
        self.timer.begin(started);

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        self.scene.update(&self.queue);
        self.scene.draw(&mut encoder, &view);

        // Over the top of the shaded frame, and only here: the overlay reports
        // on the renderer rather than being part of what it renders, so the
        // screenshot path in `crate::headless`, which shares `Scene::draw`,
        // stays free of it. It also wants `&mut` to lay the text out, which
        // `Scene::draw` does not take.
        if let Some(hud) = self.hud.as_mut() {
            // Last frame's submit cost paired with this frame's interval: the
            // one being drawn cannot report a time it has not finished taking.
            hud.draw(
                &self.device,
                &self.queue,
                &mut encoder,
                crate::hud::Target {
                    view: &view,
                    resolution: glam::UVec2::new(self.config.width, self.config.height),
                    scale_factor: self.window.scale_factor() as f32,
                },
                &self.timer.text(),
            );
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        self.timer.end(started.elapsed());
        self.queue.present(frame);

        // `present` consumed the frame, so the surface can be reconfigured now.
        if stale {
            self.reconfigure();
        }
    }
}
