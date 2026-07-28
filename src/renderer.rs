use std::sync::Arc;

use winit::window::Window;

use crate::camera::Camera;
use crate::scene::{Scene, create_depth_view};

/// Owns the GPU device and swapchain for a single window, and the scene it draws.
pub struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// Sized to match the swapchain, so it is rebuilt whenever that is.
    depth: wgpu::TextureView,
    scene: Scene,
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
                power_preference: wgpu::PowerPreference::default(),
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

        let depth = create_depth_view(&device, config.width, config.height);
        // Sized for the surface as it is now. A later resize only changes the
        // aspect: the clipmap's textures are allocated once, and rebuilding
        // them mid-flight would mean refilling every window from disk.
        let scene = Scene::new(
            &device,
            format,
            glam::UVec2::new(config.width, config.height),
            terrain_root,
        )?;

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            depth,
            scene,
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
        // A depth attachment has to match the colour target's dimensions exactly,
        // so it cannot outlive the old swapchain size.
        self.depth = create_depth_view(&self.device, width, height);
        // Without this the projection would stretch the scene to fit the new
        // viewport instead of widening the field of view.
        self.scene.camera.aspect = aspect_ratio(&self.config);
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

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        self.scene.update(&self.queue);
        self.scene.draw(&mut encoder, &view, &self.depth);

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);

        // `present` consumed the frame, so the surface can be reconfigured now.
        if stale {
            self.reconfigure();
        }
    }
}

/// Viewport aspect ratio the camera should project with.
fn aspect_ratio(config: &wgpu::SurfaceConfiguration) -> f32 {
    config.width as f32 / config.height as f32
}
