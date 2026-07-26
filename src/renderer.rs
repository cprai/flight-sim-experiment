use std::sync::Arc;

use winit::window::Window;

/// Owns the GPU device and swapchain for a single window.
pub struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    pub async fn new(
        window: Arc<Window>,
        display: winit::event_loop::OwnedDisplayHandle,
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
                // Keep the limits conservative so the app also runs on WebGL/older GPUs.
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
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

        let pipeline = create_triangle_pipeline(&device, format);

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
        })
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.config.width && height == self.config.height)
        {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
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

        draw_triangle(&mut encoder, &view, &self.pipeline);

        self.queue.submit(std::iter::once(encoder.finish()));
        self.queue.present(frame);

        // `present` consumed the frame, so the surface can be reconfigured now.
        if stale {
            self.reconfigure();
        }
    }
}

/// Background the triangle is drawn over.
pub const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.05,
    g: 0.06,
    b: 0.09,
    a: 1.0,
};

/// Builds the pipeline that draws the triangle baked into `shader.wgsl`.
///
/// Shared with the headless test so both exercise the same shader and state.
pub fn create_triangle_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("triangle pipeline"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Records a clear + triangle draw into `view`.
pub fn draw_triangle(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("triangle pass"),
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

    pass.set_pipeline(pipeline);
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: u32 = 256;

    /// Renders the triangle to an offscreen texture and returns the RGBA8 pixels.
    ///
    /// Lets CI verify the shader and pipeline without a window or display server.
    fn render_offscreen() -> Vec<u8> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no wgpu adapter available");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("failed to create device");

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

        // `SIZE * 4` is already a multiple of the 256-byte copy alignment.
        let bytes_per_row = SIZE * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: u64::from(bytes_per_row * SIZE),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let pipeline = create_triangle_pipeline(&device, format);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        draw_triangle(&mut encoder, &view, &pipeline);
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
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

        let pixels = readback
            .get_mapped_range(..)
            .expect("buffer not mapped")
            .to_vec();
        readback.unmap();
        pixels
    }

    fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i..i + 4].try_into().unwrap()
    }

    #[test]
    fn renders_a_triangle_with_interpolated_vertex_colors() {
        let pixels = render_offscreen();

        // Outside the triangle: the clear color, which is dark and blue-dominant.
        // The target is sRGB, so the linear clear values encode to roughly 63,69,85.
        let [r, g, b, a] = pixel(&pixels, 2, 2);
        assert_eq!(a, 255, "background should be opaque");
        assert!(
            r < 110 && g < 110 && b < 110,
            "corner should be dark, got {r},{g},{b}"
        );
        assert!(
            b > r && b > g,
            "corner should be the blue-ish clear color, got {r},{g},{b}"
        );

        // Just inside each vertex the interpolated color should favor that vertex's channel.
        // Coordinates are NDC points mapped to pixels: x = (nx + 1)/2 * SIZE, y = (1 - ny)/2 * SIZE.
        let [r, g, b, _] = pixel(&pixels, 128, 70); // near (0.0, 0.45) — red vertex
        assert!(
            r > g && r > b,
            "top of triangle should be red-dominant, got {r},{g},{b}"
        );

        let [r, g, b, _] = pixel(&pixels, 64, 185); // near (-0.5, -0.45) — green vertex
        assert!(
            g > r && g > b,
            "bottom-left should be green-dominant, got {r},{g},{b}"
        );

        let [r, g, b, _] = pixel(&pixels, 192, 185); // near (0.5, -0.45) — blue vertex
        assert!(
            b > r && b > g,
            "bottom-right should be blue-dominant, got {r},{g},{b}"
        );
    }
}
