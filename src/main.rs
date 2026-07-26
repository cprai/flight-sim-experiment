mod camera;
mod renderer;
mod scene;

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::window::{Window, WindowId};

use crate::renderer::Renderer;

struct App {
    display: OwnedDisplayHandle,
    renderer: Option<Renderer>,
}

impl App {
    fn new(display: OwnedDisplayHandle) -> Self {
        Self {
            display,
            renderer: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // `resumed` fires again after suspension on mobile; only build once.
        if self.renderer.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("flight-sim")
            .with_inner_size(LogicalSize::new(1280, 720));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                log::error!("failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(Renderer::new(window, self.display.clone())) {
            Ok(renderer) => self.renderer = Some(renderer),
            Err(err) => {
                log::error!("failed to initialize renderer: {err:?}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                renderer.render();
                // Drive a continuous render loop rather than redrawing only on demand.
                renderer.window().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,flight_sim=info"),
    )
    .init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(event_loop.owned_display_handle());
    event_loop.run_app(&mut app)?;

    Ok(())
}
