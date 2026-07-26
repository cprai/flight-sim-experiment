mod camera;
mod controls;
mod renderer;
mod scene;

use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

use crate::controls::FlyController;
use crate::renderer::Renderer;

struct App {
    display: OwnedDisplayHandle,
    renderer: Option<Renderer>,
    controls: FlyController,
    /// When the last frame was drawn, for the timestep the controls integrate over.
    last_frame: Instant,
}

impl App {
    fn new(display: OwnedDisplayHandle) -> Self {
        Self {
            display,
            renderer: None,
            controls: FlyController::default(),
            last_frame: Instant::now(),
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
            Ok(renderer) => {
                self.controls = FlyController::new(renderer.camera());
                self.last_frame = Instant::now();
                self.renderer = Some(renderer);
            }
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
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                // Synthetic events report the keys already down when the window
                // regains focus; taking them would undo `release_all`.
                is_synthetic: false,
                ..
            } => self.controls.key(code, state.is_pressed()),
            // Reading shift from the modifier state rather than from its own key
            // events keeps it correct when the key goes down or up while another
            // window has focus.
            WindowEvent::ModifiersChanged(modifiers) => {
                self.controls.set_boost(modifiers.state().shift_key());
            }
            WindowEvent::MouseInput {
                button: MouseButton::Left,
                state,
                ..
            } => self.controls.set_looking(state.is_pressed()),
            WindowEvent::Focused(false) => self.controls.release_all(),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                if let Some(renderer) = self.renderer.as_mut() {
                    self.controls.update(renderer.camera_mut(), dt);
                    renderer.render();
                    // Drive a continuous render loop rather than redrawing only on demand.
                    renderer.window().request_redraw();
                }
            }
            _ => {}
        }
    }

    /// Raw mouse deltas drive the look, rather than the cursor positions in
    /// [`WindowEvent::CursorMoved`]: they carry no pointer acceleration and keep
    /// arriving once a drag reaches the edge of the screen, where the cursor
    /// stops and its positions would repeat.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.controls.mouse_motion(dx as f32, dy as f32);
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
