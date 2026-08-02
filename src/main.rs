mod camera;
mod controls;
mod deferred;
mod headless;
mod hud;
mod palette;
mod renderer;
mod scene;
mod terrain;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use glam::UVec2;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

use crate::controls::FlyController;
use crate::headless::Placement;
use crate::renderer::Renderer;

/// Size of the window, and of a screenshot that does not ask for another.
const DEFAULT_SIZE: UVec2 = UVec2::new(1280, 720);

/// Where the terrain comes from, and whether to fly over it or photograph it.
///
/// The pyramid is far too large to carry in the repository -- a box a few
/// kilometres square is hundreds of megabytes -- so there is no default path to
/// fall back on. `terrain-download` fetches the measurements and
/// `terrain-process` turns them into the tree this reads; `--terrain` points at
/// what the second of those wrote, not the first.
#[derive(Parser, Debug)]
#[command(about = "Fly over terrain streamed from a tile pyramid", long_about = None)]
struct Arguments {
    /// Directory holding the tile pyramid, with a subdirectory per product.
    #[arg(short, long, value_name = "DIR")]
    terrain: PathBuf,

    /// Render a single frame to this PNG and exit, without opening a window.
    ///
    /// Presenting a swapchain needs a display server; drawing into a texture
    /// does not. This is the way in on a machine that has the GPU but no
    /// screen -- a container given `/dev/dri` and nothing else.
    #[arg(short = 'o', long, value_name = "FILE")]
    screenshot: Option<PathBuf>,

    /// Where to put the camera, as `x,y,z,yaw,pitch`: metres, then degrees.
    ///
    /// Without it the opening view is kept, which frames the whole extent and
    /// so looks at whatever is most of the box rather than at any part of it.
    #[arg(long, value_name = "X,Y,Z,YAW,PITCH", requires = "screenshot")]
    camera: Option<Placement>,

    /// Size of the screenshot, as `WIDTHxHEIGHT`. Defaults to the window's.
    ///
    /// Not merely a crop: how much ground the clipmap keeps resident is chosen
    /// so a texel lands on about a pixel, so this changes what is loaded.
    #[arg(long, value_name = "WxH", requires = "screenshot", value_parser = parse_size)]
    size: Option<UVec2>,
}

/// Reads `WIDTHxHEIGHT` for [`Arguments::size`].
fn parse_size(text: &str) -> Result<UVec2, String> {
    let (width, height) = text
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("expected WIDTHxHEIGHT, got {text:?}"))?;
    let read = |side: &str, value: &str| {
        value
            .trim()
            .parse::<u32>()
            .map_err(|err| format!("{side} of {text:?}: {err}"))
            .and_then(|n| {
                (n > 0)
                    .then_some(n)
                    .ok_or_else(|| format!("{side} is zero"))
            })
    };
    Ok(UVec2::new(read("width", width)?, read("height", height)?))
}

struct App {
    display: OwnedDisplayHandle,
    terrain: PathBuf,
    renderer: Option<Renderer>,
    controls: FlyController,
    /// When the last frame was drawn, for the timestep the controls integrate over.
    last_frame: Instant,
}

impl App {
    fn new(display: OwnedDisplayHandle, terrain: PathBuf) -> Self {
        Self {
            display,
            terrain,
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
            .with_inner_size(LogicalSize::new(DEFAULT_SIZE.x, DEFAULT_SIZE.y));
        let window = match event_loop.create_window(attributes) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                log::error!("failed to create window: {err}");
                event_loop.exit();
                return;
            }
        };

        match pollster::block_on(Renderer::new(window, self.display.clone(), &self.terrain)) {
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

    let arguments = Arguments::parse();

    // Before the event loop rather than inside it: building one already fails on
    // a machine with no display server, which is exactly where this mode is for.
    if let Some(output) = arguments.screenshot.as_deref() {
        return headless::run(
            &arguments.terrain,
            arguments.size.unwrap_or(DEFAULT_SIZE),
            arguments.camera,
            output,
        );
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(event_loop.owned_display_handle(), arguments.terrain);
    event_loop.run_app(&mut app)?;

    Ok(())
}
