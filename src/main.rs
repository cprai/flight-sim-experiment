mod camera;
mod controls;
mod deferred;
mod headless;
mod hud;
mod memory;
mod palette;
mod profile;
mod renderer;
mod reproject;
mod scene;
mod sky;
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
use crate::headless::{Placement, SunAngles};
use crate::renderer::Renderer;

/// Size of the window, and of a screenshot that does not ask for another.
const DEFAULT_SIZE: UVec2 = UVec2::new(1280, 720);

/// Fly over terrain, photograph it, or find out what drawing it costs.
///
/// The pyramid is far too large to carry in the repository -- a box a few
/// kilometres square is hundreds of megabytes -- so there is no default path to
/// fall back on. `terrain-download` fetches the measurements and
/// `terrain-process` turns them into the tree this reads; `--terrain` points at
/// what the second of those wrote, not the first.
#[derive(Parser, Debug)]
#[command(about = "Fly over terrain streamed from a tile pyramid", long_about = None)]
struct Arguments {
    #[command(subcommand)]
    mode: Mode,
}

/// Where the terrain comes from. Every mode needs it and none can guess it.
///
/// Repeated into each mode rather than made a global argument, because clap
/// does not allow a global to be required and this genuinely is.
#[derive(clap::Args, Debug)]
struct Terrain {
    /// Directory holding the tile pyramid, with a subdirectory per product.
    #[arg(short, long, value_name = "DIR")]
    terrain: PathBuf,
}

/// Which of the four ways to run.
///
/// Subcommands rather than flags because the modes do not share arguments:
/// `--camera` means nothing to a window you can steer, and an output path means
/// nothing to a run that measures. As flags those had to be bound together with
/// clap `requires` attributes that said so only after the fact.
#[derive(clap::Subcommand, Debug)]
enum Mode {
    /// Open a window and fly.
    Fly(Terrain),

    /// Open a window and fly, with the frame breakdown drawn in the corner.
    FlyProfile(Terrain),

    /// Render a single frame to a PNG and exit, without opening a window.
    ///
    /// Presenting a swapchain needs a display server; drawing into a texture
    /// does not. This is the way in on a machine that has the GPU but no
    /// screen -- a container given `/dev/dri` and nothing else. It reports no
    /// timings: one cold frame is an image, not a measurement.
    Render {
        #[command(flatten)]
        terrain: Terrain,

        /// Where to write the PNG.
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,

        #[command(flatten)]
        view: View,

        /// How many frames to draw before the one that is written.
        ///
        /// One frame cannot show anything carried between frames -- there is no
        /// frame before it to carry from -- so a run that wants to see what
        /// reuse looks like has to draw a few and keep the last.
        #[arg(long, default_value_t = 1, value_name = "N")]
        frames: u32,
    },

    /// Settle the terrain, then measure frames and print where the time went.
    ///
    /// Writes no image. Prints a table to stdout: one row per step of the
    /// frame, GPU and CPU kept apart, with the spread across the run.
    Profile {
        #[command(flatten)]
        terrain: Terrain,

        #[command(flatten)]
        view: View,

        /// How many frames to measure, after a few discarded to warm up.
        #[arg(long, default_value_t = 60, value_name = "N")]
        frames: u32,
    },
}

/// What a headless mode looks at, and how big the frame is.
#[derive(clap::Args, Debug)]
struct View {
    /// Where to put the camera, as `x,y,z,yaw,pitch`: metres, then degrees.
    ///
    /// Without it the opening view is kept, which frames the whole extent and
    /// so looks at whatever is most of the box rather than at any part of it.
    #[arg(long, value_name = "X,Y,Z,YAW,PITCH")]
    camera: Option<Placement>,

    /// Size of the frame, as `WIDTHxHEIGHT`. Defaults to the window's.
    ///
    /// Not merely a crop: the viewport decides the finest clipmap level worth
    /// filling, so this changes what is drawn as well as how much of it.
    #[arg(long, value_name = "WxH", value_parser = parse_size)]
    size: Option<UVec2>,

    /// How fast to fly forward, in metres per second.
    ///
    /// Zero, the default, holds the camera exactly still. A still camera is the
    /// best case for anything a frame reuses from the one before it: every
    /// pixel of ground lands back where it came from, and nothing is left for
    /// the march. Flying is what says whether that survives ground coming into
    /// view for the first time.
    #[arg(long, default_value_t = 0.0, value_name = "M/S")]
    motion: f32,

    /// Where the sun is, as `ELEVATION,AZIMUTH` in degrees.
    ///
    /// Elevation is measured from the horizon and may be negative, which puts
    /// the sun below it; azimuth is a compass bearing, zero north and ninety
    /// east. Without it the sun sits where it always has, 45 degrees up in the
    /// south-east, so every existing invocation draws the frame it drew before.
    #[arg(long, value_name = "ELEVATION,AZIMUTH")]
    sun: Option<SunAngles>,
}

/// Reads `WIDTHxHEIGHT` for [`View::size`].
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
    /// Whether to instrument the frame and draw the breakdown over it.
    profiling: bool,
    renderer: Option<Renderer>,
    controls: FlyController,
    /// When the last frame was drawn, for the timestep the controls integrate over.
    last_frame: Instant,
}

impl App {
    fn new(display: OwnedDisplayHandle, terrain: PathBuf, profiling: bool) -> Self {
        Self {
            display,
            terrain,
            profiling,
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

        match pollster::block_on(Renderer::new(
            window,
            self.display.clone(),
            &self.terrain,
            self.profiling,
        )) {
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

    // The headless modes run before the event loop is ever built, deliberately:
    // `EventLoop::new` fails outright on a machine with no display server, which
    // is exactly where those modes are for.
    let (terrain, profiling) = match arguments.mode {
        Mode::Render {
            terrain,
            output,
            view,
            frames,
        } => {
            return headless::render(
                &terrain.terrain,
                view.size.unwrap_or(DEFAULT_SIZE),
                view.camera,
                view.sun,
                headless::Flight {
                    frames,
                    speed: view.motion,
                },
                &output,
            );
        }
        Mode::Profile {
            terrain,
            view,
            frames,
        } => {
            return headless::profile(
                &terrain.terrain,
                view.size.unwrap_or(DEFAULT_SIZE),
                view.camera,
                view.sun,
                headless::Flight {
                    frames,
                    speed: view.motion,
                },
            );
        }
        Mode::Fly(terrain) => (terrain.terrain, false),
        Mode::FlyProfile(terrain) => (terrain.terrain, true),
    };

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(event_loop.owned_display_handle(), terrain, profiling);
    event_loop.run_app(&mut app)?;

    Ok(())
}
