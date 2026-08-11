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
/// Subcommands rather than flags because the modes hardly share arguments:
/// `--camera` means nothing to a window you can steer, and an output path means
/// nothing to a run that measures. As flags those had to be bound together with
/// clap `requires` attributes that said so only after the fact. The one thing
/// all four do share, [`Sky`], is flattened into each rather than made a global:
/// clap cannot require a global, and a global would sit before the subcommand,
/// which reads oddly for something this specific to the picture.
#[derive(clap::Subcommand, Debug)]
enum Mode {
    /// Open a window and fly.
    Fly {
        #[command(flatten)]
        terrain: Terrain,

        #[command(flatten)]
        sky: Sky,
    },

    /// Open a window and fly, with the frame breakdown drawn in the corner.
    FlyProfile {
        #[command(flatten)]
        terrain: Terrain,

        #[command(flatten)]
        sky: Sky,
    },

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

    #[command(flatten)]
    sky: Sky,
}

/// Where the sun is, which every mode needs and none can infer.
///
/// Its own struct, flattened into all four modes, so the flag is declared once
/// and cannot drift between them. The atmosphere is a pure function of the
/// camera and this, so a `fly` run and the `render` that reproduces a frame from
/// it have to be able to say the same thing.
#[derive(clap::Args, Debug)]
struct Sky {
    /// Where the sun is, as `ELEVATION,AZIMUTH` in degrees.
    ///
    /// Elevation is measured from the horizon and may be negative, which puts
    /// the sun below it; azimuth is a compass bearing, zero north and ninety
    /// east. Without it the sun sits where it always has, 45 degrees up in the
    /// south-east, so every existing invocation draws the frame it drew before.
    // Not a doc comment: clap prints those as help, and this is for whoever
    // edits the flag rather than whoever runs it. `allow_hyphen_values` is what
    // lets a sun below the horizon be written `--sun -3,120` as well as
    // `--sun=-3,120`; without it clap reads the leading minus as the start of
    // another flag and rejects the command. That is a trap worth spending the
    // setting on, because the elevations worth looking at most are the ones
    // near and below zero. The cost is that `--sun` will swallow whatever
    // follows it, which `SunAngles` then rejects for not being two numbers.
    #[arg(long, value_name = "ELEVATION,AZIMUTH", allow_hyphen_values = true)]
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
    /// Where the sun goes once the renderer exists, since the window is built
    /// on `resumed` rather than here and there is no scene to put it in yet.
    sun: Option<SunAngles>,
    renderer: Option<Renderer>,
    controls: FlyController,
    /// When the last frame was drawn, for the timestep the controls integrate over.
    last_frame: Instant,
}

impl App {
    fn new(
        display: OwnedDisplayHandle,
        terrain: PathBuf,
        profiling: bool,
        sun: Option<SunAngles>,
    ) -> Self {
        Self {
            display,
            terrain,
            profiling,
            sun,
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
            self.sun,
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
    let (terrain, profiling, sun) = match arguments.mode {
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
                view.sky.sun,
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
                view.sky.sun,
                headless::Flight {
                    frames,
                    speed: view.motion,
                },
            );
        }
        Mode::Fly { terrain, sky } => (terrain.terrain, false, sky.sun),
        Mode::FlyProfile { terrain, sky } => (terrain.terrain, true, sky.sun),
    };

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(event_loop.owned_display_handle(), terrain, profiling, sun);
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use clap::CommandFactory;

    /// Where a parsed command puts the sun, whichever mode it is.
    ///
    /// Written as one function over all four so that a mode which quietly
    /// stopped carrying `--sun` would fail here rather than be skipped.
    fn sun_of(argv: &[&str]) -> Option<SunAngles> {
        let arguments = Arguments::try_parse_from(argv)
            .unwrap_or_else(|err| panic!("{argv:?} did not parse: {err}"));
        match arguments.mode {
            Mode::Fly { sky, .. } | Mode::FlyProfile { sky, .. } => sky.sun,
            Mode::Render { view, .. } | Mode::Profile { view, .. } => view.sky.sun,
        }
    }

    /// clap's own check that the derived command is well formed -- duplicate
    /// argument ids, a flattened struct colliding with its host, and the like.
    /// Worth running because `Sky` is now flattened into four places, two of
    /// them through `View`, and a collision would otherwise surface as a panic
    /// the first time somebody ran the binary.
    #[test]
    fn the_command_tree_is_well_formed() {
        Arguments::command().debug_assert();
    }

    #[test]
    fn every_mode_takes_the_sun() {
        let angles = SunAngles {
            elevation_degrees: 5.0,
            azimuth_degrees: 120.0,
        };
        assert_eq!(
            sun_of(&["flight-sim", "fly", "-t", "x", "--sun", "5,120"]),
            Some(angles)
        );
        assert_eq!(
            sun_of(&["flight-sim", "fly-profile", "-t", "x", "--sun", "5,120"]),
            Some(angles)
        );
        assert_eq!(
            sun_of(&[
                "flight-sim",
                "render",
                "-t",
                "x",
                "-o",
                "y",
                "--sun",
                "5,120"
            ]),
            Some(angles)
        );
        assert_eq!(
            sun_of(&["flight-sim", "profile", "-t", "x", "--sun", "5,120"]),
            Some(angles)
        );
    }

    /// Left out, the sun stays wherever `Sun::default` puts it, so every
    /// invocation that predates the flag draws the frame it always drew.
    #[test]
    fn leaving_the_sun_out_leaves_it_alone() {
        assert_eq!(sun_of(&["flight-sim", "fly", "-t", "x"]), None);
        assert_eq!(
            sun_of(&["flight-sim", "render", "-t", "x", "-o", "y"]),
            None
        );
    }

    /// The elevations worth looking at are the ones near and below zero, so a
    /// leading minus has to survive both spellings. Without
    /// `allow_hyphen_values` the separated form is rejected as an unknown flag.
    #[test]
    fn a_sun_below_the_horizon_parses_either_way() {
        let dusk = SunAngles {
            elevation_degrees: -3.0,
            azimuth_degrees: 120.0,
        };
        assert_eq!(
            sun_of(&["flight-sim", "fly", "-t", "x", "--sun", "-3,120"]),
            Some(dusk)
        );
        assert_eq!(
            sun_of(&["flight-sim", "fly", "-t", "x", "--sun=-3,120"]),
            Some(dusk)
        );
    }

    /// `allow_hyphen_values` buys the above at the price of `--sun` swallowing
    /// whatever follows it. That is tolerable only because the value still has
    /// to parse as two numbers, so a swallowed flag is an error and not a
    /// silently wrong sun.
    #[test]
    fn the_sun_still_refuses_what_is_not_two_numbers() {
        for bad in ["--motion", "5", "5,120,90", "up,east"] {
            assert!(
                Arguments::try_parse_from([
                    "flight-sim",
                    "render",
                    "-t",
                    "x",
                    "-o",
                    "y",
                    "--sun",
                    bad
                ])
                .is_err(),
                "{bad:?} was accepted as a sun"
            );
        }
    }
}
