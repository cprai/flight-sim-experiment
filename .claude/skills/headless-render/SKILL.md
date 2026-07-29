---
name: headless-render
description: How to render a frame of the terrain to a PNG with no window, look at it, and use it for graphics debugging. Use whenever a change needs to be seen rather than asserted -- shader and clipmap work, "does this actually draw", "show me what it looks like", checking for seams, popping, holes, missing tiles, or wrong colour -- and whenever the app must be run at all in an environment with no display server, which is the normal case here. Covers placing the camera, reading the log the run prints, telling a GPU problem from a data problem, and the traps that make a frame look like nothing changed.
---

# Rendering a frame headless

The windowed app cannot start without a display server: `EventLoop::new` fails
outright when `DISPLAY` and `WAYLAND_DISPLAY` are unset, which they are in this
container. `--screenshot` skips winit entirely, draws into a texture, and writes
a PNG. It is the only way to run the renderer here, and the only way to *see* the
result of a graphics change.

```
cargo build --release
./target/release/flight-sim --terrain assets/terrain --screenshot /tmp/frame.png
```

Then open it with the Read tool, which renders PNGs inline. Look at the frame.
A description of what a shader change ought to do is not evidence that it did.

Always `--release`. Timings from a debug build mean nothing, and the tile
decoding on the way in is unoptimised there.

`assets/terrain` is the pyramid this repo has installed. It is not in version
control; `terrain-download` writes one. Without a pyramid there is nothing to
render and no fallback path.

## Placing the camera

`--camera=x,y,z,yaw,pitch` -- position in metres, angles in degrees.

- World space is right-handed and Y-up: **+X east, +Y up, -Z north** (so +Z is
  south). See the `Camera` doc comment in `src/camera.rs`.
- The origin is the **centre of the raster**, not a corner, so coordinates run
  either side of zero.
- Yaw is a compass heading: 0 faces north, positive turns right/east. Pitch is
  measured from the horizon, positive nose-up, so looking down at ground is
  **negative**. Roll is not exposed.
- Vertical field of view is fixed at 60 degrees (`FOV_Y_DEGREES`).

Work out the bounds from the line the run prints:

```
terrain: 98304 x 114688 texels at 1 m, levels up to 8
```

Half of each side, times the metres per texel, is the coordinate range: here x
spans about ±49 km and z about ±57 km. Anything outside that is off the data and
renders as void.

**A leading minus needs the `=` form.** `--camera -40000,...` is read by clap as
a flag and fails with `unexpected argument '-4' found`. Write
`--camera=-40000,3000,50000,0,-15`.

Without `--camera` you get `Camera::overlooking`: high above the middle of the
southern edge, looking north, pitched to take in the whole extent. Convenient,
and a trap -- see below.

## Reading the log

Every run prints, at `info`:

```
using adapter: AdapterInfo { ..., device_type: DiscreteGpu, backend: Vulkan, ... }
terrain: 98304 x 114688 texels at 1 m, levels up to 8
clipmap: 7 levels of 4096 texels, 1195 MiB of texture, reaching 114688 texels from the camera
built the scene in 9.61ms
camera at [0, 11782.164, 57344] facing Quat(...)
filled the windows in 968.39ms
rendered one frame in 34.15ms
```

- **`device_type`** decides whether any timing below it is worth quoting. If it
  says `Cpu`, the software rasterizer took the job and the numbers describe
  llvm, not the GPU. `WGPU_POWER_PREF` (already `high` in the devcontainer)
  chooses; see ca3791f.
- **`filled the windows`** is disk: tile reads and pyramid reductions. It is
  routinely 20-40x the frame time and says nothing about the shaders. Do not
  report it as render cost.
- **`rendered one frame`** is the draw plus the readback.
- **`camera at ...`** echoes where the view actually ended up, which is how you
  confirm `--camera` parsed the way you meant.

## Comparing two frames

Output is deterministic: the same arguments over the same pyramid produce
byte-identical PNGs across runs. So `cmp` is a real test.

```
./target/release/flight-sim ... -o /tmp/before.png     # stash, then edit
./target/release/flight-sim ... -o /tmp/after.png
cmp -s /tmp/before.png /tmp/after.png && echo "no visible change"
```

`cmp` reporting identical means the change did not reach a single pixel of that
view. That is a genuine finding -- but read the next section before concluding
the code is dead.

## Traps that make a change look like nothing happened

- **The default view frames the whole extent.** It looks at whatever is most of
  the box, so a change confined to one region, one level, or near ground renders
  byte-identical and looks inert. Aim the camera at the thing you changed.
- **Only one frame is drawn, after a single clipmap update.** Anything that
  settles over several frames -- streaming that catches up, hysteresis, an
  incremental update path -- will not appear. Use the `FLIGHT_SIM_WALK` harness
  below for those.
- **`--size` is not a crop.** The clipmap sizes its windows so a texel lands on
  about a pixel, so the viewport changes what is resident: `480x270` gives 9
  levels of 1024 texels and 96 MiB of texture, `1280x720` gives 7 levels of 4096
  and 1195 MiB. A small render is cheap and quick to look at, but it is a
  different clipmap -- never compare frames taken at different sizes, and never
  conclude from a small one that the big one is fine.

## Telling a GPU or driver problem from a shader or data problem

Render the same view on the software rasterizer and compare:

```
WGPU_BACKEND=vulkan VK_DRIVER_FILES=$(ls /usr/share/vulkan/icd.d/lvp*.json) \
  ./target/release/flight-sim --terrain assets/terrain -o /tmp/soft.png
```

Same wrong picture on both means the shader or the data is wrong. Different
pictures point at the driver or at undefined behaviour the hardware happens to
resolve differently. It is slow -- roughly 20-30x the frame time -- so use it to
settle a question, not as a habit.

## Deeper clipmap experiments

`--screenshot` deliberately exposes only camera and size. The knobs for
measuring the clipmap itself live in the ignored `dump_installed_terrain` test
in `src/scene.rs`, which renders through the same `headless::capture` and writes
`terrain.png` into the temp dir:

```
FLIGHT_SIM_TERRAIN=assets/terrain cargo test --release -- --ignored --nocapture dump_installed
```

- `FLIGHT_SIM_CAMERA` -- same `x,y,z,yaw,pitch` as `--camera`.
- `FLIGHT_SIM_NEAR_RINGS` -- how far the rasterized near field reaches before
  the raymarched far field takes over. Infinity rasterizes everything, zero
  raymarches everything; rendering the same view at both is how the two paths
  get checked against each other.
- `FLIGHT_SIM_WINDOW` -- window size in texels, for weighing detail against the
  texture memory and fill time it costs.
- `FLIGHT_SIM_WALK` -- steps the camera in one-metre increments before drawing,
  which exercises the incremental window updates instead of one cold fill.

## Cost

A 1280x720 PNG is about 1 MB, and reading an image into context is not free.
For a quick "did it draw at all", render small. For anything you intend to judge
-- seams, popping, aliasing, colour -- render at the size you actually care
about, because the clipmap differs.
