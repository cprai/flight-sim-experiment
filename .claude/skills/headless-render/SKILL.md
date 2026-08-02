---
name: headless-render
description: How to render a frame of the terrain to a PNG with no window, look at it, and use it for graphics debugging. Use whenever a change needs to be *seen* rather than asserted -- shader and clipmap work, "does this actually draw", "show me what it looks like", checking for seams, popping, holes, missing tiles, or wrong colour -- and whenever the app must be run at all in an environment with no display server, which is the normal case here. Covers placing the camera, reading the log the run prints, telling a GPU problem from a data problem, and the traps that make a frame look like nothing changed. For what a frame *costs* rather than what it looks like, use headless-profile instead: this mode reports no timings at all.
---

# Rendering a frame headless

The windowed app cannot start without a display server: `EventLoop::new` fails
outright when `DISPLAY` and `WAYLAND_DISPLAY` are unset, which they are in this
container. The `render` subcommand skips winit entirely, draws into a texture,
and writes a PNG. It is the only way to run the renderer here, and the only way
to *see* the result of a graphics change.

```
cargo build --release
./target/release/flight-sim render --terrain assets/terrain --output /tmp/frame.png
```

**This mode reports no timings, deliberately.** One cold frame carries first-use
pipeline compilation and whatever the tile reads left behind, so it is an image
and not a measurement. `flight-sim profile` is the mode that answers what a
frame costs -- see the `headless-profile` skill.

Then open it with the Read tool, which renders PNGs inline. Look at the frame.
A description of what a shader change ought to do is not evidence that it did.

Always `--release`. Timings from a debug build mean nothing, and the tile
decoding on the way in is unoptimised there.

`assets/terrain` is the pyramid this repo has installed. It is not in version
control, and it is built in two steps: `terrain-download` fetches elevation and
colour into `assets/download`, then `terrain-process` copies those across to
`assets/terrain` and adds the max pyramid the far field is marched through.
`--terrain` wants the second directory. Without one there is nothing to render
and no fallback path; a tree missing its `dtm-max` product fails at startup
rather than drawing something wrong.

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

Every run prints, at `info` on stderr:

```
using adapter: AdapterInfo { ..., device_type: DiscreteGpu, backend: Vulkan, ... }
terrain: 98304 x 114688 texels at 1 m, levels up to 8
terrain: 8 levels of 8 x 8 tiles, 4096 texels each, 1536 MiB of texture, reaching 196608 texels from the camera
built the scene in 11.35ms
camera at [0, 11782.164, 57344] facing Quat(...)
filled every level in 1.09s
```

Nothing goes to stdout. A `render` run whose stdout is empty is working.

- **`device_type`** decides whether the frame you are looking at came off the
  GPU at all. If it says `Cpu`, the software rasterizer took the job.
  `WGPU_POWER_PREF` (already `high` in the devcontainer) chooses; see ca3791f.
- **`filled every level`** is disk: tile reads and pyramid reductions. It is
  routinely 20-40x the frame time and says nothing about the shaders. Do not
  report it as render cost.
- **The second `terrain:` line** is what the clipmap allocated. It does not vary
  with `--size` -- see the trap below.
- **`camera at ...`** echoes where the view actually ended up, which is how you
  confirm `--camera` parsed the way you meant.

## Comparing two frames

Output is deterministic: the same arguments over the same pyramid produce
byte-identical PNGs across runs. So `cmp` is a real test.

```
./target/release/flight-sim render ... -o /tmp/before.png   # stash, then edit
./target/release/flight-sim render ... -o /tmp/after.png
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
- **`--size` is not a crop.** The viewport's pixel angle decides the *finest*
  level worth filling at a given altitude -- `detail_base` in
  `src/terrain/residency.rs` -- so a small render can be reading coarser ground
  than a big one from the same camera. That is what
  `a_wider_window_reads_finer_ground_at_the_same_distance` in `src/scene.rs`
  pins down. So a small render is cheap and quick to look at, but never compare
  frames taken at different sizes, and never conclude from a small one that the
  big one is fine.

  What `--size` does *not* change is the allocation. `Residency::level_count` is
  a function of the raster and the tile square alone, clamped to the levels the
  pyramid actually has; the viewport never enters it. Every size from `160x90`
  to `2560x1440` reports the same `8 levels of 8 x 8 tiles, 4096 texels each,
  1536 MiB` on this pyramid. If you are looking for a memory or reach effect
  from the viewport, there is not one -- `FLIGHT_SIM_TILES` below is the knob.

## Telling a GPU or driver problem from a shader or data problem

Render the same view on the software rasterizer and compare:

```
WGPU_BACKEND=vulkan VK_DRIVER_FILES=$(ls /usr/share/vulkan/icd.d/lvp*.json) \
  ./target/release/flight-sim render --terrain assets/terrain -o /tmp/soft.png
```

Same wrong picture on both means the shader or the data is wrong. Different
pictures point at the driver or at undefined behaviour the hardware happens to
resolve differently. It is slow -- roughly 20-30x the frame time -- so use it to
settle a question, not as a habit.

## Deeper clipmap experiments

`render` deliberately exposes only camera and size. The knobs for
measuring the clipmap itself live in the ignored `dump_installed_terrain` test
in `src/scene.rs`, which renders through the same `headless::capture` and writes
`terrain.png` into the temp dir:

```
FLIGHT_SIM_TERRAIN=assets/terrain cargo test --release -- --ignored --nocapture dump_installed
```

- `FLIGHT_SIM_TERRAIN` -- the pyramid to open. Required; the test panics
  without it rather than defaulting anywhere.
- `FLIGHT_SIM_CAMERA` -- same `x,y,z,yaw,pitch` as `--camera`.
- `FLIGHT_SIM_TILES` -- overrides `Residency::tiles_across`, a power of two. The
  one knob that trades texture memory for reach, and the only place it is
  reachable at all. This is what to reach for when `--size` did not move the
  clipmap, because it never does.
- `FLIGHT_SIM_WALK` -- steps the camera in one-metre increments before drawing,
  which exercises the incremental window updates instead of one cold fill.

The size is fixed at 960x540 in the test itself. Its `filled every level` line
also reports the finest level that survived `detail_base`, which `render` does
not print. Everything it prints goes to stderr through `eprintln!`, so
`--nocapture` is what reaches it, not `>`.

`FLIGHT_SIM_TILES=4` on the installed pyramid gives `9 levels of 4 x 4 tiles,
2048 texels each, 432 MiB` against the default `8 levels of 8 x 8 tiles, 4096
texels each, 1536 MiB`: a level finer and a third of the memory, reaching
131072 texels instead of 196608.

## Cost

A 1280x720 PNG is about 1 MB, and reading an image into context is not free.
For a quick "did it draw at all", render small. For anything you intend to judge
-- seams, popping, aliasing, colour -- render at the size you actually care
about, because the clipmap differs.
