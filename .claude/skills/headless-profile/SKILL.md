---
name: headless-profile
description: How to measure what a frame of terrain costs, with no window, and read the per-step breakdown it prints. Use whenever the question is speed rather than appearance -- "is this faster", "why is it slow", "what does the raymarch cost", checking a shader or clipmap change for a regression, or comparing two approaches. Covers the profile subcommand, what each GPU and CPU row means, which numbers are trustworthy, and the things this mode deliberately cannot see. For what a frame looks like rather than what it costs, use headless-render instead.
---

# Measuring a frame headless

```
cargo build --release
./target/release/flight-sim profile --terrain assets/terrain
```

Settles the terrain, throws away a few frames to warm up, then draws 60 and
prints where the time went. Writes no image; the whole output is a table on
stdout.

Always `--release`. A debug build measures rustc's inlining decisions, not the
renderer.

```
step               min    median      mean
frame          1.90 ms   2.41 ms   4.00 ms  (414.8 fps median)
gpu            1.74 ms   1.84 ms   1.84 ms
  reproject    0.23 ms   0.23 ms   0.23 ms
  compact      0.04 ms   0.04 ms   0.04 ms
  args         0.00 ms   0.00 ms   0.00 ms
  march        1.41 ms   1.49 ms   1.51 ms
  risk         0.01 ms   0.01 ms   0.01 ms
  reach        0.01 ms   0.01 ms   0.01 ms
  shading      0.02 ms   0.02 ms   0.02 ms
cpu            0.08 ms   0.11 ms   0.11 ms
  camera       0.00 ms   0.00 ms   0.00 ms
  terrain      0.00 ms   0.00 ms   0.00 ms
    advance    0.00 ms   0.00 ms   0.00 ms
    read       0.00 ms   0.00 ms   0.00 ms
    convert    0.00 ms   0.00 ms   0.00 ms
    write      0.00 ms   0.00 ms   0.00 ms
  encode       0.01 ms   0.01 ms   0.01 ms
  submit       0.07 ms   0.10 ms   0.09 ms

60 frames, 0 tile uploads
921600 pixels: 70.1% reprojected from the last frame, 12.5% sky, 17.4% marched
of which 3546 pixels were abandoned and 0 ran out of steps
```

## Reading it

**`gpu` and `cpu` are different clocks measuring overlapping work. Never add
them together.** The GPU rows come from timestamps the hardware writes at pass
boundaries; the CPU rows come from `Instant` spans around the code that records
and submits. A frame is roughly `max` of the two, not the sum.

- **`frame`** is wall clock between iterations, which here is essentially the
  GPU time plus the blocking poll. Nothing presents, so there is no vsync and
  this is not capped at a refresh rate. The fps beside it is the median
  converted, not an average of rates.
- **`gpu`** wraps every pass of the frame. Its children are where a shader
  change shows up:
  - **`reproject`** scatters the previous frame's ground into this camera, one
    point per pixel. Roughly fixed at a given resolution -- it does the same
    work whatever the scene is -- so a change here means the splat itself
    changed.
  - **`compact`** settles every pixel that scatter answered and lists the rest.
    Cheap, and reads as noise unless the list-building changed.
  - **`args`** turns that list's length into a dispatch size. Three integers;
    it rounds to zero and is only in the table so nothing is unaccounted for.
  - **`march`** is the raymarch itself, and still the bulk of the frame -- 1.49
    of 1.84 ms above. It runs only over the pixels the reprojection left, so it
    moves both when the shader changes *and* when the share of carried pixels
    does. Read it beside the coverage line before concluding a shader edit
    caused it.
  - **`risk`** reduces the motion field to one number per dither cell. A
    hundredth of a millisecond.
  - **`reach`** works out, per cell, the nearest ground that can have swept
    across it since the last frame -- which is what decides whether a carried
    answer, sky or ground, is still the nearest thing along its ray. One thread
    per cell, so also a hundredth of a millisecond, and flat in the resolution
    rather than growing with it.
  - **`shading`** is the deferred resolve: one fullscreen triangle reading the
    G-buffer back. Two hundredths of a millisecond. If a change makes this row
    grow noticeably, that is the finding.
  - **`hud`** appears only in the windowed overlay, never here.
- **The coverage line** under the table is how every pixel of the frame was
  settled. The three shares are the three paths through the compaction and add
  up to 100%:
  - **reprojected** -- a splat from the previous frame landed here, so this
    frame answered it for nothing. This is what the reprojection is judged on.
  - **sky** -- the ceiling test called it sky without casting a ray or
    consulting any history. Free either way, which is why it is kept apart from
    the reprojected share rather than counted with it. Zero whenever the camera
    is below the highest resident peak, because then a ray heading for the
    horizon has to be walked to the end of its budget before it can be called
    sky.
  - **marched** -- what was left, and what the `march` row is the cost of.

  **The second line only appears when there is something wrong to report**, and
  it describes the marched share rather than adding to it:
  - **abandoned** -- rays that gave up instead of answering: the eye inside the
    terrain, or ground that was not resident. They draw as sky, which is not
    what they mean, and they are marched again every frame because a frame that
    did not know is not worth carrying. A few thousand at the edge of the
    raster is ordinary -- the corners of a wide view reach past the loaded
    square. A large share that will not come down means the march cannot see
    ground it is standing over.
  - **ran out of steps** -- rays that exhausted `march_steps` and were painted
    as ground wherever they had got to. This is the one failure that puts
    terrain colour where sky belongs, so a number climbing here is worth
    chasing even while the frame still looks plausible.

  It is read back from the GPU on one extra frame after the measured run, so it
  costs the timings nothing. That frame is one more step along the same flight,
  not a redraw of the last measured one -- a still camera is the reprojection's
  best case and would report the same share whatever `--motion` was asked for.

  A `march` row that moved without the marched share moving is a real shader
  change; both moving together usually means the reprojection's share changed
  instead.

  `fly-profile` draws all of it as rows under `tiles`, so it can be watched
  changing as the camera moves. There the numbers lag further than the timing
  rows -- one read is in flight at a time, so a fresh one lands every few
  frames -- and they are not smoothed, because they are shares of the screen
  rather than times and only move when the flight does. Two rows exist only
  there:
  - **`unaccounted`** -- pixels that took no path at all through the
    compaction, in pixels rather than percent. The three shares are taken
    against the *viewport*, so they reach a hundred only when the compaction
    covered the frame; anything else is a region nothing wrote this frame,
    still holding whatever the last pass to reach it left there. Non-zero for a
    frame or two after a resize, because the read lags the size. Non-zero for
    longer is a bug.
  - **`eye` and `ceiling`** -- the camera's height and the highest ground
    anywhere resident, in metres. The one comparison that settles a climbing
    ray for free needs the first above the second, which is why `sky` reads
    0.0% from any camera below the peaks. The ceiling is taken across every
    tile *slot* rather than across the square in use, so a tile of somewhere
    else that has not been written over yet still counts and it can sit well
    above anything on screen.
- **`cpu`** is the recording side, and on a settled scene it is noise.
  - **`terrain`** and its four children are the tile streaming: `advance`
    decides what is wanted, `read` pulls tiles off disk, `convert` narrows the
    maxima to half floats and rescales the normals, `write` hands the bytes to
    `queue.write_texture`.
  - **`submit`** is usually the largest CPU row because the staging belt's
    copies are flushed there.
- **`tile uploads`** at the bottom is the count across the whole run, and it is
  what explains the `terrain` rows. Zero means nothing streamed.

**Three columns, not one, on purpose.** A median far below the mean is a step
that mostly costs nothing and occasionally costs a lot, which is the shape a
streaming hitch makes and the shape an average erases. Quote the median for
"how fast is it" and the gap between min and mean for "how steady is it".

## What this mode cannot see

- **Streaming cost.** The scene is settled before measuring, so nothing is
  pending and the `terrain` rows read zero. That is the point -- it holds still
  the one variable that would otherwise swamp the others -- but it means this
  cannot answer "what does crossing a tile boundary cost". For that, use
  `fly-profile` and fly, or `FLIGHT_SIM_WALK` in the `dump_installed_terrain`
  harness (see the `headless-render` skill).
- **The tile uploads on the GPU.** They leave through `queue.write_texture` onto
  wgpu's staging belt, not through the command encoder the scopes wrap, so no
  timestamp can reach them. They are inside `submit` on the CPU side and
  unattributed on the GPU side.
- **Anything inside a pass.** There is one dispatch or draw per pass, so
  `march` cannot be split further without splitting the shader.
- **What the reprojection got wrong.** The coverage line says how much of the
  frame was carried rather than marched; it says nothing about whether the
  carried pixels were *right*. Only a picture shows that -- use
  `headless-render` with `--frames` and `--motion`.
- **How coverage varied over the run.** The line is one frame, not an average
  of the sixty. Reading it back per frame would mean waiting on a map inside
  the loop, which would measure the readback rather than the frame.

## Trusting the numbers

- **`device_type` in the adapter line decides whether any of it is worth
  quoting.** If it says `Cpu`, llvmpipe took the job and you are timing LLVM.
  The default view at 1280x720 costs on the order of a hundred milliseconds
  there against under two on the discrete GPU, so a run that landed on the
  wrong adapter is obvious from the magnitude alone.
  `WGPU_POWER_PREF` (already `high` in the devcontainer) chooses.
- **An adapter without timestamp support simply omits the GPU rows** rather than
  reporting zeros, so a missing measurement never reads as a fast one. The
  device asks only for the timer features the adapter has.
- **Compare like with like.** `--size` changes the finest clipmap level worth
  filling, so a measurement at one size says nothing about another. Hold it
  fixed across a before/after.
- **`--frames N`** for a longer or shorter run; 60 is the default and a few more
  are drawn and discarded first, because the first use of a pipeline compiles
  it.
- **`--motion M/S` decides what is being measured, and the default flatters
  it.** With the camera still -- the default, and what every run before this
  option existed did -- every pixel of ground lands back on the pixel it came
  from, so the reprojection carries all it possibly can and the march does only
  the share the dither drops. That is a best case that never happens in flight.
  Flying uncovers ground the previous frame never saw, which has to be marched.
  Quote both, and never compare a still run against a moving one.

## A before/after comparison

```
./target/release/flight-sim profile --terrain assets/terrain \
  --camera=-40000,3000,50000,0,-15 > /tmp/before.txt   # stash, then edit
./target/release/flight-sim profile --terrain assets/terrain \
  --camera=-40000,3000,50000,0,-15 > /tmp/after.txt
diff -u /tmp/before.txt /tmp/after.txt
```

Same camera, same size and same `--motion` both times, or the comparison means
nothing. `--camera` takes the same `x,y,z,yaw,pitch` as `render`, and a leading
minus needs the `=` form -- see the `headless-render` skill for the coordinate
frame and for how to pick a view that actually contains what you changed.

Run it twice, once still and once flying:

```
./target/release/flight-sim profile --terrain assets/terrain   --camera=-40000,3000,50000,0,-15 --motion 50 > /tmp/after-moving.txt
```

Confirm the change reached the pixels too: a shader edit that speeds `march` up
and also changes the image is a different finding from one that does not.
`headless-render` and its `cmp` test are how you tell.

**Take one number from a view, not from the view.** How much the reprojection
can carry depends heavily on what is on screen, so the same build measures very
differently from different cameras -- a view filled with ground carries far
more than one with a lot of sky, because sky that the ceiling test cannot settle
is marched whenever the dither drops it. Two cameras is a much better report
than one.
