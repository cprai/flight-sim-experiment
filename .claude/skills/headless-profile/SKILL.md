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
frame          5.04 ms   5.25 ms   5.32 ms  (190.6 fps median)
gpu            4.93 ms   5.10 ms   5.10 ms
  geometry     4.90 ms   5.06 ms   5.06 ms
  shading      0.02 ms   0.02 ms   0.02 ms
cpu            0.05 ms   0.06 ms   0.07 ms
  camera       0.00 ms   0.00 ms   0.00 ms
  terrain      0.00 ms   0.00 ms   0.00 ms
    advance    0.00 ms   0.00 ms   0.00 ms
    read       0.00 ms   0.00 ms   0.00 ms
    convert    0.00 ms   0.00 ms   0.00 ms
    write      0.00 ms   0.00 ms   0.00 ms
  encode       0.00 ms   0.01 ms   0.01 ms
  submit       0.04 ms   0.05 ms   0.06 ms

60 frames, 0 tile uploads
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
- **`gpu`** wraps both render passes. Its children are where a shader change
  shows up:
  - **`geometry`** is the raymarch into the G-buffer. It is essentially the
    whole frame -- 5.06 of 5.10 ms above -- because it is one fullscreen
    triangle whose fragment shader walks the clipmap. Any terrain shader change
    lands here.
  - **`shading`** is the deferred resolve: one more fullscreen triangle reading
    the G-buffer back. Two hundredths of a millisecond. If a change makes this
    row grow noticeably, that is the finding.
  - **`hud`** appears only in the windowed overlay, never here.
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
- **Anything inside a pass.** There is one draw call per pass, and the near
  field and far field are branches of one fragment shader rather than separate
  pipelines, so `geometry` cannot be split further without splitting the shader.

## Trusting the numbers

- **`device_type` in the adapter line decides whether any of it is worth
  quoting.** If it says `Cpu`, llvmpipe took the job and you are timing LLVM.
  The default view at 1280x720 costs about 189 ms there against 5 ms on the
  discrete GPU -- a factor of 36, so a run that landed on the wrong adapter is
  obvious from the magnitude alone.
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

## A before/after comparison

```
./target/release/flight-sim profile --terrain assets/terrain \
  --camera=-40000,3000,50000,0,-15 > /tmp/before.txt   # stash, then edit
./target/release/flight-sim profile --terrain assets/terrain \
  --camera=-40000,3000,50000,0,-15 > /tmp/after.txt
diff -u /tmp/before.txt /tmp/after.txt
```

Same camera and same size both times, or the comparison means nothing.
`--camera` takes the same `x,y,z,yaw,pitch` as `render`, and a leading minus
needs the `=` form -- see the `headless-render` skill for the coordinate frame
and for how to pick a view that actually contains what you changed.

Confirm the change reached the pixels too: a shader edit that speeds `geometry`
up and also changes the image is a different finding from one that does not.
`headless-render` and its `cmp` test are how you tell.
