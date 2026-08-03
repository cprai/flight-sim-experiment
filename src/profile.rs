//! Where a frame's time actually went.
//!
//! Two kinds of measurement, kept apart because they cannot be compared.
//!
//! **GPU**, through [`wgpu_profiler`]: timestamps written at the boundaries of
//! the render passes, read back a frame or two later. This is the only honest
//! account of what the hardware did, and it covers the passes and nothing else.
//! Note in particular that the tile uploads are *not* in it: they go onto wgpu's
//! staging belt through `queue.write_texture` and are flushed outside the
//! encoder these scopes wrap.
//!
//! **CPU**, through plain [`Instant`] spans: the streaming work that the GPU
//! clock is blind to, which is where a frame that stutters usually lost its
//! time -- a tile read is a per-row deflate decode of a whole TIFF, and the
//! heights and maxima are then rewritten texel by texel.
//!
//! Both are off unless a run asked for them. The scopes below are no-ops when
//! [`profiler`] was built disabled, and the CPU spans are an [`Option`] that
//! stays [`None`], so an unprofiled frame does not so much as read the clock.

use std::time::{Duration, Instant};

use wgpu_profiler::{GpuProfiler, GpuProfilerSettings, GpuTimerQueryResult};

/// A duration in milliseconds, and the rate a frame of that length sustains.
pub fn ms_and_fps(frame: Duration) -> (f64, f64) {
    let seconds = frame.as_secs_f64();
    // A frame the clock could not separate from the one before it has no rate.
    // Dividing anyway would print `inf fps`.
    let fps = if seconds > 0.0 { 1.0 / seconds } else { 0.0 };
    (seconds * 1e3, fps)
}

/// The timer features of `adapter` that exist, and none that do not.
///
/// Masking rather than demanding is what keeps a device request that would
/// otherwise fail outright working on an adapter without timestamps;
/// [`wgpu_profiler`] then simply records no time for those scopes rather than
/// failing. Both device sites share this so the two cannot drift apart -- the
/// headless device is documented as asking for exactly what the windowed one
/// does, and every offscreen test goes through it.
pub fn timer_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    adapter.features() & GpuProfiler::ALL_WGPU_TIMER_FEATURES
}

/// A profiler that measures, or one that is entirely inert.
///
/// `enabled` is [`GpuProfilerSettings::enable_timer_queries`], which the crate
/// documents as the way to switch profiling off at runtime: every scope becomes
/// a no-op and no query set is ever allocated. That is what lets [`Scene::draw`]
/// take a profiler unconditionally instead of an [`Option`] threaded through
/// every pass.
///
/// [`Scene::draw`]: crate::scene::Scene::draw
pub fn profiler(device: &wgpu::Device, enabled: bool) -> GpuProfiler {
    GpuProfiler::new(
        device,
        GpuProfilerSettings {
            enable_timer_queries: enabled,
            // Debug groups are emitted even with timers off and cost a driver
            // call per pass. Nothing here is being looked at in RenderDoc.
            enable_debug_groups: false,
            ..Default::default()
        },
    )
    // The only failure is an invalid `max_num_pending_frames`, and the default
    // is not zero.
    .expect("the default profiler settings are valid")
}

/// How long the smoothed values take to close most of the way on a new level.
const TAU: f64 = 0.25;

/// Folds `sample` into `previous`, weighting it by how much time `dt` covers.
///
/// The coefficient comes from the elapsed time rather than being a constant per
/// frame, so a row takes the same quarter second to settle at 30 fps as it does
/// at 300 rather than tracking however often it happens to be sampled.
fn smooth(previous: Option<Duration>, sample: Duration, dt: Duration) -> Duration {
    // Nothing to decay from on the first sample; a fixed start would have to be
    // wrong and would then take TAU to stop being wrong.
    let Some(previous) = previous else {
        return sample;
    };
    let alpha = 1.0 - (-dt.as_secs_f64() / TAU).exp();
    previous.mul_f64(1.0 - alpha) + sample.mul_f64(alpha)
}

/// What the terrain spent bringing tiles in, for one frame.
///
/// Split the way the work splits: deciding, reading, converting, uploading. The
/// tile count belongs beside them because it is what explains the other four --
/// a frame that read four tiles and one that read none differ by a factor no
/// duration on its own accounts for.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Terrain {
    /// Choosing which tiles are wanted. Pure arithmetic, no disk.
    pub advance: Duration,
    /// Pulling tiles off disk, which is a deflate decode per row.
    pub read: Duration,
    /// Exaggerating heights and narrowing maxima to half floats.
    pub convert: Duration,
    /// Handing the bytes to `queue.write_texture`.
    pub write: Duration,
    /// Tiles brought in, counting each level separately.
    pub tiles: u32,
}

/// A stopwatch that only runs when the run asked to be timed.
///
/// Reading the clock is cheap but not free, and the places this is used are
/// inside the per-tile loops. Off, it neither reads the clock nor branches on
/// anything but a [`None`].
#[derive(Clone, Copy)]
pub struct Clock(Option<Instant>);

impl Clock {
    pub fn start(on: bool) -> Self {
        Self(on.then(Instant::now))
    }

    /// How long it has been running, or nothing if it never started.
    pub fn elapsed(self) -> Duration {
        self.0.map(|started| started.elapsed()).unwrap_or_default()
    }
}

/// What the CPU spent on one frame, step by step.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Cpu {
    /// Writing the camera uniform.
    pub camera: Duration,
    pub terrain: Terrain,
    /// Recording the render passes into the encoder.
    pub encode: Duration,
    /// `queue.submit`, which is where the staging belt's copies are flushed.
    pub submit: Duration,
}

/// One labelled duration in the readout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    /// Nesting, for indentation. A child's time is part of its parent's.
    pub depth: usize,
    pub label: String,
    pub value: Duration,
    /// Whether to show the rate this duration works out to.
    pub rate: bool,
}

/// Everything measured about a single frame.
#[derive(Clone, Default, Debug)]
pub struct Frame {
    /// Wall clock since the previous frame started.
    pub interval: Duration,
    pub cpu: Cpu,
    /// GPU scopes, parent before child, as the profiler nested them.
    pub gpu: Vec<Row>,
    /// How the pixels were settled, when a read has come back.
    ///
    /// A count rather than a duration, so it is not a [`Row`]. It also lags
    /// further than the GPU rows do -- see [`crate::reproject::CoverageReader`]
    /// -- and is [`None`] until the first read lands.
    pub coverage: Option<crate::reproject::Coverage>,
    /// Pixels of the target, which is what the coverage is a share *of*.
    ///
    /// Carried rather than derived from the coverage's own sum, so that a sum
    /// which falls short of the screen shows up as shares that do not reach a
    /// hundred instead of being normalised away. Zero where the caller does not
    /// know, which is every headless path.
    pub pixels: u32,
    /// The highest ground anywhere resident, and the camera's own height, both
    /// in metres.
    ///
    /// Together they say whether the one comparison that settles a climbing ray
    /// for free can fire at all: it needs the eye above the ceiling, and the
    /// ceiling is taken across every tile slot rather than across the square
    /// actually in use, so it can sit far above anything on screen.
    pub ceiling: f32,
    pub eye: f32,
}

/// Flattens the profiler's tree into rows, keeping the nesting as `depth`.
///
/// A scope whose feature was unavailable carries no time; it is dropped rather
/// than shown as zero, so a missing measurement never reads as a fast one.
fn flatten(results: &[GpuTimerQueryResult], depth: usize, into: &mut Vec<Row>) {
    for result in results {
        if let Some(time) = &result.time {
            into.push(Row {
                depth,
                label: result.label.clone(),
                value: Duration::from_secs_f64((time.end - time.start).max(0.0)),
                rate: false,
            });
        }
        flatten(&result.nested_queries, depth + 1, into);
    }
}

impl Frame {
    /// Takes the GPU side from a finished profiler frame.
    pub fn take_gpu(&mut self, results: &[GpuTimerQueryResult]) {
        self.gpu.clear();
        flatten(results, 0, &mut self.gpu);
    }

    /// Every row to show, in the order they are drawn or printed.
    pub fn rows(&self) -> Vec<Row> {
        let row = |depth, label: &str, value, rate| Row {
            depth,
            label: label.to_owned(),
            value,
            rate,
        };
        let terrain = &self.cpu.terrain;
        let terrain_total = terrain.advance + terrain.read + terrain.convert + terrain.write;

        let mut rows = vec![row(0, "frame", self.interval, true)];
        // The GPU scopes bring their own labels and nesting -- the outer one is
        // named "gpu" where it is opened -- so they go in as they come. They sit
        // apart from the CPU rows below because the two are different clocks
        // measuring overlapping work, and adding them up would be nonsense.
        rows.extend(self.gpu.iter().cloned());
        rows.push(row(
            0,
            "cpu",
            self.cpu.camera + terrain_total + self.cpu.encode + self.cpu.submit,
            false,
        ));
        rows.push(row(1, "camera", self.cpu.camera, false));
        rows.push(row(1, "terrain", terrain_total, false));
        rows.push(row(2, "advance", terrain.advance, false));
        rows.push(row(2, "read", terrain.read, false));
        rows.push(row(2, "convert", terrain.convert, false));
        rows.push(row(2, "write", terrain.write, false));
        rows.push(row(1, "encode", self.cpu.encode, false));
        rows.push(row(1, "submit", self.cpu.submit, false));
        rows
    }
}

/// Widest label column the rows use, indentation included, plus a space.
const LABEL: usize = 12;

/// One row as text, in fixed-width columns a monospace face lines up.
fn line(row: &Row) -> String {
    let label = format!("{:indent$}{}", "", row.label, indent = row.depth * 2);
    let (ms, fps) = ms_and_fps(row.value);
    if row.rate {
        format!("{label:<LABEL$}{ms:>7.2} ms {fps:>6.1} fps")
    } else {
        format!("{label:<LABEL$}{ms:>7.2} ms")
    }
}

/// The rows of one frame, smoothed towards each new sample.
///
/// Raw per-frame numbers bounce over a wide enough range that the digits blur,
/// and the finer the step the worse it is. The overlay shows these; the
/// headless table does not, because a run of frames can be summarised properly.
#[derive(Default)]
pub struct Smoothed {
    rows: Vec<Row>,
    tiles: u32,
    coverage: Option<crate::reproject::Coverage>,
    pixels: u32,
    ceiling: f32,
    eye: f32,
}

impl Smoothed {
    /// Folds `frame` in, or starts again if its shape changed.
    pub fn update(&mut self, frame: &Frame) {
        let incoming = frame.rows();
        self.tiles = frame.cpu.terrain.tiles;
        // Not smoothed, and not for want of somewhere to keep the state: these
        // are shares of the screen rather than times, and they move only when
        // the flight does. What makes a timing row unreadable raw is that it
        // jitters every frame around a level that is not moving at all.
        self.coverage = frame.coverage;
        self.pixels = frame.pixels;
        self.ceiling = frame.ceiling;
        self.eye = frame.eye;
        // The GPU rows appear a frame or two late and the hud scope comes and
        // goes, so the row set is not fixed. Smoothing across a changed shape
        // would pair a value with the wrong label.
        let same = self.rows.len() == incoming.len()
            && self
                .rows
                .iter()
                .zip(&incoming)
                .all(|(old, new)| old.label == new.label && old.depth == new.depth);
        if !same {
            self.rows = incoming;
            return;
        }
        for (old, new) in self.rows.iter_mut().zip(&incoming) {
            old.value = smooth(Some(old.value), new.value, frame.interval);
        }
    }

    /// The block of text to draw.
    pub fn text(&self) -> String {
        let mut text = String::new();
        for row in &self.rows {
            text.push_str(&line(row));
            text.push('\n');
        }
        text.push_str(&format!("{:<LABEL$}{:>7}", "tiles", self.tiles));
        // What the reprojection is buying, which no timing row can show: the
        // `march` row moves when the shader changes and when the share of the
        // frame handed to it changes, and these are how the two are told apart
        // while flying.
        //
        // Against the pixels on screen, not against the compaction's own sum.
        // Every pixel takes exactly one of the three paths, so the shares reach
        // a hundred when the compaction covered the frame -- and fall short,
        // visibly, when it did not. Normalising by the sum instead would report
        // a healthy-looking hundred percent while most of the screen sat
        // untouched, holding whatever the last pass to reach it left there.
        // The read lags by a frame or two, so the shortfall is expected for a
        // frame or two after a resize; `unaccounted` says how large it is.
        if let Some(coverage) = self.coverage {
            let pixels = f64::from(if self.pixels > 0 {
                self.pixels
            } else {
                coverage.total().max(1)
            });
            let mut share = |label: &str, count: u32| {
                let percent = 100.0 * f64::from(count) / pixels;
                text.push_str(&format!("\n{label:<LABEL$}{percent:>8.1} %"));
            };
            share("reprojected", coverage.reprojected);
            share("sky", coverage.sky);
            share("marched", coverage.marched);
            // Of the marched share, how much of it failed rather than worked.
            // Both are subsets of `marched`, so they do not join the sum.
            share("  abandoned", coverage.abandoned);
            share("  spent", coverage.spent);
            // What the march actually stored, against what it was handed. The
            // two agree or the frame is not being drawn, however plausible it
            // looks -- nothing clears the G-buffer, so pixels the march skipped
            // keep whatever last reached them.
            share("  written", coverage.wrote);
            // Signed, and against the viewport: short means pixels took no
            // path at all, over means the tally is counting more than one
            // frame's worth, which would mean it is not being cleared.
            let unaccounted = i64::from(self.pixels) - i64::from(coverage.total());
            text.push_str(&format!("\n{:<LABEL$}{unaccounted:>7} px", "unaccounted"));
            text.push_str(&format!(
                "\n{:<LABEL$}{:>7} wg",
                "dispatch", coverage.groups
            ));
        }
        // Not a share of anything, so it is written plainly: the eye against
        // the highest resident ground. A climbing ray is settled as sky for
        // free only above the second, and only then does a sky-filled view stop
        // being marched.
        text.push_str(&format!("\n{:<LABEL$}{:>8.0} m", "eye", self.eye));
        text.push_str(&format!("\n{:<LABEL$}{:>8.0} m", "ceiling", self.ceiling));
        text
    }
}

/// The smallest, middle and average of a run of samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stats {
    pub min: Duration,
    pub median: Duration,
    pub mean: Duration,
}

/// Summarises `samples`, which it sorts in place.
///
/// All three, because one number hides what matters: a median far below the
/// mean is a step that mostly costs nothing and occasionally costs a lot, which
/// is the shape a streaming hitch makes and the shape an average erases.
pub fn stats(samples: &mut [Duration]) -> Stats {
    samples.sort_unstable();
    let total: Duration = samples.iter().sum();
    Stats {
        min: samples.first().copied().unwrap_or_default(),
        median: samples.get(samples.len() / 2).copied().unwrap_or_default(),
        mean: total.checked_div(samples.len() as u32).unwrap_or_default(),
    }
}

/// The whole run as a table, one line per step.
///
/// Frames whose row shape differs from the first are dropped rather than
/// misaligned; in practice that is only the leading frames whose GPU results
/// had not come back yet.
pub fn table(frames: &[Frame]) -> String {
    let Some(first) = frames.first() else {
        return "no frames measured".to_owned();
    };
    let shape = first.rows();

    let mut text = format!(
        "{:<LABEL$}{:>10}{:>10}{:>10}\n",
        "step", "min", "median", "mean"
    );
    for (index, row) in shape.iter().enumerate() {
        let mut samples: Vec<Duration> = frames
            .iter()
            .map(|frame| frame.rows())
            .filter(|rows| rows.len() == shape.len() && rows[index].label == row.label)
            .map(|rows| rows[index].value)
            .collect();
        if samples.is_empty() {
            continue;
        }
        let stats = stats(&mut samples);
        let label = format!("{:indent$}{}", "", row.label, indent = row.depth * 2);
        let ms = |value: Duration| ms_and_fps(value).0;
        text.push_str(&format!(
            "{label:<LABEL$}{:>7.2} ms{:>7.2} ms{:>7.2} ms",
            ms(stats.min),
            ms(stats.median),
            ms(stats.mean)
        ));
        if row.rate {
            // On the frame row the median is the one worth converting: a mean
            // frame rate over a run is not a rate anything sustained.
            text.push_str(&format!("  ({:.1} fps median)", ms_and_fps(stats.median).1));
        }
        text.push('\n');
    }

    let tiles: u32 = frames.iter().map(|frame| frame.cpu.terrain.tiles).sum();
    text.push_str(&format!(
        "\n{} frames, {tiles} tile uploads\n",
        frames.len()
    ));
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sixtieth_of_a_second_is_sixty_fps() {
        let (ms, fps) = ms_and_fps(Duration::from_secs_f64(1.0 / 60.0));
        // Loose because a `Duration` is whole nanoseconds and a sixtieth of a
        // second is not one, so the round trip cannot be exact.
        assert!((ms - 16.667).abs() < 1e-3, "{ms}");
        assert!((fps - 60.0).abs() < 1e-3, "{fps}");
    }

    #[test]
    fn an_unmeasurable_frame_has_no_rate() {
        assert_eq!(ms_and_fps(Duration::ZERO), (0.0, 0.0));
    }

    #[test]
    fn the_first_sample_is_taken_whole() {
        let sample = Duration::from_millis(7);
        assert_eq!(smooth(None, sample, sample), sample);
    }

    #[test]
    fn smoothing_moves_towards_the_sample_without_reaching_it() {
        let previous = Duration::from_millis(10);
        let sample = Duration::from_millis(20);
        let smoothed = smooth(Some(previous), sample, Duration::from_millis(16));
        assert!(smoothed > previous && smoothed < sample, "{smoothed:?}");
    }

    /// A long enough step should land nearly on the sample, a short one barely move.
    #[test]
    fn a_longer_step_weighs_the_sample_more_heavily() {
        let previous = Some(Duration::from_millis(10));
        let sample = Duration::from_millis(20);
        let brief = smooth(previous, sample, Duration::from_millis(1));
        let long = smooth(previous, sample, Duration::from_secs(1));
        assert!(brief < long, "{brief:?} {long:?}");
        assert!(long > Duration::from_millis(19), "{long:?}");
    }

    fn frame(interval: u64, read: u64) -> Frame {
        Frame {
            interval: Duration::from_millis(interval),
            cpu: Cpu {
                terrain: Terrain {
                    read: Duration::from_millis(read),
                    tiles: 1,
                    ..Terrain::default()
                },
                ..Cpu::default()
            },
            gpu: Vec::new(),
            coverage: None,
            pixels: 0,
            ceiling: 0.0,
            eye: 0.0,
        }
    }

    #[test]
    fn a_parent_row_totals_its_children() {
        let mut sample = frame(16, 3);
        sample.cpu.terrain.advance = Duration::from_millis(1);
        sample.cpu.camera = Duration::from_millis(2);
        let rows = sample.rows();
        let find = |label: &str| {
            rows.iter()
                .find(|row| row.label == label)
                .expect(label)
                .value
        };
        assert_eq!(find("terrain"), Duration::from_millis(4));
        assert_eq!(find("cpu"), Duration::from_millis(6));
    }

    #[test]
    fn the_columns_line_up_whatever_the_digits() {
        let widths = |sample: &Frame| {
            let mut smoothed = Smoothed::default();
            smoothed.update(sample);
            smoothed.text().lines().map(str::len).collect::<Vec<_>>()
        };
        assert_eq!(widths(&frame(2, 1)), widths(&frame(120, 115)));
    }

    /// The overlay is the only place the reprojection's share is visible while
    /// flying, which is the only place the ground it carries can be *seen*
    /// going stale at the same time.
    #[test]
    fn the_overlay_reports_how_the_pixels_were_settled() {
        let mut sample = frame(16, 4);
        let mut smoothed = Smoothed::default();

        // Nothing until a read has come back, rather than three zeroes that
        // would read as a reprojection carrying nothing at all.
        smoothed.update(&sample);
        assert!(
            !smoothed.text().contains("reprojected"),
            "{}",
            smoothed.text()
        );

        sample.pixels = 1000;
        sample.coverage = Some(crate::reproject::Coverage {
            marched: 200,
            reprojected: 700,
            sky: 100,
            abandoned: 30,
            spent: 20,
            wrote: 200,
            groups: 4,
        });
        smoothed.update(&sample);
        let text = smoothed.text();
        for (label, percent) in [
            ("reprojected", 70.0),
            ("sky", 10.0),
            ("marched", 20.0),
            ("  abandoned", 3.0),
            ("  spent", 2.0),
            ("  written", 20.0),
        ] {
            let expected = format!("{label:<LABEL$}{percent:>8.1} %");
            assert!(text.contains(&expected), "no {expected:?} in {text}");
        }

        // Same width as the timing rows, so the right-aligned block does not
        // grow a third ragged edge.
        let width = |label: &str| {
            text.lines()
                .find(|line| line.starts_with(label))
                .unwrap_or_else(|| panic!("no {label} line in {text}"))
                .len()
        };
        assert_eq!(width("reprojected"), width("cpu"));
        for label in ["unaccounted", "dispatch", "eye", "ceiling"] {
            assert_eq!(width(label), width("cpu"), "{label} line is a ragged width");
        }

        // The three paths are shares of the screen, not of each other, so a
        // compaction that covered only part of it must show as shares that fall
        // short rather than as a tidy hundred percent.
        sample.pixels = 2000;
        smoothed.update(&sample);
        let short = smoothed.text();
        assert!(
            short.contains(&format!("{:<LABEL$}{:>8.1} %", "marched", 10.0)),
            "{short}"
        );
        assert!(
            short.contains(&format!("{:<LABEL$}{:>7} px", "unaccounted", 1000)),
            "{short}"
        );
    }

    #[test]
    fn a_changed_row_shape_restarts_rather_than_mispairing() {
        let mut smoothed = Smoothed::default();
        smoothed.update(&frame(16, 4));
        let mut with_gpu = frame(16, 4);
        with_gpu.gpu = vec![Row {
            depth: 1,
            label: "geometry".to_owned(),
            value: Duration::from_millis(3),
            rate: false,
        }];
        smoothed.update(&with_gpu);
        assert!(smoothed.text().contains("geometry"), "{}", smoothed.text());
    }

    #[test]
    fn stats_report_the_spread_not_just_the_average() {
        let mut samples = [
            Duration::from_millis(1),
            Duration::from_millis(1),
            Duration::from_millis(100),
        ];
        let stats = stats(&mut samples);
        assert_eq!(stats.min, Duration::from_millis(1));
        assert_eq!(stats.median, Duration::from_millis(1));
        assert_eq!(stats.mean, Duration::from_millis(34));
    }

    #[test]
    fn an_empty_run_says_so_rather_than_panicking() {
        assert_eq!(table(&[]), "no frames measured");
    }

    #[test]
    fn the_table_counts_every_frame_and_upload() {
        let text = table(&[frame(16, 4), frame(17, 4)]);
        assert!(text.contains("2 frames, 2 tile uploads"), "{text}");
        assert!(text.contains("read"), "{text}");
    }
}
