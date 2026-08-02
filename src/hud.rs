//! The frame time readout drawn over the corner of the window.
//!
//! Two numbers, because one is not enough to say anything. The swapchain is
//! configured [`wgpu::PresentMode::AutoVsync`], so the interval between frames
//! sits on the refresh rate whenever the renderer has any headroom at all and
//! reads 60 fps whether a frame cost one millisecond or fifteen. That number is
//! still the one worth showing -- it is what the window actually does -- but on
//! its own it hides every change until the moment the renderer falls off the
//! refresh rate entirely. The time spent recording and submitting the frame,
//! measured with the wait for the swapchain image left outside it, is what moves
//! when the work changes.
//!
//! Neither is shown raw. A frame time bounces over a wide enough range that the
//! digits blur, so both are smoothed towards the incoming samples.

use std::time::{Duration, Instant};

use glam::UVec2;
use glyphon::cosmic_text::Align;
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

/// A duration in milliseconds, and the rate a frame of that length sustains.
///
/// Shared with [`crate::headless`], which has no screen to draw on and prints
/// the same pair instead.
pub fn ms_and_fps(frame: Duration) -> (f64, f64) {
    let seconds = frame.as_secs_f64();
    // A frame the clock could not separate from the one before it has no rate.
    // Dividing anyway would print `inf fps`.
    let fps = if seconds > 0.0 { 1.0 / seconds } else { 0.0 };
    (seconds * 1e3, fps)
}

/// How long the smoothed values take to close most of the way on a new level.
///
/// A quarter second is about the shortest that holds the hundredths digit still
/// enough to read while still settling fast enough to feel like it belongs to
/// what is on screen.
const TAU: f64 = 0.25;

/// Folds `sample` into `previous`, weighting it by how much time `dt` covers.
///
/// The coefficient comes from the elapsed time rather than being a constant per
/// frame, so the readout takes the same quarter second to settle at 30 fps as it
/// does at 300 rather than tracking however often it happens to be sampled.
fn smooth(previous: Option<Duration>, sample: Duration, dt: Duration) -> Duration {
    // Nothing to decay from on the first sample; a fixed start would have to be
    // wrong and would then take TAU to stop being wrong.
    let Some(previous) = previous else {
        return sample;
    };
    let alpha = 1.0 - (-dt.as_secs_f64() / TAU).exp();
    previous.mul_f64(1.0 - alpha) + sample.mul_f64(alpha)
}

/// The smoothed frame timings, and the text they read as.
#[derive(Default)]
pub struct FrameTimer {
    /// When the last frame began, for the interval to the next one.
    last_frame: Option<Instant>,
    /// That interval unsmoothed, which is the timestep the averages weight by.
    interval: Option<Duration>,
    frame: Option<Duration>,
    cpu: Option<Duration>,
}

impl FrameTimer {
    /// Opens a frame at `now`, taking the interval since the last one.
    pub fn begin(&mut self, now: Instant) {
        let Some(previous) = self.last_frame.replace(now) else {
            return;
        };
        let interval = now.duration_since(previous);
        self.interval = Some(interval);
        self.frame = Some(smooth(self.frame, interval, interval));
    }

    /// Closes it with the time taken to record and submit its commands.
    ///
    /// Weighted by the frame interval rather than by `cpu` itself: the
    /// coefficient stands for how much wall clock has gone by since the last
    /// sample, and a frame that was cheap to record still took a whole frame.
    pub fn end(&mut self, cpu: Duration) {
        self.cpu = Some(smooth(self.cpu, cpu, self.interval.unwrap_or(cpu)));
    }

    /// The two lines to draw, in fixed-width columns.
    ///
    /// The widths are what keep the readout from twitching sideways as digits
    /// come and go; a monospace face makes them line up.
    pub fn text(&self) -> String {
        format!(
            "{}\n{}",
            Self::line("frame", self.frame),
            Self::line("cpu", self.cpu)
        )
    }

    fn line(label: &str, value: Option<Duration>) -> String {
        // Before the second frame there is no interval to report, and an
        // unmeasured value should look unmeasured rather than like a zero.
        let Some(value) = value else {
            return format!("{label:<5} {:>6} ms {:>6} fps", "--", "--");
        };
        let (ms, fps) = ms_and_fps(value);
        format!("{label:<5} {ms:>6.2} ms {fps:>6.1} fps")
    }
}

/// Gap from the top and right edges of the window, in logical pixels.
const MARGIN: f32 = 12.0;
/// Text size and line spacing, in logical pixels.
const FONT_SIZE: f32 = 14.0;
const LINE_HEIGHT: f32 = 18.0;

/// White text over a black shadow, so the readout survives any background.
const TEXT: Color = Color::rgb(255, 255, 255);
const SHADOW: Color = Color::rgb(0, 0, 0);

/// The glyph atlas and text pipeline that draw [`FrameTimer::text`].
pub struct Hud {
    font_system: FontSystem,
    swash_cache: SwashCache,
    viewport: Viewport,
    atlas: TextAtlas,
    renderer: TextRenderer,
    buffer: Buffer,
    /// What the buffer currently holds, so it is only reshaped when it changed.
    shaped: Option<Shaped>,
}

/// The inputs a shaped buffer is still valid for.
struct Shaped {
    text: String,
    resolution: UVec2,
    scale_factor: f32,
}

/// The surface the overlay is going onto, and how big it is.
///
/// `resolution` is in physical pixels, as [`Viewport`] and the render pass both
/// want; `scale_factor` is what turns the sizes below into those pixels, so the
/// readout stays one physical size across displays of different density.
pub struct Target<'a> {
    pub view: &'a wgpu::TextureView,
    pub resolution: UVec2,
    pub scale_factor: f32,
}

fn metrics(scale_factor: f32) -> Metrics {
    Metrics::new(FONT_SIZE * scale_factor, LINE_HEIGHT * scale_factor)
}

impl Hud {
    /// Builds the overlay for a target of `format`, or [`None`] with no fonts.
    ///
    /// `cosmic-text` does not fall back to blank text when its font database is
    /// empty -- it panics out of shaping with "no default font found" -- and a
    /// container with no fontconfig installed is exactly that case. Declining to
    /// build here leaves the renderer as it was rather than taking the window
    /// down over a debug readout.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        scale_factor: f32,
    ) -> Option<Self> {
        let mut font_system = FontSystem::new();
        if font_system.db_mut().is_empty() {
            log::warn!("no system fonts found; the frame time overlay is disabled");
            return None;
        }

        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        // The default `ColorMode::Accurate` keeps colours in sRGB, which is what
        // the sRGB surface format `Renderer::new` picks out wants.
        let mut atlas = TextAtlas::new(device, queue, &cache, format);
        let renderer =
            TextRenderer::new(&mut atlas, device, wgpu::MultisampleState::default(), None);
        let buffer = Buffer::new(&mut font_system, metrics(scale_factor));

        Some(Self {
            font_system,
            swash_cache,
            viewport,
            atlas,
            renderer,
            buffer,
            shaped: None,
        })
    }

    /// Records a pass drawing `text` into the top right corner of `view`.
    ///
    /// Loads rather than clears: the shading pass has already written every
    /// pixel and this goes over it.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: Target,
        text: &str,
    ) {
        let Target {
            view,
            resolution,
            scale_factor,
        } = target;

        // The frame these glyphs were rasterized for has been submitted, so
        // whatever of it is still held in the atlas can go before this frame
        // asks for more room.
        self.atlas.trim();

        let stale = self.shaped.as_ref().is_none_or(|shaped| {
            shaped.text != text
                || shaped.resolution != resolution
                || shaped.scale_factor != scale_factor
        });
        if stale {
            self.shape(resolution, scale_factor, text);
        }

        self.viewport.update(
            queue,
            Resolution {
                width: resolution.x,
                height: resolution.y,
            },
        );

        let bounds = TextBounds {
            left: 0,
            top: 0,
            right: resolution.x as i32,
            bottom: resolution.y as i32,
        };
        let top = MARGIN * scale_factor;
        // Terrain runs from pale rock to near-black shadow and the sky above it
        // is bright, so a single colour is unreadable somewhere in most frames.
        // A copy offset a pixel underneath is an outline for one more draw.
        let offset = scale_factor.max(1.0).round();
        let areas = [
            TextArea {
                buffer: &self.buffer,
                left: offset,
                top: top + offset,
                scale: 1.0,
                bounds,
                default_color: SHADOW,
                custom_glyphs: &[],
            },
            TextArea {
                buffer: &self.buffer,
                left: 0.0,
                top,
                scale: 1.0,
                bounds,
                default_color: TEXT,
                custom_glyphs: &[],
            },
        ];

        // A full atlas is the documented failure here. Losing the readout for a
        // frame is not worth ending the run over.
        if let Err(err) = self.renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.atlas,
            &self.viewport,
            areas,
            &mut self.swash_cache,
        ) {
            log::warn!("failed to prepare the frame time overlay: {err}");
            return;
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let Err(err) = self.renderer.render(&self.atlas, &self.viewport, &mut pass) {
            log::warn!("failed to draw the frame time overlay: {err}");
        }
    }

    /// Lays the text out again, right against the inside of the margin.
    ///
    /// Aligning right inside a buffer narrower than the window by the margin is
    /// what puts the readout in the corner: the alternative is measuring the
    /// laid-out width and subtracting it, which needs the shaping done first.
    fn shape(&mut self, resolution: UVec2, scale_factor: f32, text: &str) {
        let margin = MARGIN * scale_factor;
        self.buffer.set_metrics_and_size(
            metrics(scale_factor),
            Some(resolution.x as f32 - margin),
            Some(resolution.y as f32),
        );
        self.buffer.set_text(
            text,
            &Attrs::new().family(Family::Monospace),
            // The readout is digits and latin labels in a font this chose, so
            // there is nothing for fallback or complex shaping to do, and this
            // runs inside the very frame it is timing.
            Shaping::Basic,
            Some(Align::Right),
        );
        self.buffer.shape_until_scroll(&mut self.font_system, false);
        self.shaped = Some(Shaped {
            text: text.to_owned(),
            resolution,
            scale_factor,
        });
    }
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

    #[test]
    fn the_columns_line_up_whatever_the_digits() {
        let short = FrameTimer {
            frame: Some(Duration::from_micros(1500)),
            cpu: Some(Duration::from_micros(200)),
            ..FrameTimer::default()
        };
        let long = FrameTimer {
            frame: Some(Duration::from_millis(120)),
            cpu: Some(Duration::from_millis(115)),
            ..FrameTimer::default()
        };
        let widths = |timer: &FrameTimer| timer.text().lines().map(str::len).collect::<Vec<_>>();
        assert_eq!(widths(&short), widths(&long));
        assert_eq!(widths(&short), widths(&FrameTimer::default()));
    }

    #[test]
    fn an_untimed_frame_reads_as_unmeasured() {
        let text = FrameTimer::default().text();
        assert!(text.contains("frame"), "{text}");
        assert!(text.contains("--"), "{text}");
    }
}
