//! Drawing a block of text over the corner of the window.
//!
//! Only the drawing. What the text says, and what it was measured with, is
//! [`crate::profile`]; this turns a string into glyphs and nothing more, so the
//! same readout can be printed by a run that has no window to draw it in.

use glam::UVec2;
use glyphon::cosmic_text::Align;
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport,
};

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
