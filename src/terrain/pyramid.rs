//! Decoding rasters into a mip pyramid the clipmap can read windows out of.
//!
//! The clipmap draws distant ground on a coarser grid than nearby ground, so it
//! needs the raster pre-filtered at every power-of-two reduction. Coarse levels
//! are tiny -- the whole chain costs about a third more than the base level --
//! but they are what stops distant terrain shimmering as the camera moves.
//!
//! Nothing here is uploaded whole. The GPU only ever sees the small windows the
//! clipmap asks for through [`RasterSource`], which is the boundary that keeps
//! residency independent of how large the dataset grows.

use std::io::{BufReader, Read, Seek};
use std::sync::LazyLock;

use anyhow::{Context, Result, bail};
use glam::{IVec2, UVec2};
use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult, Limits};

use crate::terrain::geotiff::Georeferencing;

/// Whether the colour raster's values are already sRGB-encoded for display.
///
/// Visual satellite products almost always are, and they carry no colour-space
/// tag to say so. If a linear-reflectance source is ever used instead the mips
/// would be filtered twice through the same curve and the terrain would look
/// washed out; flipping this is the whole fix.
const COLOUR_IS_SRGB_ENCODED: bool = true;

/// A rectangular window of texels, at some level of detail.
///
/// Implementors own the data however they like -- in memory, memory-mapped, or
/// fetched from disk on demand -- so long as a window can be produced on
/// request. The clipmap deliberately never asks for a whole level.
pub trait RasterSource {
    /// Number of levels, where 0 is the full-resolution raster.
    fn level_count(&self) -> u32;

    /// Copies `size` texels starting at `origin` into `out`, tightly packed.
    ///
    /// Reads outside the raster clamp to the nearest edge texel, so a window
    /// hanging off the edge of the world repeats its border rather than
    /// producing a hole. `out` must hold `size.x * size.y * texel_bytes()`.
    fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]);
}

/// A texel that knows how to combine with its neighbours to build a coarser mip.
pub trait Texel: Copy + Default + bytemuck::Pod {
    /// Averages the one to four finer texels that a coarse texel covers.
    ///
    /// Fewer than four arrive at a level whose width or height is odd, where
    /// the last column or row has no partner to pair with.
    fn box_filter(samples: &[Self]) -> Self;
}

impl Texel for f32 {
    fn box_filter(samples: &[Self]) -> Self {
        // A plain mean is the right low-pass for a height field: coarse levels
        // become a smoothed surface rather than an aliased subsample of it.
        // Peaks do lose a little height, which is the intended trade -- taking
        // the maximum instead would keep silhouettes but inflate the terrain
        // and break the continuity the morph between levels depends on.
        samples.iter().sum::<f32>() / samples.len() as f32
    }
}

/// A colour texel, sRGB-encoded, in the RGBA order the GPU samples.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Srgb8(pub [u8; 4]);

impl Texel for Srgb8 {
    fn box_filter(samples: &[Self]) -> Self {
        // Averaging sRGB values directly averages the wrong quantity: the
        // encoding is a curve, so the mean of two encoded values is darker than
        // the encoding of their mean. Decode, average, re-encode. It shows most
        // at high-contrast boundaries, which is exactly where terrain is
        // interesting.
        let mut channels = [0.0f32; 4];
        for sample in samples {
            for (accumulated, &value) in channels.iter_mut().zip(sample.0.iter()) {
                *accumulated += srgb_to_linear(value);
            }
        }
        let scale = 1.0 / samples.len() as f32;
        Self(std::array::from_fn(|i| linear_to_srgb(channels[i] * scale)))
    }
}

/// Decoding is a per-texel cost over tens of millions of texels, and there are
/// only 256 possible inputs, so it is worth doing exactly 256 times.
static SRGB_TO_LINEAR: LazyLock<[f32; 256]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let value = i as f32 / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    })
});

fn srgb_to_linear(value: u8) -> f32 {
    SRGB_TO_LINEAR[usize::from(value)]
}

fn linear_to_srgb(value: f32) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

/// One level of a [`Pyramid`].
#[derive(Clone, Debug)]
pub struct Level<T> {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<T>,
}

impl<T: Texel> Level<T> {
    pub fn new(width: u32, height: u32, texels: Vec<T>) -> Self {
        assert_eq!(
            texels.len(),
            (width as usize) * (height as usize),
            "level data does not match its dimensions"
        );
        Self {
            width,
            height,
            texels,
        }
    }

    fn get(&self, x: u32, y: u32) -> T {
        self.texels[(y as usize) * (self.width as usize) + (x as usize)]
    }
}

/// A raster and every power-of-two reduction of it, held in memory.
#[derive(Clone, Debug)]
pub struct Pyramid<T> {
    levels: Vec<Level<T>>,
}

impl<T: Texel> Pyramid<T> {
    /// Builds the reduction chain from a full-resolution level, down to 1x1.
    pub fn build(base: Level<T>) -> Self {
        let mut levels = vec![base];
        while {
            let last = levels.last().expect("a pyramid always has a base level");
            last.width > 1 || last.height > 1
        } {
            levels.push(reduce(levels.last().expect("just checked")));
        }
        Self { levels }
    }

    /// Size of `level` in texels.
    #[cfg(test)]
    pub fn level_size(&self, level: u32) -> UVec2 {
        let level = &self.levels[(level as usize).min(self.levels.len() - 1)];
        UVec2::new(level.width, level.height)
    }

    #[cfg(test)]
    fn level(&self, level: u32) -> &Level<T> {
        &self.levels[level as usize]
    }
}

impl Pyramid<f32> {
    /// The lowest and highest value in the full-resolution level.
    pub fn value_range(&self) -> (f32, f32) {
        self.levels[0]
            .texels
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
    }
}

/// Halves a level, rounding up so a single odd texel still gets a home.
fn reduce<T: Texel>(fine: &Level<T>) -> Level<T> {
    let width = (fine.width.div_ceil(2)).max(1);
    let height = (fine.height.div_ceil(2)).max(1);

    let mut texels = Vec::with_capacity((width as usize) * (height as usize));
    let mut samples = Vec::with_capacity(4);
    for y in 0..height {
        for x in 0..width {
            samples.clear();
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                    if sx < fine.width && sy < fine.height {
                        samples.push(fine.get(sx, sy));
                    }
                }
            }
            texels.push(T::box_filter(&samples));
        }
    }

    Level::new(width, height, texels)
}

impl<T: Texel> RasterSource for Pyramid<T> {
    fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]) {
        let level = &self.levels[(level as usize).min(self.levels.len() - 1)];
        let (max_x, max_y) = (level.width as i32 - 1, level.height as i32 - 1);

        let texels: &mut [T] = bytemuck::cast_slice_mut(
            &mut out[..(size.x as usize) * (size.y as usize) * size_of::<T>()],
        );
        for row in 0..size.y {
            let y = (origin.y + row as i32).clamp(0, max_y) as u32;
            for column in 0..size.x {
                let x = (origin.x + column as i32).clamp(0, max_x) as u32;
                texels[(row as usize) * (size.x as usize) + (column as usize)] = level.get(x, y);
            }
        }
    }
}

/// A raster loaded from disk, with its placement on the ground.
pub struct LoadedRaster<T> {
    pub placement: Georeferencing,
    pub pyramid: Pyramid<T>,
}

/// Loads a single-band raster of elevations, in the file's own units.
pub fn load_heights(path: &str) -> Result<LoadedRaster<f32>> {
    load(path, |samples, channels| {
        // Only the first band carries elevation; some producers pad a height
        // raster out to three identical bands.
        samples.iter().step_by(channels).copied().collect()
    })
    .with_context(|| format!("loading heights from {path}"))
}

/// Loads a colour raster into the sRGB RGBA texels the GPU samples.
pub fn load_colours(path: &str) -> Result<LoadedRaster<Srgb8>> {
    load(path, |samples, channels| {
        samples
            .chunks_exact(channels)
            .map(|texel| {
                let mut rgba = [255u8; 4];
                for (out, &value) in rgba.iter_mut().zip(texel.iter()).take(3) {
                    let unit = value.clamp(0.0, 1.0);
                    *out = if COLOUR_IS_SRGB_ENCODED {
                        (unit * 255.0).round() as u8
                    } else {
                        linear_to_srgb(unit)
                    };
                }
                // A single-band source is grey, so spread it across RGB rather
                // than leaving green and blue at zero.
                if channels == 1 {
                    rgba[1] = rgba[0];
                    rgba[2] = rgba[0];
                }
                Srgb8(rgba)
            })
            .collect()
    })
    .with_context(|| format!("loading colours from {path}"))
}

/// Decodes a raster strip by strip, converting each with `convert`.
///
/// Going strip by strip rather than decoding the whole image keeps only one
/// strip of intermediate `f32` samples alive at a time, which for a wide raster
/// is the difference between a hundred kilobytes and hundreds of megabytes.
fn load<T: Texel>(
    path: &str,
    convert: impl Fn(&[f32], usize) -> Vec<T>,
) -> Result<LoadedRaster<T>> {
    let file = std::fs::File::open(path).context("opening raster")?;
    let mut decoder = Decoder::new(BufReader::new(file))
        .context("reading TIFF header")?
        // The default limit is a guard against hostile files; these are ours.
        .with_limits(Limits::unlimited());

    let placement = Georeferencing::read(&mut decoder)?;
    let (width, height) = (placement.width, placement.height);
    let channels = channel_count(&mut decoder)?;

    let mut texels = Vec::with_capacity((width as usize) * (height as usize));
    let strips = decoder.strip_count().context("counting strips")?;
    for strip in 0..strips {
        let (strip_width, strip_height) = decoder.chunk_data_dimensions(strip);
        let samples = to_f32(
            decoder
                .read_chunk(strip)
                .with_context(|| format!("decoding strip {strip}"))?,
        )?;

        let expected = (strip_width as usize) * (strip_height as usize) * channels;
        if samples.len() < expected {
            bail!(
                "strip {strip} decoded to {} samples, expected {expected}",
                samples.len()
            );
        }
        texels.extend(convert(&samples[..expected], channels));
    }

    let expected = (width as usize) * (height as usize);
    if texels.len() != expected {
        bail!(
            "raster decoded to {} texels, expected {expected}",
            texels.len()
        );
    }

    Ok(LoadedRaster {
        placement,
        pyramid: Pyramid::build(Level::new(width, height, texels)),
    })
}

fn channel_count<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<usize> {
    Ok(match decoder.colortype().context("reading colour type")? {
        ColorType::Gray(_) => 1,
        ColorType::RGB(_) => 3,
        ColorType::RGBA(_) | ColorType::CMYK(_) => 4,
        // Anything the named types do not cover, which for a height field is
        // usually elevation plus something alongside it -- `terrain-download`
        // writes a second band recording which source each texel came from.
        // Only band zero is read, so the extras cost nothing but their bytes.
        // Zero is rejected because the callers stride by this, and a stride of
        // zero panics rather than failing cleanly.
        ColorType::Multiband { num_samples: 0, .. } => {
            bail!("raster claims to have no bands at all")
        }
        ColorType::Multiband { num_samples, .. } => usize::from(num_samples),
        other => bail!("raster has unsupported colour type {other:?}"),
    })
}

/// Normalizes a decoded strip to `f32`.
///
/// Integer rasters are scaled to 0..1 so colour data reads the same whatever
/// its bit depth, while float rasters pass through untouched because their
/// values are already meaningful -- metres, for a height field.
fn to_f32(result: DecodingResult) -> Result<Vec<f32>> {
    Ok(match result {
        DecodingResult::F32(values) => values,
        DecodingResult::F64(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::U8(values) => values.into_iter().map(|v| f32::from(v) / 255.0).collect(),
        DecodingResult::U16(values) => values
            .into_iter()
            .map(|v| f32::from(v) / f32::from(u16::MAX))
            .collect(),
        DecodingResult::I16(values) => values.into_iter().map(f32::from).collect(),
        DecodingResult::I32(values) => values.into_iter().map(|v| v as f32).collect(),
        DecodingResult::U32(values) => values.into_iter().map(|v| v as f32).collect(),
        other => bail!("raster has an unsupported sample type: {other:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: u32, height: u32) -> Level<f32> {
        let texels = (0..width * height).map(|i| i as f32).collect();
        Level::new(width, height, texels)
    }

    #[test]
    fn a_pyramid_halves_until_it_reaches_a_single_texel() {
        let pyramid = Pyramid::build(ramp(8, 4));

        let sizes: Vec<_> = (0..pyramid.level_count())
            .map(|l| pyramid.level_size(l).to_array())
            .collect();
        assert_eq!(sizes, [[8, 4], [4, 2], [2, 1], [1, 1]]);
    }

    #[test]
    fn odd_dimensions_round_up_and_still_reach_one() {
        let pyramid = Pyramid::build(ramp(5, 3));

        let sizes: Vec<_> = (0..pyramid.level_count())
            .map(|l| pyramid.level_size(l).to_array())
            .collect();
        assert_eq!(sizes, [[5, 3], [3, 2], [2, 1], [1, 1]]);
    }

    #[test]
    fn a_coarse_texel_is_the_mean_of_the_four_it_covers() {
        // 2x2 blocks of a row-major ramp: [0 1 / 4 5] averages to 2.5.
        let pyramid = Pyramid::build(ramp(4, 4));
        assert_eq!(pyramid.level(1).get(0, 0), (0.0 + 1.0 + 4.0 + 5.0) / 4.0);
        assert_eq!(
            pyramid.level(1).get(1, 1),
            (10.0 + 11.0 + 14.0 + 15.0) / 4.0
        );
    }

    #[test]
    fn an_odd_edge_averages_only_the_texels_that_exist() {
        // Width 3 leaves the last coarse column covering a single fine column,
        // so it must not be dragged towards zero by a phantom partner.
        let pyramid = Pyramid::build(ramp(3, 2));
        assert_eq!(pyramid.level(1).get(1, 0), (2.0 + 5.0) / 2.0);
    }

    #[test]
    fn colour_mips_are_filtered_in_linear_space() {
        let black = Srgb8([0, 0, 0, 255]);
        let white = Srgb8([255, 255, 255, 255]);
        let checker = Level::new(2, 2, vec![black, white, white, black]);

        let averaged = Pyramid::build(checker).level(1).get(0, 0);

        // Half the light, not half the encoded value: mid-grey is 188, not 128.
        // Averaging in sRGB space here would visibly darken every coastline and
        // snow boundary in the coarse levels.
        assert_eq!(averaged.0[0], 188, "expected a linear-space average");
    }

    #[test]
    fn reading_past_an_edge_repeats_the_border_texel() {
        let pyramid = Pyramid::build(ramp(4, 4));

        let mut out = vec![0u8; 3 * 3 * size_of::<f32>()];
        pyramid.read_rect(0, IVec2::new(-2, -2), UVec2::new(3, 3), &mut out);

        // Every sample of a window entirely off the top-left corner clamps to
        // the corner texel, so a window hanging off the world has no holes.
        let values: &[f32] = bytemuck::cast_slice(&out);
        assert_eq!(values, &[0.0; 9]);
    }

    #[test]
    fn a_window_reads_the_texels_it_asks_for() {
        let pyramid = Pyramid::build(ramp(4, 4));

        let mut out = vec![0u8; 2 * 2 * size_of::<f32>()];
        pyramid.read_rect(0, IVec2::new(1, 2), UVec2::new(2, 2), &mut out);

        let values: &[f32] = bytemuck::cast_slice(&out);
        assert_eq!(values, &[9.0, 10.0, 13.0, 14.0]);
    }

    #[test]
    fn the_value_range_spans_the_full_resolution_level() {
        let pyramid = Pyramid::build(Level::new(2, 2, vec![-7.5f32, 0.0, 3.0, 12.25]));
        assert_eq!(pyramid.value_range(), (-7.5, 12.25));
    }

    /// A height raster carrying a second band alongside the elevation, which is
    /// the shape `terrain-download` writes: band zero is the height in metres,
    /// band one records which source it came from.
    ///
    /// The `tiff` crate has no two-sample colour type, but the trait is public,
    /// so the missing one is declared here the same way the downloader declares
    /// it. Both sides being spelled out separately is deliberate: this test is
    /// checking that the loader accepts the format, not that the two crates
    /// share a definition.
    struct Gray32FloatPlusOne;

    impl tiff::encoder::colortype::ColorType for Gray32FloatPlusOne {
        type Inner = f32;
        const TIFF_VALUE: tiff::tags::PhotometricInterpretation =
            tiff::tags::PhotometricInterpretation::BlackIsZero;
        const BITS_PER_SAMPLE: &'static [u16] = &[32, 32];
        const SAMPLE_FORMAT: &'static [tiff::tags::SampleFormat] =
            &[tiff::tags::SampleFormat::IEEEFP; 2];

        fn horizontal_predict(_: &[f32], _: &mut Vec<f32>) {
            unreachable!("written without a predictor")
        }
    }

    fn two_band_geotiff(width: u32, height: u32) -> Vec<u8> {
        use std::io::Cursor;

        use tiff::encoder::TiffEncoder;
        use tiff::tags::Tag;

        let mut buffer = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buffer).expect("failed to start encoding");
            let mut image = encoder
                .new_image::<Gray32FloatPlusOne>(width, height)
                .expect("failed to start image");
            image.rows_per_strip(2).expect("failed to set strip height");
            {
                let directory = image.encoder();
                directory
                    .write_tag(Tag::Unknown(33550), &[0.001f64, 0.002, 0.0][..])
                    .expect("failed to write pixel scale");
                directory
                    .write_tag(
                        Tag::Unknown(33922),
                        &[0.0f64, 0.0, 0.0, 10.0, 45.0, 0.0][..],
                    )
                    .expect("failed to write tiepoint");
                // Geographic, area pixels, degrees.
                directory
                    .write_tag(
                        Tag::Unknown(34735),
                        &[
                            1u16, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2054, 0, 1, 9102,
                        ][..],
                    )
                    .expect("failed to write geo keys");
            }

            let mut samples = Vec::with_capacity((width * height * 2) as usize);
            for i in 0..width * height {
                samples.push(100.0 + i as f32);
                samples.push(1.0);
            }
            image.write_data(&samples).expect("failed to write pixels");
        }
        buffer.into_inner()
    }

    #[test]
    fn a_height_raster_with_a_second_band_loads_its_elevations() {
        let (width, height) = (4u32, 4u32);
        let path =
            std::env::temp_dir().join(format!("flight-sim-two-band-{}.tiff", std::process::id()));
        std::fs::write(&path, two_band_geotiff(width, height)).expect("failed to write");

        let loaded = load_heights(path.to_str().expect("non-UTF-8 temp path"))
            .expect("a two-band height raster should load");
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.placement.width, width);
        assert_eq!(loaded.placement.height, height);

        // Band one is dropped; only the elevations survive.
        let level = loaded.pyramid.level(0);
        for y in 0..height {
            for x in 0..width {
                let expected = 100.0 + (y * width + x) as f32;
                assert_eq!(level.get(x, y), expected, "texel ({x}, {y})");
            }
        }
    }

    /// Decodes the rasters actually on disk, end to end.
    ///
    /// Ignored because the assets are not in version control. Run it with
    /// `cargo test -- --ignored --nocapture` after swapping the data; the
    /// figures are printed rather than asserted so this stays independent of
    /// whichever dataset is installed.
    #[test]
    #[ignore = "requires the raster assets, which are not in version control"]
    fn the_installed_rasters_decode_into_pyramids() {
        let started = std::time::Instant::now();
        let heights = load_heights(crate::terrain::HEIGHT_RASTER_PATH).expect("failed to load");
        let colours = load_colours(crate::terrain::COLOUR_RASTER_PATH).expect("failed to load");
        eprintln!("decoded both rasters in {:.2?}", started.elapsed());

        let (lowest, highest) = heights.pyramid.value_range();
        eprintln!(
            "heights: {} levels, {lowest:.3} .. {highest:.3}",
            heights.pyramid.level_count()
        );
        eprintln!("colours: {} levels", colours.pyramid.level_count());

        assert!(
            lowest.is_finite() && highest.is_finite(),
            "heights must be finite"
        );
        assert!(highest > lowest, "a height field should vary");

        // Both rasters cover the same ground, so they must reduce identically.
        assert_eq!(heights.pyramid.level_count(), colours.pyramid.level_count());
        for level in 0..heights.pyramid.level_count() {
            assert_eq!(
                heights.pyramid.level_size(level),
                colours.pyramid.level_size(level),
                "level {level} differs between the rasters"
            );
        }

        // The coarsest level is a single texel, which is what lets the outermost
        // clipmap ring cover the whole dataset however large it is.
        let coarsest = heights.pyramid.level_count() - 1;
        assert_eq!(heights.pyramid.level_size(coarsest), UVec2::ONE);
    }
}
