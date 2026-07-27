//! Writing the result out as a GeoTIFF the simulator can load.
//!
//! Two bands, interleaved: the elevation in metres, and a note of where that
//! elevation came from. The provenance band has to be 32-bit floats like the
//! first, because the `tiff` crate's encoder describes an image with a single
//! sample type -- which is what doubles the file to carry a value that only
//! ever takes three states.
//!
//! The georeferencing tags predate any registry the `tiff` crate knows, so they
//! are written by number, the same way `src/terrain/geotiff.rs` reads them.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use tiff::encoder::colortype::ColorType;
use tiff::encoder::{Compression, DeflateLevel, TiffEncoder};
use tiff::tags::{PhotometricInterpretation, SampleFormat, Tag};

use crate::bbox::OutputGrid;

/// GeoTIFF places a raster with private tags read by number.
const TAG_EXTRA_SAMPLES: u16 = 338;
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
const TAG_GEO_ASCII_PARAMS: u16 = 34737;
const TAG_GDAL_NODATA: u16 = 42113;

/// GeoKeyDirectory entries. Only the ones that change how a texel's ground size
/// is computed matter to the reader, but the datum is recorded too so the file
/// is honest about what it holds.
const KEY_MODEL_TYPE: u16 = 1024;
const KEY_RASTER_TYPE: u16 = 1025;
const KEY_GEOGRAPHIC_TYPE: u16 = 2048;
const KEY_ANGULAR_UNITS: u16 = 2054;

const MODEL_TYPE_GEOGRAPHIC: u16 = 2;
const RASTER_TYPE_PIXEL_IS_AREA: u16 = 1;
const ANGULAR_UNITS_DEGREE: u16 = 9102;

/// NAD83(CSRS) as longitude and latitude.
///
/// Not 4326. The source mosaics are NAD83(CSRS), which differs from WGS 84 by a
/// metre or two in Canada -- small, but not nothing against one-metre pixels,
/// and no shift is applied anywhere in the pipeline. Labelling the output 4326
/// would be claiming a conversion that never happened.
const EPSG_NAD83_CSRS_GEOGRAPHIC: u16 = 4617;

/// Matching the existing assets, which the renderer already reads happily.
const ROWS_PER_STRIP: u32 = 8;

/// Where a pixel's elevation came from, stored in the second band.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provenance {
    /// Neither mosaic had data here; the elevation is the nodata value.
    Missing = 0,
    /// From the one-metre mosaic.
    OneMetre = 1,
    /// From the two-metre mosaic, interpolated onto the output grid.
    TwoMetre = 2,
}

impl Provenance {
    pub fn as_f32(self) -> f32 {
        self as u8 as f32
    }
}

/// A single-band-plus-provenance image of 32-bit floats.
///
/// The `tiff` crate ships colour types for one, three and four samples but not
/// two, and the trait is public and unsealed, so the missing one is declared
/// here. `horizontal_predict` is unreachable because no predictor is used.
pub struct Gray32FloatWithProvenance;

impl ColorType for Gray32FloatWithProvenance {
    type Inner = f32;
    const TIFF_VALUE: PhotometricInterpretation = PhotometricInterpretation::BlackIsZero;
    const BITS_PER_SAMPLE: &'static [u16] = &[32, 32];
    const SAMPLE_FORMAT: &'static [SampleFormat] = &[SampleFormat::IEEEFP; 2];

    fn horizontal_predict(_: &[Self::Inner], _: &mut Vec<Self::Inner>) {
        unreachable!("the elevation raster is written without a predictor")
    }
}

/// Builds the GeoKeyDirectory describing a longitude/latitude raster in degrees
/// on NAD83(CSRS).
///
/// The layout is a four-value header -- version, revision, minor revision, and
/// the number of keys -- followed by four values per key.
fn geo_key_directory() -> Vec<u16> {
    let keys: [(u16, u16); 4] = [
        (KEY_MODEL_TYPE, MODEL_TYPE_GEOGRAPHIC),
        (KEY_RASTER_TYPE, RASTER_TYPE_PIXEL_IS_AREA),
        (KEY_GEOGRAPHIC_TYPE, EPSG_NAD83_CSRS_GEOGRAPHIC),
        (KEY_ANGULAR_UNITS, ANGULAR_UNITS_DEGREE),
    ];

    let mut directory = vec![1, 1, 0, keys.len() as u16];
    for (key, value) in keys {
        // A zero location means the value is held inline, in a count of one.
        directory.extend_from_slice(&[key, 0, 1, value]);
    }
    directory
}

/// Writes the elevation and provenance bands to `path`.
///
/// `samples` is interleaved: elevation, provenance, elevation, provenance, and
/// so on, in row-major order from the north-west corner.
pub fn write_geotiff(path: &Path, grid: &OutputGrid, samples: &[f32], nodata: f32) -> Result<()> {
    let expected = (grid.width as usize)
        .checked_mul(grid.height as usize)
        .and_then(|n| n.checked_mul(2))
        .context("the output raster does not fit in memory")?;
    anyhow::ensure!(
        samples.len() == expected,
        "expected {expected} interleaved samples for a {} x {} raster, got {}",
        grid.width,
        grid.height,
        samples.len()
    );

    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = TiffEncoder::new(BufWriter::new(file))
        .with_context(|| format!("starting {}", path.display()))?
        .with_compression(Compression::Deflate(DeflateLevel::Balanced));

    let mut image = encoder
        .new_image::<Gray32FloatWithProvenance>(grid.width, grid.height)
        .context("starting the image")?;
    image
        .rows_per_strip(ROWS_PER_STRIP)
        .context("setting the strip height")?;

    {
        let directory = image.encoder();

        // Band two is auxiliary, not alpha; without this a reader is entitled
        // to treat it as coverage and composite with it.
        directory
            .write_tag(Tag::Unknown(TAG_EXTRA_SAMPLES), &[0u16][..])
            .context("writing ExtraSamples")?;
        directory
            .write_tag(
                Tag::Unknown(TAG_MODEL_PIXEL_SCALE),
                &[grid.degrees_per_pixel_x, grid.degrees_per_pixel_y, 0.0][..],
            )
            .context("writing ModelPixelScale")?;
        directory
            .write_tag(
                Tag::Unknown(TAG_MODEL_TIEPOINT),
                &[0.0, 0.0, 0.0, grid.box_.west, grid.box_.north, 0.0][..],
            )
            .context("writing ModelTiepoint")?;
        directory
            .write_tag(
                Tag::Unknown(TAG_GEO_KEY_DIRECTORY),
                &geo_key_directory()[..],
            )
            .context("writing the GeoKeyDirectory")?;
        directory
            .write_tag(Tag::Unknown(TAG_GEO_ASCII_PARAMS), "NAD83(CSRS)|")
            .context("writing GeoAsciiParams")?;
        // One value, applied to both bands. Harmless for provenance, which only
        // ever takes 0, 1 or 2.
        directory
            .write_tag(Tag::Unknown(TAG_GDAL_NODATA), format!("{nodata}").as_str())
            .context("writing GDAL_NODATA")?;
    }

    image
        .write_data(samples)
        .with_context(|| format!("writing pixels to {}", path.display()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use tiff::decoder::Decoder;

    use super::*;
    use crate::bbox::{LatLon, LatLonBox};

    fn grid() -> OutputGrid {
        let box_ = LatLonBox::from_corners(
            LatLon {
                latitude: 49.633,
                longitude: -123.307,
            },
            LatLon {
                latitude: 49.637,
                longitude: -123.303,
            },
        )
        .expect("failed to build a box");
        // Coarse, to keep the test raster small.
        OutputGrid::cover(box_, 100.0).expect("failed to cover")
    }

    /// `name` must differ per test: these run in parallel, and two tests
    /// sharing a path race to write and delete the same file.
    fn write_to_temp(name: &str, grid: &OutputGrid, samples: &[f32]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "terrain-download-{}-{name}.tif",
            std::process::id()
        ));
        write_geotiff(&path, grid, samples, -32767.0).expect("failed to write");
        path
    }

    #[test]
    fn the_written_raster_round_trips_with_its_placement_intact() {
        let grid = grid();
        let count = (grid.width as usize) * (grid.height as usize);
        let mut samples = Vec::with_capacity(count * 2);
        for i in 0..count {
            samples.push(100.0 + i as f32);
            samples.push(Provenance::OneMetre.as_f32());
        }

        let path = write_to_temp("round-trip", &grid, &samples);
        let bytes = std::fs::read(&path).expect("failed to read back");
        std::fs::remove_file(&path).ok();

        let mut decoder = Decoder::new(Cursor::new(bytes)).expect("failed to decode");

        assert_eq!(
            decoder.dimensions().expect("no dimensions"),
            (grid.width, grid.height)
        );
        assert_eq!(
            decoder.colortype().expect("no colour type"),
            tiff::ColorType::Multiband {
                bit_depth: 32,
                num_samples: 2
            }
        );

        let scale = decoder
            .get_tag_f64_vec(Tag::Unknown(TAG_MODEL_PIXEL_SCALE))
            .expect("no pixel scale");
        assert!((scale[0] - grid.degrees_per_pixel_x).abs() < 1e-15);
        assert!((scale[1] - grid.degrees_per_pixel_y).abs() < 1e-15);
        assert_eq!(scale[2], 0.0);

        let tiepoint = decoder
            .get_tag_f64_vec(Tag::Unknown(TAG_MODEL_TIEPOINT))
            .expect("no tiepoint");
        assert_eq!(&tiepoint[0..3], &[0.0, 0.0, 0.0]);
        assert!((tiepoint[3] - grid.box_.west).abs() < 1e-15);
        assert!((tiepoint[4] - grid.box_.north).abs() < 1e-15);

        let keys = decoder
            .get_tag_u32_vec(Tag::Unknown(TAG_GEO_KEY_DIRECTORY))
            .expect("no geo keys");
        assert_eq!(keys[0..4], [1, 1, 0, 4]);
        assert_eq!(keys[4..8], [u32::from(KEY_MODEL_TYPE), 0, 1, 2]);
        assert_eq!(keys[8..12], [u32::from(KEY_RASTER_TYPE), 0, 1, 1]);
        assert_eq!(
            keys[12..16],
            [
                u32::from(KEY_GEOGRAPHIC_TYPE),
                0,
                1,
                u32::from(EPSG_NAD83_CSRS_GEOGRAPHIC)
            ]
        );
        assert_eq!(
            keys[16..20],
            [
                u32::from(KEY_ANGULAR_UNITS),
                0,
                1,
                u32::from(ANGULAR_UNITS_DEGREE)
            ]
        );

        assert_eq!(
            decoder
                .get_tag_ascii_string(Tag::Unknown(TAG_GDAL_NODATA))
                .expect("no nodata"),
            "-32767"
        );
        assert_eq!(
            decoder
                .get_tag_u32_vec(Tag::Unknown(TAG_EXTRA_SAMPLES))
                .expect("no extra samples"),
            vec![0]
        );
    }

    /// The simulator reads elevations with `samples.iter().step_by(channels)`.
    /// This is that, spelled out, against a file this module actually produced.
    #[test]
    fn the_first_band_survives_being_read_back() {
        let grid = grid();
        let count = (grid.width as usize) * (grid.height as usize);
        let mut samples = Vec::with_capacity(count * 2);
        for i in 0..count {
            samples.push(100.0 + i as f32);
            samples.push(if i % 3 == 0 {
                Provenance::TwoMetre.as_f32()
            } else {
                Provenance::OneMetre.as_f32()
            });
        }

        let path = write_to_temp("first-band", &grid, &samples);
        let bytes = std::fs::read(&path).expect("failed to read back");
        std::fs::remove_file(&path).ok();

        let mut decoder = Decoder::new(Cursor::new(bytes)).expect("failed to decode");
        let tiff::decoder::DecodingResult::F32(read) =
            decoder.read_image().expect("failed to read the image")
        else {
            panic!("expected 32-bit floats");
        };
        assert_eq!(read.len(), count * 2);

        let elevations: Vec<f32> = read.iter().step_by(2).copied().collect();
        let provenance: Vec<f32> = read.iter().skip(1).step_by(2).copied().collect();
        for i in 0..count {
            assert_eq!(elevations[i], 100.0 + i as f32);
            let expected = if i % 3 == 0 { 2.0 } else { 1.0 };
            assert_eq!(provenance[i], expected, "provenance at {i}");
        }
    }

    #[test]
    fn a_sample_count_that_does_not_match_the_grid_is_refused() {
        let grid = grid();
        let error = write_geotiff(
            &std::env::temp_dir().join("terrain-download-should-not-exist.tif"),
            &grid,
            &[0.0; 4],
            -32767.0,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("interleaved samples"), "{error}");
    }
}
