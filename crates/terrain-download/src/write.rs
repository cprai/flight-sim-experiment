//! Writing one tile of the pyramid out as a GeoTIFF.
//!
//! Every tile is a complete, self-describing GeoTIFF of [`TILE_SIZE`] squared
//! texels, placed by its own tiepoint. Nothing about a tile depends on the box
//! that was downloaded, so two runs over neighbouring ground produce files that
//! sit together on the same lattice.
//!
//! Tiles are written **uncompressed, one row per strip**. That is the whole
//! reason the renderer can read them synchronously in the middle of a frame: a
//! clipmap window moves by a thin strip at a time, and reading a strip out of a
//! compressed tile would mean inflating the entire tile -- about a millisecond
//! per hundred kilobytes, several tiles per frame, across two rasters. With one
//! row per strip the reader touches only the rows it wants and the page cache
//! does the rest. The cost is disk, which is the trade this design is making.
//!
//! The georeferencing tags predate any registry the `tiff` crate knows, so they
//! are written by number, the same way `src/terrain/geotiff.rs` reads them.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result};
use terrain_tiles::TILE_SIZE;
use tiff::encoder::colortype::{Gray32Float, RGB8};
use tiff::encoder::{Compression, TiffEncoder};
use tiff::tags::Tag;

/// GeoTIFF places a raster with private tags read by number.
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;
const TAG_GEO_ASCII_PARAMS: u16 = 34737;
const TAG_GDAL_NODATA: u16 = 42113;

/// GeoKeyDirectory entries, which must appear in ascending order of key.
const KEY_MODEL_TYPE: u16 = 1024;
const KEY_RASTER_TYPE: u16 = 1025;
const KEY_PROJECTED_TYPE: u16 = 3072;
const KEY_LINEAR_UNITS: u16 = 3076;

const MODEL_TYPE_PROJECTED: u16 = 1;
const RASTER_TYPE_PIXEL_IS_AREA: u16 = 1;
const LINEAR_UNITS_METRE: u16 = 9001;

/// NAD83(CSRS) / Canada Atlas Lambert, in metres.
///
/// The tiles are written on the grid HRDEM is already published on, rather than
/// resampled to longitude and latitude as an earlier version of this tool did.
/// Tile boundaries then fall on whole metres, which are source pixel edges, so
/// the finest level is a copy rather than an interpolation.
pub const EPSG_LAMBERT: u16 = 3979;

/// One row per strip, so a reader can seek to the rows it needs.
const ROWS_PER_STRIP: u32 = 1;

/// Where a tile sits on the ground.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TilePlacement {
    /// Easting of the tile's western edge.
    pub west: f64,
    /// Northing of the tile's northern edge.
    pub north: f64,
    /// Ground size of one of this tile's texels, which is `2^level` metres.
    pub metres_per_texel: f64,
}

/// Builds the GeoKeyDirectory describing a raster on EPSG:3979 in metres.
///
/// The layout is a four-value header -- version, revision, minor revision, and
/// the number of keys -- followed by four values per key.
fn geo_key_directory() -> Vec<u16> {
    let keys: [(u16, u16); 4] = [
        (KEY_MODEL_TYPE, MODEL_TYPE_PROJECTED),
        (KEY_RASTER_TYPE, RASTER_TYPE_PIXEL_IS_AREA),
        (KEY_PROJECTED_TYPE, EPSG_LAMBERT),
        (KEY_LINEAR_UNITS, LINEAR_UNITS_METRE),
    ];

    let mut directory = vec![1, 1, 0, keys.len() as u16];
    for (key, value) in keys {
        // A zero location means the value is held inline, in a count of one.
        directory.extend_from_slice(&[key, 0, 1, value]);
    }
    directory
}

/// Writes the placement tags every tile shares.
fn write_placement<W, K, C>(
    image: &mut tiff::encoder::ImageEncoder<'_, W, C, K>,
    placement: TilePlacement,
    nodata: Option<f32>,
) -> Result<()>
where
    W: std::io::Write + std::io::Seek,
    K: tiff::encoder::TiffKind,
    C: tiff::encoder::colortype::ColorType,
{
    let directory = image.encoder();

    directory
        .write_tag(
            Tag::Unknown(TAG_MODEL_PIXEL_SCALE),
            &[placement.metres_per_texel, placement.metres_per_texel, 0.0][..],
        )
        .context("writing ModelPixelScale")?;
    directory
        .write_tag(
            Tag::Unknown(TAG_MODEL_TIEPOINT),
            &[0.0, 0.0, 0.0, placement.west, placement.north, 0.0][..],
        )
        .context("writing ModelTiepoint")?;
    directory
        .write_tag(
            Tag::Unknown(TAG_GEO_KEY_DIRECTORY),
            &geo_key_directory()[..],
        )
        .context("writing the GeoKeyDirectory")?;
    directory
        .write_tag(
            Tag::Unknown(TAG_GEO_ASCII_PARAMS),
            "NAD83(CSRS) / Canada Atlas Lambert|",
        )
        .context("writing GeoAsciiParams")?;
    if let Some(nodata) = nodata {
        directory
            .write_tag(Tag::Unknown(TAG_GDAL_NODATA), format!("{nodata}").as_str())
            .context("writing GDAL_NODATA")?;
    }
    Ok(())
}

/// Creates the level directory a tile lives in.
fn prepare(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    Ok(())
}

/// Expected sample count for a tile of `bands` bands.
fn expected_samples(bands: usize) -> usize {
    (TILE_SIZE as usize) * (TILE_SIZE as usize) * bands
}

/// Writes one elevation tile: a single band of 32-bit floats, in metres.
///
/// The provenance band an earlier version carried is gone. The renderer only
/// ever read band zero, and interleaving a second band doubled both the bytes
/// on disk and the bytes moved for every strip the clipmap asks for. The
/// one-metre and two-metre percentages it existed to explain are still counted
/// and printed at the end of a run.
pub fn write_height_tile(
    path: &Path,
    placement: TilePlacement,
    samples: &[f32],
    nodata: f32,
) -> Result<()> {
    let expected = expected_samples(1);
    anyhow::ensure!(
        samples.len() == expected,
        "expected {expected} samples for a {TILE_SIZE} x {TILE_SIZE} tile, got {}",
        samples.len()
    );
    prepare(path)?;

    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = TiffEncoder::new(BufWriter::new(file))
        .with_context(|| format!("starting {}", path.display()))?
        .with_compression(Compression::Uncompressed);

    let mut image = encoder
        .new_image::<Gray32Float>(TILE_SIZE, TILE_SIZE)
        .context("starting the image")?;
    image
        .rows_per_strip(ROWS_PER_STRIP)
        .context("setting the strip height")?;

    write_placement(&mut image, placement, Some(nodata))?;

    image
        .write_data(samples)
        .with_context(|| format!("writing texels to {}", path.display()))?;
    Ok(())
}

/// Writes one colour tile: three bands of eight-bit sRGB.
pub fn write_colour_tile(path: &Path, placement: TilePlacement, samples: &[u8]) -> Result<()> {
    let expected = expected_samples(3);
    anyhow::ensure!(
        samples.len() == expected,
        "expected {expected} interleaved samples for a {TILE_SIZE} x {TILE_SIZE} tile, got {}",
        samples.len()
    );
    prepare(path)?;

    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut encoder = TiffEncoder::new(BufWriter::new(file))
        .with_context(|| format!("starting {}", path.display()))?
        .with_compression(Compression::Uncompressed);

    let mut image = encoder
        .new_image::<RGB8>(TILE_SIZE, TILE_SIZE)
        .context("starting the image")?;
    image
        .rows_per_strip(ROWS_PER_STRIP)
        .context("setting the strip height")?;

    // Black is the mosaic's own nodata, and it survives into the tiles.
    write_placement(&mut image, placement, Some(0.0))?;

    image
        .write_data(samples)
        .with_context(|| format!("writing texels to {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use tiff::decoder::Decoder;

    use super::*;

    fn placement() -> TilePlacement {
        TilePlacement {
            west: -1_974_272.0,
            north: 524_288.0,
            metres_per_texel: 4.0,
        }
    }

    /// `name` must differ per test: these run in parallel, and two tests
    /// sharing a path race to write and delete the same file.
    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir()
            .join(format!("terrain-download-{}-{name}", std::process::id()))
            .join("02")
            .join("-482_-128.tif")
    }

    #[test]
    fn a_height_tile_round_trips_with_its_placement_intact() {
        let samples: Vec<f32> = (0..expected_samples(1)).map(|i| 100.0 + i as f32).collect();
        let path = temp_path("height-round-trip");
        write_height_tile(&path, placement(), &samples, -32767.0).expect("failed to write");
        let bytes = std::fs::read(&path).expect("failed to read back");
        let _ = std::fs::remove_dir_all(path.parent().and_then(|p| p.parent()).expect("no root"));

        let mut decoder = Decoder::new(Cursor::new(bytes)).expect("failed to decode");
        assert_eq!(
            decoder.dimensions().expect("no dimensions"),
            (TILE_SIZE, TILE_SIZE)
        );
        assert_eq!(
            decoder.colortype().expect("no colour type"),
            tiff::ColorType::Gray(32)
        );

        let scale = decoder
            .get_tag_f64_vec(Tag::Unknown(TAG_MODEL_PIXEL_SCALE))
            .expect("no pixel scale");
        assert_eq!(scale[0], 4.0);
        assert_eq!(scale[1], 4.0);
        assert_eq!(scale[2], 0.0);

        let tiepoint = decoder
            .get_tag_f64_vec(Tag::Unknown(TAG_MODEL_TIEPOINT))
            .expect("no tiepoint");
        assert_eq!(&tiepoint[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(tiepoint[3], placement().west);
        assert_eq!(tiepoint[4], placement().north);

        let keys = decoder
            .get_tag_u32_vec(Tag::Unknown(TAG_GEO_KEY_DIRECTORY))
            .expect("no geo keys");
        assert_eq!(keys[0..4], [1, 1, 0, 4]);
        assert_eq!(keys[4..8], [u32::from(KEY_MODEL_TYPE), 0, 1, 1]);
        assert_eq!(keys[8..12], [u32::from(KEY_RASTER_TYPE), 0, 1, 1]);
        assert_eq!(
            keys[12..16],
            [u32::from(KEY_PROJECTED_TYPE), 0, 1, u32::from(EPSG_LAMBERT)]
        );
        assert_eq!(
            keys[16..20],
            [
                u32::from(KEY_LINEAR_UNITS),
                0,
                1,
                u32::from(LINEAR_UNITS_METRE)
            ]
        );

        assert_eq!(
            decoder
                .get_tag_ascii_string(Tag::Unknown(TAG_GDAL_NODATA))
                .expect("no nodata"),
            "-32767"
        );

        let tiff::decoder::DecodingResult::F32(read) =
            decoder.read_image().expect("failed to read the image")
        else {
            panic!("expected 32-bit floats");
        };
        assert_eq!(read, samples);
    }

    /// One row per strip is what makes a partial read cheap, so it is worth
    /// asserting rather than assuming the encoder honoured it.
    #[test]
    fn a_tile_is_written_one_row_per_strip_and_uncompressed() {
        let samples = vec![1.0f32; expected_samples(1)];
        let path = temp_path("strips");
        write_height_tile(&path, placement(), &samples, -32767.0).expect("failed to write");
        let bytes = std::fs::read(&path).expect("failed to read back");
        let _ = std::fs::remove_dir_all(path.parent().and_then(|p| p.parent()).expect("no root"));

        let mut decoder = Decoder::new(Cursor::new(bytes)).expect("failed to decode");
        assert_eq!(
            decoder.strip_count().expect("no strips"),
            TILE_SIZE,
            "one strip per row"
        );
        assert_eq!(
            decoder
                .get_tag_u32(Tag::Compression)
                .expect("no compression tag"),
            1,
            "1 is uncompressed"
        );
    }

    #[test]
    fn a_colour_tile_round_trips() {
        let samples: Vec<u8> = (0..expected_samples(3)).map(|i| (i % 251) as u8).collect();
        let path = temp_path("colour-round-trip");
        write_colour_tile(&path, placement(), &samples).expect("failed to write");
        let bytes = std::fs::read(&path).expect("failed to read back");
        let _ = std::fs::remove_dir_all(path.parent().and_then(|p| p.parent()).expect("no root"));

        let mut decoder = Decoder::new(Cursor::new(bytes)).expect("failed to decode");
        assert_eq!(
            decoder.colortype().expect("no colour type"),
            tiff::ColorType::RGB(8)
        );
        let tiff::decoder::DecodingResult::U8(read) =
            decoder.read_image().expect("failed to read the image")
        else {
            panic!("expected bytes");
        };
        assert_eq!(read, samples);
    }

    #[test]
    fn a_sample_count_that_does_not_fill_a_tile_is_refused() {
        let error = write_height_tile(
            &std::env::temp_dir().join("terrain-download-should-not-exist.tif"),
            placement(),
            &[0.0; 4],
            -32767.0,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("samples for a"), "{error}");
    }
}
