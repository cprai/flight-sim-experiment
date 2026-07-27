//! Pulling the pixels that matter out of rasters far too large to download.
//!
//! A single one-metre mosaic item is half a million pixels square: the British
//! Columbia block is about 142 GiB on disk. They are Cloud-Optimized GeoTIFFs,
//! so the layout is designed to be read piecemeal over HTTP range requests --
//! the tile offsets live in the header and each 512 x 512 tile is independently
//! compressed. `async-tiff` does that fetching; this module decides which tiles
//! are worth asking for and where their pixels land.
//!
//! Most tiles are not worth asking for. HRDEM only exists over surveyed LiDAR
//! areas, so in the British Columbia block 86% of tiles are entirely nodata.
//! Those compress to a recognisably tiny 3994 bytes, and skipping them on the
//! strength of the byte count alone -- before any request is made -- is the
//! largest single saving in the tool.

use anyhow::{Context, Result, bail, ensure};
use async_tiff::ImageFileDirectory;
use async_tiff::decoder::DecoderRegistry;
use async_tiff::metadata::TiffMetadataReader;
use async_tiff::metadata::cache::ReadaheadMetadataCache;
use async_tiff::reader::ReqwestReader;

use crate::stac::{EXPECTED_EPSG, SourceItem};

/// A 512 x 512 block of 32-bit floats that is entirely nodata compresses, with
/// LZW and no predictor, to this many bytes. Verified identical across all four
/// published products.
const EMPTY_TILE_BYTES: u64 = 3994;

/// Byte counts at or below this are taken to mean a tile holds no data.
///
/// Twice the observed size, to allow for a producer whose encoder differs
/// slightly. There is a wide gap to land in: the smallest tile of real terrain
/// seen in the mosaic is 118 139 bytes, fifteen times this limit.
const EMPTY_TILE_LIMIT: u64 = EMPTY_TILE_BYTES * 2;

/// Whether a tile's compressed size says it holds nothing but nodata.
///
/// Zero is how GDAL records a block it never wrote. Everything else relies on
/// an all-nodata tile being a constant, which LZW crushes to a size no tile of
/// real terrain comes close to.
fn is_empty_byte_count(bytes: u64) -> bool {
    bytes <= EMPTY_TILE_LIMIT
}

/// A rectangle of the EPSG:3979 grid held in memory at one resolution.
///
/// The origin is the *edge* of the north-west pixel, not its centre, because
/// these rasters are area-sampled. That distinction is what makes the one-metre
/// and two-metre grids line up correctly: they share an origin edge but their
/// pixel centres sit half a metre apart, and every lookup here goes through
/// ground metres rather than pixel indices so the offset never has to be
/// applied by hand.
pub struct Window {
    pub origin_x: f64,
    pub origin_y: f64,
    pub metres_per_pixel: f64,
    pub width: u32,
    pub height: u32,
    pub nodata: f32,
    pixels: Vec<f32>,
}

impl Window {
    /// Allocates a window covering at least the given extent in metres, snapped
    /// outwards to the resolution's own grid.
    pub fn covering(
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
        metres_per_pixel: f64,
        nodata: f32,
    ) -> Result<Self> {
        let origin_x = (min_x / metres_per_pixel).floor() * metres_per_pixel;
        let origin_y = (max_y / metres_per_pixel).ceil() * metres_per_pixel;
        let width = ((max_x - origin_x) / metres_per_pixel).ceil().max(1.0);
        let height = ((origin_y - min_y) / metres_per_pixel).ceil().max(1.0);

        ensure!(
            width <= f64::from(u32::MAX) && height <= f64::from(u32::MAX),
            "the source window would be {width} x {height} pixels"
        );

        let width = width as u32;
        let height = height as u32;
        let count = (width as usize)
            .checked_mul(height as usize)
            .context("the source window does not fit in memory")?;

        Ok(Self {
            origin_x,
            origin_y,
            metres_per_pixel,
            width,
            height,
            nodata,
            pixels: vec![nodata; count],
        })
    }

    /// Writes one pixel directly, so tests can build a window without a server.
    #[cfg(test)]
    pub fn set_for_test(&mut self, x: u32, y: u32, value: f32) {
        self.pixels[(y as usize) * (self.width as usize) + x as usize] = value;
    }

    /// Samples the window at a point in projected metres, bilinearly.
    ///
    /// Returns `None` if the point falls outside the window or if any of the
    /// four pixels the interpolation needs is nodata. Refusing to interpolate
    /// across a hole keeps invented values from creeping one pixel into every
    /// gap, at the cost of eroding the edge of real coverage by the same pixel.
    pub fn sample(&self, x: f64, y: f64) -> Option<f32> {
        // Position in pixel-centre space: 0.0 is the centre of pixel 0.
        let fx = (x - self.origin_x) / self.metres_per_pixel - 0.5;
        let fy = (self.origin_y - y) / self.metres_per_pixel - 0.5;

        let x0 = fx.floor();
        let y0 = fy.floor();
        if x0 < 0.0 || y0 < 0.0 {
            return None;
        }
        let x0 = x0 as u32;
        let y0 = y0 as u32;
        if x0 + 1 >= self.width || y0 + 1 >= self.height {
            return None;
        }

        let tx = fx - f64::from(x0);
        let ty = fy - f64::from(y0);
        let row = |y: u32| (y as usize) * (self.width as usize);

        let a = self.pixels[row(y0) + x0 as usize];
        let b = self.pixels[row(y0) + x0 as usize + 1];
        let c = self.pixels[row(y0 + 1) + x0 as usize];
        let d = self.pixels[row(y0 + 1) + x0 as usize + 1];
        if a == self.nodata || b == self.nodata || c == self.nodata || d == self.nodata {
            return None;
        }

        let top = f64::from(a) + (f64::from(b) - f64::from(a)) * tx;
        let bottom = f64::from(c) + (f64::from(d) - f64::from(c)) * tx;
        Some((top + (bottom - top) * ty) as f32)
    }
}

/// One opened remote raster, ready to be asked for tiles.
pub struct SourceRaster {
    pub item: SourceItem,
    ifd: ImageFileDirectory,
    reader: ReqwestReader,
    nodata: f32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: usize,
    tiles_down: usize,
    /// Easting of the western edge of column 0.
    tie_x: f64,
    /// Northing of the northern edge of row 0.
    tie_y: f64,
    metres_per_pixel: f64,
}

impl SourceRaster {
    /// Opens the raster's header and checks it is the shape everything
    /// downstream assumes.
    pub async fn open(client: &reqwest::Client, item: SourceItem) -> Result<Self> {
        let url = reqwest::Url::parse(&item.href)
            .with_context(|| format!("{} has an unusable href {}", item.id, item.href))?;
        let reader = ReqwestReader::new(client.clone(), url);
        let cache = ReadaheadMetadataCache::new(reader.clone());

        let mut metadata = TiffMetadataReader::try_open(&cache)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("opening {}", item.href))?;

        // Only the full-resolution image is wanted. Reading just the first
        // directory also avoids pulling the overviews' tile tables, which for
        // the one-metre blocks is another few megabytes of offsets.
        let ifd = metadata
            .read_next_ifd(&cache)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("reading the header of {}", item.href))?
            .with_context(|| format!("{} has no image directory", item.href))?;

        let metres_per_pixel = item.resolution.metres();
        let raster = Self::validated(item, ifd, reader, metres_per_pixel)?;
        Ok(raster)
    }

    fn validated(
        item: SourceItem,
        ifd: ImageFileDirectory,
        reader: ReqwestReader,
        metres_per_pixel: f64,
    ) -> Result<Self> {
        let describe = |what: &str| format!("{} {what}", item.id);

        let (tile_width, tile_height) = match (ifd.tile_width(), ifd.tile_height()) {
            (Some(w), Some(h)) => (w, h),
            _ => bail!("{} is not tiled, so it cannot be read piecemeal", item.id),
        };

        ensure!(
            ifd.samples_per_pixel() == 1,
            "{}",
            describe(&format!(
                "has {} bands, but an elevation raster should have one",
                ifd.samples_per_pixel()
            ))
        );

        let projected = ifd
            .geo_key_directory()
            .and_then(|keys| keys.projected_type)
            .map(u32::from);
        ensure!(
            projected == Some(EXPECTED_EPSG),
            "{}",
            describe(&format!(
                "declares projection {projected:?}, not EPSG:{EXPECTED_EPSG}"
            ))
        );

        let scale = ifd
            .model_pixel_scale()
            .with_context(|| describe("has no ModelPixelScale, so its scale is unknown"))?;
        ensure!(
            scale.len() >= 2
                && (scale[0] - metres_per_pixel).abs() < 1e-9
                && (scale[1] - metres_per_pixel).abs() < 1e-9,
            "{}",
            describe(&format!(
                "has pixel scale {scale:?}, but {} was expected",
                metres_per_pixel
            ))
        );

        let tiepoint = ifd
            .model_tiepoint()
            .with_context(|| describe("has no ModelTiepoint, so its position is unknown"))?;
        ensure!(
            tiepoint.len() >= 6,
            "{}",
            describe("has a truncated ModelTiepoint")
        );
        ensure!(
            tiepoint[0] == 0.0 && tiepoint[1] == 0.0,
            "{}",
            describe("ties a point other than its own origin, which is unsupported")
        );

        let nodata_text = ifd
            .gdal_nodata()
            .with_context(|| describe("declares no nodata value, so gaps cannot be told apart"))?;
        let nodata: f32 = nodata_text.trim().parse().with_context(|| {
            describe(&format!("has an unreadable nodata value {nodata_text:?}"))
        })?;

        let (tiles_across, tiles_down) = ifd
            .tile_count()
            .with_context(|| describe("does not say how many tiles it has"))?;

        Ok(Self {
            item,
            reader,
            nodata,
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
            tie_x: tiepoint[3],
            tie_y: tiepoint[4],
            metres_per_pixel,
            ifd,
        })
    }

    pub fn nodata(&self) -> f32 {
        self.nodata
    }

    /// Whether a tile is known to hold nothing but nodata, judged from its
    /// compressed size without fetching it.
    fn tile_is_empty(&self, index: usize) -> bool {
        match self.ifd.tile_byte_counts().and_then(|c| c.get(index)) {
            Some(&bytes) => is_empty_byte_count(bytes),
            // An absent entry cannot be fetched either way.
            None => true,
        }
    }

    fn tile_bytes(&self, index: usize) -> u64 {
        self.ifd
            .tile_byte_counts()
            .and_then(|c| c.get(index))
            .copied()
            .unwrap_or(0)
    }

    /// Whether the raster holds data at a point given in projected metres.
    ///
    /// Answered at tile granularity from the header alone, which is what makes
    /// the coverage estimate cheap.
    pub fn has_data_at(&self, x: f64, y: f64) -> bool {
        let Some((column, row)) = self.pixel_at(x, y) else {
            return false;
        };
        let tile_x = column / self.tile_width as usize;
        let tile_y = row / self.tile_height as usize;
        if tile_x >= self.tiles_across || tile_y >= self.tiles_down {
            return false;
        }
        !self.tile_is_empty(tile_y * self.tiles_across + tile_x)
    }

    /// The pixel containing a point in projected metres, if it is inside.
    fn pixel_at(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let column = ((x - self.tie_x) / self.metres_per_pixel).floor();
        let row = ((self.tie_y - y) / self.metres_per_pixel).floor();
        if column < 0.0 || row < 0.0 {
            return None;
        }
        let (column, row) = (column as usize, row as usize);
        if column >= self.ifd.image_width() as usize || row >= self.ifd.image_height() as usize {
            return None;
        }
        Some((column, row))
    }

    /// The tiles this raster would contribute to `window`, in row-major order
    /// so that neighbouring tiles merge into contiguous range requests.
    fn tiles_for(&self, window: &Window) -> Vec<(usize, usize)> {
        // The window's extent as pixel indices in this raster.
        let left = (window.origin_x - self.tie_x) / self.metres_per_pixel;
        let top = (self.tie_y - window.origin_y) / self.metres_per_pixel;
        let right = left + f64::from(window.width);
        let bottom = top + f64::from(window.height);

        let clamp = |value: f64, limit: usize| value.max(0.0).min(limit as f64) as usize;
        let first_column = clamp(left.floor(), self.ifd.image_width() as usize);
        let last_column = clamp(right.ceil(), self.ifd.image_width() as usize);
        let first_row = clamp(top.floor(), self.ifd.image_height() as usize);
        let last_row = clamp(bottom.ceil(), self.ifd.image_height() as usize);
        if first_column >= last_column || first_row >= last_row {
            return Vec::new();
        }

        let first_tile_x = first_column / self.tile_width as usize;
        let last_tile_x = (last_column - 1) / self.tile_width as usize;
        let first_tile_y = first_row / self.tile_height as usize;
        let last_tile_y = (last_row - 1) / self.tile_height as usize;

        let mut wanted = Vec::new();
        for tile_y in first_tile_y..=last_tile_y.min(self.tiles_down - 1) {
            for tile_x in first_tile_x..=last_tile_x.min(self.tiles_across - 1) {
                if !self.tile_is_empty(tile_y * self.tiles_across + tile_x) {
                    wanted.push((tile_x, tile_y));
                }
            }
        }
        wanted
    }

    /// Bytes that `fill` would download for this window.
    pub fn bytes_for(&self, window: &Window) -> u64 {
        self.tiles_for(window)
            .into_iter()
            .map(|(tile_x, tile_y)| self.tile_bytes(tile_y * self.tiles_across + tile_x))
            .sum()
    }

    /// Fetches this raster's contribution to `window` and returns the number of
    /// bytes downloaded.
    pub async fn fill(&self, window: &mut Window, batch: usize) -> Result<u64> {
        let wanted = self.tiles_for(window);
        if wanted.is_empty() {
            return Ok(0);
        }

        let registry = DecoderRegistry::default();
        let mut downloaded = 0;
        let mut done = 0;

        for chunk in wanted.chunks(batch.max(1)) {
            let tiles = self
                .ifd
                .fetch_tiles(chunk, &self.reader)
                .await
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("fetching tiles from {}", self.item.id))?;

            for tile in tiles {
                let (tile_x, tile_y) = (tile.x(), tile.y());
                downloaded += self.tile_bytes(tile_y * self.tiles_across + tile_x);

                let array = tile
                    .decode(&registry)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .with_context(|| {
                        format!("decoding tile ({tile_x}, {tile_y}) of {}", self.item.id)
                    })?;
                let async_tiff::TypedArray::Float32(values) = array.data() else {
                    bail!(
                        "{} holds {:?} samples, but elevations should be 32-bit floats",
                        self.item.id,
                        array.data_type()
                    );
                };
                self.blit(window, tile_x, tile_y, values);
            }

            done += chunk.len();
            log::debug!("{}: {done}/{} tiles", self.item.id, wanted.len());
        }

        Ok(downloaded)
    }

    /// Copies one decoded tile into the window.
    ///
    /// Nodata is skipped rather than written. Where two mosaic blocks overlap
    /// they each pad their edge with nodata, and writing it would punch a hole
    /// through the neighbour's real data.
    fn blit(&self, window: &mut Window, tile_x: usize, tile_y: usize, values: &[f32]) {
        let tile_width = self.tile_width as usize;
        let tile_height = self.tile_height as usize;
        let first_column = tile_x * tile_width;
        let first_row = tile_y * tile_height;

        // Where this tile's north-west corner sits in the window.
        let offset_x = ((self.tie_x - window.origin_x) / self.metres_per_pixel).round() as i64
            + first_column as i64;
        let offset_y = ((window.origin_y - self.tie_y) / self.metres_per_pixel).round() as i64
            + first_row as i64;

        let image_width = self.ifd.image_width() as usize;
        let image_height = self.ifd.image_height() as usize;

        for row in 0..tile_height {
            // Tiles are padded out to a full tile even at the image edge.
            if first_row + row >= image_height {
                break;
            }
            let target_y = offset_y + row as i64;
            if target_y < 0 || target_y >= i64::from(window.height) {
                continue;
            }
            let target_row = (target_y as usize) * (window.width as usize);

            for column in 0..tile_width {
                if first_column + column >= image_width {
                    break;
                }
                let target_x = offset_x + column as i64;
                if target_x < 0 || target_x >= i64::from(window.width) {
                    continue;
                }
                let value = values[row * tile_width + column];
                if value == self.nodata {
                    continue;
                }
                window.pixels[target_row + target_x as usize] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window whose pixels can be written directly, for testing `sample`.
    fn window_with(metres_per_pixel: f64, width: u32, height: u32, values: &[f32]) -> Window {
        let mut window = Window::covering(
            0.0,
            -(f64::from(height) * metres_per_pixel),
            f64::from(width) * metres_per_pixel,
            0.0,
            metres_per_pixel,
            -32767.0,
        )
        .expect("failed to allocate");
        assert_eq!((window.width, window.height), (width, height));
        window.pixels.copy_from_slice(values);
        window
    }

    #[test]
    fn a_window_snaps_outwards_to_its_own_grid() {
        let window = Window::covering(-1000.5, 499.25, -900.5, 600.75, 2.0, -32767.0)
            .expect("failed to allocate");
        assert_eq!(window.origin_x, -1002.0);
        assert_eq!(window.origin_y, 602.0);
        assert!(window.origin_x <= -1000.5);
        assert!(window.origin_y >= 600.75);
        // Wide and tall enough to contain the requested extent.
        assert!(window.origin_x + f64::from(window.width) * 2.0 >= -900.5);
        assert!(window.origin_y - f64::from(window.height) * 2.0 <= 499.25);
    }

    #[test]
    fn sampling_a_pixel_centre_returns_that_pixel() {
        let window = window_with(1.0, 2, 2, &[10.0, 20.0, 30.0, 40.0]);
        // Centre of pixel (0, 0) is half a metre in from the origin edge.
        let value = window.sample(0.5, -0.5).expect("expected a sample");
        assert!((value - 10.0).abs() < 1e-6, "{value}");
    }

    #[test]
    fn sampling_between_pixels_interpolates() {
        let window = window_with(1.0, 2, 2, &[0.0, 10.0, 0.0, 10.0]);
        let value = window.sample(1.0, -0.5).expect("expected a sample");
        assert!((value - 5.0).abs() < 1e-6, "{value}");
    }

    #[test]
    fn a_hole_in_any_corner_refuses_the_sample() {
        let window = window_with(1.0, 2, 2, &[10.0, 20.0, 30.0, -32767.0]);
        assert_eq!(window.sample(1.0, -1.0), None);
        // ...and the same window away from the hole is still unusable, because
        // every interior point of a 2x2 window touches all four pixels.
        assert_eq!(window.sample(0.6, -0.6), None);
    }

    #[test]
    fn sampling_outside_the_window_returns_nothing() {
        let window = window_with(1.0, 2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(window.sample(-5.0, -0.5), None);
        assert_eq!(window.sample(0.5, 5.0), None);
        // The outer half-pixel has no second pixel to interpolate towards.
        assert_eq!(window.sample(0.1, -0.5), None);
    }

    /// The two mosaics share an origin edge but not their pixel centres: a
    /// one-metre pixel is centred half a metre in, a two-metre pixel a full
    /// metre in. Sampling by ground position rather than index is what keeps
    /// them consistent, and this pins that down.
    #[test]
    fn the_two_resolutions_agree_about_ground_position() {
        let fine = window_with(1.0, 4, 4, &[0.0; 16]);
        let coarse = window_with(2.0, 2, 2, &[0.0; 4]);
        assert_eq!(fine.origin_x, coarse.origin_x);
        assert_eq!(fine.origin_y, coarse.origin_y);

        // Centre of the fine grid's pixel 0 versus the coarse grid's pixel 0.
        let fine_centre = fine.origin_x + 0.5 * fine.metres_per_pixel;
        let coarse_centre = coarse.origin_x + 0.5 * coarse.metres_per_pixel;
        assert!((coarse_centre - fine_centre - 0.5).abs() < 1e-12);
    }

    /// Byte counts taken from the real mosaic: a block GDAL never wrote, the
    /// all-nodata tile that fills most of the grid, and the four smallest real
    /// tiles seen in a scan of tile rows 940 to 950 of the British Columbia
    /// block.
    #[test]
    fn an_all_nodata_tile_is_recognised_by_its_compressed_size() {
        assert!(is_empty_byte_count(0), "a sparse block holds no data");
        assert!(is_empty_byte_count(EMPTY_TILE_BYTES));

        for real in [118_139, 380_423, 593_849, 689_583, 1_085_450] {
            assert!(
                !is_empty_byte_count(real),
                "{real} bytes is a tile of real terrain"
            );
        }
    }
}
