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
use futures::StreamExt;

use crate::resample::MetreExtent;
use crate::retry;
use crate::stac::SourceItem;

/// A 512 x 512 block of 32-bit floats that is entirely nodata compresses, with
/// LZW and no predictor, to this many bytes. Verified identical across all four
/// published products.
const EMPTY_TILE_BYTES: u64 = 3994;

/// Byte counts at or below this mean an HRDEM tile holds nothing but nodata.
///
/// Twice the observed size, to allow for a producer whose encoder differs
/// slightly. There is a wide gap to land in: the smallest tile of real terrain
/// seen in the mosaic is 118 139 bytes, fifteen times this limit.
pub const ELEVATION_EMPTY_TILE_LIMIT: u64 = EMPTY_TILE_BYTES * 2;

/// The same test for MRDEM, which cannot afford HRDEM's tolerance.
///
/// MRDEM is written with the same LZW-and-no-predictor settings, so its empty
/// tiles are the same 3994 bytes -- but the gap above them is gone. Across the
/// whole national raster 58 206 of its 118 012 tiles are exactly 3994 bytes and
/// the next size up is 4014, twenty bytes away, where HRDEM's next size up is a
/// hundred thousand bytes away.
///
/// Resolution is why. An MRDEM tile is 512 pixels of 30 m, so it covers 15 km
/// square; a tile holding one sliver of coastline and 99% ocean compresses
/// almost as small as an empty one. Doubling this limit the way HRDEM's does
/// would discard 732 tiles that hold real coast. So the test is exact equality
/// with the empty size in all but name, and the margin is measurement rather
/// than allowance.
pub const MRDEM_EMPTY_TILE_LIMIT: u64 = EMPTY_TILE_BYTES;

/// Whether a tile's compressed size says it holds nothing but nodata.
///
/// Zero always counts: that is how a sparse block is recorded, and there are no
/// bytes to fetch either way. Above that the judgement is the caller's, because
/// it depends entirely on the raster. An all-nodata HRDEM tile is a constant
/// that LZW crushes to 3994 bytes while real terrain never compresses below
/// 118 139, so a threshold between them is safe. Imagery has no such gap --
/// see `RasterSpec::empty_tile_limit`.
fn is_empty_byte_count(bytes: u64, limit: u64) -> bool {
    bytes == 0 || bytes <= limit
}

/// One decoded source tile, in the raster's own pixel lattice.
///
/// This is the only shape source pixels are ever held in. It exists between the
/// moment a tile finishes decoding and the moment whatever is consuming it has
/// taken what it needs, and its buffer is handed straight to the next tile --
/// so the resident set is one source tile rather than a copy of everything the
/// block covers.
pub struct Patch {
    /// Easting of the western edge of this patch's column 0.
    pub west: f64,
    /// Northing of the northern edge of its row 0.
    pub north: f64,
    pub metres_per_pixel: f64,
    /// Columns that are part of the raster.
    ///
    /// Fewer than `stride` in the last tile of a row: a COG pads its edge tiles
    /// out to a full tile, and that padding is not ground.
    pub width: usize,
    /// Rows that are part of the raster, for the same reason.
    pub height: usize,
    /// Columns per row as stored, which is the tile width.
    pub stride: usize,
    pub bands: usize,
    pub nodata: f32,
    /// Interleaved samples, `stride` columns per row.
    pub values: Vec<f32>,
}

impl Patch {
    /// One pixel's bands, or `None` where it is nodata.
    ///
    /// Nodata is reported rather than returned so that callers skip it instead
    /// of writing it. Where two mosaic blocks overlap they each pad their edge
    /// with nodata, and writing it would punch a hole through the neighbour's
    /// real data.
    pub fn texel(&self, x: usize, y: usize) -> Option<&[f32]> {
        let at = (y * self.stride + x) * self.bands;
        let sample = &self.values[at..at + self.bands];
        (!sample.iter().all(|&v| v == self.nodata)).then_some(sample)
    }
}

/// A rectangle of one projected grid, held in memory.
///
/// The origin is the *edge* of the north-west pixel, not its centre, because
/// these rasters are area-sampled. That distinction is what makes the one-metre
/// and two-metre elevation grids line up correctly: they share an origin edge
/// but their pixel centres sit half a metre apart, and every lookup here goes
/// through ground metres rather than pixel indices so the offset never has to
/// be applied by hand.
///
/// Samples are held interleaved and always as `f32`, whatever the source stored.
/// Imagery arrives as bytes and is widened on the way in, which costs four times
/// the memory but means one interpolation path serves elevation and colour alike
/// -- and the arithmetic has to happen in floating point regardless.
pub struct Window {
    pub origin_x: f64,
    pub origin_y: f64,
    pub metres_per_pixel: f64,
    pub width: u32,
    pub height: u32,
    pub bands: usize,
    pub nodata: f32,
    pixels: Vec<f32>,
}

impl Window {
    /// Allocates a window covering at least the given extent in metres, snapped
    /// outwards to the source's own grid.
    pub fn covering(
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
        metres_per_pixel: f64,
        bands: usize,
        nodata: f32,
    ) -> Result<Self> {
        ensure!(bands >= 1, "a window needs at least one band");

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
            .and_then(|n| n.checked_mul(bands))
            .context("the source window does not fit in memory")?;

        let mut pixels = Vec::new();
        pixels
            .try_reserve_exact(count)
            .context("the source window does not fit in memory")?;
        pixels.resize(count, nodata);

        Ok(Self {
            origin_x,
            origin_y,
            metres_per_pixel,
            width,
            height,
            bands,
            nodata,
            pixels,
        })
    }

    /// The ground this window covers, edge to edge.
    pub fn extent(&self) -> MetreExtent {
        MetreExtent {
            min_x: self.origin_x,
            min_y: self.origin_y - f64::from(self.height) * self.metres_per_pixel,
            max_x: self.origin_x + f64::from(self.width) * self.metres_per_pixel,
            max_y: self.origin_y,
        }
    }

    /// Writes one pixel directly, so tests can build a window without a server.
    #[cfg(test)]
    pub fn set_for_test(&mut self, x: u32, y: u32, values: &[f32]) {
        let at = ((y as usize) * (self.width as usize) + x as usize) * self.bands;
        self.pixels[at..at + self.bands].copy_from_slice(values);
    }

    /// Whether the pixel at a flat index is nodata in every band.
    ///
    /// All bands rather than any: imagery declares nodata as black across the
    /// whole pixel, and a single zero channel is an ordinary dark colour.
    fn is_nodata(&self, at: usize) -> bool {
        self.pixels[at..at + self.bands]
            .iter()
            .all(|&v| v == self.nodata)
    }

    /// Copies one decoded source tile into the window.
    ///
    /// The patch carries its own ground position, so the window does not need
    /// to know which raster it came from -- which is what lets several mosaic
    /// blocks fill one window without any of them being held open.
    pub fn absorb(&mut self, patch: &Patch) {
        debug_assert_eq!(self.bands, patch.bands);
        debug_assert_eq!(self.metres_per_pixel, patch.metres_per_pixel);

        let offset_x = ((patch.west - self.origin_x) / self.metres_per_pixel).round() as i64;
        let offset_y = ((self.origin_y - patch.north) / self.metres_per_pixel).round() as i64;

        for row in 0..patch.height {
            let target_y = offset_y + row as i64;
            if target_y < 0 || target_y >= i64::from(self.height) {
                continue;
            }
            let target_row = (target_y as usize) * (self.width as usize);

            for column in 0..patch.width {
                let target_x = offset_x + column as i64;
                if target_x < 0 || target_x >= i64::from(self.width) {
                    continue;
                }
                let Some(sample) = patch.texel(column, row) else {
                    continue;
                };
                let to = (target_row + target_x as usize) * self.bands;
                self.pixels[to..to + self.bands].copy_from_slice(sample);
            }
        }
    }

    /// Samples the window at a point in projected metres, bilinearly, writing
    /// one value per band into `out`.
    ///
    /// Returns `false` if the point falls outside the window or if any of the
    /// four pixels the interpolation needs is nodata. Refusing to interpolate
    /// across a hole keeps invented values from creeping one pixel into every
    /// gap, at the cost of eroding the edge of real coverage by the same pixel.
    pub fn sample_into(&self, x: f64, y: f64, out: &mut [f32]) -> bool {
        debug_assert_eq!(out.len(), self.bands);

        // Position in pixel-centre space: 0.0 is the centre of pixel 0.
        let fx = (x - self.origin_x) / self.metres_per_pixel - 0.5;
        let fy = (self.origin_y - y) / self.metres_per_pixel - 0.5;

        let x0 = fx.floor();
        let y0 = fy.floor();
        if x0 < 0.0 || y0 < 0.0 {
            return false;
        }
        let x0 = x0 as u32;
        let y0 = y0 as u32;
        if x0 + 1 >= self.width || y0 + 1 >= self.height {
            return false;
        }

        let tx = fx - f64::from(x0);
        let ty = fy - f64::from(y0);
        let at = |x: u32, y: u32| ((y as usize) * (self.width as usize) + x as usize) * self.bands;
        let corners = [
            at(x0, y0),
            at(x0 + 1, y0),
            at(x0, y0 + 1),
            at(x0 + 1, y0 + 1),
        ];
        if corners.iter().any(|&c| self.is_nodata(c)) {
            return false;
        }

        for (band, value) in out.iter_mut().enumerate() {
            let a = f64::from(self.pixels[corners[0] + band]);
            let b = f64::from(self.pixels[corners[1] + band]);
            let c = f64::from(self.pixels[corners[2] + band]);
            let d = f64::from(self.pixels[corners[3] + band]);
            let top = a + (b - a) * tx;
            let bottom = c + (d - c) * tx;
            *value = (top + (bottom - top) * ty) as f32;
        }
        true
    }
}

/// What a source raster is expected to look like, so the header can be checked
/// against it rather than against one hard-coded product.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RasterSpec {
    /// The projection the raster must declare.
    pub epsg: u32,
    /// Ground sample distance in metres, which is also its pixel scale.
    pub metres_per_pixel: f64,
    /// How many samples each pixel carries.
    pub bands: usize,
    /// The value to treat as absent, if the file does not say.
    ///
    /// HRDEM declares `-32767` in a GDAL_NODATA tag; the Sentinel-2 mosaics
    /// record their nodata only in STAC, so it has to be supplied here.
    pub fallback_nodata: f32,
    /// Compressed size at or below which a tile is assumed to be all nodata
    /// and skipped without fetching.
    ///
    /// Zero disables the guess, leaving only genuinely absent blocks skipped.
    /// That is the right setting whenever empty and nearly-empty tiles are not
    /// cleanly separable by size. The Sentinel-2 mosaics are such a case: an
    /// all-black 256-pixel tile deflates to 213 bytes, but a coastal tile that
    /// is 0.22% land and the rest ocean comes to 723, and one 5% land to 6850.
    /// Any threshold high enough to catch the empties would throw away the
    /// shoreline.
    pub empty_tile_limit: u64,
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
    bands: usize,
    empty_tile_limit: u64,
}

impl SourceRaster {
    /// Opens the raster's header and checks it is the shape everything
    /// downstream assumes.
    pub async fn open(
        client: &reqwest::Client,
        item: SourceItem,
        spec: RasterSpec,
    ) -> Result<Self> {
        let url = reqwest::Url::parse(&item.href)
            .with_context(|| format!("{} has an unusable href {}", item.id, item.href))?;
        let reader = ReqwestReader::new(client.clone(), url);
        let cache = ReadaheadMetadataCache::new(reader.clone());

        // Opening and reading the first directory are retried as one unit, so
        // the reader is rebuilt per attempt rather than resumed mid-way through
        // a header it may have read only half of. It costs nothing: the
        // readahead cache already holds whatever the previous attempt got.
        //
        // Only the full-resolution image is wanted. Reading just the first
        // directory also avoids pulling the overviews' tile tables, which for
        // the one-metre blocks is another few megabytes of offsets.
        let ifd = retry::retrying(
            format_args!("reading the header of {}", item.href),
            retry::is_transient_tiff,
            || async {
                let mut metadata = TiffMetadataReader::try_open(&cache).await?;
                metadata.read_next_ifd(&cache).await
            },
        )
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("reading the header of {}", item.href))?
        .with_context(|| format!("{} has no image directory", item.href))?;

        Self::validated(item, ifd, reader, spec)
    }

    fn validated(
        item: SourceItem,
        ifd: ImageFileDirectory,
        reader: ReqwestReader,
        spec: RasterSpec,
    ) -> Result<Self> {
        let metres_per_pixel = spec.metres_per_pixel;
        let describe = |what: &str| format!("{} {what}", item.id);

        let (tile_width, tile_height) = match (ifd.tile_width(), ifd.tile_height()) {
            (Some(w), Some(h)) => (w, h),
            _ => bail!("{} is not tiled, so it cannot be read piecemeal", item.id),
        };

        ensure!(
            usize::from(ifd.samples_per_pixel()) == spec.bands,
            "{}",
            describe(&format!(
                "has {} bands, but {} were expected",
                ifd.samples_per_pixel(),
                spec.bands
            ))
        );

        let projected = ifd
            .geo_key_directory()
            .and_then(|keys| keys.projected_type)
            .map(u32::from);
        ensure!(
            projected == Some(spec.epsg),
            "{}",
            describe(&format!(
                "declares projection {projected:?}, not EPSG:{}",
                spec.epsg
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

        // HRDEM states its nodata in the file; the Sentinel-2 mosaics leave the
        // tag off and declare it in STAC instead, so fall back to the spec.
        let nodata: f32 = match ifd.gdal_nodata() {
            Some(text) => text
                .trim()
                .parse()
                .with_context(|| describe(&format!("has an unreadable nodata value {text:?}")))?,
            None => spec.fallback_nodata,
        };

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
            bands: spec.bands,
            empty_tile_limit: spec.empty_tile_limit,
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
            Some(&bytes) => is_empty_byte_count(bytes, self.empty_tile_limit),
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

    /// The tiles this raster would contribute to an extent, in row-major order
    /// so that neighbouring tiles merge into contiguous range requests.
    ///
    /// Takes an extent rather than a window so that the download can be sized
    /// before anything is allocated -- a window large enough to hold a whole
    /// box no longer exists, and building one just to count bytes would defeat
    /// the point of working block by block.
    fn tiles_for(&self, extent: MetreExtent) -> Vec<(usize, usize)> {
        // The extent as pixel indices in this raster.
        let left = (extent.min_x - self.tie_x) / self.metres_per_pixel;
        let top = (self.tie_y - extent.max_y) / self.metres_per_pixel;
        let right = (extent.max_x - self.tie_x) / self.metres_per_pixel;
        let bottom = (self.tie_y - extent.min_y) / self.metres_per_pixel;

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

    /// Bytes that `fill` would download for an extent.
    pub fn bytes_for(&self, extent: MetreExtent) -> u64 {
        self.tiles_for(extent)
            .into_iter()
            .map(|(tile_x, tile_y)| self.tile_bytes(tile_y * self.tiles_across + tile_x))
            .sum()
    }

    /// Where the raster's north-west corner sits, in projected metres.
    pub fn origin(&self) -> (f64, f64) {
        (self.tie_x, self.tie_y)
    }

    pub fn metres_per_pixel(&self) -> f64 {
        self.metres_per_pixel
    }

    /// Fetches this raster's contribution to an extent, handing each tile to
    /// `sink` as it decodes and dropping it as soon as `sink` returns.
    ///
    /// Returns the number of bytes downloaded.
    ///
    /// `concurrency` range requests are kept in flight at once. That is the
    /// whole speed story of this module: `async-tiff`'s `ReqwestReader` does not
    /// override `AsyncFileReader::get_byte_ranges`, and the trait's default
    /// implementation awaits the ranges strictly one after another, so before
    /// this every tile of a download paid a full round trip on its own.
    ///
    /// Tiles are decoded on this thread as they arrive rather than across a
    /// thread pool. Decoding a batch on all cores was tried and made no
    /// difference to wall clock -- a download of this shape is bound by the
    /// network, not by LZW -- while multiplying the resident set by the number
    /// of threads holding a tile at once.
    ///
    /// Memory is bounded by `concurrency`, not by the size of the extent: one
    /// decoded tile exists at a time, and its buffer is reused by the next.
    pub async fn stream(
        &self,
        extent: MetreExtent,
        concurrency: usize,
        mut sink: impl FnMut(&Patch),
    ) -> Result<u64> {
        let wanted = self.tiles_for(extent);
        if wanted.is_empty() {
            return Ok(0);
        }

        let registry = DecoderRegistry::default();
        let mut downloaded = 0;
        let mut done = 0;
        let mut samples = Vec::new();

        let mut arrivals =
            futures::stream::iter(wanted.iter().map(|&(x, y)| self.fetch_tile(x, y)))
                .buffer_unordered(concurrency.max(1));

        while let Some(tile) = arrivals.next().await {
            let tile = tile?;
            downloaded += self.tile_bytes(tile.y() * self.tiles_across + tile.x());

            let patch = self.decode(tile, &registry, samples)?;
            sink(&patch);
            // Take the buffer back for the next tile, so a block's worth of
            // tiles costs one allocation rather than one each.
            samples = patch.values;

            done += 1;
            if done % 64 == 0 {
                log::debug!("{}: {done}/{} tiles", self.item.id, wanted.len());
            }
        }

        Ok(downloaded)
    }

    /// Fetches one tile, trying again if the network rather than the file was
    /// what failed.
    ///
    /// Retrying here rather than around the whole block is deliberate: a block
    /// is up to a few hundred range requests and re-running it would refetch
    /// every one of them, where the failure is almost always a single request
    /// that would have succeeded a moment later.
    async fn fetch_tile(&self, x: usize, y: usize) -> Result<async_tiff::Tile> {
        retry::retrying(
            format_args!("tile ({x}, {y}) of {}", self.item.id),
            retry::is_transient_tiff,
            || self.ifd.fetch_tile(x, y, &self.reader),
        )
        .await
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("fetching a tile of {}", self.item.id))
    }

    /// Decompresses one tile into a patch placed on the raster's own lattice.
    fn decode(
        &self,
        tile: async_tiff::Tile,
        registry: &DecoderRegistry,
        mut values: Vec<f32>,
    ) -> Result<Patch> {
        let (tile_x, tile_y) = (tile.x(), tile.y());
        let array = tile
            .decode(registry)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("decoding tile ({tile_x}, {tile_y}) of {}", self.item.id))?;

        // Elevation arrives as floats and imagery as bytes; both are widened to
        // f32 so one blit and one interpolator serve each. Extending a cleared
        // buffer rather than collecting a new one is what lets the pool recycle
        // it: after the first batch these never allocate again.
        values.clear();
        match array.data() {
            async_tiff::TypedArray::Float32(v) => values.extend_from_slice(v),
            async_tiff::TypedArray::UInt8(v) => values.extend(v.iter().map(|&b| f32::from(b))),
            async_tiff::TypedArray::UInt16(v) => values.extend(v.iter().map(|&b| f32::from(b))),
            other => bail!(
                "{} holds {:?} samples, which this tool cannot read",
                self.item.id,
                std::mem::discriminant(other)
            ),
        }

        let first_column = tile_x * self.tile_width as usize;
        let first_row = tile_y * self.tile_height as usize;
        Ok(Patch {
            west: self.tie_x + first_column as f64 * self.metres_per_pixel,
            north: self.tie_y - first_row as f64 * self.metres_per_pixel,
            metres_per_pixel: self.metres_per_pixel,
            // Edge tiles are padded out to a full tile; the padding is not part
            // of the raster and must not reach the output.
            width: (self.ifd.image_width() as usize)
                .saturating_sub(first_column)
                .min(self.tile_width as usize),
            height: (self.ifd.image_height() as usize)
                .saturating_sub(first_row)
                .min(self.tile_height as usize),
            stride: self.tile_width as usize,
            bands: self.bands,
            nodata: self.nodata,
            values,
        })
    }

    /// Fetches this raster's contribution to `window`.
    ///
    /// The staging step for sources that have to be interpolated: the window
    /// holds enough neighbouring pixels for bilinear sampling, which a single
    /// tile does not. Sources drawn on the output's own lattice skip it and go
    /// straight into the canvas.
    pub async fn fill(&self, window: &mut Window, concurrency: usize) -> Result<u64> {
        let extent = window.extent();
        self.stream(extent, concurrency, |patch| window.absorb(patch))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Samples a single-band window, for the tests written before bands existed.
    fn sample_one(window: &Window, x: f64, y: f64) -> Option<f32> {
        let mut out = [0.0f32];
        window.sample_into(x, y, &mut out).then_some(out[0])
    }

    /// A window whose pixels can be written directly, for testing sampling.
    fn window_with(metres_per_pixel: f64, width: u32, height: u32, values: &[f32]) -> Window {
        let mut window = Window::covering(
            0.0,
            -(f64::from(height) * metres_per_pixel),
            f64::from(width) * metres_per_pixel,
            0.0,
            metres_per_pixel,
            1,
            -32767.0,
        )
        .expect("failed to allocate");
        assert_eq!((window.width, window.height), (width, height));
        window.pixels.copy_from_slice(values);
        window
    }

    #[test]
    fn a_window_snaps_outwards_to_its_own_grid() {
        let window = Window::covering(-1000.5, 499.25, -900.5, 600.75, 2.0, 1, -32767.0)
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
        let value = sample_one(&window, 0.5, -0.5).expect("expected a sample");
        assert!((value - 10.0).abs() < 1e-6, "{value}");
    }

    #[test]
    fn sampling_between_pixels_interpolates() {
        let window = window_with(1.0, 2, 2, &[0.0, 10.0, 0.0, 10.0]);
        let value = sample_one(&window, 1.0, -0.5).expect("expected a sample");
        assert!((value - 5.0).abs() < 1e-6, "{value}");
    }

    #[test]
    fn a_hole_in_any_corner_refuses_the_sample() {
        let window = window_with(1.0, 2, 2, &[10.0, 20.0, 30.0, -32767.0]);
        assert_eq!(sample_one(&window, 1.0, -1.0), None);
        // ...and the same window away from the hole is still unusable, because
        // every interior point of a 2x2 window touches all four pixels.
        assert_eq!(sample_one(&window, 0.6, -0.6), None);
    }

    #[test]
    fn sampling_outside_the_window_returns_nothing() {
        let window = window_with(1.0, 2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(sample_one(&window, -5.0, -0.5), None);
        assert_eq!(sample_one(&window, 0.5, 5.0), None);
        // The outer half-pixel has no second pixel to interpolate towards.
        assert_eq!(sample_one(&window, 0.1, -0.5), None);
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
    fn an_all_nodata_elevation_tile_is_recognised_by_its_compressed_size() {
        let limit = ELEVATION_EMPTY_TILE_LIMIT;
        assert!(
            is_empty_byte_count(0, limit),
            "a sparse block holds no data"
        );
        assert!(is_empty_byte_count(EMPTY_TILE_BYTES, limit));

        for real in [118_139, 380_423, 593_849, 689_583, 1_085_450] {
            assert!(
                !is_empty_byte_count(real, limit),
                "{real} bytes is a tile of real terrain"
            );
        }
    }

    /// Sizes measured from a real Sentinel-2 mosaic tile. Only the 213-byte
    /// tile is genuinely blank; the rest are coastline, mostly ocean nodata
    /// with a sliver of land. Reusing the elevation threshold here would throw
    /// every one of them away, which is why the limit belongs to the spec.
    #[test]
    fn imagery_tiles_are_not_guessed_at_from_their_size() {
        for bytes in [213, 723, 812, 2_549, 3_617, 4_570, 5_194, 6_276, 6_850] {
            assert!(
                !is_empty_byte_count(bytes, 0),
                "{bytes} bytes must be fetched rather than assumed empty"
            );
            assert!(
                is_empty_byte_count(bytes, ELEVATION_EMPTY_TILE_LIMIT),
                "{bytes} bytes shows why the elevation threshold cannot be reused"
            );
        }
        assert!(is_empty_byte_count(0, 0), "a sparse block is still skipped");
    }
}
