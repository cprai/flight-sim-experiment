//! Reading clipmap windows straight off the tile pyramid on disk.
//!
//! This is the whole point of the tile format. Nothing is cached and nothing is
//! resident: a request for a window opens the few tiles it crosses, reads only
//! the rows it needs out of them, copies those into the caller's staging buffer,
//! and closes the files again. The only terrain bytes this type owns between
//! calls are one tile row of scratch.
//!
//! That is affordable because tiles are written uncompressed with one row per
//! strip, so a row is a seek and a read rather than a decode. A clipmap window
//! moves by a thin strip at a time -- often a single column of 256 texels -- and
//! inflating a whole compressed tile for that would cost milliseconds per tile
//! per frame. The page cache holds the rows that get touched repeatedly, which
//! is the right place for that cache to live.
//!
//! A missing tile file means no data there, not an error. Tiles with nothing
//! under them are never written, so absence is how sparse coverage is recorded.

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use glam::{IVec2, UVec2};
use terrain_tiles::{Manifest, Srgb8, TILE_SIZE, Texel, Tile, TileGrid};
use tiff::decoder::{Decoder, DecodingResult, Limits};

use crate::terrain::geotiff::Georeferencing;
use crate::terrain::pyramid::RasterSource;

/// A texel type that knows how to come back out of a stored tile.
pub trait TileSample: Texel {
    /// How many samples per texel the file holds, which need not be how many
    /// the GPU wants: colour is stored as three bytes and read as four.
    const BANDS: usize;

    /// The texel meaning "nothing known here".
    fn nodata(nodata: f32) -> Self;

    /// Copies `out.len()` texels out of one decoded row, starting at `first`.
    fn read_span(row: &DecodingResult, first: usize, out: &mut [Self]) -> Result<()>;
}

impl TileSample for f32 {
    const BANDS: usize = 1;

    fn nodata(nodata: f32) -> Self {
        nodata
    }

    fn read_span(row: &DecodingResult, first: usize, out: &mut [Self]) -> Result<()> {
        let DecodingResult::F32(values) = row else {
            bail!("an elevation tile decoded to something other than 32-bit floats");
        };
        out.copy_from_slice(&values[first..first + out.len()]);
        Ok(())
    }
}

impl TileSample for Srgb8 {
    const BANDS: usize = 3;

    fn nodata(_: f32) -> Self {
        // Black is the mosaics' own nodata, and opaque so the shader's test is
        // about the colour rather than about coverage.
        Srgb8([0, 0, 0, 255])
    }

    fn read_span(row: &DecodingResult, first: usize, out: &mut [Self]) -> Result<()> {
        let DecodingResult::U8(values) = row else {
            bail!("a colour tile decoded to something other than bytes");
        };
        for (index, texel) in out.iter_mut().enumerate() {
            let at = (first + index) * 3;
            *texel = Srgb8([values[at], values[at + 1], values[at + 2], 255]);
        }
        Ok(())
    }
}

/// A product's tile pyramid, read on demand.
pub struct TileStore<T> {
    root: PathBuf,
    manifest: Manifest,
    grid: TileGrid,
    /// One decoded tile row. The only terrain data held between calls.
    row: RefCell<Vec<T>>,
    /// Whether a read error has already been reported.
    ///
    /// `read_rect` cannot fail by contract -- the clipmap has no way to draw a
    /// window that did not arrive -- so a broken tile fills with nodata and says
    /// so once, rather than logging on every frame for as long as the camera
    /// stays put.
    complained: Cell<bool>,
    texel: PhantomData<T>,
}

impl<T: TileSample> TileStore<T> {
    /// Opens a product directory, reading its manifest.
    pub fn open(root: &Path) -> Result<Self> {
        let manifest = Manifest::read(root)?;
        anyhow::ensure!(
            manifest.bands as usize == T::BANDS,
            "{} holds {} bands per texel, expected {}",
            root.display(),
            manifest.bands,
            T::BANDS
        );
        let store = Self {
            root: root.to_path_buf(),
            grid: manifest.grid(),
            manifest,
            row: RefCell::new(Vec::new()),
            complained: Cell::new(false),
            texel: PhantomData,
        };
        store.verify_placement()?;
        Ok(store)
    }

    /// Checks one tile's own GeoTIFF tags against what the manifest claims.
    ///
    /// Terrain is placed from the manifest, not from the tiles: reading tags for
    /// every tile on every frame would be work for nothing, and the tiles carry
    /// them mainly so other tools can open them. But a manifest that disagreed
    /// with its tiles would put the whole landscape somewhere else and nothing
    /// would say so, which is the kind of failure that is very hard to see and
    /// very easy to prevent. One tile at open time settles it.
    ///
    /// The coarsest level is used because it is one or two tiles wherever the
    /// data is, so the check does not depend on guessing which fine tiles were
    /// written. A product with nothing in it at all has nothing to check.
    fn verify_placement(&self) -> Result<()> {
        let level = self.manifest.max_level();
        let (tile, _, _) = self.manifest.tile_of_texel(level, 0, 0);
        let path = self.grid.tile_path(&self.root, level, tile);
        if !path.exists() {
            return Ok(());
        }

        let mut decoder = open_tile(&path)?;
        let placement = Georeferencing::read(&mut decoder)
            .with_context(|| format!("reading the placement of {}", path.display()))?;

        let (west, north) = self.grid.tile_origin_metres(level, tile);
        let metres = self.grid.metres_per_texel(level);
        anyhow::ensure!(
            placement.origin() == [west, north],
            "{} says it sits at {:?} but the manifest puts it at {:?}",
            path.display(),
            placement.origin(),
            [west, north]
        );
        anyhow::ensure!(
            placement.metres_per_texel_x == metres && placement.metres_per_texel_z == metres,
            "{} has {} x {} m texels but the manifest says {metres} m",
            path.display(),
            placement.metres_per_texel_x,
            placement.metres_per_texel_z
        );
        anyhow::ensure!(
            placement.width == TILE_SIZE && placement.height == TILE_SIZE,
            "{} is {} x {} texels, not {TILE_SIZE} square",
            path.display(),
            placement.width,
            placement.height
        );
        Ok(())
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Where this product's ground sits, as the renderer describes placement.
    ///
    /// Always in level-0 terms even for colour, which is stored coarser, so that
    /// the two products describe one raster and the clipmap can index both with
    /// the same texel coordinates.
    pub fn placement(&self) -> Georeferencing {
        Georeferencing::projected(
            self.manifest.extent_texels[0],
            self.manifest.extent_texels[1],
            self.manifest.base_metres_per_texel,
            self.manifest.origin_metres,
        )
    }

    /// Reports a read failure once, then stays quiet.
    fn complain(&self, error: &anyhow::Error) {
        if !self.complained.replace(true) {
            log::warn!(
                "reading terrain tiles from {} failed, drawing nodata there: {error:?}",
                self.root.display()
            );
        }
    }

    /// The stored level a request for `level` is served from, and by how many
    /// powers of two it has to be magnified.
    ///
    /// Colour is stored from level 4 up, so a request for a finer level reads
    /// the finest that exists and repeats each texel. Coarser than the top is
    /// clamped the same way an in-memory pyramid clamps, so a clipmap ring
    /// beyond the data still has something to read.
    fn stored_level(&self, level: u32) -> (u32, u32) {
        let stored = level
            .max(self.manifest.base_level)
            .min(self.manifest.max_level());
        (stored, stored - stored.min(level))
    }

    /// The clamped global texel indices, at `stored`, for a run of output texels.
    ///
    /// Clamping here is what gives the border-repeat behaviour the clipmap
    /// relies on: a window hanging off the edge of the world reads the edge
    /// texel rather than a hole. The shift is what magnifies colour, whose
    /// finest stored level is coarser than the finest the clipmap asks for.
    fn axis(&self, level: u32, stored: u32, start: i32, count: u32, vertical: bool) -> Vec<i64> {
        let shift = stored - level.min(stored);
        let origin = self.manifest.origin_texels(level);
        let stored_origin = self.manifest.origin_texels(stored);
        let size = self.manifest.size_texels(stored);

        let (first, stored_first, extent) = if vertical {
            (origin.1, stored_origin.1, i64::from(size.1))
        } else {
            (origin.0, stored_origin.0, i64::from(size.0))
        };
        let last = stored_first + extent - 1;

        (0..i64::from(count))
            .map(|step| {
                let global = first + i64::from(start) + step;
                global.div_euclid(1 << shift).clamp(stored_first, last)
            })
            .collect()
    }
}

/// Which tile a global texel index belongs to, and where inside it.
fn split(index: i64) -> (i32, usize) {
    let size = i64::from(TILE_SIZE);
    (
        index.div_euclid(size) as i32,
        index.rem_euclid(size) as usize,
    )
}

/// Groups a run of global texel indices into the tiles they fall in.
///
/// The indices are non-decreasing, so each tile owns one contiguous stretch of
/// them. Returns `(tile index, range of positions)` pairs.
fn runs(indices: &[i64]) -> Vec<(i32, std::ops::Range<usize>)> {
    let mut runs: Vec<(i32, std::ops::Range<usize>)> = Vec::new();
    for (position, &index) in indices.iter().enumerate() {
        let (tile, _) = split(index);
        match runs.last_mut() {
            Some((last, range)) if *last == tile => range.end = position + 1,
            _ => runs.push((tile, position..position + 1)),
        }
    }
    runs
}

impl<T: TileSample> RasterSource for TileStore<T> {
    fn level_count(&self) -> u32 {
        // Counted from level 0 even when nothing is stored there, because the
        // clipmap indexes both products with the same level numbers.
        self.manifest.max_level() + 1
    }

    fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]) {
        let texels: &mut [T] = bytemuck::cast_slice_mut(
            &mut out[..(size.x as usize) * (size.y as usize) * size_of::<T>()],
        );
        texels.fill(T::nodata(self.manifest.nodata));
        if size.x == 0 || size.y == 0 {
            return;
        }

        let (stored, _) = self.stored_level(level);
        let columns = self.axis(level, stored, origin.x, size.x, false);
        let rows = self.axis(level, stored, origin.y, size.y, true);

        if let Err(error) = self.gather(stored, &columns, &rows, size.x as usize, texels) {
            self.complain(&error);
        }
    }
}

impl<T: TileSample> TileStore<T> {
    /// Copies every texel the request needs, one tile at a time.
    fn gather(
        &self,
        stored: u32,
        columns: &[i64],
        rows: &[i64],
        stride: usize,
        out: &mut [T],
    ) -> Result<()> {
        let column_runs = runs(columns);
        let mut scratch = self.row.borrow_mut();
        scratch.resize(TILE_SIZE as usize, T::nodata(self.manifest.nodata));

        for (tile_y, row_range) in runs(rows) {
            for (tile_x, column_range) in &column_runs {
                let tile = Tile::new(*tile_x, tile_y);
                let path = self.grid.tile_path(&self.root, stored, tile);
                // Absent is the ordinary case: a tile with nothing under it is
                // never written, and the nodata already in `out` is the answer.
                if !path.exists() {
                    continue;
                }
                let mut decoder = open_tile(&path)?;

                // Repeated output rows land on the same stored row when a
                // coarse level is being magnified, so the last one read is kept
                // rather than decoded again.
                let mut loaded: Option<usize> = None;
                for position in row_range.clone() {
                    let (_, row_in_tile) = split(rows[position]);
                    if loaded != Some(row_in_tile) {
                        let decoded =
                            decoder.read_chunk(row_in_tile as u32).with_context(|| {
                                format!("reading row {row_in_tile} of {}", path.display())
                            })?;
                        T::read_span(&decoded, 0, &mut scratch[..])?;
                        loaded = Some(row_in_tile);
                    }
                    let destination = position * stride;
                    for column in column_range.clone() {
                        let (_, column_in_tile) = split(columns[column]);
                        out[destination + column] = scratch[column_in_tile];
                    }
                }
            }
        }
        Ok(())
    }
}

fn open_tile(path: &Path) -> Result<Decoder<std::io::BufReader<std::fs::File>>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    Decoder::new(std::io::BufReader::new(file))
        .map(|decoder| decoder.with_limits(Limits::unlimited()))
        .with_context(|| format!("reading the header of {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrain_tiles::COLOUR_BASE_LEVEL;

    const NODATA: f32 = -32767.0;

    /// Writes a tile the same way `terrain-download` does: uncompressed, one row
    /// per strip, placed by its own tiepoint. Spelled out here rather than
    /// shared, so this is testing that the reader accepts the format rather than
    /// that two crates agree on one definition of it.
    fn write_tile<C>(path: &Path, grid: &TileGrid, level: u32, tile: Tile, data: &[C::Inner])
    where
        C: tiff::encoder::colortype::ColorType,
        [C::Inner]: tiff::encoder::TiffValue,
    {
        std::fs::create_dir_all(path.parent().expect("no parent")).expect("failed to create");
        let file = std::fs::File::create(path).expect("failed to create");
        let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))
            .expect("failed to start")
            .with_compression(tiff::encoder::Compression::Uncompressed);
        let mut image = encoder
            .new_image::<C>(TILE_SIZE, TILE_SIZE)
            .expect("failed to start the image");
        image.rows_per_strip(1).expect("failed to set strips");

        let (west, north) = grid.tile_origin_metres(level, tile);
        let metres = grid.metres_per_texel(level);
        {
            let directory = image.encoder();
            directory
                .write_tag(tiff::tags::Tag::Unknown(33550), &[metres, metres, 0.0][..])
                .expect("failed to write scale");
            directory
                .write_tag(
                    tiff::tags::Tag::Unknown(33922),
                    &[0.0f64, 0.0, 0.0, west, north, 0.0][..],
                )
                .expect("failed to write tiepoint");
            // Projected, area pixels, EPSG:3979, metres.
            directory
                .write_tag(
                    tiff::tags::Tag::Unknown(34735),
                    &[
                        1u16, 1, 0, 4, 1024, 0, 1, 1, 1025, 0, 1, 1, 3072, 0, 1, 3979, 3076, 0, 1,
                        9001,
                    ][..],
                )
                .expect("failed to write geo keys");
        }
        image.write_data(data).expect("failed to write texels");
    }

    fn manifest(product: &str, base_level: u32, level_count: u32, bands: u32) -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: product.into(),
            epsg: 3979,
            tile_size: TILE_SIZE,
            base_level,
            level_count,
            base_metres_per_texel: 1.0,
            // Two level-0 tiles square, and a multiple of the 8192 m snap.
            origin_metres: [-1_974_272.0, 524_288.0],
            extent_texels: [8_192, 8_192],
            bands,
            nodata: NODATA,
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("flight-sim-tiles-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    /// Wraps a global texel index into the range the stamp below can hold.
    ///
    /// Global indices run to a couple of million, and a stamp combining two of
    /// them would land well past the 2^24 integers an `f32` holds exactly --
    /// which would make the fixture, not the reader, the thing under test.
    /// Wrapping keeps every stamp exact and still unique across any window
    /// smaller than 2048 texels, which is far larger than anything read here.
    const STAMP_PERIOD: i64 = 2048;

    fn stamp(global_x: i64, global_y: i64) -> f32 {
        (global_x.rem_euclid(STAMP_PERIOD) * STAMP_PERIOD + global_y.rem_euclid(STAMP_PERIOD))
            as f32
    }

    /// Every texel carries its own global position, so a misplaced read is
    /// obvious rather than merely wrong-looking.
    fn stamped(tile: Tile) -> Vec<f32> {
        let size = TILE_SIZE as i64;
        (0..(TILE_SIZE as usize).pow(2))
            .map(|i| {
                let (x, y) = (i as i64 % size, i as i64 / size);
                stamp(i64::from(tile.x) * size + x, i64::from(tile.y) * size + y)
            })
            .collect()
    }

    /// Four elevation tiles, laid out as a 2x2 covering the manifest's extent.
    fn elevation_store(name: &str, missing: Option<Tile>) -> TileStore<f32> {
        let root = temp_root(name);
        let manifest = manifest("dtm", 0, 5, 1);
        manifest.write(&root).expect("failed to write the manifest");
        let grid = manifest.grid();

        let (first, _, _) = manifest.tile_of_texel(0, 0, 0);
        for dy in 0..2 {
            for dx in 0..2 {
                let tile = Tile::new(first.x + dx, first.y + dy);
                if Some(tile) == missing {
                    continue;
                }
                write_tile::<tiff::encoder::colortype::Gray32Float>(
                    &grid.tile_path(&root, 0, tile),
                    &grid,
                    0,
                    tile,
                    &stamped(tile),
                );
            }
        }
        // The coarsest level is what `open` checks its placement against.
        let top = manifest.max_level();
        let (top_tile, _, _) = manifest.tile_of_texel(top, 0, 0);
        write_tile::<tiff::encoder::colortype::Gray32Float>(
            &grid.tile_path(&root, top, top_tile),
            &grid,
            top,
            top_tile,
            &vec![1.0f32; (TILE_SIZE as usize).pow(2)],
        );

        TileStore::open(&root).expect("failed to open")
    }

    fn read(store: &TileStore<f32>, level: u32, origin: IVec2, size: UVec2) -> Vec<f32> {
        let mut out = vec![0f32; (size.x as usize) * (size.y as usize)];
        store.read_rect(level, origin, size, bytemuck::cast_slice_mut(&mut out));
        out
    }

    /// What the texel at a raster position should hold, given how `stamped`
    /// builds it. Computed from the manifest rather than from the reader.
    fn expected(manifest: &Manifest, column: i64, row: i64) -> f32 {
        let (origin_column, origin_row) = manifest.origin_texels(0);
        stamp(origin_column + column, origin_row + row)
    }

    #[test]
    fn a_window_spanning_four_tiles_is_assembled_in_the_right_order() {
        let store = elevation_store("four-tiles", None);
        // Straddling the seam where all four tiles meet.
        let origin = IVec2::new(TILE_SIZE as i32 - 2, TILE_SIZE as i32 - 2);
        let got = read(&store, 0, origin, UVec2::splat(4));

        for row in 0..4 {
            for column in 0..4 {
                let want = expected(
                    store.manifest(),
                    i64::from(origin.x) + column,
                    i64::from(origin.y) + row,
                );
                assert_eq!(
                    got[(row * 4 + column) as usize],
                    want,
                    "texel ({column}, {row})"
                );
            }
        }
    }

    /// A tile with nothing under it is never written, so the reader has to treat
    /// absence as a hole rather than as a failure -- and must still deliver the
    /// tiles either side of it.
    #[test]
    fn a_missing_tile_reads_as_nodata_without_disturbing_its_neighbours() {
        let store = elevation_store("missing", None);
        let (first, _, _) = store.manifest().tile_of_texel(0, 0, 0);
        let holed = elevation_store("missing-hole", Some(Tile::new(first.x + 1, first.y)));

        let origin = IVec2::new(TILE_SIZE as i32 - 2, 4);
        let whole = read(&store, 0, origin, UVec2::new(4, 2));
        let got = read(&holed, 0, origin, UVec2::new(4, 2));

        for row in 0..2usize {
            // The first two columns are in the surviving western tile.
            for column in 0..2usize {
                assert_eq!(got[row * 4 + column], whole[row * 4 + column], "kept");
            }
            // The last two are in the tile that was never written.
            for column in 2..4usize {
                assert_eq!(got[row * 4 + column], NODATA, "hole");
            }
        }
    }

    #[test]
    fn reading_past_the_edge_repeats_the_border_texel() {
        let store = elevation_store("border", None);
        // Two texels north and west of the raster, two inside it.
        let origin = IVec2::new(-2, -2);
        let got = read(&store, 0, origin, UVec2::splat(4));

        for row in 0..4i64 {
            for column in 0..4i64 {
                let want = expected(
                    store.manifest(),
                    (i64::from(origin.x) + column).max(0),
                    (i64::from(origin.y) + row).max(0),
                );
                assert_eq!(got[(row * 4 + column) as usize], want, "({column}, {row})");
            }
        }
        // Which must actually have exercised the clamp.
        assert_eq!(got[0], expected(store.manifest(), 0, 0));
        assert_ne!(got[0], got[15], "the two corners are different texels");
    }

    /// The clipmap refreshes a moved window as thin strips. Those have to agree
    /// with a full read of the same ground, or the terrain would depend on how
    /// the camera got there.
    #[test]
    fn a_thin_strip_reads_the_same_texels_as_the_window_containing_it() {
        let store = elevation_store("strips", None);
        let origin = IVec2::new(500, 500);
        let size = UVec2::new(24, 24);
        let whole = read(&store, 0, origin, size);

        for column in 0..size.x {
            let strip = read(
                &store,
                0,
                IVec2::new(origin.x + column as i32, origin.y),
                UVec2::new(1, size.y),
            );
            for row in 0..size.y as usize {
                assert_eq!(
                    strip[row],
                    whole[row * size.x as usize + column as usize],
                    "column {column}, row {row}"
                );
            }
        }
    }

    #[test]
    fn a_coarser_level_reads_from_its_own_tiles() {
        let store = elevation_store("levels", None);
        // The coarsest level was written as a constant, so it is unmistakable.
        let top = store.manifest().max_level();
        let got = read(&store, top, IVec2::ZERO, UVec2::splat(4));
        assert!(got.iter().all(|&v| v == 1.0), "got {got:?}");
    }

    /// Colour is stored from level 4 up, so the clipmap's finer levels have to
    /// be served by magnifying it. Sixteen level-0 texels share one stored texel.
    #[test]
    fn colour_magnifies_the_finest_level_it_actually_has() {
        let root = temp_root("colour");
        let manifest = manifest("albedo", COLOUR_BASE_LEVEL, 1, 3);
        manifest.write(&root).expect("failed to write the manifest");
        let grid = manifest.grid();

        let (tile, _, _) = manifest.tile_of_texel(COLOUR_BASE_LEVEL, 0, 0);
        let mut data = vec![0u8; (TILE_SIZE as usize).pow(2) * 3];
        // A distinct colour per stored column, so magnification is visible.
        for i in 0..(TILE_SIZE as usize).pow(2) {
            let column = (i % TILE_SIZE as usize) as u8;
            data[i * 3..i * 3 + 3].copy_from_slice(&[column, 40, 200]);
        }
        write_tile::<tiff::encoder::colortype::RGB8>(
            &grid.tile_path(&root, COLOUR_BASE_LEVEL, tile),
            &grid,
            COLOUR_BASE_LEVEL,
            tile,
            &data,
        );

        let store = TileStore::<Srgb8>::open(&root).expect("failed to open");
        assert_eq!(
            store.level_count(),
            COLOUR_BASE_LEVEL + 1,
            "levels are numbered from zero even where nothing is stored"
        );

        let magnified = 1 << COLOUR_BASE_LEVEL;
        let mut out = vec![Srgb8::default(); 64];
        store.read_rect(
            0,
            IVec2::ZERO,
            UVec2::new(64, 1),
            bytemuck::cast_slice_mut(&mut out),
        );
        for (index, texel) in out.iter().enumerate() {
            let stored_column = (index / magnified) as u8;
            assert_eq!(
                *texel,
                Srgb8([stored_column, 40, 200, 255]),
                "level-0 texel {index} should come from stored column {stored_column}"
            );
        }
    }

    /// A manifest that disagreed with its tiles would put the whole landscape
    /// somewhere else without failing, which is the one thing `open` checks for.
    #[test]
    fn a_tile_placed_somewhere_else_is_refused() {
        let root = temp_root("mismatch");
        let manifest = manifest("dtm", 0, 5, 1);
        manifest.write(&root).expect("failed to write the manifest");
        let grid = manifest.grid();

        let top = manifest.max_level();
        let (tile, _, _) = manifest.tile_of_texel(top, 0, 0);
        // Written as though it were the tile to the east of where it is filed.
        write_tile::<tiff::encoder::colortype::Gray32Float>(
            &grid.tile_path(&root, top, tile),
            &grid,
            top,
            Tile::new(tile.x + 1, tile.y),
            &vec![1.0f32; (TILE_SIZE as usize).pow(2)],
        );

        let error = match TileStore::<f32>::open(&root) {
            Ok(_) => panic!("a displaced tile should be refused"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("manifest puts it at"), "{error}");
    }
}
