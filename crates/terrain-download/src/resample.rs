//! Turning projected source pixels into a block of the output tile grid.
//!
//! The output is drawn on EPSG:3979 in metres, which is the grid HRDEM is
//! already published on. For elevation that makes the resample an identity: an
//! output texel centre lands exactly on a source pixel centre, and the fast path
//! below copies it rather than interpolating. Colour comes from Web Mercator and
//! genuinely has to be reprojected, so both paths exist.
//!
//! Work is done a block at a time rather than over the whole download, because
//! the whole download no longer fits anywhere. A block is a few tiles square; it
//! is filled, cut into tiles, written, and dropped before the next one starts.
//!
//! One metre is preferred, two metre fills what is left, and MRDEM at 30 m
//! fills whatever neither survey reached. The switch is abrupt by design: a
//! texel comes from one source or another, never a blend of them. Where two
//! disagree in height -- different survey years, all on CGVD2013 -- that shows
//! up as a step at the boundary rather than being smoothed into something
//! untraceable. The step against MRDEM is the largest of the three, and it is
//! still preferable to inventing a ramp between a measurement and a model.

use anyhow::{Context, Result};
use terrain_tiles::TILE_SIZE;

use crate::project::Projector;
use crate::source::{Patch, Window};

/// A rectangle of ground, in one CRS's metres.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MetreExtent {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl MetreExtent {
    /// An extent that nothing has been added to yet.
    fn empty() -> Self {
        Self {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    fn include(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    /// Grows the extent by `margin` on every side.
    pub fn expanded(self, margin: f64) -> Self {
        Self {
            min_x: self.min_x - margin,
            min_y: self.min_y - margin,
            max_x: self.max_x + margin,
            max_y: self.max_y + margin,
        }
    }
}

/// How many texels of slack to leave around a source window.
///
/// Bilinear interpolation reads the pixel on either side of the sample point,
/// so the window has to reach one pixel beyond the outermost sample; two makes
/// it robust to the rounding in between.
pub const BILINEAR_MARGIN: f64 = 2.0;

/// How many points to sample along each edge when reprojecting an extent.
///
/// A rectangle on one projected grid is a slightly curved quadrilateral on
/// another, so the envelope's extremes can sit part-way along an edge rather
/// than at a corner -- meridian convergence alone rotates a Lambert-aligned box
/// by about 25 degrees at 123W. Walking the boundary needs no special case for
/// that. 256 points across a block of at most a few tens of kilometres puts
/// consecutive samples a hundred metres or so apart, comfortably inside the
/// bilinear margin.
const EDGE_SAMPLES: u32 = 256;

/// Where a block of the output sits, and how finely it samples the ground.
///
/// The origin is the *edge* of the north-west texel, not its centre, matching
/// the PixelIsArea convention the tiles are written with.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Grid {
    /// Easting of the western edge of column 0.
    pub west: f64,
    /// Northing of the northern edge of row 0.
    pub north: f64,
    pub width: u32,
    pub height: u32,
    pub metres_per_texel: f64,
}

impl Grid {
    /// The ground this block covers.
    pub fn extent(&self) -> MetreExtent {
        MetreExtent {
            min_x: self.west,
            min_y: self.north - f64::from(self.height) * self.metres_per_texel,
            max_x: self.west + f64::from(self.width) * self.metres_per_texel,
            max_y: self.north,
        }
    }

    /// The ground position a texel samples, at its centre.
    pub fn centre_of(&self, column: u32, row: u32) -> (f64, f64) {
        (
            self.west + (f64::from(column) + 0.5) * self.metres_per_texel,
            self.north - (f64::from(row) + 0.5) * self.metres_per_texel,
        )
    }

    pub fn texel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Whether pixels of `metres_per_pixel`, laid out from (`west`, `north`),
    /// land exactly on this grid's texels.
    ///
    /// When they do, a source tile can be copied straight into the canvas and
    /// never needs staging in a window at all. This is the case the whole
    /// choice of EPSG:3979 was made for: HRDEM sits on an integer-metre lattice
    /// and the tile grid's boundaries are multiples of 512 m, so an output
    /// texel centre falls precisely on a source pixel centre.
    ///
    /// A thousandth of a texel is the tolerance, because both origins are
    /// snapped to whole metres -- a real alignment is exact and anything else
    /// is far away.
    pub fn aligns_with(&self, west: f64, north: f64, metres_per_pixel: f64) -> bool {
        if metres_per_pixel != self.metres_per_texel {
            return false;
        }
        let columns = (self.west - west) / metres_per_pixel;
        let rows = (north - self.north) / metres_per_pixel;
        (columns - columns.round()).abs() <= 1e-3 && (rows - rows.round()).abs() <= 1e-3
    }
}

/// The extent a source has to cover to fill `grid`.
///
/// `projector` is `None` when the source is drawn on the same CRS as the
/// output, which is the case for both HRDEM mosaics; then the answer is the
/// block's own extent plus a margin. For a source on another CRS the block's
/// boundary is walked and reprojected.
pub fn source_extent(
    grid: &Grid,
    projector: Option<&Projector>,
    source_metres_per_pixel: f64,
) -> Result<MetreExtent> {
    let margin = BILINEAR_MARGIN * source_metres_per_pixel;
    let Some(projector) = projector else {
        return Ok(grid.extent().expanded(margin));
    };

    let extent = grid.extent();
    let mut outline = Vec::with_capacity(4 * (EDGE_SAMPLES as usize + 1));
    for step in 0..=EDGE_SAMPLES {
        let fraction = f64::from(step) / f64::from(EDGE_SAMPLES);
        let x = extent.min_x + fraction * (extent.max_x - extent.min_x);
        let y = extent.min_y + fraction * (extent.max_y - extent.min_y);
        outline.push((x, extent.min_y));
        outline.push((x, extent.max_y));
        outline.push((extent.min_x, y));
        outline.push((extent.max_x, y));
    }

    projector
        .to_source(&mut outline)
        .context("projecting the outline of a block")?;

    let mut projected = MetreExtent::empty();
    for (x, y) in outline {
        anyhow::ensure!(
            x.is_finite() && y.is_finite(),
            "a block of the requested box does not project onto the source grid"
        );
        projected.include(x, y);
    }
    Ok(projected.expanded(margin))
}

/// Where a texel's value came from.
///
/// No longer written to disk -- the tiles carry elevation alone -- but still
/// tracked, because it is what implements the preference between the sources
/// and what the percentages printed at the end are counted from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provenance {
    /// No source had data here; the elevation is the nodata value.
    Missing = 0,
    /// From the one-metre mosaic.
    OneMetre = 1,
    /// From the two-metre mosaic, interpolated onto the output grid.
    TwoMetre = 2,
    /// From MRDEM at 30 m, interpolated onto the output grid.
    ///
    /// Worth telling apart from the two above rather than lumping in as
    /// "filled": it is thirty times coarser than the ground around it, so the
    /// share of a download that comes from here is the one number that says how
    /// much of the box is coarse. A run reporting most of its texels from this
    /// tier has been pointed at ground HRDEM never surveyed.
    ThirtyMetre = 3,
}

impl Provenance {
    /// Marks a texel as filled where there is only one source to fill it from.
    ///
    /// Colour comes from a single mosaic, but the canvas still has to know
    /// which texels are done. Reusing the first tier says that without adding a
    /// variant that would mean nothing for elevation.
    pub const FILLED: Self = Self::OneMetre;
}

/// What each source contributed, as a count of output texels.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Tally {
    pub one_metre: u64,
    pub two_metre: u64,
    pub thirty_metre: u64,
    pub missing: u64,
}

impl Tally {
    pub fn total(&self) -> u64 {
        self.one_metre + self.two_metre + self.thirty_metre + self.missing
    }

    /// Accumulates another block's counts into this one.
    pub fn add(&mut self, other: Tally) {
        self.one_metre += other.one_metre;
        self.two_metre += other.two_metre;
        self.thirty_metre += other.thirty_metre;
        self.missing += other.missing;
    }

    /// The four shares as percentages, which is what gets printed.
    pub fn percentages(&self) -> Shares {
        let total = self.total();
        if total == 0 {
            return Shares::default();
        }
        let share = |n: u64| 100.0 * n as f64 / total as f64;
        Shares {
            one_metre: share(self.one_metre),
            two_metre: share(self.two_metre),
            thirty_metre: share(self.thirty_metre),
            missing: share(self.missing),
        }
    }
}

/// The tiers' shares of a download, as percentages.
///
/// A named struct rather than a tuple because there are now four of them and
/// three are easy to transpose. `let (one, two, none) = ...` silently became
/// wrong when the third tier was added; `shares.missing` cannot.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct Shares {
    pub one_metre: f64,
    pub two_metre: f64,
    pub thirty_metre: f64,
    pub missing: f64,
}

/// One block of the output being filled in.
pub struct Canvas {
    pub grid: Grid,
    pub bands: usize,
    nodata: f32,
    values: Vec<f32>,
    provenance: Vec<u8>,
}

impl Canvas {
    pub fn new(grid: Grid, bands: usize, nodata: f32) -> Result<Self> {
        anyhow::ensure!(bands >= 1, "a canvas needs at least one band");
        let mut canvas = Self {
            grid,
            bands,
            nodata,
            values: Vec::new(),
            provenance: Vec::new(),
        };
        canvas.reset(grid)?;
        Ok(canvas)
    }

    /// Points the canvas at another block, clearing it and keeping its buffers.
    ///
    /// Every block of a download is the same size but for the last in a row or
    /// column, so the allocation made for the first serves all of them.
    ///
    /// Reuse matters more than it looks. Allocating a fresh canvas per block
    /// meant handing back tens of megabytes to whichever allocator arena the
    /// dropping thread belonged to, and glibc does not readily return those to
    /// the kernel: peak RSS climbed with every block -- 231, 247, 257, 267 MiB
    /// across a four-block download -- even though only one canvas was ever
    /// live. Running the same binary under `MALLOC_ARENA_MAX=2` came out 64 MiB
    /// lower, which is what identified the cause.
    pub fn reset(&mut self, grid: Grid) -> Result<()> {
        let texels = (grid.width as usize)
            .checked_mul(grid.height as usize)
            .context("the block does not fit in memory")?;
        let count = texels
            .checked_mul(self.bands)
            .context("the block does not fit in memory")?;

        self.grid = grid;
        self.values.clear();
        self.values
            .try_reserve(count)
            .context("the block does not fit in memory")?;
        self.values.resize(count, self.nodata);

        self.provenance.clear();
        self.provenance
            .try_reserve(texels)
            .context("the block does not fit in memory")?;
        self.provenance.resize(texels, Provenance::Missing as u8);
        Ok(())
    }

    fn is_filled(&self, index: usize) -> bool {
        self.provenance[index] != Provenance::Missing as u8
    }

    /// The data bands, interleaved, row-major from the north-west corner.
    ///
    /// Only the tests read the whole block; the tool cuts it into tiles.
    #[cfg(test)]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Counts what each source ended up contributing to this block.
    pub fn tally(&self) -> Tally {
        let mut tally = Tally::default();
        for &source in &self.provenance {
            if source == Provenance::OneMetre as u8 {
                tally.one_metre += 1;
            } else if source == Provenance::TwoMetre as u8 {
                tally.two_metre += 1;
            } else if source == Provenance::ThirtyMetre as u8 {
                tally.thirty_metre += 1;
            } else {
                tally.missing += 1;
            }
        }
        tally
    }

    /// Whether any texel is still waiting for a value.
    pub fn has_holes(&self) -> bool {
        self.provenance.contains(&(Provenance::Missing as u8))
    }

    /// Copies one decoded source tile straight in, and returns how many texels
    /// it filled.
    ///
    /// Only for sources on this canvas's own lattice -- [`Grid::aligns_with`]
    /// is the test, and the caller is expected to have made it. Taking tiles
    /// one at a time is what keeps source pixels out of memory: the tile is
    /// dropped the moment this returns, where staging a whole block in a window
    /// first held a second copy of everything the block covered.
    ///
    /// Copying is also more correct than interpolating, not merely faster.
    /// Bilinear refuses any sample whose four neighbours include a hole, so
    /// interpolating an aligned grid would erode a texel of real data from the
    /// edge of every gap for nothing.
    ///
    /// Texels that already have a value are left alone, which is what makes a
    /// later pass a fallback rather than an overwrite.
    pub fn absorb(&mut self, patch: &Patch, provenance: Provenance) -> u64 {
        debug_assert_eq!(self.bands, patch.bands);
        debug_assert!(
            self.grid
                .aligns_with(patch.west, patch.north, patch.metres_per_pixel)
        );

        let metres = self.grid.metres_per_texel;
        let column_offset = ((patch.west - self.grid.west) / metres).round() as i64;
        let row_offset = ((self.grid.north - patch.north) / metres).round() as i64;
        let mut filled = 0;

        for row in 0..patch.height {
            let y = row_offset + row as i64;
            if y < 0 || y >= i64::from(self.grid.height) {
                continue;
            }
            let first = (y as usize) * (self.grid.width as usize);

            for column in 0..patch.width {
                let x = column_offset + column as i64;
                if x < 0 || x >= i64::from(self.grid.width) {
                    continue;
                }
                let index = first + x as usize;
                if self.is_filled(index) {
                    continue;
                }
                let Some(sample) = patch.texel(column, row) else {
                    continue;
                };
                let at = index * self.bands;
                self.values[at..at + self.bands].copy_from_slice(sample);
                self.provenance[index] = provenance as u8;
                filled += 1;
            }
        }

        filled
    }

    /// Fills every texel this window can supply, leaving the rest alone.
    ///
    /// Only texels that are still holes are considered, so calling this with
    /// the two-metre window after the one-metre window is what implements the
    /// preference between them.
    ///
    /// `projector` is `None` when the window shares this canvas's CRS.
    pub fn fill_from(
        &mut self,
        window: &Window,
        projector: Option<&Projector>,
        provenance: Provenance,
    ) -> Result<u64> {
        anyhow::ensure!(
            window.bands == self.bands,
            "the source has {} bands but the output has {}",
            window.bands,
            self.bands
        );

        let mut row = vec![(0.0f64, 0.0f64); self.grid.width as usize];
        let mut sample = vec![self.nodata; self.bands];
        let mut filled = 0;

        for y in 0..self.grid.height {
            let first = (y as usize) * (self.grid.width as usize);

            // Skip a row that has nothing left to fill, which after the
            // one-metre pass is usually most of them.
            if (0..self.grid.width as usize).all(|x| self.is_filled(first + x)) {
                continue;
            }

            for (x, point) in row.iter_mut().enumerate() {
                *point = self.grid.centre_of(x as u32, y);
            }
            if let Some(projector) = projector {
                projector
                    .to_source(&mut row)
                    .with_context(|| format!("projecting output row {y}"))?;
            }

            for (x, &(metres_x, metres_y)) in row.iter().enumerate() {
                let index = first + x;
                if self.is_filled(index) {
                    continue;
                }
                if window.sample_into(metres_x, metres_y, &mut sample) {
                    let at = index * self.bands;
                    self.values[at..at + self.bands].copy_from_slice(&sample);
                    self.provenance[index] = provenance as u8;
                    filled += 1;
                }
            }
        }

        Ok(filled)
    }

    /// The extent, in the source's metres, of the texels still unfilled.
    ///
    /// Used to size the two-metre window: there is no point fetching tiles for
    /// ground the one-metre mosaic already covered. Returns `None` when nothing
    /// is left to fill.
    pub fn hole_extent(
        &self,
        projector: Option<&Projector>,
        source_metres_per_pixel: f64,
    ) -> Result<Option<MetreExtent>> {
        let mut row = vec![(0.0f64, 0.0f64); self.grid.width as usize];
        let mut extent = MetreExtent::empty();
        let mut any = false;

        for y in 0..self.grid.height {
            let first = (y as usize) * (self.grid.width as usize);
            if (0..self.grid.width as usize).all(|x| self.is_filled(first + x)) {
                continue;
            }

            for (x, point) in row.iter_mut().enumerate() {
                *point = self.grid.centre_of(x as u32, y);
            }
            if let Some(projector) = projector {
                projector
                    .to_source(&mut row)
                    .with_context(|| format!("projecting output row {y}"))?;
            }

            for (x, &(metres_x, metres_y)) in row.iter().enumerate() {
                if self.is_filled(first + x) {
                    continue;
                }
                any = true;
                extent.include(metres_x, metres_y);
            }
        }

        if !any {
            return Ok(None);
        }
        Ok(Some(
            extent.expanded(BILINEAR_MARGIN * source_metres_per_pixel),
        ))
    }

    /// Copies one tile's worth of texels out of the block, into `out`.
    ///
    /// `column` and `row` are in tiles, measured from the block's north-west
    /// corner. Returns `false` when every texel is nodata, which is the signal
    /// not to write the tile at all -- an unwritten tile is how the output stays
    /// sparse over the large parts of any box that HRDEM does not cover.
    pub fn tile_samples(&self, column: u32, row: u32, out: &mut Vec<f32>) -> bool {
        let size = TILE_SIZE as usize;
        let first_column = (column as usize) * size;
        let first_row = (row as usize) * size;
        debug_assert!(first_column + size <= self.grid.width as usize);
        debug_assert!(first_row + size <= self.grid.height as usize);

        out.clear();
        out.reserve(size * size * self.bands);
        let mut any = false;
        for y in 0..size {
            let start = ((first_row + y) * (self.grid.width as usize) + first_column) * self.bands;
            out.extend_from_slice(&self.values[start..start + size * self.bands]);

            let texel = (first_row + y) * (self.grid.width as usize) + first_column;
            any |= (0..size).any(|x| self.is_filled(texel + x));
        }
        any
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::{EPSG_LAMBERT, EPSG_WEB_MERCATOR};

    const NODATA: f32 = -32767.0;

    /// A block a couple of tiles across, placed where HRDEM actually has data.
    fn grid(metres_per_texel: f64, width: u32, height: u32) -> Grid {
        Grid {
            west: -1_974_272.0,
            north: 524_288.0,
            width,
            height,
            metres_per_texel,
        }
    }

    /// A window covering `grid`, whose texels take the value `value` gives for
    /// their centre in metres.
    fn window_over(grid: &Grid, metres_per_pixel: f64, value: impl Fn(f64, f64) -> f32) -> Window {
        let extent = source_extent(grid, None, metres_per_pixel).expect("failed to extend");
        let mut window = Window::covering(
            extent.min_x,
            extent.min_y,
            extent.max_x,
            extent.max_y,
            metres_per_pixel,
            1,
            NODATA,
        )
        .expect("failed to allocate");

        for y in 0..window.height {
            for x in 0..window.width {
                let metres_x = window.origin_x + (f64::from(x) + 0.5) * metres_per_pixel;
                let metres_y = window.origin_y - (f64::from(y) + 0.5) * metres_per_pixel;
                window.set_for_test(x, y, &[value(metres_x, metres_y)]);
            }
        }
        window
    }

    #[test]
    fn a_grids_extent_matches_its_texel_count() {
        let grid = grid(1.0, 512, 256);
        let extent = grid.extent();
        assert_eq!(extent.max_x - extent.min_x, 512.0);
        assert_eq!(extent.max_y - extent.min_y, 256.0);
        assert_eq!(extent.max_y, grid.north);
        assert_eq!(extent.min_x, grid.west);
    }

    #[test]
    fn texel_centres_sit_half_a_texel_inside_the_edges() {
        let grid = grid(2.0, 8, 8);
        assert_eq!(grid.centre_of(0, 0), (grid.west + 1.0, grid.north - 1.0));
        assert_eq!(grid.centre_of(7, 7), (grid.west + 15.0, grid.north - 15.0));
    }

    /// Elevation never leaves EPSG:3979, so its source extent is the block plus
    /// the bilinear margin and nothing else.
    fn margin(metres_per_pixel: f64) -> f64 {
        BILINEAR_MARGIN * metres_per_pixel
    }

    #[test]
    fn a_source_on_the_same_crs_needs_only_the_block_and_a_margin() {
        let grid = grid(1.0, 64, 64);
        let extent = source_extent(&grid, None, 2.0).expect("failed");
        assert_eq!(extent.min_x, grid.west - margin(2.0));
        assert_eq!(extent.max_y, grid.north + margin(2.0));
    }

    /// Web Mercator is a different grid entirely, so the block's outline has to
    /// be walked. The result must contain every corner, and be larger than the
    /// block because the two grids are rotated relative to each other.
    #[test]
    fn a_source_on_another_crs_covers_every_corner_of_the_block() {
        let projector =
            Projector::between(EPSG_LAMBERT, EPSG_WEB_MERCATOR).expect("failed to build");
        let grid = grid(1.0, 4096, 4096);
        let extent = source_extent(&grid, Some(&projector), 19.0).expect("failed");

        let block = grid.extent();
        for (x, y) in [
            (block.min_x, block.min_y),
            (block.min_x, block.max_y),
            (block.max_x, block.min_y),
            (block.max_x, block.max_y),
        ] {
            let mut corner = [(x, y)];
            projector.to_source(&mut corner).expect("failed to project");
            let (px, py) = corner[0];
            assert!(
                px >= extent.min_x && px <= extent.max_x,
                "easting {px} outside {}..{}",
                extent.min_x,
                extent.max_x
            );
            assert!(
                py >= extent.min_y && py <= extent.max_y,
                "northing {py} outside {}..{}",
                extent.min_y,
                extent.max_y
            );
        }
    }

    #[test]
    fn a_fresh_canvas_is_all_holes() {
        let grid = grid(1.0, 32, 32);
        let canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        let tally = canvas.tally();
        assert_eq!(tally.missing, grid.texel_count());
        assert_eq!(tally.one_metre, 0);
        assert!(canvas.has_holes());
    }

    #[test]
    fn a_covering_window_fills_every_texel_once() {
        let grid = grid(1.0, 64, 64);
        let window = window_over(&grid, 1.0, |_, _| 500.0);

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        let filled = canvas
            .fill_from(&window, None, Provenance::OneMetre)
            .expect("failed to fill");

        assert_eq!(filled, grid.texel_count());
        assert!(!canvas.has_holes());
        assert_eq!(canvas.tally().one_metre, grid.texel_count());
        for &value in canvas.values() {
            assert_eq!(value, 500.0);
        }
    }

    /// A patch of source pixels laid on part of a grid, whose values come from
    /// `value` applied to each pixel's centre in metres.
    fn patch_over(
        grid: &Grid,
        column: usize,
        row: usize,
        size: usize,
        value: impl Fn(f64, f64) -> f32,
    ) -> Patch {
        let metres = grid.metres_per_texel;
        let west = grid.west + column as f64 * metres;
        let north = grid.north - row as f64 * metres;
        let mut values = Vec::with_capacity(size * size);
        for y in 0..size {
            for x in 0..size {
                values.push(value(
                    west + (x as f64 + 0.5) * metres,
                    north - (y as f64 + 0.5) * metres,
                ));
            }
        }
        Patch {
            west,
            north,
            metres_per_pixel: metres,
            width: size,
            height: size,
            stride: size,
            bands: 1,
            nodata: NODATA,
            values,
        }
    }

    /// HRDEM's integer-metre lattice against the tile grid's 512 m boundaries,
    /// which is the case the direct copy exists for.
    #[test]
    fn a_source_on_whole_metres_aligns_with_the_output() {
        let grid = grid(1.0, 64, 64);
        assert!(grid.aligns_with(-2_000_000.0, 1_000_000.0, 1.0));
        // Half a metre off is not alignment.
        assert!(!grid.aligns_with(-1_999_999.5, 1_000_000.0, 1.0));
        // Nor is the right lattice at the wrong pixel size: a two-metre source
        // has to be interpolated onto a one-metre grid.
        assert!(!grid.aligns_with(-2_000_000.0, 1_000_000.0, 2.0));
    }

    /// The claim the whole choice of EPSG:3979 rests on: a one-metre source on
    /// the output's own lattice is copied, not interpolated, so the values come
    /// out bit for bit identical to what the mosaic holds.
    #[test]
    fn an_aligned_patch_is_copied_exactly() {
        let grid = grid(1.0, 64, 64);
        // A ramp with plenty of bits set, so an interpolation that happened to
        // land near the right answer would still differ.
        let value = |x: f64, y: f64| (x * 0.317_209_886 + y * 0.577_215_664) as f32;

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        let mut filled = 0;
        // Four patches, as the four source tiles covering the block would be.
        for (column, row) in [(0, 0), (32, 0), (0, 32), (32, 32)] {
            let patch = patch_over(&grid, column, row, 32, value);
            filled += canvas.absorb(&patch, Provenance::OneMetre);
        }

        assert_eq!(filled, grid.texel_count());
        for y in 0..grid.height {
            for x in 0..grid.width {
                let (metres_x, metres_y) = grid.centre_of(x, y);
                let expected = value(metres_x, metres_y);
                let got = canvas.values()[(y as usize) * (grid.width as usize) + x as usize];
                assert_eq!(got, expected, "texel {x},{y}");
            }
        }
    }

    /// A source that is not on the output's lattice is interpolated instead,
    /// and the two routes agree where there are no holes to disagree about.
    #[test]
    fn a_half_texel_offset_has_to_be_interpolated() {
        // The window snaps its own origin to whole pixels, so the offset has to
        // be put on the canvas: a block half a metre east of the source lattice.
        let source = grid(1.0, 32, 32);
        let window = window_over(&source, 1.0, |_, _| 250.0);
        let grid = Grid {
            west: source.west + 0.5,
            ..source
        };
        assert!(
            !grid.aligns_with(window.origin_x, window.origin_y, window.metres_per_pixel),
            "should not align"
        );

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        canvas
            .fill_from(&window, None, Provenance::OneMetre)
            .expect("failed to fill");
        for &value in canvas.values() {
            assert!((value - 250.0).abs() < 1e-3, "{value}");
        }
    }

    /// Copying rather than interpolating is what keeps the texel next to a hole.
    /// Bilinear refuses any sample with a nodata neighbour, so routing an
    /// aligned source through the window would eat a one-texel border around
    /// every gap for nothing.
    #[test]
    fn absorbing_does_not_erode_the_texels_beside_a_hole() {
        let grid = grid(1.0, 32, 32);
        let hole = grid.centre_of(16, 16);
        let holed = |x: f64, y: f64| {
            if (x - hole.0).abs() < 0.5 && (y - hole.1).abs() < 0.5 {
                NODATA
            } else {
                42.0
            }
        };

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        canvas.absorb(&patch_over(&grid, 0, 0, 32, holed), Provenance::OneMetre);
        let tally = canvas.tally();
        assert_eq!(tally.missing, 1, "only the hole itself should be missing");
        assert_eq!(tally.one_metre, grid.texel_count() - 1);

        // The same data through the window loses a ring around the hole, which
        // is what the direct path is avoiding.
        let mut interpolated = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        interpolated
            .fill_from(&window_over(&grid, 1.0, holed), None, Provenance::OneMetre)
            .expect("failed to fill");
        // An aligned sample point lands on a pixel centre, so the hole is a
        // corner of the 2x2 the interpolation reads rather than its middle: the
        // texels lost are the hole and the three to its north and west.
        assert_eq!(
            interpolated.tally().missing,
            4,
            "bilinear should refuse the hole and the three texels sharing its corner"
        );
    }

    /// A later pass fills only what the first left behind, whichever route each
    /// took: the one-metre mosaic is absorbed and the two-metre one interpolated.
    #[test]
    fn an_absorbed_patch_is_not_overwritten_by_a_later_pass() {
        let grid = grid(1.0, 64, 64);
        let midpoint = grid.west + f64::from(grid.width) * 0.5;

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        let patch = patch_over(
            &grid,
            0,
            0,
            64,
            |x, _| {
                if x < midpoint { 100.0 } else { NODATA }
            },
        );
        let absorbed = canvas.absorb(&patch, Provenance::OneMetre);
        assert!(absorbed > 0 && absorbed < grid.texel_count());

        canvas
            .fill_from(
                &window_over(&grid, 2.0, |_, _| 900.0),
                None,
                Provenance::TwoMetre,
            )
            .expect("failed to fill");

        let tally = canvas.tally();
        assert_eq!(tally.one_metre, absorbed, "one metre was overwritten");
        assert_eq!(tally.missing, 0, "two metre should have covered the rest");
    }

    /// Nothing outside a patch is touched, so tiles of a block can be absorbed
    /// in any order and a patch hanging off the edge is clipped rather than
    /// wrapping.
    #[test]
    fn a_patch_only_writes_where_it_lands() {
        let grid = grid(1.0, 32, 32);
        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");

        // Placed so that only its south-east quadrant overlaps the canvas.
        let mut patch = patch_over(&grid, 0, 0, 8, |_, _| 7.0);
        patch.west -= 4.0;
        patch.north += 4.0;
        assert_eq!(canvas.absorb(&patch, Provenance::OneMetre), 16);

        for y in 0..grid.height {
            for x in 0..grid.width {
                let got = canvas.values()[(y as usize) * (grid.width as usize) + x as usize];
                let inside = x < 4 && y < 4;
                assert_eq!(got, if inside { 7.0 } else { NODATA }, "texel {x},{y}");
            }
        }
    }

    /// The heart of the fallback: two metre data only reaches texels the one
    /// metre pass left behind.
    #[test]
    fn two_metre_data_fills_only_the_holes_left_by_one_metre() {
        let grid = grid(1.0, 64, 64);
        let midpoint = grid.west + f64::from(grid.width) * 0.5;
        let fine = window_over(&grid, 1.0, |x, _| if x < midpoint { 100.0 } else { NODATA });
        let coarse = window_over(&grid, 2.0, |_, _| 900.0);

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        canvas
            .fill_from(&fine, None, Provenance::OneMetre)
            .expect("failed to fill");
        let after_fine = canvas.tally();
        assert!(after_fine.one_metre > 0, "one metre filled nothing");
        assert!(after_fine.missing > 0, "nothing left for two metre");

        canvas
            .fill_from(&coarse, None, Provenance::TwoMetre)
            .expect("failed to fill");
        let tally = canvas.tally();

        assert_eq!(
            tally.one_metre, after_fine.one_metre,
            "one metre was overwritten"
        );
        assert_eq!(tally.missing, 0, "two metre should have covered the rest");
        assert_eq!(tally.two_metre, after_fine.missing);
        assert_eq!(tally.total(), grid.texel_count());
    }

    #[test]
    fn texels_no_source_covers_stay_missing() {
        let grid = grid(1.0, 32, 32);
        let empty = window_over(&grid, 1.0, |_, _| NODATA);

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        let filled = canvas
            .fill_from(&empty, None, Provenance::OneMetre)
            .expect("failed to fill");

        assert_eq!(filled, 0);
        assert_eq!(canvas.tally().missing, grid.texel_count());
        for &value in canvas.values() {
            assert_eq!(value, NODATA);
        }
    }

    #[test]
    fn percentages_add_up() {
        let half = Tally {
            one_metre: 30,
            two_metre: 10,
            thirty_metre: 5,
            missing: 5,
        };
        let mut tally = half;
        tally.add(half);
        let shares = tally.percentages();
        assert!((shares.one_metre - 60.0).abs() < 1e-9);
        assert!((shares.two_metre - 20.0).abs() < 1e-9);
        assert!((shares.thirty_metre - 10.0).abs() < 1e-9);
        assert!((shares.missing - 10.0).abs() < 1e-9);
        let total = shares.one_metre + shares.two_metre + shares.thirty_metre + shares.missing;
        assert!((total - 100.0).abs() < 1e-9);
    }

    /// The thirty-metre tier is a fallback below the other two, not a peer: it
    /// may only fill what they left, exactly as two metres may only fill what
    /// one metre left. This is what stops MRDEM's 30 m ground from landing on
    /// top of surveyed LiDAR.
    #[test]
    fn thirty_metre_data_fills_only_the_holes_left_by_the_mosaics() {
        let grid = grid(1.0, 64, 64);
        let half = grid.west + f64::from(grid.width) * 0.5;
        let three_quarters = grid.west + f64::from(grid.width) * 0.75;

        // One metre covers the west half, two metres a quarter more, and the
        // last quarter has nothing under it until MRDEM reaches it.
        let fine = window_over(&grid, 1.0, |x, _| if x < half { 100.0 } else { NODATA });
        let coarse = window_over(
            &grid,
            2.0,
            |x, _| {
                if x < three_quarters { 200.0 } else { NODATA }
            },
        );
        let medium = window_over(&grid, 30.0, |_, _| 300.0);

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        canvas
            .fill_from(&fine, None, Provenance::OneMetre)
            .expect("failed to fill");
        canvas
            .fill_from(&coarse, None, Provenance::TwoMetre)
            .expect("failed to fill");

        let after_mosaics = canvas.tally();
        assert!(after_mosaics.one_metre > 0, "one metre filled nothing");
        assert!(after_mosaics.two_metre > 0, "two metre filled nothing");
        assert!(after_mosaics.missing > 0, "nothing left for thirty metre");

        canvas
            .fill_from(&medium, None, Provenance::ThirtyMetre)
            .expect("failed to fill");
        let tally = canvas.tally();

        assert_eq!(
            tally.one_metre, after_mosaics.one_metre,
            "one metre was overwritten"
        );
        assert_eq!(
            tally.two_metre, after_mosaics.two_metre,
            "two metre was overwritten"
        );
        assert_eq!(tally.thirty_metre, after_mosaics.missing);
        assert_eq!(
            tally.missing, 0,
            "thirty metre should have covered the rest"
        );
        assert_eq!(tally.total(), grid.texel_count());
    }

    #[test]
    fn the_hole_extent_is_none_once_everything_is_filled() {
        let grid = grid(1.0, 32, 32);
        let window = window_over(&grid, 1.0, |_, _| 500.0);

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        assert!(canvas.hole_extent(None, 2.0).expect("failed").is_some());

        canvas
            .fill_from(&window, None, Provenance::OneMetre)
            .expect("failed to fill");
        assert!(canvas.hole_extent(None, 2.0).expect("failed").is_none());
    }

    #[test]
    fn a_tile_is_cut_out_of_the_block_in_row_major_order() {
        let grid = grid(1.0, TILE_SIZE * 2, TILE_SIZE);
        let window = window_over(&grid, 1.0, |x, _| x as f32);
        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        canvas
            .fill_from(&window, None, Provenance::OneMetre)
            .expect("failed to fill");

        let mut samples = Vec::new();
        assert!(canvas.tile_samples(1, 0, &mut samples), "tile has data");
        assert_eq!(samples.len(), (TILE_SIZE as usize).pow(2));
        // The second tile across starts one tile east of the block's origin.
        let (expected_x, _) = grid.centre_of(TILE_SIZE, 0);
        assert_eq!(samples[0], expected_x as f32);
    }

    #[test]
    fn a_tile_with_nothing_in_it_reports_itself_empty() {
        let grid = grid(1.0, TILE_SIZE, TILE_SIZE);
        let canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        let mut samples = Vec::new();
        assert!(!canvas.tile_samples(0, 0, &mut samples), "should be empty");
    }
}
