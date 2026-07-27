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
//! One metre is preferred and two metre fills what is left. The switch is
//! abrupt by design: a texel comes from one mosaic or the other, never a blend
//! of both. Where the two disagree in height -- different survey years, both on
//! CGVD2013 -- that shows up as a step at the boundary rather than being
//! smoothed into something untraceable.

use anyhow::{Context, Result};
use terrain_tiles::TILE_SIZE;

use crate::project::Projector;
use crate::source::Window;

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
/// tracked, because it is what implements the preference between the two
/// mosaics and what the percentages printed at the end are counted from.
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
    pub missing: u64,
}

impl Tally {
    pub fn total(&self) -> u64 {
        self.one_metre + self.two_metre + self.missing
    }

    /// Accumulates another block's counts into this one.
    pub fn add(&mut self, other: Tally) {
        self.one_metre += other.one_metre;
        self.two_metre += other.two_metre;
        self.missing += other.missing;
    }

    /// The three shares as percentages, which is what gets printed.
    pub fn percentages(&self) -> (f64, f64, f64) {
        let total = self.total();
        if total == 0 {
            return (0.0, 0.0, 0.0);
        }
        let share = |n: u64| 100.0 * n as f64 / total as f64;
        (
            share(self.one_metre),
            share(self.two_metre),
            share(self.missing),
        )
    }
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
        let texels = (grid.width as usize)
            .checked_mul(grid.height as usize)
            .context("the block does not fit in memory")?;
        let count = texels
            .checked_mul(bands)
            .context("the block does not fit in memory")?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .context("the block does not fit in memory")?;
        values.resize(count, nodata);

        let mut provenance = Vec::new();
        provenance
            .try_reserve_exact(texels)
            .context("the block does not fit in memory")?;
        provenance.resize(texels, Provenance::Missing as u8);

        Ok(Self {
            grid,
            bands,
            nodata,
            values,
            provenance,
        })
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

    /// Whether the window is drawn on exactly the same lattice as this canvas.
    ///
    /// When it is, an output texel centre falls precisely on a source pixel
    /// centre and the value can be copied. This is not only faster: bilinear
    /// interpolation refuses a sample whose four neighbours include a hole, so
    /// interpolating an aligned grid would erode a texel of real data from the
    /// edge of every gap for no reason at all.
    fn aligned_offset(&self, window: &Window) -> Option<(i64, i64)> {
        if window.metres_per_pixel != self.grid.metres_per_texel {
            return None;
        }
        let metres = self.grid.metres_per_texel;
        let columns = (self.grid.west - window.origin_x) / metres;
        let rows = (window.origin_y - self.grid.north) / metres;
        // A thousandth of a texel: the two origins are each snapped to whole
        // metres, so a real alignment is exact and anything else is far off.
        if (columns - columns.round()).abs() > 1e-3 || (rows - rows.round()).abs() > 1e-3 {
            return None;
        }
        Some((columns.round() as i64, rows.round() as i64))
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

        if projector.is_none()
            && let Some(offset) = self.aligned_offset(window)
        {
            return Ok(self.copy_from(window, offset, provenance));
        }

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

    /// Copies straight across when the two grids share a lattice.
    fn copy_from(&mut self, window: &Window, offset: (i64, i64), provenance: Provenance) -> u64 {
        let (column_offset, row_offset) = offset;
        let mut filled = 0;

        for y in 0..self.grid.height {
            let source_row = row_offset + i64::from(y);
            if source_row < 0 || source_row >= i64::from(window.height) {
                continue;
            }
            let first = (y as usize) * (self.grid.width as usize);

            for x in 0..self.grid.width {
                let index = first + x as usize;
                if self.is_filled(index) {
                    continue;
                }
                let source_column = column_offset + i64::from(x);
                if source_column < 0 || source_column >= i64::from(window.width) {
                    continue;
                }
                let Some(sample) = window.texel_at(source_column as u32, source_row as u32) else {
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

    /// The claim the whole choice of EPSG:3979 rests on: a one-metre source on
    /// the output's own lattice is copied, not interpolated, so the values come
    /// out bit for bit identical to what the mosaic holds.
    #[test]
    fn an_aligned_source_is_copied_exactly() {
        let grid = grid(1.0, 64, 64);
        // A ramp with plenty of bits set, so an interpolation that happened to
        // land near the right answer would still differ.
        let value = |x: f64, y: f64| (x * 0.317_209_886 + y * 0.577_215_664) as f32;
        let window = window_over(&grid, 1.0, value);

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        canvas
            .fill_from(&window, None, Provenance::OneMetre)
            .expect("failed to fill");

        for y in 0..grid.height {
            for x in 0..grid.width {
                let (metres_x, metres_y) = grid.centre_of(x, y);
                let expected = value(metres_x, metres_y);
                let got = canvas.values()[(y as usize) * (grid.width as usize) + x as usize];
                assert_eq!(got, expected, "texel {x},{y}");
            }
        }
    }

    /// A misaligned source falls back to interpolating, and the two paths agree
    /// where there are no holes to disagree about.
    #[test]
    fn a_half_texel_offset_falls_back_to_interpolating() {
        // The window snaps its own origin to whole pixels, so the offset has to
        // be put on the canvas: a block half a metre east of the source lattice.
        let source = grid(1.0, 32, 32);
        let window = window_over(&source, 1.0, |_, _| 250.0);
        let grid = Grid {
            west: source.west + 0.5,
            ..source
        };

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        assert!(canvas.aligned_offset(&window).is_none(), "should not align");
        canvas
            .fill_from(&window, None, Provenance::OneMetre)
            .expect("failed to fill");
        for &value in canvas.values() {
            assert!((value - 250.0).abs() < 1e-3, "{value}");
        }
    }

    /// Copying rather than interpolating is what keeps the texel next to a hole.
    /// Bilinear refuses any sample with a nodata neighbour, so the slow path
    /// would eat a one-texel border around every gap.
    #[test]
    fn copying_does_not_erode_the_texels_beside_a_hole() {
        let grid = grid(1.0, 32, 32);
        let hole = grid.centre_of(16, 16);
        let window = window_over(&grid, 1.0, |x, y| {
            if (x - hole.0).abs() < 0.5 && (y - hole.1).abs() < 0.5 {
                NODATA
            } else {
                42.0
            }
        });

        let mut canvas = Canvas::new(grid, 1, NODATA).expect("failed to allocate");
        canvas
            .fill_from(&window, None, Provenance::OneMetre)
            .expect("failed to fill");

        let tally = canvas.tally();
        assert_eq!(tally.missing, 1, "only the hole itself should be missing");
        assert_eq!(tally.one_metre, grid.texel_count() - 1);
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
        let mut tally = Tally {
            one_metre: 35,
            two_metre: 10,
            missing: 5,
        };
        tally.add(Tally {
            one_metre: 35,
            two_metre: 10,
            missing: 5,
        });
        let (one, two, none) = tally.percentages();
        assert!((one - 70.0).abs() < 1e-9);
        assert!((two - 20.0).abs() < 1e-9);
        assert!((none - 10.0).abs() < 1e-9);
        assert!((one + two + none - 100.0).abs() < 1e-9);
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
