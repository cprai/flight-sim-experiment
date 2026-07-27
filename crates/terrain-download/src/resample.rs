//! Turning projected source pixels into the longitude/latitude raster asked for.
//!
//! The source mosaics are drawn on a Lambert grid in metres; the output is a
//! grid of degrees. Those do not line up, so every output pixel is placed by
//! projecting its centre into metres and interpolating there. Doing it in that
//! direction -- pulling from the source rather than pushing to the output --
//! means every output pixel gets a value exactly once, with no gaps or
//! double-writes to reconcile afterwards.
//!
//! One metre is preferred and two metre fills what is left. The switch is
//! abrupt by design: a pixel comes from one mosaic or the other, never a blend
//! of both, so the provenance band tells the whole truth about where a value
//! came from. Where the two disagree in height -- different survey years, both
//! on CGVD2013 -- that shows up as a step at the boundary rather than being
//! smoothed into something untraceable.

use anyhow::{Context, Result};

use crate::bbox::OutputGrid;
use crate::project::Projector;
use crate::source::Window;
use crate::write::Provenance;

/// The extent of the output grid once projected into metres.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MetreExtent {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

/// How many pixels of slack to leave around the source window.
///
/// Bilinear interpolation reads the pixel on either side of the sample point,
/// so the window has to reach one pixel beyond the outermost sample; two makes
/// it robust to the rounding in between.
const BILINEAR_MARGIN: f64 = 2.0;

/// Projects the outline of the output grid to find the source extent it needs.
///
/// The boundary is walked rather than just the four corners. Most of the time
/// the corners would do: the projection sends a point to
/// `x = r sin t`, `y = r0 - r cos t`, where `r` depends only on latitude and
/// `t` only on longitude, and each of those products is monotonic in each
/// variable -- so the extremes sit at corners.
///
/// The exception is `cos t`, which peaks where `t` is zero, on the central
/// meridian at 95 degrees west. A box straddling it has a southern edge that
/// bows below both of its corners: for 100W..90W at 49N..51N the dip is
/// 14.4 km, which taking corners alone would simply fail to fetch. Walking the
/// boundary costs one projection per edge pixel and needs no special case.
pub fn projected_extent(
    grid: &OutputGrid,
    projector: &Projector,
    metres_per_pixel: f64,
) -> Result<MetreExtent> {
    let mut outline = Vec::with_capacity(2 * (grid.width as usize + grid.height as usize));
    for x in 0..grid.width {
        let longitude = grid.longitude_of(x);
        outline.push((longitude, grid.latitude_of(0)));
        outline.push((longitude, grid.latitude_of(grid.height - 1)));
    }
    for y in 0..grid.height {
        let latitude = grid.latitude_of(y);
        outline.push((grid.longitude_of(0), latitude));
        outline.push((grid.longitude_of(grid.width - 1), latitude));
    }

    projector
        .to_metres(&mut outline)
        .context("projecting the outline of the requested box")?;

    let mut extent = MetreExtent {
        min_x: f64::INFINITY,
        min_y: f64::INFINITY,
        max_x: f64::NEG_INFINITY,
        max_y: f64::NEG_INFINITY,
    };
    for (x, y) in outline {
        anyhow::ensure!(
            x.is_finite() && y.is_finite(),
            "the requested box does not project onto the Canada Atlas Lambert grid"
        );
        extent.min_x = extent.min_x.min(x);
        extent.min_y = extent.min_y.min(y);
        extent.max_x = extent.max_x.max(x);
        extent.max_y = extent.max_y.max(y);
    }

    let margin = BILINEAR_MARGIN * metres_per_pixel;
    Ok(MetreExtent {
        min_x: extent.min_x - margin,
        min_y: extent.min_y - margin,
        max_x: extent.max_x + margin,
        max_y: extent.max_y + margin,
    })
}

/// The output raster being filled in, as interleaved elevation and provenance.
pub struct Canvas {
    pub width: u32,
    pub height: u32,
    /// Elevation, provenance, elevation, provenance, ... row-major from the
    /// north-west corner.
    pub samples: Vec<f32>,
}

/// What each source contributed, as a count of output pixels.
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

impl Canvas {
    pub fn new(grid: &OutputGrid, nodata: f32) -> Result<Self> {
        let count = (grid.width as usize)
            .checked_mul(grid.height as usize)
            .and_then(|n| n.checked_mul(2))
            .context("the output raster does not fit in memory")?;

        let mut samples = Vec::new();
        samples
            .try_reserve_exact(count)
            .context("the output raster does not fit in memory")?;
        for _ in 0..count / 2 {
            samples.push(nodata);
            samples.push(Provenance::Missing.as_f32());
        }

        Ok(Self {
            width: grid.width,
            height: grid.height,
            samples,
        })
    }

    fn is_filled(&self, index: usize) -> bool {
        self.samples[index * 2 + 1] != Provenance::Missing.as_f32()
    }

    fn set(&mut self, index: usize, elevation: f32, provenance: Provenance) {
        self.samples[index * 2] = elevation;
        self.samples[index * 2 + 1] = provenance.as_f32();
    }

    /// Counts what each source ended up contributing.
    pub fn tally(&self) -> Tally {
        let mut tally = Tally::default();
        for pair in self.samples.chunks_exact(2) {
            if pair[1] == Provenance::OneMetre.as_f32() {
                tally.one_metre += 1;
            } else if pair[1] == Provenance::TwoMetre.as_f32() {
                tally.two_metre += 1;
            } else {
                tally.missing += 1;
            }
        }
        tally
    }

    /// Whether any pixel is still waiting for a value.
    pub fn has_holes(&self) -> bool {
        self.samples
            .chunks_exact(2)
            .any(|pair| pair[1] == Provenance::Missing.as_f32())
    }

    /// Fills every pixel this window can supply, leaving the rest alone.
    ///
    /// Only pixels that are still holes are considered, so calling this with
    /// the two-metre window after the one-metre window is what implements the
    /// preference between them.
    pub fn fill_from(
        &mut self,
        grid: &OutputGrid,
        projector: &Projector,
        window: &Window,
        provenance: Provenance,
    ) -> Result<u64> {
        let mut row = vec![(0.0f64, 0.0f64); self.width as usize];
        let mut filled = 0;

        for y in 0..self.height {
            let first = (y as usize) * (self.width as usize);

            // Skip a row that has nothing left to fill, which after the
            // one-metre pass is usually most of them.
            if (0..self.width as usize).all(|x| self.is_filled(first + x)) {
                continue;
            }

            let latitude = grid.latitude_of(y);
            for (x, point) in row.iter_mut().enumerate() {
                *point = (grid.longitude_of(x as u32), latitude);
            }
            projector
                .to_metres(&mut row)
                .with_context(|| format!("projecting output row {y}"))?;

            for (x, &(metres_x, metres_y)) in row.iter().enumerate() {
                let index = first + x;
                if self.is_filled(index) {
                    continue;
                }
                if let Some(elevation) = window.sample(metres_x, metres_y) {
                    self.set(index, elevation, provenance);
                    filled += 1;
                }
            }
        }

        Ok(filled)
    }

    /// The extent, in metres, of the pixels still unfilled.
    ///
    /// Used to size the two-metre window: there is no point fetching tiles for
    /// ground the one-metre mosaic already covered. Returns `None` when nothing
    /// is left to fill.
    pub fn hole_extent(
        &self,
        grid: &OutputGrid,
        projector: &Projector,
        metres_per_pixel: f64,
    ) -> Result<Option<MetreExtent>> {
        let mut row = vec![(0.0f64, 0.0f64); self.width as usize];
        let mut extent = MetreExtent {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        };
        let mut any = false;

        for y in 0..self.height {
            let first = (y as usize) * (self.width as usize);
            if (0..self.width as usize).all(|x| self.is_filled(first + x)) {
                continue;
            }

            let latitude = grid.latitude_of(y);
            for (x, point) in row.iter_mut().enumerate() {
                *point = (grid.longitude_of(x as u32), latitude);
            }
            projector
                .to_metres(&mut row)
                .with_context(|| format!("projecting output row {y}"))?;

            for (x, &(metres_x, metres_y)) in row.iter().enumerate() {
                if self.is_filled(first + x) {
                    continue;
                }
                any = true;
                extent.min_x = extent.min_x.min(metres_x);
                extent.min_y = extent.min_y.min(metres_y);
                extent.max_x = extent.max_x.max(metres_x);
                extent.max_y = extent.max_y.max(metres_y);
            }
        }

        if !any {
            return Ok(None);
        }

        let margin = BILINEAR_MARGIN * metres_per_pixel;
        Ok(Some(MetreExtent {
            min_x: extent.min_x - margin,
            min_y: extent.min_y - margin,
            max_x: extent.max_x + margin,
            max_y: extent.max_y + margin,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::{LatLon, LatLonBox};

    const NODATA: f32 = -32767.0;

    fn small_grid(metres_per_pixel: f64) -> OutputGrid {
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
        OutputGrid::cover(box_, metres_per_pixel).expect("failed to cover")
    }

    /// A window covering the whole projected extent of `grid`, filled by
    /// `value`, which is given the pixel's position in metres.
    fn window_over(
        grid: &OutputGrid,
        projector: &Projector,
        metres_per_pixel: f64,
        value: impl Fn(f64, f64) -> f32,
    ) -> Window {
        let extent =
            projected_extent(grid, projector, metres_per_pixel).expect("failed to project");
        let mut window = Window::covering(
            extent.min_x,
            extent.min_y,
            extent.max_x,
            extent.max_y,
            metres_per_pixel,
            NODATA,
        )
        .expect("failed to allocate");

        for y in 0..window.height {
            for x in 0..window.width {
                let metres_x = window.origin_x + (f64::from(x) + 0.5) * metres_per_pixel;
                let metres_y = window.origin_y - (f64::from(y) + 0.5) * metres_per_pixel;
                window.set_for_test(x, y, value(metres_x, metres_y));
            }
        }
        window
    }

    fn extent_of(box_: LatLonBox, projector: &Projector) -> (MetreExtent, MetreExtent) {
        let grid = OutputGrid::cover(box_, 200.0).expect("failed to cover");
        // No margin, so the two are directly comparable.
        let walked = projected_extent(&grid, projector, 0.0).expect("failed to project");

        let mut corners = MetreExtent {
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        };
        for (longitude, latitude) in [
            (box_.west, box_.south),
            (box_.west, box_.north),
            (box_.east, box_.south),
            (box_.east, box_.north),
        ] {
            let (x, y) = projector
                .point_to_metres(longitude, latitude)
                .expect("failed to project");
            corners.min_x = corners.min_x.min(x);
            corners.min_y = corners.min_y.min(y);
            corners.max_x = corners.max_x.max(x);
            corners.max_y = corners.max_y.max(y);
        }
        (walked, corners)
    }

    /// Away from the central meridian the projection is monotonic in both
    /// coordinates, so the corners already bound the region and walking the
    /// boundary finds nothing more. Pinned so the next test means something.
    #[test]
    fn west_of_the_central_meridian_the_corners_already_bound_the_box() {
        let projector = Projector::new().expect("failed to build");
        let box_ = LatLonBox::from_corners(
            LatLon {
                latitude: 49.0,
                longitude: -125.0,
            },
            LatLon {
                latitude: 51.0,
                longitude: -120.0,
            },
        )
        .expect("failed to build a box");
        let (walked, corners) = extent_of(box_, &projector);

        // Within a grid pixel; the walk samples pixel centres, not the edge.
        for (a, b) in [
            (walked.min_x, corners.min_x),
            (walked.min_y, corners.min_y),
            (walked.max_x, corners.max_x),
            (walked.max_y, corners.max_y),
        ] {
            assert!((a - b).abs() < 250.0, "{a} vs {b}");
        }
    }

    /// The case that makes walking the boundary necessary. `cos t` peaks on the
    /// central meridian, so a box straddling 95 degrees west has a southern
    /// edge that bows south of both its corners -- by 14.4 km for this box.
    /// Taking corners alone would leave that strip unfetched.
    #[test]
    fn a_box_straddling_the_central_meridian_bows_past_its_corners() {
        let projector = Projector::new().expect("failed to build");
        let box_ = LatLonBox::from_corners(
            LatLon {
                latitude: 49.0,
                longitude: -100.0,
            },
            LatLon {
                latitude: 51.0,
                longitude: -90.0,
            },
        )
        .expect("failed to build a box");
        let (walked, corners) = extent_of(box_, &projector);

        let dip = corners.min_y - walked.min_y;
        assert!(
            (dip - 14_372.0).abs() < 100.0,
            "expected the southern edge to dip about 14.4 km, got {dip}"
        );

        // The other three are still corner-bound, as the algebra predicts.
        for (a, b) in [
            (walked.min_x, corners.min_x),
            (walked.max_x, corners.max_x),
            (walked.max_y, corners.max_y),
        ] {
            assert!((a - b).abs() < 250.0, "{a} vs {b}");
        }
    }

    #[test]
    fn a_fresh_canvas_is_all_holes() {
        let grid = small_grid(100.0);
        let canvas = Canvas::new(&grid, NODATA).expect("failed to allocate");
        let tally = canvas.tally();
        assert_eq!(tally.one_metre, 0);
        assert_eq!(tally.two_metre, 0);
        assert_eq!(tally.missing, grid.pixel_count());
        assert!(canvas.has_holes());
    }

    #[test]
    fn a_covering_window_fills_every_pixel_once() {
        let projector = Projector::new().expect("failed to build");
        let grid = small_grid(20.0);
        let window = window_over(&grid, &projector, 20.0, |_, _| 500.0);

        let mut canvas = Canvas::new(&grid, NODATA).expect("failed to allocate");
        let filled = canvas
            .fill_from(&grid, &projector, &window, Provenance::OneMetre)
            .expect("failed to fill");

        assert_eq!(filled, grid.pixel_count());
        assert!(!canvas.has_holes());
        let tally = canvas.tally();
        assert_eq!(tally.one_metre, grid.pixel_count());
        assert_eq!(tally.missing, 0);

        for pair in canvas.samples.chunks_exact(2) {
            assert!((pair[0] - 500.0).abs() < 1e-3, "{}", pair[0]);
        }
    }

    /// The heart of the fallback: two metre data only reaches pixels the one
    /// metre pass left behind, and the provenance band says which is which.
    #[test]
    fn two_metre_data_fills_only_the_holes_left_by_one_metre() {
        let projector = Projector::new().expect("failed to build");
        let grid = small_grid(20.0);

        // One metre covers only the western half, by easting.
        let midpoint = {
            let extent = projected_extent(&grid, &projector, 1.0).expect("failed to project");
            (extent.min_x + extent.max_x) / 2.0
        };
        let fine = window_over(&grid, &projector, 20.0, |x, _| {
            if x < midpoint { 100.0 } else { NODATA }
        });
        let coarse = window_over(&grid, &projector, 40.0, |_, _| 900.0);

        let mut canvas = Canvas::new(&grid, NODATA).expect("failed to allocate");
        canvas
            .fill_from(&grid, &projector, &fine, Provenance::OneMetre)
            .expect("failed to fill");
        let tally_after_fine = canvas.tally();
        assert!(tally_after_fine.one_metre > 0, "one metre filled nothing");
        assert!(tally_after_fine.missing > 0, "nothing left for two metre");

        canvas
            .fill_from(&grid, &projector, &coarse, Provenance::TwoMetre)
            .expect("failed to fill");
        let tally = canvas.tally();

        assert_eq!(
            tally.one_metre, tally_after_fine.one_metre,
            "one metre was overwritten"
        );
        assert_eq!(tally.missing, 0, "two metre should have covered the rest");
        assert_eq!(tally.two_metre, tally_after_fine.missing);
        assert_eq!(tally.total(), grid.pixel_count());

        // Every pixel holds the value of whichever source claimed it.
        for pair in canvas.samples.chunks_exact(2) {
            if pair[1] == Provenance::OneMetre.as_f32() {
                assert!((pair[0] - 100.0).abs() < 1e-3, "{}", pair[0]);
            } else {
                assert!((pair[0] - 900.0).abs() < 1e-3, "{}", pair[0]);
            }
        }
    }

    #[test]
    fn pixels_no_source_covers_stay_missing() {
        let projector = Projector::new().expect("failed to build");
        let grid = small_grid(20.0);
        let empty = window_over(&grid, &projector, 20.0, |_, _| NODATA);

        let mut canvas = Canvas::new(&grid, NODATA).expect("failed to allocate");
        let filled = canvas
            .fill_from(&grid, &projector, &empty, Provenance::OneMetre)
            .expect("failed to fill");

        assert_eq!(filled, 0);
        let tally = canvas.tally();
        assert_eq!(tally.missing, grid.pixel_count());
        for pair in canvas.samples.chunks_exact(2) {
            assert_eq!(pair[0], NODATA);
            assert_eq!(pair[1], Provenance::Missing.as_f32());
        }
    }

    #[test]
    fn percentages_add_up() {
        let tally = Tally {
            one_metre: 70,
            two_metre: 20,
            missing: 10,
        };
        let (one, two, none) = tally.percentages();
        assert!((one - 70.0).abs() < 1e-9);
        assert!((two - 20.0).abs() < 1e-9);
        assert!((none - 10.0).abs() < 1e-9);
        assert!((one + two + none - 100.0).abs() < 1e-9);
    }

    #[test]
    fn the_hole_extent_is_none_once_everything_is_filled() {
        let projector = Projector::new().expect("failed to build");
        let grid = small_grid(20.0);
        let window = window_over(&grid, &projector, 20.0, |_, _| 500.0);

        let mut canvas = Canvas::new(&grid, NODATA).expect("failed to allocate");
        assert!(
            canvas
                .hole_extent(&grid, &projector, 2.0)
                .expect("failed")
                .is_some()
        );

        canvas
            .fill_from(&grid, &projector, &window, Provenance::OneMetre)
            .expect("failed to fill");
        assert!(
            canvas
                .hole_extent(&grid, &projector, 2.0)
                .expect("failed")
                .is_none()
        );
    }
}
