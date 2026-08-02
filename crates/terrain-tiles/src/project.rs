//! Converting between the grids the output and the sources are drawn on.
//!
//! The output tile grid is EPSG:3979, NAD83(CSRS) / Canada Atlas Lambert, in
//! metres -- the grid HRDEM is already published on. Two conversions are needed
//! around it.
//!
//! The first is from the longitude and latitude the user types, EPSG:4617, to
//! decide which ground the requested box covers. Both sides sit on GRS 80 with a
//! null `towgs84`, so proj4rs's datum step does nothing and the transform is a
//! pure projection.
//!
//! The second is from the output grid to a source on a different one. Elevation
//! needs none of this -- it is already on EPSG:3979, which is the whole reason
//! that CRS was chosen -- but the Sentinel-2 mosaics are EPSG:3857, Web
//! Mercator, whose definition carries `+nadgrids=@null` so no shift is applied
//! there either. A rectangle on one of these grids is not a rectangle on the
//! other: meridian convergence rotates them relative to each other by about 25
//! degrees at 123W, which is why callers walk a boundary rather than projecting
//! four corners.
//!
//! The price of holding the geographic side at EPSG:4617 is that NAD83(CSRS)
//! and WGS 84 differ by a metre or two in Canada and nothing here models it --
//! below a pixel of 16 m imagery, but worth knowing against a 1 m elevation
//! raster.

use anyhow::{Context, Result};
use proj4rs::proj::Proj;

/// EPSG code for NAD83(CSRS) as longitude and latitude in degrees.
pub const EPSG_GEOGRAPHIC: u16 = 4617;
/// WGS 84 / Pseudo-Mercator. The Sentinel-2 cloud-free mosaics.
pub const EPSG_WEB_MERCATOR: u16 = 3857;

// The output grid's own code lives beside the writer that stamps it into every
// tile; re-exported here so projection callers need only this module.
pub use crate::write::EPSG_LAMBERT;

/// Transforms points from one CRS to another.
pub struct Projector {
    from: Proj,
    to: Proj,
    from_epsg: u16,
    to_epsg: u16,
    /// Whether the input side is longitude and latitude rather than metres.
    ///
    /// proj4rs speaks radians, so an angular side needs converting on the way
    /// in or out. Tracked from the constructor rather than asked of `Proj`, so
    /// that the one mistake this module exists to prevent is decided in exactly
    /// one place.
    from_is_angular: bool,
}

fn build(epsg: u16) -> Result<Proj> {
    Proj::from_epsg_code(epsg)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("building EPSG:{epsg}"))
}

impl Projector {
    /// Builds a projector from degrees of longitude and latitude onto a
    /// projected CRS in metres.
    pub fn from_geographic(target_epsg: u16) -> Result<Self> {
        Ok(Self {
            from: build(EPSG_GEOGRAPHIC)?,
            to: build(target_epsg)?,
            from_epsg: EPSG_GEOGRAPHIC,
            to_epsg: target_epsg,
            from_is_angular: true,
        })
    }

    /// Builds a projector between two projected CRSs, both in metres.
    pub fn between(from_epsg: u16, to_epsg: u16) -> Result<Self> {
        Ok(Self {
            from: build(from_epsg)?,
            to: build(to_epsg)?,
            from_epsg,
            to_epsg,
            from_is_angular: false,
        })
    }

    /// Transforms points onto the target CRS, in place.
    ///
    /// Named for its use: callers hold output-grid coordinates and want the
    /// source coordinates that correspond.
    pub fn to_source(&self, points: &mut [(f64, f64)]) -> Result<()> {
        if self.from_is_angular {
            for point in points.iter_mut() {
                point.0 = point.0.to_radians();
                point.1 = point.1.to_radians();
            }
        }
        proj4rs::transform::transform(&self.from, &self.to, points)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| {
                format!(
                    "projecting EPSG:{} to EPSG:{}",
                    self.from_epsg, self.to_epsg
                )
            })
    }

    /// Transforms points back, in place.
    ///
    /// Used to turn the snapped output extent back into the longitude/latitude
    /// box the catalogues are searched by, and by the tests to check the
    /// forward transform against coordinates the services publish.
    pub fn to_output(&self, points: &mut [(f64, f64)]) -> Result<()> {
        proj4rs::transform::transform(&self.to, &self.from, points)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("projecting from EPSG:{}", self.to_epsg))?;
        if self.from_is_angular {
            for point in points.iter_mut() {
                point.0 = point.0.to_degrees();
                point.1 = point.1.to_degrees();
            }
        }
        Ok(())
    }

    /// Projects one point. The pipelines work in batches; this is for tests
    /// and one-off lookups.
    pub fn point_to_source(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        let mut one = [(x, y)];
        self.to_source(&mut one)?;
        Ok(one[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for a coordinate that should be exact, in metres.
    const TIGHT: f64 = 1e-3;

    /// Web Mercator control points, computed independently from the closed
    /// form -- `x = R * lon`, `y = R * ln(tan(pi/4 + lat/2))` on the sphere of
    /// radius 6378137 that EPSG:3857 defines. Pins the second projection the
    /// same way the Lambert one is pinned.
    #[test]
    fn web_mercator_matches_independently_computed_values() {
        let projector = Projector::from_geographic(EPSG_WEB_MERCATOR).expect("failed to build");
        for (longitude, latitude, expect_x, expect_y) in [
            (0.0, 0.0, 0.0, 0.0),
            (-123.305, 49.635, -13_726_249.812_3, 6_383_302.591_0),
            (-123.307, 49.633, -13_726_472.451_2, 6_382_958.836_2),
        ] {
            let (x, y) = projector
                .point_to_source(longitude, latitude)
                .expect("failed to project");
            assert!((x - expect_x).abs() < 0.01, "easting {x} != {expect_x}");
            assert!((y - expect_y).abs() < 0.01, "northing {y} != {expect_y}");
        }
    }

    /// The two targets must place the same ground at the same longitude and
    /// latitude, or the elevation and the imagery would not overlay.
    #[test]
    fn both_targets_round_trip_to_the_same_degrees() {
        let lambert = Projector::from_geographic(EPSG_LAMBERT).expect("failed to build");
        let mercator = Projector::from_geographic(EPSG_WEB_MERCATOR).expect("failed to build");
        for point in [(-123.305, 49.635), (-114.07, 51.05)] {
            let mut a = [point];
            let mut b = [point];
            lambert.to_source(&mut a).expect("failed");
            mercator.to_source(&mut b).expect("failed");
            lambert.to_output(&mut a).expect("failed");
            mercator.to_output(&mut b).expect("failed");
            assert!((a[0].0 - b[0].0).abs() < 1e-9, "{a:?} vs {b:?}");
            assert!((a[0].1 - b[0].1).abs() < 1e-9, "{a:?} vs {b:?}");
        }
    }

    /// Colour is fetched by projecting the output grid straight onto the mosaic
    /// grid, so that composite transform has to land where going through
    /// longitude and latitude would.
    #[test]
    fn lambert_to_mercator_agrees_with_going_via_degrees() {
        let direct = Projector::between(EPSG_LAMBERT, EPSG_WEB_MERCATOR).expect("failed to build");
        let to_lambert = Projector::from_geographic(EPSG_LAMBERT).expect("failed to build");
        let to_mercator = Projector::from_geographic(EPSG_WEB_MERCATOR).expect("failed to build");

        for (longitude, latitude) in [(-123.305, 49.635), (-114.07, 51.05), (-95.0, 49.0)] {
            let lambert = to_lambert
                .point_to_source(longitude, latitude)
                .expect("failed");
            let expected = to_mercator
                .point_to_source(longitude, latitude)
                .expect("failed");
            let got = direct
                .point_to_source(lambert.0, lambert.1)
                .expect("failed");
            assert!(
                (got.0 - expected.0).abs() < TIGHT,
                "{got:?} vs {expected:?}"
            );
            assert!(
                (got.1 - expected.1).abs() < TIGHT,
                "{got:?} vs {expected:?}"
            );
        }
    }

    /// A projected input must not be treated as degrees. Radians would be
    /// nonsense at these magnitudes, so this pins that the angular conversion
    /// is skipped for a metre-to-metre projector.
    #[test]
    fn a_projected_input_is_not_converted_from_degrees() {
        let direct = Projector::between(EPSG_LAMBERT, EPSG_LAMBERT).expect("failed to build");
        let (x, y) = direct
            .point_to_source(-1_956_653.44, 517_123.37)
            .expect("failed");
        assert!((x + 1_956_653.44).abs() < TIGHT, "easting {x}");
        assert!((y - 517_123.37).abs() < TIGHT, "northing {y}");
    }

    #[test]
    fn the_false_origin_projects_to_zero() {
        let projector = Projector::from_geographic(EPSG_LAMBERT).expect("failed to build");
        let (x, y) = projector
            .point_to_source(-95.0, 49.0)
            .expect("failed to project");
        assert!(x.abs() < TIGHT, "easting {x}");
        assert!(y.abs() < TIGHT, "northing {y}");
    }

    /// Guards the one mistake this wrapper exists to prevent. Radians would put
    /// this point tens of millions of metres away, so the check is coarse on
    /// purpose -- it is asserting the unit, not the projection.
    #[test]
    fn coordinates_are_taken_as_degrees_not_radians() {
        let projector = Projector::from_geographic(EPSG_LAMBERT).expect("failed to build");
        let (x, y) = projector
            .point_to_source(-123.1, 49.7)
            .expect("failed to project");
        assert!(x.abs() < 3.0e6 && y.abs() < 3.0e6, "{x} {y}");
    }

    /// Control points computed independently of proj4rs, by evaluating the
    /// Lambert Conformal Conic 2SP formulae for EPSG:3979's published
    /// parameters -- standard parallels 49N and 77N, false origin 49N 95W, on
    /// GRS 80. If the crate ever changes its maths, this notices.
    #[test]
    fn control_points_match_independently_computed_values() {
        let projector = Projector::from_geographic(EPSG_LAMBERT).expect("failed to build");
        for (longitude, latitude, expect_x, expect_y) in [
            (-123.1, 49.7, -1_956_653.44, 517_123.37),
            (-123.305_40, 49.634_73, -1_973_119.85, 516_927.57),
        ] {
            let (x, y) = projector
                .point_to_source(longitude, latitude)
                .expect("failed to project");
            assert!((x - expect_x).abs() < 0.01, "easting {x} != {expect_x}");
            assert!((y - expect_y).abs() < 0.01, "northing {y} != {expect_y}");
        }
    }

    #[test]
    fn projecting_there_and_back_returns_the_original_degrees() {
        let projector = Projector::from_geographic(EPSG_LAMBERT).expect("failed to build");
        let original = [(-123.307, 49.633), (-95.0, 49.0), (-60.0, 76.0)];
        let mut points = original;

        projector.to_source(&mut points).expect("failed to project");
        projector.to_output(&mut points).expect("failed to invert");

        for (got, want) in points.iter().zip(original.iter()) {
            // A millionth of a degree is about a tenth of a millimetre.
            assert!((got.0 - want.0).abs() < 1e-9, "{got:?} != {want:?}");
            assert!((got.1 - want.1).abs() < 1e-9, "{got:?} != {want:?}");
        }
    }

    /// The north-west corner of mosaic block `2_4`, which the STAC item
    /// declares as covering longitudes -126.85..-117.04 and latitudes
    /// 49.39..55.40. Ties the projection to what the service actually says.
    #[test]
    fn a_block_corner_falls_inside_the_declared_geographic_extent() {
        let projector = Projector::from_geographic(EPSG_LAMBERT).expect("failed to build");
        let mut corner = [(-2_000_000.0, 1_000_000.0)];
        projector.to_output(&mut corner).expect("failed to invert");
        let (longitude, latitude) = corner[0];
        assert!(
            (-126.852_867..=-117.040_404).contains(&longitude),
            "longitude {longitude}"
        );
        assert!(
            (49.392_567..=55.396_741).contains(&latitude),
            "latitude {latitude}"
        );
    }
}
