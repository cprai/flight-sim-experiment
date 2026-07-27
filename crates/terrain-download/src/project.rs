//! Converting the requested longitudes and latitudes into the coordinates the
//! source rasters are drawn on.
//!
//! Each source has its own projection and none of them is longitude/latitude.
//! HRDEM is EPSG:3979, NAD83(CSRS) / Canada Atlas Lambert, in metres; the
//! Sentinel-2 mosaics are EPSG:3857, Web Mercator. A longitude/latitude box is
//! not a rectangle in either -- in Lambert the meridians converge and the
//! parallels bow, in Mercator the north-south scale stretches with latitude --
//! so everything downstream works in projected metres and asks this module
//! where a given output pixel lands.
//!
//! The geographic side is always EPSG:4617, NAD83(CSRS) as longitude and
//! latitude, whatever the target. For HRDEM that makes the transform a pure
//! projection: both sides sit on GRS 80 with a null `towgs84`, so proj4rs's
//! datum step does nothing. Web Mercator is nominally WGS 84 and its definition
//! carries `+nadgrids=@null`, so no shift is applied there either. Holding the
//! geographic side fixed is what makes the elevation and the imagery land on
//! the same ground: they are placed by the same interpretation of the box the
//! user typed. The price is that NAD83(CSRS) and WGS 84 differ by a metre or
//! two in Canada and nothing here models it -- below a pixel of 10 m imagery,
//! but worth knowing against a 1 m elevation raster.

use anyhow::{Context, Result};
use proj4rs::proj::Proj;

/// EPSG code for NAD83(CSRS) as longitude and latitude in degrees.
pub const EPSG_GEOGRAPHIC: u16 = 4617;
/// NAD83(CSRS) / Canada Atlas Lambert, in metres. The HRDEM mosaics.
pub const EPSG_LAMBERT: u16 = 3979;
/// WGS 84 / Pseudo-Mercator. The Sentinel-2 cloud-free mosaics.
pub const EPSG_WEB_MERCATOR: u16 = 3857;

/// Projects between the geographic coordinates the user types and the metres
/// one particular source raster is drawn on.
pub struct Projector {
    geographic: Proj,
    target: Proj,
    target_epsg: u16,
}

impl Projector {
    /// Builds a projector onto `target_epsg`.
    pub fn new(target_epsg: u16) -> Result<Self> {
        Ok(Self {
            geographic: Proj::from_epsg_code(EPSG_GEOGRAPHIC)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("building EPSG:{EPSG_GEOGRAPHIC}"))?,
            target: Proj::from_epsg_code(target_epsg)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("building EPSG:{target_epsg}"))?,
            target_epsg,
        })
    }

    /// Converts degrees of longitude and latitude to eastings and northings in
    /// metres, in place.
    ///
    /// proj4rs speaks radians, so the conversion happens here and only here --
    /// callers deal in degrees throughout, which is the only way to keep the
    /// mistake from spreading.
    pub fn to_metres(&self, points: &mut [(f64, f64)]) -> Result<()> {
        for point in points.iter_mut() {
            point.0 = point.0.to_radians();
            point.1 = point.1.to_radians();
        }
        proj4rs::transform::transform(&self.geographic, &self.target, points)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("projecting to EPSG:{}", self.target_epsg))
    }

    /// Converts eastings and northings in metres back to degrees, in place.
    ///
    /// The pipeline only ever goes the other way; this exists so the tests can
    /// check the forward transform against coordinates the service publishes.
    #[cfg(test)]
    pub fn to_degrees(&self, points: &mut [(f64, f64)]) -> Result<()> {
        proj4rs::transform::transform(&self.target, &self.geographic, points)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("projecting from EPSG:{}", self.target_epsg))?;
        for point in points.iter_mut() {
            point.0 = point.0.to_degrees();
            point.1 = point.1.to_degrees();
        }
        Ok(())
    }

    /// Projects one point. The pipeline works in rows; this is for tests.
    #[cfg(test)]
    pub fn point_to_metres(&self, longitude: f64, latitude: f64) -> Result<(f64, f64)> {
        let mut one = [(longitude, latitude)];
        self.to_metres(&mut one)?;
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
        let projector = Projector::new(EPSG_WEB_MERCATOR).expect("failed to build");
        for (longitude, latitude, expect_x, expect_y) in [
            (0.0, 0.0, 0.0, 0.0),
            (-123.305, 49.635, -13_726_249.812_3, 6_383_302.591_0),
            (-123.307, 49.633, -13_726_472.451_2, 6_382_958.836_2),
        ] {
            let (x, y) = projector
                .point_to_metres(longitude, latitude)
                .expect("failed to project");
            assert!((x - expect_x).abs() < 0.01, "easting {x} != {expect_x}");
            assert!((y - expect_y).abs() < 0.01, "northing {y} != {expect_y}");
        }
    }

    /// The two targets must place the same ground at the same longitude and
    /// latitude, or the elevation and the imagery would not overlay.
    #[test]
    fn both_targets_round_trip_to_the_same_degrees() {
        let lambert = Projector::new(EPSG_LAMBERT).expect("failed to build");
        let mercator = Projector::new(EPSG_WEB_MERCATOR).expect("failed to build");
        for point in [(-123.305, 49.635), (-114.07, 51.05)] {
            let mut a = [point];
            let mut b = [point];
            lambert.to_metres(&mut a).expect("failed");
            mercator.to_metres(&mut b).expect("failed");
            lambert.to_degrees(&mut a).expect("failed");
            mercator.to_degrees(&mut b).expect("failed");
            assert!((a[0].0 - b[0].0).abs() < 1e-9, "{a:?} vs {b:?}");
            assert!((a[0].1 - b[0].1).abs() < 1e-9, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn the_false_origin_projects_to_zero() {
        let projector = Projector::new(EPSG_LAMBERT).expect("failed to build");
        let (x, y) = projector
            .point_to_metres(-95.0, 49.0)
            .expect("failed to project");
        assert!(x.abs() < TIGHT, "easting {x}");
        assert!(y.abs() < TIGHT, "northing {y}");
    }

    /// Guards the one mistake this wrapper exists to prevent. Radians would put
    /// this point tens of millions of metres away, so the check is coarse on
    /// purpose -- it is asserting the unit, not the projection.
    #[test]
    fn coordinates_are_taken_as_degrees_not_radians() {
        let projector = Projector::new(EPSG_LAMBERT).expect("failed to build");
        let (x, y) = projector
            .point_to_metres(-123.1, 49.7)
            .expect("failed to project");
        assert!(x.abs() < 3.0e6 && y.abs() < 3.0e6, "{x} {y}");
    }

    /// Control points computed independently of proj4rs, by evaluating the
    /// Lambert Conformal Conic 2SP formulae for EPSG:3979's published
    /// parameters -- standard parallels 49N and 77N, false origin 49N 95W, on
    /// GRS 80. If the crate ever changes its maths, this notices.
    #[test]
    fn control_points_match_independently_computed_values() {
        let projector = Projector::new(EPSG_LAMBERT).expect("failed to build");
        for (longitude, latitude, expect_x, expect_y) in [
            (-123.1, 49.7, -1_956_653.44, 517_123.37),
            (-123.305_40, 49.634_73, -1_973_119.85, 516_927.57),
        ] {
            let (x, y) = projector
                .point_to_metres(longitude, latitude)
                .expect("failed to project");
            assert!((x - expect_x).abs() < 0.01, "easting {x} != {expect_x}");
            assert!((y - expect_y).abs() < 0.01, "northing {y} != {expect_y}");
        }
    }

    #[test]
    fn projecting_there_and_back_returns_the_original_degrees() {
        let projector = Projector::new(EPSG_LAMBERT).expect("failed to build");
        let original = [(-123.307, 49.633), (-95.0, 49.0), (-60.0, 76.0)];
        let mut points = original;

        projector.to_metres(&mut points).expect("failed to project");
        projector.to_degrees(&mut points).expect("failed to invert");

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
        let projector = Projector::new(EPSG_LAMBERT).expect("failed to build");
        let mut corner = [(-2_000_000.0, 1_000_000.0)];
        projector.to_degrees(&mut corner).expect("failed to invert");
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
