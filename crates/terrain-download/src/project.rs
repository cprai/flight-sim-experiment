//! Converting the requested longitudes and latitudes into the coordinates the
//! source rasters are drawn on.
//!
//! HRDEM is published in EPSG:3979, NAD83(CSRS) / Canada Atlas Lambert -- a
//! Lambert Conformal Conic projection in metres. A longitude/latitude box does
//! not map to a rectangle there: meridians converge and parallels bow, so the
//! box's edges are curved. Everything downstream therefore works in projected
//! metres and asks this module where a given output pixel lands.
//!
//! The pairing of EPSG:4617 with EPSG:3979 is deliberate. 4617 is NAD83(CSRS)
//! expressed as longitude and latitude, so both sides of the transform sit on
//! the same GRS 80 ellipsoid with a null `towgs84`, and proj4rs's datum step
//! does nothing. Using 4326 instead would name a different datum -- WGS 84,
//! which differs from NAD83(CSRS) by a metre or two in Canada -- and invite a
//! shift that the definitions would then decline to apply anyway.

use anyhow::{Context, Result};
use proj4rs::proj::Proj;

/// EPSG code for NAD83(CSRS) as longitude and latitude in degrees.
pub const EPSG_GEOGRAPHIC: u16 = 4617;
/// EPSG code for NAD83(CSRS) / Canada Atlas Lambert, in metres.
pub const EPSG_LAMBERT: u16 = 3979;

/// Projects between the geographic coordinates the user types and the metres
/// the source rasters use.
pub struct Projector {
    geographic: Proj,
    lambert: Proj,
}

impl Projector {
    pub fn new() -> Result<Self> {
        Ok(Self {
            geographic: Proj::from_epsg_code(EPSG_GEOGRAPHIC)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("building EPSG:{EPSG_GEOGRAPHIC}"))?,
            lambert: Proj::from_epsg_code(EPSG_LAMBERT)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| format!("building EPSG:{EPSG_LAMBERT}"))?,
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
        proj4rs::transform::transform(&self.geographic, &self.lambert, points)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("projecting to EPSG:3979")
    }

    /// Converts eastings and northings in metres back to degrees, in place.
    ///
    /// The pipeline only ever goes the other way; this exists so the tests can
    /// check the forward transform against coordinates the service publishes.
    #[cfg(test)]
    pub fn to_degrees(&self, points: &mut [(f64, f64)]) -> Result<()> {
        proj4rs::transform::transform(&self.lambert, &self.geographic, points)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("projecting from EPSG:3979")?;
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

    #[test]
    fn the_false_origin_projects_to_zero() {
        let projector = Projector::new().expect("failed to build");
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
        let projector = Projector::new().expect("failed to build");
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
        let projector = Projector::new().expect("failed to build");
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
        let projector = Projector::new().expect("failed to build");
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
        let projector = Projector::new().expect("failed to build");
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
