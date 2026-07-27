//! The geographic box the user asks for, and the output grid laid over it.
//!
//! Two corners come in on the command line in either diagonal order; what comes
//! out is a north-up box plus the raster that covers it exactly. "Exactly"
//! matters: the raster's outer edges land on the requested longitudes and
//! latitudes, so the file's extent is the box the user typed rather than
//! whatever fell out of rounding a pixel count.

use std::str::FromStr;

use anyhow::{Result, bail, ensure};

/// A single `lat,lon` pair as typed on the command line.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LatLon {
    pub latitude: f64,
    pub longitude: f64,
}

impl FromStr for LatLon {
    type Err = anyhow::Error;

    fn from_str(text: &str) -> Result<Self> {
        let (latitude, longitude) = text
            .split_once(',')
            .ok_or_else(|| anyhow::anyhow!("expected `lat,lon`, got `{text}`"))?;
        let parse = |part: &str, what: &str| -> Result<f64> {
            part.trim()
                .parse::<f64>()
                .map_err(|_| anyhow::anyhow!("{what} `{}` is not a number", part.trim()))
        };
        let latitude = parse(latitude, "latitude")?;
        let longitude = parse(longitude, "longitude")?;

        ensure!(
            (-90.0..=90.0).contains(&latitude),
            "latitude {latitude} is outside -90..90; the order is `lat,lon`"
        );
        ensure!(
            (-180.0..=180.0).contains(&longitude),
            "longitude {longitude} is outside -180..180"
        );

        Ok(Self {
            latitude,
            longitude,
        })
    }
}

/// A north-up geographic box in degrees.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct LatLonBox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl LatLonBox {
    /// Builds a box from two opposite corners given in either order.
    pub fn from_corners(a: LatLon, b: LatLon) -> Result<Self> {
        let box_ = Self {
            west: a.longitude.min(b.longitude),
            south: a.latitude.min(b.latitude),
            east: a.longitude.max(b.longitude),
            north: a.latitude.max(b.latitude),
        };

        if box_.west == box_.east || box_.south == box_.north {
            bail!(
                "the two corners describe a box with no area: \
                 {:.6},{:.6} to {:.6},{:.6}",
                a.latitude,
                a.longitude,
                b.latitude,
                b.longitude
            );
        }

        Ok(box_)
    }

    pub fn width_degrees(&self) -> f64 {
        self.east - self.west
    }

    pub fn height_degrees(&self) -> f64 {
        self.north - self.south
    }

    /// The latitude at which ground distances for the whole box are evaluated.
    pub fn centre_latitude(&self) -> f64 {
        (self.south + self.north) / 2.0
    }

    /// Renders as the `bbox` query parameter STAC expects: west,south,east,north.
    pub fn to_stac_bbox(self) -> String {
        format!(
            "{:.9},{:.9},{:.9},{:.9}",
            self.west, self.south, self.east, self.north
        )
    }
}

/// Metres of ground per degree of latitude and of longitude, on the WGS 84
/// ellipsoid, at one latitude.
///
/// Duplicated from `metres_per_degree` in the simulator's
/// `src/terrain/geotiff.rs`, which is the definition the renderer uses when it
/// loads the result. `flight-sim` is a binary-only package with no library
/// target, so the function cannot be imported; keeping the two in step matters
/// more than the twenty lines saved by sharing them.
pub fn metres_per_degree(latitude_degrees: f64) -> (f64, f64) {
    /// Semi-major axis of the WGS 84 ellipsoid, in metres.
    const SEMI_MAJOR_AXIS: f64 = 6_378_137.0;
    /// Its flattening, as the reciprocal the standard quotes.
    const INVERSE_FLATTENING: f64 = 298.257_223_563;

    let flattening = 1.0 / INVERSE_FLATTENING;
    let eccentricity_squared = flattening * (2.0 - flattening);

    let latitude = latitude_degrees.to_radians();
    let w = (1.0 - eccentricity_squared * latitude.sin().powi(2)).sqrt();

    let meridional = SEMI_MAJOR_AXIS * (1.0 - eccentricity_squared) / w.powi(3);
    let prime_vertical = SEMI_MAJOR_AXIS / w;

    let per_radian_to_per_degree = std::f64::consts::PI / 180.0;
    (
        per_radian_to_per_degree * meridional,
        per_radian_to_per_degree * prime_vertical * latitude.cos(),
    )
}

/// The raster laid over the requested box.
///
/// Pixels are area samples, so the value at column `i` describes the ground
/// between `west + i * degrees_per_pixel_x` and one pixel further east, and is
/// sampled at the middle of that span.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct OutputGrid {
    pub box_: LatLonBox,
    pub width: u32,
    pub height: u32,
    pub degrees_per_pixel_x: f64,
    pub degrees_per_pixel_y: f64,
}

impl OutputGrid {
    /// Lays a grid of roughly `metres_per_pixel` square texels over the box.
    ///
    /// The pixel counts are chosen first, then the pixel size is recomputed as
    /// the box divided by the count. That second step is what makes the extent
    /// exact; sizing pixels first and multiplying back would leave the eastern
    /// and southern edges short or long by a fraction of a pixel.
    pub fn cover(box_: LatLonBox, metres_per_pixel: f64) -> Result<Self> {
        let (metres_per_degree_lat, metres_per_degree_lon) =
            metres_per_degree(box_.centre_latitude());

        let width = ((box_.width_degrees() * metres_per_degree_lon) / metres_per_pixel).round();
        let height = ((box_.height_degrees() * metres_per_degree_lat) / metres_per_pixel).round();

        ensure!(
            width >= 1.0 && height >= 1.0,
            "the box is smaller than one {metres_per_pixel} m pixel"
        );
        ensure!(
            width <= f64::from(u32::MAX) && height <= f64::from(u32::MAX),
            "the box needs {width} x {height} pixels, which will not fit in a raster"
        );

        let width = width as u32;
        let height = height as u32;

        Ok(Self {
            box_,
            width,
            height,
            degrees_per_pixel_x: box_.width_degrees() / f64::from(width),
            degrees_per_pixel_y: box_.height_degrees() / f64::from(height),
        })
    }

    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Longitude of the centre of column `x`.
    pub fn longitude_of(&self, x: u32) -> f64 {
        self.box_.west + (f64::from(x) + 0.5) * self.degrees_per_pixel_x
    }

    /// Latitude of the centre of row `y`. Row 0 is the northern edge.
    pub fn latitude_of(&self, y: u32) -> f64 {
        self.box_.north - (f64::from(y) + 0.5) * self.degrees_per_pixel_y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corner(latitude: f64, longitude: f64) -> LatLon {
        LatLon {
            latitude,
            longitude,
        }
    }

    #[test]
    fn a_pair_is_parsed_as_latitude_then_longitude() {
        let parsed: LatLon = "49.633,-123.307".parse().expect("failed to parse");
        assert_eq!(parsed, corner(49.633, -123.307));
    }

    #[test]
    fn a_latitude_beyond_the_pole_is_rejected_as_a_swapped_pair() {
        let error = "-123.307,49.633".parse::<LatLon>().unwrap_err().to_string();
        assert!(error.contains("the order is `lat,lon`"), "{error}");
    }

    #[test]
    fn corners_given_in_either_diagonal_describe_the_same_box() {
        let a = corner(49.637, -123.303);
        let b = corner(49.633, -123.307);
        let one = LatLonBox::from_corners(a, b).expect("failed to build");
        let other = LatLonBox::from_corners(b, a).expect("failed to build");
        assert_eq!(one, other);
        assert_eq!(one.west, -123.307);
        assert_eq!(one.north, 49.637);
    }

    #[test]
    fn a_box_with_no_area_is_rejected() {
        let a = corner(49.633, -123.307);
        let error = LatLonBox::from_corners(a, corner(49.633, -123.0))
            .unwrap_err()
            .to_string();
        assert!(error.contains("no area"), "{error}");
    }

    #[test]
    fn the_grid_covers_the_requested_box_exactly() {
        let box_ = LatLonBox::from_corners(corner(49.633, -123.307), corner(49.637, -123.303))
            .expect("failed to build");
        let grid = OutputGrid::cover(box_, 1.0).expect("failed to cover");

        // Outer edges, which sit half a pixel beyond the outermost centres.
        let west_edge = grid.longitude_of(0) - grid.degrees_per_pixel_x / 2.0;
        let east_edge = grid.longitude_of(grid.width - 1) + grid.degrees_per_pixel_x / 2.0;
        let north_edge = grid.latitude_of(0) + grid.degrees_per_pixel_y / 2.0;
        let south_edge = grid.latitude_of(grid.height - 1) - grid.degrees_per_pixel_y / 2.0;

        assert!((west_edge - box_.west).abs() < 1e-12, "{west_edge}");
        assert!((east_edge - box_.east).abs() < 1e-12, "{east_edge}");
        assert!((north_edge - box_.north).abs() < 1e-12, "{north_edge}");
        assert!((south_edge - box_.south).abs() < 1e-12, "{south_edge}");
    }

    #[test]
    fn grid_texels_are_about_a_metre_across_in_both_directions() {
        let box_ = LatLonBox::from_corners(corner(49.63, -123.31), corner(49.65, -123.29))
            .expect("failed to build");
        let grid = OutputGrid::cover(box_, 1.0).expect("failed to cover");
        let (per_lat, per_lon) = metres_per_degree(box_.centre_latitude());

        let metres_x = grid.degrees_per_pixel_x * per_lon;
        let metres_y = grid.degrees_per_pixel_y * per_lat;
        assert!((metres_x - 1.0).abs() < 0.01, "{metres_x}");
        assert!((metres_y - 1.0).abs() < 0.01, "{metres_y}");
    }

    #[test]
    fn metres_per_degree_matches_published_values() {
        // The same reference values the simulator's copy is checked against.
        for (latitude, expect_lat, expect_lon) in [
            (0.0, 110_574.0, 111_320.0),
            (45.0, 111_132.0, 78_847.0),
            (60.0, 111_412.0, 55_800.0),
        ] {
            let (lat, lon) = metres_per_degree(latitude);
            assert!((lat - expect_lat).abs() < 2.0, "{latitude}: {lat}");
            assert!((lon - expect_lon).abs() < 2.0, "{latitude}: {lon}");
        }
    }
}
