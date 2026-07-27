//! The geographic box the user asks for.
//!
//! Two corners come in on the command line in either diagonal order; what comes
//! out is a north-up box in degrees. Turning that into ground is `extent`'s
//! job, because the output is drawn on a projected grid and the box is only the
//! request, not the result -- the tiles written cover a little more than was
//! asked for, snapped out to whole tile boundaries.

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

    /// Renders as the `bbox` query parameter STAC expects: west,south,east,north.
    pub fn to_stac_bbox(self) -> String {
        format!(
            "{:.9},{:.9},{:.9},{:.9}",
            self.west, self.south, self.east, self.north
        )
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
}
