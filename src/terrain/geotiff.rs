//! Reading a raster's placement on the ground out of its GeoTIFF tags.
//!
//! A plain TIFF says how big an image is in pixels; the GeoTIFF tags say where
//! those pixels sit on the Earth and how far apart they are. Everything in this
//! module is derived from the file at runtime, so swapping the dataset for one
//! covering somewhere else -- at a different latitude, resolution, or in a
//! different coordinate system -- needs no code change.

use std::io::{Read, Seek};

use anyhow::{Context, Result, bail};
use tiff::decoder::Decoder;
use tiff::tags::Tag;

/// GeoTIFF stores its placement in private tags that predate any registry the
/// `tiff` crate knows about, so they are read by number.
const TAG_MODEL_PIXEL_SCALE: u16 = 33550;
const TAG_MODEL_TIEPOINT: u16 = 33922;
const TAG_MODEL_TRANSFORMATION: u16 = 34264;
const TAG_GEO_KEY_DIRECTORY: u16 = 34735;

/// Keys within the GeoKeyDirectory. Only those that change how a texel's ground
/// size is computed are read; the rest describe the datum, which does not.
const KEY_MODEL_TYPE: u16 = 1024;
const KEY_RASTER_TYPE: u16 = 1025;
const KEY_ANGULAR_UNITS: u16 = 2054;
const KEY_LINEAR_UNITS: u16 = 3076;

/// Whether the raster's coordinates are already a distance or still an angle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelType {
    /// Coordinates are a linear distance in some projection's units.
    Projected,
    /// Coordinates are longitude and latitude, so they must be converted.
    Geographic,
}

/// Whether a stored value describes a texel's area or the point at its corner.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RasterType {
    /// The value covers the whole texel, so the sample sits at its centre.
    PixelIsArea,
    /// The value is sampled exactly at the texel's corner coordinate.
    PixelIsPoint,
}

/// Where a raster sits in the world and how much ground each texel covers.
///
/// The world is right-handed and Y-up with +X east and -Z north, matching the
/// camera. Raster row 0 is the northern edge, so advancing a row moves +Z.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Georeferencing {
    pub width: u32,
    pub height: u32,
    /// Metres of ground per raster column, eastward.
    pub metres_per_texel_x: f64,
    /// Metres of ground per raster row, southward.
    pub metres_per_texel_z: f64,
    /// Added to a texel index to reach the position it samples: a half texel for
    /// area pixels, nothing for point pixels.
    pub centre_offset: f64,
    /// The raster's north-west corner in the file's own coordinates.
    ///
    /// Never used for placement -- the world origin is the raster centre -- but
    /// kept so two rasters can be checked for covering the same ground.
    origin: [f64; 2],
}

impl Georeferencing {
    /// Reads the placement of the image the decoder is currently positioned on.
    pub fn read<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<Self> {
        let (width, height) = decoder.dimensions().context("reading image dimensions")?;

        let scale = decoder
            .get_tag_f64_vec(Tag::Unknown(TAG_MODEL_PIXEL_SCALE))
            .ok()
            .filter(|s| s.len() >= 2);
        let tiepoint = decoder
            .get_tag_f64_vec(Tag::Unknown(TAG_MODEL_TIEPOINT))
            .ok()
            .filter(|t| t.len() >= 6);

        let (scale, tiepoint) = match (scale, tiepoint) {
            (Some(scale), Some(tiepoint)) => (scale, tiepoint),
            _ => {
                // A ModelTransformation can express rotation and shear, which
                // would stop rows and columns running along the world axes. The
                // clipmap's regular grid depends on that, so rather than
                // silently mis-place the terrain, say what is unsupported.
                if decoder
                    .find_tag(Tag::Unknown(TAG_MODEL_TRANSFORMATION))
                    .ok()
                    .flatten()
                    .is_some()
                {
                    bail!(
                        "raster is placed with a ModelTransformation; only \
                         axis-aligned ModelPixelScale + ModelTiepoint is supported"
                    );
                }
                bail!("raster has no GeoTIFF placement tags, so its scale is unknown");
            }
        };

        let keys = GeoKeys::read(decoder)?;

        // A tiepoint ties one raster point to one model point. Anything other
        // than the origin would need the offset applied before the centring
        // below, which no dataset seen in the wild bothers with.
        if tiepoint[0] != 0.0 || tiepoint[1] != 0.0 {
            bail!("raster's tiepoint is not at its origin, which is unsupported");
        }

        let (metres_per_texel_x, metres_per_texel_z) = match keys.model_type {
            ModelType::Projected => {
                let to_metres = keys.linear_units_to_metres()?;
                (scale[0] * to_metres, scale[1] * to_metres)
            }
            ModelType::Geographic => {
                let to_degrees = keys.angular_units_to_degrees()?;
                let degrees_x = scale[0] * to_degrees;
                let degrees_z = scale[1] * to_degrees;
                // A degree of longitude shortens towards the poles, so the
                // conversion needs a latitude to be evaluated at. Using the
                // raster's centre splits the resulting error evenly between its
                // northern and southern edges instead of piling it at one end.
                let north_edge = tiepoint[4] * to_degrees;
                let centre_latitude = north_edge - degrees_z * f64::from(height) * 0.5;
                let (per_degree_z, per_degree_x) = metres_per_degree(centre_latitude);
                (degrees_x * per_degree_x, degrees_z * per_degree_z)
            }
        };

        if !(metres_per_texel_x.is_finite() && metres_per_texel_x > 0.0)
            || !(metres_per_texel_z.is_finite() && metres_per_texel_z > 0.0)
        {
            bail!(
                "raster's texel size is not a positive distance: \
                 {metres_per_texel_x} x {metres_per_texel_z} m"
            );
        }

        Ok(Self {
            width,
            height,
            metres_per_texel_x,
            metres_per_texel_z,
            centre_offset: match keys.raster_type {
                RasterType::PixelIsArea => 0.5,
                RasterType::PixelIsPoint => 0.0,
            },
            origin: [tiepoint[3], tiepoint[4]],
        })
    }

    /// The raster's north-west corner in the file's own coordinates.
    ///
    /// Only used to check a tile against the manifest that claims to describe
    /// it; placement itself goes through the world origin at the raster centre.
    pub fn origin(&self) -> [f64; 2] {
        self.origin
    }

    /// The raster's ground size in metres, east-west then north-south.
    pub fn world_extent(&self) -> (f64, f64) {
        (
            f64::from(self.width) * self.metres_per_texel_x,
            f64::from(self.height) * self.metres_per_texel_z,
        )
    }

    /// World-space X and Z of the point sampled by texel (`col`, `row`) of `level`.
    ///
    /// `level` is a mip level: each step doubles the ground a texel covers. The
    /// world origin is the raster's centre, which keeps coordinates small enough
    /// that `f32` geometry stays smooth. A raster spanning much more than a
    /// thousand kilometres would need camera-relative rendering instead.
    ///
    /// Levels share one lattice: texel `j` of level `l` sits exactly where texel
    /// `j * 2^l` of level 0 does. That is what lets a coarse grid and the finer
    /// one nested inside it meet without a seam, and it is why the half-texel
    /// centring offset is *not* scaled by the level -- doing so would place each
    /// level's samples half of its own texel further out than the level inside
    /// it, leaving a hairline gap all the way around every ring.
    ///
    /// The cost is that a box-filtered mip's value is the average of the ground
    /// starting at its sample rather than centred on it, so coarse levels sit
    /// half of their own texel off. A level is only ever drawn at a distance
    /// where its texels are around a pixel across, so that error stays under
    /// half a pixel.
    pub fn world_of_texel(&self, level: u32, col: f64, row: f64) -> (f64, f64) {
        let texels = f64::from(1u32 << level);
        let c = self.centre_offset;
        (
            (col * texels + c - f64::from(self.width) * 0.5) * self.metres_per_texel_x,
            (row * texels + c - f64::from(self.height) * 0.5) * self.metres_per_texel_z,
        )
    }

    /// A placement on a projected grid in metres, from a tile pyramid's manifest.
    ///
    /// Tiles are not read through [`Georeferencing::read`] the way a single
    /// raster is: the pyramid is many files and its placement lives in the
    /// manifest beside them, so it is built directly. `PixelIsArea` to match
    /// what the tiles themselves are written with.
    pub fn projected(width: u32, height: u32, metres_per_texel: f64, origin: [f64; 2]) -> Self {
        Self {
            width,
            height,
            metres_per_texel_x: metres_per_texel,
            metres_per_texel_z: metres_per_texel,
            centre_offset: 0.5,
            origin,
        }
    }

    /// A square, north-up placement with square texels, for tests.
    ///
    /// Lets rendering tests build terrain without a file, and without caring
    /// where on the planet it is supposed to be.
    #[cfg(test)]
    pub fn square(width: u32, height: u32, metres_per_texel: f64) -> Self {
        Self {
            width,
            height,
            metres_per_texel_x: metres_per_texel,
            metres_per_texel_z: metres_per_texel,
            centre_offset: 0.5,
            origin: [0.0, 0.0],
        }
    }

    /// World XZ of the first and last samples the raster holds.
    ///
    /// Ground outside this is not described by the data at all, so it is where
    /// the terrain has to stop.
    pub fn data_bounds(&self) -> ((f64, f64), (f64, f64)) {
        (
            self.world_of_texel(0, 0.0, 0.0),
            self.world_of_texel(0, f64::from(self.width - 1), f64::from(self.height - 1)),
        )
    }

    /// Which texel of the full-resolution raster covers a world position.
    ///
    /// The inverse of [`Georeferencing::world_of_texel`] at level 0, and
    /// fractional: the clipmap needs to know where the camera falls between
    /// texels, not just which one it is over.
    pub fn texel_of_world(&self, world_x: f64, world_z: f64) -> glam::DVec2 {
        glam::DVec2::new(
            world_x / self.metres_per_texel_x + f64::from(self.width) * 0.5 - self.centre_offset,
            world_z / self.metres_per_texel_z + f64::from(self.height) * 0.5 - self.centre_offset,
        )
    }
}

/// Metres per degree of latitude and of longitude at a latitude, on the WGS 84
/// ellipsoid.
///
/// Both shrink towards the poles, longitude far faster than latitude.
pub fn metres_per_degree(latitude_degrees: f64) -> (f64, f64) {
    /// Semi-major axis of the WGS 84 ellipsoid, in metres.
    const SEMI_MAJOR_AXIS: f64 = 6_378_137.0;
    /// Its flattening, as the reciprocal the standard quotes.
    const INVERSE_FLATTENING: f64 = 298.257_223_563;

    let flattening = 1.0 / INVERSE_FLATTENING;
    let eccentricity_squared = flattening * (2.0 - flattening);

    let latitude = latitude_degrees.to_radians();
    let w = (1.0 - eccentricity_squared * latitude.sin().powi(2)).sqrt();

    // Radius of curvature along the meridian, and perpendicular to it.
    let meridional = SEMI_MAJOR_AXIS * (1.0 - eccentricity_squared) / w.powi(3);
    let prime_vertical = SEMI_MAJOR_AXIS / w;

    let per_radian_to_per_degree = std::f64::consts::PI / 180.0;
    (
        per_radian_to_per_degree * meridional,
        per_radian_to_per_degree * prime_vertical * latitude.cos(),
    )
}

/// The handful of GeoKeyDirectory entries that affect a texel's ground size.
struct GeoKeys {
    model_type: ModelType,
    raster_type: RasterType,
    angular_units: u16,
    linear_units: u16,
}

impl GeoKeys {
    /// Unit codes, from the EPSG register the GeoTIFF spec defers to.
    const UNIT_RADIAN: u16 = 9101;
    const UNIT_DEGREE: u16 = 9102;
    const UNIT_METRE: u16 = 9001;
    const UNIT_FOOT: u16 = 9002;
    const UNIT_US_SURVEY_FOOT: u16 = 9003;

    fn read<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<Self> {
        let directory = decoder
            .get_tag_u16_vec(Tag::Unknown(TAG_GEO_KEY_DIRECTORY))
            .context("raster has no GeoKeyDirectory, so its coordinate system is unknown")?;

        // The directory opens with a four-value header whose last entry counts
        // the keys that follow, each itself four values wide.
        let Some(&key_count) = directory.get(3) else {
            bail!("raster's GeoKeyDirectory is truncated before its header ends");
        };
        let entries = directory
            .get(4..)
            .unwrap_or_default()
            .chunks_exact(4)
            .take(usize::from(key_count));

        let mut keys = Self {
            // Defaults per the GeoTIFF spec, used when a key is simply absent.
            model_type: ModelType::Geographic,
            raster_type: RasterType::PixelIsArea,
            angular_units: Self::UNIT_DEGREE,
            linear_units: Self::UNIT_METRE,
        };

        for entry in entries {
            let (id, location, value) = (entry[0], entry[1], entry[3]);
            // A non-zero location means the value lives in another tag. None of
            // the keys read here are ever stored that way.
            if location != 0 {
                continue;
            }
            match id {
                KEY_MODEL_TYPE => {
                    keys.model_type = match value {
                        1 => ModelType::Projected,
                        2 => ModelType::Geographic,
                        // 3 is a geocentric cartesian frame, whose axes do not
                        // line up with a north-up raster at all.
                        other => bail!("raster uses unsupported GTModelType {other}"),
                    }
                }
                KEY_RASTER_TYPE => {
                    keys.raster_type = match value {
                        1 => RasterType::PixelIsArea,
                        2 => RasterType::PixelIsPoint,
                        other => bail!("raster uses unsupported GTRasterType {other}"),
                    }
                }
                KEY_ANGULAR_UNITS => keys.angular_units = value,
                KEY_LINEAR_UNITS => keys.linear_units = value,
                _ => {}
            }
        }

        Ok(keys)
    }

    fn angular_units_to_degrees(&self) -> Result<f64> {
        match self.angular_units {
            Self::UNIT_DEGREE => Ok(1.0),
            Self::UNIT_RADIAN => Ok(180.0 / std::f64::consts::PI),
            other => bail!("raster uses unsupported angular unit {other}"),
        }
    }

    fn linear_units_to_metres(&self) -> Result<f64> {
        match self.linear_units {
            Self::UNIT_METRE => Ok(1.0),
            Self::UNIT_FOOT => Ok(0.3048),
            // Defined as exactly 1200/3937 metres, a hair longer than a foot.
            Self::UNIT_US_SURVEY_FOOT => Ok(1200.0 / 3937.0),
            other => bail!("raster uses unsupported linear unit {other}"),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::io::Cursor;

    use tiff::encoder::{TiffEncoder, colortype};

    use super::*;

    /// A raster small enough to build quickly but tall enough to span several
    /// strips, so the last one is a partial strip like a real file's.
    const WIDTH: u32 = 20;
    const HEIGHT: u32 = 30;

    /// Packs GeoKeys into the flat directory layout the tag stores.
    ///
    /// Every key here is a plain value held inline, which is how the four keys
    /// that matter are always written.
    pub(crate) fn geo_key_directory(keys: &[(u16, u16)]) -> Vec<u16> {
        let mut directory = vec![1, 1, 0, keys.len() as u16];
        for &(id, value) in keys {
            directory.extend_from_slice(&[id, 0, 1, value]);
        }
        directory
    }

    /// Keys describing a longitude/latitude raster measured in degrees.
    fn geographic_keys() -> Vec<u16> {
        geo_key_directory(&[
            (KEY_MODEL_TYPE, 2),
            (KEY_RASTER_TYPE, 1),
            (KEY_ANGULAR_UNITS, 9102),
        ])
    }

    /// Builds a single-band GeoTIFF in memory carrying the given placement.
    ///
    /// Synthesizing the file keeps these tests independent of any particular
    /// dataset, and lets them cover placements no single real file has.
    pub(crate) fn synthetic_geotiff(
        width: u32,
        height: u32,
        pixel_scale: &[f64],
        tiepoint: &[f64],
        geo_keys: &[u16],
    ) -> Vec<u8> {
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buffer).expect("failed to start encoding");
            let mut image = encoder
                .new_image::<colortype::Gray32Float>(width, height)
                .expect("failed to start image");
            image.rows_per_strip(8).expect("failed to set strip height");
            {
                let directory = image.encoder();
                if !pixel_scale.is_empty() {
                    directory
                        .write_tag(Tag::Unknown(TAG_MODEL_PIXEL_SCALE), pixel_scale)
                        .expect("failed to write pixel scale");
                }
                if !tiepoint.is_empty() {
                    directory
                        .write_tag(Tag::Unknown(TAG_MODEL_TIEPOINT), tiepoint)
                        .expect("failed to write tiepoint");
                }
                if !geo_keys.is_empty() {
                    directory
                        .write_tag(Tag::Unknown(TAG_GEO_KEY_DIRECTORY), geo_keys)
                        .expect("failed to write geo keys");
                }
            }
            let pixels = vec![0.0f32; (width * height) as usize];
            image.write_data(&pixels).expect("failed to write pixels");
        }
        buffer.into_inner()
    }

    fn read_placement(bytes: &[u8]) -> Result<Georeferencing> {
        let mut decoder = Decoder::new(Cursor::new(bytes))?;
        Georeferencing::read(&mut decoder)
    }

    /// A north-up geographic raster placed at a mid-northern latitude.
    fn geographic_raster() -> Georeferencing {
        let bytes = synthetic_geotiff(
            WIDTH,
            HEIGHT,
            &[0.001, 0.002, 0.0],
            &[0.0, 0.0, 0.0, 10.0, 45.0, 0.0],
            &geographic_keys(),
        );
        read_placement(&bytes).expect("failed to read placement")
    }

    #[test]
    fn metres_per_degree_matches_published_values() {
        // Reference values for the WGS 84 ellipsoid, to the nearest metre.
        for (latitude, expect_lat, expect_lon) in [
            (0.0, 110_574.0, 111_320.0),
            (45.0, 111_132.0, 78_847.0),
            (60.0, 111_412.0, 55_800.0),
        ] {
            let (lat, lon) = metres_per_degree(latitude);
            assert!(
                (lat - expect_lat).abs() < 2.0,
                "latitude degree at {latitude}: got {lat}, expected {expect_lat}"
            );
            assert!(
                (lon - expect_lon).abs() < 2.0,
                "longitude degree at {latitude}: got {lon}, expected {expect_lon}"
            );
        }
    }

    #[test]
    fn a_degree_of_longitude_shrinks_towards_the_poles() {
        let (_, equator) = metres_per_degree(0.0);
        let (_, mid) = metres_per_degree(45.0);
        let (_, high) = metres_per_degree(70.0);
        assert!(equator > mid && mid > high, "{equator} {mid} {high}");
    }

    #[test]
    fn a_geographic_raster_converts_its_degrees_to_metres() {
        let placement = geographic_raster();

        // The reference latitude is the raster's centre, not its northern edge.
        let centre_latitude = 45.0 - 0.002 * f64::from(HEIGHT) * 0.5;
        let (per_degree_z, per_degree_x) = metres_per_degree(centre_latitude);

        assert!((placement.metres_per_texel_x - 0.001 * per_degree_x).abs() < 1e-6);
        assert!((placement.metres_per_texel_z - 0.002 * per_degree_z).abs() < 1e-6);
    }

    #[test]
    fn a_projected_raster_passes_its_pixel_scale_straight_through() {
        let bytes = synthetic_geotiff(
            WIDTH,
            HEIGHT,
            &[30.0, 25.0, 0.0],
            &[0.0, 0.0, 0.0, 500_000.0, 4_000_000.0, 0.0],
            &geo_key_directory(&[
                (KEY_MODEL_TYPE, 1),
                (KEY_RASTER_TYPE, 1),
                (KEY_LINEAR_UNITS, 9001),
            ]),
        );
        let placement = read_placement(&bytes).expect("failed to read placement");

        // Already metres, so no latitude is involved at all.
        assert_eq!(placement.metres_per_texel_x, 30.0);
        assert_eq!(placement.metres_per_texel_z, 25.0);
    }

    #[test]
    fn a_projected_raster_in_feet_is_converted_to_metres() {
        let bytes = synthetic_geotiff(
            WIDTH,
            HEIGHT,
            &[100.0, 100.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &geo_key_directory(&[
                (KEY_MODEL_TYPE, 1),
                (KEY_RASTER_TYPE, 1),
                (KEY_LINEAR_UNITS, 9002),
            ]),
        );
        let placement = read_placement(&bytes).expect("failed to read placement");

        assert!((placement.metres_per_texel_x - 30.48).abs() < 1e-9);
    }

    #[test]
    fn area_pixels_sample_half_a_texel_in_from_point_pixels() {
        let area = geographic_raster();
        let bytes = synthetic_geotiff(
            WIDTH,
            HEIGHT,
            &[0.001, 0.002, 0.0],
            &[0.0, 0.0, 0.0, 10.0, 45.0, 0.0],
            &geo_key_directory(&[
                (KEY_MODEL_TYPE, 2),
                (KEY_RASTER_TYPE, 2),
                (KEY_ANGULAR_UNITS, 9102),
            ]),
        );
        let point = read_placement(&bytes).expect("failed to read placement");

        assert_eq!(area.centre_offset, 0.5);
        assert_eq!(point.centre_offset, 0.0);

        let (area_x, _) = area.world_of_texel(0, 0.0, 0.0);
        let (point_x, _) = point.world_of_texel(0, 0.0, 0.0);
        assert!((area_x - point_x - 0.5 * area.metres_per_texel_x).abs() < 1e-9);
    }

    #[test]
    fn the_first_raster_row_lies_north_of_the_last() {
        let placement = geographic_raster();
        let (_, first) = placement.world_of_texel(0, 0.0, 0.0);
        let (_, last) = placement.world_of_texel(0, 0.0, f64::from(HEIGHT - 1));

        // North is -Z, so row 0 must be the more negative of the two.
        assert!(first < last, "row 0 at {first} should be north of {last}");
        assert!(
            first < 0.0 && last > 0.0,
            "the raster should straddle the origin"
        );
    }

    #[test]
    fn texels_are_centred_on_the_world_origin_and_span_the_extent() {
        let placement = geographic_raster();
        let (extent_x, extent_z) = placement.world_extent();

        let (west, north) = placement.world_of_texel(0, 0.0, 0.0);
        let (east, south) =
            placement.world_of_texel(0, f64::from(WIDTH - 1), f64::from(HEIGHT - 1));

        // Sample points sit half a texel inside each edge, so the span they
        // cover is one texel short of the full extent.
        assert!((east - west - (extent_x - placement.metres_per_texel_x)).abs() < 1e-6);
        assert!((south - north - (extent_z - placement.metres_per_texel_z)).abs() < 1e-6);
        assert!((west + east).abs() < 1e-6, "should be centred east-west");
        assert!(
            (north + south).abs() < 1e-6,
            "should be centred north-south"
        );
    }

    #[test]
    fn world_positions_map_back_to_the_texels_they_came_from() {
        let placement = geographic_raster();
        for (col, row) in [(0.0, 0.0), (7.0, 11.0), (19.0, 29.0)] {
            let (x, z) = placement.world_of_texel(0, col, row);
            let texel = placement.texel_of_world(x, z);
            assert!(
                (texel.x - col).abs() < 1e-6,
                "{texel} should be ({col}, {row})"
            );
            assert!(
                (texel.y - row).abs() < 1e-6,
                "{texel} should be ({col}, {row})"
            );
        }
    }

    #[test]
    fn every_level_samples_the_same_lattice() {
        let placement = geographic_raster();

        // Texel j of level l has to land exactly on texel j * 2^l of level 0,
        // or a coarse grid and the finer one nested inside it would not meet.
        for level in 0..4 {
            for texel in 0..4 {
                let (coarse, _) = placement.world_of_texel(level, f64::from(texel), 0.0);
                let fine_texel = f64::from(texel * (1 << level));
                let (fine, _) = placement.world_of_texel(0, fine_texel, 0.0);
                assert!(
                    (fine - coarse).abs() < 1e-9,
                    "level {level} texel {texel} sits at {coarse}, level 0 puts it at {fine}"
                );
            }
        }
    }

    #[test]
    fn a_coarser_mip_texel_covers_proportionally_more_ground() {
        let placement = geographic_raster();

        let (near, _) = placement.world_of_texel(2, 0.0, 0.0);
        let (far, _) = placement.world_of_texel(2, 1.0, 0.0);
        assert!(
            (far - near - 4.0 * placement.metres_per_texel_x).abs() < 1e-9,
            "a level 2 texel should span four of level 0's"
        );
    }

    #[test]
    fn a_raster_without_placement_tags_is_rejected() {
        let bytes = synthetic_geotiff(WIDTH, HEIGHT, &[], &[], &geographic_keys());
        let error = read_placement(&bytes).expect_err("should refuse to guess a scale");
        assert!(
            error.to_string().contains("no GeoTIFF placement tags"),
            "unhelpful error: {error}"
        );
    }

    #[test]
    fn a_raster_placed_by_transformation_matrix_is_rejected() {
        // A ModelTransformation can rotate the raster off the world axes, which
        // the regular terrain grid cannot represent.
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut encoder = TiffEncoder::new(&mut buffer).unwrap();
            let mut image = encoder
                .new_image::<colortype::Gray32Float>(WIDTH, HEIGHT)
                .unwrap();
            image.rows_per_strip(8).unwrap();
            {
                let directory = image.encoder();
                let transformation = [0.0f64; 16];
                directory
                    .write_tag(Tag::Unknown(TAG_MODEL_TRANSFORMATION), &transformation[..])
                    .unwrap();
                directory
                    .write_tag(Tag::Unknown(TAG_GEO_KEY_DIRECTORY), &geographic_keys()[..])
                    .unwrap();
            }
            image
                .write_data(&vec![0.0f32; (WIDTH * HEIGHT) as usize])
                .unwrap();
        }
        let error = read_placement(&buffer.into_inner()).expect_err("should refuse to rotate");
        assert!(
            error.to_string().contains("ModelTransformation"),
            "unhelpful error: {error}"
        );
    }

    #[test]
    fn an_unsupported_unit_is_rejected_rather_than_assumed() {
        let bytes = synthetic_geotiff(
            WIDTH,
            HEIGHT,
            &[0.001, 0.002, 0.0],
            &[0.0, 0.0, 0.0, 10.0, 45.0, 0.0],
            // 9105 is grads, which nothing here knows how to convert.
            &geo_key_directory(&[(KEY_MODEL_TYPE, 2), (KEY_ANGULAR_UNITS, 9105)]),
        );
        assert!(
            read_placement(&bytes).is_err(),
            "should refuse unknown units"
        );
    }
}
