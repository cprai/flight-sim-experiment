//! Terrain built from a georeferenced height raster and a matching colour raster.

pub mod clipmap;
pub mod geotiff;
pub mod gpu;
pub mod mesh;
pub mod pyramid;

/// Elevations, in metres, on whatever vertical datum the file was authored with.
pub const HEIGHT_RASTER_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/dem.tiff");

/// Surface colour, on the same grid as the height raster.
///
/// Resolved against the crate root rather than the working directory so the
/// binary finds its data wherever it is launched from.
pub const COLOUR_RASTER_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/albedo.tiff");
