//! Terrain streamed from a pyramid of georeferenced tiles on disk.

pub mod clipmap;
pub mod geotiff;
pub mod gpu;
pub mod mesh;
pub mod pyramid;
pub mod tiles;

/// Bare-ground elevations, in metres, and the directory they are preferred from.
pub const ELEVATION_PRODUCTS: [&str; 2] = ["dtm", "dsm"];

/// Surface colour, over the same ground as the elevation.
pub const COLOUR_PRODUCT: &str = "albedo";

/// Any elevation below this means "no measurement here".
///
/// HRDEM writes -32767 and the tiles carry it through, but the exact value is
/// not worth threading from the manifest into the shader: the deepest ground on
/// Earth is a fraction of this, so anything below it is a hole however the
/// producer chose to spell it.
pub const NODATA_BELOW: f32 = -30_000.0;
