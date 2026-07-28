//! Terrain streamed from a pyramid of georeferenced tiles on disk.

pub mod clipmap;
pub mod geotiff;
pub mod gpu;
pub mod maxima;
pub mod mesh;
pub mod pyramid;
pub mod tiles;

/// Bare-ground elevations, in metres, and the directory they are preferred from.
pub const ELEVATION_PRODUCTS: [&str; 2] = ["dtm", "dsm"];

/// Surface colour, over the same ground as the elevation.
pub const COLOUR_PRODUCT: &str = "albedo";

/// Any elevation below this means "no measurement here".
///
/// Defined beside the filter that has to drop such texels when it builds a
/// coarse level, rather than copied here: the renderer and the reduction have to
/// agree on what a hole is or a level of the pyramid will quietly average some
/// in. The shader carries its own copy, which cannot be shared and is marked to
/// be kept in step.
pub use terrain_tiles::NODATA_BELOW;
