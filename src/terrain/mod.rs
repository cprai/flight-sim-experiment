//! Terrain streamed from a pyramid of georeferenced tiles on disk.

pub mod geotiff;
pub mod gpu;
pub mod maxima;
pub mod pyramid;
pub mod residency;
pub mod tiles;

/// Bare-ground elevations, in metres, and the directory they are read from.
///
/// One product rather than a list to fall back through. A surface model was the
/// other candidate and it is not one the renderer can use interchangeably: it
/// stands the sensor's own reading on top of the ground, so trees and roofs
/// become terrain a ray collides with, and the shading a hillside gets is the
/// canopy's. Anything wanting both would have to draw them as two surfaces
/// rather than pick one at startup.
pub const ELEVATION_PRODUCT: &str = "dtm";

/// Any elevation below this means "no measurement here".
///
/// Defined beside the filter that has to drop such texels when it builds a
/// coarse level, rather than copied here: the renderer and the reduction have to
/// agree on what a hole is or a level of the pyramid will quietly average some
/// in. The shader carries its own copy, which cannot be shared and is marked to
/// be kept in step.
pub use terrain_tiles::NODATA_BELOW;
