//! Which tiles a product's ground occupies at a level.

use terrain_tiles::{Manifest, Tile};

/// The tiles a product's ground occupies at one level: the north-west one, and
/// how many there are on each axis.
///
/// Derived from the manifest rather than by halving the level below, because a
/// coarse level's tiles are only a clean halving while the origin stays on a
/// tile boundary.
pub fn tile_range(manifest: &Manifest, level: u32) -> (Tile, u32, u32) {
    let (width, height) = manifest.size_texels(level);
    let (first, _, _) = manifest.tile_of_texel(level, 0, 0);
    let (last, _, _) = manifest.tile_of_texel(level, i64::from(width) - 1, i64::from(height) - 1);
    (
        first,
        (last.x - first.x + 1) as u32,
        (last.y - first.y + 1) as u32,
    )
}
