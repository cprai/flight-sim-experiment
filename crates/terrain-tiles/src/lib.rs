//! The tile grid the terrain is stored on, shared by the writer and the reader.
//!
//! Terrain is kept as square tiles on a global quadtree grid anchored at the
//! projection's own origin. A tile at level `L` holds [`TILE_SIZE`] squared
//! texels of `2^L` metres each, so it spans `TILE_SIZE * 2^L` metres of ground,
//! and its edges land on multiples of that span measured from the origin. Level
//! `L + 1` tiles therefore cover exactly four level-`L` tiles, with no rounding
//! anywhere in the chain.
//!
//! Anchoring globally rather than to the downloaded box is what makes two
//! downloads of neighbouring ground line up: the grid does not depend on which
//! box was asked for. It also means a coarse tile may hang off the edge of the
//! data, which is fine -- the part with nothing behind it is nodata.
//!
//! The grid is in projected metres, EPSG:3979 for everything this project
//! fetches. That is the CRS HRDEM is published on, on an integer-metre lattice,
//! so tile boundaries are always source pixel boundaries and the finest level is
//! a copy rather than a resampling.
//!
//! This crate exists because the downloader and the renderer have to agree on
//! all of it exactly. A disagreement of one tile displaces the terrain by half a
//! kilometre and nothing would report an error.

use std::path::{Path, PathBuf};

pub mod manifest;
pub mod maxima;
pub mod read;
pub mod texel;
pub mod write;

pub use manifest::{COLOUR_BASE_LEVEL, Manifest};
pub use texel::{
    COLOUR_IS_SRGB_ENCODED, NODATA_BELOW, Srgb8, Texel, linear_to_srgb, srgb_to_linear,
};

/// What marks a product directory as a max pyramid rather than a measurement.
pub const MAXIMA_SUFFIX: &str = "-max";

/// What a max pyramid's product directory is called, given the elevation it was
/// reduced from.
///
/// A suffix rather than one shared name because `dtm` and `dsm` are different
/// surfaces and each needs its own bound: the renderer picks whichever elevation
/// product is installed and must get the pyramid built from that one, not from
/// the other. Derived here so the tool that writes it and the renderer that
/// opens it cannot spell it differently.
pub fn maxima_product(elevation: &str) -> String {
    format!("{elevation}{MAXIMA_SUFFIX}")
}

/// Side length of every tile, in texels, at every level.
///
/// 512 is a compromise between two costs that pull opposite ways. Larger tiles
/// mean fewer files -- a 57 x 90 km box at one metre is already 38 000 tiles at
/// this size -- while smaller tiles mean less waste when the renderer wants a
/// thin strip, since it must touch every tile the strip crosses. 512 keeps a
/// 256-texel clipmap window inside a 2 x 2 neighbourhood at every level.
pub const TILE_SIZE: u32 = 512;

/// A tile's position on the grid, in tiles.
///
/// Both axes are signed and both count away from the projection origin in the
/// same direction the raster's own indices run: `x` eastward, `y` *southward*.
/// Southward rather than northward so that tile rows and texel rows agree in
/// sign, which removes a class of off-by-one that is otherwise very easy to
/// write. Everything this project fetches is north-west of EPSG:3979's origin
/// at 95W 49N, so both indices are usually negative.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Tile {
    pub x: i32,
    pub y: i32,
}

impl Tile {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Which levels a product is stored at, and how large its texels are.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TileGrid {
    /// The projected CRS the grid is anchored to.
    pub epsg: u32,
    /// The finest level actually stored.
    ///
    /// Zero for elevation. Colour starts at 4, because Sentinel-2 is a 19 m
    /// product and storing it at one metre would be storing sixteen times its
    /// own resolution in invented detail.
    pub base_level: u32,
    /// Ground size of a level-0 texel, whatever `base_level` is.
    ///
    /// Level numbering is absolute so that two products with different base
    /// levels still mean the same thing by "level 4".
    pub base_metres_per_texel: f64,
}

impl TileGrid {
    /// Ground size of one texel at `level`.
    pub fn metres_per_texel(&self, level: u32) -> f64 {
        self.base_metres_per_texel * f64::from(1u32 << level)
    }

    /// Ground size of one whole tile at `level`.
    pub fn tile_span_metres(&self, level: u32) -> f64 {
        self.metres_per_texel(level) * f64::from(TILE_SIZE)
    }

    /// The tile covering a point, given as an easting and a northing.
    pub fn tile_of_metres(&self, level: u32, easting: f64, northing: f64) -> Tile {
        let span = self.tile_span_metres(level);
        Tile::new(
            (easting / span).floor() as i32,
            // Negated because `y` runs southward while northings run north.
            ((-northing) / span).floor() as i32,
        )
    }

    /// The north-west corner of a tile, as an easting and a northing.
    pub fn tile_origin_metres(&self, level: u32, tile: Tile) -> (f64, f64) {
        let span = self.tile_span_metres(level);
        (f64::from(tile.x) * span, -f64::from(tile.y) * span)
    }

    /// Where a tile's file lives under a product directory.
    ///
    /// One directory per level keeps any single directory to a few thousand
    /// entries even for a large box, which matters for both the filesystem and
    /// for anyone looking at it.
    pub fn tile_path(&self, root: &Path, level: u32, tile: Tile) -> PathBuf {
        root.join(format!("{level:02}"))
            .join(format!("{}_{}.tif", tile.x, tile.y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> TileGrid {
        TileGrid {
            epsg: 3979,
            base_level: 0,
            base_metres_per_texel: 1.0,
        }
    }

    #[test]
    fn a_tile_spans_its_texels() {
        let grid = grid();
        for level in 0..8 {
            assert_eq!(
                grid.tile_span_metres(level),
                grid.metres_per_texel(level) * f64::from(TILE_SIZE),
                "level {level}"
            );
        }
        assert_eq!(grid.tile_span_metres(0), 512.0);
        assert_eq!(grid.tile_span_metres(4), 8192.0);
    }

    /// The property the whole quadtree rests on: one coarse tile is exactly the
    /// four fine tiles beneath it, with no rounding at any level.
    #[test]
    fn a_coarse_tile_covers_exactly_four_fine_ones() {
        let grid = grid();
        for level in 0..12 {
            assert_eq!(
                grid.tile_span_metres(level + 1),
                grid.tile_span_metres(level) * 2.0,
                "level {level}"
            );
        }
    }

    #[test]
    fn a_points_tile_contains_it() {
        let grid = grid();
        // Squamish in EPSG:3979, which is north-west of the false origin, so
        // both indices come out negative.
        for (easting, northing) in [
            (-1_973_119.85, 516_927.57),
            (-1_956_653.44, 517_123.37),
            (0.0, 0.0),
            (511.9, -0.1),
        ] {
            for level in 0..6 {
                let tile = grid.tile_of_metres(level, easting, northing);
                let (west, north) = grid.tile_origin_metres(level, tile);
                let span = grid.tile_span_metres(level);
                assert!(
                    west <= easting && easting < west + span,
                    "level {level}: {easting} not in {west}..{}",
                    west + span
                );
                assert!(
                    north >= northing && northing > north - span,
                    "level {level}: {northing} not in {}..{north}",
                    north - span
                );
            }
        }
    }

    /// A truncating cast would fold the two tiles either side of the origin into
    /// one, which is the classic way a global grid goes wrong near its anchor.
    #[test]
    fn indices_floor_rather_than_truncate_towards_zero() {
        let grid = grid();
        assert_eq!(grid.tile_of_metres(0, -1.0, 0.0).x, -1);
        assert_eq!(grid.tile_of_metres(0, -512.0, 0.0).x, -1);
        assert_eq!(grid.tile_of_metres(0, -513.0, 0.0).x, -2);
        assert_eq!(grid.tile_of_metres(0, 0.0, 0.0).x, 0);
        // Northings mirror: just north of the origin is tile row -1.
        assert_eq!(grid.tile_of_metres(0, 0.0, 1.0).y, -1);
        assert_eq!(grid.tile_of_metres(0, 0.0, 0.0).y, 0);
        assert_eq!(grid.tile_of_metres(0, 0.0, -1.0).y, 0);
    }

    #[test]
    fn a_tiles_origin_maps_back_to_that_tile() {
        let grid = grid();
        for level in 0..6 {
            for tile in [
                Tile::new(0, 0),
                Tile::new(-3855, -1010),
                Tile::new(7, -3),
                Tile::new(-1, -1),
            ] {
                let (west, north) = grid.tile_origin_metres(level, tile);
                assert_eq!(
                    grid.tile_of_metres(level, west, north),
                    tile,
                    "level {level}"
                );
            }
        }
    }

    #[test]
    fn tile_paths_carry_the_level_and_the_indices() {
        let grid = grid();
        let path = grid.tile_path(Path::new("/tmp/terrain/dtm"), 4, Tile::new(-239, -122));
        assert_eq!(
            path,
            Path::new("/tmp/terrain/dtm/04/-239_-122.tif"),
            "got {}",
            path.display()
        );
    }

    /// Colour is stored coarser than elevation, but level numbering is shared,
    /// so a level-4 tile is the same ground whichever product it belongs to.
    #[test]
    fn products_with_different_base_levels_share_a_lattice() {
        let elevation = grid();
        let colour = TileGrid {
            base_level: 4,
            ..grid()
        };
        assert_eq!(elevation.tile_span_metres(4), colour.tile_span_metres(4));
        let point = (-1_973_119.85, 516_927.57);
        assert_eq!(
            elevation.tile_of_metres(4, point.0, point.1),
            colour.tile_of_metres(4, point.0, point.1)
        );
    }
}
