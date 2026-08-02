//! What a product directory holds, written beside its tiles.
//!
//! Without this the renderer would have to scan tens of thousands of files to
//! learn the extent and the level count before it could draw anything, and it
//! would still be guessing at the parts a filename does not carry. One small
//! file answers all of it, and it is the thing two products are compared
//! through to decide they describe the same ground.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::{TILE_SIZE, Tile, TileGrid};

/// The file a product directory is described by.
pub const MANIFEST_NAME: &str = "manifest.json";

/// Level at which colour is stored, and so the granularity everything snaps to.
///
/// Sentinel-2's cloud-free mosaics arrive at about 19 m, so level 4 -- 16 m --
/// is the closest the grid comes without inventing detail. Because the download
/// extent is snapped to a whole tile at *this* level, both the 512 m elevation
/// grid and the 8192 m colour grid tile it exactly, which is what lets the two
/// manifests be compared for equality rather than for approximate agreement.
pub const COLOUR_BASE_LEVEL: u32 = 4;

/// Level at which ground-cover materials are stored.
///
/// Materials come from OpenStreetMap vector data, which has no resolution of
/// its own, so this is a choice rather than a snap to a source. Level 2 --
/// 4 m texels -- keeps the shoreline of a pond and the edge of a pitch
/// legible while costing a quarter of what level 0 would, and the renderer
/// serves finer requests by repeating texels, as it already does for colour.
pub const MATERIAL_BASE_LEVEL: u32 = 2;

/// Describes one product's tile tree.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Manifest {
    /// Bumped whenever a reader could otherwise misinterpret an older tree.
    pub version: u32,
    /// `dtm`, `dsm` or `albedo`.
    pub product: String,
    /// The projected CRS the grid is anchored to.
    pub epsg: u32,
    /// Always [`TILE_SIZE`]; stored so a mismatch is an error, not corruption.
    pub tile_size: u32,
    /// The finest level stored: 0 for elevation, [`COLOUR_BASE_LEVEL`] for colour.
    pub base_level: u32,
    /// How many levels are stored, counting up from `base_level`.
    pub level_count: u32,
    /// Ground size of a level-0 texel, whatever `base_level` is.
    pub base_metres_per_texel: f64,
    /// North-west corner of the covered ground, as an easting and a northing.
    pub origin_metres: [f64; 2],
    /// Size of the covered ground, in *level-0* texels.
    ///
    /// Level 0 even when nothing is stored there, so that two products of one
    /// download hold identical numbers and can be compared directly.
    pub extent_texels: [u32; 2],
    /// Samples per texel in the stored files: 1 for elevation, 3 for colour.
    pub bands: u32,
    /// The value meaning "nothing known here".
    pub nodata: f32,
}

impl Manifest {
    /// The only version this code writes or accepts.
    pub const VERSION: u32 = 1;

    /// Where a product's manifest sits inside its directory.
    pub fn path_in(root: &Path) -> PathBuf {
        root.join(MANIFEST_NAME)
    }

    /// Reads and validates a manifest.
    ///
    /// Validation is deliberately strict. Every field here is used to place
    /// terrain on the ground, and a wrong one shows up as scenery in the wrong
    /// place rather than as a failure, so it is worth refusing loudly.
    pub fn read(root: &Path) -> Result<Self> {
        let path = Self::path_in(root);
        let text =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let manifest: Self =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        manifest
            .validate()
            .with_context(|| format!("in {}", path.display()))?;
        Ok(manifest)
    }

    /// Writes the manifest, creating the product directory if it is not there.
    pub fn write(&self, root: &Path) -> Result<()> {
        self.validate()?;
        fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
        let path = Self::path_in(root);
        let text = serde_json::to_string_pretty(self)?;
        fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == Self::VERSION,
            "manifest version {} is not the {} this build understands",
            self.version,
            Self::VERSION
        );
        ensure!(
            self.tile_size == TILE_SIZE,
            "tiles are {} texels but this build is built for {TILE_SIZE}",
            self.tile_size
        );
        ensure!(self.level_count >= 1, "a product needs at least one level");
        ensure!(self.bands >= 1, "a product needs at least one band");
        ensure!(
            self.base_metres_per_texel.is_finite() && self.base_metres_per_texel > 0.0,
            "a level-0 texel is not a positive size: {}",
            self.base_metres_per_texel
        );
        ensure!(
            self.origin_metres.iter().all(|v| v.is_finite()),
            "the origin is not a finite position: {:?}",
            self.origin_metres
        );
        ensure!(
            self.extent_texels[0] > 0 && self.extent_texels[1] > 0,
            "the extent is empty: {:?}",
            self.extent_texels
        );

        // Every stored level has to divide the extent exactly, or the coarsest
        // level would describe a different piece of ground than the finest.
        let step = 1u32
            .checked_shl(self.max_level())
            .context("level count is absurdly large")?;
        ensure!(
            self.extent_texels[0].is_multiple_of(step)
                && self.extent_texels[1].is_multiple_of(step),
            "extent {:?} does not divide evenly into level {} texels of {step}",
            self.extent_texels,
            self.max_level()
        );
        Ok(())
    }

    /// The grid this product's tiles sit on.
    pub fn grid(&self) -> TileGrid {
        TileGrid {
            epsg: self.epsg,
            base_level: self.base_level,
            base_metres_per_texel: self.base_metres_per_texel,
        }
    }

    /// The coarsest level stored.
    pub fn max_level(&self) -> u32 {
        self.base_level + self.level_count - 1
    }

    /// Ground size of one texel at `level`.
    pub fn metres_per_texel(&self, level: u32) -> f64 {
        self.base_metres_per_texel * f64::from(1u32 << level)
    }

    /// Size of the covered ground at `level`, in that level's texels.
    pub fn size_texels(&self, level: u32) -> (u32, u32) {
        (
            (self.extent_texels[0] >> level).max(1),
            (self.extent_texels[1] >> level).max(1),
        )
    }

    /// Where the raster's north-west corner sits on the global texel lattice.
    ///
    /// Returned as a column and a row, both counting away from the projection
    /// origin in the direction the raster's own indices run. `i64` because a
    /// level-0 index in metres reaches a few million and the arithmetic below
    /// adds tile-sized offsets to it.
    pub fn origin_texels(&self, level: u32) -> (i64, i64) {
        let metres = self.metres_per_texel(level);
        (
            (self.origin_metres[0] / metres).round() as i64,
            (-self.origin_metres[1] / metres).round() as i64,
        )
    }

    /// The tile holding a texel, and where in that tile it sits.
    pub fn tile_of_texel(&self, level: u32, column: i64, row: i64) -> (Tile, u32, u32) {
        let (origin_column, origin_row) = self.origin_texels(level);
        let (global_column, global_row) = (origin_column + column, origin_row + row);
        let size = i64::from(TILE_SIZE);
        (
            Tile::new(
                global_column.div_euclid(size) as i32,
                global_row.div_euclid(size) as i32,
            ),
            global_column.rem_euclid(size) as u32,
            global_row.rem_euclid(size) as u32,
        )
    }

    /// Whether two products describe exactly the same ground.
    ///
    /// Exact equality rather than a tolerance, because both manifests are
    /// written by one run over one snapped extent. Resolutions may differ --
    /// colour is stored coarser than elevation -- but the ground may not.
    pub fn covers_same_ground_as(&self, other: &Self) -> bool {
        self.epsg == other.epsg
            && self.origin_metres == other.origin_metres
            && self.extent_texels == other.extent_texels
            && self.base_metres_per_texel == other.base_metres_per_texel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: "dtm".into(),
            epsg: 3979,
            tile_size: TILE_SIZE,
            base_level: 0,
            level_count: 5,
            base_metres_per_texel: 1.0,
            // A multiple of 8192, as the downloader's snap guarantees.
            origin_metres: [-1_974_272.0, 524_288.0],
            extent_texels: [16_384, 8_192],
            bands: 1,
            nodata: -32767.0,
        }
    }

    #[test]
    fn a_manifest_round_trips_through_json() {
        let directory = std::env::temp_dir().join("terrain-tiles-round-trip");
        let original = manifest();
        original.write(&directory).expect("failed to write");
        let read = Manifest::read(&directory).expect("failed to read");
        assert_eq!(read, original);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn a_future_version_is_refused_rather_than_guessed_at() {
        let manifest = Manifest {
            version: Manifest::VERSION + 1,
            ..manifest()
        };
        let message = manifest.validate().expect_err("should refuse").to_string();
        assert!(message.contains("version"), "got {message}");
    }

    #[test]
    fn an_extent_that_does_not_divide_into_levels_is_refused() {
        let manifest = Manifest {
            extent_texels: [16_384 + 1, 8_192],
            ..manifest()
        };
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn levels_halve_the_extent() {
        let manifest = manifest();
        assert_eq!(manifest.size_texels(0), (16_384, 8_192));
        assert_eq!(manifest.size_texels(1), (8_192, 4_096));
        assert_eq!(manifest.size_texels(4), (1_024, 512));
        assert_eq!(manifest.max_level(), 4);
    }

    /// The origin is snapped to a whole colour tile, so it lands on a tile
    /// boundary at every level the elevation grid uses.
    #[test]
    fn the_origin_lands_on_a_tile_boundary_up_to_the_colour_level() {
        let manifest = manifest();
        for level in 0..=COLOUR_BASE_LEVEL {
            let (tile, column, row) = manifest.tile_of_texel(level, 0, 0);
            assert_eq!((column, row), (0, 0), "level {level} lands mid-tile");
            let grid = manifest.grid();
            let (west, north) = grid.tile_origin_metres(level, tile);
            assert_eq!(
                (west, north),
                (manifest.origin_metres[0], manifest.origin_metres[1]),
                "level {level}"
            );
        }
    }

    #[test]
    fn texels_walk_into_the_next_tile_at_the_boundary() {
        let manifest = manifest();
        let (first, column, _) = manifest.tile_of_texel(0, 0, 0);
        assert_eq!(column, 0);
        let (same, column, _) = manifest.tile_of_texel(0, i64::from(TILE_SIZE) - 1, 0);
        assert_eq!(same, first);
        assert_eq!(column, TILE_SIZE - 1);
        let (next, column, _) = manifest.tile_of_texel(0, i64::from(TILE_SIZE), 0);
        assert_eq!(next, Tile::new(first.x + 1, first.y));
        assert_eq!(column, 0);
    }

    /// Reading a texel to the north-west of the raster is legal -- the clipmap's
    /// windows hang off the edge of the world -- so the indices must go
    /// negative cleanly rather than wrapping.
    #[test]
    fn negative_texels_reach_the_tile_to_the_north_west() {
        let manifest = manifest();
        let (first, _, _) = manifest.tile_of_texel(0, 0, 0);
        let (before, column, row) = manifest.tile_of_texel(0, -1, -1);
        assert_eq!(before, Tile::new(first.x - 1, first.y - 1));
        assert_eq!((column, row), (TILE_SIZE - 1, TILE_SIZE - 1));
    }

    #[test]
    fn products_of_one_download_cover_the_same_ground() {
        let elevation = manifest();
        let colour = Manifest {
            product: "albedo".into(),
            base_level: COLOUR_BASE_LEVEL,
            level_count: 1,
            bands: 3,
            nodata: 0.0,
            ..manifest()
        };
        assert!(elevation.covers_same_ground_as(&colour));

        let elsewhere = Manifest {
            origin_metres: [0.0, 0.0],
            ..colour
        };
        assert!(!elevation.covers_same_ground_as(&elsewhere));
    }
}
