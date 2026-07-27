//! The ground a download covers, snapped out to whole tiles.
//!
//! The user asks for a longitude/latitude box; what gets written is a set of
//! tiles on a fixed global lattice, which almost never lines up with what was
//! asked for. This module is the join between the two: it projects the box onto
//! EPSG:3979, grows it out to tile boundaries, and works out how many levels the
//! pyramid needs.
//!
//! Everything is snapped to a whole *colour* tile -- 8192 m -- rather than a
//! whole elevation tile. That is a multiple of every tile span from level 0 up
//! to level 4, so both products tile the same ground exactly and their manifests
//! can be compared for equality rather than for approximate agreement.

use anyhow::{Context, Result};
use terrain_tiles::{COLOUR_BASE_LEVEL, Manifest, TILE_SIZE, Tile, TileGrid};

use crate::bbox::LatLonBox;
use crate::project::{self, Projector};
use crate::resample::Grid;

/// Ground size of a level-0 texel.
///
/// One metre, which is HRDEM's native sampling. Nothing resamples elevation on
/// the way in because the output grid is the grid it was published on.
pub const BASE_METRES_PER_TEXEL: f64 = 1.0;

/// The extent is snapped outwards to a multiple of this, in metres.
pub const SNAP_METRES: f64 =
    BASE_METRES_PER_TEXEL * (TILE_SIZE as f64) * (1u32 << COLOUR_BASE_LEVEL) as f64;

/// How many points to sample along each edge of the requested box.
///
/// The projection sends a point to `x = r sin t`, `y = r0 - r cos t`, where `r`
/// depends only on latitude and `t` only on longitude. Both products are
/// monotonic in each variable, so away from the central meridian the corners
/// already bound the region. The exception is `cos t`, which peaks at 95 degrees
/// west: a box straddling it has a southern edge that bows below both of its
/// corners, by 14.4 km for 100W..90W at 49N..51N. Walking the boundary finds
/// that with no special case, and 1024 points puts samples a kilometre apart
/// even on a box ten degrees wide.
const EDGE_SAMPLES: u32 = 1024;

/// The coarsest level that will ever be built.
///
/// At level 13 one tile spans 4194 km, which is wider than the country. The cap
/// exists because the extent is snapped to 8192 m -- two to the thirteen -- and
/// every level has to divide it exactly; a coarser level would not.
const MAX_LEVEL: u32 = 13;

/// One block of the output: a few tiles square, filled and written as a unit.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Block {
    /// The block's north-west tile, in global tile indices.
    pub tile: Tile,
    pub tiles_across: u32,
    pub tiles_down: u32,
    /// The texels the block covers.
    pub grid: Grid,
}

/// The ground a download covers, on the tile lattice.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TileExtent {
    /// Easting of the western edge, a multiple of [`SNAP_METRES`].
    pub west: f64,
    /// Northing of the northern edge, a multiple of [`SNAP_METRES`].
    pub north: f64,
    /// Width in level-0 texels.
    pub width: u32,
    /// Height in level-0 texels.
    pub height: u32,
    /// The coarsest level worth building.
    pub max_level: u32,
}

impl TileExtent {
    /// Projects the requested box and snaps it out to whole tiles.
    ///
    /// The result always covers at least one tile, so asking for a square metre
    /// of ground still produces something the renderer can open.
    pub fn cover(box_: LatLonBox) -> Result<Self> {
        let projector = Projector::from_geographic(project::EPSG_LAMBERT)?;

        let mut outline = Vec::with_capacity(4 * (EDGE_SAMPLES as usize + 1));
        for step in 0..=EDGE_SAMPLES {
            let fraction = f64::from(step) / f64::from(EDGE_SAMPLES);
            let longitude = box_.west + fraction * box_.width_degrees();
            let latitude = box_.south + fraction * box_.height_degrees();
            outline.push((longitude, box_.south));
            outline.push((longitude, box_.north));
            outline.push((box_.west, latitude));
            outline.push((box_.east, latitude));
        }
        projector
            .to_source(&mut outline)
            .context("projecting the requested box onto the Canada Atlas Lambert grid")?;

        let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
        let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (x, y) in outline {
            anyhow::ensure!(
                x.is_finite() && y.is_finite(),
                "the requested box does not project onto the Canada Atlas Lambert grid"
            );
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        let west = (min_x / SNAP_METRES).floor() * SNAP_METRES;
        let north = (max_y / SNAP_METRES).ceil() * SNAP_METRES;
        let east = (max_x / SNAP_METRES).ceil() * SNAP_METRES;
        let south = (min_y / SNAP_METRES).floor() * SNAP_METRES;

        // At least one snap unit on each axis, so a degenerate box still writes
        // a tile rather than an empty tree.
        let width = ((east - west).max(SNAP_METRES) / BASE_METRES_PER_TEXEL).round();
        let height = ((north - south).max(SNAP_METRES) / BASE_METRES_PER_TEXEL).round();
        anyhow::ensure!(
            width <= f64::from(u32::MAX) && height <= f64::from(u32::MAX),
            "the requested box covers {width} x {height} texels, which is too large to index"
        );
        let (width, height) = (width as u32, height as u32);

        Ok(Self {
            west,
            north,
            width,
            height,
            max_level: max_level_for(width.max(height)),
        })
    }

    /// The ground the tiles actually cover, back in degrees.
    ///
    /// Both catalogues are searched geographically, and searching by the box
    /// the user typed is wrong once it has been snapped outwards: the extent is
    /// a whole 8192 m tile on each side, so a box a few hundred metres across
    /// grows into one eight kilometres across. Searching by the box found only
    /// the imagery square under its centre and left the rest of the extent
    /// black -- a diagonal cut across the tile, because Lambert grid north is
    /// rotated about 25 degrees from true north at 123W and the square's
    /// northern edge crosses the tile at an angle.
    ///
    /// The boundary is walked rather than inverted at the corners, for the same
    /// reason the forward direction walks it.
    pub fn geographic_box(&self) -> Result<LatLonBox> {
        let projector = Projector::from_geographic(project::EPSG_LAMBERT)?;
        let extent = self.grid(0).extent();

        let mut outline = Vec::with_capacity(4 * (EDGE_SAMPLES as usize + 1));
        for step in 0..=EDGE_SAMPLES {
            let fraction = f64::from(step) / f64::from(EDGE_SAMPLES);
            let x = extent.min_x + fraction * (extent.max_x - extent.min_x);
            let y = extent.min_y + fraction * (extent.max_y - extent.min_y);
            outline.push((x, extent.min_y));
            outline.push((x, extent.max_y));
            outline.push((extent.min_x, y));
            outline.push((extent.max_x, y));
        }
        projector
            .to_output(&mut outline)
            .context("inverting the output extent back to longitude and latitude")?;

        let (mut west, mut south) = (f64::INFINITY, f64::INFINITY);
        let (mut east, mut north) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (longitude, latitude) in outline {
            anyhow::ensure!(
                longitude.is_finite() && latitude.is_finite(),
                "the output extent does not invert to longitude and latitude"
            );
            west = west.min(longitude);
            south = south.min(latitude);
            east = east.max(longitude);
            north = north.max(latitude);
        }

        Ok(LatLonBox {
            west,
            south,
            east,
            north,
        })
    }

    /// Size of the extent at `level`, in that level's texels.
    pub fn size_texels(&self, level: u32) -> (u32, u32) {
        ((self.width >> level).max(1), (self.height >> level).max(1))
    }

    /// The texel grid the extent covers at `level`.
    pub fn grid(&self, level: u32) -> Grid {
        let (width, height) = self.size_texels(level);
        Grid {
            west: self.west,
            north: self.north,
            width,
            height,
            metres_per_texel: BASE_METRES_PER_TEXEL * f64::from(1u32 << level),
        }
    }

    /// The tiles at `level` that any of the extent falls inside.
    ///
    /// Worked out from the ground each tile covers rather than by dividing the
    /// texel count, because the extent is only guaranteed to land on a tile
    /// boundary up to [`COLOUR_BASE_LEVEL`]. Above that a tile spans more than
    /// the snap unit, so the extent can sit part-way across one.
    pub fn tile_range(&self, level: u32) -> (Tile, u32, u32) {
        let grid = self.tile_grid();
        let first = grid.tile_of_metres(level, self.west, self.north);
        // A hair inside the far edge, so an extent ending exactly on a boundary
        // does not claim the tile beyond it.
        let inside = BASE_METRES_PER_TEXEL * 0.5;
        let last = grid.tile_of_metres(
            level,
            self.west + f64::from(self.width) - inside,
            self.north - f64::from(self.height) + inside,
        );
        (
            first,
            (last.x - first.x + 1) as u32,
            (last.y - first.y + 1) as u32,
        )
    }

    /// How many tiles across and down the extent is at `level`.
    pub fn tiles(&self, level: u32) -> (u32, u32) {
        let (_, across, down) = self.tile_range(level);
        (across, down)
    }

    /// The lattice the tiles sit on.
    pub fn tile_grid(&self) -> TileGrid {
        TileGrid {
            epsg: u32::from(project::EPSG_LAMBERT),
            base_level: 0,
            base_metres_per_texel: BASE_METRES_PER_TEXEL,
        }
    }

    /// How many tiles the finest level of a product would hold.
    ///
    /// Counted before anything is fetched, as the guard against a box that
    /// would take days and fill the disk. It is an upper bound: tiles with no
    /// data under them are never written.
    pub fn tile_count(&self, base_level: u32) -> u64 {
        let (across, down) = self.tiles(base_level);
        u64::from(across) * u64::from(down)
    }

    /// Splits `level` into blocks of at most `block_tiles` square.
    ///
    /// Blocks are the unit of work: one is filled from the network, cut into
    /// tiles, written, and dropped before the next begins, so peak memory
    /// depends on the block size rather than on the size of the box.
    pub fn blocks(&self, level: u32, block_tiles: u32) -> Vec<Block> {
        let block_tiles = block_tiles.max(1);
        let (first, across, down) = self.tile_range(level);
        let tile_grid = self.tile_grid();
        let metres_per_texel = BASE_METRES_PER_TEXEL * f64::from(1u32 << level);

        let mut blocks = Vec::new();
        let mut row = 0;
        while row < down {
            let tiles_down = block_tiles.min(down - row);
            let mut column = 0;
            while column < across {
                let tiles_across = block_tiles.min(across - column);
                let tile = Tile::new(first.x + column as i32, first.y + row as i32);
                // Placed from the tile's own origin rather than the extent's, so
                // a block still lands correctly at a level whose tiles are
                // coarser than the extent is aligned to.
                let (west, north) = tile_grid.tile_origin_metres(level, tile);
                blocks.push(Block {
                    tile,
                    tiles_across,
                    tiles_down,
                    grid: Grid {
                        west,
                        north,
                        width: tiles_across * TILE_SIZE,
                        height: tiles_down * TILE_SIZE,
                        metres_per_texel,
                    },
                });
                column += tiles_across;
            }
            row += tiles_down;
        }
        blocks
    }

    /// The manifest describing a product written over this extent.
    pub fn manifest(&self, product: &str, base_level: u32, bands: u32, nodata: f32) -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: product.to_string(),
            epsg: u32::from(project::EPSG_LAMBERT),
            tile_size: TILE_SIZE,
            base_level,
            level_count: self.max_level - base_level + 1,
            base_metres_per_texel: BASE_METRES_PER_TEXEL,
            origin_metres: [self.west, self.north],
            extent_texels: [self.width, self.height],
            bands,
            nodata,
        }
    }
}

/// The coarsest level worth building for an extent this many texels across.
///
/// Stops as soon as the extent fits inside a single tile's *span*, so the
/// coarsest level is at most two tiles on a side. Requiring it to fall inside
/// one actual tile would not always terminate: an extent straddling a boundary
/// can keep straddling however far the span doubles.
fn max_level_for(texels: u32) -> u32 {
    let mut level = COLOUR_BASE_LEVEL;
    while level < MAX_LEVEL && (texels >> level) > TILE_SIZE {
        level += 1;
    }
    level
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::LatLon;

    fn corner(latitude: f64, longitude: f64) -> LatLon {
        LatLon {
            latitude,
            longitude,
        }
    }

    /// The box every other part of this project is checked against.
    fn squamish() -> LatLonBox {
        LatLonBox::from_corners(corner(49.633, -123.307), corner(49.637, -123.303))
            .expect("failed to build a box")
    }

    #[test]
    fn the_extent_lands_on_whole_tile_boundaries() {
        let extent = TileExtent::cover(squamish()).expect("failed to cover");
        assert_eq!(extent.west % SNAP_METRES, 0.0, "west {}", extent.west);
        assert_eq!(extent.north % SNAP_METRES, 0.0, "north {}", extent.north);
        assert_eq!(f64::from(extent.width) % SNAP_METRES, 0.0);
        assert_eq!(f64::from(extent.height) % SNAP_METRES, 0.0);
    }

    /// A box far smaller than a tile still has to produce one.
    #[test]
    fn a_tiny_box_still_covers_a_whole_tile() {
        let tiny = LatLonBox::from_corners(corner(49.6330, -123.3070), corner(49.6331, -123.3069))
            .expect("failed to build a box");
        let extent = TileExtent::cover(tiny).expect("failed to cover");
        assert!(f64::from(extent.width) >= SNAP_METRES);
        assert!(f64::from(extent.height) >= SNAP_METRES);
        assert_eq!(extent.tiles(COLOUR_BASE_LEVEL), (1, 1));
    }

    /// The requested ground must end up inside what gets written, or the
    /// download would quietly miss the corner the user asked for.
    #[test]
    fn the_requested_box_falls_inside_the_snapped_extent() {
        let box_ = squamish();
        let extent = TileExtent::cover(box_).expect("failed to cover");
        let projector = Projector::from_geographic(project::EPSG_LAMBERT).expect("failed to build");

        for (longitude, latitude) in [
            (box_.west, box_.south),
            (box_.west, box_.north),
            (box_.east, box_.south),
            (box_.east, box_.north),
        ] {
            let (x, y) = projector
                .point_to_source(longitude, latitude)
                .expect("failed to project");
            assert!(
                x >= extent.west && x <= extent.west + f64::from(extent.width),
                "easting {x} outside the extent"
            );
            assert!(
                y <= extent.north && y >= extent.north - f64::from(extent.height),
                "northing {y} outside the extent"
            );
        }
    }

    /// The catalogues are searched by this, so it has to describe the ground the
    /// tiles cover rather than the ground that was asked for. Searching by the
    /// typed box instead found only the imagery square under the box's centre
    /// and left the rest of the extent black.
    #[test]
    fn the_geographic_box_covers_everything_the_tiles_will() {
        let box_ = squamish();
        let extent = TileExtent::cover(box_).expect("failed to cover");
        let search = extent.geographic_box().expect("failed to invert");

        assert!(search.west <= box_.west, "{} vs {}", search.west, box_.west);
        assert!(search.east >= box_.east, "{} vs {}", search.east, box_.east);
        assert!(search.south <= box_.south);
        assert!(search.north >= box_.north);

        // The snap grows a 400 m box to 8192 m, so the search box has to be
        // substantially larger -- this is the failure the test exists for.
        assert!(
            search.north - search.south > 0.05,
            "the search box spans only {} degrees of latitude",
            search.north - search.south
        );

        // Every corner of the extent must invert back inside it.
        let projector = Projector::from_geographic(project::EPSG_LAMBERT).expect("failed to build");
        let corner = extent.grid(0).extent();
        for (x, y) in [
            (corner.min_x, corner.min_y),
            (corner.min_x, corner.max_y),
            (corner.max_x, corner.min_y),
            (corner.max_x, corner.max_y),
        ] {
            let mut point = [(x, y)];
            projector.to_output(&mut point).expect("failed to invert");
            let (longitude, latitude) = point[0];
            assert!(
                (search.west..=search.east).contains(&longitude),
                "longitude {longitude} outside the search box"
            );
            assert!(
                (search.south..=search.north).contains(&latitude),
                "latitude {latitude} outside the search box"
            );
        }
    }

    /// The whole point of the snap: elevation and colour tile the same ground.
    #[test]
    fn both_products_tile_the_extent_exactly() {
        let extent = TileExtent::cover(squamish()).expect("failed to cover");
        for level in [0, COLOUR_BASE_LEVEL] {
            let (width, height) = extent.size_texels(level);
            assert_eq!(width % TILE_SIZE, 0, "level {level} width {width}");
            assert_eq!(height % TILE_SIZE, 0, "level {level} height {height}");
        }

        let elevation = extent.manifest("dtm", 0, 1, -32767.0);
        let colour = extent.manifest("albedo", COLOUR_BASE_LEVEL, 3, 0.0);
        assert!(elevation.covers_same_ground_as(&colour));
        assert_eq!(elevation.max_level(), colour.max_level());
    }

    #[test]
    fn levels_stop_once_the_extent_fits_a_single_tile_span() {
        assert_eq!(max_level_for(TILE_SIZE), COLOUR_BASE_LEVEL);
        // 8192 texels needs level 4 exactly: 8192 >> 4 == 512.
        assert_eq!(max_level_for(8_192), 4);
        // 16384 needs one more.
        assert_eq!(max_level_for(16_384), 5);
        assert_eq!(max_level_for(u32::MAX), MAX_LEVEL);
    }

    /// Every level has to divide the extent exactly, which is what the manifest
    /// refuses to accept otherwise.
    #[test]
    fn a_manifest_over_the_extent_validates() {
        for box_ in [
            squamish(),
            LatLonBox::from_corners(corner(49.0, -124.0), corner(50.0, -123.0))
                .expect("failed to build a box"),
        ] {
            let extent = TileExtent::cover(box_).expect("failed to cover");
            let directory = std::env::temp_dir().join(format!(
                "terrain-download-extent-{}-{}",
                std::process::id(),
                extent.width
            ));
            extent
                .manifest("dtm", 0, 1, -32767.0)
                .write(&directory)
                .expect("elevation manifest should validate");
            extent
                .manifest("albedo", COLOUR_BASE_LEVEL, 3, 0.0)
                .write(&directory)
                .expect("colour manifest should validate");
            let _ = std::fs::remove_dir_all(&directory);
        }
    }

    #[test]
    fn blocks_tile_the_extent_without_gaps_or_overlaps() {
        let box_ = LatLonBox::from_corners(corner(49.0, -124.0), corner(49.3, -123.6))
            .expect("failed to build a box");
        let extent = TileExtent::cover(box_).expect("failed to cover");

        for block_tiles in [1, 3, 8] {
            let blocks = extent.blocks(0, block_tiles);
            let (across, down) = extent.tiles(0);
            let total: u32 = blocks.iter().map(|b| b.tiles_across * b.tiles_down).sum();
            assert_eq!(total, across * down, "block size {block_tiles}");

            let first = extent.tile_range(0).0;
            let mut seen = std::collections::HashSet::new();
            for block in &blocks {
                // Every block's grid must agree with its tile indices.
                let offset_x = (block.tile.x - first.x) as u32;
                let offset_y = (block.tile.y - first.y) as u32;
                let span = f64::from(TILE_SIZE) * BASE_METRES_PER_TEXEL;
                assert_eq!(block.grid.west, extent.west + f64::from(offset_x) * span);
                assert_eq!(block.grid.north, extent.north - f64::from(offset_y) * span);

                for y in 0..block.tiles_down {
                    for x in 0..block.tiles_across {
                        let tile = Tile::new(block.tile.x + x as i32, block.tile.y + y as i32);
                        assert!(seen.insert(tile), "{tile:?} covered twice");
                    }
                }
            }
        }
    }

    /// A block's grid has to describe the same ground its tiles do, or the
    /// texels written into a tile would come from somewhere else entirely.
    #[test]
    fn a_blocks_grid_matches_where_its_tiles_sit() {
        let extent = TileExtent::cover(squamish()).expect("failed to cover");
        let grid = extent.tile_grid();
        for block in extent.blocks(0, 4) {
            let (west, north) = grid.tile_origin_metres(0, block.tile);
            assert_eq!(block.grid.west, west);
            assert_eq!(block.grid.north, north);
            assert_eq!(block.grid.width, block.tiles_across * TILE_SIZE);
        }
    }

    #[test]
    fn the_tile_count_matches_what_the_blocks_hold() {
        let extent = TileExtent::cover(squamish()).expect("failed to cover");
        let counted: u32 = extent
            .blocks(0, 8)
            .iter()
            .map(|b| b.tiles_across * b.tiles_down)
            .sum();
        assert_eq!(u64::from(counted), extent.tile_count(0));
    }
}
