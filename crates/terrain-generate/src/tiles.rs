//! Reading and writing rectangles of a product, a whole tile at a time.
//!
//! Coordinates here are *manifest texels* at a level: column and row counted
//! from the raster's north-west corner. Reads outside the raster, or over tiles
//! that were never written, come back as nodata rather than clamping to the
//! border.
//!
//! # Why this is a copy
//!
//! This is `terrain-process/src/tiles.rs`, duplicated. That crate is a binary,
//! so nothing can depend on it, and the alternative was to lift these two
//! functions into `terrain-tiles` -- which would have meant a rayon dependency
//! in the crate the renderer links, behind a feature flag, for the sake of a
//! hundred lines of block copying.
//!
//! What is *not* duplicated is the part that would be dangerous to duplicate.
//! The rules of the max pyramid live in `terrain_tiles::maxima` and both tools
//! call them there; only this blocking around them is repeated, and it is
//! ordinary rectangle arithmetic that fails loudly rather than subtly.

use std::path::Path;

use anyhow::{Context, Result};
use terrain_tiles::read::read_height_tile;
use terrain_tiles::write::{TilePlacement, write_height_tile};
use terrain_tiles::{Manifest, TILE_SIZE, Tile};

/// Copies `size` cells starting at `origin` into `out`, tightly packed.
pub fn read_rect(
    root: &Path,
    manifest: &Manifest,
    level: u32,
    origin: (i64, i64),
    size: (usize, usize),
    out: &mut [f32],
) -> Result<()> {
    out.fill(manifest.nodata);
    if size.0 == 0 || size.1 == 0 {
        return Ok(());
    }

    let grid = manifest.grid();
    let span = i64::from(TILE_SIZE);
    let (first, _, _) = manifest.tile_of_texel(level, origin.0, origin.1);
    let (last, _, _) = manifest.tile_of_texel(
        level,
        origin.0 + size.0 as i64 - 1,
        origin.1 + size.1 as i64 - 1,
    );
    // Where this level's texel zero sits on the global lattice, so a tile's
    // index can be turned back into the texels it covers.
    let (origin_column, origin_row) = manifest.origin_texels(level);

    for tile_y in first.y..=last.y {
        for tile_x in first.x..=last.x {
            let path = grid.tile_path(root, level, Tile::new(tile_x, tile_y));
            // A tile with nothing under it was never written, and the nodata
            // already in `out` is the answer.
            let Some(values) = read_height_tile(&path)? else {
                continue;
            };

            // The tile's own texels, in the same coordinates as `origin`.
            let west = i64::from(tile_x) * span - origin_column;
            let north = i64::from(tile_y) * span - origin_row;
            let from_x = origin.0.max(west);
            let from_y = origin.1.max(north);
            let to_x = (origin.0 + size.0 as i64).min(west + span);
            let to_y = (origin.1 + size.1 as i64).min(north + span);
            if from_x >= to_x || from_y >= to_y {
                continue;
            }

            let columns = (to_x - from_x) as usize;
            for row in from_y..to_y {
                let source = ((row - north) * span + from_x - west) as usize;
                let target = ((row - origin.1) as usize) * size.0 + (from_x - origin.0) as usize;
                out[target..target + columns].copy_from_slice(&values[source..source + columns]);
            }
        }
    }
    Ok(())
}

/// Cuts a block of cells into tiles and writes the ones holding anything.
///
/// `cells` is `across * down` whole tiles, and `corner` is the north-west one.
/// Returns how many tiles were written; a tile whose ground is entirely
/// unmeasured is left out, so the pyramid stays as sparse as the raster it
/// bounds. A generated landscape has no holes in it, so in practice every tile
/// is written -- but the rule is the format's, not this tool's.
pub fn write_block(
    root: &Path,
    manifest: &Manifest,
    level: u32,
    corner: Tile,
    across: usize,
    down: usize,
    cells: &[f32],
) -> Result<u64> {
    let tile = TILE_SIZE as usize;
    let width = across * tile;
    assert_eq!(cells.len(), width * down * tile);

    let grid = manifest.grid();
    let mut out = vec![0f32; tile * tile];
    let mut written = 0;
    for row in 0..down {
        for column in 0..across {
            for line in 0..tile {
                let from = (row * tile + line) * width + column * tile;
                out[line * tile..(line + 1) * tile].copy_from_slice(&cells[from..from + tile]);
            }
            if out.iter().all(|cell| *cell == manifest.nodata) {
                continue;
            }

            let at = Tile::new(corner.x + column as i32, corner.y + row as i32);
            let (west, north) = grid.tile_origin_metres(level, at);
            write_height_tile(
                &grid.tile_path(root, level, at),
                TilePlacement {
                    west,
                    north,
                    metres_per_texel: grid.metres_per_texel(level),
                },
                &out,
                manifest.nodata,
            )
            .with_context(|| format!("writing level {level} tile {at:?}"))?;
            written += 1;
        }
    }
    Ok(written)
}
