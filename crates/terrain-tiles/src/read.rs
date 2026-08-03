//! Reading a written tile back off disk.
//!
//! Both tools that build coarse levels work the same way: write the base level,
//! then walk up reading children back rather than keeping them in memory. That
//! keeps the peak cost at a handful of tiles however large the box is, and the
//! tiles were just written so the page cache usually still has them.
//!
//! The renderer does not go through here. It wants thin strips out of tiles it
//! never holds whole, which is what [`crate::write`]'s one-row-per-strip layout
//! is for; see `src/terrain/tiles.rs`. This is the whole-tile path the writers
//! need.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use tiff::decoder::{Decoder, DecodingResult, Limits};

use crate::{TILE_SIZE, Tile};

/// The four level-`L-1` tiles a level-`L` tile is made of, in reading order.
///
/// A tile covers `[x * span, (x + 1) * span)`, and the finer span is half, so
/// its children are `2x` and `2x + 1` on each axis. Both indices count away
/// from the projection origin in the same direction, so the same doubling
/// works for rows as for columns.
pub fn children(tile: Tile) -> [Tile; 4] {
    [
        Tile::new(tile.x * 2, tile.y * 2),
        Tile::new(tile.x * 2 + 1, tile.y * 2),
        Tile::new(tile.x * 2, tile.y * 2 + 1),
        Tile::new(tile.x * 2 + 1, tile.y * 2 + 1),
    ]
}

/// Reads a tile back, or `None` if it was never written.
///
/// An absent file is the ordinary case, not an error: tiles with nothing under
/// them are skipped, which is how a box over patchy coverage stays small.
pub fn read_tile(path: &Path, bands: usize) -> Result<Option<DecodingResult>> {
    if !path.exists() {
        return Ok(None);
    }
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("reading the header of {}", path.display()))?
        .with_limits(Limits::unlimited());

    let (width, height) = decoder
        .dimensions()
        .with_context(|| format!("reading the size of {}", path.display()))?;
    ensure!(
        width == TILE_SIZE && height == TILE_SIZE,
        "{} is {width} x {height}, not a {TILE_SIZE} x {TILE_SIZE} tile",
        path.display()
    );

    let image = decoder
        .read_image()
        .with_context(|| format!("decoding {}", path.display()))?;
    let expected = (TILE_SIZE as usize).pow(2) * bands;
    let got = match &image {
        DecodingResult::F32(values) => values.len(),
        DecodingResult::U8(values) => values.len(),
        DecodingResult::U16(values) => values.len(),
        DecodingResult::U32(values) => values.len(),
        other => bail!(
            "{} holds an unexpected sample type {other:?}",
            path.display()
        ),
    };
    ensure!(
        got == expected,
        "{} decoded to {got} samples, expected {expected}",
        path.display()
    );
    Ok(Some(image))
}

/// Reads a single-band tile back as floats, or `None` if it was never written.
pub fn read_height_tile(path: &Path) -> Result<Option<Vec<f32>>> {
    match read_tile(path, 1)? {
        None => Ok(None),
        Some(DecodingResult::F32(values)) => Ok(Some(values)),
        Some(_) => bail!(
            "{} holds something other than 32-bit floats",
            path.display()
        ),
    }
}

/// Reads a single-band tile back as material ids, or `None` if it was never
/// written.
pub fn read_material_tile(path: &Path) -> Result<Option<Vec<u32>>> {
    match read_tile(path, 1)? {
        None => Ok(None),
        Some(DecodingResult::U32(values)) => Ok(Some(values)),
        Some(_) => bail!(
            "{} holds something other than 32-bit unsigned ids",
            path.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tiles_children_are_the_four_beneath_it() {
        assert_eq!(
            children(Tile::new(3, -5)),
            [
                Tile::new(6, -10),
                Tile::new(7, -10),
                Tile::new(6, -9),
                Tile::new(7, -9),
            ]
        );
        // The doubling has to keep working through zero, where a truncating
        // divide would fold two tiles into one on the way back down.
        assert_eq!(children(Tile::new(-1, -1))[0], Tile::new(-2, -2));
    }

    #[test]
    fn a_tile_that_was_never_written_reads_as_absent() {
        let path = std::env::temp_dir().join("terrain-tiles-absent.tif");
        assert!(
            read_tile(&path, 1)
                .expect("absence is not an error")
                .is_none()
        );
    }
}
