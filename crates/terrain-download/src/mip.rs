//! Building each coarser level of the pyramid from the four tiles beneath it.
//!
//! The base level is written first, block by block, and then this walks up:
//! every level-`L` tile is made from the up-to-four level-`L-1` tiles it covers,
//! read back off disk. Reading them back rather than keeping them in memory is
//! what keeps the peak cost at five tiles however large the download is, and the
//! tiles were just written so the page cache usually still has them.
//!
//! Two rules make holes behave. A missing or nodata child is left out of the
//! average rather than being counted as zero, so a coarse texel over the edge of
//! a survey is the mean of the real data under it. And a coarse texel with no
//! valid children at all stays nodata, so a hole never fills itself in as the
//! levels coarsen -- nor does it spread.

use std::path::Path;

use anyhow::{Context, Result};
use terrain_tiles::{Srgb8, TILE_SIZE, Texel, Tile, TileGrid};
use tiff::decoder::{Decoder, DecodingResult, Limits};

use crate::extent::TileExtent;
use crate::write::{self, TilePlacement};

/// The four level-`L-1` tiles a level-`L` tile is made of, in reading order.
///
/// A tile covers `[x * span, (x + 1) * span)`, and the finer span is half, so
/// its children are `2x` and `2x + 1` on each axis. Both indices count away
/// from the projection origin in the same direction, so the same doubling
/// works for rows as for columns.
fn children(tile: Tile) -> [Tile; 4] {
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
fn read_tile(path: &Path, bands: usize) -> Result<Option<DecodingResult>> {
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
    anyhow::ensure!(
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
        other => anyhow::bail!(
            "{} holds an unexpected sample type {other:?}",
            path.display()
        ),
    };
    anyhow::ensure!(
        got == expected,
        "{} decoded to {got} samples, expected {expected}",
        path.display()
    );
    Ok(Some(image))
}

/// Half a tile: the side of the quadrant one child fills in its parent.
const HALF: usize = (TILE_SIZE / 2) as usize;

/// Builds every level above `base_level`, reading children back off disk.
///
/// Returns how many tiles were written.
pub fn build_levels(
    root: &Path,
    extent: &TileExtent,
    base_level: u32,
    bands: usize,
    nodata: f32,
) -> Result<u64> {
    let grid = extent.tile_grid();
    let mut written = 0;

    for level in (base_level + 1)..=extent.max_level {
        let (first, across, down) = extent.tile_range(level);
        let mut level_written = 0;

        for row in 0..down {
            for column in 0..across {
                let tile = Tile::new(first.x + column as i32, first.y + row as i32);
                if reduce_tile(root, &grid, level, tile, bands, nodata)? {
                    level_written += 1;
                }
            }
        }

        log::info!(
            "level {level}: wrote {level_written} of {} tiles",
            across * down
        );
        written += level_written;
    }

    Ok(written)
}

/// Builds one coarse tile. Returns whether it held anything worth writing.
fn reduce_tile(
    root: &Path,
    grid: &TileGrid,
    level: u32,
    tile: Tile,
    bands: usize,
    nodata: f32,
) -> Result<bool> {
    let children = children(tile);
    let mut loaded = Vec::with_capacity(4);
    for child in children {
        let path = grid.tile_path(root, level - 1, child);
        loaded.push(read_tile(&path, bands)?);
    }
    if loaded.iter().all(Option::is_none) {
        return Ok(false);
    }

    let placement = {
        let (west, north) = grid.tile_origin_metres(level, tile);
        TilePlacement {
            west,
            north,
            metres_per_texel: grid.metres_per_texel(level),
        }
    };
    let path = grid.tile_path(root, level, tile);

    if bands == 1 {
        let mut parent = vec![nodata; (TILE_SIZE as usize).pow(2)];
        let mut any = false;
        for (index, child) in loaded.iter().enumerate() {
            let Some(DecodingResult::F32(values)) = child else {
                continue;
            };
            any |= reduce_quadrant(&mut parent, index, nodata, f32::box_filter, |x, y| {
                let value = values[y * (TILE_SIZE as usize) + x];
                (value != nodata).then_some(value)
            });
        }
        if !any {
            return Ok(false);
        }
        write::write_height_tile(&path, placement, &parent, nodata)?;
    } else {
        let mut parent = vec![0u8; (TILE_SIZE as usize).pow(2) * 3];
        let mut any = false;
        for (index, child) in loaded.iter().enumerate() {
            let Some(DecodingResult::U8(values)) = child else {
                continue;
            };
            any |= reduce_colour_quadrant(&mut parent, index, values);
        }
        if !any {
            return Ok(false);
        }
        write::write_colour_tile(&path, placement, &parent)?;
    }

    Ok(true)
}

/// Fills one quadrant of a single-band parent from one child.
///
/// `sample` returns the child's value at a texel, or `None` where it is nodata.
/// Returns whether anything was written.
fn reduce_quadrant(
    parent: &mut [f32],
    quadrant: usize,
    nodata: f32,
    filter: impl Fn(&[f32]) -> f32,
    sample: impl Fn(usize, usize) -> Option<f32>,
) -> bool {
    let (offset_x, offset_y) = ((quadrant % 2) * HALF, (quadrant / 2) * HALF);
    let mut samples = Vec::with_capacity(4);
    let mut any = false;

    for y in 0..HALF {
        for x in 0..HALF {
            samples.clear();
            for dy in 0..2 {
                for dx in 0..2 {
                    if let Some(value) = sample(x * 2 + dx, y * 2 + dy) {
                        samples.push(value);
                    }
                }
            }
            let at = (offset_y + y) * (TILE_SIZE as usize) + offset_x + x;
            parent[at] = if samples.is_empty() {
                nodata
            } else {
                any = true;
                filter(&samples)
            };
        }
    }
    any
}

/// Fills one quadrant of an RGB parent from one child.
///
/// Black is the mosaics' own nodata, so a black child texel is left out of the
/// average and a quadrant of pure black stays black. Averaging goes through
/// [`Srgb8`], which decodes the curve before taking the mean -- averaging the
/// encoded bytes instead would darken every mip.
fn reduce_colour_quadrant(parent: &mut [u8], quadrant: usize, values: &[u8]) -> bool {
    let (offset_x, offset_y) = ((quadrant % 2) * HALF, (quadrant / 2) * HALF);
    let mut samples = Vec::with_capacity(4);
    let mut any = false;

    for y in 0..HALF {
        for x in 0..HALF {
            samples.clear();
            for dy in 0..2 {
                for dx in 0..2 {
                    let at = ((y * 2 + dy) * (TILE_SIZE as usize) + x * 2 + dx) * 3;
                    let rgb = &values[at..at + 3];
                    if rgb != [0, 0, 0] {
                        samples.push(Srgb8([rgb[0], rgb[1], rgb[2], 255]));
                    }
                }
            }
            let at = ((offset_y + y) * (TILE_SIZE as usize) + offset_x + x) * 3;
            if samples.is_empty() {
                parent[at..at + 3].copy_from_slice(&[0, 0, 0]);
            } else {
                any = true;
                let mixed = Srgb8::box_filter(&samples);
                parent[at..at + 3].copy_from_slice(&mixed.0[..3]);
            }
        }
    }
    any
}

#[cfg(test)]
mod tests {
    use super::*;

    const NODATA: f32 = -32767.0;

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

    /// Four known values must come out as their mean, in the right quadrant.
    #[test]
    fn a_quadrant_reduces_to_the_mean_of_each_group_of_four() {
        let mut parent = vec![NODATA; (TILE_SIZE as usize).pow(2)];
        let child: Vec<f32> = (0..(TILE_SIZE as usize).pow(2))
            .map(|i| (i % (TILE_SIZE as usize)) as f32)
            .collect();

        // Quadrant 3 is the south-east one.
        let any = reduce_quadrant(&mut parent, 3, NODATA, f32::box_filter, |x, y| {
            Some(child[y * (TILE_SIZE as usize) + x])
        });
        assert!(any);

        // Columns 0 and 1 of the child average to 0.5, and land at the parent's
        // first column inside that quadrant.
        let at = HALF * (TILE_SIZE as usize) + HALF;
        assert_eq!(parent[at], 0.5);
        assert_eq!(parent[at + 1], 2.5);
        // Nothing outside the quadrant was touched.
        assert_eq!(parent[0], NODATA);
    }

    #[test]
    fn nodata_children_are_left_out_of_the_average() {
        let mut parent = vec![NODATA; (TILE_SIZE as usize).pow(2)];
        // Only the very first texel of the group of four is real.
        let any = reduce_quadrant(&mut parent, 0, NODATA, f32::box_filter, |x, y| {
            (x == 0 && y == 0).then_some(10.0)
        });
        assert!(any);
        assert_eq!(
            parent[0], 10.0,
            "should be the one real value, not a quarter"
        );
        assert_eq!(parent[1], NODATA, "nothing under it at all");
    }

    #[test]
    fn a_quadrant_with_nothing_under_it_writes_nothing() {
        let mut parent = vec![NODATA; (TILE_SIZE as usize).pow(2)];
        let any = reduce_quadrant(&mut parent, 0, NODATA, f32::box_filter, |_, _| None);
        assert!(!any);
        assert!(parent.iter().all(|&v| v == NODATA));
    }

    /// Colour has to average the light, not the bytes. Half black and half
    /// white encodes well above the midpoint.
    #[test]
    fn colour_quadrants_average_in_linear_light() {
        let mut parent = vec![0u8; (TILE_SIZE as usize).pow(2) * 3];
        let mut child = vec![0u8; (TILE_SIZE as usize).pow(2) * 3];
        // Two of every four texels white, the other two a dark grey that is not
        // black, so nothing is treated as nodata.
        for i in 0..(TILE_SIZE as usize).pow(2) {
            let value = if (i / (TILE_SIZE as usize)).is_multiple_of(2) {
                255
            } else {
                1
            };
            child[i * 3..i * 3 + 3].copy_from_slice(&[value; 3]);
        }

        assert!(reduce_colour_quadrant(&mut parent, 0, &child));
        assert!(
            parent[0] > 180,
            "half the light should encode above the midpoint, got {}",
            parent[0]
        );
    }

    #[test]
    fn black_colour_texels_are_treated_as_nodata() {
        let mut parent = vec![9u8; (TILE_SIZE as usize).pow(2) * 3];
        let child = vec![0u8; (TILE_SIZE as usize).pow(2) * 3];
        assert!(!reduce_colour_quadrant(&mut parent, 0, &child));
        assert_eq!(&parent[0..3], &[0, 0, 0], "should be cleared to nodata");
    }
}
