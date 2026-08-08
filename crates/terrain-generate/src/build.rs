//! Building the max pyramid the far field is marched through.
//!
//! A depth takes in two things: the depth below it, halved, and the elevation
//! level of its own number, closed. Depth 0 has no depth below and is the
//! closure of the raster alone; every other depth is the greater of the two.
//! See `terrain_tiles::maxima` for why both terms are needed -- that module is
//! where the rules live, and this only blocks the work up.
//!
//! # The canopy is already in the heights
//!
//! It used to enter here. The renderer grew crowns while marching, so a cell had
//! to be raised by the tallest tree that could stand over it or rays would pass
//! straight through the forest -- and nothing would report it, because a pyramid
//! that is too low draws as scattered holes in the far field rather than as an
//! error.
//!
//! `emit::heights` now writes the trees into the elevation itself, so the
//! surface the renderer draws *is* the surface this closes, and the pyramid needs
//! no canopy term at all. The bound is tighter for it: every forest cell used to
//! carry the tallest crown its cover could grow above the ground it bounded,
//! which is up to twenty-eight metres of empty air for a ray to descend into and
//! find nothing in.
//!
//! The work is blocked because closing a square reaches a sample past it. A cell
//! at the east edge of a tile needs a sample from the tile next door, so a tile
//! cannot be built alone; a block of `n x n` tiles reads `n + 1` tiles of
//! elevation across, which at four tiles is a quarter of overhang rather than
//! the whole neighbour a tile-at-a-time pass would re-read. Halving needs no
//! overhang at all, so the depth below is read at exactly four child tiles per
//! parent, which is the floor.
//!
//! # Why this is a copy
//!
//! This is `terrain-process/src/build.rs`, duplicated, for the reason set out
//! in `crate::tiles`: that crate is a binary and cannot be depended on. The
//! recurrence itself is not copied -- `quad_max` and `reduce_max` are called
//! out of `terrain-tiles`, where they are stated once and tested against the
//! definition.
//!
//! It matters here more than it does for a rectangle copy. A max pyramid that
//! is too low does not fail: rays pass through ground the ceiling said was not
//! there, and the far field draws with holes scattered through it. So the two
//! copies must not drift, and both call the same rules to make sure they cannot
//! drift in the way that matters.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use rayon::prelude::*;
use terrain_tiles::maxima::{quad_max, reduce_max};
use terrain_tiles::{Manifest, TILE_SIZE, Tile};

use crate::tiles::{read_rect, write_block};

/// How many tiles square a block covers.
///
/// Four is 2048 cells, so a worker building a depth above zero holds a 4096
/// square of the depth below and three 2048 squares beside it, about 118 MB.
const BLOCK_TILES: u32 = 4;

/// How many blocks to build at once.
///
/// Held down rather than taken from the core count because each worker owns the
/// buffers above. Four is about 400 MB resident, and the pass is bound by
/// reading and writing tiles rather than by taking maxima.
const BLOCK_THREADS: usize = 4;

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

/// Builds the whole max pyramid of `elevation`, which must already be written
/// under `root`, and writes it beside it.
///
/// Returns how many tiles were written. The manifest goes last, as every
/// product's does, so a killed run leaves a directory the renderer refuses
/// rather than one it opens and reads holes out of.
pub fn maxima(root: &Path, elevation: &str, source: &Manifest) -> Result<u64> {
    let name = terrain_tiles::maxima_product(elevation);
    let source_root = root.join(elevation);
    let maxima_root = root.join(&name);
    anyhow::ensure!(
        source_root.is_dir(),
        "{} holds no {elevation} to reduce; generate it first, or drop \
         `--product {name}`",
        root.display()
    );

    // Everything about the ground is the elevation's; only what the values mean
    // differs, and the pyramid is written at exactly the levels the elevation
    // is.
    let manifest = Manifest {
        product: name.clone(),
        ..source.clone()
    };

    let started = std::time::Instant::now();
    let mut written = 0;
    for level in 0..=manifest.max_level() {
        written += build_depth(&source_root, &maxima_root, &manifest, level)?;
    }
    manifest.write(&maxima_root)?;
    log::info!("built {name}: {written} tiles in {:.1?}", started.elapsed());
    Ok(written)
}

/// Builds one depth of `maxima_root`, reading depth `level - 1` of it, or the
/// elevation under `source_root` when `level` is zero.
///
/// Returns how many tiles were written.
fn build_depth(
    source_root: &Path,
    maxima_root: &Path,
    manifest: &Manifest,
    level: u32,
) -> Result<u64> {
    let (first, across, down) = tile_range(manifest, level);
    let blocks: Vec<(u32, u32)> = (0..down.div_ceil(BLOCK_TILES))
        .flat_map(|row| (0..across.div_ceil(BLOCK_TILES)).map(move |column| (column, row)))
        .collect();
    let total = blocks.len() as u64;
    log::info!(
        "depth {level}: {} tiles in {total} blocks of {BLOCK_TILES} x {BLOCK_TILES}",
        across * down
    );

    // A depth reads its child at twice its own indices, which is only the same
    // ground if the two levels' origins line up exactly.
    if level > 0 {
        let (fine, coarse) = (
            manifest.origin_texels(level - 1),
            manifest.origin_texels(level),
        );
        anyhow::ensure!(
            fine == (coarse.0 * 2, coarse.1 * 2),
            "level {level} sits at {coarse:?} but level {} sits at {fine:?}, which is not twice it",
            level - 1
        );
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(BLOCK_THREADS)
        .build()
        .context("building the thread pool")?;

    let done = AtomicU64::new(0);
    let written: u64 = pool
        .install(|| {
            blocks
                .into_par_iter()
                .map(|(column, row)| {
                    let corner = Tile::new(
                        first.x + (column * BLOCK_TILES) as i32,
                        first.y + (row * BLOCK_TILES) as i32,
                    );
                    let built = build_block(
                        source_root,
                        maxima_root,
                        manifest,
                        level,
                        corner,
                        (across - column * BLOCK_TILES).min(BLOCK_TILES) as usize,
                        (down - row * BLOCK_TILES).min(BLOCK_TILES) as usize,
                    );
                    let at = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if at.is_multiple_of(200) || at == total {
                        log::info!("depth {level}: {at} of {total} blocks");
                    }
                    built
                })
                .collect::<Result<Vec<u64>>>()
        })?
        .iter()
        .sum();

    log::info!("depth {level}: wrote {written} of {} tiles", across * down);
    Ok(written)
}

/// Builds the `across x down` tiles of one block.
fn build_block(
    source_root: &Path,
    maxima_root: &Path,
    manifest: &Manifest,
    level: u32,
    corner: Tile,
    across: usize,
    down: usize,
) -> Result<u64> {
    let tile = TILE_SIZE as usize;
    let (width, height) = (across * tile, down * tile);
    // Where this block's first cell sits, in this level's own texel indices.
    let origin = (
        i64::from(corner.x) * i64::from(TILE_SIZE) - manifest.origin_texels(level).0,
        i64::from(corner.y) * i64::from(TILE_SIZE) - manifest.origin_texels(level).1,
    );

    // This level's own samples, closed: the maximum of the four around each
    // cell. Read one wider than the block, because closing reaches a sample past
    // the cell it produces.
    let mut samples = vec![0f32; (width + 1) * (height + 1)];
    read_rect(
        source_root,
        manifest,
        level,
        origin,
        (width + 1, height + 1),
        &mut samples,
    )?;
    let mut cells = vec![0f32; width * height];
    quad_max(&samples, width as u32, height as u32, &mut cells);
    let mut any = samples.iter().any(|value| *value != manifest.nodata);
    drop(samples);

    if level > 0 {
        // ... against every bound the depth below already carries, halved. Two
        // adjacent closed squares share their boundary, so a plain 2x2 maximum
        // covers the whole of the coarse square and needs no overhang.
        let (child_width, child_height) = (2 * width, 2 * height);
        let mut child = vec![0f32; child_width * child_height];
        read_rect(
            maxima_root,
            manifest,
            level - 1,
            (origin.0 * 2, origin.1 * 2),
            (child_width, child_height),
            &mut child,
        )?;
        any |= child.iter().any(|value| *value != manifest.nodata);

        let mut carried = vec![0f32; width * height];
        reduce_max(
            &child,
            child_width as u32,
            child_height as u32,
            &mut carried,
        );
        drop(child);
        for (cell, below) in cells.iter_mut().zip(carried) {
            *cell = cell.max(below);
        }
    }

    if !any {
        return Ok(0);
    }
    write_block(maxima_root, manifest, level, corner, across, down, &cells)
}
