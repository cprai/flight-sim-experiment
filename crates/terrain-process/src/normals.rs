//! Building a pyramid of surface normals from the finest elevation there is.
//!
//! The renderer shades by the normal of the ground a ray met, and a normal is a
//! derivative. Taking it from whichever level the march happened to stop at
//! would flatten the far field, because a coarse height texel is already a
//! smoothed surface and its slopes are the slopes of that smoothing. So every
//! normal here is computed at level 0 and averaged down, the way a normal map
//! keeps its detail through minification: the mean of the sixty-four one-metre
//! normals under an eight-metre texel still carries the roughness of what is
//! under it, where the normal of the averaged heights would not.
//!
//! Nothing finer than the pyramid's base level is written. The fine levels
//! exist only as the terms of that mean, and storing them would cost
//! sixty-four times the disk to describe lighting nothing gets close enough to
//! see. Averaging straight from level 0 to the base, rather than through the
//! levels between, is not only cheaper: every stored normal is renormalised to
//! unit length, so folding in steps would let a quadrant whose normals nearly
//! cancel come back at full strength and outvote one whose normals agree, and
//! would over-weight a lone real normal sitting in a quadrant of holes. One
//! mean over every valid level-0 normal weights each by the ground it covers,
//! which is what an area average should do. Above the base the levels are
//! folded in steps regardless -- there is nothing else left to fold -- and the
//! same caveat applies to them.
//!
//! Tile edges are where this is easy to get wrong. A central difference at the
//! east edge of a tile needs the height in the tile next door, so the base pass
//! reads a one-texel halo on all four sides and lets [`read_rect`] span
//! whatever tiles that touches. Past the raster the halo comes back nodata
//! rather than clamped to the border, and the difference falls back to
//! one-sided, so the last row of a survey slopes the way its two real samples
//! do rather than the way a repeated edge would.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use rayon::prelude::*;
use terrain_tiles::read::{children, read_normal_tile};
use terrain_tiles::write::{TilePlacement, write_normal_tile};
use terrain_tiles::{Manifest, NODATA_BELOW, Normal, TILE_SIZE, Texel, Tile, TileGrid};

use crate::build::tile_range;
use crate::tiles::read_rect;

/// How many tiles to build at once.
///
/// One output tile at the base level covers `TILE_SIZE * 2^base` level-0 texels
/// on a side -- 4096 at the level this ships at -- so a worker holds a 4098
/// square of heights, about 67 MB, plus a running sum for the tile it is
/// filling. Four workers is under 300 MB, a little less than the max pyramid's
/// own pass, and the work is bound by reading tiles rather than by the
/// arithmetic over them.
const NORMAL_THREADS: usize = 4;

/// Half a tile: the side of the quadrant one child fills in its parent.
const HALF: usize = (TILE_SIZE / 2) as usize;

/// Builds every level of `normal_root`, from `manifest.base_level` up.
///
/// `source` is the elevation the normals are derived from, read at level 0 out
/// of `source_root`; `manifest` describes the pyramid being written. Returns
/// how many tiles were written.
pub fn build(
    source_root: &Path,
    normal_root: &Path,
    source: &Manifest,
    manifest: &Manifest,
) -> Result<u64> {
    let base = manifest.base_level;

    // The base level addresses its heights by multiplying its own texel indices
    // out to level 0, which is only the same ground if the two origins line up
    // exactly. The download's snapping guarantees it; a raster that broke it
    // would shift every normal by a texel and nothing else would say so.
    let (fine, coarse) = (source.origin_texels(0), manifest.origin_texels(base));
    let scale = 1i64 << base;
    anyhow::ensure!(
        fine == (coarse.0 * scale, coarse.1 * scale),
        "level {base} sits at {coarse:?}, which is not level 0's {fine:?} divided by {scale}"
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(NORMAL_THREADS)
        .build()
        .context("building the thread pool")?;

    let mut written = pool.install(|| build_base(source_root, normal_root, source, manifest))?;
    for level in (base + 1)..=manifest.max_level() {
        written += pool.install(|| reduce_level(normal_root, manifest, level))?;
    }
    Ok(written)
}

/// Builds the finest stored level, one tile per work item, from level-0 heights.
fn build_base(
    source_root: &Path,
    normal_root: &Path,
    source: &Manifest,
    manifest: &Manifest,
) -> Result<u64> {
    let level = manifest.base_level;
    let (first, across, down) = tile_range(manifest, level);
    let tiles: Vec<Tile> = (0..down)
        .flat_map(|row| {
            (0..across).map(move |column| Tile::new(first.x + column as i32, first.y + row as i32))
        })
        .collect();
    let total = tiles.len() as u64;
    log::info!(
        "level {level}: {total} tiles, each from {} level-0 texels square",
        (TILE_SIZE as usize) << level
    );

    let done = AtomicU64::new(0);
    let written: u64 = tiles
        .into_par_iter()
        .map(|tile| {
            let built = build_tile(source_root, normal_root, source, manifest, tile);
            let at = done.fetch_add(1, Ordering::Relaxed) + 1;
            if at.is_multiple_of(100) || at == total {
                log::info!("level {level}: {at} of {total} tiles");
            }
            built
        })
        .collect::<Result<Vec<u64>>>()?
        .iter()
        .sum();

    log::info!("level {level}: wrote {written} of {total} tiles");
    Ok(written)
}

/// Builds one base-level tile: the mean of the level-0 normals under each texel.
fn build_tile(
    source_root: &Path,
    normal_root: &Path,
    source: &Manifest,
    manifest: &Manifest,
    tile: Tile,
) -> Result<u64> {
    let out_side = TILE_SIZE as usize;
    let scale = 1usize << manifest.base_level;
    let side = out_side * scale;

    // The tile's north-west texel, in level-0 texels counted from the raster's
    // own corner, and then one further out for the halo the differences read.
    let (origin_column, origin_row) = manifest.origin_texels(manifest.base_level);
    let origin = (
        (i64::from(tile.x) * i64::from(TILE_SIZE) - origin_column) * scale as i64 - 1,
        (i64::from(tile.y) * i64::from(TILE_SIZE) - origin_row) * scale as i64 - 1,
    );
    let span = side + 2;
    let mut heights = vec![0f32; span * span];
    read_rect(source_root, source, 0, origin, (span, span), &mut heights)?;

    let metres = source.metres_per_texel(0) as f32;
    // Sums of unit normals, and how many went into each, per output texel.
    let mut sums = vec![[0f32; 3]; out_side * out_side];
    let mut counts = vec![0u32; out_side * out_side];

    for y in 0..side {
        for x in 0..side {
            let at = |dx: usize, dy: usize| heights[(y + dy) * span + x + dx];
            let here = at(1, 1);
            if here < NODATA_BELOW {
                continue;
            }
            // A central difference where both neighbours are ground, one-sided
            // where only one is, and flat where the texel stands alone.
            let slope = |low: f32, high: f32| match (low < NODATA_BELOW, high < NODATA_BELOW) {
                (false, false) => (high - low) / (2.0 * metres),
                (false, true) => (here - low) / metres,
                (true, false) => (high - here) / metres,
                (true, true) => 0.0,
            };
            let east = slope(at(0, 1), at(2, 1));
            let south = slope(at(1, 0), at(1, 2));

            // The normal of a height field, before it is scaled to unit length.
            let inverse = 1.0 / (east * east + 1.0 + south * south).sqrt();
            let index = (y / scale) * out_side + x / scale;
            let sum = &mut sums[index];
            sum[0] += -east * inverse;
            sum[1] += inverse;
            sum[2] += -south * inverse;
            counts[index] += 1;
        }
    }
    drop(heights);

    let mut out = vec![Normal::NODATA; out_side * out_side];
    let mut any = false;
    for ((texel, sum), count) in out.iter_mut().zip(&sums).zip(&counts) {
        if *count == 0 {
            continue;
        }
        any = true;
        *texel = Normal::from_unit(sum[0], sum[1], sum[2]);
    }
    if !any {
        return Ok(0);
    }

    write_tile(
        normal_root,
        &manifest.grid(),
        manifest.base_level,
        tile,
        &out,
    )?;
    Ok(1)
}

/// Builds one level above the base, each tile from the four beneath it.
fn reduce_level(normal_root: &Path, manifest: &Manifest, level: u32) -> Result<u64> {
    let (first, across, down) = tile_range(manifest, level);
    let tiles: Vec<Tile> = (0..down)
        .flat_map(|row| {
            (0..across).map(move |column| Tile::new(first.x + column as i32, first.y + row as i32))
        })
        .collect();
    let total = tiles.len() as u64;

    let written: u64 = tiles
        .into_par_iter()
        .map(|tile| reduce_tile(normal_root, manifest, level, tile))
        .collect::<Result<Vec<u64>>>()?
        .iter()
        .sum();

    log::info!("level {level}: wrote {written} of {total} tiles");
    Ok(written)
}

/// Builds one coarse tile from the up-to-four beneath it.
fn reduce_tile(normal_root: &Path, manifest: &Manifest, level: u32, tile: Tile) -> Result<u64> {
    let grid = manifest.grid();
    let side = TILE_SIZE as usize;
    let mut out = vec![Normal::NODATA; side * side];
    let mut any = false;

    for (quadrant, child) in children(tile).into_iter().enumerate() {
        let path = grid.tile_path(normal_root, level - 1, child);
        let Some(values) = read_normal_tile(&path)? else {
            continue;
        };
        let (offset_x, offset_y) = ((quadrant % 2) * HALF, (quadrant / 2) * HALF);
        let mut samples = Vec::with_capacity(4);

        for y in 0..HALF {
            for x in 0..HALF {
                samples.clear();
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sample = values[(y * 2 + dy) * side + x * 2 + dx];
                        // A hole is left out of the mean rather than averaged
                        // in as a direction, and four holes stay a hole.
                        if !sample.is_nodata() {
                            samples.push(sample);
                        }
                    }
                }
                if !samples.is_empty() {
                    any = true;
                    out[(offset_y + y) * side + offset_x + x] = Normal::box_filter(&samples);
                }
            }
        }
    }

    if !any {
        return Ok(0);
    }
    write_tile(normal_root, &grid, level, tile, &out)?;
    Ok(1)
}

/// Writes one tile, placed where the grid says it belongs.
fn write_tile(
    normal_root: &Path,
    grid: &TileGrid,
    level: u32,
    tile: Tile,
    out: &[Normal],
) -> Result<()> {
    let (west, north) = grid.tile_origin_metres(level, tile);
    write_normal_tile(
        &grid.tile_path(normal_root, level, tile),
        TilePlacement {
            west,
            north,
            metres_per_texel: grid.metres_per_texel(level),
        },
        out,
    )
    .with_context(|| format!("writing level {level} tile {tile:?}"))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use terrain_tiles::write::write_height_tile;

    use super::*;

    /// Wide enough for two tiles at the base level, so there is a seam to test
    /// across, and one tile tall.
    const WIDE: u32 = 4096;
    const TALL: u32 = 1024;
    const NODATA: f32 = -32767.0;

    /// Level 2 rather than the 3 this ships at, so a fixture is sixteen level-0
    /// texels per stored one instead of sixty-four: enough averaging to be a
    /// real test, few enough tiles to be a fast one.
    const BASE: u32 = 2;

    fn source() -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: "dtm".into(),
            epsg: 3979,
            tile_size: TILE_SIZE,
            base_level: 0,
            level_count: BASE + 2,
            base_metres_per_texel: 1.0,
            origin_metres: [-1_974_272.0, 524_288.0],
            extent_texels: [WIDE, TALL],
            bands: 1,
            nodata: NODATA,
        }
    }

    fn normals(source: &Manifest) -> Manifest {
        Manifest {
            product: "dtm-normal".into(),
            base_level: BASE,
            level_count: source.max_level() - BASE + 1,
            nodata: f32::from(Normal::NODATA.to_sample()),
            ..source.clone()
        }
    }

    /// The whole raster, as one array, which is also the oracle every test
    /// compares against.
    fn raster(height: impl Fn(u32, u32) -> f32) -> Vec<f32> {
        (0..TALL)
            .flat_map(|y| (0..WIDE).map(move |x| (x, y)))
            .map(|(x, y)| height(x, y))
            .collect()
    }

    /// Writes the level-0 elevation tiles the pass reads.
    fn write_source(root: &Path, manifest: &Manifest, raster: &[f32]) {
        let grid = manifest.grid();
        let (first, across, down) = tile_range(manifest, 0);
        let side = TILE_SIZE as usize;
        for row in 0..down {
            for column in 0..across {
                let tile = Tile::new(first.x + column as i32, first.y + row as i32);
                let mut out = vec![NODATA; side * side];
                for y in 0..side {
                    for x in 0..side {
                        let (gx, gy) = (column as usize * side + x, row as usize * side + y);
                        out[y * side + x] = raster[gy * WIDE as usize + gx];
                    }
                }
                let (west, north) = grid.tile_origin_metres(0, tile);
                write_height_tile(
                    &grid.tile_path(root, 0, tile),
                    TilePlacement {
                        west,
                        north,
                        metres_per_texel: 1.0,
                    },
                    &out,
                    NODATA,
                )
                .expect("failed to write a source tile");
            }
        }
    }

    /// `name` must differ per test: these run in parallel, and two sharing a
    /// path would race to write and delete the same files.
    fn run(name: &str, height: impl Fn(u32, u32) -> f32) -> (PathBuf, Manifest, Vec<f32>) {
        let root = std::env::temp_dir().join(format!(
            "terrain-process-normals-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        let source = source();
        let manifest = normals(&source);
        let raster = raster(height);
        write_source(&root.join("dtm"), &source, &raster);
        build(
            &root.join("dtm"),
            &root.join("dtm-normal"),
            &source,
            &manifest,
        )
        .expect("failed to build");
        (root, manifest, raster)
    }

    /// Reads one stored level back as a single array, nodata where a tile was
    /// never written.
    fn read_level(root: &Path, manifest: &Manifest, level: u32) -> (Vec<Normal>, usize, usize) {
        let grid = manifest.grid();
        let (first, across, down) = tile_range(manifest, level);
        let side = TILE_SIZE as usize;
        let (width, height) = (across as usize * side, down as usize * side);
        let mut out = vec![Normal::NODATA; width * height];
        for row in 0..down as usize {
            for column in 0..across as usize {
                let tile = Tile::new(first.x + column as i32, first.y + row as i32);
                let Some(values) = read_normal_tile(&grid.tile_path(root, level, tile))
                    .expect("failed to read a tile")
                else {
                    continue;
                };
                for y in 0..side {
                    let from = y * side;
                    let to = (row * side + y) * width + column * side;
                    out[to..to + side].copy_from_slice(&values[from..from + side]);
                }
            }
        }
        (out, width, height)
    }

    /// The normal a plane has everywhere, computed the way the pass does.
    fn plane(east: f32, south: f32) -> Normal {
        let inverse = 1.0 / (east * east + 1.0 + south * south).sqrt();
        Normal::from_unit(-east * inverse, inverse, -south * inverse)
    }

    /// A tilted plane has one normal, and every texel of every level must carry
    /// it: any sign error between a raster row and world +Z, or between a
    /// column and +X, shows up here and nowhere else.
    #[test]
    fn a_tilted_plane_carries_its_own_normal_at_every_level() {
        let (east, south) = (0.25f32, -0.5f32);
        let (root, manifest, _) = run("plane", |x, y| east * x as f32 + south * y as f32);
        let expected = plane(east, south);

        for level in manifest.base_level..=manifest.max_level() {
            let (values, width, _) = read_level(&root.join("dtm-normal"), &manifest, level);
            // A coarse tile hangs off the edge of the raster, and the part of
            // it with nothing behind it is a hole rather than a slope.
            let (across, down) = manifest.size_texels(level);
            let mut checked = 0;
            for y in 0..down as usize {
                for x in 0..across as usize {
                    // The raster's own edge differences one-sidedly rather than
                    // centrally, which is still exact for a plane.
                    assert_eq!(
                        values[y * width + x],
                        expected,
                        "level {level} texel ({x}, {y})"
                    );
                    checked += 1;
                }
            }
            assert!(
                checked > 1000,
                "level {level} only checked {checked} texels"
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The cross-tile case. A normal at a tile's edge has to be the one a
    /// seamless raster would give, which means differencing against the tile
    /// next door rather than against a repeat of the edge texel.
    #[test]
    fn a_normal_at_a_tile_seam_matches_a_seamless_raster() {
        // Rolling ground, so neighbouring texels genuinely differ and a
        // clamped edge would be visibly wrong.
        let height = |x: u32, y: u32| {
            let (x, y) = (x as f32, y as f32);
            40.0 * (x * 0.011).sin() + 25.0 * (y * 0.017).cos() + 0.03 * x
        };
        let (root, manifest, raster) = run("seam", height);
        let (values, width, _) = read_level(&root.join("dtm-normal"), &manifest, BASE);

        let scale = 1usize << BASE;
        let wide = WIDE as usize;
        let tall = TALL as usize;
        // The same mean, taken off the seamless array rather than off tiles.
        let oracle = |column: usize, row: usize| {
            let mut sum = [0f32; 3];
            for y in row * scale..(row + 1) * scale {
                for x in column * scale..(column + 1) * scale {
                    let at = |x: usize, y: usize| raster[y * wide + x];
                    let east = if x == 0 {
                        at(1, y) - at(0, y)
                    } else if x + 1 == wide {
                        at(x, y) - at(x - 1, y)
                    } else {
                        (at(x + 1, y) - at(x - 1, y)) / 2.0
                    };
                    let south = if y == 0 {
                        at(x, 1) - at(x, 0)
                    } else if y + 1 == tall {
                        at(x, y) - at(x, y - 1)
                    } else {
                        (at(x, y + 1) - at(x, y - 1)) / 2.0
                    };
                    let inverse = 1.0 / (east * east + 1.0 + south * south).sqrt();
                    sum[0] += -east * inverse;
                    sum[1] += inverse;
                    sum[2] += -south * inverse;
                }
            }
            Normal::from_unit(sum[0], sum[1], sum[2])
        };

        // The seam between the two base-level tiles, and its neighbourhood.
        let seam = TILE_SIZE as usize;
        let mut checked = 0;
        let mut differing = 0;
        for row in (0..TALL as usize / scale).step_by(7) {
            for column in seam - 3..seam + 3 {
                let stored = values[row * width + column];
                assert_eq!(stored, oracle(column, row), "texel ({column}, {row})");
                if column > seam - 3 && stored != values[row * width + column - 1] {
                    differing += 1;
                }
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} texels were checked");
        // A flat fixture would pass the equality above without proving
        // anything, so the ground either side of the seam has to actually
        // differ from texel to texel.
        assert!(
            differing > checked / 4,
            "only {differing} of {checked} texels differ from their neighbour"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Ground with nothing measured under it stays a hole all the way up, and
    /// the ground beside it is still shaded by what is actually there.
    #[test]
    fn a_hole_stays_a_hole_and_leaves_its_neighbours_alone() {
        let scale = 1usize << BASE;
        // A hole covering whole stored texels, so there is no partial texel to
        // reason about, plus a plane everywhere else. Every other column of
        // them, over the southern quarter of the raster.
        let (east, south) = (0.125f32, 0.25f32);
        let rows = 200..(TALL as usize / scale);
        let hole = |x: u32, y: u32| {
            (x as usize / scale).is_multiple_of(2) && rows.contains(&(y as usize / scale))
        };
        let (root, manifest, _) = run("hole", |x, y| {
            if hole(x, y) {
                NODATA
            } else {
                east * x as f32 + south * y as f32
            }
        });

        let (values, width, _) = read_level(&root.join("dtm-normal"), &manifest, BASE);
        let expected = plane(east, south);
        let across = manifest.size_texels(BASE).0 as usize;
        let mut holes = 0;
        for row in rows.clone() {
            for column in 0..across {
                let value = values[row * width + column];
                if column.is_multiple_of(2) {
                    assert!(value.is_nodata(), "texel ({column}, {row}) filled a hole");
                    holes += 1;
                } else {
                    // Its east and west neighbours are gone, so the horizontal
                    // slope is whatever one side gives; the southward one is
                    // untouched, and neither may be wild.
                    assert_eq!(
                        value.south, expected.south,
                        "texel ({column}, {row}) leans the wrong way north to south"
                    );
                }
            }
        }
        assert!(holes > 100, "only {holes} holes were checked");

        // A coarse texel spans one hole column and one real one, so it takes
        // the mean of what is actually there rather than becoming a hole.
        let (coarse, coarse_width, _) = read_level(&root.join("dtm-normal"), &manifest, BASE + 1);
        let coarse_row = rows.start / 2 + 4;
        assert!(
            coarse[coarse_row * coarse_width..coarse_row * coarse_width + 8]
                .iter()
                .all(|value| !value.is_nodata()),
            "a coarse texel with real ground under half of it should not be a hole"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The point of the whole design: the fine levels are computed and thrown
    /// away, so they must not be on disk.
    #[test]
    fn no_level_finer_than_the_base_is_written() {
        let (root, manifest, _) = run("levels", |x, _| x as f32 * 0.01);
        let normal_root = root.join("dtm-normal");
        for level in 0..manifest.base_level {
            let directory = normal_root.join(format!("{level:02}"));
            assert!(
                !directory.exists(),
                "level {level} was written to {}",
                directory.display()
            );
        }
        for level in manifest.base_level..=manifest.max_level() {
            assert!(
                normal_root.join(format!("{level:02}")).exists(),
                "level {level} is missing"
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Ground with nothing under it at all writes nothing, so the pyramid
    /// stays as sparse as the raster.
    #[test]
    fn a_tile_with_nothing_under_it_is_never_written() {
        let (root, manifest, _) = run("empty", |_, _| NODATA);
        let normal_root = root.join("dtm-normal");
        for level in manifest.base_level..=manifest.max_level() {
            let directory = normal_root.join(format!("{level:02}"));
            assert!(
                !directory.exists() || directory.read_dir().expect("unreadable").count() == 0,
                "level {level} wrote a tile over ground with no measurements"
            );
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
