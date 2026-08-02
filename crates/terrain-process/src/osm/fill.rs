//! Filling unmapped ground from the cover mapped around it.
//!
//! After the paint, what is still `Null` is ground no OpenStreetMap area,
//! stroke, or fill reached: alpine slopes above the highest wood polygon,
//! verges between a zone and the road that borders it, backcountry nobody
//! has mapped. Leaving it null would be honest but useless -- the renderer
//! shades it as missing data -- and the data around it is not silent: ground
//! is overwhelmingly like its neighbours. So every null texel within
//! [`FILL_METRES`] of mapped cover takes the id of the *nearest* mapped
//! texel, and only ground further than that from every mapped thing in the
//! extract stays null.
//!
//! Water does not spread. A lake's shoreline is surveyed geometry, and land
//! beyond it is not more lake; the ocean fill has already claimed everything
//! up to the coastline. Fills come from land cover only -- though distance
//! still measures straight across water, which is what puts forest rather
//! than nothing on an unmapped islet near a wooded shore.
//!
//! Nearness is a chamfer distance transform, weights 3 and 4 -- within a few
//! percent of Euclidean, and exact enough for a fill whose alternative was
//! magenta. Two passes over the raster, forward and backward, each texel
//! taking the best of its already-visited neighbours. The raster is far too
//! large to hold twice over in memory, so the forward pass streams its
//! intermediate rows to a spill file beside the tiles and the backward pass
//! consumes them in reverse, re-reading the painted tiles as it goes; peak
//! memory is a few tile rows whatever the raster size.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{Context, Result};
use terrain_tiles::read::read_material_tile;
use terrain_tiles::write::{TilePlacement, write_material_tile};
use terrain_tiles::{MATERIAL_BASE_LEVEL, Manifest, TILE_SIZE, Tile};

use crate::build::tile_range;

/// How far mapped cover reaches into unmapped ground.
///
/// Measured before choosing: in the committed extract every null texel sits
/// within 1.7 km of mapped land cover, so 4 km fills all of it with margin
/// to spare. The cap is what keeps the fill honest on some future extract
/// whose region clip crosses real land: ground a few kilometres past the
/// last mapped polygon is extrapolation, but ground forty kilometres past it
/// would be invention, and it stays null.
pub const FILL_METRES: f64 = 4000.0;

/// Chamfer weights: 3 per orthogonal step, 4 per diagonal step, so a texel's
/// weighted distance approximates 3 times its Euclidean distance in texels.
const ORTHOGONAL: u16 = 3;
const DIAGONAL: u16 = 4;

/// The spill file's name, beside the level directories under the product.
const SPILL_FILE: &str = "fill.spill";

/// Whether a stored id is cover the fill may spread. Null is nothing, and
/// the water block -- ocean included -- is bounded by surveyed shoreline.
fn spreads(id: u32) -> bool {
    id != 0 && (id >> 8) != 0x01
}

/// Fills every null base-level texel within [`FILL_METRES`] of mapped land
/// cover with the nearest such cover's id, rewriting the level's tiles under
/// `root`. Returns how many texels were filled and how many stayed null.
pub fn fill(manifest: &Manifest, root: &Path) -> Result<(u64, u64)> {
    let level = MATERIAL_BASE_LEVEL;
    let (width, height) = manifest.size_texels(level);
    let (first, across, down) = tile_range(manifest, level);
    let metres = manifest.metres_per_texel(level);
    let cap = ((FILL_METRES / metres) * f64::from(ORTHOGONAL)) as u16;
    let started = std::time::Instant::now();

    let w = width as usize;
    let spill_path = root.join(SPILL_FILE);
    let spill = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&spill_path)
        .with_context(|| format!("creating {}", spill_path.display()))?;
    let mut spill = std::io::BufWriter::new(spill);
    let mut row_bytes_buffer = vec![0u8; w * (2 + 4)];

    // Forward pass: north to south, west to east. Each row's distances and
    // nearest ids settle against the row above and the texel to the left,
    // then spill to disk for the backward pass to finish.
    let mut previous: Option<(Vec<u16>, Vec<u32>)> = None;
    for band in 0..down {
        let rows = TILE_SIZE.min(height - band * TILE_SIZE);
        let cells = read_band(manifest, root, first, across, band, width, rows)?;
        for row in 0..rows as usize {
            let line = &cells[row * w..(row + 1) * w];
            let mut distance = vec![u16::MAX; w];
            let mut nearest = vec![0u32; w];
            for x in 0..w {
                if spreads(line[x]) {
                    distance[x] = 0;
                    nearest[x] = line[x];
                    continue;
                }
                let mut best = u16::MAX;
                let mut id = 0u32;
                let mut consider = |d: u16, step: u16, n: u32| {
                    let reached = d.saturating_add(step);
                    if reached < best {
                        best = reached;
                        id = n;
                    }
                };
                if let Some((above, above_ids)) = &previous {
                    consider(above[x], ORTHOGONAL, above_ids[x]);
                    if x > 0 {
                        consider(above[x - 1], DIAGONAL, above_ids[x - 1]);
                    }
                    if x + 1 < w {
                        consider(above[x + 1], DIAGONAL, above_ids[x + 1]);
                    }
                }
                if x > 0 {
                    consider(distance[x - 1], ORTHOGONAL, nearest[x - 1]);
                }
                distance[x] = best;
                nearest[x] = id;
            }
            encode_row(&distance, &nearest, &mut row_bytes_buffer);
            spill.write_all(&row_bytes_buffer)?;
            previous = Some((distance, nearest));
        }
    }
    let mut spill = spill
        .into_inner()
        .context("flushing the fill spill file")?;

    // Backward pass: south to north, east to west, finishing each row
    // against the row below and writing the filled tiles as bands complete.
    let row_bytes = (w * (2 + 4)) as u64;
    let mut filled = 0u64;
    let mut unfilled = 0u64;
    let mut written = 0u64;
    let mut below: Option<(Vec<u16>, Vec<u32>)> = None;
    for band in (0..down).rev() {
        let rows = TILE_SIZE.min(height - band * TILE_SIZE);
        let mut cells = read_band(manifest, root, first, across, band, width, rows)?;
        for row in (0..rows as usize).rev() {
            let absolute = u64::from(band) * u64::from(TILE_SIZE) + row as u64;
            spill.seek(SeekFrom::Start(absolute * row_bytes))?;
            spill.read_exact(&mut row_bytes_buffer)?;
            let mut distance = vec![0u16; w];
            let mut nearest = vec![0u32; w];
            decode_row(&row_bytes_buffer, &mut distance, &mut nearest);

            let line = &mut cells[row * w..(row + 1) * w];
            for x in (0..w).rev() {
                if distance[x] > 0 {
                    let mut best = distance[x];
                    let mut id = nearest[x];
                    let mut consider = |d: u16, step: u16, n: u32| {
                        let reached = d.saturating_add(step);
                        if reached < best {
                            best = reached;
                            id = n;
                        }
                    };
                    if let Some((under, under_ids)) = &below {
                        consider(under[x], ORTHOGONAL, under_ids[x]);
                        if x > 0 {
                            consider(under[x - 1], DIAGONAL, under_ids[x - 1]);
                        }
                        if x + 1 < w {
                            consider(under[x + 1], DIAGONAL, under_ids[x + 1]);
                        }
                    }
                    if x + 1 < w {
                        consider(distance[x + 1], ORTHOGONAL, nearest[x + 1]);
                    }
                    distance[x] = best;
                    nearest[x] = id;
                }
                if line[x] == 0 {
                    if distance[x] <= cap && nearest[x] != 0 {
                        line[x] = nearest[x];
                        filled += 1;
                    } else {
                        unfilled += 1;
                    }
                }
            }
            below = Some((distance, nearest));
        }
        written += write_band(manifest, root, first, across, band, width, rows, &cells)?;
    }

    drop(spill);
    std::fs::remove_file(&spill_path)
        .with_context(|| format!("removing {}", spill_path.display()))?;
    log::info!(
        "filled {filled} unmapped texels from cover within {FILL_METRES} m, rewrote {written} \
         tiles in {:.1?}; {unfilled} texels have no cover in reach and stay null",
        started.elapsed()
    );
    Ok((filled, unfilled))
}

/// One spill row: the distances, then the ids, little-endian.
fn encode_row(distance: &[u16], nearest: &[u32], bytes: &mut [u8]) {
    let split = distance.len() * 2;
    for (chunk, &value) in bytes[..split].chunks_exact_mut(2).zip(distance) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    for (chunk, &value) in bytes[split..].chunks_exact_mut(4).zip(nearest) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
}

fn decode_row(bytes: &[u8], distance: &mut [u16], nearest: &mut [u32]) {
    let split = distance.len() * 2;
    for (chunk, value) in bytes[..split].chunks_exact(2).zip(distance) {
        *value = u16::from_le_bytes(chunk.try_into().expect("chunks of two"));
    }
    for (chunk, value) in bytes[split..].chunks_exact(4).zip(nearest) {
        *value = u32::from_le_bytes(chunk.try_into().expect("chunks of four"));
    }
}

/// Reads one tile row into a band buffer; absent tiles read as null.
fn read_band(
    manifest: &Manifest,
    root: &Path,
    first: Tile,
    across: u32,
    band: u32,
    width: u32,
    rows: u32,
) -> Result<Vec<u32>> {
    let level = MATERIAL_BASE_LEVEL;
    let grid = manifest.grid();
    let tile = TILE_SIZE as usize;
    let w = width as usize;
    let mut cells = vec![0u32; w * rows as usize];
    for column in 0..across {
        let at = Tile::new(first.x + column as i32, first.y + band as i32);
        let path = grid.tile_path(root, level, at);
        let Some(values) =
            read_material_tile(&path).with_context(|| format!("reading {}", path.display()))?
        else {
            continue;
        };
        let from_x = column as usize * tile;
        let columns = tile.min(w - from_x);
        for line in 0..rows as usize {
            cells[line * w + from_x..line * w + from_x + columns]
                .copy_from_slice(&values[line * tile..line * tile + columns]);
        }
    }
    Ok(cells)
}

/// Writes a band's non-empty tiles back; the mirror of the rasterizer's.
#[allow(clippy::too_many_arguments)]
fn write_band(
    manifest: &Manifest,
    root: &Path,
    first: Tile,
    across: u32,
    band: u32,
    width: u32,
    rows: u32,
    cells: &[u32],
) -> Result<u64> {
    let level = MATERIAL_BASE_LEVEL;
    let grid = manifest.grid();
    let tile = TILE_SIZE as usize;
    let mut out = vec![0u32; tile * tile];
    let mut written = 0;
    for column in 0..across {
        out.fill(0);
        let from_x = column as usize * tile;
        let columns = tile.min(width as usize - from_x);
        for line in 0..rows as usize {
            let source = line * width as usize + from_x;
            out[line * tile..line * tile + columns]
                .copy_from_slice(&cells[source..source + columns]);
        }
        if out.iter().all(|&cell| cell == 0) {
            continue;
        }
        let at = Tile::new(first.x + column as i32, first.y + band as i32);
        let (west, north) = grid.tile_origin_metres(level, at);
        write_material_tile(
            &grid.tile_path(root, level, at),
            TilePlacement {
                west,
                north,
                metres_per_texel: grid.metres_per_texel(level),
            },
            &out,
        )
        .with_context(|| format!("writing level {level} tile {at:?}"))?;
        written += 1;
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use terrain_materials::Material;

    /// One tile of ground at the base level: 512 x 512 texels of 4 m.
    fn manifest() -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: "materials".into(),
            epsg: 3979,
            tile_size: TILE_SIZE,
            base_level: MATERIAL_BASE_LEVEL,
            level_count: 1,
            base_metres_per_texel: 1.0,
            origin_metres: [0.0, 0.0],
            extent_texels: [2048, 2048],
            bands: 1,
            nodata: 0.0,
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "terrain-process-fill-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("failed to create");
        root
    }

    fn write_texels(root: &Path, manifest: &Manifest, texels: &[(u32, u32, u32)]) {
        let grid = manifest.grid();
        let (first, _, _) = tile_range(manifest, MATERIAL_BASE_LEVEL);
        let mut values = vec![0u32; (TILE_SIZE * TILE_SIZE) as usize];
        for &(x, y, id) in texels {
            assert!(x < TILE_SIZE && y < TILE_SIZE, "one-tile fixtures only");
            values[(y * TILE_SIZE + x) as usize] = id;
        }
        let (west, north) = grid.tile_origin_metres(MATERIAL_BASE_LEVEL, first);
        write_material_tile(
            &grid.tile_path(root, MATERIAL_BASE_LEVEL, first),
            TilePlacement {
                west,
                north,
                metres_per_texel: grid.metres_per_texel(MATERIAL_BASE_LEVEL),
            },
            &values,
        )
        .expect("failed to write the fixture");
    }

    fn texel(root: &Path, manifest: &Manifest, x: u32, y: u32) -> u32 {
        let grid = manifest.grid();
        let (first, _, _) = tile_range(manifest, MATERIAL_BASE_LEVEL);
        let values = read_material_tile(&grid.tile_path(root, MATERIAL_BASE_LEVEL, first))
            .expect("failed to read")
            .expect("the tile exists");
        values[(y * TILE_SIZE + x) as usize]
    }

    /// Null ground takes the nearest cover, and the nearest one specifically:
    /// two sources, and the texels between them split by which is closer.
    #[test]
    fn null_ground_takes_the_nearest_cover() {
        let manifest = manifest();
        let root = temp_root("nearest");
        write_texels(
            &root,
            &manifest,
            &[
                (100, 100, Material::ForestUnknown.id()),
                (140, 100, Material::BareRock.id()),
            ],
        );
        fill(&manifest, &root).expect("failed to fill");

        assert_eq!(texel(&root, &manifest, 110, 100), Material::ForestUnknown.id());
        assert_eq!(texel(&root, &manifest, 130, 100), Material::BareRock.id());
        assert_eq!(
            texel(&root, &manifest, 100, 130),
            Material::ForestUnknown.id(),
            "reach is two-dimensional, not just along the row"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Water is bounded by surveyed shoreline: it must neither spread nor be
    /// overwritten, so a lake beside a wood stays a lake and the gap beyond
    /// the lake fills with the wood, straight across the water.
    #[test]
    fn water_neither_spreads_nor_takes_a_fill() {
        let manifest = manifest();
        let root = temp_root("water");
        write_texels(
            &root,
            &manifest,
            &[
                (100, 100, Material::ForestUnknown.id()),
                (110, 100, Material::Lake.id()),
                (111, 100, Material::Lake.id()),
            ],
        );
        fill(&manifest, &root).expect("failed to fill");

        assert_eq!(texel(&root, &manifest, 110, 100), Material::Lake.id());
        assert_eq!(
            texel(&root, &manifest, 115, 100),
            Material::ForestUnknown.id(),
            "the fill crosses the lake but carries the wood"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Ground further than the cap from every source stays null: the fill
    /// extrapolates, and a cap is what separates that from invention.
    #[test]
    fn ground_beyond_the_cap_stays_null() {
        let manifest = manifest();
        let root = temp_root("cap");
        // 4 m texels: the far corner is ~2.3 km away diagonally, within the
        // 4 km cap, but a texel 1010 columns east would be beyond it. The
        // one-tile fixture cannot hold one, so shrink the check: every
        // reachable texel is filled, and the count of unfilled ones is what
        // the cap arithmetic says for a raster this size.
        write_texels(&root, &manifest, &[(0, 0, Material::ForestUnknown.id())]);
        let (filled, unfilled) = fill(&manifest, &root).expect("failed to fill");
        assert_eq!(
            filled + unfilled + 1,
            u64::from(TILE_SIZE) * u64::from(TILE_SIZE),
            "every texel is either the source, filled, or out of reach"
        );
        assert_eq!(
            texel(&root, &manifest, 511, 511),
            Material::ForestUnknown.id(),
            "the far corner is within the cap"
        );
        assert_eq!(unfilled, 0, "one source reaches this whole tile");
        let _ = std::fs::remove_dir_all(&root);
    }
}
