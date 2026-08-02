//! Coarse material levels: each texel the commonest id on the ground it names.
//!
//! Averaging is meaningless for ids, so the material pyramid reduces by
//! *mode*: a coarse texel holds the id covering the most base-level ground
//! beneath it. The mode is taken over the base level directly, not over the
//! level below -- mode is not associative, and a mode of modes drifts: four
//! children can each pick a narrow winner while the honest count over the
//! footprint says otherwise. Counting is associative, so every level is
//! built the same way, by counting base texels, and level 8 is exactly as
//! honest as level 3.
//!
//! Each coarse tile counts into a dense table: 512 squared texels by one slot
//! per material -- about 39 MB of `u16`, which the deepest footprint (4096
//! base texels at level 8) cannot overflow. Four workers of that is the same
//! memory ceiling the other passes budget. Reading the whole base level once
//! per level costs pages the cache still holds from the paint.
//!
//! `Null` never wins a count. It is the absence of data, not a kind of
//! ground, and letting it out-vote real cover would erode the mapped world's
//! edges a little more at every level. A coarse texel is `Null` only when
//! nothing at all was counted beneath it -- the same rule the elevation mips
//! apply to their nodata. Ties between real materials go to the higher
//! [`super::classify::precedence`], then the lower id, so reruns are
//! byte-identical.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use rayon::prelude::*;
use terrain_materials::Material;
use terrain_tiles::read::read_material_tile;
use terrain_tiles::write::{TilePlacement, write_material_tile};
use terrain_tiles::{MATERIAL_BASE_LEVEL, Manifest, TILE_SIZE, Tile};

use super::classify::precedence;
use crate::build::tile_range;

/// How many coarse tiles build at once; see the module doc for the budget.
const TILE_THREADS: usize = 4;

/// Dense material table: id to slot, slot to id, and the tie-break key.
struct Palette {
    /// Slot for an id, addressed by `(id >> 8) * 256 + (id & 0xff)`; ids sit
    /// in category blocks, so the table is small and total.
    slots: Vec<u16>,
    ids: Vec<u32>,
    precedences: Vec<u8>,
}

const NO_SLOT: u16 = u16::MAX;

impl Palette {
    fn new() -> Self {
        let highest = Material::ALL
            .iter()
            .map(|material| material.id())
            .max()
            .expect("the enum is not empty");
        let mut slots = vec![NO_SLOT; (highest as usize) + 1];
        let mut ids = Vec::with_capacity(Material::ALL.len());
        let mut precedences = Vec::with_capacity(Material::ALL.len());
        for (slot, &material) in Material::ALL.iter().enumerate() {
            slots[material.id() as usize] = slot as u16;
            ids.push(material.id());
            precedences.push(precedence(material));
        }
        Self {
            slots,
            ids,
            precedences,
        }
    }

    fn slot(&self, id: u32) -> Option<usize> {
        let slot = *self.slots.get(id as usize)?;
        (slot != NO_SLOT).then_some(slot as usize)
    }
}

/// Builds every level above the base, coarsest last.
///
/// Returns how many tiles were written.
pub fn build_levels(manifest: &Manifest, root: &Path) -> Result<u64> {
    let palette = Palette::new();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(TILE_THREADS)
        .build()
        .context("building the thread pool")?;

    let mut written = 0u64;
    for level in (MATERIAL_BASE_LEVEL + 1)..=manifest.max_level() {
        let (first, across, down) = tile_range(manifest, level);
        let tiles: Vec<Tile> = (0..down)
            .flat_map(|row| {
                (0..across)
                    .map(move |column| Tile::new(first.x + column as i32, first.y + row as i32))
            })
            .collect();
        let total = tiles.len() as u64;
        let started = std::time::Instant::now();

        let done = AtomicU64::new(0);
        let level_written: u64 = pool
            .install(|| {
                tiles
                    .into_par_iter()
                    .map(|tile| {
                        let wrote = build_tile(manifest, root, &palette, level, tile)?;
                        let at = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if at.is_multiple_of(200) || at == total {
                            log::info!("level {level}: {at} of {total} tiles");
                        }
                        Ok(u64::from(wrote))
                    })
                    .collect::<Result<Vec<u64>>>()
            })?
            .iter()
            .sum();

        log::info!(
            "level {level}: wrote {level_written} of {total} tiles in {:.1?}",
            started.elapsed()
        );
        written += level_written;
    }
    Ok(written)
}

/// Counts the base ground under one coarse tile and writes the modes.
///
/// Returns whether the tile held anything worth writing.
fn build_tile(
    manifest: &Manifest,
    root: &Path,
    palette: &Palette,
    level: u32,
    tile: Tile,
) -> Result<bool> {
    let grid = manifest.grid();
    let span = i64::from(TILE_SIZE);
    let shift = level - MATERIAL_BASE_LEVEL;
    let materials = palette.ids.len();
    let mut counts = vec![0u16; (TILE_SIZE as usize).pow(2) * materials];

    // The base tiles under this tile: indices double per level of descent.
    let base_first = (i64::from(tile.x) << shift, i64::from(tile.y) << shift);
    let mut any = false;
    for base_y in 0..1i64 << shift {
        for base_x in 0..1i64 << shift {
            let base = Tile::new(
                (base_first.0 + base_x) as i32,
                (base_first.1 + base_y) as i32,
            );
            let path = grid.tile_path(root, MATERIAL_BASE_LEVEL, base);
            let Some(values) = read_material_tile(&path)? else {
                continue;
            };
            any = true;
            for (index, &id) in values.iter().enumerate() {
                if id == 0 {
                    continue;
                }
                let Some(slot) = palette.slot(id) else {
                    anyhow::bail!("{} holds unknown material id {id:#x}", path.display());
                };
                // Where this base texel lands inside the coarse tile.
                let gx = i64::from(base.x) * span + (index as i64 % span);
                let gy = i64::from(base.y) * span + (index as i64 / span);
                let cx = ((gx >> shift) - i64::from(tile.x) * span) as usize;
                let cy = ((gy >> shift) - i64::from(tile.y) * span) as usize;
                counts[(cy * TILE_SIZE as usize + cx) * materials + slot] += 1;
            }
        }
    }
    if !any {
        return Ok(false);
    }

    let mut out = vec![0u32; (TILE_SIZE as usize).pow(2)];
    let mut wrote_any = false;
    for (texel, cell) in out.iter_mut().enumerate() {
        let slice = &counts[texel * materials..(texel + 1) * materials];
        let mut best: Option<(u16, u8, usize)> = None;
        for (slot, &count) in slice.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let candidate = (count, palette.precedences[slot], slot);
            let better = match best {
                None => true,
                Some((c, p, s)) => {
                    // More ground wins; then the higher layer; then the
                    // lower id, which is the lower slot.
                    (count, palette.precedences[slot]) > (c, p)
                        || (count, palette.precedences[slot]) == (c, p) && slot < s
                }
            };
            if better {
                best = Some(candidate);
            }
        }
        if let Some((_, _, slot)) = best {
            *cell = palette.ids[slot];
            wrote_any = true;
        }
    }
    if !wrote_any {
        return Ok(false);
    }

    let (west, north) = grid.tile_origin_metres(level, tile);
    write_material_tile(
        &grid.tile_path(root, level, tile),
        TilePlacement {
            west,
            north,
            metres_per_texel: grid.metres_per_texel(level),
        },
        &out,
    )
    .with_context(|| format!("writing level {level} tile {tile:?}"))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// One base tile, reduced through levels 3 and 4.
    fn manifest() -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: "materials".into(),
            epsg: 3979,
            tile_size: TILE_SIZE,
            base_level: MATERIAL_BASE_LEVEL,
            level_count: 3,
            base_metres_per_texel: 1.0,
            origin_metres: [0.0, 0.0],
            extent_texels: [2048, 2048],
            bands: 1,
            nodata: 0.0,
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("terrain-process-mip-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    fn write_base(root: &Path, manifest: &Manifest, value: impl Fn(u32, u32) -> u32) {
        let grid = manifest.grid();
        let samples: Vec<u32> = (0..TILE_SIZE * TILE_SIZE)
            .map(|index| value(index % TILE_SIZE, index / TILE_SIZE))
            .collect();
        let tile = Tile::new(0, 0);
        let (west, north) = grid.tile_origin_metres(MATERIAL_BASE_LEVEL, tile);
        write_material_tile(
            &grid.tile_path(root, MATERIAL_BASE_LEVEL, tile),
            TilePlacement {
                west,
                north,
                metres_per_texel: grid.metres_per_texel(MATERIAL_BASE_LEVEL),
            },
            &samples,
        )
        .expect("failed to write the base");
    }

    fn read_level(root: &Path, manifest: &Manifest, level: u32) -> Vec<u32> {
        let grid = manifest.grid();
        read_material_tile(&grid.tile_path(root, level, Tile::new(0, 0)))
            .expect("failed to read")
            .expect("the level should have been written")
    }

    /// The pattern where an honest count and a mode-of-modes disagree. In
    /// every 4 x 4 base block the top half is mostly Lake and the bottom all
    /// Forest: 6 Lake against 10 Forest. A mode of the four 2 x 2 modes
    /// would see Lake, Lake, Forest, Forest and give the tie to Lake's
    /// higher layer -- the wrong answer this module exists to not produce.
    #[test]
    fn a_coarse_texel_counts_the_base_not_the_level_below() {
        let manifest = manifest();
        let root = temp_root("exact");
        let lake = Material::Lake.id();
        let forest = Material::ForestUnknown.id();
        write_base(&root, &manifest, |x, y| {
            if y % 4 < 2 {
                // Three Lake and one Forest per 2 x 2.
                if x % 2 == 1 && y % 2 == 1 {
                    forest
                } else {
                    lake
                }
            } else {
                forest
            }
        });
        build_levels(&manifest, &root).expect("failed to build");

        let level3 = read_level(&root, &manifest, 3);
        assert_eq!(level3[0], lake, "the top quadrants lean Lake");
        // Tile row 1 -- index 512 in the 512-wide tile -- covers base rows
        // 2 and 3, the all-Forest half of the block.
        assert_eq!(level3[512], forest);

        let level4 = read_level(&root, &manifest, 4);
        assert_eq!(
            level4[0], forest,
            "10 Forest beats 6 Lake over the full footprint"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Null is absence, not ground: one real texel outvotes 4095 empty ones,
    /// and only a fully empty footprint stays Null.
    #[test]
    fn null_never_outvotes_ground() {
        let manifest = manifest();
        let root = temp_root("null");
        let lake = Material::Lake.id();
        write_base(
            &root,
            &manifest,
            |x, y| if x == 0 && y == 0 { lake } else { 0 },
        );
        build_levels(&manifest, &root).expect("failed to build");

        let level4 = read_level(&root, &manifest, 4);
        assert_eq!(level4[0], lake, "one texel of water carries its cell");
        assert_eq!(level4[1], 0, "an empty footprint stays empty");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A count tie between materials on the same footprint goes to the
    /// higher layer, so water is never buried by an equal amount of grass.
    #[test]
    fn ties_go_to_the_higher_layer() {
        let manifest = manifest();
        let root = temp_root("ties");
        let lake = Material::Lake.id();
        let grass = Material::Grass.id();
        write_base(
            &root,
            &manifest,
            |x, _| if x % 2 == 0 { grass } else { lake },
        );
        build_levels(&manifest, &root).expect("failed to build");

        let level3 = read_level(&root, &manifest, 3);
        assert_eq!(level3[0], lake, "two grass, two lake: water wins the tie");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// An entirely empty region writes nothing at any level.
    #[test]
    fn an_empty_base_writes_no_coarse_tiles() {
        let manifest = manifest();
        let root = temp_root("empty");
        // No base tile at all.
        let written = build_levels(&manifest, &root).expect("failed to build");
        assert_eq!(written, 0);
        let _ = std::fs::remove_dir_all(&root);
    }
}
