//! Turning the per-texel functions into the product directories on disk.
//!
//! Every level is generated rather than reduced. Nothing here reads a level it
//! has already written, so the levels can be built in any order and each one is
//! only ever the same function evaluated on its own lattice, band-limited to
//! its own texels -- see `detail` for why that is the better of the two ways to
//! build a pyramid.
//!
//! The work is one tile per task. A tile is a quarter of a megabyte of samples
//! and about a megabyte of file, so a worker's whole state is one buffer and
//! there is nothing to share, which is what lets this run on every core the
//! machine has without any of the blocking the max pyramid needs.
//!
//! A tile may hang off the edge of the raster: the grid is anchored at the
//! projection's origin, so a coarse tile's span does not divide the download's
//! extent. The part with nothing behind it is written as nodata, exactly as
//! `terrain-download` writes it, rather than as a repeat of the last real
//! texel.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use rayon::prelude::*;
use terrain_materials::Material;
use terrain_tiles::write::{TilePlacement, write_height_tile, write_material_tile};
use terrain_tiles::{Manifest, TILE_SIZE, Tile};

use crate::build::tile_range;
use crate::classify::material;
use crate::detail::{self, OUTSIDE};
use crate::fields::Fields;
use crate::shape::Relief;

/// How many tiles to build between progress lines.
const REPORT_EVERY: u64 = 500;

/// Where one texel of a tile sits, or that it sits outside the raster.
///
/// Returned as raster metres -- east from the western edge, south from the
/// northern one -- because that is the coordinate system both per-texel
/// functions are written in.
fn texel_metres(
    manifest: &Manifest,
    level: u32,
    tile: Tile,
    column: usize,
    row: usize,
) -> Option<[f32; 2]> {
    let (origin_column, origin_row) = manifest.origin_texels(level);
    let (width, height) = manifest.size_texels(level);
    let raster_column = i64::from(tile.x) * i64::from(TILE_SIZE) + column as i64 - origin_column;
    let raster_row = i64::from(tile.y) * i64::from(TILE_SIZE) + row as i64 - origin_row;
    if raster_column < 0
        || raster_row < 0
        || raster_column >= i64::from(width)
        || raster_row >= i64::from(height)
    {
        return None;
    }
    // Half a texel in, because the format's rasters are `PixelIsArea`: a texel
    // is a square of ground and its sample is the middle of that square.
    let metres = manifest.metres_per_texel(level) as f32;
    Some([
        (raster_column as f32 + 0.5) * metres,
        (raster_row as f32 + 0.5) * metres,
    ])
}

/// Runs `build` over every tile of every stored level of a product.
///
/// The closure is handed a level, a tile and the level's texel size, and
/// returns how many tiles it wrote -- one, or none if the tile turned out to
/// hold nothing.
fn over_tiles<F>(manifest: &Manifest, name: &str, build: F) -> Result<u64>
where
    F: Fn(u32, Tile, f32) -> Result<u64> + Sync,
{
    let started = std::time::Instant::now();
    let mut written = 0;
    for level in manifest.base_level..=manifest.max_level() {
        let (first, across, down) = tile_range(manifest, level);
        let tiles: Vec<Tile> = (0..down)
            .flat_map(|row| {
                (0..across)
                    .map(move |column| Tile::new(first.x + column as i32, first.y + row as i32))
            })
            .collect();
        let total = tiles.len() as u64;
        log::info!("{name} level {level}: {total} tiles");

        let texel = manifest.metres_per_texel(level) as f32;
        let done = AtomicU64::new(0);
        let level_written: u64 = tiles
            .into_par_iter()
            .map(|tile| {
                let built = build(level, tile, texel);
                let at = done.fetch_add(1, Ordering::Relaxed) + 1;
                if at.is_multiple_of(REPORT_EVERY) || at == total {
                    log::info!("{name} level {level}: {at} of {total} tiles");
                }
                built
            })
            .collect::<Result<Vec<u64>>>()?
            .iter()
            .sum();
        written += level_written;
    }
    log::info!("wrote {name}: {written} tiles in {:.1?}", started.elapsed());
    Ok(written)
}

/// Where a tile sits on the ground.
fn placement(manifest: &Manifest, level: u32, tile: Tile) -> TilePlacement {
    let grid = manifest.grid();
    let (west, north) = grid.tile_origin_metres(level, tile);
    TilePlacement {
        west,
        north,
        metres_per_texel: grid.metres_per_texel(level),
    }
}

/// Writes the elevation product: every level, generated from the fields.
///
/// The trees are in here, at every level. Nothing grows a crown at run time any
/// more: a ray meets a tree by meeting the ground, which is the whole of why the
/// canopy stopped costing three quarters of the frame.
///
/// Baked at each level's own texel size, which is what `terrain_canopy::baked`
/// takes it for -- a texel narrower than a crown is asking about one point of one
/// tree, and a texel many crowns wide has to stand for the lot.
pub fn heights(
    root: &Path,
    manifest: &Manifest,
    fields: &Fields,
    seed: u32,
    relief: Relief,
) -> Result<u64> {
    let product = root.join(&manifest.product);
    let cell = fields.metres_per_cell;
    let written = over_tiles(manifest, &manifest.product, |level, tile, texel| {
        let side = TILE_SIZE as usize;
        let mut samples = vec![OUTSIDE; side * side];
        let mut any = false;
        for row in 0..side {
            for column in 0..side {
                let Some([x, y]) = texel_metres(manifest, level, tile, column, row) else {
                    continue;
                };
                let sample = fields.sample(x, y);
                let ground = detail::ground(&sample, texel, cell);
                let bare = detail::height(&sample, &ground, x, y, texel, seed);
                let grown = crate::classify::trees(&sample, &ground, x, y, texel, seed, relief);
                samples[row * side + column] =
                    bare + terrain_canopy::baked(x, y, &standing(&grown, x, y), texel).height;
                any = true;
            }
        }
        if !any {
            return Ok(0);
        }
        let path = manifest.grid().tile_path(&product, level, tile);
        write_height_tile(
            &path,
            placement(manifest, level, tile),
            &samples,
            manifest.nodata,
        )
        .with_context(|| format!("writing level {level} tile {tile:?}"))?;
        Ok(1)
    })?;

    // Last, so a killed run leaves a directory the renderer refuses rather than
    // one it opens and reads holes out of.
    manifest.write(&product)?;
    Ok(written)
}

/// The cover a point stands under, from what the classifier grew there.
///
/// One place, because both products ask the same question and an answer that
/// differed between them would draw a tree and paint a meadow. The clumping is
/// applied here rather than in the classifier because it is a *look*, not a fact
/// about the landscape: it thins and thickens a stand inside itself so that a
/// classifier writing one density per texel does not draw as one flat density.
fn standing(grown: &crate::classify::Trees, x: f32, y: f32) -> terrain_canopy::Cover {
    terrain_canopy::Cover {
        density: grown.density * terrain_canopy::clump(x, y),
        health: grown.health,
    }
}

/// Writes the ground-cover product: every level, classified from the fields.
///
/// Including the trees themselves. Where the crowns cover enough of a texel it is
/// written as `Material::Canopy` rather than as the floor of the stand, which is
/// the only thing that tells the renderer a pixel is a treetop -- once the crowns
/// are baked into the heights, a hit on one is indistinguishable from a hit on a
/// hillock, and the march has nothing left to ask.
///
/// The share comes out of the same `terrain_canopy::baked` call the elevation
/// used, over the same block of ground, so a texel cannot be raised as a tree
/// here and painted as open ground there.
pub fn materials(
    root: &Path,
    manifest: &Manifest,
    fields: &Fields,
    seed: u32,
    relief: Relief,
) -> Result<u64> {
    let product = root.join(&manifest.product);
    let cell = fields.metres_per_cell;
    // Counted at the finest stored level, which is the one the classifier's
    // thresholds were chosen against.
    let counted: Mutex<BTreeMap<u32, u64>> = Mutex::new(BTreeMap::new());
    let written = over_tiles(manifest, &manifest.product, |level, tile, texel| {
        let side = TILE_SIZE as usize;
        // Null, which is both the format's nodata and what ground outside the
        // raster means.
        let mut ids = vec![Material::Null.id(); side * side];
        let mut any = false;
        for row in 0..side {
            for column in 0..side {
                let Some([x, y]) = texel_metres(manifest, level, tile, column, row) else {
                    continue;
                };
                let sample = fields.sample(x, y);
                let ground = detail::ground(&sample, texel, cell);
                let floor = material(&sample, &ground, x, y, texel, seed, relief);
                let grown = crate::classify::trees(&sample, &ground, x, y, texel, seed, relief);
                let share = terrain_canopy::baked(x, y, &standing(&grown, x, y), texel).share;
                // One rule at every level, and it means two different things
                // without needing to be told which: close up the block is inside
                // a single crown or the gap beside it, so this asks "is this
                // texel a treetop"; far out it spans a stand, so it asks "is this
                // mostly wood". Both are the question the pixel wants answered.
                ids[row * side + column] = if share >= terrain_canopy::PAINTED {
                    Material::Canopy.id()
                } else {
                    floor.id()
                };
                any = true;
            }
        }
        if !any {
            return Ok(0);
        }
        if level == manifest.base_level {
            let mut counts = counted.lock().expect("a tile panicked while counting");
            for id in &ids {
                *counts.entry(*id).or_insert(0u64) += 1;
            }
        }
        let path = manifest.grid().tile_path(&product, level, tile);
        write_material_tile(&path, placement(manifest, level, tile), &ids)
            .with_context(|| format!("writing level {level} tile {tile:?}"))?;
        Ok(1)
    })?;

    report(
        &counted
            .into_inner()
            .expect("a tile panicked while counting"),
    );
    manifest.write(&product)?;
    Ok(written)
}

/// Logs what the landscape was covered in, commonest first.
///
/// The only feedback there is on whether the classifier's thresholds are set
/// anywhere near right. A landscape that comes out a tenth water, or all one
/// kind of forest, is wrong in a way that a single rendered frame can easily
/// fail to show and that this cannot miss.
fn report(counts: &BTreeMap<u32, u64>) {
    let total: u64 = counts.values().sum();
    if total == 0 {
        return;
    }
    let mut ordered: Vec<(&u32, &u64)> = counts.iter().collect();
    ordered.sort_by_key(|(id, count)| (std::cmp::Reverse(**count), **id));
    for (id, count) in ordered {
        let share = *count as f64 * 100.0 / total as f64;
        match Material::try_from_u32(*id) {
            Some(material) => log::info!("materials: {share:6.2}% {material:?}"),
            // Unreachable while the classifier only returns variants, and worth
            // saying rather than hiding: a stray id draws as magenta ground.
            None => log::warn!("materials: {share:6.2}% is the unassigned id {id:#x}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrain_tiles::read::{read_height_tile, read_material_tile};

    fn manifest(product: &str, base_level: u32, nodata: f32) -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: product.into(),
            epsg: 3979,
            tile_size: TILE_SIZE,
            base_level,
            level_count: 4 - base_level,
            base_metres_per_texel: 1.0,
            // On a level-3 tile boundary, which is the coarsest here.
            origin_metres: [-1_990_656.0, 536_576.0],
            extent_texels: [2048, 1024],
            bands: 1,
            nodata,
        }
    }

    /// The same span the generator defaults to, so a test writes the heights a
    /// real run would.
    fn relief() -> Relief {
        Relief {
            valley_metres: 700.0,
            peak_metres: 2600.0,
        }
    }

    fn fields() -> Fields {
        let mut fields = Fields::new([2048.0, 1024.0], 32.0);
        crate::shape::raise(
            &mut fields,
            Relief {
                valley_metres: 700.0,
                peak_metres: 2600.0,
            },
            5,
        );
        crate::flow::route(&mut fields);
        fields
    }

    fn temp(name: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("terrain-generate-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    /// The tiles have to be readable through the reader the renderer actually
    /// uses, at the placement their grid position calls for. Anything less than
    /// a round trip through the real files would not catch a tile written to
    /// the wrong path or with the wrong tiepoint, and neither of those reports
    /// an error at run time -- they move the scenery.
    #[test]
    fn an_elevation_product_round_trips_through_the_readers() {
        let root = temp("heights-round-trip");
        let manifest = manifest("dtm", 0, -32767.0);
        let written = heights(&root, &manifest, &fields(), 3, relief()).expect("failed to write");
        assert!(written > 0);

        let read = Manifest::read(&root.join("dtm")).expect("the manifest must validate");
        assert_eq!(read, manifest);

        let (first, across, down) = tile_range(&manifest, 0);
        assert_eq!((across, down), (4, 2));
        let path = manifest.grid().tile_path(&root.join("dtm"), 0, first);
        let samples = read_height_tile(&path)
            .expect("failed to read")
            .expect("the north-west tile must exist");
        assert_eq!(samples.len(), (TILE_SIZE * TILE_SIZE) as usize);
        assert!(
            samples
                .iter()
                .all(|value| *value > 500.0 && *value < 3000.0),
            "a texel came out outside any plausible height"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Every level has to be there, or the renderer's clipmap has a ring with
    /// nothing behind it.
    #[test]
    fn every_stored_level_is_written() {
        let root = temp("every-level");
        let manifest = manifest("dtm", 0, -32767.0);
        heights(&root, &manifest, &fields(), 3, relief()).expect("failed to write");
        for level in manifest.base_level..=manifest.max_level() {
            let directory = root.join("dtm").join(format!("{level:02}"));
            let count = std::fs::read_dir(&directory)
                .with_context(|| format!("reading {}", directory.display()))
                .expect("a level directory must exist")
                .count();
            let (_, across, down) = tile_range(&manifest, level);
            assert_eq!(count as u32, across * down, "level {level}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Materials are ids, not numbers to interpolate, so the round trip has to
    /// come back with ids the shared enum recognises. An unassigned id draws as
    /// magenta and nothing reports it.
    #[test]
    fn every_material_written_is_one_the_shared_enum_knows() {
        let root = temp("materials-round-trip");
        let manifest = manifest("materials", 2, 0.0);
        let written = materials(
            &root,
            &manifest,
            &fields(),
            3,
            Relief {
                valley_metres: 700.0,
                peak_metres: 2600.0,
            },
        )
        .expect("failed to write");
        assert!(written > 0);

        let (first, _, _) = tile_range(&manifest, 2);
        let path = manifest.grid().tile_path(&root.join("materials"), 2, first);
        let ids = read_material_tile(&path)
            .expect("failed to read")
            .expect("the north-west tile must exist");
        let side = TILE_SIZE as usize;
        let mut inside = 0;
        for (index, id) in ids.iter().enumerate() {
            let material =
                Material::try_from_u32(*id).unwrap_or_else(|| panic!("{id:#x} is not a material"));
            // Ground outside the raster is `Null` by design; ground inside it
            // must always have been classified as something.
            if texel_metres(&manifest, 2, first, index % side, index / side).is_some() {
                assert_ne!(
                    material,
                    Material::Null,
                    "texel {index} was left unclassified"
                );
                inside += 1;
            } else {
                assert_eq!(
                    material,
                    Material::Null,
                    "texel {index} is outside the raster"
                );
            }
        }
        assert!(inside > 0, "the north-west tile holds no raster at all");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A coarse tile hangs off the edge of the raster, because the grid is
    /// anchored at the projection's origin rather than at the extent. What is
    /// behind the overhang is nothing, and it has to read as nothing.
    #[test]
    fn ground_outside_the_raster_is_written_as_a_hole() {
        let manifest = manifest("dtm", 0, -32767.0);
        let (first, _, _) = tile_range(&manifest, 3);
        // Level 3 tiles span 4096 m and the extent is 2048 wide, so the
        // north-west tile of the coarsest level reaches past it.
        let inside = texel_metres(&manifest, 3, first, 0, 0);
        assert!(inside.is_some(), "the first texel must be inside");
        let outside = texel_metres(&manifest, 3, first, TILE_SIZE as usize - 1, 0);
        assert!(outside.is_none(), "the last texel must be outside");
    }

    /// Texels sample the middle of the ground they cover, which is what
    /// `PixelIsArea` means. Half a texel out and every product would be offset
    /// from every other by half a texel of its own level, which draws as
    /// material boundaries that do not sit on the terrain they belong to.
    #[test]
    fn a_texel_samples_the_middle_of_the_ground_it_covers() {
        let manifest = manifest("dtm", 0, -32767.0);
        let (first, _, _) = tile_range(&manifest, 0);
        assert_eq!(texel_metres(&manifest, 0, first, 0, 0), Some([0.5, 0.5]));
        assert_eq!(texel_metres(&manifest, 0, first, 3, 7), Some([3.5, 7.5]));

        // ... and a level-2 texel covers four metres, so its middle is two in.
        let (first, _, _) = tile_range(&manifest, 2);
        assert_eq!(texel_metres(&manifest, 2, first, 0, 0), Some([2.0, 2.0]));
    }
}
