//! Turning the per-texel functions into the product directories on disk.
//!
//! Every level is generated rather than reduced. Nothing here reads a level it
//! has already written, so the levels can be built in any order and each one is
//! only ever the same function evaluated on its own lattice, band-limited to
//! its own texels -- see `detail` for why that is the better of the two ways to
//! build a pyramid.
//!
//! The work is one tile per dispatch. The per-texel functions run on the GPU --
//! see `texels` for the driver and `emit.wgsl` for the transcription -- and
//! what is left on the CPU is the half a GPU is no use for: masking the
//! overhang, encoding a TIFF and writing it. Those run across every core while
//! the next tile is being computed, which is why the tiles are pulled through
//! `par_bridge` rather than collected first: a worker takes one tile, and the
//! dispatch that fills it happens under the bridge's own lock, so the shared
//! output buffers are never written by two tiles at once and nothing queues up
//! in memory ahead of the writers.
//!
//! A tile may hang off the edge of the raster: the grid is anchored at the
//! projection's origin, so a coarse tile's span does not divide the download's
//! extent. The part with nothing behind it is written as nodata, exactly as
//! `terrain-download` writes it, rather than as a repeat of the last real
//! texel. The shader computes it anyway -- a branch per texel to skip ground
//! that is about to be overwritten costs more than the texel does -- so the
//! overhang is punched out after the readback instead, over the rectangle that
//! actually has raster behind it.

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use rayon::prelude::*;
use terrain_materials::Material;
use terrain_tiles::write::{TilePlacement, write_height_tile, write_material_tile};
use terrain_tiles::{Manifest, TILE_SIZE, Tile};

use crate::detail::OUTSIDE;
use crate::gpu::Gpu;
use crate::texels::Texels;
use crate::tiles::tile_range;

/// How many tiles to build between progress lines.
const REPORT_EVERY: u64 = 500;

/// Which columns and rows of a tile have raster behind them, or `None` if none
/// of it does.
///
/// A rectangle rather than a per-texel test, because that is what it always is:
/// the raster is an axis-aligned block and so is a tile, so the part of one
/// inside the other is a block too. Punching the overhang out of a finished
/// tile is then two `fill`s and a short loop rather than a quarter of a million
/// bounds checks.
fn inside(manifest: &Manifest, level: u32, tile: Tile) -> Option<(Range<usize>, Range<usize>)> {
    let (origin_column, origin_row) = manifest.origin_texels(level);
    let (width, height) = manifest.size_texels(level);
    let side = i64::from(TILE_SIZE);
    let span = |first: i64, extent: u32| -> Range<usize> {
        let low = (-first).clamp(0, side);
        let high = (i64::from(extent) - first).clamp(low, side);
        low as usize..high as usize
    };
    let columns = span(i64::from(tile.x) * side - origin_column, width);
    let rows = span(i64::from(tile.y) * side - origin_row, height);
    if columns.is_empty() || rows.is_empty() {
        return None;
    }
    Some((columns, rows))
}

/// Where a tile's first texel sits, in raster metres -- east from the western
/// edge of the raster, south from the northern one.
///
/// That is the coordinate system every per-texel function is written in, and
/// the shader walks a tile from here by adding whole texels, so this is the
/// only place the tile's position on the ground is worked out.
///
/// Half a texel in, because the format's rasters are `PixelIsArea`: a texel is
/// a square of ground and its sample is the middle of that square. It may be
/// negative, for a tile that hangs off the western or northern edge.
fn tile_origin_metres(manifest: &Manifest, level: u32, tile: Tile) -> [f32; 2] {
    let (origin_column, origin_row) = manifest.origin_texels(level);
    let metres = manifest.metres_per_texel(level) as f32;
    let at = |index: i32, origin: i64| {
        ((i64::from(index) * i64::from(TILE_SIZE) - origin) as f32 + 0.5) * metres
    };
    [at(tile.x, origin_column), at(tile.y, origin_row)]
}

/// Where one texel of a tile sits, or that it sits outside the raster.
///
/// The two halves of what a tile's position means, put back together: which
/// texels count, and where the first one is. Nothing in a run needs this -- the
/// shader walks a tile from its origin, and the overhang is punched out
/// afterwards -- but it is what the walk has to agree with, so it stays as the
/// thing the tests state that agreement against.
#[cfg(test)]
fn texel_metres(
    manifest: &Manifest,
    level: u32,
    tile: Tile,
    column: usize,
    row: usize,
) -> Option<[f32; 2]> {
    let (columns, rows) = inside(manifest, level, tile)?;
    if !columns.contains(&column) || !rows.contains(&row) {
        return None;
    }
    let origin = tile_origin_metres(manifest, level, tile);
    let metres = manifest.metres_per_texel(level) as f32;
    Some([
        origin[0] + column as f32 * metres,
        origin[1] + row as f32 * metres,
    ])
}

/// Replaces every texel of a finished tile that has no raster behind it.
fn punch_out<T: Copy>(manifest: &Manifest, level: u32, tile: Tile, samples: &mut [T], hole: T) {
    let side = TILE_SIZE as usize;
    let Some((columns, rows)) = inside(manifest, level, tile) else {
        samples.fill(hole);
        return;
    };
    if columns.len() == side && rows.len() == side {
        return;
    }
    for row in 0..side {
        let line = &mut samples[row * side..(row + 1) * side];
        if rows.contains(&row) {
            line[..columns.start].fill(hole);
            line[columns.end..].fill(hole);
        } else {
            line.fill(hole);
        }
    }
}

/// Builds every tile of every stored level of a product and writes it.
///
/// `sample` is handed a level, a tile and the level's texel size and returns
/// the tile's samples, or nothing if the tile has no raster behind it at all.
/// It runs one tile at a time, because it drives the device. `write` runs on
/// every core.
fn over_tiles<T, S, W>(manifest: &Manifest, name: &str, mut sample: S, write: W) -> Result<u64>
where
    T: Send,
    S: FnMut(u32, Tile, f32) -> Option<T> + Send,
    W: Fn(u32, Tile, T) -> Result<()> + Sync,
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
        // Progress is counted here rather than beside the write because this is
        // the stage that runs in order; the bridge lets the writers fall at most
        // a core's worth of tiles behind, so it is still a report of the run
        // rather than of a queue filling up.
        let sample = &mut sample;
        let level_written: u64 = tiles
            .into_iter()
            .filter_map(|tile| {
                let built = sample(level, tile, texel);
                let at = done.fetch_add(1, Ordering::Relaxed) + 1;
                if at.is_multiple_of(REPORT_EVERY) || at == total {
                    log::info!("{name} level {level}: {at} of {total} tiles");
                }
                built.map(|samples| (tile, samples))
            })
            .par_bridge()
            .map(|(tile, samples)| write(level, tile, samples).map(|()| 1u64))
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
pub fn heights(root: &Path, manifest: &Manifest, gpu: &Gpu, texels: &Texels) -> Result<u64> {
    let product = root.join(&manifest.product);
    let written = over_tiles(
        manifest,
        &manifest.product,
        |level, tile, texel| {
            inside(manifest, level, tile)?;
            let origin = tile_origin_metres(manifest, level, tile);
            Some(texels.tile(gpu, origin, texel).0)
        },
        |level, tile, mut samples| {
            punch_out(manifest, level, tile, &mut samples, OUTSIDE);
            let path = manifest.grid().tile_path(&product, level, tile);
            write_height_tile(
                &path,
                placement(manifest, level, tile),
                &samples,
                manifest.nodata,
            )
            .with_context(|| format!("writing level {level} tile {tile:?}"))
        },
    )?;

    // Last, so a killed run leaves a directory the renderer refuses rather than
    // one it opens and reads holes out of.
    manifest.write(&product)?;
    Ok(written)
}

/// Writes the ground-cover product: every level, classified from the fields.
///
/// Including the trees themselves. Where the crowns cover enough of a texel it is
/// written as `Material::Canopy` rather than as the floor of the stand, which is
/// the only thing that tells the renderer a pixel is a treetop -- once the crowns
/// are baked into the heights, a hit on one is indistinguishable from a hit on a
/// hillock, and the march has nothing left to ask.
///
/// The id comes out of the same `cs_texel` dispatch the elevation does, over the
/// same block of ground, so a texel cannot be raised as a tree here and painted
/// as open ground there. Each product still runs its own dispatch, since either
/// can be asked for without the other, and the walk is cheap enough on the
/// device that computing both twice is not worth a shared cache of tiles.
pub fn materials(root: &Path, manifest: &Manifest, gpu: &Gpu, texels: &Texels) -> Result<u64> {
    let product = root.join(&manifest.product);
    // Counted at the finest stored level, which is the one the classifier's
    // thresholds were chosen against.
    let counted: Mutex<BTreeMap<u32, u64>> = Mutex::new(BTreeMap::new());
    let written = over_tiles(
        manifest,
        &manifest.product,
        |level, tile, texel| {
            inside(manifest, level, tile)?;
            let origin = tile_origin_metres(manifest, level, tile);
            Some(texels.tile(gpu, origin, texel).1)
        },
        |level, tile, mut ids| {
            // Null, which is both the format's nodata and what ground outside
            // the raster means.
            punch_out(manifest, level, tile, &mut ids, Material::Null.id());
            if level == manifest.base_level {
                let mut counts = counted.lock().expect("a tile panicked while counting");
                for id in &ids {
                    *counts.entry(*id).or_insert(0u64) += 1;
                }
            }
            let path = manifest.grid().tile_path(&product, level, tile);
            write_material_tile(&path, placement(manifest, level, tile), &ids)
                .with_context(|| format!("writing level {level} tile {tile:?}"))
        },
    )?;

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
mod measure {
    use super::*;

    use terrain_tiles::MATERIAL_PRODUCT;

    use crate::fields::Fields;

    /// What one tile of each product costs, level by level.
    ///
    /// The same measurement that said porting was worth doing, kept so the
    /// before and after are the same number rather than two different ones.
    /// Level matters because the crowns and stones are sampled over the ground
    /// a texel covers: a texel at level 3 spans 8 m and one at level 8 spans
    /// 256 m, and on the CPU that made the coarse levels far more expensive
    /// than their tile counts suggested.
    ///
    /// What it now includes on top of the walk is a dispatch, a blocking
    /// readback, the overhang punched out, and a TIFF encoded and written --
    /// everything a real tile costs. Against the CPU column that is a
    /// comparison in the tile's favour rather than the shader's, which is the
    /// direction a speed claim should be biased.
    ///
    /// Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn measure_what_a_texel_costs_by_level() {
        let mut fields = Fields::new([49152.0, 57344.0], 16.0);
        let relief = crate::shape::Relief {
            valley_metres: 700.0,
            peak_metres: 2600.0,
        };
        crate::shape::raise(&mut fields, relief, 0);
        crate::flow::route(&mut fields);

        let gpu = crate::gpu::test_gpu();
        let texels = Texels::new(&gpu, &fields, TILE_SIZE, 0, relief);
        drop(fields);

        // The real emitters over a one-tile raster, rather than the loops
        // rewritten here. Reimplementing them is how a measurement quietly
        // stops measuring the thing it is named after: written that way, the
        // materials column left out the canopy walk that `materials` does and
        // read a hundredth of the truth.
        let root = std::env::temp_dir().join("terrain-generate-texel-cost");
        println!("level  texel      heights   materials");
        for level in 3..=8u32 {
            let span = TILE_SIZE << level;
            let manifest = |product: &str, nodata: f32| Manifest {
                version: Manifest::VERSION,
                product: product.into(),
                epsg: 3979,
                tile_size: TILE_SIZE,
                base_level: level,
                level_count: 1,
                base_metres_per_texel: 1.0,
                origin_metres: [-1_990_656.0, 536_576.0],
                extent_texels: [span, span],
                bands: 1,
                nodata,
            };

            let at = std::time::Instant::now();
            heights(&root, &manifest("dtm", -32767.0), &gpu, &texels).expect("heights");
            let dtm = at.elapsed();

            let at = std::time::Instant::now();
            materials(&root, &manifest(MATERIAL_PRODUCT, 0.0), &gpu, &texels).expect("materials");
            let cover = at.elapsed();

            let _ = std::fs::remove_dir_all(&root);
            println!(
                "{level:5}  {:5} m  {dtm:>9.2?}  {cover:>9.2?}",
                1u32 << level
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrain_tiles::read::{read_height_tile, read_material_tile};

    use crate::fields::Fields;
    use crate::shape::Relief;

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

    /// A device with that landscape on it, which is what both emitters take
    /// now. Seed 3 rather than 0 so a test would notice the seed being ignored.
    fn driver() -> (crate::gpu::Gpu, Texels) {
        let gpu = crate::gpu::test_gpu();
        let texels = Texels::new(&gpu, &fields(), TILE_SIZE, 3, relief());
        (gpu, texels)
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
        let (gpu, texels) = driver();
        let written = heights(&root, &manifest, &gpu, &texels).expect("failed to write");
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

    /// A tile has to hold the ground its grid position names.
    ///
    /// This is what the wiring to the device could get wrong without anything
    /// else noticing. The shader is handed one origin per tile and walks the
    /// rest by adding whole texels, so an origin off by a texel, by half a
    /// texel, or by a whole tile produces a landscape that is perfectly
    /// plausible, passes every range check, and is in the wrong place -- and
    /// the seam only shows where two tiles meet.
    ///
    /// So the expected origin is worked out here from the raster indices
    /// directly, the way `terrain-download` names ground, rather than from the
    /// function under test. A tile away from the corner, because an index
    /// scaled wrongly still lands on zero at the first one.
    ///
    /// Exact equality, not a tolerance: both sides are the same dispatch of the
    /// same shader, so anything but a bit-for-bit match is a different origin.
    #[test]
    fn a_written_tile_holds_the_ground_its_grid_position_names() {
        let root = temp("heights-placement");
        let manifest = manifest("dtm", 0, -32767.0);
        let (gpu, texels) = driver();
        heights(&root, &manifest, &gpu, &texels).expect("failed to write");

        let (first, across, down) = tile_range(&manifest, 0);
        let tile = Tile::new(first.x + across as i32 - 1, first.y + down as i32 - 1);
        assert!(
            inside(&manifest, 0, tile).is_some_and(|(columns, rows)| {
                columns.len() == TILE_SIZE as usize && rows.len() == TILE_SIZE as usize
            }),
            "the tile under test must be full, or the overhang explains a difference"
        );

        let path = manifest.grid().tile_path(&root.join("dtm"), 0, tile);
        let written = read_height_tile(&path)
            .expect("failed to read")
            .expect("the tile must exist");

        let (origin_column, origin_row) = manifest.origin_texels(0);
        let metres = manifest.metres_per_texel(0) as f32;
        let corner = |index: i32, origin: i64| {
            ((i64::from(index) * i64::from(TILE_SIZE) - origin) as f32 + 0.5) * metres
        };
        let wanted = texels
            .tile(
                &gpu,
                [corner(tile.x, origin_column), corner(tile.y, origin_row)],
                metres,
            )
            .0;
        assert_eq!(
            written,
            wanted,
            "the tile on disk is not the ground at {:?}",
            (tile.x, tile.y)
        );
        // ... and it is ground, rather than two identical buffers of nothing.
        assert!(
            written.iter().any(|height| *height > 500.0),
            "the tile under test holds nothing to compare"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Every level has to be there, or the renderer's clipmap has a ring with
    /// nothing behind it.
    #[test]
    fn every_stored_level_is_written() {
        let root = temp("every-level");
        let manifest = manifest("dtm", 0, -32767.0);
        let (gpu, texels) = driver();
        heights(&root, &manifest, &gpu, &texels).expect("failed to write");
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
        let (gpu, texels) = driver();
        let written = materials(&root, &manifest, &gpu, &texels).expect("failed to write");
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
