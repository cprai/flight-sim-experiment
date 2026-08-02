//! Turns a downloaded tile pyramid into the one the simulator flies over.
//!
//! `terrain-download` fetches what was measured: elevation and colour, each as a
//! mip chain of tiles. The renderer needs one thing more that no survey
//! publishes -- a quadtree of maximum heights, which is what lets a ray skip
//! empty air instead of stepping through it. This builds that, once, and writes
//! it beside the products it was reduced from.
//!
//! It used to be built at runtime, block by block, as the camera moved. That
//! cost a second or three every time the clipmap filled from cold and about a
//! millisecond a frame in steady flight, all of it re-deriving the same
//! maxima over the same ground on every launch. The ground does not change.
//!
//! The output is a complete tree: the source products copied across, plus a
//! `<product>-max` pyramid for each elevation product, plus a `materials`
//! pyramid of ground-cover ids built from the raw OpenStreetMap extract when
//! the download carries one -- see `osm`. The renderer opens this directory
//! and nothing else, so what it draws is one directory's worth of files
//! rather than two trees that have to be kept in step.
//!
//! ```text
//! terrain-download --output assets/download ...
//! terrain-process --input assets/download --output assets/terrain
//! flight-sim --terrain assets/terrain
//! ```

mod build;
mod osm;
mod tiles;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use terrain_tiles::{MATERIAL_PRODUCT, MAXIMA_SUFFIX, Manifest, maxima_product};

use build::tile_range;

#[derive(Parser, Debug)]
#[command(about = "Build the max pyramid the far field is marched through", long_about = None)]
struct Arguments {
    /// Directory `terrain-download` wrote, with a subdirectory per product.
    #[arg(short, long, value_name = "DIR")]
    input: PathBuf,

    /// Directory to build the renderer's tree in.
    #[arg(short, long, value_name = "DIR")]
    output: PathBuf,

    /// Only process these products, rather than everything under `--input`.
    #[arg(long, value_name = "NAME")]
    product: Vec<String>,

    /// Skip copying the source products, and only build the max pyramids.
    ///
    /// For a re-run after a change to how the pyramid is reduced, where the
    /// tens of gigabytes of measurements beside it have not moved.
    #[arg(long)]
    no_copy: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,terrain_process=info"),
    )
    .init();

    let arguments = Arguments::parse();
    let products = discover(&arguments.input, &arguments.product)?;
    // Materials are built rather than copied: raw OpenStreetMap data under
    // `osm/`, recognised by the record the downloader leaves there, becomes
    // a product directory that does not exist under `--input` at all.
    let wants = |name: &str| {
        arguments.product.is_empty() || arguments.product.iter().any(|wanted| wanted == name)
    };
    let osm_record = arguments.input.join("osm").join(osm::read::RECORD_FILE);
    let build_materials = osm_record.exists() && wants(MATERIAL_PRODUCT);
    anyhow::ensure!(
        !products.is_empty() || build_materials,
        "{} holds no product directory with a {} in it",
        arguments.input.display(),
        terrain_tiles::manifest::MANIFEST_NAME
    );

    for (name, manifest) in &products {
        if !arguments.no_copy {
            copy_product(&arguments.input.join(name), &arguments.output.join(name))?;
        }
        // Colour has nothing to bound, and neither do material ids. Only a
        // single-band *elevation* product does, and a pyramid already
        // reduced from one is not reduced again.
        if manifest.bands == 1 && !name.ends_with(MAXIMA_SUFFIX) && name != MATERIAL_PRODUCT {
            build_maxima(&arguments.input, &arguments.output, name, manifest)?;
        }
    }

    if build_materials {
        // The grid comes from whichever product is already there --
        // unfiltered, so `--product materials` alone still finds one.
        let (_, reference) = discover(&arguments.input, &[])?
            .into_iter()
            .next()
            .context("materials need an existing product to take the grid from")?;
        osm::build(&arguments.input, &arguments.output, &reference)?;
    }
    Ok(())
}

/// The products under `root`, in a stable order, with their manifests.
fn discover(root: &Path, wanted: &[String]) -> Result<Vec<(String, Manifest)>> {
    let mut products = Vec::new();
    let entries = std::fs::read_dir(root).with_context(|| format!("reading {}", root.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("reading {}", root.display()))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if !entry.path().is_dir() || !Manifest::path_in(&entry.path()).exists() {
            continue;
        }
        if !wanted.is_empty() && !wanted.contains(&name) {
            continue;
        }
        let manifest = Manifest::read(&entry.path())?;
        products.push((name, manifest));
    }
    products.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(products)
}

/// Copies one product directory across, skipping files already there.
///
/// The skip is what makes a re-run cheap: the measurements do not change, so
/// after the first pass this costs a stat per tile rather than a copy. Size
/// rather than a checksum, because a tile is a fixed-size file written whole --
/// a truncated one differs in length, and nothing rewrites a tile in place.
fn copy_product(source: &Path, destination: &Path) -> Result<()> {
    let files = walk(source)?;
    let copied = std::sync::atomic::AtomicU64::new(0);
    log::info!(
        "copying {} to {}: {} files",
        source.display(),
        destination.display(),
        files.len()
    );

    files.par_iter().try_for_each(|relative| -> Result<()> {
        let from = source.join(relative);
        let to = destination.join(relative);
        let length = from.metadata().map(|meta| meta.len()).unwrap_or(0);
        if to.metadata().is_ok_and(|meta| meta.len() == length) {
            return Ok(());
        }
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        std::fs::copy(&from, &to)
            .with_context(|| format!("copying {} to {}", from.display(), to.display()))?;
        copied.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    })?;

    log::info!(
        "copied {} files, {} already there",
        copied.load(std::sync::atomic::Ordering::Relaxed),
        files.len() as u64 - copied.load(std::sync::atomic::Ordering::Relaxed)
    );
    Ok(())
}

/// Every file under `root`, as paths relative to it.
fn walk(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let entries = std::fs::read_dir(&directory)
            .with_context(|| format!("reading {}", directory.display()))?;
        for entry in entries {
            let path = entry
                .with_context(|| format!("reading {}", directory.display()))?
                .path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.push(
                    path.strip_prefix(root)
                        .expect("walked from the root")
                        .to_path_buf(),
                );
            }
        }
    }
    Ok(files)
}

/// Builds one elevation product's max pyramid, reading `input` and writing
/// beside the copies under `output`.
///
/// Read from the download rather than from the copy so that `--no-copy` is a
/// complete run on its own: rebuilding the pyramid after a change to how it is
/// reduced should not depend on tens of gigabytes of measurements having been
/// duplicated first.
fn build_maxima(input: &Path, output: &Path, product: &str, source: &Manifest) -> Result<()> {
    let name = maxima_product(product);
    let source_root = input.join(product);
    let maxima_root = output.join(&name);

    // Everything about the ground is the source's; only what the values mean
    // differs, and the pyramid is written at exactly the levels the elevation
    // is. Coarser ones are derived rather than measured, so the renderer folds
    // them in memory instead -- the coarsest stored level of a raster this size
    // is under a megabyte.
    let manifest = Manifest {
        product: name.clone(),
        ..source.clone()
    };

    let (_, across, down) = tile_range(source, 0);
    log::info!(
        "building {name}: {across} x {down} tiles at depth 0, depths up to {}",
        manifest.max_level()
    );
    let started = std::time::Instant::now();

    let mut written = 0;
    for level in 0..=manifest.max_level() {
        written += build::build_depth(&source_root, &maxima_root, &manifest, level)?;
    }

    // Written last, so a run killed partway leaves a directory the renderer
    // refuses to open rather than one it opens and reads holes out of.
    manifest.write(&maxima_root)?;
    log::info!("built {name}: {written} tiles in {:.1?}", started.elapsed());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use terrain_tiles::write::{TilePlacement, write_height_tile};
    use terrain_tiles::{NODATA_BELOW, TILE_SIZE, Tile};

    /// Five tiles square at the base.
    ///
    /// Wider than one block, which is four, so the pass has to stitch blocks
    /// together as well as tiles. A cell at a block's east edge is built from an
    /// overhang read out of the next block's tiles, and getting that wrong is
    /// invisible at any size that fits in a single block.
    const SIDE: u32 = TILE_SIZE * 5;
    const LEVELS: u32 = 3;
    const NODATA: f32 = -32767.0;

    fn manifest() -> Manifest {
        Manifest {
            version: Manifest::VERSION,
            product: "dtm".into(),
            epsg: 3979,
            tile_size: TILE_SIZE,
            base_level: 0,
            level_count: LEVELS,
            base_metres_per_texel: 1.0,
            origin_metres: [-1_974_272.0, 524_288.0],
            extent_texels: [SIDE, SIDE],
            bands: 1,
            nodata: NODATA,
        }
    }

    /// Ridged, so a maximum is a real choice rather than whichever corner was
    /// picked, with a block of unmeasured ground in one corner.
    fn height(x: u32, y: u32) -> f32 {
        if x < 40 && y < 40 {
            return NODATA;
        }
        let ridge = ((x * 7 + y * 13) % 17) as f32;
        ridge * ridge * 0.5 - f32::from(x.is_multiple_of(11)) * 40.0
    }

    /// The elevation mip chain, as `terrain-download` would have written it:
    /// each level the mean of the four texels under it, holes left out.
    ///
    /// The pyramid is a bound on *these*, not on the raster alone -- a coarse
    /// level's samples answer for ground beyond any one cell -- so the tool
    /// reads them and so must anything checking it.
    fn mips() -> Vec<(u32, Vec<f32>)> {
        let base: Vec<f32> = (0..SIDE * SIDE)
            .map(|index| height(index % SIDE, index / SIDE))
            .collect();
        let mut chain = vec![(SIDE, base)];
        for _ in 1..LEVELS {
            let (span, fine) = chain.last().expect("never empty");
            let (span, half) = (*span, span / 2);
            let mut coarse = vec![0.0; (half * half) as usize];
            for y in 0..half as usize {
                for x in 0..half as usize {
                    let at = |dx: usize, dy: usize| fine[(2 * y + dy) * span as usize + 2 * x + dx];
                    let real: Vec<f32> = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)]
                        .into_iter()
                        .filter(|value| *value > NODATA_BELOW)
                        .collect();
                    coarse[y * half as usize + x] = if real.is_empty() {
                        NODATA
                    } else {
                        real.iter().sum::<f32>() / real.len() as f32
                    };
                }
            }
            chain.push((half, coarse));
        }
        chain
    }

    /// Writes the source elevation product, every level of it.
    fn write_source(root: &Path, manifest: &Manifest, mips: &[(u32, Vec<f32>)]) {
        let grid = manifest.grid();
        let tile = TILE_SIZE as usize;
        for (level, (span, values)) in mips.iter().enumerate() {
            let level = level as u32;
            let (first, across, down) = tile_range(manifest, level);
            for row in 0..down {
                for column in 0..across {
                    let samples: Vec<f32> = (0..tile * tile)
                        .map(|index| {
                            let x = column * TILE_SIZE + (index % tile) as u32;
                            let y = row * TILE_SIZE + (index / tile) as u32;
                            if x < *span && y < *span {
                                values[(y * span + x) as usize]
                            } else {
                                NODATA
                            }
                        })
                        .collect();
                    let at = Tile::new(first.x + column as i32, first.y + row as i32);
                    let (west, north) = grid.tile_origin_metres(level, at);
                    write_height_tile(
                        &grid.tile_path(root, level, at),
                        TilePlacement {
                            west,
                            north,
                            metres_per_texel: grid.metres_per_texel(level),
                        },
                        &samples,
                        manifest.nodata,
                    )
                    .expect("failed to write a source tile");
                }
            }
        }
    }

    /// Reads a whole level of a written product back, row-major.
    ///
    /// Whole rather than a cell at a time: a tile is a file to open and decode,
    /// and these tests walk every cell of every depth.
    fn read_level(root: &Path, manifest: &Manifest, level: u32) -> Vec<f32> {
        let (width, height) = manifest.size_texels(level);
        let mut out = vec![manifest.nodata; (width as usize) * (height as usize)];
        let (first, across, down) = tile_range(manifest, level);
        let (_, offset_x, offset_y) = manifest.tile_of_texel(level, 0, 0);
        assert_eq!(
            (offset_x, offset_y),
            (0, 0),
            "this raster's origin should sit on a tile boundary"
        );

        let tile = TILE_SIZE as usize;
        for row in 0..down {
            for column in 0..across {
                let at = Tile::new(first.x + column as i32, first.y + row as i32);
                let path = manifest.grid().tile_path(root, level, at);
                let Some(values) =
                    terrain_tiles::read::read_height_tile(&path).expect("failed to read")
                else {
                    continue;
                };
                let (at_x, at_y) = (column as usize * tile, row as usize * tile);
                let columns = tile.min(width as usize - at_x);
                let rows = tile.min(height as usize - at_y);
                for line in 0..rows {
                    let to = (at_y + line) * width as usize + at_x;
                    out[to..to + columns]
                        .copy_from_slice(&values[line * tile..line * tile + columns]);
                }
            }
        }
        out
    }

    fn build(name: &str) -> (PathBuf, Manifest, Vec<(u32, Vec<f32>)>) {
        let root =
            std::env::temp_dir().join(format!("terrain-process-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let source = manifest();
        let mips = mips();
        write_source(&root.join("dtm"), &source, &mips);
        build_maxima(&root, &root, "dtm", &source).expect("failed to build");
        (root.join(maxima_product("dtm")), source, mips)
    }

    /// The definition, read straight off the mip chain: the greatest, over every
    /// level at or above this depth, of that level's samples across the closed
    /// square the cell is named after.
    fn defined(mips: &[(u32, Vec<f32>)], depth: u32, i: u32, j: u32) -> f32 {
        let mut highest = f32::NEG_INFINITY;
        for (level, (span, values)) in mips.iter().enumerate().take(depth as usize + 1) {
            let step = 1u32 << (depth - level as u32);
            for y in j * step..=(j + 1) * step {
                for x in i * step..=(i + 1) * step {
                    if x < *span && y < *span {
                        highest = highest.max(values[(y * span + x) as usize]);
                    }
                }
            }
        }
        highest
    }

    /// Every cell the tool writes is exactly the bound it claims to be, checked
    /// through the files the renderer will actually open.
    ///
    /// Equality rather than a bound in one direction, because both directions
    /// matter and they fail differently: too low is a ray passing through solid
    /// ground, which draws as sky on the far side of a ridge, and too high is
    /// the march descending into cells it could have skipped. It also pins the
    /// block seams -- a cell built at the east edge of one block must come out
    /// identical to what a block boundary elsewhere would have produced.
    #[test]
    fn every_written_cell_is_the_bound_the_levels_under_it_ask_for() {
        let (root, source, mips) = build("bounds");
        let maxima = Manifest {
            product: maxima_product("dtm"),
            ..source
        };

        for depth in 0..=maxima.max_level() {
            let span = SIDE >> depth;
            let cells = read_level(&root, &maxima, depth);
            for j in 0..span {
                for i in 0..span {
                    assert_eq!(
                        cells[(j * span + i) as usize],
                        defined(&mips, depth, i, j),
                        "depth {depth} cell ({i}, {j})"
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(root.parent().expect("under a root"));
    }

    /// The property the march depends on: a cell is at or above every height
    /// sample of every clipmap level that reads it, across the closed square it
    /// covers.
    ///
    /// The far corner of that square is the case worth having a test for. At its
    /// finest depth the march intersects a bilinear patch through four of
    /// clipmap level `l`'s height samples, and those are means over `2^l` raster
    /// texels each, so the sample at a cell's far corner answers for ground the
    /// cell does not cover.
    #[test]
    fn a_written_cell_bounds_the_samples_of_every_level_that_reads_it() {
        let (root, source, mips) = build("levels");
        let maxima = Manifest {
            product: maxima_product("dtm"),
            ..source
        };

        let mut checked = 0;
        for depth in 0..=maxima.max_level() {
            let cells = read_level(&root, &maxima, depth);
            let span = SIDE >> depth;
            for level in 0..=depth {
                let mip = depth - level;
                let (mip_span, values) = &mips[level as usize];
                for j in 0..span {
                    for i in 0..span {
                        let ceiling = cells[(j * span + i) as usize];
                        for t_y in j << mip..=(j + 1) << mip {
                            for t_x in i << mip..=(i + 1) << mip {
                                if t_x >= *mip_span || t_y >= *mip_span {
                                    continue;
                                }
                                let height = values[(t_y * mip_span + t_x) as usize];
                                assert!(
                                    ceiling >= height,
                                    "depth {depth} as level {level} mip {mip}: cell ({i}, {j}) \
                                     claims {ceiling} but texel ({t_x}, {t_y}) is at {height}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 10_000, "only {checked} texels were checked");
        let _ = std::fs::remove_dir_all(root.parent().expect("under a root"));
    }

    /// Unmeasured ground must stay recognisable as such rather than becoming a
    /// ceiling of its own, and must not bury the real ground beside it.
    #[test]
    fn a_hole_stays_a_hole_and_does_not_lower_its_neighbours() {
        let (root, source, mips) = build("holes");
        let maxima = Manifest {
            product: maxima_product("dtm"),
            ..source
        };

        for depth in 0..=maxima.max_level() {
            let cells = read_level(&root, &maxima, depth);
            let span = SIDE >> depth;
            // Deep inside the unmeasured corner nothing has been invented ...
            assert!(
                cells[0] < NODATA_BELOW,
                "depth {depth} invented ground at {} over a hole",
                cells[0]
            );
            // ... and the first cell that can see past it reports the ground
            // there rather than being dragged down towards the sentinel.
            let edge = (40 >> depth) + 1;
            assert!(
                cells[(edge * span + edge) as usize] > NODATA_BELOW,
                "depth {depth} let the hole bury the ground beside it"
            );
            assert_eq!(
                cells[(edge * span + edge) as usize],
                defined(&mips, depth, edge, edge)
            );
        }
        let _ = std::fs::remove_dir_all(root.parent().expect("under a root"));
    }
}
