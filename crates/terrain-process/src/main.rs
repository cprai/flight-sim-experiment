//! Turns a downloaded tile pyramid into the one the simulator flies over.
//!
//! `terrain-download` fetches what was measured: elevation and colour, each as a
//! mip chain of tiles. The renderer needs one thing more that no survey
//! publishes -- what kind of ground each texel is -- so this copies the
//! measurements across and builds a `materials` pyramid of ground-cover ids
//! beside them, out of the raw OpenStreetMap extract the download left under
//! `osm/`. The renderer opens this directory and nothing else, so what it draws
//! is one directory's worth of files rather than two trees that have to be kept
//! in step.
//!
//! # There used to be a third product here
//!
//! A `<product>-max` quadtree of maximum heights: what lets a ray skip empty air
//! instead of stepping through it. It was built here because building it at
//! runtime cost a second or three every time the clipmap filled from cold and
//! about a millisecond a frame in steady flight, all of it re-deriving the same
//! maxima over the same ground on every launch.
//!
//! `ecffa05` moved it back onto the GPU and it is cheaper there than it ever was
//! on disk -- the whole raster is resident, so the chain is derived once at load
//! in a compute pass and the ceilings it derives are *tighter* than the stored
//! ones, because they close only over levels a ray can actually descend into.
//! Nothing has opened a `-max` directory since. It cost as much as the elevation
//! it was reduced from, 57 GB for the survey this flies, and writing it was the
//! single most expensive thing this tool did.
//!
//! ```text
//! terrain-download --output assets/download ...
//! terrain-process --input assets/download --output assets/terrain
//! flight-sim --terrain assets/terrain
//! ```
//!
//! # What comes out is a fraction of what goes in
//!
//! Only levels at and above `--base-level` are copied, which defaults to the
//! level the renderer holds its raster resident from. Everything finer is
//! downloaded, reduced and left behind: the renderer generates the levels under
//! its base -- fractal detail, tree crowns and stones, all pure functions of
//! position -- and has not opened a stored one since `9ad0ca5`.
//!
//! Measured over the survey this flies: `dtm` 57 GB to 926 MB and `materials`
//! 3.6 GB to 926 MB, with the `-max` product that used to sit beside them gone
//! entirely. The same three cameras draw the 1.9 GB tree and the 117 GB one
//! *pixel for pixel identically*, which is the only interesting thing to say
//! about a change like this.

mod tiles;
mod osm;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use terrain_tiles::{MATERIAL_PRODUCT, Manifest, RESIDENT_BASE_LEVEL};


#[derive(Parser, Debug)]
#[command(about = "Build the tree the simulator flies over", long_about = None)]
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

    /// Skip copying the source products, and only build the derived pyramids.
    ///
    /// For a re-run after a change to how a pyramid is reduced, where the tens
    /// of gigabytes of measurements beside it have not moved.
    #[arg(long)]
    no_copy: bool,

    /// The finest level to write, as a level of the download's own pyramid.
    ///
    /// Defaults to what the renderer holds. Every finer level is skipped rather
    /// than copied, which for a metre survey is the difference between 57 GB of
    /// elevation and under one: the renderer keeps its whole raster resident
    /// from this level up and generates everything under it, so a finer level
    /// would be opened by nothing.
    ///
    /// Zero writes the download whole, which is what to pass when measuring a
    /// finer base against what it costs.
    #[arg(long, value_name = "LEVEL", default_value_t = RESIDENT_BASE_LEVEL)]
    base_level: u32,

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
        anyhow::ensure!(
            arguments.base_level <= manifest.max_level(),
            "{name} stores levels {}..={} and --base-level asks for {}",
            manifest.base_level,
            manifest.max_level(),
            arguments.base_level
        );
        if !arguments.no_copy {
            let base = arguments.base_level.max(manifest.base_level);
            copy_product(
                &arguments.input.join(name),
                &arguments.output.join(name),
                base,
            )?;
            // The copy's own manifest, not the source's: the tree being written
            // starts at a different level from the one being read, and a
            // manifest that said otherwise would send the renderer looking for
            // directories that were deliberately left behind.
            Manifest {
                base_level: base,
                level_count: manifest.max_level() - base + 1,
                ..manifest.clone()
            }
            .write(&arguments.output.join(name))?;
        }
    }

    if build_materials {
        // The grid comes from whichever product is already there --
        // unfiltered, so `--product materials` alone still finds one.
        let (_, reference) = discover(&arguments.input, &[])?
            .into_iter()
            .next()
            .context("materials need an existing product to take the grid from")?;
        osm::build(
            &arguments.input,
            &arguments.output,
            &reference,
            arguments.base_level,
        )?;
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

/// Copies one product directory across, from `base` up, skipping files already
/// there.
///
/// The skip is what makes a re-run cheap: the measurements do not change, so
/// after the first pass this costs a stat per tile rather than a copy. Size
/// rather than a checksum, because a tile is a fixed-size file written whole --
/// a truncated one differs in length, and nothing rewrites a tile in place.
///
/// The manifest is not copied. The caller writes one describing the tree that
/// came out rather than the one that went in.
fn copy_product(source: &Path, destination: &Path, base: u32) -> Result<()> {
    let files: Vec<PathBuf> = walk(source)?
        .into_iter()
        .filter(|relative| level_of(relative).is_none_or(|level| level >= base))
        .filter(|relative| relative != Path::new(terrain_tiles::manifest::MANIFEST_NAME))
        .collect();
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

/// Which level a tile's path belongs to, or [`None`] for anything that is not
/// under a level directory.
///
/// Tiles live in a directory named for their level, two digits wide, so this is
/// the whole of what a copy needs to know to leave the fine ones behind.
fn level_of(relative: &Path) -> Option<u32> {
    relative
        .components()
        .next()?
        .as_os_str()
        .to_str()?
        .parse()
        .ok()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A copy leaves the fine levels behind by reading the level out of a path,
    /// so this is the whole of what decides whether 57 GB is written or one.
    #[test]
    fn a_paths_level_is_the_directory_it_sits_in() {
        assert_eq!(level_of(Path::new("03/-239_-122.tif")), Some(3));
        assert_eq!(level_of(Path::new("00/0_0.tif")), Some(0));
        assert_eq!(level_of(Path::new("12/1_1.tif")), Some(12));
        // Anything that is not a level directory is kept whatever the base is,
        // which is the safe direction: a file this cannot read is a file it has
        // no business deleting.
        assert_eq!(level_of(Path::new("manifest.json")), None);
        assert_eq!(level_of(Path::new("notes/03/x.tif")), None);
    }
}
