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

mod tiles;
mod osm;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use terrain_tiles::{MATERIAL_PRODUCT, Manifest};


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

    for (name, _) in &products {
        if !arguments.no_copy {
            copy_product(&arguments.input.join(name), &arguments.output.join(name))?;
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
