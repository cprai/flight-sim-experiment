//! Painting OpenStreetMap ground cover into a materials tile pyramid.
//!
//! The downloader leaves a raw regional extract beside the rasters it
//! fetches; this module turns that extract into the `materials` product: one
//! `u32` [`terrain_materials::Material`] id per texel, on the same grid as every
//! other product, mip levels included. The stages, each its own submodule:
//!
//! - [`read`]: three passes over the `.osm.pbf`, keeping only what paints.
//! - [`classify`]: OSM's tag vocabulary folded onto the material enum.
//! - [`assemble`]: ids into rings, rings into polygons with holes, ways
//!   into strokes for the roads.
//! - [`coastline`]: the sea, which OSM implies rather than maps.
//! - [`rasterize`]: polygons and strokes painted into base tiles, in layers.
//! - [`fill`]: unmapped ground takes the nearest mapped cover, to a limit.
//! - [`mip`]: coarse levels, each texel the commonest id beneath it.
//!
//! The grid is not the extract's to choose: the manifest is cloned from
//! whichever product is already in the download, so `materials` covers
//! exactly the ground the elevation does and the renderer's same-ground
//! check holds by construction. Only the levels differ -- base level
//! [`MATERIAL_BASE_LEVEL`], because vector data has no resolution of its own.

use std::path::Path;

use anyhow::{Context, Result};
use terrain_materials::Material;
use terrain_tiles::{MATERIAL_BASE_LEVEL, MATERIAL_PRODUCT, Manifest};

pub mod assemble;
pub mod classify;
pub mod coastline;
pub mod fill;
pub mod mip;
pub mod rasterize;
pub mod read;

/// Builds the materials product from the extract under `input/osm`.
///
/// `reference` is any product manifest from the same download; everything
/// about the ground is copied from it. The materials manifest is written
/// last, so a run killed partway leaves a directory the renderer refuses to
/// open rather than a pyramid with holes.
pub fn build(input: &Path, output: &Path, reference: &Manifest) -> Result<()> {
    anyhow::ensure!(
        reference.max_level() >= MATERIAL_BASE_LEVEL,
        "the {} product stores nothing at or above level {MATERIAL_BASE_LEVEL}",
        reference.product
    );
    let manifest = Manifest {
        product: MATERIAL_PRODUCT.into(),
        base_level: MATERIAL_BASE_LEVEL,
        level_count: reference.max_level() - MATERIAL_BASE_LEVEL + 1,
        bands: 1,
        nodata: 0.0,
        ..reference.clone()
    };
    let root = output.join(MATERIAL_PRODUCT);
    let started = std::time::Instant::now();

    let osm_dir = input.join("osm");
    let record = read::SourceRecord::read(&osm_dir)?;
    log::info!("building {MATERIAL_PRODUCT} from the {} extract", record.region);
    let mut extract = read::read_extract(&osm_dir.join(&record.file))?;
    extract.nodes.project()?;

    let mut polygons = assemble::polygons(&extract);
    let strokes = assemble::strokes(&extract);
    match coastline::ocean(&extract, coastline::Rect::of_manifest(&manifest)) {
        Some(sea) => polygons.push(sea),
        None => log::warn!("no sea could be closed; the ocean will be unpainted"),
    }

    let base = rasterize::rasterize(&polygons, &strokes, &manifest, &root)?;
    fill::fill(&manifest, &root)?;
    let coarse = mip::build_levels(&manifest, &root)?;
    manifest.write(&root)?;
    log::info!(
        "built {MATERIAL_PRODUCT}: {base} base tiles and {coarse} coarse tiles in {:.1?}",
        started.elapsed()
    );
    log_histogram(&manifest, &root).context("summarising the ground")?;
    Ok(())
}

/// Logs what the ground came out as, at one coarse level, so a run ends with
/// evidence rather than silence. Ocean and forest dominating is a pass; a
/// map that is mostly Null is a bug report.
fn log_histogram(manifest: &Manifest, root: &Path) -> Result<()> {
    use terrain_tiles::read::read_material_tile;
    let level = 4;
    let grid = manifest.grid();
    let (first, across, down) = crate::build::tile_range(manifest, level);
    let mut counts: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for row in 0..down {
        for column in 0..across {
            let tile = terrain_tiles::Tile::new(first.x + column as i32, first.y + row as i32);
            if let Some(values) = read_material_tile(&grid.tile_path(root, level, tile))? {
                for &id in &values {
                    *counts.entry(id).or_default() += 1;
                }
            }
        }
    }
    let (width, height) = manifest.size_texels(level);
    let total = u64::from(width) * u64::from(height);
    let mapped: u64 = counts
        .iter()
        .filter(|&(&id, _)| id != 0)
        .map(|(_, &count)| count)
        .sum();
    let mut sorted: Vec<(u32, u64)> = counts
        .into_iter()
        .filter(|&(id, _)| id != 0)
        .collect();
    sorted.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    let summary: Vec<String> = sorted
        .iter()
        .take(8)
        .map(|&(id, count)| {
            let name = Material::try_from_u32(id)
                .map_or_else(|| format!("{id:#x}"), |material| format!("{material:?}"));
            format!("{name} {:.1}%", 100.0 * count as f64 / total as f64)
        })
        .collect();
    log::info!(
        "ground at level {level}: {}; {:.1}% unmapped",
        summary.join(", "),
        100.0 * (total - mapped) as f64 / total as f64
    );
    Ok(())
}
