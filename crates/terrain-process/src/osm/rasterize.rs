//! Painting polygons into the base level, one row of tiles at a time.
//!
//! The unit of work is a *band*: every tile in one tile row, painted into a
//! single buffer spanning the raster's full width. Bands exist because the
//! obvious alternative -- rasterizing each tile independently -- has a seam
//! problem: scanline parity counts crossings from infinity, so a tile in the
//! middle of a lake contains no edges at all and can only be filled correctly
//! by geometry it does not intersect. Painting whole rows from the full,
//! unclipped geometry makes a seam impossible by construction, and each
//! raster row is computed exactly once.
//!
//! Within a band, polygons paint in one global order -- layer, then area
//! descending, then discovery order -- so overlaps resolve the same way in
//! every band. Later paints overwrite earlier ones: the painter's algorithm,
//! with [`super::classify::precedence`] deciding who paints later. Area
//! breaks ties within a layer because a polygon that contains another always
//! has the larger boundary; painting the container first lets the contained
//! one -- a clearing in a forest, both vegetation -- show through.
//!
//! Fill rule: even-odd across all a polygon's rings, sampled at texel
//! centres, with the half-open crossing rule (an edge counts where
//! `low <= centre < high`) so a vertex shared by two edges counts once and
//! abutting polygons neither gap nor double-paint.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use rayon::prelude::*;
use terrain_tiles::write::{TilePlacement, write_material_tile};
use terrain_tiles::{MATERIAL_BASE_LEVEL, Manifest, TILE_SIZE, Tile};

use super::assemble::Polygon;
use crate::build::tile_range;

/// How many bands paint at once.
///
/// A band buffer is the raster's width by [`TILE_SIZE`] of `u32` -- about
/// 50 MB for the committed box -- so four workers hold about 200 MB, the
/// same ceiling the max-pyramid pass budgets for.
const BAND_THREADS: usize = 4;

/// Paints every polygon into base-level tiles under `root`.
///
/// Returns how many tiles were written. Tiles that end up entirely
/// [`terrain_tiles::Material::Null`] are not written at all, matching the
/// convention that absence means "nothing known here".
pub fn rasterize(polygons: &[Polygon], manifest: &Manifest, root: &Path) -> Result<u64> {
    let level = MATERIAL_BASE_LEVEL;
    let (width, height) = manifest.size_texels(level);
    let (first, across, down) = tile_range(manifest, level);
    let (_, offset_x, offset_y) = manifest.tile_of_texel(level, 0, 0);
    anyhow::ensure!(
        (offset_x, offset_y) == (0, 0),
        "the raster origin does not sit on a level-{level} tile boundary"
    );

    // One global paint order, shared by every band so no seam can disagree.
    let mut order: Vec<usize> = (0..polygons.len()).collect();
    order.sort_by(|&a, &b| {
        polygons[a]
            .layer
            .cmp(&polygons[b].layer)
            .then(polygons[b].area.total_cmp(&polygons[a].area))
            .then(a.cmp(&b))
    });

    let started = std::time::Instant::now();
    log::info!(
        "painting {} polygons into {across} x {down} tiles at level {level}",
        polygons.len()
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(BAND_THREADS)
        .build()
        .context("building the thread pool")?;
    let done = AtomicU64::new(0);
    let written: u64 = pool
        .install(|| {
            (0..down)
                .into_par_iter()
                .map(|band| {
                    let built = paint_band(
                        polygons, &order, manifest, root, first, across, band, width, height,
                    );
                    let at = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if at.is_multiple_of(8) || at == u64::from(down) {
                        log::info!("painted band {at} of {down}");
                    }
                    built
                })
                .collect::<Result<Vec<u64>>>()
        })?
        .iter()
        .sum();

    log::info!("painted {written} tiles in {:.1?}", started.elapsed());
    Ok(written)
}

/// Paints one band -- tile row `band` -- and writes its non-empty tiles.
#[allow(clippy::too_many_arguments)]
fn paint_band(
    polygons: &[Polygon],
    order: &[usize],
    manifest: &Manifest,
    root: &Path,
    first: Tile,
    across: u32,
    band: u32,
    width: u32,
    height: u32,
) -> Result<u64> {
    let level = MATERIAL_BASE_LEVEL;
    let metres = manifest.metres_per_texel(level);
    let west = manifest.origin_metres[0];
    let north = manifest.origin_metres[1];

    let rows_start = band * TILE_SIZE;
    let rows = TILE_SIZE.min(height - rows_start);
    // The band's ground, for the bbox rejection below.
    let band_north = north - f64::from(rows_start) * metres;
    let band_south = band_north - f64::from(rows) * metres;

    let mut cells = vec![0u32; width as usize * rows as usize];
    let mut crossings: Vec<(u32, f64)> = Vec::new();
    for &index in order {
        let polygon = &polygons[index];
        if polygon.bbox[1] >= band_north
            || polygon.bbox[3] <= band_south
            || polygon.bbox[0] >= west + f64::from(width) * metres
            || polygon.bbox[2] <= west
        {
            continue;
        }

        // Every edge crossing of every texel-centre row this band holds.
        crossings.clear();
        for ring in &polygon.rings {
            for pair in ring.windows(2) {
                let ((x1, y1), (x2, y2)) = (pair[0], pair[1]);
                let (low, high) = (y1.min(y2), y1.max(y2));
                if low == high {
                    continue;
                }
                // Rows whose centre northing sits in `low <= y < high`. Row
                // centres run south as rows grow, so `high` bounds the first
                // row and `low` the last.
                let first_row = (((north - high) / metres - 0.5).floor() as i64 + 1)
                    .max(i64::from(rows_start));
                let last_row = (((north - low) / metres - 0.5).floor() as i64)
                    .min(i64::from(rows_start + rows) - 1);
                for row in first_row..=last_row {
                    let centre = north - (row as f64 + 0.5) * metres;
                    let x = x1 + (centre - y1) * (x2 - x1) / (y2 - y1);
                    crossings.push(((row - i64::from(rows_start)) as u32, x));
                }
            }
        }

        crossings.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let id = polygon.material.id();
        let mut at = 0;
        while at < crossings.len() {
            let row = crossings[at].0;
            let end = crossings[at..]
                .iter()
                .position(|&(r, _)| r != row)
                .map_or(crossings.len(), |offset| at + offset);
            // Pairs of crossings bound filled spans: even-odd. An odd count
            // is degenerate geometry; the unpaired crossing paints nothing.
            for span in crossings[at..end].chunks_exact(2) {
                // Both ends clamp to the raster: a span can lie entirely
                // outside it, and an unclamped start walks off the buffer.
                let from = (((span[0].1 - west) / metres - 0.5).ceil() as i64)
                    .clamp(0, i64::from(width));
                let to = (((span[1].1 - west) / metres - 0.5).ceil() as i64)
                    .clamp(from, i64::from(width));
                let line = row as usize * width as usize;
                for cell in &mut cells[line + from as usize..line + to as usize] {
                    *cell = id;
                }
            }
            at = end;
        }
    }

    write_band(manifest, root, first, across, band, width, rows, &cells)
}

/// Cuts a painted band into tiles and writes the ones holding anything.
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
    use terrain_tiles::read::read_material_tile;
    use terrain_tiles::{Material, TileGrid};

    /// Two tiles across, one down, at level 2: 1024 x 512 texels of 4 m.
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
            extent_texels: [4096, 2048],
            bands: 1,
            nodata: 0.0,
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "terrain-process-rasterize-{}-{name}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    /// A closed rectangle ring in metres.
    fn rectangle(west: f64, south: f64, east: f64, north: f64) -> Vec<(f64, f64)> {
        vec![
            (west, south),
            (east, south),
            (east, north),
            (west, north),
            (west, south),
        ]
    }

    fn texel(root: &Path, manifest: &Manifest, column: i64, row: i64) -> u32 {
        let (tile, x, y) = manifest.tile_of_texel(MATERIAL_BASE_LEVEL, column, row);
        let grid: TileGrid = manifest.grid();
        let path = grid.tile_path(root, MATERIAL_BASE_LEVEL, tile);
        match read_material_tile(&path).expect("failed to read") {
            Some(values) => values[(y * TILE_SIZE + x) as usize],
            None => 0,
        }
    }

    /// The ground is at northings 0 down to -2048; a polygon over the first
    /// texels paints exactly the texels whose centres it covers.
    #[test]
    fn a_polygon_paints_the_texels_whose_centres_it_covers() {
        let manifest = manifest();
        let root = temp_root("centres");
        // Covers x in [0, 40), y in [-8, 0): columns 0..10, rows 0..2.
        let polygon = Polygon::new(
            Material::Lake,
            vec![rectangle(0.0, -8.0, 40.0, 0.0)],
        )
        .expect("a ring");
        let written = rasterize(&[polygon], &manifest, &root).expect("failed to paint");
        assert_eq!(written, 1, "one tile holds paint");

        assert_eq!(texel(&root, &manifest, 0, 0), Material::Lake.id());
        assert_eq!(texel(&root, &manifest, 9, 1), Material::Lake.id());
        assert_eq!(texel(&root, &manifest, 10, 0), 0, "east of the edge");
        assert_eq!(texel(&root, &manifest, 0, 2), 0, "south of the edge");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A lake inside a forest survives painting because water's layer paints
    /// later, whichever order the polygons arrived in.
    #[test]
    fn higher_layers_paint_over_lower_ones() {
        let manifest = manifest();
        let root = temp_root("layers");
        let forest = Polygon::new(
            Material::ForestUnknown,
            vec![rectangle(0.0, -400.0, 400.0, 0.0)],
        )
        .expect("a ring");
        let lake = Polygon::new(Material::Lake, vec![rectangle(100.0, -300.0, 300.0, -100.0)])
            .expect("a ring");
        // Deliberately hand the lake over first.
        rasterize(&[lake, forest], &manifest, &root).expect("failed to paint");

        assert_eq!(texel(&root, &manifest, 1, 1), Material::ForestUnknown.id());
        assert_eq!(texel(&root, &manifest, 50, 50), Material::Lake.id());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Within a layer the smaller polygon paints last: a clearing inside a
    /// wood shows, both being vegetation.
    #[test]
    fn within_a_layer_the_smaller_area_wins() {
        let manifest = manifest();
        let root = temp_root("area");
        let wood = Polygon::new(
            Material::ForestUnknown,
            vec![rectangle(0.0, -400.0, 400.0, 0.0)],
        )
        .expect("a ring");
        let clearing = Polygon::new(Material::Grass, vec![rectangle(100.0, -300.0, 300.0, -100.0)])
            .expect("a ring");
        rasterize(&[wood, clearing], &manifest, &root).expect("failed to paint");

        assert_eq!(texel(&root, &manifest, 1, 1), Material::ForestUnknown.id());
        assert_eq!(texel(&root, &manifest, 50, 50), Material::Grass.id());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A hole is a ring inside a ring: even-odd leaves it unpainted and the
    /// layer below shows through.
    #[test]
    fn a_hole_leaves_what_was_painted_below() {
        let manifest = manifest();
        let root = temp_root("hole");
        // A zone underneath (layer 1) and a holed wood on top (layer 2): the
        // hole must show the zone, not the wood and not Null.
        let zone = Polygon::new(
            Material::Residential,
            vec![rectangle(0.0, -400.0, 400.0, 0.0)],
        )
        .expect("a ring");
        let wood_with_hole = Polygon::new(
            Material::ForestUnknown,
            vec![
                rectangle(0.0, -400.0, 400.0, 0.0),
                rectangle(100.0, -300.0, 300.0, -100.0),
            ],
        )
        .expect("two rings");
        rasterize(&[zone, wood_with_hole], &manifest, &root).expect("failed to paint");

        assert_eq!(texel(&root, &manifest, 1, 1), Material::ForestUnknown.id());
        assert_eq!(
            texel(&root, &manifest, 50, 50),
            Material::Residential.id(),
            "the hole shows the layer below"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Tiles the paint never reaches are not written at all.
    #[test]
    fn untouched_tiles_are_absent() {
        let manifest = manifest();
        let root = temp_root("absent");
        let polygon = Polygon::new(Material::Lake, vec![rectangle(0.0, -40.0, 40.0, 0.0)])
            .expect("a ring");
        rasterize(&[polygon], &manifest, &root).expect("failed to paint");

        let grid = manifest.grid();
        let (tile, _, _) = manifest.tile_of_texel(MATERIAL_BASE_LEVEL, 600, 0);
        assert!(
            read_material_tile(&grid.tile_path(&root, MATERIAL_BASE_LEVEL, tile))
                .expect("readable")
                .is_none(),
            "the second tile holds nothing and should not exist"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
