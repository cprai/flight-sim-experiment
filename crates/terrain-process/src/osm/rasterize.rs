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
//!
//! Strokes -- roads, rail lines, runways -- are not polygons and paint by a
//! different rule: every texel whose centre lies within half the stroke's
//! width of its polyline, segment by segment. A segment's reach is a
//! capsule, which is convex, so each texel row it touches is one clean span.
//! Strokes paint after the ground layers and before wetland and open water,
//! so a road embankment crosses a park but never dams a river: the water,
//! painting later, cuts it where they overlap, which is also what keeps an
//! unbridged culvert crossing from reading as a weir.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use rayon::prelude::*;
use terrain_tiles::write::{TilePlacement, write_material_tile};
use terrain_tiles::{MATERIAL_BASE_LEVEL, Manifest, TILE_SIZE, Tile};

use super::assemble::{Polygon, Stroke};
use crate::build::tile_range;

/// The first layer that paints *over* the strokes: wetland, then open water.
const LAYERS_ABOVE_STROKES: u8 = 4;

/// How many bands paint at once.
///
/// A band buffer is the raster's width by [`TILE_SIZE`] of `u32` -- about
/// 50 MB for the committed box -- so four workers hold about 200 MB, the
/// same ceiling the max-pyramid pass budgets for.
const BAND_THREADS: usize = 4;

/// Paints every polygon and stroke into base-level tiles under `root`.
///
/// Returns how many tiles were written. Tiles that end up entirely
/// [`terrain_materials::Material::Null`] are not written at all, matching the
/// convention that absence means "nothing known here".
pub fn rasterize(
    polygons: &[Polygon],
    strokes: &[Stroke],
    manifest: &Manifest,
    root: &Path,
) -> Result<u64> {
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

    // Everything below wetland paints first, then the strokes, then the
    // water; the order vector is already sorted by layer, so the boundary
    // is a partition point.
    let split = order.partition_point(|&index| polygons[index].layer < LAYERS_ABOVE_STROKES);

    let started = std::time::Instant::now();
    log::info!(
        "painting {} polygons and {} strokes into {across} x {down} tiles at level {level}",
        polygons.len(),
        strokes.len()
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
                        polygons, &order, split, strokes, manifest, root, first, across, band,
                        width, height,
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

/// One band's place on the raster, shared by the polygon and stroke painters.
struct Band {
    metres: f64,
    /// The raster's west edge and north edge, in grid metres.
    west: f64,
    north: f64,
    /// The band's first raster row and how many rows it holds.
    rows_start: u32,
    rows: u32,
    /// The raster's full width in texels.
    width: u32,
    /// The band's own ground, for bbox rejection.
    band_north: f64,
    band_south: f64,
}

/// Paints one band -- tile row `band` -- and writes its non-empty tiles.
#[allow(clippy::too_many_arguments)]
fn paint_band(
    polygons: &[Polygon],
    order: &[usize],
    split: usize,
    strokes: &[Stroke],
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
    let north = manifest.origin_metres[1];
    let rows_start = band * TILE_SIZE;
    let rows = TILE_SIZE.min(height - rows_start);
    let band_north = north - f64::from(rows_start) * metres;
    let at = Band {
        metres,
        west: manifest.origin_metres[0],
        north,
        rows_start,
        rows,
        width,
        band_north,
        band_south: band_north - f64::from(rows) * metres,
    };

    let mut cells = vec![0u32; width as usize * rows as usize];
    let mut crossings: Vec<(u32, f64)> = Vec::new();
    for &index in &order[..split] {
        paint_polygon(&polygons[index], &at, &mut cells, &mut crossings);
    }
    for stroke in strokes {
        paint_stroke(stroke, &at, &mut cells);
    }
    for &index in &order[split..] {
        paint_polygon(&polygons[index], &at, &mut cells, &mut crossings);
    }

    write_band(manifest, root, first, across, band, width, rows, &cells)
}

/// Paints one polygon's texels into a band buffer.
fn paint_polygon(polygon: &Polygon, at: &Band, cells: &mut [u32], crossings: &mut Vec<(u32, f64)>) {
    let (metres, west, north) = (at.metres, at.west, at.north);
    let (rows_start, rows, width) = (at.rows_start, at.rows, at.width);
    {
        if polygon.bbox[1] >= at.band_north
            || polygon.bbox[3] <= at.band_south
            || polygon.bbox[0] >= west + f64::from(width) * metres
            || polygon.bbox[2] <= west
        {
            return;
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
                let first_row =
                    (((north - high) / metres - 0.5).floor() as i64 + 1).max(i64::from(rows_start));
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
        let mut cursor = 0;
        while cursor < crossings.len() {
            let row = crossings[cursor].0;
            let end = crossings[cursor..]
                .iter()
                .position(|&(r, _)| r != row)
                .map_or(crossings.len(), |offset| cursor + offset);
            // Pairs of crossings bound filled spans: even-odd. An odd count
            // is degenerate geometry; the unpaired crossing paints nothing.
            for span in crossings[cursor..end].chunks_exact(2) {
                // Both ends clamp to the raster: a span can lie entirely
                // outside it, and an unclamped start walks off the buffer.
                let from =
                    (((span[0].1 - west) / metres - 0.5).ceil() as i64).clamp(0, i64::from(width));
                let to = (((span[1].1 - west) / metres - 0.5).ceil() as i64)
                    .clamp(from, i64::from(width));
                let line = row as usize * width as usize;
                for cell in &mut cells[line + from as usize..line + to as usize] {
                    *cell = id;
                }
            }
            cursor = end;
        }
    }
}

/// Paints one stroke's texels into a band buffer: every texel whose centre
/// lies within half the width of the polyline.
///
/// Segment by segment. A segment's reach is a capsule -- a rectangle with a
/// disc on each end -- and a capsule is convex, so its intersection with a
/// texel-centre row is a single span whose ends come from whichever of the
/// three parts reaches furthest at that northing. The discs also make the
/// joints between segments seamless, whatever angle they meet at.
fn paint_stroke(stroke: &Stroke, at: &Band, cells: &mut [u32]) {
    let (metres, west, north) = (at.metres, at.west, at.north);
    let radius = stroke.width * 0.5;
    if stroke.bbox[1] - radius >= at.band_north
        || stroke.bbox[3] + radius <= at.band_south
        || stroke.bbox[0] - radius >= west + f64::from(at.width) * metres
        || stroke.bbox[2] + radius <= west
    {
        return;
    }

    let id = stroke.material.id();
    for pair in stroke.points.windows(2) {
        let ((x1, y1), (x2, y2)) = (pair[0], pair[1]);
        let (dx, dy) = (x2 - x1, y2 - y1);
        let length = dx.hypot(dy);
        if length == 0.0 {
            // Duplicate consecutive nodes; the neighbours' caps cover it.
            continue;
        }
        // Unit normal, for the rectangle's corners.
        let (nx, ny) = (-dy / length, dx / length);

        // Rows whose centre northing the capsule can reach.
        let (low, high) = (y1.min(y2) - radius, y1.max(y2) + radius);
        let first_row =
            (((north - high) / metres - 0.5).ceil() as i64).max(i64::from(at.rows_start));
        let last_row = (((north - low) / metres - 0.5).floor() as i64)
            .min(i64::from(at.rows_start + at.rows) - 1);
        for row in first_row..=last_row {
            let centre = north - (row as f64 + 0.5) * metres;
            let mut span: Option<(f64, f64)> = None;
            let mut reach = |x: f64| {
                span = Some(match span {
                    None => (x, x),
                    Some((lo, hi)) => (lo.min(x), hi.max(x)),
                });
            };

            // The end discs' chords at this northing.
            for (px, py) in [(x1, y1), (x2, y2)] {
                let rise = centre - py;
                if rise.abs() <= radius {
                    let half = (radius * radius - rise * rise).sqrt();
                    reach(px - half);
                    reach(px + half);
                }
            }
            // The rectangle: where its four edges cross this northing.
            let corners = [
                (x1 + nx * radius, y1 + ny * radius),
                (x2 + nx * radius, y2 + ny * radius),
                (x2 - nx * radius, y2 - ny * radius),
                (x1 - nx * radius, y1 - ny * radius),
            ];
            for edge in 0..4 {
                let (ax, ay) = corners[edge];
                let (bx, by) = corners[(edge + 1) % 4];
                if (ay - centre) * (by - centre) <= 0.0 && ay != by {
                    reach(ax + (centre - ay) * (bx - ax) / (by - ay));
                }
            }

            let Some((lo, hi)) = span else { continue };
            let from = (((lo - west) / metres - 0.5).ceil() as i64).clamp(0, i64::from(at.width));
            let to = ((((hi - west) / metres - 0.5).floor() + 1.0) as i64)
                .clamp(from, i64::from(at.width));
            let line = (row - i64::from(at.rows_start)) as usize * at.width as usize;
            for cell in &mut cells[line + from as usize..line + to as usize] {
                *cell = id;
            }
        }
    }
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
    use terrain_materials::Material;
    use terrain_tiles::TileGrid;
    use terrain_tiles::read::read_material_tile;

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
        let polygon =
            Polygon::new(Material::Lake, vec![rectangle(0.0, -8.0, 40.0, 0.0)]).expect("a ring");
        let written = rasterize(&[polygon], &[], &manifest, &root).expect("failed to paint");
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
        let lake = Polygon::new(
            Material::Lake,
            vec![rectangle(100.0, -300.0, 300.0, -100.0)],
        )
        .expect("a ring");
        // Deliberately hand the lake over first.
        rasterize(&[lake, forest], &[], &manifest, &root).expect("failed to paint");

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
        let clearing = Polygon::new(
            Material::Grass,
            vec![rectangle(100.0, -300.0, 300.0, -100.0)],
        )
        .expect("a ring");
        rasterize(&[wood, clearing], &[], &manifest, &root).expect("failed to paint");

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
        rasterize(&[zone, wood_with_hole], &[], &manifest, &root).expect("failed to paint");

        assert_eq!(texel(&root, &manifest, 1, 1), Material::ForestUnknown.id());
        assert_eq!(
            texel(&root, &manifest, 50, 50),
            Material::Residential.id(),
            "the hole shows the layer below"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A stroke paints the texels whose 4 m centres lie within half its
    /// width of the line, rounded ends included, and no others.
    #[test]
    fn a_stroke_paints_a_capsule_of_its_width() {
        let manifest = manifest();
        let root = temp_root("stroke");
        // A 16 m road along northing -100, so half-width 8: row centres run
        // -(4r + 2), and rows 23 (-94) through 26 (-106) are within reach.
        let stroke = Stroke {
            material: Material::Paved,
            width: 16.0,
            points: vec![(100.0, -100.0), (200.0, -100.0)],
            bbox: [100.0, -100.0, 200.0, -100.0],
        };
        rasterize(&[], &[stroke], &manifest, &root).expect("failed to paint");

        assert_eq!(
            texel(&root, &manifest, 37, 24),
            Material::Paved.id(),
            "on the line"
        );
        assert_eq!(
            texel(&root, &manifest, 37, 23),
            Material::Paved.id(),
            "6 m north"
        );
        assert_eq!(
            texel(&root, &manifest, 37, 26),
            Material::Paved.id(),
            "6 m south"
        );
        assert_eq!(
            texel(&root, &manifest, 37, 22),
            0,
            "10 m north is past the edge"
        );
        assert_eq!(
            texel(&root, &manifest, 37, 27),
            0,
            "10 m south is past the edge"
        );
        // The cap disc rounds the end: the centreline reach is x in
        // [92, 208], and column centres run 4c + 2.
        assert_eq!(
            texel(&root, &manifest, 23, 24),
            Material::Paved.id(),
            "west cap"
        );
        assert_eq!(
            texel(&root, &manifest, 51, 24),
            Material::Paved.id(),
            "east cap"
        );
        assert_eq!(texel(&root, &manifest, 22, 24), 0, "beyond the west cap");
        assert_eq!(texel(&root, &manifest, 52, 24), 0, "beyond the east cap");
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Water paints after the strokes: a river crossed by an untagged road
    /// stays a river, cutting the road where they overlap.
    #[test]
    fn water_paints_over_a_stroke() {
        let manifest = manifest();
        let root = temp_root("stroke-water");
        let river = Polygon::new(Material::River, vec![rectangle(140.0, -400.0, 160.0, 0.0)])
            .expect("a ring");
        let road = Stroke {
            material: Material::Paved,
            width: 16.0,
            points: vec![(100.0, -100.0), (200.0, -100.0)],
            bbox: [100.0, -100.0, 200.0, -100.0],
        };
        rasterize(&[river], &[road], &manifest, &root).expect("failed to paint");

        assert_eq!(
            texel(&root, &manifest, 30, 24),
            Material::Paved.id(),
            "west of the river"
        );
        assert_eq!(
            texel(&root, &manifest, 37, 24),
            Material::River.id(),
            "the crossing"
        );
        assert_eq!(
            texel(&root, &manifest, 45, 24),
            Material::Paved.id(),
            "east of the river"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// Tiles the paint never reaches are not written at all.
    #[test]
    fn untouched_tiles_are_absent() {
        let manifest = manifest();
        let root = temp_root("absent");
        let polygon =
            Polygon::new(Material::Lake, vec![rectangle(0.0, -40.0, 40.0, 0.0)]).expect("a ring");
        rasterize(&[polygon], &[], &manifest, &root).expect("failed to paint");

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
