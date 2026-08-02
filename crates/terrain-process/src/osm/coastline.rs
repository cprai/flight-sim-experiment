//! The sea, which OpenStreetMap implies rather than maps.
//!
//! There is no ocean polygon in OSM. The sea is defined by `natural=coastline`
//! ways -- directed so that land is on the left and water on the right -- and
//! anything on the water side of them, all the way out, is ocean. To paint
//! Howe Sound and the Strait of Georgia at all, this module has to turn that
//! implication into rings: clip the coastline to the raster's rectangle and
//! close it along the rectangle's edges, walking in the direction that keeps
//! water on the right, so that each closed cycle encloses sea.
//!
//! The walk direction is the crux. With north up, the perimeter is walked
//! clockwise -- north edge eastward, east edge southward, south edge westward,
//! west edge northward -- because that keeps the rectangle's interior on the
//! right, the same side the coastline keeps its water. A chain that exits the
//! rectangle is continued along the perimeter to the next chain that enters,
//! and the cycle closes around water, however many chains and corners it takes.
//!
//! One ugliness is load-bearing. Geofabrik clips the extract to the region
//! polygon, which crosses this raster's interior, so the mainland coastline
//! simply stops mid-sea where the region ends -- a dangling end strictly
//! inside the rectangle. Such an end is extended straight to the nearest
//! perimeter point: invented coastline, but it lies where the extract has no
//! data at all, and without it the mainland chain cannot close and the whole
//! ocean vanishes. The extension is bounded; a dangle further than
//! [`DANGLE_METRES`] from the edge is a sign something else is wrong, and is
//! dropped with a warning instead.
//!
//! What this cannot know: land whose coastline lives in a *different* extract.
//! Nanaimo and Gabriola Island sit inside this raster, but their shoreline is
//! in the `vancouver-island` extract, so nothing here says they are land and
//! the ocean fill paints over them. The fix is merging a second extract,
//! deliberately not attempted yet.

use terrain_tiles::{Manifest, Material};

use super::assemble::{Polygon, resolve, stitch};
use super::read::Extract;

/// How far a dangling coastline end may be extended to reach the perimeter.
///
/// The region boundary runs mid-strait, so real dangles sit tens of
/// kilometres inside the west edge; 25 km covers the strait's half-width with
/// room to spare while still catching a chain that ends somewhere absurd.
pub const DANGLE_METRES: f64 = 25_000.0;

/// A point this close to a rectangle edge counts as on it.
///
/// Clipped crossings are computed by interpolation and land within rounding
/// of the edge; a real coastline vertex a centimetre from the exact raster
/// boundary is indistinguishable from a crossing and harmlessly treated as
/// one.
const EDGE_EPSILON: f64 = 0.01;

/// The raster's ground, as the rectangle the sea is clipped to.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl Rect {
    /// The ground a product's manifest covers.
    pub fn of_manifest(manifest: &Manifest) -> Self {
        let west = manifest.origin_metres[0];
        let north = manifest.origin_metres[1];
        Self {
            west,
            south: north - f64::from(manifest.extent_texels[1]) * manifest.base_metres_per_texel,
            east: west + f64::from(manifest.extent_texels[0]) * manifest.base_metres_per_texel,
            north,
        }
    }

    fn contains(&self, (x, y): (f64, f64)) -> bool {
        (self.west..=self.east).contains(&x) && (self.south..=self.north).contains(&y)
    }

    fn on_perimeter(&self, (x, y): (f64, f64)) -> bool {
        self.contains((x, y))
            && ((x - self.west).abs() < EDGE_EPSILON
                || (self.east - x).abs() < EDGE_EPSILON
                || (y - self.south).abs() < EDGE_EPSILON
                || (self.north - y).abs() < EDGE_EPSILON)
    }

    fn perimeter_length(&self) -> f64 {
        2.0 * ((self.east - self.west) + (self.north - self.south))
    }

    /// Where a perimeter point sits on the clockwise walk from the north-west
    /// corner: north edge eastward, east edge southward, south edge westward,
    /// west edge northward. This order keeps the interior -- the water a
    /// cycle encloses -- on the right, matching the coastline convention.
    fn perimeter_t(&self, (x, y): (f64, f64)) -> f64 {
        let (width, height) = (self.east - self.west, self.north - self.south);
        if (self.north - y).abs() < EDGE_EPSILON {
            x - self.west
        } else if (self.east - x).abs() < EDGE_EPSILON {
            width + (self.north - y)
        } else if (y - self.south).abs() < EDGE_EPSILON {
            width + height + (self.east - x)
        } else {
            2.0 * width + height + (y - self.south)
        }
    }

    /// The corners passed walking clockwise from `from` to `to`, exclusive.
    fn corners_between(&self, from: f64, to: f64) -> Vec<(f64, f64)> {
        let (width, height) = (self.east - self.west, self.north - self.south);
        let corners = [
            (width, (self.east, self.north)),
            (width + height, (self.east, self.south)),
            (2.0 * width + height, (self.west, self.south)),
            (self.perimeter_length(), (self.west, self.north)),
        ];
        let span = if to > from {
            to - from
        } else {
            to - from + self.perimeter_length()
        };
        let mut passed: Vec<(f64, (f64, f64))> = corners
            .iter()
            .map(|&(t, point)| {
                let ahead = if t > from {
                    t - from
                } else {
                    t - from + self.perimeter_length()
                };
                (ahead, point)
            })
            .filter(|&(ahead, _)| ahead < span)
            .collect();
        passed.sort_by(|a, b| a.0.total_cmp(&b.0));
        passed.into_iter().map(|(_, point)| point).collect()
    }
}

/// One coastline piece inside the rectangle, endpoints on its perimeter.
struct Crossing {
    points: Vec<(f64, f64)>,
    /// Perimeter positions of the first and last point.
    entry: f64,
    exit: f64,
}

/// The ocean as one even-odd polygon, or `None` when there is no coastline
/// to imply one (a landlocked extract, or nothing survived validation).
pub fn ocean(extract: &Extract, rect: Rect) -> Option<Polygon> {
    let stitched = stitch(&extract.coastlines);

    // Resolve to grid metres. A chain touching a node the extract lost is
    // split at the gap rather than dropped; each run stands alone.
    let mut rings: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut chains: Vec<Vec<(f64, f64)>> = Vec::new();
    for chain in &stitched.closed {
        match resolve(chain, &extract.nodes) {
            Some(ring) => rings.push(ring),
            None => chains.extend(resolve_runs(chain, extract)),
        }
    }
    for chain in &stitched.open {
        match resolve(chain, &extract.nodes) {
            Some(points) => chains.push(points),
            None => chains.extend(resolve_runs(chain, extract)),
        }
    }

    // Rings entirely inside pass through as islands; rings that cross the
    // boundary are opened and clipped like any chain; rings entirely outside
    // have nothing to say about this ground.
    let mut islands = Vec::new();
    for ring in rings {
        if ring.iter().all(|&point| rect.contains(point)) {
            islands.push(ring);
        } else if ring.iter().any(|&point| rect.contains(point)) {
            chains.push(ring);
        }
    }

    // Clip to the rectangle and put every surviving endpoint on the
    // perimeter, extending region-clip dangles or dropping hopeless chains.
    let mut crossings: Vec<Crossing> = Vec::new();
    let mut dropped = 0u64;
    let mut extended = 0u64;
    for chain in &chains {
        for mut piece in clip_chain(chain, rect) {
            let mut ok = true;
            for end in [false, true] {
                let point = if end {
                    *piece.last().expect("clipped pieces are never empty")
                } else {
                    piece[0]
                };
                if rect.on_perimeter(point) {
                    continue;
                }
                match nearest_perimeter(point, rect) {
                    Some(projected) => {
                        extended += 1;
                        if end {
                            piece.push(projected);
                        } else {
                            piece.insert(0, projected);
                        }
                    }
                    None => {
                        ok = false;
                    }
                }
            }
            if !ok || piece.len() < 2 {
                dropped += 1;
                continue;
            }
            let entry = rect.perimeter_t(piece[0]);
            let exit = rect.perimeter_t(*piece.last().expect("checked"));
            crossings.push(Crossing {
                points: piece,
                entry,
                exit,
            });
        }
    }
    if extended > 0 {
        log::info!(
            "extended {extended} dangling coastline ends to the raster edge \
             (the region clip ends mid-sea)"
        );
    }
    if dropped > 0 {
        log::warn!("dropped {dropped} coastline pieces that could not reach the raster edge");
    }

    if crossings.is_empty() {
        if !islands.is_empty() {
            // An island with no surrounding sea would fill as sea.
            log::warn!(
                "no coastline crosses the raster edge; dropping {} closed rings \
                 rather than guessing which side is water",
                islands.len()
            );
        }
        return None;
    }

    // Walking the perimeter from every exit must reach an entry: two exits in
    // a row means two chains claim contradictory water and the fill would be
    // garbage. The data does contain such pieces -- fragments of coastline
    // the region clip stranded, spurs around islands whose better half is in
    // another extract -- so a violation is repaired by dropping the shorter
    // of the two offending pieces (fragments are short; the mainland is not)
    // and validating again. Only a repair that will not converge aborts.
    let mut alive = vec![true; crossings.len()];
    let mut order: Vec<(f64, usize, bool)>;
    loop {
        order = Vec::new();
        for (index, crossing) in crossings.iter().enumerate() {
            if alive[index] {
                order.push((crossing.entry, index, true));
                order.push((crossing.exit, index, false));
            }
        }
        if order.is_empty() {
            log::error!("no coastline pieces survived validation; the sea cannot be filled");
            return None;
        }
        order.sort_by(|a, b| a.0.total_cmp(&b.0));

        let violation = (0..order.len()).find(|&pair| {
            order[pair].2 == order[(pair + 1) % order.len()].2
        });
        let Some(pair) = violation else {
            break;
        };
        let metres = |index: usize| -> f64 {
            crossings[index]
                .points
                .windows(2)
                .map(|pair| (pair[1].0 - pair[0].0).hypot(pair[1].1 - pair[0].1))
                .sum()
        };
        let (a, b) = (order[pair].1, order[(pair + 1) % order.len()].1);
        let drop = if metres(a) <= metres(b) { a } else { b };
        let ends = (
            crossings[drop].points[0],
            *crossings[drop].points.last().expect("never empty"),
        );
        log::warn!(
            "coastline crossings do not alternate near t = {:.0} of {:.0}; dropping \
             the {:.0} m piece from ({:.0}, {:.0}) to ({:.0}, {:.0})",
            order[pair].0,
            rect.perimeter_length(),
            metres(drop),
            ends.0.0,
            ends.0.1,
            ends.1.0,
            ends.1.1,
        );
        alive[drop] = false;
    }

    // Link: follow a chain in, walk the perimeter clockwise from its exit to
    // the next entry, follow that chain, until the cycle closes.
    let next_entry: std::collections::HashMap<usize, usize> = order
        .iter()
        .enumerate()
        .filter(|(_, entry)| !entry.2)
        .map(|(position, &(_, index, _))| {
            let (_, following, _) = order[(position + 1) % order.len()];
            (index, following)
        })
        .collect();

    let mut sea_rings: Vec<Vec<(f64, f64)>> = Vec::new();
    // Dropped pieces are already "visited": no cycle may pick them up.
    let mut visited: Vec<bool> = alive.iter().map(|&kept| !kept).collect();
    for start in 0..crossings.len() {
        if visited[start] {
            continue;
        }
        let mut ring: Vec<(f64, f64)> = Vec::new();
        let mut current = start;
        loop {
            visited[current] = true;
            ring.extend_from_slice(&crossings[current].points);
            let following = next_entry[&current];
            ring.extend(
                rect.corners_between(crossings[current].exit, crossings[following].entry),
            );
            if following == start {
                break;
            }
            current = following;
        }
        ring.push(ring[0]);
        sea_rings.push(ring);
    }

    // Islands hold only where sea surrounds them; one whose sea chain was
    // dropped would invert and fill as water.
    let (kept, orphaned): (Vec<_>, Vec<_>) = islands
        .into_iter()
        .partition(|ring| inside(&sea_rings, ring[0]));
    if !orphaned.is_empty() {
        log::warn!(
            "dropped {} coastline rings that no sea surrounds",
            orphaned.len()
        );
    }

    log::info!(
        "closed the sea: {} rings along the raster edge, {} islands",
        sea_rings.len(),
        kept.len()
    );
    sea_rings.extend(kept);
    Polygon::new(Material::Ocean, sea_rings)
}

/// Resolves a chain that has missing nodes into its resolvable runs.
fn resolve_runs(chain: &[i64], extract: &Extract) -> Vec<Vec<(f64, f64)>> {
    let mut runs = Vec::new();
    let mut run = Vec::new();
    for &id in chain {
        match extract.nodes.get(id) {
            Some(point) => run.push(point),
            None => {
                if run.len() >= 2 {
                    runs.push(std::mem::take(&mut run));
                } else {
                    run.clear();
                }
            }
        }
    }
    if run.len() >= 2 {
        runs.push(run);
    }
    runs
}

/// Clips a polyline to the rectangle, splitting where it leaves and returns.
fn clip_chain(chain: &[(f64, f64)], rect: Rect) -> Vec<Vec<(f64, f64)>> {
    let mut pieces = Vec::new();
    let mut current: Vec<(f64, f64)> = Vec::new();
    for pair in chain.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let Some((t0, t1)) = clip_segment(a, b, rect) else {
            continue;
        };
        let lerp = |t: f64| (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
        let (pa, pb) = (lerp(t0), lerp(t1));
        if current.is_empty() {
            current.push(pa);
        }
        current.push(pb);
        if t1 < 1.0 {
            // The segment exits; the piece is complete.
            pieces.push(std::mem::take(&mut current));
        }
    }
    if current.len() >= 2 {
        pieces.push(current);
    }
    pieces
}

/// Liang-Barsky: the parameter span of a segment inside the rectangle.
fn clip_segment(a: (f64, f64), b: (f64, f64), rect: Rect) -> Option<(f64, f64)> {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    for (p, q) in [
        (-dx, a.0 - rect.west),
        (dx, rect.east - a.0),
        (-dy, a.1 - rect.south),
        (dy, rect.north - a.1),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
            continue;
        }
        let r = q / p;
        if p < 0.0 {
            t0 = t0.max(r);
        } else {
            t1 = t1.min(r);
        }
    }
    (t0 <= t1).then_some((t0, t1))
}

/// The nearest perimeter point within [`DANGLE_METRES`], for a chain end the
/// region clip left hanging inside the rectangle.
fn nearest_perimeter((x, y): (f64, f64), rect: Rect) -> Option<(f64, f64)> {
    let candidates = [
        (x - rect.west, (rect.west, y)),
        (rect.east - x, (rect.east, y)),
        (y - rect.south, (x, rect.south)),
        (rect.north - y, (x, rect.north)),
    ];
    let (distance, point) = candidates
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .expect("four candidates");
    (distance <= DANGLE_METRES).then_some(point)
}

/// Whether a point is inside a ring set, even-odd, by a ray cast east.
fn inside(rings: &[Vec<(f64, f64)>], (x, y): (f64, f64)) -> bool {
    let mut crossings = 0;
    for ring in rings {
        for pair in ring.windows(2) {
            let ((x1, y1), (x2, y2)) = (pair[0], pair[1]);
            if (y1 > y) != (y2 > y) {
                let cross_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1);
                if cross_x > x {
                    crossings += 1;
                }
            }
        }
    }
    crossings % 2 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osm::read::Nodes;
    use terrain_tiles::Material;

    fn rect() -> Rect {
        Rect {
            west: 0.0,
            south: 0.0,
            east: 10.0,
            north: 10.0,
        }
    }

    /// Builds an extract holding only coastline geometry, on a synthetic
    /// node table where node id `n` sits at the `n`th supplied point.
    fn extract(points: &[(f64, f64)], coastlines: Vec<Vec<i64>>) -> Extract {
        Extract {
            areas: Vec::new(),
            relations: Vec::new(),
            coastlines,
            members: std::collections::HashMap::new(),
            nodes: Nodes::for_tests((0..points.len() as i64).collect(), points.to_vec()),
        }
    }

    /// A west-to-east chain with water on its right (south): the sea ring
    /// must close around the southern half and leave the north dry.
    #[test]
    fn one_crossing_chain_closes_around_the_water_side() {
        let extract = extract(&[(-2.0, 5.0), (12.0, 5.0)], vec![vec![0, 1]]);
        let sea = ocean(&extract, rect()).expect("there should be sea");
        assert_eq!(sea.material, Material::Ocean);
        assert_eq!(sea.rings.len(), 1);
        assert!(inside(&sea.rings, (5.0, 2.0)), "south of the coast is sea");
        assert!(!inside(&sea.rings, (5.0, 8.0)), "north of the coast is land");
    }

    /// The same coast drawn the other way round encloses the north instead:
    /// direction is the only thing that says which side is water.
    #[test]
    fn reversing_the_chain_floods_the_other_side() {
        let extract = extract(&[(12.0, 5.0), (-2.0, 5.0)], vec![vec![0, 1]]);
        let sea = ocean(&extract, rect()).expect("there should be sea");
        assert!(!inside(&sea.rings, (5.0, 2.0)));
        assert!(inside(&sea.rings, (5.0, 8.0)));
    }

    /// A chain that enters and leaves through the same edge takes a bite out
    /// of the land rather than sweeping the whole rectangle.
    #[test]
    fn a_same_edge_bite_stays_a_bite() {
        // In through the north edge at x = 7, around the pocket, back out at
        // x = 3. Walking south the right hand points west, so the pocket
        // between the arms is the water.
        let extract = extract(
            &[(7.0, 12.0), (7.0, 6.0), (3.0, 6.0), (3.0, 12.0)],
            vec![vec![0, 1, 2, 3]],
        );
        let sea = ocean(&extract, rect()).expect("there should be sea");
        assert!(inside(&sea.rings, (5.0, 8.0)), "inside the bite");
        assert!(!inside(&sea.rings, (5.0, 4.0)), "south of the bite");
        assert!(!inside(&sea.rings, (1.0, 8.0)), "west of the bite");
    }

    /// The same walk the other way round is a peninsula: the pocket is land
    /// and the sea wraps around the outside of it.
    #[test]
    fn the_reverse_walk_is_a_peninsula() {
        let extract = extract(
            &[(3.0, 12.0), (3.0, 6.0), (7.0, 6.0), (7.0, 12.0)],
            vec![vec![0, 1, 2, 3]],
        );
        let sea = ocean(&extract, rect()).expect("there should be sea");
        assert!(!inside(&sea.rings, (5.0, 8.0)), "the pocket is land");
        assert!(inside(&sea.rings, (5.0, 4.0)), "south of it is sea");
        assert!(inside(&sea.rings, (1.0, 8.0)), "west of it is sea");
    }

    #[test]
    fn an_island_ring_inside_the_sea_is_kept_and_one_on_land_is_dropped() {
        // Coast at y=5, water south. An island ring in the south, a pond-like
        // ring in the north where there is no sea around it.
        let extract = extract(
            &[
                (-2.0, 5.0),
                (12.0, 5.0),
                (4.0, 2.0),
                (6.0, 2.0),
                (5.0, 3.0),
                (4.0, 8.0),
                (6.0, 8.0),
                (5.0, 9.0),
            ],
            vec![vec![0, 1], vec![2, 3, 4, 2], vec![5, 6, 7, 5]],
        );
        let sea = ocean(&extract, rect()).expect("there should be sea");
        assert_eq!(sea.rings.len(), 2, "the sea and one island");
        assert!(
            !inside(&sea.rings, (5.0, 2.3)),
            "the island is not sea even though the strip around it is"
        );
        assert!(inside(&sea.rings, (3.0, 2.1)), "beside the island is sea");
    }

    /// The mainland chain really does end mid-rectangle in this project's
    /// extract; the invented segment to the nearest edge is what lets the
    /// sea close at all.
    #[test]
    fn a_dangling_end_is_extended_to_the_nearest_edge() {
        // Enters from the east heading west, water right (north), and stops
        // dead in the middle.
        let extract = extract(&[(12.0, 5.0), (5.0, 5.0)], vec![vec![0, 1]]);
        let sea = ocean(&extract, rect()).expect("there should be sea");
        // The dangle at (5, 5) projects to the south edge... its nearest
        // edge is ambiguous at the centre; any consistent choice closes the
        // ring. What matters: north-east of the drawn coast is water.
        assert!(inside(&sea.rings, (8.0, 8.0)));
        assert!(!inside(&sea.rings, (8.0, 2.0)));
    }

    /// Two parallel west-to-east chains cannot both be honoured: each says
    /// the ground south of it is water and the ground north is land, so the
    /// strip between them is claimed both ways. The repair drops one and
    /// fills honestly from what remains.
    #[test]
    fn contradictory_chains_drop_an_offender_and_fill_from_the_rest() {
        let extract = extract(
            &[(-2.0, 5.0), (12.0, 5.0), (-2.0, 3.0), (12.0, 3.0)],
            vec![vec![0, 1], vec![2, 3]],
        );
        let sea = ocean(&extract, rect()).expect("the survivor still implies sea");
        // Whichever chain survives, south of y = 3 is water on both
        // accounts and north of y = 5 is land on both.
        assert!(inside(&sea.rings, (5.0, 1.0)));
        assert!(!inside(&sea.rings, (5.0, 8.0)));
    }

    #[test]
    fn no_coastline_means_no_sea() {
        let extract = extract(&[], vec![]);
        assert!(ocean(&extract, rect()).is_none());
    }

    #[test]
    fn perimeter_positions_run_clockwise_from_the_north_west() {
        let rect = rect();
        let nw = rect.perimeter_t((0.0, 10.0));
        let ne = rect.perimeter_t((10.0, 10.0));
        let se = rect.perimeter_t((10.0, 0.0));
        let sw = rect.perimeter_t((0.0, 0.0));
        assert_eq!(nw, 0.0);
        assert!(nw < ne && ne < se && se < sw);
        // Mid-edges land between their corners.
        assert!((rect.perimeter_t((5.0, 10.0)) - 5.0).abs() < 1e-9);
        assert!(rect.perimeter_t((0.0, 5.0)) > sw);
    }

    #[test]
    fn corners_between_wraps_around_the_start() {
        let rect = rect();
        // From the middle of the west edge, clockwise past the north-west
        // and north-east corners to the middle of the east edge.
        let passed = rect.corners_between(
            rect.perimeter_t((0.0, 5.0)),
            rect.perimeter_t((10.0, 5.0)),
        );
        assert_eq!(passed, vec![(0.0, 10.0), (10.0, 10.0)]);
    }
}
