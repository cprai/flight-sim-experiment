//! Turning ids into rings, and rings into polygons ready to paint.
//!
//! A closed classified way is already a polygon. A multipolygon relation is
//! not: it names member ways that are arbitrary fragments -- a lake shore
//! drawn in five pieces, an island in one -- and the fragments only become
//! rings when chained end to end by shared node ids. The stitcher here does
//! that for relations, and `coastline` borrows it for the sea, which arrives
//! as the same kind of fragments at a larger scale.
//!
//! Node ids, not coordinates: two ways that meet do so at the *same node*,
//! so id equality is exact where coordinate equality would need a tolerance
//! and a wrong tolerance would weld across a narrow channel.
//!
//! Everything downstream fills a polygon's rings even-odd, so no ring is
//! outer or inner here; a hole is just a ring inside another. What cannot be
//! resolved -- a member way the extract does not carry, a ring that will not
//! close, a node the region clip removed -- is dropped ring by ring with a
//! count, keeping the rest of its polygon, because a lake missing one shore
//! fragment still holds more truth as a partial lake than as nothing.

use terrain_materials::Material;

use super::classify::precedence;
use super::read::{Extract, Nodes};

/// One paintable polygon: a material and the rings that bound it.
pub struct Polygon {
    pub material: Material,
    /// The painter's layer, from [`precedence`], carried here so sorting
    /// never re-derives it.
    pub layer: u8,
    /// Closed rings in grid metres; first and last point of each are equal.
    /// Filled even-odd across all of them.
    pub rings: Vec<Vec<(f64, f64)>>,
    /// `[min_x, min_y, max_x, max_y]` over every ring.
    pub bbox: [f64; 4],
    /// Sum of the rings' absolute areas, in square metres. The same-layer
    /// paint order: a container's boundary always encloses more than what it
    /// contains, so bigger paints first and smaller paints over it.
    pub area: f64,
}

impl Polygon {
    /// Builds a polygon from resolved rings, or `None` if none survived.
    pub fn new(material: Material, rings: Vec<Vec<(f64, f64)>>) -> Option<Self> {
        if rings.is_empty() {
            return None;
        }
        let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let mut area = 0.0;
        for ring in &rings {
            for &(x, y) in ring {
                bbox[0] = bbox[0].min(x);
                bbox[1] = bbox[1].min(y);
                bbox[2] = bbox[2].max(x);
                bbox[3] = bbox[3].max(y);
            }
            area += ring_area(ring).abs();
        }
        Some(Self {
            material,
            layer: precedence(material),
            rings,
            bbox,
            area,
        })
    }
}

/// Signed area of a closed ring, by the shoelace formula. Positive when the
/// ring runs counter-clockwise in an x-east, y-north frame.
pub fn ring_area(ring: &[(f64, f64)]) -> f64 {
    let mut doubled = 0.0;
    for pair in ring.windows(2) {
        let ((x1, y1), (x2, y2)) = (pair[0], pair[1]);
        doubled += x1 * y2 - x2 * y1;
    }
    doubled * 0.5
}

/// What the stitcher made of a pile of way fragments.
pub struct Stitched {
    /// Chains whose ends met: first and last id equal.
    pub closed: Vec<Vec<i64>>,
    /// Chains that ran out of continuations with their ends apart.
    pub open: Vec<Vec<i64>>,
}

/// Chains way fragments end to end by shared endpoint ids.
///
/// Greedy: take an unused fragment, extend its tail with whichever unused
/// fragment starts or ends there (reversing as needed), and when the tail is
/// stuck, flip the chain once to try the other end. A junction where three
/// fragments meet -- invalid for the boundaries this is used on, but the data
/// is what mappers typed -- takes whichever continuation comes first.
pub fn stitch(ways: &[Vec<i64>]) -> Stitched {
    let mut closed = Vec::new();
    let mut open = Vec::new();
    let mut used = vec![false; ways.len()];

    // Every fragment end, so continuations are found by lookup rather than
    // by scanning every unused fragment at every step.
    let mut by_end: std::collections::HashMap<i64, Vec<usize>> = std::collections::HashMap::new();
    for (index, way) in ways.iter().enumerate() {
        if way.len() < 2 {
            used[index] = true;
            continue;
        }
        for id in [way[0], *way.last().expect("length checked")] {
            by_end.entry(id).or_default().push(index);
        }
    }

    for start in 0..ways.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut chain = ways[start].clone();
        let mut flipped = false;
        loop {
            if chain.first() == chain.last() && chain.len() >= 4 {
                closed.push(chain);
                break;
            }
            let tail = *chain.last().expect("chains never empty");
            let next = by_end
                .get(&tail)
                .into_iter()
                .flatten()
                .copied()
                .find(|&index| !used[index]);
            match next {
                Some(index) => {
                    used[index] = true;
                    let way = &ways[index];
                    if way[0] == tail {
                        chain.extend_from_slice(&way[1..]);
                    } else {
                        chain.extend(way.iter().rev().skip(1));
                    }
                }
                None if !flipped => {
                    // The tail is stuck; the head may not be.
                    chain.reverse();
                    flipped = true;
                }
                None => {
                    if flipped {
                        // Restore the fragments' own direction. Closed rings
                        // do not care, but the coastline reads water off the
                        // right-hand side of an *open* chain, and handing it
                        // back reversed would flood the land.
                        chain.reverse();
                    }
                    open.push(chain);
                    break;
                }
            }
        }
    }
    Stitched { closed, open }
}

/// Resolves a chain of node ids into grid-metre points, or `None` if any
/// node is not in the extract.
pub fn resolve(chain: &[i64], nodes: &Nodes) -> Option<Vec<(f64, f64)>> {
    chain.iter().map(|&id| nodes.get(id)).collect()
}

/// Every classified area in the extract, as paintable polygons.
pub fn polygons(extract: &Extract) -> Vec<Polygon> {
    let mut out = Vec::with_capacity(extract.areas.len() + extract.relations.len());
    let mut dropped_rings = 0u64;
    let mut dropped_polygons = 0u64;

    for area in &extract.areas {
        match resolve(&area.refs, &extract.nodes).and_then(|ring| {
            Polygon::new(area.material, vec![ring])
        }) {
            Some(polygon) => out.push(polygon),
            None => dropped_polygons += 1,
        }
    }

    for relation in &extract.relations {
        let fragments: Vec<Vec<i64>> = relation
            .members
            .iter()
            .filter_map(|way| extract.members.get(way).cloned())
            .collect();
        dropped_rings += (relation.members.len() - fragments.len()) as u64;

        let stitched = stitch(&fragments);
        dropped_rings += stitched.open.len() as u64;
        let rings: Vec<Vec<(f64, f64)>> = stitched
            .closed
            .iter()
            .filter_map(|ring| {
                let resolved = resolve(ring, &extract.nodes);
                if resolved.is_none() {
                    dropped_rings += 1;
                }
                resolved
            })
            .collect();
        match Polygon::new(relation.material, rings) {
            Some(polygon) => out.push(polygon),
            None => dropped_polygons += 1,
        }
    }

    if dropped_rings > 0 || dropped_polygons > 0 {
        log::info!(
            "assembly dropped {dropped_rings} rings and {dropped_polygons} whole polygons \
             (missing members, missing nodes, or rings that will not close)"
        );
    }
    log::info!("assembled {} polygons", out.len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragments_stitch_into_a_ring_whichever_way_they_point() {
        // A square drawn as three fragments, the middle one reversed.
        let ways = vec![vec![1, 2, 3], vec![5, 4, 3], vec![5, 6, 1]];
        let stitched = stitch(&ways);
        assert!(stitched.open.is_empty());
        assert_eq!(stitched.closed.len(), 1);
        let ring = &stitched.closed[0];
        assert_eq!(ring.first(), ring.last());
        assert_eq!(ring.len(), 7, "six corners and the repeat: {ring:?}");
    }

    #[test]
    fn a_fragment_pile_with_a_gap_comes_out_open() {
        let ways = vec![vec![1, 2, 3], vec![4, 5, 1]];
        let stitched = stitch(&ways);
        assert!(stitched.closed.is_empty());
        assert_eq!(stitched.closed.len() + stitched.open.len(), 1);
        // Both fragments joined into one chain despite the gap at 3-4.
        assert_eq!(stitched.open[0].len(), 5);
    }

    /// A chain whose tail is stuck but whose head continues must flip rather
    /// than give up: which end of a fragment pile the walk starts from is
    /// arbitrary, and coastline chains are exactly this shape.
    #[test]
    fn a_chain_stuck_at_one_end_continues_from_the_other() {
        let ways = vec![vec![2, 3], vec![1, 2]];
        let stitched = stitch(&ways);
        assert!(stitched.closed.is_empty());
        assert_eq!(stitched.open.len(), 1);
        // In the fragments' own direction, not the direction the walk
        // happened to build it: 1 to 2 to 3, as the pieces are drawn.
        assert_eq!(stitched.open[0], vec![1, 2, 3]);
    }

    #[test]
    fn two_separate_rings_stay_separate() {
        let ways = vec![vec![1, 2, 3, 1], vec![10, 11, 12, 10]];
        let stitched = stitch(&ways);
        assert_eq!(stitched.closed.len(), 2);
        assert!(stitched.open.is_empty());
    }

    #[test]
    fn the_shoelace_signs_by_winding_and_measures_the_square() {
        let counter_clockwise = vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)];
        assert_eq!(ring_area(&counter_clockwise), 16.0);
        let clockwise: Vec<_> = counter_clockwise.iter().rev().copied().collect();
        assert_eq!(ring_area(&clockwise), -16.0);
    }

    #[test]
    fn a_polygons_bbox_and_area_span_all_its_rings() {
        let outer = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)];
        let hole = vec![(2.0, 2.0), (4.0, 2.0), (4.0, 4.0), (2.0, 4.0), (2.0, 2.0)];
        let polygon =
            Polygon::new(Material::Lake, vec![outer, hole]).expect("two rings is a polygon");
        assert_eq!(polygon.bbox, [0.0, 0.0, 10.0, 10.0]);
        // Absolute areas add; the hole is a fill-rule matter, not a sum.
        assert_eq!(polygon.area, 104.0);
        assert_eq!(polygon.layer, 5, "lakes paint on the water layer");
    }

    #[test]
    fn a_polygon_with_no_rings_is_none() {
        assert!(Polygon::new(Material::Lake, vec![]).is_none());
    }
}
