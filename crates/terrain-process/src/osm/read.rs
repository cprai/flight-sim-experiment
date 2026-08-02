//! Pulling the areas this pipeline paints out of a `.osm.pbf` extract.
//!
//! The file is a stream of blocks -- nodes, then ways, then relations -- and
//! elements refer to each other by id only: a way is a list of node ids, a
//! multipolygon relation a list of way ids. Geometry for a relation therefore
//! cannot be gathered in one pass, because by the time the relation streams
//! past, its member ways (mostly untagged, so nothing marked them as wanted)
//! are already gone. The reader makes three passes instead:
//!
//! 1. Ways and relations: classify every way's tags, keeping the node refs of
//!    classified closed ways and of coastline ways, and keep every classified
//!    multipolygon relation's member list.
//! 2. Member ways: node refs for the ways pass 1's relations named.
//! 3. Nodes: coordinates for exactly the node ids the kept ways reference.
//!
//! A pass over the whole extract is about two seconds, so three passes cost
//! far less than the alternative -- holding every way's refs from pass 1
//! onward -- would cost in memory. What is kept is only what will be painted:
//! for this project's extract, tens of megabytes out of a hundred-megabyte
//! file.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use osmpbf::{Element, ElementReader};
use serde::Deserialize;
use terrain_tiles::Material;
use terrain_tiles::project::{EPSG_LAMBERT, Projector};

use super::classify::classify;

/// The record `terrain-download` writes beside the extract.
///
/// Deliberately not `manifest.json` -- that name is how product directories
/// are recognised as tile pyramids, and the osm directory is not one. Only
/// the fields this tool needs are read; the record carries more.
#[derive(Debug, Deserialize)]
pub struct SourceRecord {
    /// The extract's file name inside the osm directory.
    pub file: String,
    /// The Geofabrik region id, for logging.
    pub region: String,
}

/// The file the record sits in, kept in step with `terrain-download`.
pub const RECORD_FILE: &str = "source.json";

impl SourceRecord {
    /// Reads the record out of the download's `osm` directory.
    pub fn read(osm_dir: &Path) -> Result<Self> {
        let path = osm_dir.join(RECORD_FILE);
        let text =
            std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

/// One classified closed way: a single ring of ground cover.
pub struct Area {
    pub material: Material,
    /// Node ids around the ring; first and last are the same node.
    pub refs: Vec<i64>,
}

/// One classified multipolygon relation, geometry still by reference.
///
/// Member roles are not kept. Outer versus inner only matters to a fill rule
/// that needs to know which ring is the hole; this pipeline fills even-odd
/// across all of a polygon's rings, where a ring inside a ring is a hole by
/// construction and the roles -- frequently missing or wrong in the data --
/// have nothing to add.
pub struct MultiPolygon {
    pub material: Material,
    /// Ids of the member ways.
    pub members: Vec<i64>,
}

/// Everything the extract holds that this pipeline paints.
pub struct Extract {
    pub areas: Vec<Area>,
    pub relations: Vec<MultiPolygon>,
    /// `natural=coastline` ways, directed with land on the left.
    pub coastlines: Vec<Vec<i64>>,
    /// Node refs for every way a relation names, by way id.
    pub members: HashMap<i64, Vec<i64>>,
    pub nodes: Nodes,
}

/// The coordinates of every node the kept ways reference.
///
/// A sorted id column beside a position column, rather than a hash map: the
/// ids are looked up millions of times but built once, and the sorted layout
/// is both half the memory and the natural output of deduplicating the refs.
pub struct Nodes {
    ids: Vec<i64>,
    /// Longitude and latitude in degrees as read; eastings and northings in
    /// metres after [`Nodes::project`].
    positions: Vec<(f64, f64)>,
}

impl Nodes {
    /// The position of a node, or `None` for an id the extract never carried
    /// -- which happens where the region clip cut a way's far end off.
    pub fn get(&self, id: i64) -> Option<(f64, f64)> {
        self.ids
            .binary_search(&id)
            .ok()
            .map(|index| self.positions[index])
    }

    /// Projects every position from degrees onto the tile grid's metres, in
    /// one batch, so everything downstream works in grid coordinates.
    pub fn project(&mut self) -> Result<()> {
        let projector = Projector::from_geographic(EPSG_LAMBERT)?;
        projector
            .to_source(&mut self.positions)
            .context("projecting the extract's nodes onto the grid")
    }

    /// A node table from parallel columns, for tests that need geometry
    /// without a pbf. `ids` must already be sorted.
    #[cfg(test)]
    pub fn for_tests(ids: Vec<i64>, positions: Vec<(f64, f64)>) -> Self {
        assert!(ids.is_sorted(), "the id column is binary searched");
        assert_eq!(ids.len(), positions.len());
        Self { ids, positions }
    }
}

/// Reads the extract in three passes; see the module doc for why three.
pub fn read_extract(path: &Path) -> Result<Extract> {
    let started = std::time::Instant::now();

    // Pass 1: every way's tags meet the classifier once.
    let mut areas = Vec::new();
    let mut relations = Vec::new();
    let mut coastlines = Vec::new();
    let mut unclosed = 0u64;
    reader(path)?
        .for_each(|element| match element {
            Element::Way(way) => {
                let tags: Vec<(&str, &str)> = way.tags().collect();
                // `refs()`, never `raw_refs()`: the raw slice is delta coded,
                // and reading it as ids would scatter every vertex.
                if tags.iter().any(|&(k, v)| k == "natural" && v == "coastline") {
                    coastlines.push(way.refs().collect());
                    return;
                }
                let Some(material) = classify(&tags) else {
                    return;
                };
                let refs: Vec<i64> = way.refs().collect();
                // A lone way must close on itself to hold ground. Open ones
                // are either mapping errors or rings the region clip cut,
                // and a guessed closure would paint ground that is not there.
                if refs.len() >= 4 && refs.first() == refs.last() {
                    areas.push(Area { material, refs });
                } else {
                    unclosed += 1;
                }
            }
            Element::Relation(relation) => {
                let tags: Vec<(&str, &str)> = relation.tags().collect();
                if !tags.iter().any(|&(k, v)| k == "type" && v == "multipolygon") {
                    return;
                }
                let Some(material) = classify(&tags) else {
                    return;
                };
                let members: Vec<i64> = relation
                    .members()
                    .filter(|member| member.member_type == osmpbf::RelMemberType::Way)
                    .map(|member| member.member_id)
                    .collect();
                if !members.is_empty() {
                    relations.push(MultiPolygon { material, members });
                }
            }
            Element::Node(_) | Element::DenseNode(_) => {}
        })
        .context("scanning ways and relations")?;

    // Pass 2: geometry for the ways the relations named.
    let mut wanted_ways: Vec<i64> = relations
        .iter()
        .flat_map(|relation| relation.members.iter().copied())
        .collect();
    wanted_ways.sort_unstable();
    wanted_ways.dedup();
    let mut members: HashMap<i64, Vec<i64>> = HashMap::with_capacity(wanted_ways.len());
    reader(path)?
        .for_each(|element| {
            if let Element::Way(way) = element
                && wanted_ways.binary_search(&way.id()).is_ok()
            {
                members.insert(way.id(), way.refs().collect());
            }
        })
        .context("scanning relation member ways")?;

    // Pass 3: coordinates for exactly the nodes the kept ways reference.
    let mut ids: Vec<i64> = areas
        .iter()
        .map(|area| &area.refs)
        .chain(coastlines.iter())
        .chain(members.values())
        .flatten()
        .copied()
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let mut positions = vec![(f64::NAN, f64::NAN); ids.len()];
    reader(path)?
        .for_each(|element| {
            let (id, position) = match element {
                Element::Node(node) => (node.id(), (node.lon(), node.lat())),
                Element::DenseNode(node) => (node.id(), (node.lon(), node.lat())),
                _ => return,
            };
            if let Ok(index) = ids.binary_search(&id) {
                positions[index] = position;
            }
        })
        .context("scanning node coordinates")?;

    let missing = positions.iter().filter(|p| p.0.is_nan()).count();
    log::info!(
        "read {}: {} areas, {} multipolygons over {} member ways, {} coastline ways, \
         {} nodes in {:.1?}",
        path.display(),
        areas.len(),
        relations.len(),
        members.len(),
        coastlines.len(),
        ids.len(),
        started.elapsed()
    );
    if unclosed > 0 {
        log::info!("skipped {unclosed} classified ways that do not close");
    }
    if missing > 0 {
        // Rings touching these fail to resolve later, individually.
        log::warn!("{missing} referenced nodes are not in the extract");
    }

    Ok(Extract {
        areas,
        relations,
        coastlines,
        members,
        nodes: Nodes { ids, positions },
    })
}

fn reader(path: &Path) -> Result<ElementReader<std::io::BufReader<std::fs::File>>> {
    ElementReader::from_path(path).with_context(|| format!("opening {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_lookup_finds_present_ids_and_refuses_absent_ones() {
        let nodes = Nodes {
            ids: vec![-7, 3, 19],
            positions: vec![(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        };
        assert_eq!(nodes.get(-7), Some((1.0, 2.0)));
        assert_eq!(nodes.get(19), Some((5.0, 6.0)));
        assert_eq!(nodes.get(0), None);
    }

    #[test]
    fn the_source_record_reads_the_fields_it_needs_and_ignores_the_rest() {
        let dir = std::env::temp_dir().join(format!("terrain-process-osm-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("failed to create");
        std::fs::write(
            dir.join(RECORD_FILE),
            r#"{"region": "southcoast-admreg", "file": "x.osm.pbf", "md5": "abc"}"#,
        )
        .expect("failed to write");
        let record = SourceRecord::read(&dir).expect("failed to read");
        assert_eq!(record.file, "x.osm.pbf");
        assert_eq!(record.region, "southcoast-admreg");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
