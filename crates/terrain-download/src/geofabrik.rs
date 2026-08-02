//! Raw OpenStreetMap data for the box, as one Geofabrik extract.
//!
//! OSM is a single worldwide database, and nobody serves an arbitrary clip of
//! it without either a job queue or a quarter-degree ceiling. Geofabrik
//! publishes daily `.osm.pbf` extracts of every administrative region,
//! unauthenticated, and is the distribution point openstreetmap.org itself
//! points bulk consumers at. So the tool downloads the smallest region that
//! contains the whole requested box and leaves clipping to the processing
//! step -- for this project's ground that is BC's South Coast administrative
//! region, about 110 MB. Which region that is comes from the server's own
//! machine-readable index, not from a hardcoded table, so the tool works
//! anywhere Geofabrik covers.
//!
//! The `-latest.osm.pbf` name is not a file but a redirect to the dated
//! publication of the day (`...-260731.osm.pbf`), which is immutable once
//! written. The download pins that dated URL: a resume is then a plain range
//! request against bytes that cannot change, rather than a race against the
//! nightly replacement. The pinned URL, its ETag and its length live in a
//! sidecar next to the partial file, and a partial whose sidecar does not
//! match what the server currently offers is discarded rather than continued
//! -- what a spliced file produces (a corrupt pbf, discovered only when the
//! processing step reads it) is far worse than a fresh download. The
//! published `.md5` is checked at the end either way.

use std::collections::HashMap;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;

use crate::bbox::LatLonBox;
use crate::coverage;
use crate::retry;

/// Where the region index lives under the download root.
pub const INDEX_PATH: &str = "index-v1.json";

/// The metadata written beside the extract.
///
/// Deliberately not `manifest.json`: that name is how `terrain-process`
/// recognises a tile pyramid, and this directory holds a single vector file
/// that the process step must skip, not resample.
pub const RECORD_FILE: &str = "source.json";

/// One region of the Geofabrik index: an extract and the ground it covers.
pub struct Region {
    pub id: String,
    pub parent: Option<String>,
    pub name: String,
    pub pbf_url: Option<String>,
    polygons: Vec<Polygon>,
}

/// One outer ring and the holes cut out of it, in degrees.
struct Polygon {
    exterior: Vec<(f64, f64)>,
    holes: Vec<Vec<(f64, f64)>>,
}

#[derive(Deserialize)]
struct Index {
    features: Vec<Feature>,
}

#[derive(Deserialize)]
struct Feature {
    properties: Properties,
    geometry: Geometry,
}

#[derive(Deserialize)]
struct Properties {
    id: String,
    parent: Option<String>,
    name: String,
    #[serde(default)]
    urls: Urls,
}

#[derive(Deserialize, Default)]
struct Urls {
    pbf: Option<String>,
}

/// GeoJSON positions may carry more than two numbers, so each one is read as
/// a list and only its first two members are kept.
#[derive(Deserialize)]
#[serde(tag = "type")]
enum Geometry {
    Polygon {
        coordinates: Vec<Vec<Vec<f64>>>,
    },
    MultiPolygon {
        coordinates: Vec<Vec<Vec<Vec<f64>>>>,
    },
}

fn ring(positions: Vec<Vec<f64>>) -> Vec<(f64, f64)> {
    positions
        .into_iter()
        .filter_map(|p| Some((*p.first()?, *p.get(1)?)))
        .collect()
}

fn polygon(mut rings: Vec<Vec<Vec<f64>>>) -> Option<Polygon> {
    if rings.is_empty() {
        return None;
    }
    let exterior = ring(rings.remove(0));
    let holes = rings.into_iter().map(ring).collect();
    Some(Polygon { exterior, holes })
}

impl From<Feature> for Region {
    fn from(feature: Feature) -> Self {
        let polygons = match feature.geometry {
            Geometry::Polygon { coordinates } => polygon(coordinates).into_iter().collect(),
            Geometry::MultiPolygon { coordinates } => {
                coordinates.into_iter().filter_map(polygon).collect()
            }
        };
        Self {
            id: feature.properties.id,
            parent: feature.properties.parent,
            name: feature.properties.name,
            pbf_url: feature.properties.urls.pbf,
            polygons,
        }
    }
}

/// Fetches the region index: every extract the server offers, with the
/// geometry it covers.
pub async fn fetch_index(client: &reqwest::Client, root: &str) -> Result<Vec<Region>> {
    let url = format!("{}/{INDEX_PATH}", root.trim_end_matches('/'));
    let body = retry::retrying(&url, retry::is_transient, || async {
        client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await
    })
    .await
    .with_context(|| format!("requesting {url}"))?;

    let index: Index =
        serde_json::from_str(&body).with_context(|| format!("parsing the index from {url}"))?;
    Ok(index.features.into_iter().map(Region::from).collect())
}

impl Region {
    /// Whether the region's geometry covers this point.
    fn contains_point(&self, longitude: f64, latitude: f64) -> bool {
        self.polygons.iter().any(|polygon| {
            ring_contains(&polygon.exterior, longitude, latitude)
                && !polygon
                    .holes
                    .iter()
                    .any(|hole| ring_contains(hole, longitude, latitude))
        })
    }

    /// Whether the region covers the whole box, tested on a 5x5 lattice of
    /// sample points.
    ///
    /// Not a true polygon-against-polygon test: a boundary poking into the box
    /// between two samples goes unseen. What that costs is choosing a parent
    /// region -- more data, still correct -- or a spurious "no region" that
    /// `--osm-region` overrides; what it saves is a clipping dependency for a
    /// choice the box sizes involved never make wrong in practice, since
    /// extract boundaries are drawn generously around their regions.
    pub fn contains_box(&self, box_: LatLonBox) -> bool {
        const SAMPLES: u32 = 5;
        (0..SAMPLES).all(|row| {
            let latitude =
                box_.south + box_.height_degrees() * f64::from(row) / f64::from(SAMPLES - 1);
            (0..SAMPLES).all(|column| {
                let longitude =
                    box_.west + box_.width_degrees() * f64::from(column) / f64::from(SAMPLES - 1);
                self.contains_point(longitude, latitude)
            })
        })
    }

    /// The area of the geometry's bounding box, in square degrees. Only used
    /// to break ties between equally deep regions, so the crudeness is fine.
    fn footprint(&self) -> f64 {
        let mut west = f64::INFINITY;
        let mut south = f64::INFINITY;
        let mut east = f64::NEG_INFINITY;
        let mut north = f64::NEG_INFINITY;
        for polygon in &self.polygons {
            for &(longitude, latitude) in &polygon.exterior {
                west = west.min(longitude);
                south = south.min(latitude);
                east = east.max(longitude);
                north = north.max(latitude);
            }
        }
        if west > east {
            return 0.0;
        }
        (east - west) * (north - south)
    }
}

/// Even-odd ray casting: a ray east from the point crosses the ring an odd
/// number of times exactly when the point is inside.
///
/// None of the regions this tool would choose straddle the antimeridian, so
/// longitude is treated as plain x. (Geofabrik's index does hold a couple
/// that do -- Fiji -- but Canadian terrain never selects them.)
fn ring_contains(ring: &[(f64, f64)], longitude: f64, latitude: f64) -> bool {
    let mut inside = false;
    let mut previous = match ring.last() {
        Some(&point) => point,
        None => return false,
    };
    for &(x, y) in ring {
        let (px, py) = previous;
        if (y > latitude) != (py > latitude) && longitude < (px - x) * (latitude - y) / (py - y) + x
        {
            inside = !inside;
        }
        previous = (x, y);
    }
    inside
}

/// How many ancestors a region has. The cap guards against a cyclic index --
/// never observed, but an infinite loop is a bad way to learn about one.
fn depth(by_id: &HashMap<&str, &Region>, region: &Region) -> usize {
    let mut steps = 0;
    let mut current = region;
    while let Some(parent) = &current.parent {
        match by_id.get(parent.as_str()) {
            Some(next) if steps < 100 => {
                current = next;
                steps += 1;
            }
            _ => break,
        }
    }
    steps
}

/// Picks the smallest extract that covers the whole box: the deepest region in
/// the hierarchy containing it, ties broken by footprint.
///
/// Deepest rather than smallest-by-area as the primary key because the
/// hierarchy is what Geofabrik actually promises -- a child is a subset of its
/// parent -- while comparing areas across unrelated regions would let a small
/// neighbour that merely overlaps win.
pub fn select_region(regions: &[Region], box_: LatLonBox) -> Result<&Region> {
    let by_id: HashMap<&str, &Region> = regions.iter().map(|r| (r.id.as_str(), r)).collect();

    let chosen = regions
        .iter()
        .filter(|region| region.pbf_url.is_some() && region.contains_box(box_))
        .max_by(|a, b| {
            depth(&by_id, a)
                .cmp(&depth(&by_id, b))
                .then_with(|| b.footprint().total_cmp(&a.footprint()))
        });
    if let Some(region) = chosen {
        return Ok(region);
    }

    // Name the regions that cover the centre, deepest first: the likeliest
    // reason to land here is a box straddling ground the index only covers in
    // pieces, and the fix is forcing one of these.
    let longitude = (box_.west + box_.east) / 2.0;
    let latitude = (box_.south + box_.north) / 2.0;
    let mut partial: Vec<&Region> = regions
        .iter()
        .filter(|region| region.pbf_url.is_some() && region.contains_point(longitude, latitude))
        .collect();
    partial.sort_by_key(|region| std::cmp::Reverse(depth(&by_id, region)));

    if partial.is_empty() {
        bail!(
            "no Geofabrik region covers the box, or even its centre; the extract \
             map does not reach this ground"
        );
    }
    bail!(
        "no Geofabrik region covers the whole box, though its centre falls in: {}; \
         pass --osm-region with one of those ids to download it anyway",
        partial
            .iter()
            .map(|region| region.id.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

/// Looks up the region `--osm-region` named.
pub fn find_region<'a>(regions: &'a [Region], id: &str) -> Result<&'a Region> {
    let region = regions
        .iter()
        .find(|region| region.id == id)
        .with_context(|| format!("the Geofabrik index has no region with id `{id}`"))?;
    ensure!(
        region.pbf_url.is_some(),
        "region `{id}` exists but offers no pbf extract"
    );
    Ok(region)
}

/// What one HEAD of the `latest` alias learned about today's publication.
pub struct Probe {
    /// The dated, immutable URL the alias redirected to.
    pub url: String,
    pub length: u64,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
}

/// Follows the `latest` redirect and reads the publication's vitals.
pub async fn probe(client: &reqwest::Client, latest: &str) -> Result<Probe> {
    let response = retry::retrying(latest, retry::is_transient, || async {
        client.head(latest).send().await?.error_for_status()
    })
    .await
    .with_context(|| format!("probing {latest}"))?;

    let header = |name: reqwest::header::HeaderName| {
        response
            .headers()
            .get(&name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    };
    // Read from the header rather than `content_length()`, which reports the
    // body size -- and a HEAD response has no body.
    let length = header(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.parse().ok())
        .with_context(|| format!("{latest} answered HEAD without a Content-Length"))?;

    Ok(Probe {
        url: response.url().to_string(),
        length,
        etag: header(reqwest::header::ETAG),
        last_modified: header(reqwest::header::LAST_MODIFIED),
    })
}

/// The sidecar kept beside a partial download: which publication its bytes
/// belong to, so a later run knows whether they are still worth keeping.
#[derive(Serialize, Deserialize)]
struct PartMeta {
    url: String,
    etag: Option<String>,
    total: u64,
}

/// Whether a partial download's bytes are a prefix of this publication.
///
/// The URL is dated and immutable, so matching it is the real test; the ETag
/// is checked too when both sides have one, in case that promise ever breaks.
fn part_matches(meta: &PartMeta, probe: &Probe) -> bool {
    meta.url == probe.url
        && meta.total == probe.length
        && match (&meta.etag, &probe.etag) {
            (Some(ours), Some(theirs)) => ours == theirs,
            _ => true,
        }
}

fn part_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

fn meta_path(target: &Path) -> PathBuf {
    let mut name = target.as_os_str().to_owned();
    name.push(".part.meta.json");
    PathBuf::from(name)
}

/// Decides what an existing partial download is worth, and returns how many
/// of its bytes will be reused.
///
/// Without `--resume` any partial is discarded, matching the flag's meaning on
/// the raster paths: "already on disk" is only trusted when asked for. With
/// it, the partial survives only if its sidecar pins the publication the
/// server is still offering.
pub fn reconcile_part(target: &Path, probe: &Probe, resume: bool) -> u64 {
    let part = part_path(target);
    let meta = meta_path(target);

    let length = match std::fs::metadata(&part) {
        Ok(metadata) => metadata.len(),
        Err(_) => {
            let _ = std::fs::remove_file(&meta);
            return 0;
        }
    };

    let kept = resume
        && std::fs::read_to_string(&meta)
            .ok()
            .and_then(|text| serde_json::from_str::<PartMeta>(&text).ok())
            .is_some_and(|meta| part_matches(&meta, probe));
    if kept {
        return length;
    }

    if resume {
        log::info!("the extract was republished since the partial download; starting over");
    }
    let _ = std::fs::remove_file(&part);
    let _ = std::fs::remove_file(&meta);
    0
}

/// Says what is about to be downloaded and asks whether to.
///
/// Always asks, unlike the elevation prompt with its threshold: there is no
/// "small enough to skip asking" here, because the extract is a whole region
/// and starts at hundreds of megabytes. `--yes` skips the question but not
/// the announcement -- a log of an unattended run should still say what was
/// chosen.
pub fn confirm(region: &Region, probe: &Probe, reused: u64, assume_yes: bool) -> Result<bool> {
    println!(
        "The box falls in {} (`{}`), a {} extract:\n  {}",
        region.name,
        region.id,
        coverage::describe_bytes(probe.length),
        probe.url
    );
    if reused > 0 {
        println!(
            "  resuming past {} already on disk, {} to go",
            coverage::describe_bytes(reused),
            coverage::describe_bytes(probe.length - reused)
        );
    }
    if assume_yes {
        return Ok(true);
    }

    if !std::io::stdin().is_terminal() {
        anyhow::bail!(
            "downloading it needs a confirmation and there is no terminal to ask at; \
             pass --yes to proceed"
        );
    }
    print!("Download it? [y/N] ");
    std::io::stdout().flush().context("prompting")?;

    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .context("reading the answer")?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes" | "Yes"))
}

/// The `start` and `total` of a `Content-Range: bytes start-end/total`.
fn parse_content_range(value: &str) -> Option<(u64, u64)> {
    let (range, total) = value.strip_prefix("bytes ")?.split_once('/')?;
    let (start, _) = range.split_once('-')?;
    Some((start.parse().ok()?, total.parse().ok()?))
}

/// Whether a failed transfer is worth continuing. Only network failures are:
/// an I/O error is the disk's to explain, and anything else this module bails
/// on is a fact about the server that will hold on every attempt.
fn is_transient_transfer(error: &anyhow::Error) -> bool {
    error
        .downcast_ref::<reqwest::Error>()
        .is_some_and(retry::is_transient)
}

/// Downloads the publication into the `.part` file until it is whole.
/// Returns how many bytes this run fetched.
///
/// The retry budget resets whenever the file grows, which `retry::retrying`'s
/// fixed budget deliberately does not do: five attempts is the right price for
/// a request that is cheap to restart, but this transfer runs for the better
/// part of an hour, and a blip every ten minutes of healthy progress should
/// never add up to abandoning it. A sustained outage still exhausts the budget
/// in a few seconds, exactly as it would anywhere else.
pub async fn download(client: &reqwest::Client, probe: &Probe, target: &Path) -> Result<u64> {
    let part = part_path(target);
    let meta = meta_path(target);

    // The sidecar goes down before the first byte, so no partial file ever
    // exists without the record of which publication it belongs to.
    let record = PartMeta {
        url: probe.url.clone(),
        etag: probe.etag.clone(),
        total: probe.length,
    };
    std::fs::write(&meta, serde_json::to_string_pretty(&record)?)
        .with_context(|| format!("writing {}", meta.display()))?;

    let start = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let mut wait = retry::FIRST_BACKOFF;
    let mut tries = 1;
    loop {
        let offset = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        ensure!(
            offset <= probe.length,
            "{} holds {offset} bytes but the publication is {}; delete it and re-run",
            part.display(),
            probe.length
        );
        if offset == probe.length {
            return Ok(probe.length - start);
        }

        let error = match attempt(client, probe, &part, offset).await {
            // A body that ended early but cleanly: go straight round again,
            // the range request continues from wherever it stopped.
            Ok(()) => continue,
            Err(error) => error,
        };
        if !is_transient_transfer(&error) {
            return Err(error).with_context(|| format!("downloading {}", probe.url));
        }

        let grew = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0) > offset;
        if grew {
            tries = 1;
            wait = retry::FIRST_BACKOFF;
        } else if tries >= retry::ATTEMPTS {
            return Err(error).with_context(|| {
                format!(
                    "downloading {} made no progress over {} attempts",
                    probe.url,
                    retry::ATTEMPTS
                )
            });
        } else {
            tries += 1;
        }
        log::warn!(
            "the transfer broke ({error:#}); retrying in {} ms",
            wait.as_millis()
        );
        tokio::time::sleep(wait).await;
        wait *= 2;
    }
}

/// One request: ask for everything past `offset` and stream it to disk.
/// Returns cleanly when the body ends, however short; the caller measures the
/// file to see what that meant.
async fn attempt(client: &reqwest::Client, probe: &Probe, part: &Path, offset: u64) -> Result<()> {
    let mut request = client.get(&probe.url);
    if offset > 0 {
        request = request.header(reqwest::header::RANGE, format!("bytes={offset}-"));
    }
    let response = request.send().await?;

    // The dated publication is dropped some days after a newer one appears,
    // so a partial download left alone too long can outlive its source.
    if response.status() == reqwest::StatusCode::NOT_FOUND
        || response.status() == reqwest::StatusCode::GONE
    {
        bail!(
            "{} is gone -- Geofabrik replaces extracts daily and eventually drops \
             old publications; re-run to download the current one",
            probe.url
        );
    }
    let response = response.error_for_status()?;

    if offset > 0 {
        // A 200 here would mean the server ignored the range and is sending
        // the whole file; appending that to the partial would splice garbage.
        ensure!(
            response.status() == reqwest::StatusCode::PARTIAL_CONTENT,
            "{} answered a range request with {} instead of a partial body",
            probe.url,
            response.status()
        );
        let range = response
            .headers()
            .get(reqwest::header::CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_content_range)
            .with_context(|| format!("{} sent no readable Content-Range", probe.url))?;
        ensure!(
            range == (offset, probe.length),
            "{} offered bytes {}-/{} where {offset}-/{} was asked for",
            probe.url,
            range.0,
            range.1,
            probe.length
        );
    }

    let mut file = if offset == 0 {
        tokio::fs::File::create(part).await
    } else {
        tokio::fs::OpenOptions::new().append(true).open(part).await
    }
    .with_context(|| format!("opening {}", part.display()))?;

    let mut done = offset;
    let mut decile = done * 10 / probe.length;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)
            .await
            .with_context(|| format!("writing {}", part.display()))?;
        done += chunk.len() as u64;
        ensure!(
            done <= probe.length,
            "{} sent more bytes than the {} it promised",
            probe.url,
            probe.length
        );
        if done * 10 / probe.length > decile {
            decile = done * 10 / probe.length;
            log::info!(
                "{} of {} ({}%)",
                coverage::describe_bytes(done),
                coverage::describe_bytes(probe.length),
                decile * 10
            );
        }
    }
    file.flush()
        .await
        .with_context(|| format!("flushing {}", part.display()))?;
    Ok(())
}

/// The first token of a Geofabrik `.md5` file: `<hex>  <filename>`.
fn parse_md5(text: &str) -> Option<String> {
    let token = text.split_whitespace().next()?;
    (token.len() == 32 && token.chars().all(|c| c.is_ascii_hexdigit()))
        .then(|| token.to_ascii_lowercase())
}

/// Checks the completed `.part` against the publication's published hash.
///
/// This is what proves a file assembled across several runs and range
/// requests is exactly one publication; the sidecar makes that likely, the
/// hash makes it certain. Returns the hex digest for the record.
pub async fn verify_md5(client: &reqwest::Client, probe: &Probe, target: &Path) -> Result<String> {
    let url = format!("{}.md5", probe.url);
    let text = retry::retrying(&url, retry::is_transient, || async {
        client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await
    })
    .await
    .with_context(|| format!("requesting {url}"))?;
    let expected = parse_md5(&text).with_context(|| format!("no md5 digest in {url}"))?;

    let part = part_path(target);
    let actual = tokio::task::spawn_blocking({
        let part = part.clone();
        move || -> Result<String> {
            let mut file = std::fs::File::open(&part)
                .with_context(|| format!("opening {}", part.display()))?;
            let mut context = md5::Context::new();
            let mut buffer = vec![0u8; 1 << 20];
            loop {
                let n = file.read(&mut buffer)?;
                if n == 0 {
                    break;
                }
                context.consume(&buffer[..n]);
            }
            Ok(format!("{:x}", context.finalize()))
        }
    })
    .await
    .context("hashing the download")??;

    ensure!(
        actual == expected,
        "{} hashed to {actual} where the server published {expected}; \
         delete it and re-run",
        part.display()
    );
    Ok(actual)
}

/// Promotes a verified `.part` to its final name and drops the sidecar.
pub fn finish(target: &Path) -> Result<()> {
    std::fs::rename(part_path(target), target)
        .with_context(|| format!("renaming into {}", target.display()))?;
    let _ = std::fs::remove_file(meta_path(target));
    Ok(())
}

/// A box in the shape the record file keeps it.
#[derive(Serialize)]
pub struct GeographicBox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl From<LatLonBox> for GeographicBox {
    fn from(box_: LatLonBox) -> Self {
        Self {
            west: box_.west,
            south: box_.south,
            east: box_.east,
            north: box_.north,
        }
    }
}

/// Everything the later processing step needs to know about the file: which
/// publication it is, what ground was actually wanted from it, and how to
/// tell whether it has gone stale.
#[derive(Serialize)]
pub struct SourceRecord {
    pub region: String,
    pub name: String,
    /// The dated publication actually fetched.
    pub url: String,
    /// The stable alias it came from, for checking freshness later.
    pub latest_url: String,
    pub file: String,
    pub requested_box: GeographicBox,
    pub snapped_box: GeographicBox,
    pub content_length: u64,
    pub md5: String,
    pub last_modified: Option<String>,
    pub downloaded_at_unix: u64,
}

/// Writes the record beside the extract.
pub fn write_record(root: &Path, record: &SourceRecord) -> Result<()> {
    let path = root.join(RECORD_FILE);
    std::fs::write(&path, serde_json::to_string_pretty(record)?)
        .with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed square ring from corner to corner.
    fn square(west: f64, south: f64, east: f64, north: f64) -> Vec<(f64, f64)> {
        vec![
            (west, south),
            (east, south),
            (east, north),
            (west, north),
            (west, south),
        ]
    }

    fn region(id: &str, parent: Option<&str>, pbf: bool, exterior: Vec<(f64, f64)>) -> Region {
        Region {
            id: id.to_owned(),
            parent: parent.map(str::to_owned),
            name: id.to_owned(),
            pbf_url: pbf.then(|| format!("https://example.test/{id}-latest.osm.pbf")),
            polygons: vec![Polygon {
                exterior,
                holes: Vec::new(),
            }],
        }
    }

    fn box_(west: f64, south: f64, east: f64, north: f64) -> LatLonBox {
        LatLonBox {
            west,
            south,
            east,
            north,
        }
    }

    #[test]
    fn a_point_is_inside_its_ring_and_outside_a_hole() {
        let ring = square(0.0, 0.0, 10.0, 10.0);
        assert!(ring_contains(&ring, 5.0, 5.0));
        assert!(!ring_contains(&ring, 15.0, 5.0));

        let holed = Region {
            polygons: vec![Polygon {
                exterior: square(0.0, 0.0, 10.0, 10.0),
                holes: vec![square(4.0, 4.0, 6.0, 6.0)],
            }],
            ..region("holed", None, true, Vec::new())
        };
        assert!(holed.contains_point(2.0, 2.0));
        assert!(!holed.contains_point(5.0, 5.0), "the hole is outside");
    }

    #[test]
    fn any_part_of_a_multipolygon_counts() {
        let split = Region {
            polygons: vec![
                Polygon {
                    exterior: square(0.0, 0.0, 1.0, 1.0),
                    holes: Vec::new(),
                },
                Polygon {
                    exterior: square(5.0, 5.0, 6.0, 6.0),
                    holes: Vec::new(),
                },
            ],
            ..region("split", None, true, Vec::new())
        };
        assert!(split.contains_point(5.5, 5.5));
        assert!(
            !split.contains_point(3.0, 3.0),
            "between the parts is outside"
        );
    }

    fn fixture() -> Vec<Region> {
        vec![
            region("continent", None, true, square(0.0, 0.0, 100.0, 100.0)),
            region(
                "country",
                Some("continent"),
                true,
                square(10.0, 10.0, 60.0, 60.0),
            ),
            region(
                "province",
                Some("country"),
                true,
                square(20.0, 20.0, 40.0, 40.0),
            ),
        ]
    }

    #[test]
    fn the_deepest_region_containing_the_box_wins() {
        let regions = fixture();
        let chosen = select_region(&regions, box_(25.0, 25.0, 30.0, 30.0)).expect("should select");
        assert_eq!(chosen.id, "province");
    }

    #[test]
    fn a_box_straddling_a_boundary_falls_to_the_parent() {
        let regions = fixture();
        let chosen = select_region(&regions, box_(35.0, 35.0, 45.0, 45.0)).expect("should select");
        assert_eq!(chosen.id, "country");
    }

    #[test]
    fn a_region_without_an_extract_is_never_chosen() {
        let mut regions = fixture();
        regions[2].pbf_url = None;
        let chosen = select_region(&regions, box_(25.0, 25.0, 30.0, 30.0)).expect("should select");
        assert_eq!(chosen.id, "country");
    }

    #[test]
    fn a_box_nothing_contains_names_the_partial_fits() {
        let regions = fixture();
        // Centre inside everything, but the box pokes west of the continent.
        let error = select_region(&regions, box_(-5.0, 25.0, 55.0, 30.0))
            .map(|region| region.id.clone())
            .expect_err("nothing contains this")
            .to_string();
        assert!(error.contains("--osm-region"), "{error}");
        assert!(
            error.contains("province, country, continent"),
            "deepest first: {error}"
        );
    }

    #[test]
    fn the_index_geojson_is_read_into_regions() {
        let text = r#"{"type": "FeatureCollection", "features": [
            {"type": "Feature",
             "properties": {"id": "island", "parent": "sea", "name": "Island",
                            "urls": {"pbf": "https://example.test/island.osm.pbf"}},
             "geometry": {"type": "MultiPolygon",
                          "coordinates": [[[[0,0],[4,0],[4,4],[0,4],[0,0]]]]}},
            {"type": "Feature",
             "properties": {"id": "sea", "name": "Sea"},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]]}}
        ]}"#;
        let index: Index = serde_json::from_str(text).expect("should parse");
        let regions: Vec<Region> = index.features.into_iter().map(Region::from).collect();

        assert_eq!(regions[0].id, "island");
        assert_eq!(regions[0].parent.as_deref(), Some("sea"));
        assert!(regions[0].pbf_url.is_some());
        assert!(regions[0].contains_point(2.0, 2.0));

        assert_eq!(regions[1].id, "sea");
        assert!(regions[1].pbf_url.is_none(), "no urls at all means no pbf");
        assert!(regions[1].contains_point(-5.0, 5.0));
    }

    #[test]
    fn a_content_range_yields_its_start_and_total() {
        assert_eq!(
            parse_content_range("bytes 100-11254310/11254311"),
            Some((100, 11254311))
        );
        assert_eq!(parse_content_range("bytes */11254311"), None);
        assert_eq!(parse_content_range("garbage"), None);
    }

    #[test]
    fn an_md5_line_yields_its_digest() {
        assert_eq!(
            parse_md5("3be8512de6e3a6ce4f201f75faea310f  prince-edward-island-latest.osm.pbf\n"),
            Some("3be8512de6e3a6ce4f201f75faea310f".to_owned())
        );
        assert_eq!(parse_md5("not a digest"), None);
        assert_eq!(parse_md5(""), None);
    }

    #[test]
    fn a_partial_is_only_kept_for_the_same_publication() {
        let meta = PartMeta {
            url: "https://example.test/bc-260731.osm.pbf".to_owned(),
            etag: Some("\"abc\"".to_owned()),
            total: 100,
        };
        let probe = |url: &str, etag: Option<&str>, length: u64| Probe {
            url: url.to_owned(),
            length,
            etag: etag.map(str::to_owned),
            last_modified: None,
        };

        let same = probe(
            "https://example.test/bc-260731.osm.pbf",
            Some("\"abc\""),
            100,
        );
        assert!(part_matches(&meta, &same));

        let republished = probe(
            "https://example.test/bc-260801.osm.pbf",
            Some("\"abc\""),
            100,
        );
        assert!(
            !part_matches(&meta, &republished),
            "a new date is a new file"
        );

        let mutated = probe(
            "https://example.test/bc-260731.osm.pbf",
            Some("\"xyz\""),
            100,
        );
        assert!(!part_matches(&meta, &mutated), "same name, different bytes");

        let unstamped = probe("https://example.test/bc-260731.osm.pbf", None, 100);
        assert!(
            part_matches(&meta, &unstamped),
            "a missing etag is not a mismatch"
        );
    }
}
