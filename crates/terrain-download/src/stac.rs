//! Finding which published rasters cover the requested box.
//!
//! Natural Resources Canada exposes its elevation products through one STAC
//! API. The HRDEM collections are split into items that are 500 km squares of
//! the EPSG:3979 grid; MRDEM is a single national item. Either way each item
//! carries its rasters as assets, and asking the service which items intersect
//! a longitude/latitude box is a single request -- so nothing here has to
//! understand the block naming scheme, or care how many items a collection has.
//!
//! Two shapes of the data drive the code below. Items are *not* guaranteed to
//! carry the product being asked for -- the northern blocks of the two-metre
//! mosaic are satellite-derived and publish a surface model with no terrain
//! model beside it -- so a missing asset skips an item rather than failing the
//! run. And every item is asserted to be EPSG:3979, because the whole pipeline
//! places pixels by that assumption and a silent change of projection would
//! put the terrain in the wrong place rather than produce an error.

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::bbox::LatLonBox;
use crate::retry;

/// The projection every HRDEM mosaic item is published in.
pub const EXPECTED_EPSG: u32 = 3979;

/// What to fetch. The first two are elevation from HRDEM; the others come from
/// different providers entirely and never touch the STAC catalogue.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Product {
    /// Digital terrain model: the bare ground, with vegetation and buildings
    /// removed.
    Dtm,
    /// Digital surface model: the top of whatever the sensor saw.
    Dsm,
    /// Cloud-free Sentinel-2 colour imagery, for the terrain's surface.
    Albedo,
    /// Raw OpenStreetMap data covering the box, from a Geofabrik extract.
    Osm,
}

impl Product {
    /// The key an elevation product appears under in an item's asset map.
    ///
    /// Albedo has none: it does not come from the HRDEM catalogue at all.
    pub fn asset_key(self) -> Option<&'static str> {
        match self {
            Self::Dtm => Some("dtm"),
            Self::Dsm => Some("dsm"),
            Self::Albedo | Self::Osm => None,
        }
    }

    pub fn is_elevation(self) -> bool {
        matches!(self, Self::Dtm | Self::Dsm)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Dtm => "dtm",
            Self::Dsm => "dsm",
            Self::Albedo => "albedo",
            Self::Osm => "osm",
        }
    }
}

/// Which published raster to draw from, finest first. Each tier is tried only
/// for ground the tiers above it left empty.
///
/// The first two are HRDEM, which exists only where a LiDAR survey flew. The
/// third is a different product entirely: MRDEM, a 30 m model that covers
/// Canada without gaps. It is thirty times coarser than the mosaics and is
/// there to put ground under the holes rather than to compete with them --
/// over the Squamish box the first two tiers leave 11% of the ground with no
/// elevation at all, and the renderer draws that as sky you can fly through.
///
/// MRDEM is still bare earth, not a surface model standing in for one: where
/// LiDAR exists it *is* HRDEM resampled to 30 m, and elsewhere it is Copernicus
/// GLO-30 with forest-removal and settlement-removal models applied.
// The shared `Metre` suffix is the unit, not noise: these name ground sample
// distances, and `One`/`Two`/`Thirty` alone would read as ordinals.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolution {
    OneMetre,
    TwoMetre,
    ThirtyMetre,
}

impl Resolution {
    pub const ALL: [Self; 3] = [Self::OneMetre, Self::TwoMetre, Self::ThirtyMetre];

    pub fn collection(self) -> &'static str {
        match self {
            Self::OneMetre => "hrdem-mosaic-1m",
            Self::TwoMetre => "hrdem-mosaic-2m",
            Self::ThirtyMetre => "mrdem-30",
        }
    }

    /// Ground sample distance in metres, which is also the pixel scale the
    /// item's rasters are expected to declare.
    pub fn metres(self) -> f64 {
        match self {
            Self::OneMetre => 1.0,
            Self::TwoMetre => 2.0,
            Self::ThirtyMetre => 30.0,
        }
    }

    /// How this resolution is named in messages to the user.
    pub fn label(self) -> &'static str {
        match self {
            Self::OneMetre => "1 m",
            Self::TwoMetre => "2 m",
            Self::ThirtyMetre => "30 m",
        }
    }
}

/// One raster that overlaps the requested box.
#[derive(Clone, PartialEq, Debug)]
pub struct SourceItem {
    pub id: String,
    pub href: String,
}

#[derive(Deserialize)]
struct FeatureCollection {
    #[serde(default)]
    features: Vec<Feature>,
    #[serde(default)]
    links: Vec<Link>,
}

#[derive(Deserialize)]
struct Feature {
    id: String,
    #[serde(default)]
    properties: Properties,
    #[serde(default)]
    assets: std::collections::HashMap<String, Asset>,
}

#[derive(Deserialize, Default)]
struct Properties {
    #[serde(rename = "proj:epsg")]
    epsg: Option<u32>,
    /// Earth Search tags each scene with its Sentinel-2 grid square.
    #[serde(rename = "grid:code")]
    grid_code: Option<String>,
}

#[derive(Deserialize)]
struct Asset {
    href: String,
}

#[derive(Deserialize)]
struct Link {
    rel: String,
    href: String,
}

/// Guards against a paging loop if the service ever returns a cyclic `next`.
const MAX_PAGES: usize = 32;

/// Fetches one catalogue response, retrying if the network was what failed.
///
/// These are a handful of requests at the very start of a run, so retrying them
/// buys little time -- but a download is often left unattended for half an hour,
/// and failing in the first second because a DNS lookup blinked is the most
/// annoying way for that to be wasted.
async fn get_text(client: &reqwest::Client, url: &str) -> Result<String> {
    retry::retrying(url, retry::is_transient, || async {
        client
            .get(url)
            .send()
            .await?
            .error_for_status()?
            .text()
            .await
    })
    .await
    .with_context(|| format!("requesting {url}"))
}

/// Asks one collection which of its items carry `product` over `box_`.
pub async fn find_items(
    client: &reqwest::Client,
    stac_root: &str,
    resolution: Resolution,
    product: Product,
    box_: LatLonBox,
) -> Result<Vec<SourceItem>> {
    let collection = resolution.collection();
    let mut url = format!(
        "{}/collections/{collection}/items?bbox={}&limit=100",
        stac_root.trim_end_matches('/'),
        box_.to_stac_bbox()
    );

    let mut items = Vec::new();
    for _ in 0..MAX_PAGES {
        let body = get_text(client, &url).await?;

        let page: FeatureCollection = serde_json::from_str(&body)
            .with_context(|| format!("parsing the STAC response from {url}"))?;

        for feature in &page.features {
            items.extend(read_feature(feature, resolution, product)?);
        }

        match page.links.iter().find(|link| link.rel == "next") {
            Some(next) => url = next.href.clone(),
            None => return Ok(items),
        }
    }

    bail!("the STAC service kept offering more pages of {collection} items after {MAX_PAGES}")
}

/// Turns one feature into an item to fetch, or nothing if it lacks the product.
fn read_feature(
    feature: &Feature,
    resolution: Resolution,
    product: Product,
) -> Result<Option<SourceItem>> {
    // Checked even for items that are then skipped: a projection change is a
    // fact about the collection worth failing on, not a per-item detail.
    match feature.properties.epsg {
        Some(EXPECTED_EPSG) => {}
        Some(other) => bail!(
            "item {} is published in EPSG:{other}, but this tool places pixels \
             by assuming EPSG:{EXPECTED_EPSG}",
            feature.id
        ),
        None => bail!("item {} does not say which projection it uses", feature.id),
    }

    let key = product
        .asset_key()
        .expect("only elevation products search this catalogue");
    let Some(asset) = feature.assets.get(key) else {
        // Expected, not exceptional: the satellite-derived northern blocks of
        // the two-metre mosaic publish a surface model and no terrain model.
        log::info!(
            "{} has no {} asset, so it contributes no {} data",
            feature.id,
            key,
            resolution.label()
        );
        return Ok(None);
    };

    Ok(Some(SourceItem {
        id: feature.id.clone(),
        href: asset.href.clone(),
    }))
}

/// Where the cloud-free Sentinel-2 mosaics live.
///
/// These are Earth Genome's annual composites on Source Cooperative. They are
/// the only cloud-free Sentinel-2 mosaics that can be read anonymously: the
/// Copernicus Data Space global mosaics are better data -- native ten metre,
/// quarterly, in the satellite's own UTM zone -- but their assets are `s3://`
/// URIs behind an account, and the download endpoint answers 401 without a
/// token.
///
/// The price of anonymity is resolution. These are reprojected to Web Mercator
/// at 19.1 m, which at fifty degrees north is about 12.4 m of ground, and they
/// have already been resampled once before this tool touches them.
pub const MOSAIC_ROOT: &str = "https://data.source.coop/earthgenome/sentinel2-temporal-mosaics";

/// The years published. Nothing more recent exists.
pub const MOSAIC_YEARS: [u16; 2] = [2022, 2023];

/// Roughly 45 km, in degrees of latitude. The mosaics are cut on the Sentinel-2
/// military grid, whose squares are 100 km, so sampling at least this finely
/// cannot step over one.
const MGRS_SAMPLE_SPACING_DEGREES: f64 = 0.4;

/// Caps the discovery queries for an unreasonably large box.
const MAX_MGRS_QUERIES: usize = 64;

/// The URL of one mosaic tile's true-colour image.
pub fn mosaic_href(tile: &str, year: u16) -> String {
    format!(
        "{MOSAIC_ROOT}/{tile}_{year}-01-01_{}-01-01/TCI.tif",
        year + 1
    )
}

/// Finds which Sentinel-2 grid squares cover the box.
///
/// The mosaics are published per grid square but carry no catalogue of their
/// own -- there is no `catalog.json` to read -- so the squares are discovered
/// from Earth Search, which does catalogue the underlying scenes and tags each
/// with a `grid:code`. Points across the box are queried rather than the box
/// itself, because a single search returns whichever scenes the service feels
/// like returning first and they can all belong to one square.
pub async fn find_mosaic_tiles(
    client: &reqwest::Client,
    earth_search_root: &str,
    box_: LatLonBox,
) -> Result<Vec<String>> {
    let steps = |span: f64| ((span / MGRS_SAMPLE_SPACING_DEGREES).ceil() as usize).max(1);
    let (across, down) = (steps(box_.width_degrees()), steps(box_.height_degrees()));
    anyhow::ensure!(
        across * (down + 1) <= MAX_MGRS_QUERIES,
        "that box spans too much ground to locate its imagery tiles ({} queries)",
        across * down
    );

    let mut tiles: Vec<String> = Vec::new();
    for row in 0..=down {
        for column in 0..=across {
            let longitude = box_.west + box_.width_degrees() * (column as f64 / across as f64);
            let latitude = box_.south + box_.height_degrees() * (row as f64 / down as f64);

            let url = format!(
                "{}/search?collections=sentinel-2-l2a&limit=1&bbox={:.9},{:.9},{:.9},{:.9}",
                earth_search_root.trim_end_matches('/'),
                longitude - 1e-6,
                latitude - 1e-6,
                longitude + 1e-6,
                latitude + 1e-6
            );
            let body = get_text(client, &url).await?;
            let page: FeatureCollection = serde_json::from_str(&body)
                .with_context(|| format!("parsing the Earth Search response from {url}"))?;

            for feature in &page.features {
                if let Some(code) = feature.properties.grid_code.as_deref() {
                    // Earth Search writes it as `MGRS-10UDV`; the mosaics are
                    // named by the bare square.
                    let square = code.strip_prefix("MGRS-").unwrap_or(code).to_string();
                    if !tiles.contains(&square) {
                        tiles.push(square);
                    }
                }
            }
        }
    }

    anyhow::ensure!(
        !tiles.is_empty(),
        "no Sentinel-2 grid square covers that box"
    );
    Ok(tiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trimmed from a real response for `hrdem-mosaic-1m`, keeping the fields
    /// this module reads.
    const ONE_METRE_PAGE: &str = r#"{
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "id": "2_4-mosaic-1m",
          "properties": { "proj:epsg": 3979, "proj:shape": [500000, 500000] },
          "assets": {
            "dsm": { "href": "https://example.invalid/2_4-mosaic-1m-dsm.tif" },
            "dtm": { "href": "https://example.invalid/2_4-mosaic-1m-dtm.tif" },
            "extent": { "href": "https://example.invalid/2_4-mosaic-1m-extent.geojson" }
          }
        }
      ],
      "links": [{ "rel": "self", "href": "https://example.invalid/items" }]
    }"#;

    /// The asset list really published by `4_8-mosaic-2m`: surface model only.
    const SURFACE_ONLY_PAGE: &str = r#"{
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "id": "4_8-mosaic-2m",
          "properties": { "proj:epsg": 3979 },
          "assets": {
            "coverage": { "href": "https://example.invalid/c.gpkg" },
            "dsm": { "href": "https://example.invalid/4_8-mosaic-2m-dsm.tif" },
            "dsm-vrt": { "href": "https://example.invalid/4_8-mosaic-2m-dsm.vrt" },
            "extent": { "href": "https://example.invalid/e.geojson" },
            "hillshade-dsm": { "href": "https://example.invalid/h.tif" },
            "thumbnail": { "href": "https://example.invalid/t.png" }
          }
        }
      ],
      "links": []
    }"#;

    fn parse(text: &str) -> FeatureCollection {
        serde_json::from_str(text).expect("failed to parse")
    }

    fn read_all(text: &str, resolution: Resolution, product: Product) -> Result<Vec<SourceItem>> {
        let page = parse(text);
        let mut items = Vec::new();
        for feature in &page.features {
            items.extend(read_feature(feature, resolution, product)?);
        }
        Ok(items)
    }

    #[test]
    fn the_terrain_asset_is_picked_out_of_a_real_response() {
        let items =
            read_all(ONE_METRE_PAGE, Resolution::OneMetre, Product::Dtm).expect("failed to read");
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].id, "2_4-mosaic-1m");
        assert_eq!(
            items[0].href,
            "https://example.invalid/2_4-mosaic-1m-dtm.tif"
        );
    }

    #[test]
    fn asking_for_the_surface_model_picks_the_other_asset() {
        let items =
            read_all(ONE_METRE_PAGE, Resolution::OneMetre, Product::Dsm).expect("failed to read");
        assert_eq!(
            items[0].href,
            "https://example.invalid/2_4-mosaic-1m-dsm.tif"
        );
    }

    #[test]
    fn an_item_without_the_requested_product_is_skipped_not_fatal() {
        let items = read_all(SURFACE_ONLY_PAGE, Resolution::TwoMetre, Product::Dtm)
            .expect("a missing asset should not be an error");
        assert!(items.is_empty(), "{items:?}");

        // The same item does contribute when the surface model is what is wanted.
        let items = read_all(SURFACE_ONLY_PAGE, Resolution::TwoMetre, Product::Dsm)
            .expect("failed to read");
        assert_eq!(items.len(), 1);
    }

    #[test]
    fn an_item_in_another_projection_is_refused() {
        let moved = ONE_METRE_PAGE.replace("\"proj:epsg\": 3979", "\"proj:epsg\": 4326");
        let error = read_all(&moved, Resolution::OneMetre, Product::Dtm)
            .unwrap_err()
            .to_string();
        assert!(error.contains("EPSG:4326"), "{error}");
        assert!(error.contains("EPSG:3979"), "{error}");
    }

    #[test]
    fn an_item_that_does_not_say_its_projection_is_refused() {
        let silent = ONE_METRE_PAGE.replace("\"proj:epsg\": 3979,", "");
        let error = read_all(&silent, Resolution::OneMetre, Product::Dtm)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not say"), "{error}");
    }

    #[test]
    fn the_bbox_parameter_is_west_south_east_north() {
        let box_ = LatLonBox {
            west: -123.307,
            south: 49.633,
            east: -123.303,
            north: 49.637,
        };
        assert_eq!(
            box_.to_stac_bbox(),
            "-123.307000000,49.633000000,-123.303000000,49.637000000"
        );
    }
}
