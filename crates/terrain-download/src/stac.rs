//! Finding which published rasters cover the requested box.
//!
//! Natural Resources Canada exposes HRDEM through a STAC API. Each collection
//! is split into items that are 500 km squares of the EPSG:3979 grid, and each
//! item carries its rasters as assets. Asking the service which items intersect
//! a longitude/latitude box is a single request, so nothing here has to
//! understand the block naming scheme.
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

/// The projection every HRDEM mosaic item is published in.
pub const EXPECTED_EPSG: u32 = 3979;

/// Which of the two elevation surfaces to fetch.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Product {
    /// Digital terrain model: the bare ground, with vegetation and buildings
    /// removed.
    Dtm,
    /// Digital surface model: the top of whatever the sensor saw.
    Dsm,
}

impl Product {
    /// The key this product appears under in an item's asset map.
    pub fn asset_key(self) -> &'static str {
        match self {
            Self::Dtm => "dtm",
            Self::Dsm => "dsm",
        }
    }
}

/// Which mosaic to draw from. One metre is preferred; two metre is the fallback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolution {
    OneMetre,
    TwoMetre,
}

impl Resolution {
    pub const ALL: [Self; 2] = [Self::OneMetre, Self::TwoMetre];

    pub fn collection(self) -> &'static str {
        match self {
            Self::OneMetre => "hrdem-mosaic-1m",
            Self::TwoMetre => "hrdem-mosaic-2m",
        }
    }

    /// Ground sample distance in metres, which is also the pixel scale the
    /// item's rasters are expected to declare.
    pub fn metres(self) -> f64 {
        match self {
            Self::OneMetre => 1.0,
            Self::TwoMetre => 2.0,
        }
    }

    /// How this resolution is named in messages to the user.
    pub fn label(self) -> &'static str {
        match self {
            Self::OneMetre => "1 m",
            Self::TwoMetre => "2 m",
        }
    }
}

/// One raster that overlaps the requested box.
#[derive(Clone, PartialEq, Debug)]
pub struct SourceItem {
    pub id: String,
    pub href: String,
    pub resolution: Resolution,
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
        let body = client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("requesting {url}"))?
            .error_for_status()
            .with_context(|| format!("requesting {url}"))?
            .text()
            .await
            .with_context(|| format!("reading the response to {url}"))?;

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

    let Some(asset) = feature.assets.get(product.asset_key()) else {
        // Expected, not exceptional: the satellite-derived northern blocks of
        // the two-metre mosaic publish a surface model and no terrain model.
        log::info!(
            "{} has no {} asset, so it contributes no {} data",
            feature.id,
            product.asset_key(),
            resolution.label()
        );
        return Ok(None);
    };

    Ok(Some(SourceItem {
        id: feature.id.clone(),
        href: asset.href.clone(),
        resolution,
    }))
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
        assert_eq!(items[0].resolution, Resolution::OneMetre);
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
