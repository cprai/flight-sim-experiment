//! Deciding what ground an OpenStreetMap area's tags describe.
//!
//! OpenStreetMap has no single land-cover key. The same ground arrives as
//! `natural=`, `landuse=`, `leisure=`, `water=`, `wetland=`, `golf=`, or a
//! combination, refined by modifiers like `leaf_type=`, and the vocabulary is
//! whatever mappers actually typed. This module folds that vocabulary onto
//! [`Material`], one function each way: [`classify`] from tags to a material,
//! [`precedence`] from a material to its layer.
//!
//! The mapping was written against a tag census of the extract this project
//! actually uses (BC's South Coast), not against the whole wiki: values that
//! do not occur there are still mapped when obvious, but nothing exotic is
//! guessed at. Unrecognised values are skipped rather than approximated -- a
//! wrong material paints confidently forever, while a skipped area leaves the
//! layer below showing, which is at worst incomplete.
//!
//! Roads and buildings are deliberately not here. They are urban structure
//! rather than ground cover, and will be built as meshes with directional
//! texturing later; painting them flat now would only have to be undone.
//!
//! Precedence orders overlapping areas into layers, because the output is one
//! flat raster: a lake inside a forest inside a residential zone must come out
//! water. Low layers paint first and high layers over them -- broad zones
//! under vegetation, vegetation under specific ground like beaches and golf
//! greens, everything under wetland, wetland under open water. Ocean is the
//! floor: every mapped area sits on top of the coastline fill.

use terrain_materials::Material;

/// The value a tag lookup finds, or nothing.
fn get<'a>(tags: &[(&'a str, &'a str)], key: &str) -> Option<&'a str> {
    tags.iter()
        .find(|(candidate, _)| *candidate == key)
        .map(|(_, value)| *value)
}

/// The material an area with these tags is made of, or `None` when the tags
/// do not describe ground cover this pipeline paints.
///
/// Keys are consulted from most to least specific -- `golf=` before `natural=`
/// before `landuse=` before `leisure=` -- so a fairway inside a wood inside a
/// park classifies as the fairway. `area=no` refuses everything: it is the
/// mapper saying a closed way is a loop of something linear, not a surface.
pub fn classify(tags: &[(&str, &str)]) -> Option<Material> {
    if get(tags, "area") == Some("no") {
        return None;
    }

    if let Some(value) = get(tags, "golf") {
        return golf(value);
    }
    // `water=` sometimes stands alone, and `waterway=riverbank` is the older
    // spelling of `water=river`; both mean open water whatever else is tagged.
    if let Some(value) = get(tags, "water") {
        return Some(water(value));
    }
    if get(tags, "waterway") == Some("riverbank") {
        return Some(Material::River);
    }
    if let Some(value) = get(tags, "natural") {
        return natural(value, tags);
    }
    if let Some(value) = get(tags, "wetland") {
        return Some(wetland(value));
    }
    if get(tags, "man_made") == Some("clearcut") {
        return Some(Material::Clearcut);
    }
    if let Some(value) = get(tags, "landuse") {
        return landuse(value, tags);
    }
    if let Some(value) = get(tags, "leisure") {
        return leisure(value, tags);
    }
    if get(tags, "landcover") == Some("grass") {
        return Some(Material::Grass);
    }
    None
}

fn natural(value: &str, tags: &[(&str, &str)]) -> Option<Material> {
    Some(match value {
        "water" => water(get(tags, "water").unwrap_or("")),
        "wood" => forest(tags),
        "wetland" => wetland(get(tags, "wetland").unwrap_or("")),
        "scrub" => Material::Scrub,
        "shrubbery" => Material::Shrubbery,
        "heath" => Material::Heath,
        "grassland" => Material::Grassland,
        "fell" => Material::Fell,
        "bare_rock" | "rock" | "stone" => Material::BareRock,
        "scree" => Material::Scree,
        "shingle" => Material::Shingle,
        "sand" | "dune" => Material::Sand,
        "beach" => Material::Beach,
        "glacier" => Material::Glacier,
        "mud" => Material::Mud,
        // Named ocean water. These are labels over water the coastline fill
        // already covers, but painting them costs nothing and catches sea
        // the stitching missed.
        "bay" | "strait" => Material::Ocean,
        // Everything else under `natural=` is linear (coastline, cliff,
        // ridge, tree_row) or a point feature dressed as an area, not cover.
        _ => return None,
    })
}

fn water(value: &str) -> Material {
    match value {
        "lake" | "oxbow" => Material::Lake,
        "lagoon" => Material::Lagoon,
        "pond" => Material::Pond,
        "river" => Material::River,
        "stream" | "ditch" | "drain" => Material::Stream,
        "reservoir" => Material::Reservoir,
        "basin" | "wastewater" => Material::Basin,
        "canal" => Material::Canal,
        _ => Material::WaterUnknown,
    }
}

fn wetland(value: &str) -> Material {
    match value {
        "marsh" => Material::Marsh,
        "swamp" => Material::Swamp,
        "bog" | "string_bog" => Material::Bog,
        "fen" => Material::Fen,
        "tidalflat" => Material::TidalFlat,
        "saltmarsh" => Material::SaltMarsh,
        "wet_meadow" => Material::WetMeadow,
        "reedbed" => Material::Reedbed,
        _ => Material::WetlandUnknown,
    }
}

fn forest(tags: &[(&str, &str)]) -> Material {
    match get(tags, "leaf_type").unwrap_or("") {
        "needleleaved" => Material::ForestNeedleleaved,
        "broadleaved" => Material::ForestBroadleaved,
        "mixed" => Material::ForestMixed,
        _ => Material::ForestUnknown,
    }
}

fn landuse(value: &str, tags: &[(&str, &str)]) -> Option<Material> {
    Some(match value {
        "residential" | "garages" => Material::Residential,
        "commercial" => Material::Commercial,
        "retail" => Material::Retail,
        "industrial" | "depot" | "port" => Material::Industrial,
        "institutional" | "education" | "civic" | "governmental" => Material::Institutional,
        "religious" => Material::Religious,
        "railway" => Material::Railway,
        "military" => Material::Military,
        "construction" => Material::Construction,
        "brownfield" => Material::Brownfield,
        "greenfield" => Material::Greenfield,
        "landfill" => Material::Landfill,
        "quarry" => Material::Quarry,
        "cemetery" => Material::Cemetery,
        "grass" => Material::Grass,
        "meadow" | "animal_keeping" => Material::Meadow,
        "forest" => forest(tags),
        "farmland" => Material::Farmland,
        "farmyard" => Material::Farmyard,
        "orchard" => Material::Orchard,
        "vineyard" => Material::Vineyard,
        "allotments" => Material::Allotments,
        "plant_nursery" => Material::PlantNursery,
        "greenhouse_horticulture" => Material::Greenhouses,
        "flowerbed" => Material::FlowerBed,
        "village_green" => Material::VillageGreen,
        "recreation_ground" => Material::RecreationGround,
        "basin" => Material::Basin,
        "reservoir" => Material::Reservoir,
        _ => return None,
    })
}

fn leisure(value: &str, tags: &[(&str, &str)]) -> Option<Material> {
    Some(match value {
        "park" => Material::Park,
        "garden" => Material::Garden,
        "dog_park" => Material::DogPark,
        "playground" => Material::Playground,
        "recreation_ground" => Material::RecreationGround,
        "village_green" | "common" => Material::VillageGreen,
        // The course polygon is the ground between the holes; the parts a
        // golfer would name arrive separately under `golf=`.
        "golf_course" => Material::GolfRough,
        "pitch" | "track" => pitch(tags),
        // Compounds (sports centres, marinas, stadiums) and building-scale
        // furniture (pools, bleachers) are urban structure, not ground cover.
        _ => return None,
    })
}

fn pitch(tags: &[(&str, &str)]) -> Material {
    match get(tags, "surface").unwrap_or("") {
        "artificial_turf" | "tartan" | "acrylic" | "asphalt" | "concrete" | "hard"
        | "rubber" | "clay" | "metal_grid" | "paved" | "paving_stones" | "wood" => {
            Material::PitchArtificial
        }
        _ => Material::PitchGrass,
    }
}

fn golf(value: &str) -> Option<Material> {
    Some(match value {
        "fairway" | "driving_range" => Material::GolfFairway,
        "green" => Material::GolfGreen,
        "tee" => Material::GolfTee,
        "bunker" => Material::GolfBunker,
        "rough" => Material::GolfRough,
        // Water hazards are ponds that happen to punish golfers.
        "water_hazard" | "lateral_water_hazard" => Material::Pond,
        _ => return None,
    })
}

/// Which layer a material paints on. Higher paints later, and so on top.
///
/// The layers, low to high: the ocean floor everything else sits on; broad
/// zones that describe use rather than surface; vegetation; specific ground
/// that replaces whatever grows around it; wetland, which overrides dry
/// cover; and open inland water, which nothing may bury.
pub fn precedence(material: Material) -> u8 {
    use Material::*;
    match material {
        Null | Ocean => 0,
        Residential | Commercial | Retail | Industrial | Institutional | Religious | Railway
        | Military | Brownfield | Greenfield | Farmland | Park | RecreationGround => 1,
        ForestNeedleleaved | ForestBroadleaved | ForestMixed | ForestUnknown | Clearcut
        | Scrub | Shrubbery | Heath | Grassland | Grass | Meadow | Fell | Orchard | Vineyard
        | Allotments | PlantNursery | Greenhouses | Farmyard | Garden | VillageGreen => 2,
        BareRock | Scree | Shingle | Sand | Beach | Glacier | BareEarth | Mud | Quarry
        | Landfill | Construction | Cemetery | FlowerBed | GolfFairway | GolfGreen | GolfTee
        | GolfBunker | GolfRough | PitchGrass | PitchArtificial | Playground | DogPark => 3,
        Marsh | Swamp | Bog | Fen | TidalFlat | SaltMarsh | WetMeadow | Reedbed
        | WetlandUnknown => 4,
        Lake | Pond | River | Stream | Reservoir | Basin | Canal | Lagoon | WaterUnknown => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn of(pairs: &[(&str, &str)]) -> Option<Material> {
        classify(pairs)
    }

    #[test]
    fn woods_split_by_leaf_type_and_fold_landuse_forest_in() {
        assert_eq!(of(&[("natural", "wood")]), Some(Material::ForestUnknown));
        assert_eq!(
            of(&[("natural", "wood"), ("leaf_type", "needleleaved")]),
            Some(Material::ForestNeedleleaved)
        );
        assert_eq!(
            of(&[("landuse", "forest"), ("leaf_type", "broadleaved")]),
            Some(Material::ForestBroadleaved)
        );
        assert_eq!(
            of(&[("landuse", "forest"), ("leaf_type", "mixed")]),
            Some(Material::ForestMixed)
        );
    }

    #[test]
    fn water_takes_its_subtype_from_the_water_key() {
        assert_eq!(of(&[("natural", "water")]), Some(Material::WaterUnknown));
        assert_eq!(
            of(&[("natural", "water"), ("water", "lake")]),
            Some(Material::Lake)
        );
        // `water=` standing alone still means water.
        assert_eq!(of(&[("water", "reservoir")]), Some(Material::Reservoir));
        assert_eq!(of(&[("waterway", "riverbank")]), Some(Material::River));
    }

    #[test]
    fn wetland_subtypes_come_through_and_unknown_ones_stay_wetland() {
        assert_eq!(
            of(&[("natural", "wetland"), ("wetland", "tidalflat")]),
            Some(Material::TidalFlat)
        );
        assert_eq!(
            of(&[("natural", "wetland"), ("wetland", "quagmire")]),
            Some(Material::WetlandUnknown)
        );
        assert_eq!(
            of(&[("natural", "wetland")]),
            Some(Material::WetlandUnknown)
        );
    }

    /// The keys are consulted in specificity order, so a tagged combination
    /// classifies as its most specific part.
    #[test]
    fn more_specific_keys_win_over_broader_ones() {
        assert_eq!(
            of(&[("leisure", "golf_course"), ("golf", "green")]),
            Some(Material::GolfGreen)
        );
        assert_eq!(
            of(&[("landuse", "residential"), ("natural", "wood")]),
            Some(Material::ForestUnknown)
        );
        assert_eq!(
            of(&[("landuse", "basin"), ("water", "basin")]),
            Some(Material::Basin)
        );
    }

    #[test]
    fn pitches_split_by_surface() {
        assert_eq!(of(&[("leisure", "pitch")]), Some(Material::PitchGrass));
        assert_eq!(
            of(&[("leisure", "pitch"), ("surface", "artificial_turf")]),
            Some(Material::PitchArtificial)
        );
        assert_eq!(
            of(&[("leisure", "pitch"), ("surface", "grass")]),
            Some(Material::PitchGrass)
        );
    }

    /// Linear features and urban structure must not become ground cover: a
    /// cliff is a line, a pool is furniture, a building is a later mesh.
    #[test]
    fn linear_and_structural_features_do_not_classify() {
        assert_eq!(of(&[("natural", "cliff")]), None);
        assert_eq!(of(&[("natural", "coastline")]), None);
        assert_eq!(of(&[("natural", "tree_row")]), None);
        assert_eq!(of(&[("leisure", "swimming_pool")]), None);
        assert_eq!(of(&[("building", "yes")]), None);
        assert_eq!(of(&[("highway", "pedestrian")]), None);
    }

    #[test]
    fn area_no_refuses_whatever_else_is_tagged() {
        assert_eq!(of(&[("natural", "wood"), ("area", "no")]), None);
    }

    #[test]
    fn named_sea_water_is_ocean() {
        assert_eq!(of(&[("natural", "bay")]), Some(Material::Ocean));
        assert_eq!(of(&[("natural", "strait")]), Some(Material::Ocean));
    }

    /// Every material the classifier can produce has a layer, and the layers
    /// order the way the module doc promises: zones under vegetation under
    /// specific ground under wetland under water, with ocean at the floor.
    #[test]
    fn precedence_orders_the_layers_the_ground_needs() {
        assert!(precedence(Material::Ocean) < precedence(Material::Residential));
        assert!(precedence(Material::Residential) < precedence(Material::ForestUnknown));
        assert!(precedence(Material::ForestUnknown) < precedence(Material::Beach));
        assert!(precedence(Material::Beach) < precedence(Material::Marsh));
        assert!(precedence(Material::Marsh) < precedence(Material::Lake));
        // The pair the plan calls out: a lake inside a forest stays a lake.
        assert!(precedence(Material::Lake) > precedence(Material::ForestMixed));
        for &material in Material::ALL {
            assert!(precedence(material) <= 5, "{material:?}");
        }
    }
}
