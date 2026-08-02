//! What kind of ground a texel is, as an id the whole project agrees on.
//!
//! The materials product stores one of these per texel where the albedo
//! product stores a colour. The enum lives in this crate for the same reason
//! the grid does: the tool that rasterizes OpenStreetMap ground cover and the
//! renderer that will one day turn ids into shading have to mean the same
//! thing by every number, and a disagreement would draw the wrong terrain
//! without reporting an error.
//!
//! Ids are `u32` and blocked by category -- water `0x01xx`, wetland `0x02xx`,
//! forest `0x03xx`, and so on -- rather than densely packed from 1. The blocks
//! keep room for new members next to their relatives (a new wetland subtype
//! slots into `0x02xx` without renumbering anything), and the category of an
//! unfamiliar id can be read straight off its high byte in a debugger or a
//! hex dump. Discriminants are explicit and are append-only: tiles on disk
//! hold these numbers, so a renumbering would silently repaint the ground.
//!
//! `Null` is zero: the id for ground no mapped area covers, and the value a
//! never-written tile reads back as. Zero also matches the convention that a
//! product's nodata is `0.0` for everything that is not an elevation.

/// One kind of ground cover.
///
/// Variants follow OpenStreetMap's vocabulary, folded to the granularity a
/// terrain texture can use: tags that draw the same ground share a variant,
/// and modifiers that genuinely change the picture -- a needleleaved forest
/// against a broadleaved one -- get their own.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Material {
    /// No mapped area covers this ground.
    Null = 0,

    // Water, 0x01xx. `Ocean` is not a tag: OpenStreetMap implies the sea from
    // `natural=coastline` ways, and the processor fills the water side.
    Ocean = 0x0100,
    Lake = 0x0101,
    Pond = 0x0102,
    River = 0x0103,
    Stream = 0x0104,
    Reservoir = 0x0105,
    Basin = 0x0106,
    Canal = 0x0107,
    Lagoon = 0x0108,
    /// `natural=water` with no `water=` subtype; most are small lakes.
    WaterUnknown = 0x01ff,

    // Wetland, 0x02xx, from `wetland=` on `natural=wetland` areas.
    Marsh = 0x0200,
    Swamp = 0x0201,
    Bog = 0x0202,
    Fen = 0x0203,
    TidalFlat = 0x0204,
    SaltMarsh = 0x0205,
    WetMeadow = 0x0206,
    Reedbed = 0x0207,
    WetlandUnknown = 0x02ff,

    // Forest, 0x03xx, split by `leaf_type` where the mapper recorded one.
    ForestNeedleleaved = 0x0300,
    ForestBroadleaved = 0x0301,
    ForestMixed = 0x0302,
    /// Felled forest, `man_made=clearcut`: stumps and slash, not trees.
    Clearcut = 0x0303,
    ForestUnknown = 0x03ff,

    // Scrub and grass, 0x04xx.
    Scrub = 0x0400,
    Shrubbery = 0x0401,
    Heath = 0x0402,
    Grassland = 0x0403,
    /// Maintained grass: `landuse=grass`, `landcover=grass`.
    Grass = 0x0404,
    Meadow = 0x0405,
    FlowerBed = 0x0406,
    /// High barren tundra above the treeline, `natural=fell`.
    Fell = 0x0407,

    // Bare ground, 0x05xx.
    BareRock = 0x0500,
    Scree = 0x0501,
    Shingle = 0x0502,
    Sand = 0x0503,
    Beach = 0x0504,
    Glacier = 0x0505,
    /// Unvegetated soil: `surface=ground|earth|dirt` areas.
    BareEarth = 0x0506,
    Mud = 0x0507,

    // Agriculture, 0x06xx.
    Farmland = 0x0600,
    Farmyard = 0x0601,
    Orchard = 0x0602,
    Vineyard = 0x0603,
    Allotments = 0x0604,
    PlantNursery = 0x0605,
    Greenhouses = 0x0606,

    // Developed ground, 0x07xx. Broad zones, not individual works: roads and
    // buildings are deliberately absent, waiting on mesh geometry rather than
    // ground texture.
    Residential = 0x0700,
    Commercial = 0x0701,
    Retail = 0x0702,
    Industrial = 0x0703,
    /// Schools, hospitals, civic ground: `landuse=institutional|education|civic`.
    Institutional = 0x0704,
    Religious = 0x0705,
    Railway = 0x0706,
    Military = 0x0707,
    Construction = 0x0708,
    Brownfield = 0x0709,
    Greenfield = 0x070a,
    Landfill = 0x070b,
    Quarry = 0x070c,
    Cemetery = 0x070d,

    // Maintained leisure ground, 0x08xx.
    Park = 0x0800,
    Garden = 0x0801,
    RecreationGround = 0x0802,
    VillageGreen = 0x0803,
    DogPark = 0x0804,
    GolfFairway = 0x0805,
    GolfGreen = 0x0806,
    GolfTee = 0x0807,
    GolfBunker = 0x0808,
    GolfRough = 0x0809,
    PitchGrass = 0x080a,
    PitchArtificial = 0x080b,
    Playground = 0x080c,
}

impl Material {
    /// Every variant, in id order.
    ///
    /// This is the list a counting pass sizes its tables by and the search
    /// space [`Material::try_from_u32`] answers from, so a variant missing
    /// here is a variant that silently cannot be read back.
    pub const ALL: &[Material] = &[
        Material::Null,
        Material::Ocean,
        Material::Lake,
        Material::Pond,
        Material::River,
        Material::Stream,
        Material::Reservoir,
        Material::Basin,
        Material::Canal,
        Material::Lagoon,
        Material::WaterUnknown,
        Material::Marsh,
        Material::Swamp,
        Material::Bog,
        Material::Fen,
        Material::TidalFlat,
        Material::SaltMarsh,
        Material::WetMeadow,
        Material::Reedbed,
        Material::WetlandUnknown,
        Material::ForestNeedleleaved,
        Material::ForestBroadleaved,
        Material::ForestMixed,
        Material::Clearcut,
        Material::ForestUnknown,
        Material::Scrub,
        Material::Shrubbery,
        Material::Heath,
        Material::Grassland,
        Material::Grass,
        Material::Meadow,
        Material::FlowerBed,
        Material::Fell,
        Material::BareRock,
        Material::Scree,
        Material::Shingle,
        Material::Sand,
        Material::Beach,
        Material::Glacier,
        Material::BareEarth,
        Material::Mud,
        Material::Farmland,
        Material::Farmyard,
        Material::Orchard,
        Material::Vineyard,
        Material::Allotments,
        Material::PlantNursery,
        Material::Greenhouses,
        Material::Residential,
        Material::Commercial,
        Material::Retail,
        Material::Industrial,
        Material::Institutional,
        Material::Religious,
        Material::Railway,
        Material::Military,
        Material::Construction,
        Material::Brownfield,
        Material::Greenfield,
        Material::Landfill,
        Material::Quarry,
        Material::Cemetery,
        Material::Park,
        Material::Garden,
        Material::RecreationGround,
        Material::VillageGreen,
        Material::DogPark,
        Material::GolfFairway,
        Material::GolfGreen,
        Material::GolfTee,
        Material::GolfBunker,
        Material::GolfRough,
        Material::PitchGrass,
        Material::PitchArtificial,
        Material::Playground,
    ];

    /// The id as it is stored in a tile.
    pub const fn id(self) -> u32 {
        self as u32
    }

    /// The variant a stored id stands for, or `None` for an id no version of
    /// this enum has assigned.
    ///
    /// A linear search, deliberately: it runs against [`Material::ALL`], so it
    /// cannot disagree with the list, and readers that care about speed build
    /// a table from the list once rather than calling this per texel.
    pub fn try_from_u32(id: u32) -> Option<Material> {
        Material::ALL
            .iter()
            .copied()
            .find(|material| material.id() == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The list is the contract: a variant added to the enum but not to `ALL`
    /// would rasterize fine and then be unreadable, so the count is pinned
    /// here and must be bumped together with any addition.
    #[test]
    fn the_list_holds_every_variant_exactly_once() {
        assert_eq!(Material::ALL.len(), 75);
        let ids: HashSet<u32> = Material::ALL.iter().map(|m| m.id()).collect();
        assert_eq!(ids.len(), Material::ALL.len(), "a duplicate id");
    }

    #[test]
    fn every_id_reads_back_as_its_variant() {
        for &material in Material::ALL {
            assert_eq!(Material::try_from_u32(material.id()), Some(material));
        }
    }

    #[test]
    fn an_unassigned_id_reads_back_as_none() {
        for id in [1, 0x0109, 0x02fe, 0x0900, u32::MAX] {
            assert_eq!(Material::try_from_u32(id), None, "id {id:#x}");
        }
    }

    /// The category is the high byte, which is what makes a raw id in a hex
    /// dump legible; a variant filed under the wrong block would lie about
    /// what kind of ground it is.
    #[test]
    fn ids_sit_in_their_category_blocks() {
        assert_eq!(Material::Null.id(), 0);
        for &material in Material::ALL {
            if material == Material::Null {
                continue;
            }
            let block = material.id() >> 8;
            assert!(
                (1..=8).contains(&block),
                "{material:?} sits outside every block: {:#x}",
                material.id()
            );
        }
        assert_eq!(Material::Ocean.id() >> 8, 1);
        assert_eq!(Material::Marsh.id() >> 8, 2);
        assert_eq!(Material::ForestNeedleleaved.id() >> 8, 3);
        assert_eq!(Material::Scrub.id() >> 8, 4);
        assert_eq!(Material::BareRock.id() >> 8, 5);
        assert_eq!(Material::Farmland.id() >> 8, 6);
        assert_eq!(Material::Residential.id() >> 8, 7);
        assert_eq!(Material::Park.id() >> 8, 8);
    }
}
