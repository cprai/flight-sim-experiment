//! One flat colour per ground-cover material, as the shading pass's table.
//!
//! This is a stand-in for real material shading: enough to see the ground
//! cover the pipeline painted, not what any of it will finally look like.
//! Each material gets a single colour chosen by hand, hue by category --
//! waters blue, forests dark green, developed ground grey -- so that an
//! aerial view reads like a land-cover map.
//!
//! The colours live in an exhaustive `match`: adding a material to
//! `terrain-materials` without deciding its colour is a compile error here,
//! not a black pixel at runtime. Ids the enum has *not* assigned -- `Null`,
//! and any gap inside the category blocks -- stay pure magenta, the colour
//! of missing data, which no real material is allowed to resemble.
//!
//! [`build`] lays the table out indexed by id, one slot per possible id up
//! to the last assigned block, so the shader turns an id into a colour with
//! one fetch and no search. The entries are stored linearised: the render
//! target is sRGB and re-encodes on write, so encoded bytes would otherwise
//! be encoded twice and the ground would wash out.

use terrain_materials::Material;
use terrain_tiles::srgb_to_linear;

/// One slot per id from 0 to one past the highest assigned id (`0x08ff`).
///
/// Kept in step with `PALETTE_SIZE` in `src/shading.wgsl`, and pinned by a
/// test below: a new `0x09xx` category block must bump both together.
pub const PALETTE_SIZE: usize = 0x0900;

/// The colour of missing data. Never a plausible ground colour, which is
/// the point: a magenta hillside is a bug report, not scenery.
pub const MAGENTA: [u8; 3] = [255, 0, 255];

/// The flat colour a material draws as, sRGB-encoded.
///
/// `pub(crate)` so the scene tests can ask what a material should look like
/// on screen instead of hard-coding a second copy of the table.
pub(crate) fn flat_colour(material: Material) -> [u8; 3] {
    use Material::*;
    match material {
        Null => MAGENTA,

        // Water: blues, darker the bigger and deeper the body.
        Ocean => [10, 45, 95],
        Lake => [30, 80, 140],
        Pond => [45, 95, 150],
        River => [40, 95, 150],
        Stream => [70, 120, 170],
        Reservoir => [35, 85, 145],
        Basin => [55, 100, 150],
        Canal => [50, 105, 160],
        Lagoon => [45, 110, 160],
        WaterUnknown => [40, 90, 145],

        // Wetland: the teal-olives of ground that is neither land nor water.
        Marsh => [90, 125, 95],
        Swamp => [70, 110, 85],
        Bog => [110, 115, 75],
        Fen => [100, 130, 90],
        TidalFlat => [150, 140, 115],
        SaltMarsh => [130, 140, 110],
        WetMeadow => [110, 140, 95],
        Reedbed => [125, 135, 80],
        WetlandUnknown => [95, 125, 90],

        // Forest: dark greens, needles darker than leaves.
        ForestNeedleleaved => [25, 70, 40],
        ForestBroadleaved => [50, 105, 50],
        ForestMixed => [38, 88, 45],
        Clearcut => [115, 95, 65],
        // The crowns, and the only entry here that is never ground. Darker than
        // any of the floors above, because a canopy from above is mostly its
        // own shadow: the light that reaches a treetop is the light that has
        // not already been caught by the branches beside it.
        Canopy => [28, 62, 36],
        ForestUnknown => [40, 90, 45],

        // Scrub and grass: lighter, yellower greens than any forest.
        Scrub => [105, 125, 70],
        Shrubbery => [95, 135, 75],
        Heath => [130, 120, 85],
        Grassland => [140, 165, 85],
        Grass => [110, 160, 80],
        Meadow => [125, 160, 85],
        FlowerBed => [160, 135, 105],
        Fell => [150, 150, 120],

        // Bare ground: greys and tans, ice nearly white.
        BareRock => [130, 130, 130],
        Scree => [150, 145, 135],
        Shingle => [160, 155, 140],
        Sand => [210, 190, 140],
        Beach => [220, 200, 150],
        Glacier => [235, 245, 250],
        BareEarth => [140, 115, 85],
        Mud => [130, 105, 80],

        // Agriculture: the yellows of worked ground.
        Farmland => [190, 170, 95],
        Farmyard => [175, 150, 100],
        Orchard => [120, 150, 70],
        Vineyard => [110, 140, 75],
        Allotments => [130, 150, 85],
        PlantNursery => [115, 155, 90],
        Greenhouses => [200, 205, 195],

        // Developed ground: greys, with a little of what the zone is for.
        Residential => [160, 155, 150],
        Commercial => [150, 145, 155],
        Retail => [165, 150, 145],
        Industrial => [120, 115, 120],
        Institutional => [155, 150, 140],
        Religious => [145, 140, 135],
        Railway => [95, 90, 95],
        Military => [110, 110, 95],
        Construction => [170, 150, 120],
        Brownfield => [140, 120, 100],
        Greenfield => [150, 160, 120],
        Landfill => [125, 110, 90],
        Quarry => [150, 140, 130],
        Cemetery => [110, 130, 105],
        Paved => [75, 75, 78],
        Building => [178, 170, 162],

        // Maintained leisure ground: the brightest greens on the map,
        // except the bunkers, which are sand.
        Park => [90, 170, 85],
        Garden => [110, 175, 95],
        RecreationGround => [100, 165, 90],
        VillageGreen => [105, 170, 85],
        DogPark => [120, 165, 95],
        GolfFairway => [80, 160, 70],
        GolfGreen => [60, 150, 60],
        GolfTee => [70, 155, 65],
        GolfBunker => [215, 195, 145],
        GolfRough => [95, 145, 75],
        PitchGrass => [85, 160, 80],
        PitchArtificial => [70, 140, 110],
        Playground => [180, 160, 130],
    }
}

/// The id-indexed colour table the shading pass uploads once.
///
/// Every slot starts magenta and only assigned ids are overwritten, so an
/// id from a newer enum than this binary -- or a corrupt texel -- draws as
/// missing data rather than as whatever the neighbouring slot holds.
pub fn build() -> Vec<[f32; 4]> {
    let linearise = |colour: [u8; 3]| {
        [
            srgb_to_linear(colour[0]),
            srgb_to_linear(colour[1]),
            srgb_to_linear(colour[2]),
            1.0,
        ]
    };
    let mut table = vec![linearise(MAGENTA); PALETTE_SIZE];
    for &material in Material::ALL {
        table[material.id() as usize] = linearise(flat_colour(material));
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table is indexed by id, so every id has to fit in it; an id past
    /// the end means a new category block landed and both this constant and
    /// the copy in `shading.wgsl` need bumping.
    #[test]
    fn every_material_id_fits_in_the_palette() {
        for &material in Material::ALL {
            assert!(
                (material.id() as usize) < PALETTE_SIZE,
                "{material:?} is {:#x}",
                material.id()
            );
        }
    }

    #[test]
    fn null_and_unassigned_ids_are_magenta() {
        let table = build();
        let magenta = [
            srgb_to_linear(255),
            srgb_to_linear(0),
            srgb_to_linear(255),
            1.0,
        ];
        assert_eq!(table[0], magenta, "Null");
        assert_eq!(table[0x0109], magenta, "an id inside the water block gap");
        assert_eq!(table[0x08ff], magenta, "the last slot");
    }

    /// Magenta means missing data and nothing else may borrow the meaning.
    #[test]
    fn no_real_material_resembles_magenta() {
        for &material in Material::ALL {
            if material == Material::Null {
                continue;
            }
            let [r, g, b] = flat_colour(material);
            assert!(
                !(r > 200 && b > 200 && g < 60),
                "{material:?} is {:?}, which reads as magenta",
                flat_colour(material)
            );
        }
    }

    /// Ground that exactly matched the sky would read as holes in the
    /// terrain. The waters are blue on purpose, but none of them may be
    /// *this* blue -- the constant is `CLEAR_COLOR` in `src/scene.rs`
    /// through the sRGB encoding.
    #[test]
    fn no_material_wears_the_skys_exact_colour() {
        use terrain_tiles::linear_to_srgb;
        let sky = [
            linear_to_srgb(0.30),
            linear_to_srgb(0.55),
            linear_to_srgb(0.85),
        ];
        for &material in Material::ALL {
            assert_ne!(flat_colour(material), sky, "{material:?}");
        }
    }
}
