//! What kind of ground a texel is, read off the same fields that shaped it.
//!
//! The measured pipeline gets its ground cover from OpenStreetMap: somebody
//! walked or traced every wood and every lake. There is nobody to ask here, so
//! the cover has to be deduced from the landscape -- and it can be, because in
//! mountain country the landscape is most of the answer. Height sets the
//! treeline and the snowline; slope decides whether anything can root at all;
//! drainage says where the water is; and the erosion passes have already
//! recorded where material was cut away and where it piled up.
//!
//! That is the whole of the method, and it is why this reads the channels the
//! simulation left rather than inventing a second set of noise fields: cover
//! that disagrees with the ground it sits on is the thing that makes a
//! generated landscape look wrong even when nobody can say why.
//!
//! Only natural cover is produced. Nothing in the agriculture, developed or
//! leisure blocks of [`Material`] can come out of here -- there is nobody in
//! this world to have built any of it.
//!
//! Like `detail`, this is a pure function of position and a sample, written to
//! port to WGSL: the same fixed loop bounds, the same `f32` arithmetic, and no
//! branch that depends on anything but the numbers passed in.

use terrain_materials::Material;

use crate::detail::{CHANNEL_FLOW, Ground};
use crate::fields::Sample;
use crate::noise::{Fractal, fbm, smoothstep};
use crate::shape::Relief;

/// Where the treeline and the permanent snow sit, as fractions of the relief.
///
/// In the Rockies both are real altitudes -- roughly 2200 m and 2900 m -- but
/// pinning them to metres would put a landscape asked for a smaller range
/// entirely below the treeline or entirely above it. As fractions they follow
/// whatever `--peak-metres` and `--valley-metres` were asked for.
///
/// They sit lower in the relief than the real ratio suggests, and that is the
/// regional fall's doing rather than an error. The ground drops a kilometre
/// from one end of the map to the other, so the peaks at the low end stand a
/// long way below the peaks at the high end; lines set at the real fractions
/// put the whole southern half of the landscape under the treeline and left it
/// a forest with no mountains in it.
const TREELINE_SHARE: f32 = 0.52;
const SNOWLINE_SHARE: f32 = 0.76;

/// How far the lines wander, as a share of the relief, and over what distance.
///
/// A treeline that ran at one altitude would draw as a contour line round every
/// mountain in the landscape. Real ones wander by a couple of hundred metres
/// with shelter, soil and wind.
const LINE_WAVELENGTH: f32 = 2_600.0;
const LINE_SHARE: f32 = 0.05;

/// How much higher a line runs on a slope facing the sun, as a share of the
/// relief.
///
/// South-facing slopes in the northern hemisphere carry trees a long way above
/// the shaded side of the same ridge, and the asymmetry is one of the most
/// visible things about a real range from the air.
const ASPECT_SHARE: f32 = 0.063;

/// How far below the treeline the forest starts giving out into scrub, as a
/// share of the relief.
const KRUMMHOLZ_SHARE: f32 = 0.137;

/// How wide the change from wooded to alpine rock is, as a share of the relief.
const ROCK_BAND_SHARE: f32 = 0.079;

/// Feature size of the mottling that breaks up every boundary.
///
/// Band-limited like any other octave, so a coarse level gets a smoother
/// boundary rather than a boundary that flickers.
const MOTTLE_WAVELENGTH: f32 = 140.0;
const MOTTLE_OCTAVES: u32 = 4;

/// How steep ground can be and still hold a glacier rather than shed it.
const ICE_STEEPNESS: f32 = 0.45;

/// How rocky ground has to be before nothing grows on it at all, below the
/// treeline and above it.
///
/// Two numbers rather than one, because the same slope means different things
/// at different heights. Below the treeline there is soil, and conifers root in
/// it on ground far steeper than anyone would walk up -- it takes a genuine
/// cliff to leave rock showing. Above it there is no soil to hold, and a slope
/// that would carry forest lower down carries nothing at all.
///
/// One threshold for both painted better than a third of a wooded landscape
/// grey.
const ROCK_THRESHOLD_WOODED: f32 = 0.86;
const ROCK_THRESHOLD_ALPINE: f32 = 0.42;

/// How much loose material a steep slope needs before it reads as talus.
const SCREE_STEEPNESS: f32 = 0.30;
const SCREE_FILLING: f32 = 0.22;

/// The heights and lines a landscape's cover is measured against.
///
/// Computed once per texel and passed around rather than recomputed per test,
/// because every branch below wants at least one of them.
struct Lines {
    treeline: f32,
    snowline: f32,
    /// The relief the whole landscape spans, in metres.
    ///
    /// Every band below is a share of it rather than a fixed number of metres.
    /// A fixed one is right for the range this crate generates by default and
    /// wrong for every other: a two-hundred-and-sixty-metre scrub belt under
    /// the treeline is a detail on a landscape spanning two kilometres and most
    /// of a landscape spanning four hundred metres.
    span: f32,
    /// Where this texel sits in the relief, `0` at the valley floor and `1` at
    /// the peak.
    band: f32,
    /// A `-1..=1` field that every threshold is nudged by.
    mottle: f32,
}

fn lines(
    sample: &Sample,
    ground: &Ground,
    x: f32,
    y: f32,
    texel_metres: f32,
    seed: u32,
    relief: Relief,
) -> Lines {
    let wobble = fbm(x, y, seed ^ 0x4b19_c2e7, Fractal::new(LINE_WAVELENGTH, 3));
    // The southward component of downhill is how much the slope faces the sun.
    let sun = sample.aspect[1] * ground.steepness;
    let base = relief.valley_metres;
    let span = relief.span();
    Lines {
        span,
        treeline: base + span * (TREELINE_SHARE + wobble * LINE_SHARE + sun * ASPECT_SHARE),
        snowline: base
            + span * (SNOWLINE_SHARE + wobble * LINE_SHARE * 0.6 + sun * ASPECT_SHARE * 1.2),
        band: ((sample.height - base) / span.max(1.0)).clamp(0.0, 1.0),
        mottle: fbm(
            x,
            y,
            seed ^ 0xa71f_63b9,
            Fractal::new(MOTTLE_WAVELENGTH, MOTTLE_OCTAVES).band_limited(2.0 * texel_metres),
        ),
    }
}

/// What grows above the treeline, which is a question of how far above.
fn alpine(sample: &Sample, ground: &Ground, lines: &Lines) -> Material {
    let bare = smoothstep(
        lines.treeline,
        lines.snowline,
        sample.height + lines.mottle * lines.span * LINE_SHARE,
    );
    if bare > 0.55 {
        // High barren tundra: the ground between the last shrub and the snow.
        Material::Fell
    } else if ground.steepness > 0.25 || sample.hardness > 0.6 {
        Material::Heath
    } else {
        Material::Grassland
    }
}

/// What grows below the treeline, on ground dry enough to hold it.
fn wooded(sample: &Sample, ground: &Ground, lines: &Lines) -> Material {
    // A band just under the treeline where the forest gives out into scrub.
    let krummholz = smoothstep(
        lines.treeline - lines.span * KRUMMHOLZ_SHARE,
        lines.treeline,
        sample.height + lines.mottle * lines.span * LINE_SHARE * 0.75,
    );
    if krummholz > 0.6 {
        return Material::Scrub;
    }
    // A Rocky Mountain slope is conifer, and overwhelmingly so. Broadleaves --
    // aspen and cottonwood -- come in on the low, wet, sheltered ground and in
    // patches after a disturbance, and mixed stands are the transition between
    // the two. Both are qualified by the mottling as well as by height, so they
    // arrive as stands rather than as an altitude band: a "mixed forest" belt
    // ringing every mountain at one height is the tell of a rule that keyed off
    // elevation alone.
    let low = lines.band < 0.30 + lines.mottle * 0.08;
    let wet = sample.flow > CHANNEL_FLOW - 3.5;
    let sunny = sample.aspect[1] > 0.25;
    if low && wet && ground.steepness < 0.3 {
        Material::ForestBroadleaved
    } else if low && (sunny || lines.mottle > 0.25) {
        Material::ForestMixed
    } else {
        Material::ForestNeedleleaved
    }
}

/// What covers a flat valley floor the water has been working on.
fn floor(sample: &Sample, ground: &Ground, lines: &Lines) -> Material {
    if ground.channel > 0.04 && ground.filling > 0.45 {
        // The gravel bars a braided river leaves either side of itself.
        return Material::Shingle;
    }
    if ground.filling > 0.7 && lines.mottle > 0.45 {
        // Outwash: the finest of what the water carried, dropped last.
        return Material::Sand;
    }
    if ground.filling < 0.1 && sample.slope > 0.12 && lines.mottle < -0.5 {
        // A bank the water has cut back to soil.
        return Material::BareEarth;
    }
    if lines.mottle > 0.15 {
        Material::Meadow
    } else {
        Material::Grassland
    }
}

/// The ground cover of one texel.
///
/// `x` and `y` are raster metres, as everywhere else in the per-texel half.
pub fn material(
    sample: &Sample,
    ground: &Ground,
    x: f32,
    y: f32,
    texel_metres: f32,
    seed: u32,
    relief: Relief,
) -> Material {
    let lines = lines(sample, ground, x, y, texel_metres, seed, relief);

    // Water first: it covers whatever is underneath it.
    if ground.lake > 0.5 {
        return Material::Lake;
    }
    if ground.channel > 0.62 {
        return Material::River;
    }
    if ground.channel > 0.28 {
        return Material::Stream;
    }
    // The margin of a lake, and the flat wet ground that never quite drains.
    let boggy = sample.slope < 0.035 && sample.flow > CHANNEL_FLOW - 2.5;
    if ground.lake > 0.12 || (boggy && ground.filling > 0.25) {
        return if sample.height > lines.treeline {
            Material::Bog
        } else {
            Material::Marsh
        };
    }

    // Permanent ice, which needs height and a slope shallow enough to hold it.
    if sample.height > lines.snowline + lines.mottle * lines.span * LINE_SHARE
        && ground.steepness < ICE_STEEPNESS
    {
        return Material::Glacier;
    }

    // Rock, and the talus under it. Both are about the ground rather than the
    // climate, so they come before the treeline is consulted at all -- a cliff
    // is bare at any altitude.
    let band = lines.span * ROCK_BAND_SHARE;
    let above_the_trees = smoothstep(lines.treeline - band, lines.treeline + band, sample.height);
    let bare = crate::noise::lerp(
        ROCK_THRESHOLD_WOODED,
        ROCK_THRESHOLD_ALPINE,
        above_the_trees,
    );
    if ground.rockiness > bare + lines.mottle * 0.10 {
        return Material::BareRock;
    }
    if ground.steepness > SCREE_STEEPNESS && ground.filling > SCREE_FILLING {
        return Material::Scree;
    }

    if sample.height > lines.treeline {
        return alpine(sample, ground, &lines);
    }
    // Flat, worked-over valley bottom, as against a wooded hillside.
    if sample.slope < 0.09 && lines.band < 0.35 {
        return floor(sample, ground, &lines);
    }
    wooded(sample, ground, &lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detail::ground as ground_of;

    fn relief() -> Relief {
        Relief {
            valley_metres: 700.0,
            peak_metres: 2600.0,
        }
    }

    fn classify(sample: &Sample) -> Material {
        let ground = ground_of(sample, 4.0, 16.0);
        material(sample, &ground, 1234.0, 5678.0, 4.0, 9, relief())
    }

    /// Dry ground at a height and a slope, and nothing else going on.
    ///
    /// A constructor rather than a literal to spread with, because `filled`
    /// tracks `height` on dry ground: overriding one and not the other quietly
    /// floods the sample under kilometres of water, and every test downstream
    /// then asserts something about a lake.
    fn dry(height: f32, slope: f32) -> Sample {
        Sample {
            height,
            hardness: 0.5,
            flow: 2.0,
            deposit: 0.0,
            filled: height,
            slope,
            aspect: [0.0, 0.0],
        }
    }

    /// Every combination of channel values has to name a material. A gap would
    /// not be an error -- it would be a `Null` texel, which the renderer draws
    /// as magenta.
    #[test]
    fn every_landscape_is_classified_as_something() {
        let mut seen = std::collections::HashSet::new();
        for height in [700.0f32, 1100.0, 1500.0, 1900.0, 2300.0, 2600.0] {
            for slope in [0.0f32, 0.05, 0.2, 0.5, 1.0, 2.0] {
                for flow in [0.0f32, 6.0, 10.0, 13.0, 17.0] {
                    for deposit in [-8.0f32, 0.0, 2.0, 10.0] {
                        for hardness in [0.0f32, 0.5, 1.0] {
                            for depth in [0.0f32, 0.2, 5.0] {
                                let sample = Sample {
                                    height,
                                    hardness,
                                    flow,
                                    deposit,
                                    filled: height + depth,
                                    slope,
                                    aspect: [0.6, -0.8],
                                };
                                let material = classify(&sample);
                                assert_ne!(
                                    material,
                                    Material::Null,
                                    "nothing classified {sample:?}"
                                );
                                seen.insert(material.id());
                            }
                        }
                    }
                }
            }
        }
        assert!(
            seen.len() >= 10,
            "only {} materials ever came out; the classifier has dead branches",
            seen.len()
        );
    }

    /// Nobody built anything in this world. Farmland, roads, buildings and
    /// pitches all have ids, and a stray one would paint a car park onto a
    /// mountainside.
    #[test]
    fn nothing_built_by_anybody_is_ever_painted() {
        for height in [700.0f32, 1400.0, 2100.0, 2600.0] {
            for slope in [0.0f32, 0.1, 0.4, 1.2, 3.0] {
                for flow in [0.0f32, 8.0, 12.0, 16.0, 20.0] {
                    for deposit in [-20.0f32, 0.0, 30.0] {
                        for depth in [0.0f32, 1.0, 40.0] {
                            for aspect in [[0.0f32, 1.0], [0.0, -1.0], [1.0, 0.0]] {
                                let sample = Sample {
                                    height,
                                    hardness: 0.4,
                                    flow,
                                    deposit,
                                    filled: height + depth,
                                    slope,
                                    aspect,
                                };
                                let block = classify(&sample).id() >> 8;
                                assert!(
                                    (0..=5).contains(&block),
                                    "{:?} is in block {block:#x}, which is something built",
                                    classify(&sample)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The bands that make a mountain look like a mountain: forest at the
    /// bottom, then bare alpine ground, then ice.
    #[test]
    fn cover_changes_with_height_in_the_order_a_mountain_does() {
        let low = classify(&dry(900.0, 0.2));
        assert_eq!(low.id() >> 8, 0x03, "{low:?} is not forest");

        let high = classify(&dry(2100.0, 0.2));
        assert!(
            matches!(high, Material::Fell | Material::Heath | Material::Grassland),
            "{high:?} is not alpine ground"
        );

        let top = classify(&dry(2580.0, 0.05));
        assert_eq!(top, Material::Glacier, "the summit is {top:?}");
    }

    #[test]
    fn standing_water_is_a_lake_and_a_trunk_drainage_is_a_river() {
        let lake = classify(&Sample {
            filled: 1210.0,
            ..dry(1200.0, 0.01)
        });
        assert_eq!(lake, Material::Lake);

        let river = classify(&Sample {
            flow: 18.0,
            ..dry(1200.0, 0.02)
        });
        assert_eq!(river, Material::River);
    }

    /// A cliff is bare at any altitude, which is why rock is decided before the
    /// treeline is consulted.
    #[test]
    fn a_cliff_is_bare_rock_wherever_it_is() {
        for height in [800.0f32, 1400.0, 2000.0, 2500.0] {
            let cliff = classify(&Sample {
                hardness: 1.0,
                ..dry(height, 2.0)
            });
            assert_eq!(cliff, Material::BareRock, "at {height} m");
        }
    }

    /// Loose material on a steep slope is talus, and talus is what a
    /// mountainside is mostly made of below its cliffs.
    #[test]
    fn loose_material_on_a_steep_slope_is_scree() {
        let talus = classify(&Sample {
            hardness: 0.1,
            deposit: 4.0,
            ..dry(2000.0, 0.55)
        });
        assert_eq!(talus, Material::Scree);
    }

    /// The lines have to wander, or they draw as contours round every mountain.
    #[test]
    fn the_treeline_is_not_a_contour_line() {
        let sample = dry(1870.0, 0.25);
        let ground = ground_of(&sample, 4.0, 16.0);
        let mut kinds = std::collections::HashSet::new();
        for step in 0..400 {
            let x = step as f32 * 60.0;
            kinds.insert(material(&sample, &ground, x, 3000.0, 4.0, 9, relief()).id());
        }
        assert!(
            kinds.len() > 1,
            "the same material all the way along a line at the treeline"
        );
    }

    /// The relief scales the landscape, so the lines have to scale with it --
    /// otherwise a range asked to span 300 m would come out entirely alpine or
    /// entirely wooded.
    #[test]
    fn the_lines_follow_the_relief_they_were_given() {
        let small = Relief {
            valley_metres: 0.0,
            peak_metres: 400.0,
        };
        let sample = dry(100.0, 0.2);
        let ground = ground_of(&sample, 4.0, 16.0);
        let low = material(&sample, &ground, 100.0, 100.0, 4.0, 9, small);
        assert_eq!(low.id() >> 8, 0x03, "{low:?} is not forest near the valley");

        let high = dry(380.0, 0.2);
        let alpine = material(&high, &ground, 100.0, 100.0, 4.0, 9, small);
        assert_ne!(
            alpine.id() >> 8,
            0x03,
            "{alpine:?} is forest above the line"
        );
    }
}
