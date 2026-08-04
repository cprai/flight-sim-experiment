//! The height of one texel: the coarse landscape, amplified.
//!
//! This is the second half of the generator and the whole of what a texel is.
//! It takes a position, the size of the texel at that position, and a sample of
//! the channels the simulation left, and returns metres. Nothing else: no
//! neighbours, no state, no ordering. Two consequences follow from that and
//! both are load-bearing.
//!
//! **Tiles cannot have seams.** Two tiles that meet evaluate the same function
//! at the same positions, so they agree by construction rather than by
//! overlapping and blending.
//!
//! **Levels are band-limited rather than filtered.** A coarse level is not the
//! box filter of the fine one; it is the same function with the octaves its
//! texels cannot represent left out. That is a better answer than filtering --
//! it is what filtering is trying to approximate -- and it means the pyramid is
//! generated rather than reduced, at four thirds of the cost of the base level.
//!
//! The detail is not decoration. What it adds is decided by what the simulation
//! found: ribs and buttresses on steep hard rock, smooth ground where sediment
//! was dropped, dead flat where water stands. A uniform layer of noise over the
//! whole landscape would undo the erosion rather than finish it.

use crate::fields::Sample;
use crate::noise::{Fractal, billow, fbm, lerp, ridged, smoothstep};

/// Feature size and depth of the fine texture over everything.
const TEXTURE_WAVELENGTH: f32 = 512.0;
const TEXTURE_OCTAVES: u32 = 10;
const TEXTURE_METRES: f32 = 5.0;

/// Feature size and depth of the ribs that run down a rock face.
///
/// Ridged rather than plain, because what a mountain face is made of is
/// buttresses with gullies between them, and those are creases -- lines --
/// rather than the round lumps a Brownian fractal gives.
const RIB_WAVELENGTH: f32 = 380.0;
const RIB_OCTAVES: u32 = 6;
const RIB_METRES: f32 = 15.0;

/// Feature size and depth of the hummocks on ground the water filled.
///
/// Billowy rather than ridged, because a moraine is a heap of what was carried
/// and dropped: rounded mounds with hollows between them, which is the exact
/// opposite of the creases a face is cut into. Only appears where the deposit
/// channel says material actually piled up, so it draws valley floors and
/// alluvial fans and nothing else.
const MORAINE_WAVELENGTH: f32 = 220.0;
const MORAINE_OCTAVES: u32 = 4;
const MORAINE_METRES: f32 = 6.0;

/// Where a slope stops counting as ground and starts counting as rock, as a
/// rise over run. About 14 to 42 degrees.
const GENTLE_SLOPE: f32 = 0.25;
const STEEP_SLOPE: f32 = 0.90;

/// How much sediment has to have piled up before the ground reads as filled.
const FILLED_METRES: f32 = 6.0;

/// How deep the water has to be before the surface is the water's rather than
/// the ground's.
///
/// The band between the two is the shoreline, and it has to be a band: a hard
/// cut-off would draw every lake with a one-texel cliff round it.
pub const SHORE_METRES: f32 = 0.4;
pub const LAKE_METRES: f32 = 2.5;

/// The drainage a channel starts at and the drainage it is fully cut at, as
/// `log2` of a cell count.
///
/// Between them is the difference between a gully that happens to collect water
/// and a river with a bed.
pub const CHANNEL_FLOW: f32 = 11.5;
pub const RIVER_FLOW: f32 = 15.5;

/// How deep a full river cuts below the ground around it.
const CHANNEL_METRES: f32 = 4.0;

/// How wide a channel is, in metres.
///
/// Not a free choice: the drainage channel is interpolated out of the coarse
/// grid, so its width is roughly a couple of those cells. It is named here
/// because the channel has to fade out on levels whose texels are wider than
/// it, exactly as an octave of noise does -- a two-cell feature sampled every
/// 256 m is not a river, it is a flicker that changes with the camera.
const CHANNEL_WIDTH_CELLS: f32 = 2.5;

/// How much of a feature survives at a given texel size, `0..=1`.
///
/// The same rule the fractals band-limit themselves by, applied to a feature
/// that comes out of the simulation rather than out of noise.
fn resolvable(feature_metres: f32, texel_metres: f32) -> f32 {
    smoothstep(0.5, 1.5, feature_metres / (2.0 * texel_metres))
}

/// What kind of ground this is, as the three fractions the detail and the
/// material classifier both key off.
///
/// Shared between the two so that the rock the height function roughens is the
/// rock the classifier paints. Deriving them twice would let the two drift, and
/// a drift shows as scree colour on ground with no scree on it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Ground {
    /// How steep this is, `0..=1`.
    pub steepness: f32,
    /// How rocky: steep, and hard enough to stand rather than slump.
    pub rockiness: f32,
    /// How much loose material has piled here, `0..=1`.
    pub filling: f32,
    /// How much of the texel is under standing water, `0..=1`.
    pub lake: f32,
    /// How much of a river channel runs through it, `0..=1`.
    pub channel: f32,
}

/// Reads the ground kind off a sample.
pub fn ground(sample: &Sample, texel_metres: f32, cell_metres: f32) -> Ground {
    let steepness = smoothstep(GENTLE_SLOPE, STEEP_SLOPE, sample.slope);
    let lake = smoothstep(SHORE_METRES, LAKE_METRES, sample.water_depth());
    let channel = smoothstep(CHANNEL_FLOW, RIVER_FLOW, sample.flow)
        * (1.0 - lake)
        * resolvable(CHANNEL_WIDTH_CELLS * cell_metres, texel_metres);
    Ground {
        steepness,
        rockiness: steepness * (0.35 + 0.65 * sample.hardness),
        filling: smoothstep(0.0, FILLED_METRES, sample.deposit),
        lake,
        channel,
    }
}

/// The height of the texel at `(x, y)`, in metres.
///
/// `x` and `y` are raster metres -- east from the western edge, south from the
/// northern one -- and `texel_metres` is the ground a texel of this level
/// covers, which is what the detail is band-limited to.
pub fn height(
    sample: &Sample,
    ground: &Ground,
    x: f32,
    y: f32,
    texel_metres: f32,
    seed: u32,
) -> f32 {
    // Nyquist: an octave whose features are narrower than two texels cannot be
    // represented at this level, and summing it anyway is how a pyramid starts
    // to shimmer.
    let finest = 2.0 * texel_metres;

    let texture = fbm(
        x,
        y,
        seed ^ 0x7c1a_3f55,
        Fractal::new(TEXTURE_WAVELENGTH, TEXTURE_OCTAVES).band_limited(finest),
    );
    // Centred, so ribs raise and gully alike rather than lifting the whole face.
    let ribs = ridged(
        x,
        y,
        seed ^ 0x2b90_d417,
        Fractal::new(RIB_WAVELENGTH, RIB_OCTAVES).band_limited(finest),
    ) - 0.5;

    // Centred as well, so the hummocks sit either side of the floor the water
    // left rather than raising the whole of it.
    let moraine = billow(
        x,
        y,
        seed ^ 0x9d54_86e1,
        Fractal::new(MORAINE_WAVELENGTH, MORAINE_OCTAVES).band_limited(finest),
    ) - 0.5;

    // Rough where it is steep, smooth where the water dropped its load.
    let texture_metres =
        TEXTURE_METRES * (0.25 + 1.5 * ground.steepness) * (1.0 - 0.7 * ground.filling);
    let rib_metres = RIB_METRES * ground.rockiness * (1.0 - ground.filling);
    let moraine_metres = MORAINE_METRES * ground.filling * (1.0 - ground.steepness);

    let land =
        sample.height + texture * texture_metres + ribs * rib_metres + moraine * moraine_metres
            - ground.channel * CHANNEL_METRES;

    // Standing water is flat, and it is the surface the renderer draws: there
    // is no separate water pass, so a lake is terrain at the lake's level.
    lerp(land, sample.filled, ground.lake)
}

/// What an elevation texel outside the raster holds.
///
/// Nothing generated is ever unknown, but a tile at a coarse level can hang off
/// the edge of the ground the manifest claims, and the part with nothing behind
/// it has to read as a hole rather than as a repeat of the last real ridge.
pub const OUTSIDE: f32 = -32767.0;

/// The sentinel has to stay under the threshold the renderer calls a hole. It
/// is written in two places -- here and the manifest -- and a disagreement
/// would draw as ground at thirty kilometres below the sea rather than as an
/// error.
const _: () = assert!(OUTSIDE < terrain_tiles::NODATA_BELOW);

#[cfg(test)]
mod tests {
    use super::*;
    use terrain_tiles::NODATA_BELOW;

    fn sample() -> Sample {
        Sample {
            height: 1500.0,
            hardness: 0.5,
            flow: 4.0,
            deposit: 0.0,
            filled: 1500.0,
            slope: 0.5,
            aspect: [1.0, 0.0],
        }
    }

    fn at(sample: &Sample, x: f32, y: f32, texel_metres: f32) -> f32 {
        let ground = ground(sample, texel_metres, 16.0);
        height(sample, &ground, x, y, texel_metres, 31)
    }

    /// Generated ground must never wander down to where the renderer would
    /// read it as a hole -- a texel that did would draw as a bottomless pit in
    /// the middle of a mountainside.
    #[test]
    fn generated_ground_never_reads_as_a_hole() {
        for x in 0..64 {
            let got = at(&sample(), x as f32 * 37.0, 11.0, 1.0);
            assert!(
                got > NODATA_BELOW,
                "generated {got} m, which reads as a hole"
            );
        }
    }

    /// Detail is detail. If it ever moved the ground by more than the amplitude
    /// it was given, the landscape would no longer be the one the simulation
    /// shaped.
    #[test]
    fn detail_stays_within_its_own_amplitude() {
        let bound =
            TEXTURE_METRES * 1.75 + RIB_METRES * 0.5 + MORAINE_METRES * 0.5 + CHANNEL_METRES;
        for i in 0..400 {
            let (x, y) = (i as f32 * 13.7, i as f32 * -7.3);
            let got = at(&sample(), x, y, 1.0);
            assert!(
                (got - sample().height).abs() <= bound,
                "detail moved the ground {} m at ({x}, {y})",
                got - sample().height
            );
        }
    }

    /// The point of band limiting: a coarse level is a smoothing of the fine
    /// one, not a different landscape. Checked as the spread of the detail over
    /// *the same ground*, which must fall, and keep falling, as texels grow.
    ///
    /// The fall is gentler than it looks like it should be, and that is
    /// correct rather than a weak test. Each octave carries half the amplitude
    /// of the one before it, so the two coarsest octaves alone are already
    /// three quarters of the whole sum; dropping everything below them takes a
    /// quarter off, not most of it. What has to vanish entirely is the detail
    /// on a level whose texels are coarser than the first octave -- there is
    /// nothing left to sum there at all.
    #[test]
    fn coarser_levels_carry_less_detail_than_finer_ones() {
        let spread = |texel_metres: f32| {
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for i in 0..2000 {
                let got = at(&sample(), i as f32 * 3.0, 512.0, texel_metres);
                low = low.min(got);
                high = high.max(got);
            }
            high - low
        };
        let level = [spread(1.0), spread(16.0), spread(64.0), spread(256.0)];
        for pair in level.windows(2) {
            assert!(
                pair[1] < pair[0],
                "detail rose from {} m to {} m as the texels grew",
                pair[0],
                pair[1]
            );
        }
        assert!(
            level[3] < level[0] * 0.6,
            "level 0 spread {} m and level 8 still spread {} m",
            level[0],
            level[3]
        );

        // Past the first octave there is nothing left to add at all.
        let flat = spread(TEXTURE_WAVELENGTH);
        assert_eq!(flat, 0.0, "detail survived texels wider than every octave");
    }

    /// Tiles are generated independently, so the only thing that can make them
    /// join is that the height at a position does not depend on which tile
    /// asked for it. This is that claim, stated directly.
    #[test]
    fn the_height_at_a_position_does_not_depend_on_who_asks() {
        for x in [0.0f32, 511.5, 512.5, 8191.5, 49151.5] {
            let first = at(&sample(), x, 1024.5, 1.0);
            let second = at(&sample(), x, 1024.5, 1.0);
            assert_eq!(first, second, "at x = {x}");
        }
    }

    /// Water is flat and the renderer draws no water of its own, so a lake has
    /// to *be* the terrain at the lake's level -- to within nothing at all, or
    /// it would ripple.
    #[test]
    fn standing_water_is_drawn_dead_flat_at_its_own_level() {
        let flooded = Sample {
            height: 1490.0,
            filled: 1500.0,
            slope: 0.02,
            ..sample()
        };
        for i in 0..200 {
            let got = at(&flooded, i as f32 * 3.1, 700.0, 1.0);
            assert!(
                (got - flooded.filled).abs() < 1e-3,
                "the lake surface came out at {got}, not {}",
                flooded.filled
            );
        }
    }

    /// The shore is a band rather than a step, or every lake would be ringed by
    /// a one-texel cliff.
    #[test]
    fn the_shoreline_is_a_ramp_rather_than_a_step() {
        let mut last = f32::NAN;
        for step in 0..=20 {
            let depth = step as f32 * (LAKE_METRES / 10.0);
            let shore = Sample {
                height: 1500.0 - depth,
                filled: 1500.0,
                slope: 0.05,
                ..sample()
            };
            let got = at(&shore, 300.0, 300.0, 1.0);
            if last.is_finite() {
                assert!(
                    (got - last).abs() < 1.5,
                    "the shore jumped {} m at a depth of {depth} m",
                    got - last
                );
            }
            last = got;
        }
    }

    /// A river channel is a couple of coarse cells wide, so it must fade out on
    /// levels whose texels are wider than that -- otherwise it is sampled once
    /// every few hundred metres and flickers as the camera moves.
    #[test]
    fn a_channel_fades_out_on_levels_too_coarse_to_hold_it() {
        let trunk = Sample {
            flow: RIVER_FLOW + 2.0,
            slope: 0.05,
            ..sample()
        };
        let carve = |texel_metres: f32| ground(&trunk, texel_metres, 16.0).channel;
        assert!(carve(1.0) > 0.9, "no channel at one metre: {}", carve(1.0));
        assert!(carve(16.0) > 0.5, "no channel at level 4: {}", carve(16.0));
        assert_eq!(carve(256.0), 0.0, "a channel survived level 8");
    }

    /// Steep hard rock has to end up rougher than a valley floor of gravel, or
    /// the detail is decoration rather than a continuation of the erosion.
    #[test]
    fn rock_is_roughened_and_filled_ground_is_not() {
        let spread = |sample: &Sample| {
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for i in 0..600 {
                let got = at(sample, i as f32 * 1.7, 4000.0, 1.0) - sample.height;
                low = low.min(got);
                high = high.max(got);
            }
            high - low
        };
        let cliff = Sample {
            slope: 1.4,
            hardness: 1.0,
            ..sample()
        };
        let flat = Sample {
            slope: 0.02,
            hardness: 0.0,
            deposit: 20.0,
            ..sample()
        };
        assert!(
            spread(&cliff) > spread(&flat) * 2.0,
            "the cliff spread {} m and the flat {} m",
            spread(&cliff),
            spread(&flat)
        );
    }

    /// Ground the water filled gets hummocks rather than the smooth plane a
    /// bare interpolation of the coarse grid would leave. Valley floors are the
    /// ground the camera is closest to, so a flat one is the most visible thing
    /// a detail pass can get wrong.
    #[test]
    fn filled_ground_is_hummocky_rather_than_flat() {
        let floor = Sample {
            slope: 0.02,
            hardness: 0.2,
            deposit: 20.0,
            ..sample()
        };
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..800 {
            let got = at(&floor, i as f32 * 2.3, 9000.0, 1.0);
            low = low.min(got);
            high = high.max(got);
        }
        assert!(
            high - low > 1.5,
            "the valley floor spread only {} m",
            high - low
        );
    }
}
