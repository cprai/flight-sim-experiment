//! Raising the hills the erosion passes then take apart.
//!
//! Nothing here pretends to be geology. What it has to produce is a starting
//! surface with the *structure* a mountain range has -- long crests rather than
//! scattered peaks, ranges separated by broad basins, and a consistent grain --
//! because erosion sharpens what it is given and cannot invent structure that
//! is not already there. Drop droplets on plain fractional Brownian motion and
//! the result is a rumpled blanket with gullies in it, however long the
//! simulation runs.
//!
//! Three things do that work:
//!
//! * a **ridged multifractal**, whose creases are lines rather than points, for
//!   the crests;
//! * a **domain warp** in front of it, so the crests bend and branch instead of
//!   running in the straight lines a fractal on an unwarped lattice produces;
//! * an **anisotropic stretch about a trend bearing**, so ranges are long in one
//!   direction and narrow across it. The Rockies run roughly north-north-west,
//!   and that single choice is most of what makes a generated range read as that
//!   range rather than as generic mountains.
//!
//! A separate hardness channel is raised at the same time. It is not terrain: it
//! is how well the rock at a point resists being cut, and it is what later turns
//! a uniform slope into benches, cliff bands and the talus below them. Erosion
//! reads it, and so does the material classifier.

use rayon::prelude::*;

use crate::fields::{Fields, Grid};
use crate::noise::{Fractal, fbm, ridged, smoothstep, value, warp};

/// The height range a landscape is asked to span, in metres.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Relief {
    /// Where the lowest valley floor ends up.
    pub valley_metres: f32,
    /// Where the highest peak ends up.
    pub peak_metres: f32,
}

impl Relief {
    pub fn span(&self) -> f32 {
        self.peak_metres - self.valley_metres
    }
}

/// Which way the ranges run, as a bearing in degrees clockwise from north.
///
/// The Rockies trend north-north-west, and a range's grain is the most
/// recognisable thing about it from the air: crests, the valleys between them,
/// and the rivers in those valleys all line up. Twenty degrees west of north is
/// the front ranges' bearing through Alberta and Montana.
///
/// Read only by the test that pins [`TREND_SIN`] and [`TREND_COS`] to it; the
/// generator itself uses those, because a sine is not a `const fn` and this is
/// evaluated once per cell of an eleven-million-cell grid.
#[allow(
    dead_code,
    reason = "the bearing the direction constants are pinned to"
)]
const TREND_DEGREES: f32 = -20.0;

/// The trend as a direction.
///
/// Written out rather than derived, because `sin_cos` is not a `const fn` and
/// this is evaluated for every cell of an eleven-million-cell grid. Pinned
/// against [`TREND_DEGREES`] by a test, so the two cannot drift.
const TREND_SIN: f32 = -0.342_020_14;
const TREND_COS: f32 = 0.939_692_6;

/// How much longer a range is than it is wide.
///
/// Applied by compressing the across-trend axis before the noise is sampled, so
/// features that would have been round come out elongated along the trend.
const TREND_STRETCH: f32 = 2.4;

/// Feature size of the crests, in metres, before the stretch.
///
/// Seven kilometres across the trend and seventeen along it. The ridges of a
/// real range sit a few kilometres apart, and a longer wavelength than this
/// puts only three or four crests on a fifty-kilometre map -- which reads as a
/// dome with some bumps on it rather than as a range.
const CREST_WAVELENGTH: f32 = 7_000.0;

/// Feature size of the warp that bends the crests, and how far it moves a point.
///
/// Nearly two kilometres is a large warp -- comparable to the crest spacing --
/// and that is the point: a timid warp leaves the fractal's lattice showing
/// through as crests that all meet at the same few angles.
const WARP_WAVELENGTH: f32 = 9_000.0;
const WARP_METRES: f32 = 1_800.0;

/// Feature size of the mask that decides where ranges are at all.
///
/// Much larger than a crest, so a whole group of crests rises or falls
/// together and the map has ranges and basins rather than uniform corrugation.
const RANGE_WAVELENGTH: f32 = 36_000.0;

/// Feature size of the rolling ground under everything.
const ROLLING_WAVELENGTH: f32 = 5_000.0;

/// How much of the height comes from the crests, and how much from the rolling
/// ground between them.
///
/// These and the regional fall add to more than the whole range, deliberately.
/// The three never reach their own extremes at the same point -- a crest at the
/// low end of the ramp is nowhere near the top of the range -- so sharing the
/// relief between them strictly would leave the mountains looking half-sized.
const CREST_SHARE: f32 = 0.62;
const ROLLING_SHARE: f32 = 0.12;

/// How sharply the crests rise out of their own field.
///
/// Above one, so that the ridged field's middle values are pushed down and the
/// range reads as peaks above a floor rather than as a plateau with dents.
const CREST_CURVE: f32 = 1.35;

/// How far the whole map falls from one end to the other, as a fraction of the
/// relief, **along the trend**.
///
/// Both halves of that matter, and between them they are the difference between
/// a landscape and a swamp.
///
/// *That it exists at all*: a range with no regional gradient is a field of
/// closed basins. Each valley is ringed by crests, water collects in it, and
/// the flow pass floods a fifth of the map. Erosion does not rescue it, because
/// erosion has nowhere to take the water either. Real ranges are not
/// flat-based; the Rockies fall from the continental divide to the plains by
/// close to two kilometres over a hundred.
///
/// *That it runs along the trend*: the crests are stretched along the trend, so
/// the valleys between them are too -- long, narrow, and parallel. A fall
/// *across* the trend runs into every one of those crests in turn and dams each
/// valley at both ends, which is a gradient that drains nothing. A fall along
/// the trend runs down the valleys themselves, and every one of them reaches
/// the edge of the map.
///
/// *That it is a gradient rather than a share of the relief*: a share is
/// scale-dependent in the worst way. The crests keep their size in metres
/// whatever box is asked for, so spreading a fixed share of the relief over a
/// bigger map halves its gradient and the basins come back -- twenty-five
/// kilometres of ground came out a twenty-fifth underwater and fifty
/// kilometres of the same landscape came out a sixth. Twenty metres a
/// kilometre is roughly what the Rockies do between the continental divide and
/// the plains, and it means the same thing at any extent.
const TILT_METRES_PER_KILOMETRE: f32 = 20.0;

/// The most of the relief the regional fall may take.
///
/// The gradient above wins on any map of a sensible size; this only bites when
/// a very large extent is asked for with a very small relief, where an
/// unbounded ramp would leave nothing over for the mountains standing on it.
const TILT_SHARE_LIMIT: f32 = 0.55;

/// Feature sizes of the two parts of the hardness field.
///
/// The broad part is which massif is hard; the banded part is the bedding
/// within it, stretched hard along the trend so that it draws as strata rather
/// than as blotches.
const HARDNESS_WAVELENGTH: f32 = 7_000.0;
const BEDDING_WAVELENGTH: f32 = 900.0;
const BEDDING_STRETCH: f32 = 9.0;

/// Turns a `-1..=1` fractal into a `0..=1` one.
fn unipolar(value: f32) -> f32 {
    value * 0.5 + 0.5
}

/// A position rotated onto the trend and squashed across it.
///
/// The returned pair is "along the trend" and "across the trend", the second
/// scaled up so that a fractal sampling it changes faster across than along.
fn on_trend(x: f32, y: f32, stretch: f32) -> [f32; 2] {
    let along = x * TREND_SIN + y * TREND_COS;
    let across = x * TREND_COS - y * TREND_SIN;
    [along / stretch, across]
}

/// How far along the trend a point is, `0` at the north-north-west end of the
/// ground and `1` at the south-south-east end.
///
/// The extent is projected onto the trend the same way the point is, so the
/// two ends are the two corners the trend actually runs between rather than
/// the corners of the box.
fn along_the_trend(x: f32, y: f32, extent: [f32; 2]) -> f32 {
    let at = x * TREND_SIN + y * TREND_COS;
    ((at - trend_low(extent)) / trend_span(extent)).clamp(0.0, 1.0)
}

/// The along-trend coordinate of the corner of `extent` that comes first.
///
/// The trend bears west of north, so its sine is negative and the box's eastern
/// edge is the low end of the axis.
fn trend_low(extent: [f32; 2]) -> f32 {
    extent[0] * TREND_SIN.min(0.0)
}

/// How far the ground reaches along the trend, in metres.
fn trend_span(extent: [f32; 2]) -> f32 {
    extent[0] * TREND_SIN.max(0.0) + extent[1] * TREND_COS - trend_low(extent)
}

/// How much of the relief the regional fall takes over a given extent.
pub fn tilt_share(extent: [f32; 2], relief: Relief) -> f32 {
    let fall = TILT_METRES_PER_KILOMETRE * trend_span(extent) / 1000.0;
    (fall / relief.span().max(1.0)).min(TILT_SHARE_LIMIT)
}

/// The starting height at a point, as a fraction of the relief.
///
/// Kept as a fraction rather than metres so that the shape does not depend on
/// what range was asked for: `--peak-metres` scales the landscape, it does not
/// change it.
fn potential(x: f32, y: f32, extent: [f32; 2], tilt_share: f32, finest: f32, seed: u32) -> f32 {
    let [along, across] = on_trend(x, y, TREND_STRETCH);
    let [wx, wy] = warp(
        along,
        across,
        seed ^ 0x51a7_3e19,
        Fractal::new(WARP_WAVELENGTH, 3),
        WARP_METRES,
    );

    let crest = ridged(
        wx,
        wy,
        seed ^ 0x1d2c_7b41,
        Fractal::new(CREST_WAVELENGTH, 9).band_limited(finest),
    );
    let range = smoothstep(
        0.18,
        0.70,
        unipolar(fbm(
            x,
            y,
            seed ^ 0x9c41_08ad,
            Fractal::new(RANGE_WAVELENGTH, 3),
        )),
    );
    let rolling = unipolar(fbm(
        x,
        y,
        seed ^ 0x77b3_51e2,
        Fractal::new(ROLLING_WAVELENGTH, 6).band_limited(finest),
    ));

    // Falling down the length of the ranges rather than across them, which is
    // the direction the valleys already run and so the one that gives each of
    // them a way off the map.
    let tilt = tilt_share * (1.0 - along_the_trend(x, y, extent));

    let ranges = range * crest.powf(CREST_CURVE) * CREST_SHARE;
    // The rolling ground is strongest between the ranges: inside one it would
    // only be roughness on a slope that erosion is about to redo anyway.
    let between = rolling * ROLLING_SHARE * (0.40 + 0.60 * range);
    (ranges + between + tilt).clamp(0.0, 1.0)
}

/// How hard the rock is at a point, `0..=1`.
fn hardness(x: f32, y: f32, seed: u32) -> f32 {
    let massif = unipolar(fbm(
        x,
        y,
        seed ^ 0x3ea1_9c57,
        Fractal::new(HARDNESS_WAVELENGTH, 4),
    ));
    let [along, across] = on_trend(x, y, BEDDING_STRETCH);
    let bedding = unipolar(value(
        along / BEDDING_WAVELENGTH,
        across / BEDDING_WAVELENGTH,
        seed ^ 0xc4d9_2b17,
    ));
    (0.62 * massif + 0.38 * bedding).clamp(0.0, 1.0)
}

/// Fills the height and hardness channels of a fresh grid.
pub fn raise(fields: &mut Fields, relief: Relief, seed: u32) {
    let width = fields.width();
    let metres_per_cell = fields.metres_per_cell;
    let extent = [
        (fields.width() - 1) as f32 * metres_per_cell,
        (fields.rows() - 1) as f32 * metres_per_cell,
    ];
    // Nothing finer than the grid can carry: an octave below two cells would
    // alias into the simulation rather than be eroded by it.
    let finest = 2.0 * metres_per_cell;
    let tilt = tilt_share(extent, relief);
    log::info!(
        "the ground falls {:.0} m along the trend, {:.0}% of the relief",
        tilt * relief.span(),
        tilt * 100.0
    );

    let Fields {
        height,
        hardness: rock,
        ..
    } = fields;
    height
        .values
        .par_chunks_mut(width)
        .zip(rock.values.par_chunks_mut(width))
        .enumerate()
        .for_each(|(row, (heights, rocks))| {
            let y = row as f32 * metres_per_cell;
            for column in 0..width {
                let x = column as f32 * metres_per_cell;
                heights[column] = relief.valley_metres
                    + relief.span() * potential(x, y, extent, tilt, finest, seed);
                rocks[column] = hardness(x, y, seed);
            }
        });
}

/// Stretches a channel so its lowest and highest values land exactly on `range`.
///
/// Run after the erosion passes rather than before them. Erosion always takes
/// height off the peaks -- that is what it is -- so a landscape raised to the
/// asked-for peak arrives somewhere below it, by an amount that depends on how
/// long the droplets ran. Rescaling at the end makes `--peak-metres` mean what
/// it says without pretending to know that amount in advance. The shapes are
/// untouched: it is one affine map applied to every node.
pub fn rescale(grid: &mut Grid, relief: Relief) {
    let (low, high) = grid.range();
    if !(high - low).is_finite() || high - low < 1e-3 {
        return;
    }
    let scale = relief.span() / (high - low);
    for value in &mut grid.values {
        *value = relief.valley_metres + (*value - low) * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn relief() -> Relief {
        Relief {
            valley_metres: 700.0,
            peak_metres: 2600.0,
        }
    }

    fn raised(metres_per_cell: f32, extent: [f32; 2]) -> Fields {
        let mut fields = Fields::new(extent, metres_per_cell);
        raise(&mut fields, relief(), 12345);
        fields
    }

    /// A landscape that ran off either end of the range it was asked for would
    /// put the camera underground or leave it a kilometre too high, since the
    /// opening viewpoint is derived from the highest ground.
    #[test]
    fn the_raised_ground_stays_inside_the_relief_it_was_asked_for() {
        let fields = raised(64.0, [16_384.0, 16_384.0]);
        let (low, high) = fields.height.range();
        assert!(low >= relief().valley_metres - 1.0, "lowest is {low}");
        assert!(high <= relief().peak_metres + 1.0, "highest is {high}");
    }

    /// The point of the whole module: mountains, not a plain with bumps. If the
    /// crests stopped forming, this is the number that would collapse.
    ///
    /// Two thirds rather than all of it. The three terms that make up the
    /// height never reach their own extremes at the same point, so the raw
    /// uplift always leaves some of the range on the table; `rescale` is what
    /// claims it, once erosion has taken what it is going to take.
    #[test]
    fn a_raised_landscape_uses_most_of_the_range_available_to_it() {
        let fields = raised(64.0, [32_768.0, 32_768.0]);
        let (low, high) = fields.height.range();
        assert!(
            high - low > relief().span() * 0.6,
            "only {} m of {} m relief was used",
            high - low,
            relief().span()
        );
    }

    /// Ranges have to run somewhere. Measured as the ratio of how fast the
    /// ground changes across the trend to how fast it changes along it: an
    /// isotropic landscape gives one, and a grained one gives well above it.
    #[test]
    fn ranges_run_along_the_trend_rather_than_in_every_direction() {
        let fields = raised(64.0, [32_768.0, 32_768.0]);
        let angle = TREND_DEGREES.to_radians();
        let (sin, cos) = angle.sin_cos();
        let step = 512.0;

        let (mut along, mut across, mut samples) = (0.0f64, 0.0f64, 0u32);
        for row in 8..(fields.rows() - 8) {
            for column in 8..(fields.width() - 8) {
                let [x, y] = fields.metres_of_node(column, row);
                let at = fields
                    .height
                    .sample_nodes(x / fields.metres_per_cell, y / fields.metres_per_cell);
                let ahead = |dx: f32, dy: f32| {
                    fields.height.sample_nodes(
                        (x + dx) / fields.metres_per_cell,
                        (y + dy) / fields.metres_per_cell,
                    )
                };
                along += f64::from((ahead(sin * step, cos * step) - at).abs());
                across += f64::from((ahead(cos * step, -sin * step) - at).abs());
                samples += 1;
            }
        }
        assert!(samples > 1000);
        assert!(
            across / along > 1.25,
            "the ground changes {:.3} across the trend for every 1 along it, \
             which is not a grain",
            across / along
        );
    }

    /// The precomputed direction has to be the bearing it claims to be. A drift
    /// between the two would turn the ranges and leave the regional fall
    /// pointing across them instead of along them, which is the one thing that
    /// floods the map.
    #[test]
    fn the_trend_direction_matches_the_bearing_it_was_taken_from() {
        let (sin, cos) = TREND_DEGREES.to_radians().sin_cos();
        assert!((sin - TREND_SIN).abs() < 1e-6, "{sin} against {TREND_SIN}");
        assert!((cos - TREND_COS).abs() < 1e-6, "{cos} against {TREND_COS}");
    }

    /// The mountains scale with the relief; the ground they stand on does not.
    ///
    /// Asking for twice the relief has to make the crests twice as tall, or
    /// `--peak-metres` would be a different landscape rather than a taller one.
    /// The regional fall is the exception, and deliberately: it is a gradient
    /// across real ground, so it stays the same number of metres however tall
    /// the peaks standing on it are asked to be. Doubling the relief therefore
    /// halves its *share* -- which is the whole difference between a range on a
    /// continental slope and a range on a table.
    #[test]
    fn the_relief_scales_the_mountains_and_leaves_the_regional_fall_alone() {
        let extent = [32_768.0, 32_768.0];
        let doubled = Relief {
            valley_metres: relief().valley_metres,
            peak_metres: relief().valley_metres + relief().span() * 2.0,
        };
        let fall = |relief: Relief| tilt_share(extent, relief) * relief.span();
        assert!(
            (fall(relief()) - fall(doubled)).abs() < 1.0,
            "the ground falls {} m at one relief and {} m at twice it",
            fall(relief()),
            fall(doubled)
        );

        // ... and what is left over -- the crests and the rolling ground -- is
        // the part that doubles.
        let peaks = |relief: Relief| {
            let mut fields = Fields::new(extent, 64.0);
            raise(&mut fields, relief, 99);
            let (low, high) = fields.height.range();
            high - low - fall(relief)
        };
        // Not exactly twice, and it cannot be: the highest and lowest ground of
        // a map are not the highest crest and the lowest valley of the ramp,
        // they are wherever the two happened to add up furthest apart. Taking
        // the fall back off is an estimate of the mountains, not a measurement.
        let (single, double) = (peaks(relief()), peaks(doubled));
        assert!(
            (double / single - 2.0).abs() < 0.25,
            "the mountains went from {single} m to {double} m, which is not twice"
        );
    }

    /// The regional fall is a gradient rather than a share, so a bigger box
    /// gets proportionally more of it. Getting this wrong is invisible at test
    /// scale and floods a sixth of a full-sized map: the crests keep their size
    /// in metres, so a fixed share spread over twice the ground is half the
    /// gradient holding them apart.
    #[test]
    fn a_wider_map_falls_further_at_the_same_gradient() {
        let small = [16_384.0, 16_384.0];
        let large = [32_768.0, 32_768.0];
        let fall = |extent: [f32; 2]| tilt_share(extent, relief()) * relief().span();
        let (near, far) = (fall(small), fall(large));
        assert!(
            far > near * 1.8,
            "{near} m over the small box, {far} m over a box twice the size"
        );
        // ... up to the cap, which is what stops a very large box from being
        // nothing but a ramp.
        assert!(tilt_share([500_000.0, 500_000.0], relief()) <= TILT_SHARE_LIMIT);
    }

    #[test]
    fn hardness_stays_between_zero_and_one_and_varies() {
        let fields = raised(64.0, [16_384.0, 16_384.0]);
        let (low, high) = fields.hardness.range();
        assert!(low >= 0.0 && high <= 1.0, "hardness spans {low}..{high}");
        assert!(high - low > 0.4, "hardness barely varies: {low}..{high}");
    }

    /// Rescaling is what makes the asked-for peak the peak that arrives, and it
    /// has to be an affine map -- anything else would re-shape the landscape
    /// after erosion had finished with it.
    #[test]
    fn rescaling_lands_exactly_on_the_range_asked_for() {
        let mut grid = Grid::filled(8, 8, 0.0);
        for (index, value) in grid.values.iter_mut().enumerate() {
            *value = 1100.0 + index as f32 * 3.0;
        }
        let before = grid.clone();
        rescale(&mut grid, relief());
        let (low, high) = grid.range();
        assert!((low - relief().valley_metres).abs() < 1e-2, "low is {low}");
        assert!((high - relief().peak_metres).abs() < 1e-2, "high is {high}");

        // Affine: equal steps before stay equal steps after.
        let steps: Vec<f32> = grid.values.windows(2).map(|w| w[1] - w[0]).collect();
        let first = steps[0];
        for (index, step) in steps.iter().enumerate() {
            assert!(
                (step - first).abs() < 1e-2,
                "step {index} is {step}, not {first}"
            );
        }
        assert_eq!(before.values.len(), grid.values.len());
    }

    #[test]
    fn rescaling_flat_ground_leaves_it_alone_rather_than_dividing_by_zero() {
        let mut grid = Grid::filled(4, 4, 1234.0);
        rescale(&mut grid, relief());
        assert!(grid.values.iter().all(|value| *value == 1234.0));
    }
}
