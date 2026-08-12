//! Fluvial incision: the pass that gives the landscape a way out.
//!
//! Droplets cut beautifully at the scale of a gully and hopelessly at the scale
//! of a range. Each one carries a fixed budget of water a fixed number of
//! steps, so it can deepen a channel that already exists but it cannot cut a
//! valley through a thousand-metre divide -- and a landscape raised from ridged
//! noise is full of divides. Left with only droplets, a fifty-kilometre map
//! comes out with a fifth of its ground under standing water: every basin
//! between two crests fills to its rim, because nothing ever cut the rim.
//!
//! What actually cuts rims, over geological time, is the river that spills over
//! them. The whole drainage above a spill point crosses it, and erosion goes
//! with discharge, so the outlet of a lake is the single fastest-eroding place
//! in a landscape. That is what this models, and one pass of it drains basins
//! that any number of droplets would not touch.
//!
//! # The model
//!
//! Stream power, the standard statement of it:
//!
//! ```text
//! dh/dt = -K A^m S^n
//! ```
//!
//! -- the rate a river lowers its bed goes as a power of the area it drains and
//! a power of its own slope. With `n = 1` it has a closed, unconditionally
//! stable implicit form (Braun and Willett, 2013): sweeping the drainage tree
//! from the outlets upwards,
//!
//! ```text
//! h'[i] = (h[i] + C h'[r]) / (1 + C),   C = K A^m dt / L
//! ```
//!
//! where `r` is the cell `i` drains into and `L` is the distance to it. There is
//! no timestep to get wrong: every value of `C` gives a height between `h[i]`
//! and `h'[r]`, so the profile can never invert and no round can overshoot. The
//! sweep is exactly `Drainage::order`, which the flood already produced in the
//! right direction.
//!
//! # Basins survive, rims do not
//!
//! Incision runs on the *filled* surface, which has no hollows in it at all, so
//! on its own it would cut every basin permanently flat and leave a landscape
//! with no lakes anywhere. The result is therefore taken as a ceiling rather
//! than as the answer: ground keeps whichever is lower of what it was and what
//! the river would have cut it to. A lake bed, which sits below the filled
//! surface, is untouched; the rim it spills over, which is *on* the filled
//! surface and carries the whole basin's drainage, is cut. Round by round the
//! rim comes down, the lake shrinks, and the ones that survive are the ones
//! deep enough to deserve to.
//!
//! # Cutting alone is not a landscape
//!
//! Every round of cutting is followed by a round of [`crate::creep`], and the
//! pair is the model rather than the cutting on its own. Stream power has a
//! runaway in it: a hollow one cell deep gathers a little more water than the
//! ground beside it, so it is cut a little deeper, so it gathers more still.
//! Nothing here opposes that, so the feedback runs all the way down to whatever
//! the grid can hold and the landscape arrives corrugated with grooves one cell
//! wide. Creep is what opposes it, and the wavelength where the two balance is
//! the spacing the valleys come out at. See that module for the measurements.

use crate::creep;
use crate::fields::Fields;
use crate::flow;

/// How fast rivers cut, per round.
///
/// Dimensionless, because the area term is a length once its exponent is
/// applied and it is divided by a length -- which also means a cell draining
/// only itself has a power of exactly this, whatever `--sim-metres` the grid is
/// at, and a cell draining `n` cells has `sqrt(n)` times it.
///
/// That square root is the whole of the tuning. It has to be small enough that
/// a ridge top, which drains nothing, keeps its height over every round -- at
/// this value a ridge gives up under a tenth of a percent of its relief per
/// round -- and the separation then does the rest: a trunk draining a hundred
/// thousand cells has three hundred times the power and grades itself flat in a
/// handful of rounds. Raise it much and the rivers cut faster but the ridges
/// come down with them, and the range planes off into hills.
const ERODIBILITY: f32 = 0.05;

/// How strongly the drainage area drives the cutting.
///
/// One half is the usual value fitted to real river profiles, and with `n = 1`
/// it gives the concave long profile every river has: steep at the head, nearly
/// flat at the mouth.
///
/// The sweep computes it as a `sqrt` rather than a `powf`, which is one
/// correctly-rounded instruction instead of a call into libm, ~10^9 times a
/// run. The assertion below is what keeps the two spellings honest: change this
/// constant and the build stops, rather than the code quietly going on taking a
/// square root of something that is no longer meant to be squared.
const AREA_EXPONENT: f32 = 0.5;
const _: () = assert!(AREA_EXPONENT == 0.5);

/// How much of the cutting hard rock refuses.
///
/// The same idea as in `hydraulic`, and deliberately weaker: a big river cuts
/// through hard rock given time, which is exactly how a gorge forms. Too strong
/// a resistance here and rivers stop at every hard bed and pond behind it,
/// which is the problem this pass exists to solve.
const HARDNESS_RESISTANCE: f32 = 0.5;

/// Runs the landscape forward `rounds` times: rivers cutting down, creep
/// filling in.
///
/// Each round re-routes the water first, because the last round moved it: a
/// basin that drained one way before its rim came down may drain another way
/// after.
pub fn rivers(fields: &mut Fields, rounds: u32) {
    let mut scratch = vec![0.0f32; fields.height.values.len()];
    // Everything the round needs, allocated once for the whole run. A round
    // touches about 280 MB of grids, and allocating them fresh each time buys
    // nothing but the page faults of filling a quarter of a gigabyte ninety-two
    // times over.
    let mut flow_scratch = flow::Scratch::default();
    let mut bed = Vec::new();
    for _ in 0..rounds {
        cut(fields, &mut flow_scratch, &mut bed);
        // The other half of the model, and it belongs inside this loop rather
        // than after it. Creep is not a tidy-up of what the rivers left: it is
        // what the rivers are competing against, round by round, and the
        // balance between the two is what decides how far apart the valleys end
        // up. Run once at the end instead, it would smooth a landscape whose
        // channel spacing had already collapsed to the cell size.
        let at = std::time::Instant::now();
        creep::settle(&mut fields.height, &mut scratch);
        log::debug!("incision: the creep took {:.2?}", at.elapsed());
    }
}

/// One round of stream-power incision, which only ever takes ground away.
fn cut(fields: &mut Fields, scratch: &mut flow::Scratch, bed: &mut Vec<f32>) {
    let width = fields.width();
    let cell_area = fields.metres_per_cell * fields.metres_per_cell;

    let drainage = scratch.drainage(fields);

    // Timed apart from the drainage, and at the same `debug` level, because
    // this sweep and the accumulation inside `flow` have the same shape -- a
    // serial walk of the flood's order scattering across a 44 MB grid -- and
    // knowing which of them is worth attacking means seeing them side by side
    // rather than as one figure for the round.
    let at = std::time::Instant::now();

    // Both reaches, worked out once rather than per cell: there are only two,
    // and which one a cell wants is a branch rather than a multiply.
    let cardinal_reach = fields.metres_per_cell;
    let diagonal_reach = std::f32::consts::SQRT_2 * fields.metres_per_cell;

    // The surface the rivers would leave if there were no hollows, built
    // downstream first so that every cell's receiver is already final. Filled
    // from the caller's buffer rather than cloned, so this is a copy into pages
    // that are already there rather than a fresh 44 MB every round.
    bed.resize(drainage.filled.len(), 0.0);
    bed.copy_from_slice(&drainage.filled);
    for (position, index) in drainage.order.iter().enumerate() {
        // The same lead the accumulation takes, in the other direction. This
        // sweep has the milder version of the same problem: one scattered read
        // of each of four grids rather than an eight-neighbour gather, so there
        // is less to overlap and less to win.
        if let Some(soon) = position.checked_add(flow::AHEAD)
            && let Some(soon) = drainage.order.get(soon)
        {
            let soon = *soon as usize;
            flow::prefetch(bed, soon);
            flow::prefetch(&drainage.area, soon);
            flow::prefetch(&fields.hardness.values, soon);
            flow::prefetch(&drainage.drains_to, soon);
        }

        let index = *index as usize;
        let into = drainage.drains_to[index];
        if into == u32::MAX {
            // Leaves the map. This is the base level everything else is
            // measured down to, and it does not move.
            continue;
        }
        let into = into as usize;
        // A receiver is always one of the eight neighbours, so the step between
        // the two indices carries the same information the two row numbers did
        // and costs two integer divisions instead of four. The row difference
        // need not be worked out at all: `dx` is in -1..=1, so the receiver is
        // on another row exactly when the step is not the column shift alone.
        //
        // Tempting and wrong: reading the whole thing off `|step|`, as neither
        // 1 nor `width`. That is only true while `width > 2` -- on a two-column
        // grid a diagonal step is also 1 -- and it would fail on a narrow test
        // grid rather than on the shipped one.
        let step = into as i64 - index as i64;
        let shift = (into % width) as i64 - (index % width) as i64;
        let diagonal = shift != 0 && step != shift;
        let reach = if diagonal {
            diagonal_reach
        } else {
            cardinal_reach
        };

        let resistance = 1.0 - HARDNESS_RESISTANCE * fields.hardness.values[index];
        let power = ERODIBILITY * resistance * (drainage.area[index] * cell_area).sqrt() / reach;
        // `bed[index]` still holds `drainage.filled[index]` -- it was cloned
        // from it and no cell appears twice in the order, so nothing has
        // written this one yet. Reading it here instead keeps a whole 44 MB
        // random-access stream out of the loop.
        bed[index] = (bed[index] + power * bed[into]) / (1.0 + power);
    }

    log::debug!("incision: the stream-power sweep took {:.2?}", at.elapsed());
    let at = std::time::Instant::now();

    // A ceiling, not an answer: hollows keep their own floor and only the
    // ground that stands proud of the water is cut.
    //
    // The deposit channel is deliberately left alone. It means "loose material
    // the water dropped here", which is what the classifier reads to find
    // talus, gravel bars and alluvial floors, and a river cutting its bed into
    // bedrock leaves no loose material anywhere. Recording the incision there
    // as well swamped the droplets' signal completely and stopped any scree at
    // all from being painted.
    for (height, ceiling) in fields.height.values.iter_mut().zip(bed.iter()) {
        *height = height.min(*ceiling);
    }

    log::debug!("incision: the ceiling took {:.2?}", at.elapsed());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::Fields;

    /// A hollow behind a rim, inside a plain falling east.
    ///
    /// The case droplets cannot solve: a lot of ground with no way out of it,
    /// so it ponds to the rim until something cuts the rim down. `rim` is how
    /// far the ring stands above the plain and `hollow` how far the floor
    /// inside it sits below.
    fn basin(side: usize, rim: f32, hollow: f32) -> Fields {
        let mut fields = Fields::new([(side - 1) as f32 * 20.0, (side - 1) as f32 * 20.0], 20.0);
        let middle = (side - 1) as f32 * 0.5;
        for row in 0..side {
            for column in 0..side {
                let index = fields.height.index(column, row);
                let (dx, dy) = (column as f32 - middle, row as f32 - middle);
                let distance = (dx * dx + dy * dy).sqrt();
                // A gentle regional fall, so that the ring is what makes the
                // hollow a hollow rather than the tilt of the ground under it.
                let plain = 1500.0 - column as f32 * 0.5;
                let inside = distance / (middle * 0.55);
                fields.height.values[index] = if inside < 1.0 {
                    plain + rim * (1.0 - (inside - 0.75).abs() / 0.75)
                        - hollow * f32::from(inside < 0.6)
                } else {
                    plain
                };
            }
        }
        fields
    }

    fn ponded(fields: &Fields) -> f64 {
        let drainage = flow::drainage(fields);
        let wet = drainage
            .filled
            .iter()
            .zip(&fields.height.values)
            .filter(|(filled, ground)| *filled - *ground > 2.5)
            .count();
        wet as f64 / fields.height.values.len() as f64
    }

    /// The whole reason the pass exists. A shallow hollow is a hollow because
    /// nothing has cut its rim yet, and the river spilling over that rim is the
    /// fastest-eroding thing in the landscape; given the chance, it drains.
    #[test]
    fn a_shallow_basin_drains_once_its_rim_is_cut() {
        let mut fields = basin(161, 40.0, 25.0);
        let before = ponded(&fields);
        assert!(before > 0.03, "the basin did not pond at all: {before}");
        rivers(&mut fields, 60);
        let after = ponded(&fields);
        assert!(
            after < before * 0.2,
            "ponding went from {before} to {after}, which is not draining"
        );
    }

    /// ... and the other half of the same claim, which is what stops the pass
    /// from simply abolishing water. A hollow hundreds of metres deep behind a
    /// high rim is a lake, and no amount of cutting at the outlet should empty
    /// it: that is what a lake *is*.
    #[test]
    fn a_deep_basin_is_left_as_a_lake() {
        let mut fields = basin(161, 200.0, 260.0);
        let before = ponded(&fields);
        rivers(&mut fields, 60);
        let after = ponded(&fields);
        assert!(
            after > before * 0.5,
            "a 260 m deep basin was drained away, from {before} to {after}"
        );
    }

    /// The implicit form's guarantee, and the reason there is no timestep to
    /// tune: a cell can never be cut below the cell it drains into, however
    /// hard the rivers are told to cut.
    #[test]
    fn incision_never_cuts_a_cell_below_the_one_it_drains_into() {
        let mut fields = basin(101, 200.0, 260.0);
        rivers(&mut fields, 40);
        let drainage = flow::drainage(&fields);
        for (index, into) in drainage.drains_to.iter().enumerate() {
            if *into == u32::MAX {
                continue;
            }
            assert!(
                drainage.filled[index] >= drainage.filled[*into as usize] - 1e-3,
                "cell {index} sits below the cell it drains into"
            );
        }
    }

    /// Incision only ever takes ground away. A round that raised anything would
    /// mean the implicit sweep had gone the wrong way round the tree.
    ///
    /// Asked of the cutting on its own, because [`rivers`] is the cutting *and*
    /// the creep, and creep raises ground -- filling a groove is the whole
    /// point of it. Keeping this claim needs the two steps separable, which is
    /// why they are.
    #[test]
    fn incision_only_lowers_the_ground() {
        let before = basin(101, 200.0, 260.0);
        let mut after = basin(101, 200.0, 260.0);
        let mut scratch = flow::Scratch::default();
        let mut bed = Vec::new();
        for _ in 0..40 {
            cut(&mut after, &mut scratch, &mut bed);
        }
        for (index, (was, now)) in before
            .height
            .values
            .iter()
            .zip(&after.height.values)
            .enumerate()
        {
            assert!(now <= was, "cell {index} rose from {was} m to {now} m");
        }
    }

    /// A plane falling south with a shallow groove down the middle of it.
    ///
    /// The groove collects the whole map's drainage while the ground either
    /// side of it, at the same height, collects only itself -- which is the
    /// cleanest way to ask whether the cutting follows the water rather than
    /// the height.
    fn grooved(side: usize) -> Fields {
        let mut fields = Fields::new([(side - 1) as f32 * 20.0, (side - 1) as f32 * 20.0], 20.0);
        let middle = (side - 1) as f32 * 0.5;
        for row in 0..side {
            for column in 0..side {
                let index = fields.height.index(column, row);
                let across = ((column as f32 - middle).abs() / 8.0).min(1.0);
                fields.height.values[index] = 1500.0 - row as f32 * 3.0 + 40.0 * across;
            }
        }
        fields
    }

    /// Erosion has to follow the water. If the cutting were spread evenly over
    /// the map it would be a blur rather than a river network, and the ridges
    /// would come down with the valleys.
    #[test]
    fn the_cutting_goes_where_the_drainage_is() {
        let side = 121;
        let before = grooved(side);
        let mut after = grooved(side);
        rivers(&mut after, 8);

        let cut = |column: usize, row: usize| {
            let index = before.height.index(column, row);
            before.height.values[index] - after.height.values[index]
        };
        // Halfway down the map, so that both have ground above and below them.
        let row = side / 2;
        let channel = cut(side / 2, row);
        let flank = cut(side - 8, row);
        // Threefold rather than the hundredfold the power terms differ by,
        // because the flank is not standing still either: its own base level is
        // falling as the channel cuts, and that knickpoint works its way up
        // every tributary. That is the landscape lowering, which is what should
        // happen -- what must not happen is the flank keeping pace with the
        // channel, which is a pass that has stopped being a river network.
        assert!(
            channel > flank * 3.0,
            "the channel was cut {channel} m and the ground beside it {flank} m"
        );
    }

    /// The regression test for the corduroy, run over a real raised landscape
    /// because that is the only place the bug lives.
    ///
    /// Cutting valleys is not the bug -- it is the pass working. The bug is the
    /// *scale* it cuts them at: with stream power and nothing opposing it,
    /// every other column becomes a channel, because the finest hollow the grid
    /// can hold is the finest hollow the feedback will find. Left in, it laid a
    /// corduroy over the whole map that was plainly visible from the air.
    ///
    /// So the claim is a comparison rather than a threshold, and it is the same
    /// one that diagnosed the bug: **the cutting must not leave more roughness
    /// at the scale of a cell than it was handed.** Uplift arrives with grain
    /// of its own -- it is a fractal down to two cells -- and rivers are
    /// entitled to rearrange it, but a pass whose features are all a kilometre
    /// across has no business multiplying what is there at thirty-two metres.
    /// Handed 1.59 m of grain, the pass without creep left 6.82 m of it.
    ///
    /// Measured on the ground no river runs over, which is where the artifact
    /// was and where it is a contradiction in terms: a cell two dozen others
    /// drain through has no river on it. The floor of a real valley is a cell
    /// or two wide and full of legitimate grain, and including it would drown
    /// the signal.
    #[test]
    fn cutting_leaves_no_more_grain_at_the_cell_scale_than_it_was_given() {
        let (side, cell) = (256, 32.0f32);
        let extent = [(side - 1) as f32 * cell, (side - 1) as f32 * cell];
        let mut fields = Fields::new(extent, cell);
        crate::shape::raise(
            &mut fields,
            crate::shape::Relief {
                valley_metres: 700.0,
                peak_metres: 2600.0,
            },
            7,
        );
        let before = fields.height.clone();
        rivers(&mut fields, 60);

        let hillslope = &flow::drainage(&fields).area;
        let grain = |grid: &crate::fields::Grid| {
            let at = |column: usize, row: usize| grid.values[row * side + column];
            let (mut total, mut cells) = (0.0f64, 0u32);
            for row in 8..side - 8 {
                for column in 8..side - 8 {
                    if hillslope[row * side + column] > 25.0 {
                        continue;
                    }
                    let mut around = 0.0f32;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            around += at(column + dx - 1, row + dy - 1);
                        }
                    }
                    total += f64::from((at(column, row) - around / 9.0).abs());
                    cells += 1;
                }
            }
            assert!(cells > 1000, "only {cells} cells of hillslope to measure");
            total / f64::from(cells)
        };

        // The rivers must have done something, or a pass that returned early
        // would pass this trivially.
        let cut = (before.values.iter().zip(&fields.height.values))
            .map(|(before, after)| f64::from(before - after))
            .fold(0.0, f64::max);
        assert!(cut > 50.0, "the rivers only cut {cut} m into the landscape");

        let (was, now) = (grain(&before), grain(&fields.height));
        assert!(
            now < was,
            "the landscape arrived with {was:.3} m of grain between neighbouring \
             cells and left with {now:.3} m, which is a corduroy"
        );
    }

    /// Nothing here may depend on anything but the landscape and the constants.
    #[test]
    fn incision_is_reproducible() {
        let mut first = basin(41, 200.0, 260.0);
        let mut second = basin(41, 200.0, 260.0);
        rivers(&mut first, 5);
        rivers(&mut second, 5);
        assert_eq!(first.height.values, second.height.values);
    }
}
