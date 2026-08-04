//! Thermal erosion: letting slopes fall to the angle they can actually stand at.
//!
//! Rock does not hold at any angle. Past its angle of repose it breaks off and
//! slides until the pile it lands on is shallow enough to stay put, and the
//! scree fans under every mountain cliff are the result. Fractal noise knows
//! nothing about that, so a freshly raised landscape is full of slopes no
//! ground could hold and spikes a single lattice point wide.
//!
//! This pass moves material downhill wherever a slope exceeds its repose angle,
//! repeatedly, until it does not. Two things fall out of that which no amount of
//! noise gives you: cliffs and talus become *one* feature rather than two, and
//! every slope in the landscape ends up at one of a small number of angles --
//! which is exactly what a real mountainside looks like, and what makes the
//! material classifier's job possible later.
//!
//! # Why it is a gather rather than a scatter
//!
//! The obvious form is a scatter: each cell works out what it sheds and adds it
//! to its neighbours. That cannot be run in parallel without either locks or
//! non-determinism, because two cells write to the same neighbour.
//!
//! So it runs as two parallel passes over read-only state instead. The first
//! works out how much each cell gives away and how that would be split; the
//! second has each cell *collect* its share from each of its eight neighbours.
//! Both are pure functions of the previous sweep, so the result does not depend
//! on how the work was divided, and the same seed gives the same landscape on
//! any number of cores.

use rayon::prelude::*;

use crate::fields::Fields;

/// How much of the excess a sweep moves.
///
/// Half, which is the largest value that cannot overshoot: moving the whole
/// excess would put the neighbour above the cell that shed to it and the pair
/// would oscillate rather than settle.
const SWEEP_SHARE: f32 = 0.5;

/// Which stage of the landscape is settling.
///
/// The angle a slope holds at is not a property of the terrain, it is a
/// property of the material, and the two materials this crate has behave very
/// differently: bedrock stands in cliffs, loose sediment does not stand at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Settling {
    /// Freshly raised rock, before any water has touched it.
    ///
    /// Held at an angle that varies with hardness, which is what turns a
    /// uniform slope into a staircase of cliff bands and benches: the soft
    /// beds slump back and the hard ones are left standing proud of them.
    Bedrock,
    /// Sediment the droplets have just dropped.
    ///
    /// One shallow angle regardless of what it landed on, because a pile of
    /// loose gravel does not care what is under it.
    Sediment,
}

impl Settling {
    /// How many sweeps to run.
    ///
    /// Bedrock starts from raw noise with slopes far past anything that could
    /// stand, and needs enough sweeps for the material to travel the length of
    /// a talus fan -- material moves one cell a sweep, so this is a distance.
    /// Sediment has only just been dropped and is nearly at rest already.
    fn sweeps(self) -> u32 {
        match self {
            Settling::Bedrock => 48,
            Settling::Sediment => 10,
        }
    }

    /// The steepest angle this material holds at, in degrees.
    ///
    /// Real dry talus rests between about 33 and 37 degrees whatever it is made
    /// of; bare rock faces stand far past vertical in places, but a heightfield
    /// cannot represent an overhang, so the hard end is the steepest slope a
    /// single-valued surface can usefully carry.
    fn repose_degrees(self, hardness: f32) -> f32 {
        match self {
            Settling::Bedrock => 34.0 + 34.0 * hardness,
            Settling::Sediment => 34.0,
        }
    }
}

/// The eight neighbours, as offsets and the ground distance to each.
const NEIGHBOURS: [(i64, i64); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

/// How much further apart two diagonal neighbours are than two orthogonal ones.
///
/// A diagonal neighbour may sit proportionally lower without the slope between
/// them being any steeper, so every fall is measured against its own reach.
fn reach(dx: i64, dy: i64) -> f32 {
    if dx != 0 && dy != 0 {
        std::f32::consts::SQRT_2
    } else {
        1.0
    }
}

/// Runs thermal erosion to convergence, or as near as the sweep count gets.
pub fn relax(fields: &mut Fields, settling: Settling) {
    let (width, rows) = (fields.width(), fields.rows());
    let metres_per_cell = fields.metres_per_cell;

    // What each cell gives away this sweep, and the total of the excesses that
    // amount is to be split between.
    let mut given = vec![0f32; width * rows];
    let mut shared = vec![0f32; width * rows];
    // The sweep reads this and writes the channel, then the two are swapped, so
    // no cell ever reads a neighbour that has already moved.
    let mut previous = vec![0f32; width * rows];

    for _ in 0..settling.sweeps() {
        std::mem::swap(&mut previous, &mut fields.height.values);
        let heights: &[f32] = &previous;
        let hardness: &[f32] = &fields.hardness.values;

        given
            .par_chunks_mut(width)
            .zip(shared.par_chunks_mut(width))
            .enumerate()
            .for_each(|(row, (given, shared))| {
                for column in 0..width {
                    let here = heights[row * width + column];
                    let hold = settling.repose_degrees(hardness[row * width + column]);
                    let fall = hold.to_radians().tan() * metres_per_cell;

                    let (mut total, mut steepest) = (0.0f32, 0.0f32);
                    for (dx, dy) in NEIGHBOURS {
                        let excess = here
                            - at(heights, width, rows, column as i64 + dx, row as i64 + dy)
                            - fall * reach(dx, dy);
                        if excess > 0.0 {
                            total += excess;
                            steepest = steepest.max(excess);
                        }
                    }
                    given[column] = SWEEP_SHARE * steepest;
                    shared[column] = total;
                }
            });

        // ... and what each cell collects, asking every neighbour for its share
        // rather than being handed one.
        let given: &[f32] = &given;
        let shared: &[f32] = &shared;
        fields
            .height
            .values
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, out)| {
                for column in 0..width {
                    let here = heights[row * width + column];
                    let mut taken = 0.0;
                    for (dx, dy) in NEIGHBOURS {
                        let (nx, ny) = (column as i64 + dx, row as i64 + dy);
                        let shed = at(given, width, rows, nx, ny);
                        if shed <= 0.0 {
                            continue;
                        }
                        let neighbour = at(heights, width, rows, nx, ny);
                        let hold = settling.repose_degrees(at(hardness, width, rows, nx, ny));
                        let excess = neighbour
                            - here
                            - hold.to_radians().tan() * metres_per_cell * reach(dx, dy);
                        if excess <= 0.0 {
                            continue;
                        }
                        taken += shed * excess / at(shared, width, rows, nx, ny);
                    }
                    out[column] = here - given[row * width + column] + taken;
                }
            });
    }
}

/// A cell of a flat grid, with anything outside clamped to the edge.
///
/// The same rule [`crate::fields::Grid::at`] uses, repeated here because these
/// sweeps work on bare slices: the two auxiliary buffers are not channels of
/// the landscape and giving them a `Grid` each would only be ceremony.
fn at(values: &[f32], width: usize, rows: usize, column: i64, row: i64) -> f32 {
    let column = column.clamp(0, width as i64 - 1) as usize;
    let row = row.clamp(0, rows as i64 - 1) as usize;
    values[row * width + column]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::Fields;

    /// A single tall cell on a flat plain.
    ///
    /// Kept low enough that the sweep count can actually carry the material to
    /// the foot of the pile it makes -- material moves one cell a sweep, so a
    /// pile that needs to spread further than the sweeps allow has not
    /// converged and cannot be asked to have.
    fn spike(width: usize, rows: usize, metres_per_cell: f32) -> Fields {
        let mut fields = Fields::new(
            [
                (width - 1) as f32 * metres_per_cell,
                (rows - 1) as f32 * metres_per_cell,
            ],
            metres_per_cell,
        );
        assert_eq!((fields.width(), fields.rows()), (width, rows));
        let index = fields.height.index(width / 2, rows / 2);
        fields.height.values[index] = 40.0;
        fields
    }

    fn total(fields: &Fields) -> f64 {
        fields.height.values.iter().map(|v| f64::from(*v)).sum()
    }

    /// Thermal erosion moves material; it does not make or destroy any. A leak
    /// would show up as the whole landscape slowly rising or sinking over the
    /// sweeps, which is invisible in a single frame and obvious in the height
    /// range at the end of a run.
    #[test]
    fn relaxing_conserves_the_material_it_moves() {
        let mut fields = spike(33, 33, 10.0);
        let before = total(&fields);
        relax(&mut fields, Settling::Sediment);
        let after = total(&fields);
        assert!(
            (before - after).abs() < before.abs() * 1e-4 + 1e-3,
            "{before} became {after}"
        );
    }

    /// The property the pass exists for. Every slope must end at or under the
    /// angle its material holds at, or the cliffs it leaves are ones no scree
    /// could have formed under.
    #[test]
    fn no_slope_is_left_steeper_than_its_material_holds() {
        let mut fields = spike(33, 33, 10.0);
        relax(&mut fields, Settling::Sediment);

        let hold = Settling::Sediment.repose_degrees(0.5).to_radians().tan();
        for row in 0..fields.rows() as i64 {
            for column in 0..fields.width() as i64 {
                let here = fields.height.at(column, row);
                for (dx, dy) in NEIGHBOURS {
                    let reach = if dx != 0 && dy != 0 {
                        std::f32::consts::SQRT_2
                    } else {
                        1.0
                    };
                    let drop = here - fields.height.at(column + dx, row + dy);
                    let allowed = hold * fields.metres_per_cell * reach;
                    assert!(
                        drop <= allowed + 1e-2,
                        "({column}, {row}) falls {drop} m to its neighbour, past {allowed} m"
                    );
                }
            }
        }
    }

    /// Harder rock has to end up standing steeper than softer rock, because
    /// that difference is what draws the cliff bands.
    #[test]
    fn harder_rock_stands_steeper_than_soft() {
        let build = |hardness: f32| {
            let mut fields = spike(21, 21, 10.0);
            fields.hardness.values.fill(hardness);
            relax(&mut fields, Settling::Bedrock);
            let middle = fields.height.at(10, 10);
            middle - fields.height.at(11, 10)
        };
        let soft = build(0.0);
        let hard = build(1.0);
        assert!(hard > soft * 1.5, "soft fell {soft} m, hard fell {hard} m");
    }

    /// Flat ground has nothing to shed, and a pass that moved material anyway
    /// would be adding a slow drift to every basin in the landscape.
    #[test]
    fn flat_ground_is_left_exactly_alone() {
        let mut fields = Fields::new([200.0, 200.0], 10.0);
        fields.height.values.fill(1234.5);
        relax(&mut fields, Settling::Bedrock);
        assert!(
            fields.height.values.iter().all(|value| *value == 1234.5),
            "flat ground moved"
        );
    }

    /// The two-pass gather exists so that the answer cannot depend on how the
    /// rows were split between threads. Checked by running the same landscape
    /// through a pool of one thread and a pool of many.
    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("failed to build a pool");
            pool.install(|| {
                let mut fields = spike(41, 37, 8.0);
                for (index, value) in fields.hardness.values.iter_mut().enumerate() {
                    *value = (index % 7) as f32 / 6.0;
                }
                relax(&mut fields, Settling::Bedrock);
                fields.height.values
            })
        };
        assert_eq!(run(1), run(8));
    }
}
