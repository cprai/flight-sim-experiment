//! Hillslope creep: the process that decides how far apart valleys are.
//!
//! Everything else in this crate takes ground away, and takes it away fastest
//! where the most water passes. On its own that is unstable, and unstable in a
//! specific way that ruins a landscape: a hollow one cell deep gathers slightly
//! more water than the ground beside it, so it is cut slightly deeper, so it
//! gathers more still. Nothing in a stream-power model opposes that, so the
//! feedback runs all the way down to the finest thing the grid can hold and the
//! landscape comes out corrugated with grooves one cell wide -- a corduroy laid
//! over every hillside, plainly visible from the air, at a spacing that says
//! nothing about the terrain and everything about `--sim-metres`.
//!
//! What opposes it in reality is that soil moves downhill without a river:
//! frost heave, tree throw, burrowing animals, rain splash, gravity on anything
//! loose. Averaged over a hillside all of that behaves like diffusion,
//!
//! ```text
//! dh/dt = D grad^2 h
//! ```
//!
//! and diffusion attacks short wavelengths hardest -- the rate goes as the
//! square of the frequency, so a two-cell groove is smoothed a hundred times
//! faster than a twenty-cell valley. Run against incision, it wins on the fine
//! stuff and loses on the coarse, and the wavelength where the two are equal is
//! the ridge-to-valley spacing of the resulting landscape. That length scale is
//! a real and measured property of real hillslopes (Perron, Kirchner and
//! Dietrich, 2009, "Formation of evenly spaced ridges and valleys", *Nature*
//! 460), and a landscape evolution model without a diffusion term does not have
//! one at all.
//!
//! So this is not a smoothing filter bolted on to hide an artifact. It is the
//! second half of the model, and leaving it out is what made the artifact.

use rayon::prelude::*;

use crate::fields::Grid;

/// The eight neighbours, and how much each counts towards the Laplacian.
///
/// Diagonals at half weight, because they are twice as far away as the square
/// of the distance and that is what a second derivative divides by. The pair
/// together is the nine-point stencil, which is very much less anisotropic than
/// the five-point one: with cardinal neighbours alone, diffusion runs faster
/// along the axes than across them and slowly rotates every feature onto the
/// grid, which is the artifact this module exists to remove rather than
/// another way of causing it.
const NEIGHBOURS: [(i64, i64, f32); 8] = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, 0.5),
    (1, -1, 0.5),
    (-1, 1, 0.5),
    (1, 1, 0.5),
];

/// What the weighted sum of the neighbour differences has to be divided by to
/// be the Laplacian in units of cells squared.
///
/// Fixed by requiring the stencil to be exact on `x^2`, whose Laplacian is two:
/// the cardinal differences sum to two and the diagonal ones to four, so
/// `(2 + 0.5 * 4) / 2 = 2`.
const STENCIL_SCALE: f32 = 2.0;

/// The most of the Laplacian one step may apply.
///
/// Explicit diffusion is stable only while a step cannot overshoot the average
/// it is moving towards. The worst case for this stencil is a stripe one cell
/// wide, whose Laplacian is `-4` times its own amplitude, so anything past
/// `0.5` inverts it and oscillates. This is well under that, because the
/// interesting question is not stability but how hard creep competes with the
/// river cutting it runs beside -- see [`STRENGTH`].
const STABILITY_LIMIT: f32 = 0.5;

/// How much creep one round applies, as a fraction of the Laplacian.
///
/// This single number sets how far apart the valleys end up, and it is worth
/// being explicit about how it was chosen.
///
/// A round of incision takes two to five per cent of a headwater cell's height
/// above its receiver, so creep matches it at the wavelength where `STRENGTH *
/// 2 * (1 - cos(2 pi / wavelength))` reaches that -- about seven cells here, or
/// a hundred metres at the default `--sim-metres`. Below it the smoothing wins
/// and grooves cannot survive; well above it creep is negligible and the rivers
/// have the landscape to themselves. A hundred metres is a plausible
/// ridge-to-valley spacing for a range of this relief, and it is at last a
/// number that comes from the model rather than from the grid: halving
/// `--sim-metres` halves the spacing in cells and leaves it where it was in
/// metres.
///
/// Picked by measuring, over a 16 km box, how much power the whole simulation
/// leaves at the two-to-three-cell scale against how much the raw uplift had
/// there. Without creep it was four hundred times as much, which is the
/// corduroy. At `0.04` it is twice; at this value it is level, so creep is
/// neither adding structure at the grid scale nor eating any; at `0.12` it is a
/// third, which is over-smoothed -- the fine drainage texture starts to go with
/// the artifact. The middle of those is what is wanted, and it is a broad
/// optimum rather than a knife edge.
const STRENGTH: f32 = 0.06;

const _: () = assert!(STRENGTH < STABILITY_LIMIT);

/// Creeps the surface one step downhill of its own curvature.
///
/// A gather, like [`crate::thermal`], and for the same reason: every cell reads
/// the previous state and writes only itself, so the result does not depend on
/// how rayon split the rows and a seed reproduces a landscape exactly.
///
/// `previous` is the caller's scratch buffer rather than a fresh allocation,
/// because this runs once per round of incision over eleven million cells.
pub fn settle(grid: &mut Grid, previous: &mut Vec<f32>) {
    let (width, rows) = (grid.width, grid.height);
    std::mem::swap(previous, &mut grid.values);
    let heights: &[f32] = previous;

    grid.values
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, out)| {
            for (column, height) in out.iter_mut().enumerate() {
                let here = heights[row * width + column];
                let inside = |dx: i64, dy: i64| {
                    let (nx, ny) = (column as i64 + dx, row as i64 + dy);
                    (nx >= 0 && ny >= 0 && nx < width as i64 && ny < rows as i64)
                        .then(|| heights[ny as usize * width + nx as usize])
                };
                let mut curvature = 0.0f32;
                for (dx, dy, weight) in NEIGHBOURS {
                    // A missing neighbour is replaced by continuing the
                    // landscape straight on, using the cell opposite: whatever
                    // slope runs off the edge keeps running.
                    //
                    // The alternatives are both wrong and wrong visibly.
                    // Clamping to the edge value mirrors the ground, so a slope
                    // meeting the boundary reads as a valley floor and the
                    // whole rim of the raster slowly rises into a lip. Dropping
                    // the term instead leaves a one-sided stencil, which is the
                    // same lip by another route -- a no-flux wall really does
                    // pile material against itself. Extrapolating gives a plane
                    // exactly zero curvature, boundary included, which is the
                    // claim `a_uniform_slope_is_left_exactly_where_it_is`
                    // makes and the only one of the three that survives it.
                    let neighbour = match inside(dx, dy) {
                        Some(height) => height,
                        None => match inside(-dx, -dy) {
                            Some(opposite) => 2.0 * here - opposite,
                            // Both sides off the grid, so there is no direction
                            // to continue in. Only reachable on a grid one or
                            // two cells across.
                            None => here,
                        },
                    };
                    curvature += weight * (neighbour - here);
                }
                *height = here + STRENGTH * curvature / STENCIL_SCALE;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(width: usize, rows: usize, at: impl Fn(usize, usize) -> f32) -> Grid {
        let mut grid = Grid::filled(width, rows, 0.0);
        for row in 0..rows {
            for column in 0..width {
                grid.values[row * width + column] = at(column, row);
            }
        }
        grid
    }

    fn stepped(mut grid: Grid, steps: u32) -> Grid {
        let mut scratch = vec![0.0; grid.values.len()];
        for _ in 0..steps {
            settle(&mut grid, &mut scratch);
        }
        grid
    }

    /// The whole claim of the module, stated as the ratio it turns on: a groove
    /// one cell wide has to die far faster than a valley twenty cells wide, or
    /// creep is a blur rather than a length scale.
    #[test]
    fn short_wavelengths_are_smoothed_far_faster_than_long_ones() {
        let ripple = |wavelength: f32| {
            let before = grid(96, 96, |column, _| {
                (column as f32 * std::f32::consts::TAU / wavelength).sin()
            });
            let after = stepped(before, 40);
            // Away from the clamped edge, where the mirror halves the ripple.
            let middle = 48 * 96;
            (32..64)
                .map(|column| after.values[middle + column].abs())
                .fold(0.0f32, f32::max)
        };
        let (fine, coarse) = (ripple(2.0), ripple(24.0));
        assert!(fine < 0.01, "a two-cell ripple survived at {fine}");
        assert!(coarse > 0.5, "a 24-cell ripple was flattened to {coarse}");
    }

    /// Creep moves ground about; it must not invent or destroy any. A drift
    /// would show as the whole landscape rising or sinking over eighty rounds,
    /// which the rescale at the end would then hide by squashing the relief.
    ///
    /// Stated over ground that is flat where it meets the edge, because that is
    /// the whole of what conservation can mean here: the boundary continues the
    /// landscape off the map rather than walling it in, so material genuinely
    /// does leave across a sloping edge, and should.
    #[test]
    fn creep_conserves_the_ground_it_moves() {
        let before = grid(64, 64, |column, row| {
            let (dx, dy) = (column as f32 - 31.5, row as f32 - 31.5);
            let distance = dx.hypot(dy);
            1000.0 + 200.0 * (1.0 - (distance / 20.0).min(1.0)).powi(2)
        });
        let total = |grid: &Grid| grid.values.iter().map(|v| f64::from(*v)).sum::<f64>();
        let was = total(&before);
        let now = total(&stepped(before, 40));
        assert!(
            (now - was).abs() / was < 1e-6,
            "the ground went from {was} to {now}"
        );
    }

    /// A plane has no curvature, so creep has nothing to do to it. This is what
    /// stops the pass from quietly eating the regional gradient the whole
    /// drainage depends on -- and, because the edges are clamped rather than
    /// skipped, it is a claim about the boundary as much as the interior.
    #[test]
    fn a_uniform_slope_is_left_exactly_where_it_is() {
        let before = grid(48, 48, |column, row| {
            900.0 + column as f32 * 2.5 - row as f32 * 1.25
        });
        let after = stepped(before.clone(), 30);
        for (index, (was, now)) in before.values.iter().zip(&after.values).enumerate() {
            assert!(
                (was - now).abs() < 1e-2,
                "cell {index} moved from {was} to {now}"
            );
        }
    }

    /// Diffusion runs equally fast in every direction or it is a grid artifact
    /// of its own. Checked as a spike, whose spread must be as wide across the
    /// diagonal as it is along the axes.
    #[test]
    fn creep_spreads_a_spike_as_far_diagonally_as_along_the_axes() {
        let side = 41;
        let middle = side / 2;
        let after = stepped(
            grid(side, side, |column, row| {
                f32::from(column == middle && row == middle) * 100.0
            }),
            30,
        );
        let at = |dx: usize, dy: usize| after.values[(middle + dy) * side + middle + dx];
        let (axis, diagonal) = (at(4, 0), at(3, 3));
        // Four cells out along the axis against 4.24 out along the diagonal, so
        // the diagonal should read a little lower and nowhere near half.
        assert!(
            diagonal > axis * 0.8 && diagonal < axis,
            "the axis holds {axis} and the diagonal {diagonal}"
        );
    }

    #[test]
    fn creep_is_reproducible() {
        let make = || {
            grid(64, 64, |column, row| {
                (column as f32 * 0.7).sin() * (row as f32 * 0.3).cos() * 50.0
            })
        };
        assert_eq!(stepped(make(), 9).values, stepped(make(), 9).values);
    }
}
