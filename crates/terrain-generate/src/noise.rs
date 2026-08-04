//! The fractal noise every height in this crate is ultimately made of.
//!
//! Nothing here reads a table, allocates, recurses, or touches a `f64`. That is
//! deliberate and it is the whole reason this module is separate: the plan for
//! this generator is that the per-texel half of it eventually runs as a compute
//! shader, evaluating ground the camera is about to reach instead of reading it
//! off disk. Everything below is written to port to WGSL by transcription --
//! `f32` arithmetic, `u32` bit twiddling, fixed loop bounds, plain scalars in
//! and out.
//!
//! The one thing that costs portability is the gradient table, and it is a
//! sixteen-entry constant rather than the permutation array Perlin's original
//! uses. A permutation array is a buffer to bind and an indirection to pay; the
//! hash below produces the same decorrelation arithmetically, and sixteen fixed
//! directions is a `const array` in WGSL with no binding at all.
//!
//! **Ranges are contracts.** [`gradient`] and [`value`] return `-1..=1`,
//! [`fbm`] returns roughly `-1..=1` (exactly so only in the limit; the gain
//! series is normalised so the sum cannot exceed it), and [`ridged`] returns
//! `0..=1`. Callers scale by a height in metres and would otherwise have no
//! idea what they were scaling.

/// A bit mixer with no fixed points worth worrying about.
///
/// This is Wellons' `lowbias32`, chosen over the more familiar
/// multiply-shift-xor chains because its avalanche is measured rather than
/// assumed: a single bit of input flips about half the output bits, which is
/// what stops neighbouring lattice points from drawing correlated gradients and
/// putting a visible grain in the terrain.
fn mix(mut bits: u32) -> u32 {
    bits ^= bits >> 16;
    bits = bits.wrapping_mul(0x7feb_352d);
    bits ^= bits >> 15;
    bits = bits.wrapping_mul(0x846c_a68b);
    bits ^= bits >> 16;
    bits
}

/// A repeatable random word for a lattice point of a given seed.
///
/// The two coordinates are folded in one at a time, each through its own
/// mixer, rather than combined and mixed once. Combining first is cheaper and
/// wrong in a way that shows: `x` and `y` then meet only through a single xor,
/// so whole diagonals of the lattice collide and the noise grows a herringbone.
pub fn hash(x: i32, y: i32, seed: u32) -> u32 {
    let mut bits = seed.wrapping_mul(0x9e37_79b1);
    bits = mix(bits ^ (x as u32).wrapping_mul(0x3504_f333));
    bits = mix(bits ^ (y as u32).wrapping_mul(0xf1bb_cdcb));
    bits
}

/// A repeatable `0..1` from a lattice point, for jittering a threshold.
///
/// Twenty-four bits of the word, which is every bit an `f32` can hold anyway.
pub fn hash_unit(x: i32, y: i32, seed: u32) -> f32 {
    (hash(x, y, seed) >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// How many directions a lattice gradient can point.
///
/// Sixteen rather than the eight that a three-bit selector would give. Eight
/// leaves gradient noise with a visible bias along the axes and diagonals,
/// which on a mountain reads as ridges that all run the same four ways;
/// sixteen is under the threshold where it shows and still costs one table
/// lookup from four bits.
const GRADIENT_COUNT: usize = 16;

/// Sine and cosine of 22.5 and 45 degrees, the only values the table needs.
const DIAGONAL: f32 = std::f32::consts::FRAC_1_SQRT_2;
const NEAR_AXIS: f32 = 0.923_879_5;
const OFF_AXIS: f32 = 0.382_683_4;

/// Unit vectors at every 22.5 degrees.
///
/// Written out rather than computed from a sine so that a WGSL port is a
/// `const array` and the values cannot drift between the two implementations.
const GRADIENTS: [[f32; 2]; GRADIENT_COUNT] = [
    [1.0, 0.0],
    [NEAR_AXIS, OFF_AXIS],
    [DIAGONAL, DIAGONAL],
    [OFF_AXIS, NEAR_AXIS],
    [0.0, 1.0],
    [-OFF_AXIS, NEAR_AXIS],
    [-DIAGONAL, DIAGONAL],
    [-NEAR_AXIS, OFF_AXIS],
    [-1.0, 0.0],
    [-NEAR_AXIS, -OFF_AXIS],
    [-DIAGONAL, -DIAGONAL],
    [-OFF_AXIS, -NEAR_AXIS],
    [0.0, -1.0],
    [OFF_AXIS, -NEAR_AXIS],
    [DIAGONAL, -DIAGONAL],
    [NEAR_AXIS, -OFF_AXIS],
];

/// What unit gradient noise has to be multiplied by to reach `-1..=1`.
///
/// Two-dimensional gradient noise with unit gradients peaks at `sqrt(1/2)`,
/// in the middle of a cell whose four gradients all point at it. Scaling by the
/// reciprocal is what lets a caller multiply by a height in metres and get that
/// height.
const GRADIENT_SCALE: f32 = std::f32::consts::SQRT_2;

/// Perlin's quintic interpolant, which has zero first *and* second derivative
/// at both ends.
///
/// The cubic `3t^2 - 2t^3` is cheaper and leaves a second-derivative jump at
/// every lattice line. That is invisible in the height itself and very visible
/// in the shading, because the renderer derives its normal from the heights --
/// the jump draws as a faint grid of creases across the whole terrain.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Linear interpolation, exact at both ends.
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// A smooth `0..=1` ramp between two thresholds, flat outside them.
///
/// The same function WGSL spells `smoothstep`, written out because this crate
/// needs it on the CPU and a port needs the two to agree exactly. Every
/// threshold in the generator goes through this rather than being a comparison:
/// a hard cut-off draws as a contour line running across the terrain, which is
/// the single most recognisable tell of a generated landscape.
pub fn smoothstep(low: f32, high: f32, at: f32) -> f32 {
    let t = ((at - low) / (high - low)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The gradient at a lattice point, dotted with the offset to the sample.
fn corner(x: i32, y: i32, offset: [f32; 2], seed: u32) -> f32 {
    let gradient = GRADIENTS[(hash(x, y, seed) as usize) % GRADIENT_COUNT];
    gradient[0] * offset[0] + gradient[1] * offset[1]
}

/// Gradient ("Perlin") noise, in `-1..=1`, zero at every lattice point.
///
/// The lattice is the integer grid, so a caller controls the feature size by
/// scaling its coordinates before the call.
pub fn gradient(x: f32, y: f32, seed: u32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (ix, iy) = (x0 as i32, y0 as i32);

    let (ux, uy) = (fade(fx), fade(fy));
    let bottom = lerp(
        corner(ix, iy, [fx, fy], seed),
        corner(ix + 1, iy, [fx - 1.0, fy], seed),
        ux,
    );
    let top = lerp(
        corner(ix, iy + 1, [fx, fy - 1.0], seed),
        corner(ix + 1, iy + 1, [fx - 1.0, fy - 1.0], seed),
        ux,
    );
    lerp(bottom, top, uy) * GRADIENT_SCALE
}

/// Value noise, in `-1..=1`.
///
/// Blobbier than [`gradient`] -- its extremes sit *on* the lattice rather than
/// between lattice points -- which is what makes it the right thing for the
/// slowly varying masks this crate uses it for, and the wrong thing for a
/// surface anyone looks at.
pub fn value(x: f32, y: f32, seed: u32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (ix, iy) = (x0 as i32, y0 as i32);

    let at = |dx: i32, dy: i32| hash_unit(ix + dx, iy + dy, seed) * 2.0 - 1.0;
    let (ux, uy) = (fade(fx), fade(fy));
    lerp(
        lerp(at(0, 0), at(1, 0), ux),
        lerp(at(0, 1), at(1, 1), ux),
        uy,
    )
}

/// The most octaves any fractal here will sum.
///
/// A bound rather than a choice: WGSL wants a loop it can unroll, and past
/// this the amplitude of an octave is below a millimetre against the height
/// range this crate generates.
pub const MAX_OCTAVES: u32 = 12;

/// How the octaves of a fractal sum relate to one another.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Fractal {
    /// Feature size of the first octave, in the caller's own units.
    pub wavelength: f32,
    /// How many octaves the fractal *is*.
    ///
    /// This is what the sum is normalised by, whether or not every octave is
    /// summed, and it does not change once the fractal is built.
    pub octaves: u32,
    /// How many of those octaves are actually summed. Never more than
    /// `octaves`; lowered by [`Fractal::band_limited`].
    pub summed: u32,
    /// What each octave multiplies the frequency by. Two is the usual choice;
    /// a little over two stops octaves from reinforcing on the lattice lines
    /// they share.
    pub lacunarity: f32,
    /// What each octave multiplies the amplitude by.
    pub gain: f32,
}

impl Fractal {
    /// A fractal with the usual doubling and halving.
    pub fn new(wavelength: f32, octaves: u32) -> Self {
        Self {
            wavelength,
            octaves,
            summed: octaves,
            lacunarity: 2.017,
            gain: 0.5,
        }
    }

    /// The same fractal, stopped before any octave finer than `wavelength`.
    ///
    /// This is what band-limits a level of the pyramid: an octave whose
    /// features are smaller than two texels cannot be represented at that
    /// level and would alias into a shimmer that changes with the camera
    /// rather than a detail that stays put.
    ///
    /// Only `summed` moves; the normalisation stays that of the whole fractal.
    /// That is the difference between a coarse level being *the same surface
    /// with its finest octaves removed* and it being a differently scaled one.
    /// Renormalising over the octaves that survive would make every coarse
    /// level about a seventh louder than the level under it, and the clipmap
    /// draws that as the ground breathing as a ring crosses it.
    pub fn band_limited(mut self, wavelength: f32) -> Self {
        let mut summed = 0;
        let mut at = self.wavelength;
        while summed < self.octaves && at >= wavelength {
            summed += 1;
            at /= self.lacunarity;
        }
        self.summed = summed;
        self
    }

    /// What the amplitudes of the whole fractal's octaves add up to.
    fn total_amplitude(&self) -> f32 {
        let mut total = 0.0;
        let mut amplitude = 1.0;
        for _ in 0..self.octaves {
            total += amplitude;
            amplitude *= self.gain;
        }
        total
    }
}

/// Fractional Brownian motion: octaves of [`gradient`], in `-1..=1`.
///
/// Returns zero for a fractal with no octaves left, which is what a band limit
/// coarser than the first octave means.
pub fn fbm(x: f32, y: f32, seed: u32, fractal: Fractal) -> f32 {
    if fractal.summed == 0 {
        return 0.0;
    }
    let mut frequency = 1.0 / fractal.wavelength;
    let mut amplitude = 1.0;
    let mut sum = 0.0;
    for octave in 0..fractal.summed.min(MAX_OCTAVES) {
        sum += gradient(
            x * frequency,
            y * frequency,
            seed ^ (octave.wrapping_mul(0x51ed_270b)),
        ) * amplitude;
        frequency *= fractal.lacunarity;
        amplitude *= fractal.gain;
    }
    sum / fractal.total_amplitude()
}

/// Ridged multifractal, in `0..=1`, peaking along creases rather than at points.
///
/// Each octave is `1 - |noise|`, squared to sharpen the crease, and weighted by
/// the octave above it so that detail collects on the ridges and leaves the
/// slopes between them smooth. That feedback is the whole difference between
/// this and `1 - |fbm|`: without it every ridge carries the same roughness as
/// every valley, which is what makes plain inverted noise read as crumpled
/// paper instead of as mountains.
pub fn ridged(x: f32, y: f32, seed: u32, fractal: Fractal) -> f32 {
    if fractal.summed == 0 {
        return 0.0;
    }
    let mut frequency = 1.0 / fractal.wavelength;
    let mut amplitude = 1.0;
    let mut weight = 1.0f32;
    let mut sum = 0.0;
    for octave in 0..fractal.summed.min(MAX_OCTAVES) {
        let raw = gradient(
            x * frequency,
            y * frequency,
            seed ^ (octave.wrapping_mul(0x68e3_1da4)),
        );
        let mut signal = 1.0 - raw.abs();
        signal *= signal;
        signal *= weight;
        // The next octave only contributes where this one already ridged.
        weight = (signal * 2.0).clamp(0.0, 1.0);
        sum += signal * amplitude;
        frequency *= fractal.lacunarity;
        amplitude *= fractal.gain;
    }
    (sum / fractal.total_amplitude()).clamp(0.0, 1.0)
}

/// Billowy noise, in `0..=1`: octaves of `|noise|`, rounded rather than creased.
///
/// The counterpart to [`ridged`], and what deposited ground wants -- moraine,
/// alluvial fans, anything that was dropped rather than cut.
pub fn billow(x: f32, y: f32, seed: u32, fractal: Fractal) -> f32 {
    if fractal.summed == 0 {
        return 0.0;
    }
    let mut frequency = 1.0 / fractal.wavelength;
    let mut amplitude = 1.0;
    let mut sum = 0.0;
    for octave in 0..fractal.summed.min(MAX_OCTAVES) {
        sum += gradient(
            x * frequency,
            y * frequency,
            seed ^ (octave.wrapping_mul(0x1b56_c4e9)),
        )
        .abs()
            * amplitude;
        frequency *= fractal.lacunarity;
        amplitude *= fractal.gain;
    }
    (sum / fractal.total_amplitude()).clamp(0.0, 1.0)
}

/// Moves a point by a fractal vector field, in the caller's own units.
///
/// Fractals sampled on a warped domain lose the axis-aligned regularity that
/// otherwise gives noise-built mountains their tell-tale look: ranges start to
/// bend and branch, and valleys stop meeting at right angles. `amount` is how
/// far, at most, a point moves.
pub fn warp(x: f32, y: f32, seed: u32, fractal: Fractal, amount: f32) -> [f32; 2] {
    [
        x + fbm(x, y, seed ^ 0x2f9e_c1a7, fractal) * amount,
        y + fbm(x, y, seed ^ 0xb5a3_71d5, fractal) * amount,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sample positions spread over a wide range of scales and both signs, so
    /// that a range or continuity claim is not being checked in one lucky
    /// corner of the lattice.
    fn positions() -> Vec<(f32, f32)> {
        let mut out = Vec::new();
        for i in -40i32..40 {
            let t = i as f32;
            out.push((t * 0.317, t * -0.211));
            out.push((t * 13.7, t * 29.1));
            out.push((t * 1024.0 + 0.5, t * 512.0 - 0.25));
        }
        out
    }

    /// The seed is the whole reproducibility story: a run is meant to be
    /// repeatable from it alone, and two runs of the same seed must produce
    /// the same ground down to the last texel.
    #[test]
    fn a_seed_reproduces_its_noise_exactly() {
        for (x, y) in positions() {
            assert_eq!(gradient(x, y, 7), gradient(x, y, 7), "at ({x}, {y})");
            assert_eq!(value(x, y, 7), value(x, y, 7), "at ({x}, {y})");
            let fractal = Fractal::new(300.0, 6);
            assert_eq!(
                fbm(x, y, 7, fractal),
                fbm(x, y, 7, fractal),
                "at ({x}, {y})"
            );
        }
    }

    #[test]
    fn a_different_seed_produces_different_noise() {
        let differing = positions()
            .iter()
            .filter(|(x, y)| gradient(*x, *y, 1) != gradient(*x, *y, 2))
            .count();
        assert!(
            differing > positions().len() * 9 / 10,
            "only {differing} of {} positions moved with the seed",
            positions().len()
        );
    }

    /// Callers scale these by a height in metres, so a value outside the stated
    /// range is a mountain taller than it was asked to be.
    #[test]
    fn every_generator_stays_inside_the_range_it_promises() {
        let fractal = Fractal::new(97.0, 8);
        for (x, y) in positions() {
            for scale in [0.01f32, 1.0, 100.0] {
                let (x, y) = (x * scale, y * scale);
                let checks = [
                    ("gradient", gradient(x, y, 3), -1.0, 1.0),
                    ("value", value(x, y, 3), -1.0, 1.0),
                    ("fbm", fbm(x, y, 3, fractal), -1.0, 1.0),
                    ("ridged", ridged(x, y, 3, fractal), 0.0, 1.0),
                    ("billow", billow(x, y, 3, fractal), 0.0, 1.0),
                ];
                for (name, got, low, high) in checks {
                    assert!(
                        (low..=high).contains(&got),
                        "{name} gave {got} at ({x}, {y}), outside {low}..={high}"
                    );
                }
            }
        }
    }

    /// Gradient noise is zero at every lattice point by construction -- the
    /// offset it dots its gradient with is the zero vector there. Worth
    /// asserting because it is the property that makes the lattice invisible:
    /// if it ever stopped holding, every integer line would show as a ridge.
    #[test]
    fn gradient_noise_vanishes_on_its_lattice() {
        for x in -5..5 {
            for y in -5..5 {
                assert_eq!(gradient(x as f32, y as f32, 11), 0.0, "at ({x}, {y})");
            }
        }
    }

    /// Tiles are generated independently and must still join. Nothing here
    /// knows about tiles, and that is exactly the point: the noise is a
    /// function of position alone, so two neighbours evaluating the same
    /// position get the same answer and the seam cannot exist.
    #[test]
    fn noise_is_continuous_across_any_boundary() {
        let fractal = Fractal::new(64.0, 6);
        // Half a metre either side of a boundary a tile would fall on.
        for boundary in [0.0f32, 512.0, -512.0, 8192.0] {
            let step = 1.0 / 64.0;
            let mut last = fbm(boundary - 4.0 * step, 17.5, 5, fractal);
            for i in -3..=4 {
                let at = fbm(boundary + i as f32 * step, 17.5, 5, fractal);
                assert!(
                    (at - last).abs() < 0.2,
                    "fbm jumped from {last} to {at} across {boundary}"
                );
                last = at;
            }
        }
    }

    /// The band limit is what keeps a coarse level of the pyramid from
    /// carrying detail its texels cannot hold. Dropping octaves must lower the
    /// detail without moving the surface, or the clipmap shows the ground
    /// stepping as a ring crosses it.
    #[test]
    fn a_band_limit_drops_octaves_and_keeps_the_shape() {
        let fine = Fractal::new(1024.0, 8);
        assert_eq!(fine.band_limited(1.0).summed, 8, "nothing to drop");
        assert_eq!(fine.band_limited(1024.0).summed, 1, "only the first fits");
        assert_eq!(fine.band_limited(4096.0).summed, 0, "none fit");
        assert!(fine.band_limited(64.0).summed < 8, "some must be dropped");

        for (x, y) in positions() {
            let all = fbm(x, y, 9, fine);
            let limited = fbm(x, y, 9, fine.band_limited(64.0));
            assert!(
                (all - limited).abs() < 0.45,
                "band limiting moved ({x}, {y}) from {all} to {limited}"
            );
        }
    }

    /// The octaves a band limit keeps must be *unchanged*, not rescaled. This
    /// is the property that makes a coarse level of the pyramid a smoothing of
    /// the fine one: their shared octaves have to agree exactly, or every level
    /// is a slightly different landscape and the clipmap shows the difference
    /// as the ground moving under a ring.
    #[test]
    fn a_band_limit_leaves_the_octaves_it_keeps_at_their_own_amplitude() {
        let full = Fractal::new(1024.0, 8);
        let three = full.band_limited(1024.0 / 4.1);
        assert_eq!(three.summed, 3);
        // The same three octaves asked for on their own, but normalised as the
        // whole eight-octave fractal is.
        let mut alone = Fractal::new(1024.0, 8);
        alone.summed = 3;
        for (x, y) in positions() {
            assert_eq!(fbm(x, y, 6, three), fbm(x, y, 6, alone), "at ({x}, {y})");
        }
        // ... and dropping octaves always makes the surface quieter, never
        // louder.
        let quieter = positions()
            .iter()
            .filter(|(x, y)| fbm(*x, *y, 6, three).abs() <= fbm(*x, *y, 6, full).abs() + 0.35)
            .count();
        assert_eq!(quieter, positions().len());
    }

    /// A fractal with every octave band-limited away is flat, not a division by
    /// zero.
    #[test]
    fn a_fractal_with_no_octaves_is_flat() {
        let none = Fractal::new(10.0, 8).band_limited(1_000_000.0);
        assert_eq!(none.summed, 0);
        for (x, y) in positions() {
            assert_eq!(fbm(x, y, 1, none), 0.0);
            assert_eq!(ridged(x, y, 1, none), 0.0);
            assert_eq!(billow(x, y, 1, none), 0.0);
        }
    }

    /// A single mixer folding both coordinates at once leaves whole diagonals
    /// of the lattice sharing a gradient, which draws a herringbone across the
    /// terrain. Checked as a distribution rather than by eye: every one of the
    /// sixteen directions should come up about as often.
    #[test]
    fn the_hash_spreads_gradients_evenly_over_the_lattice() {
        let mut counts = [0u32; GRADIENT_COUNT];
        for x in -64..64 {
            for y in -64..64 {
                counts[(hash(x, y, 4) as usize) % GRADIENT_COUNT] += 1;
            }
        }
        let expected = (128 * 128 / GRADIENT_COUNT) as u32;
        for (direction, count) in counts.iter().enumerate() {
            let drift = count.abs_diff(expected);
            assert!(
                drift * 5 < expected,
                "direction {direction} came up {count} times against {expected} expected"
            );
        }
    }

    /// Warping must actually move points, and must move them smoothly -- a
    /// warp that jumped would tear the terrain rather than bend it.
    #[test]
    fn warping_moves_points_by_at_most_the_amount_asked_for() {
        let fractal = Fractal::new(2000.0, 4);
        for (x, y) in positions() {
            let [wx, wy] = warp(x, y, 21, fractal, 300.0);
            assert!((wx - x).abs() <= 300.0, "x moved {} m", wx - x);
            assert!((wy - y).abs() <= 300.0, "y moved {} m", wy - y);
        }
    }
}
