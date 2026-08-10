//! Rounding a max pyramid cell to half precision, the only way it may be
//! rounded.
//!
//! The pyramid itself is neither read from disk nor built here any more:
//! `cs_maxima` in `src/terrain.wgsl` derives it on the GPU from the heights
//! once they are resident, which is one spelling of the recurrence rather than
//! two. See `crates/terrain-tiles/src/maxima.rs` for what the cells mean and
//! why their squares are closed.
//!
//! What remains is this rounding, and it remains because the shader carries a
//! transcription of it. WGSL's `pack2x16float` is round-to-nearest on most
//! backends and towards zero on some, and neither is towards positive infinity,
//! so the shader converts and then corrects -- against exactly the rule below.
//! Keeping the Rust one is what lets a test say the two agree.

#[cfg(test)]
use half::f16;

/// The smallest half float that is not below `height`.
///
/// The pyramid is stored at half precision, which is worth three hundred
/// megabytes at the widest window, but only a ceiling that is rounded the right
/// way stays a ceiling. Rounding to nearest would put some cells a little below
/// the ground they are supposed to bound, and a ray comparing against one of
/// those skips a cell it should have descended into -- a ridge with a hole
/// through it, which is the one failure this whole structure exists to prevent.
///
/// Rounding up costs the opposite: a cell claims ground very slightly higher
/// than it holds, so a ray descends into one it could have skipped. Half floats
/// carry eleven bits of mantissa, so that is a metre or two at the height of a
/// mountain and a great deal less near sea level.
#[cfg(test)]
pub fn ceiling_half(height: f32) -> f16 {
    let rounded = f16::from_f32(height);
    if rounded.to_f32() >= height || rounded.is_infinite() {
        return rounded;
    }
    // One step towards positive infinity. The bit patterns of half floats
    // increase away from zero within a sign, so the direction depends on it.
    let bits = rounded.to_bits();
    f16::from_bits(if rounded.is_sign_negative() {
        bits - 1
    } else {
        bits + 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ceiling that rounds down is not a ceiling.
    #[test]
    fn a_half_float_ceiling_is_never_below_what_it_bounds() {
        // Across the range terrain occupies, the sentinel it uses for nothing
        // known, and the awkward values around zero.
        let mut seed = 0x51ce_1111_2222_3333u64;
        let mut heights: Vec<f32> = vec![0.0, -0.0, 1.0, -32767.0, 8848.0, 1e-8, -1e-8];
        for _ in 0..20_000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            heights.push((seed % 4_000_000) as f32 / 1000.0 - 500.0);
        }

        for height in heights {
            let rounded = ceiling_half(height).to_f32();
            assert!(rounded >= height, "{height} rounded down to {rounded}");
            // ... and no further above it than it has to be, or the pyramid
            // would grow slack for nothing. One step down the bit pattern,
            // which runs away from zero within a sign, must be too low.
            let bits = ceiling_half(height).to_bits();
            if rounded != 0.0 {
                let below = f16::from_bits(if rounded < 0.0 { bits + 1 } else { bits - 1 });
                assert!(
                    below.to_f32() < height,
                    "{height} could have been bounded by {below} rather than {rounded}"
                );
            }
        }
    }

    /// Ground nothing is known about has to stay recognisable as such.
    #[test]
    fn the_nodata_sentinel_survives_being_halved() {
        let rounded = ceiling_half(-32767.0).to_f32();
        assert!(
            rounded < crate::terrain::NODATA_BELOW,
            "the sentinel rounded up to {rounded}, which reads as ground"
        );
    }
}
