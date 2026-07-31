//! Narrowing the max pyramid on its way into the texture.
//!
//! The quadtree the far field skips empty space with is no longer built here.
//! `terrain-process` writes it beside the elevation it bounds, as an ordinary
//! product of tiles, and the clipmap reads it with the same machinery that
//! carries the heights: [`crate::terrain::tiles::TileStore`] out,
//! `write_texture` in. See `crates/terrain-tiles/src/maxima.rs` for what the
//! cells mean and why their squares are closed.
//!
//! One thing is worth repeating here, because it is what the shifted indexing in
//! [`crate::terrain::gpu`] rests on. The product holds
//!
//! ```text
//! M[m][i, j] = max of the raster's level-0 samples over the closed square
//!              [i * 2^m, (i + 1) * 2^m] x [j * 2^m, (j + 1) * 2^m]
//! ```
//!
//! and clipmap level `l`'s depth-`m` cell covers `[i * 2^m, (i + 1) * 2^m]` in
//! level `l`'s texels, which is `[i * 2^(l+m), (i + 1) * 2^(l+m)]` in level-0
//! texels. Same square, same indices. So level `l` depth `m` is **product level
//! `l + m`**, read at the window origin shifted down by `m`, and one pyramid
//! serves every level.
//!
//! What remains here is the one conversion that has to happen at upload time,
//! because the product is written at full precision and the texture holds half.

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
