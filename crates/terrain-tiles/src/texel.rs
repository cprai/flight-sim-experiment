//! Texel types and the filter that builds one coarse texel from four fine ones.
//!
//! This lives beside the grid maths rather than in the renderer because the mip
//! chain is now built by the downloader and read by the renderer. The two have
//! to agree on what a coarse texel *means*, and the sRGB curve below is the part
//! most likely to be got subtly wrong in two places at once.

use std::sync::LazyLock;

/// Whether the colour raster's values are already sRGB-encoded for display.
///
/// Visual satellite products almost always are, and they carry no colour-space
/// tag to say so. If a linear-reflectance source is ever used instead the mips
/// would be filtered twice through the same curve and the terrain would look
/// washed out; flipping this is the whole fix.
pub const COLOUR_IS_SRGB_ENCODED: bool = true;

/// Elevations below this are a raster's nodata rather than ground.
///
/// HRDEM writes -32767 and other producers spell it differently, but the deepest
/// ground on Earth is a small fraction of this, so anything below it is a hole
/// however it was written. Kept in step with `NODATA_BELOW` in
/// `src/terrain.wgsl`.
pub const NODATA_BELOW: f32 = -30_000.0;

/// A texel that knows how to combine with its neighbours to build a coarser mip.
pub trait Texel: Copy + Default + bytemuck::Pod {
    /// Averages the one to four finer texels that a coarse texel covers.
    ///
    /// Fewer than four arrive at the edge of the data, where a neighbour is
    /// missing or holds nodata and the caller has dropped it. Callers must not
    /// pass an empty slice; a coarse texel with no valid children is
    /// [`Texel::NODATA`].
    fn box_filter(samples: &[Self]) -> Self;

    /// Whether this texel means "nothing was measured here".
    ///
    /// Nodata has to be dropped before [`Texel::box_filter`] rather than
    /// averaged into it. One -32767 among three real metres comes out around
    /// -7800: far below any ground, but nowhere near the sentinel, so nothing
    /// downstream recognises it as a hole and it draws as a pit instead.
    fn is_nodata(&self) -> bool;

    /// What a coarse texel holds when every one of its children was nodata.
    const NODATA: Self;
}

impl Texel for f32 {
    fn box_filter(samples: &[Self]) -> Self {
        // A plain mean is the right low-pass for a height field: coarse levels
        // become a smoothed surface rather than an aliased subsample of it.
        // Peaks do lose a little height, which is the intended trade -- taking
        // the maximum instead would keep silhouettes but inflate the terrain
        // and break the continuity the morph between levels depends on.
        samples.iter().sum::<f32>() / samples.len() as f32
    }

    fn is_nodata(&self) -> bool {
        *self < NODATA_BELOW
    }

    const NODATA: Self = -32767.0;
}

/// A colour texel, sRGB-encoded, in the RGBA order the GPU samples.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Srgb8(pub [u8; 4]);

impl Texel for Srgb8 {
    fn box_filter(samples: &[Self]) -> Self {
        // Averaging sRGB values directly averages the wrong quantity: the
        // encoding is a curve, so the mean of two encoded values is darker than
        // the encoding of their mean. Decode, average, re-encode. It shows most
        // at high-contrast boundaries, which is exactly where terrain is
        // interesting.
        let mut channels = [0.0f32; 4];
        for sample in samples {
            for (accumulated, &value) in channels.iter_mut().zip(sample.0.iter()) {
                *accumulated += srgb_to_linear(value);
            }
        }
        let scale = 1.0 / samples.len() as f32;
        Self(std::array::from_fn(|i| linear_to_srgb(channels[i] * scale)))
    }

    /// Imagery carries no sentinel, so black stands in for it: tiles with
    /// nothing under them are written black, and ground that is genuinely this
    /// dark is indistinguishable from no ground at all anyway.
    fn is_nodata(&self) -> bool {
        self.0[..3] == [0, 0, 0]
    }

    const NODATA: Self = Self([0, 0, 0, 255]);
}

/// Decoding is a per-texel cost over tens of millions of texels, and there are
/// only 256 possible inputs, so it is worth doing exactly 256 times.
static SRGB_TO_LINEAR: LazyLock<[f32; 256]> = LazyLock::new(|| {
    std::array::from_fn(|i| {
        let value = i as f32 / 255.0;
        if value <= 0.040_45 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    })
});

/// Decodes one sRGB-encoded byte into the light it stands for.
pub fn srgb_to_linear(value: u8) -> f32 {
    SRGB_TO_LINEAR[usize::from(value)]
}

/// Encodes a linear light value, in 0..1, as an sRGB byte.
pub fn linear_to_srgb(value: f32) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heights_average_arithmetically() {
        assert_eq!(f32::box_filter(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(f32::box_filter(&[7.0]), 7.0);
    }

    /// The whole reason `Srgb8` has its own filter. Averaging the encoded bytes
    /// would give 128; averaging the light they stand for gives noticeably more.
    #[test]
    fn colours_average_in_linear_light_not_in_srgb() {
        let black = Srgb8([0, 0, 0, 255]);
        let white = Srgb8([255, 255, 255, 255]);
        let mixed = Srgb8::box_filter(&[black, black, white, white]);
        assert_eq!(mixed.0[3], 255, "alpha should survive");
        assert!(
            mixed.0[0] > 180,
            "half the light should encode well above the midpoint, got {}",
            mixed.0[0]
        );
    }

    #[test]
    fn averaging_one_colour_returns_it_unchanged() {
        for value in [0u8, 1, 17, 128, 254, 255] {
            let only = Srgb8([value, value, value, 255]);
            assert_eq!(Srgb8::box_filter(&[only]), only, "value {value}");
        }
    }
}
