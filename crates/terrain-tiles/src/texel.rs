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

/// A surface normal, as the two horizontal components of a unit vector.
///
/// The world is Y-up with +X east and -Z north, and raster row zero is the
/// northern edge, so stepping a row moves +Z. The two components stored are
/// therefore east and south: the raster's own two directions, in the order its
/// own indices run. A height field's normal always points upwards, so the third
/// component is `sqrt(1 - east^2 - south^2)` and never needs storing. Two bytes
/// a texel rather than four is what keeps a fourth clipmap texture inside the
/// renderer's memory budget; see `Residency::texture_bytes`.
///
/// Encoding clamps to +/-127, which leaves `(-128, -128)` unreachable by any
/// real normal -- it decodes to a horizontal length above one, and a unit
/// vector's cannot exceed it -- and that is what [`Texel::NODATA`] is.
///
/// Reconstruction is worst where the ground is steepest: as `east^2 + south^2`
/// approaches one the square root turns steep, so one code of error in either
/// component moves the vertical component a long way. Measured, a round trip
/// costs a third of a degree over gentle ground and stays under one degree out
/// to a slope of seventy, but reaches five degrees within ten of vertical.
/// Every stored normal is the mean of sixty-four finer ones, which is what
/// keeps it out of that last band far more often than the ground does.
/// Hemi-octahedral packing would spread the error evenly at the same width, but
/// it fills the square and leaves no value free to mean nodata. These normals
/// are shading detail rather than geometry, so the simpler scheme with a
/// sentinel wins.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Normal {
    pub east: i8,
    pub south: i8,
}

impl Normal {
    /// The largest magnitude a component encodes to, which leaves -128 free.
    const SCALE: f32 = 127.0;

    /// Flat ground.
    pub const UP: Self = Self { east: 0, south: 0 };

    /// Packs a normal, which need not already be unit length.
    ///
    /// `up` is only read for its share of the length: a normal is stored by its
    /// two horizontal components, so a downward-facing one would come back
    /// mirrored. Height fields do not produce those.
    pub fn from_unit(east: f32, up: f32, south: f32) -> Self {
        let length = (east * east + up * up + south * south).sqrt();
        if !length.is_normal() {
            // Nothing to point at -- zero, or a length that arithmetic has
            // already lost. Flat is the answer that cannot mislead a shader.
            return Self::UP;
        }
        let encode = |value: f32| {
            (value / length * Self::SCALE)
                .round()
                .clamp(-Self::SCALE, Self::SCALE) as i8
        };
        Self {
            east: encode(east),
            south: encode(south),
        }
    }

    /// Unpacks a normal as east, up, south.
    ///
    /// Callers must drop nodata first: the sentinel is not a direction and
    /// comes back out of here longer than one.
    pub fn to_unit(self) -> [f32; 3] {
        let east = f32::from(self.east) / Self::SCALE;
        let south = f32::from(self.south) / Self::SCALE;
        [
            east,
            (1.0 - east * east - south * south).max(0.0).sqrt(),
            south,
        ]
    }

    /// How a normal sits in a tile's 16-bit sample, defined in one place.
    ///
    /// East in the low byte, so the two bytes reach the GPU in the order an
    /// `Rg8Snorm` texel wants them without anything having to swap a pair.
    pub const fn to_sample(self) -> u16 {
        (self.east as u8 as u16) | ((self.south as u8 as u16) << 8)
    }

    /// The inverse of [`Normal::to_sample`].
    pub const fn from_sample(sample: u16) -> Self {
        Self {
            east: sample as u8 as i8,
            south: (sample >> 8) as u8 as i8,
        }
    }
}

impl Texel for Normal {
    fn box_filter(samples: &[Self]) -> Self {
        // The mean of the directions, which is a normal map's own answer to
        // minification: it keeps the roughness of the ground under a coarse
        // texel, where taking the normal of the averaged heights would smooth
        // it away.
        let mut sum = [0.0f32; 3];
        for sample in samples {
            for (total, value) in sum.iter_mut().zip(sample.to_unit()) {
                *total += value;
            }
        }
        // Two opposing vertical faces sum to nothing at all. It is the only way
        // to get here, since every term's vertical component is positive.
        Self::from_unit(sum[0], sum[1], sum[2])
    }

    fn is_nodata(&self) -> bool {
        *self == Self::NODATA
    }

    const NODATA: Self = Self {
        east: i8::MIN,
        south: i8::MIN,
    };
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

    /// Every upward direction, by its slope and the angle between it and what
    /// comes back out of two bytes.
    fn directions() -> impl Iterator<Item = (u16, [f32; 3], f32)> {
        (0..90u16).flat_map(|degrees| {
            (0..72u16).map(move |step| {
                let (slope, bearing) = (
                    f32::from(degrees).to_radians(),
                    f32::from(step * 5).to_radians(),
                );
                let unit = [
                    slope.sin() * bearing.cos(),
                    slope.cos(),
                    slope.sin() * bearing.sin(),
                ];
                let back = Normal::from_unit(unit[0], unit[1], unit[2]).to_unit();
                let dot: f32 = unit.iter().zip(back).map(|(a, b)| a * b).sum();
                (degrees, unit, dot.clamp(-1.0, 1.0).acos().to_degrees())
            })
        })
    }

    /// Two bytes cannot hold a direction exactly, so what matters is the size
    /// of the error and where it falls. Measured by ten-degree bands of slope,
    /// it is a third of a degree over anything gentle, still under one at 70
    /// degrees, and only past 80 -- where reconstructing the vertical
    /// component from the other two loses its grip -- does it reach five. A
    /// stored normal is the mean of sixty-four finer ones, so the last band is
    /// far rarer than a cliff in the terrain is.
    #[test]
    fn a_packed_normal_comes_back_close_to_itself() {
        let mut worst = [0.0f32; 9];
        for (degrees, _, error) in directions() {
            let band = usize::from(degrees / 10);
            worst[band] = worst[band].max(error);
        }
        for (band, (measured, allowed)) in worst
            .iter()
            .zip([0.35, 0.35, 0.4, 0.45, 0.5, 0.55, 0.75, 1.6, 5.1])
            .enumerate()
        {
            assert!(
                *measured < allowed,
                "slopes from {} degrees came back {measured} away, over {allowed}",
                band * 10
            );
        }
        assert!(
            worst[0] > 0.01,
            "an exact round trip means nothing was packed"
        );
        assert!(worst[8] > worst[0], "the error should grow with the slope");
    }

    /// The whole reason the sentinel can be a value rather than a flag.
    #[test]
    fn no_real_direction_encodes_to_the_nodata_sentinel() {
        for (_, unit, _) in directions() {
            let packed = Normal::from_unit(unit[0], unit[1], unit[2]);
            assert!(!packed.is_nodata(), "{unit:?} packed to the sentinel");
        }
        // And a vertical wall, the extreme the clamp exists for.
        assert!(!Normal::from_unit(-1.0, 0.0, 0.0).is_nodata());
        assert!(!Normal::from_unit(-1.0, 0.0, -1.0).is_nodata());
    }

    #[test]
    fn a_sample_carries_east_in_its_low_byte() {
        let normal = Normal {
            east: -3,
            south: 100,
        };
        assert_eq!(normal.to_sample(), 0x64fd);
        assert_eq!(Normal::from_sample(normal.to_sample()), normal);
        assert_eq!(
            Normal::from_sample(Normal::NODATA.to_sample()),
            Normal::NODATA
        );
        assert_eq!(Normal::NODATA.to_sample(), 0x8080);
    }

    #[test]
    fn averaging_one_normal_returns_it_unchanged() {
        for normal in [
            Normal::UP,
            Normal {
                east: 40,
                south: -70,
            },
        ] {
            assert_eq!(Normal::box_filter(&[normal]), normal, "{normal:?}");
        }
    }

    /// Averaging directions, not their packed bytes: two slopes either side of
    /// a ridge give the flat top of it, not the mean of two byte pairs.
    #[test]
    fn opposing_walls_average_to_flat_ground() {
        let east = Normal::from_unit(1.0, 0.0, 0.0);
        let west = Normal::from_unit(-1.0, 0.0, 0.0);
        assert_eq!(Normal::box_filter(&[east, west]), Normal::UP);

        let slope = Normal::from_unit(1.0, 1.0, 0.0);
        let counter = Normal::from_unit(-1.0, 1.0, 0.0);
        assert_eq!(Normal::box_filter(&[slope, counter]), Normal::UP);
    }

    #[test]
    fn nothing_at_all_averages_to_flat_ground_rather_than_a_nan() {
        assert_eq!(Normal::from_unit(0.0, 0.0, 0.0), Normal::UP);
        assert_eq!(Normal::from_unit(f32::NAN, 1.0, 0.0), Normal::UP);
    }
}
