//! The boundary between the clipmap and wherever its texels come from.
//!
//! The clipmap draws distant ground on a coarser grid than nearby ground, so it
//! needs the raster pre-filtered at every power-of-two reduction. That filtering
//! now happens in `terrain-download`, which writes the whole chain to disk as
//! tiles; [`crate::terrain::tiles::TileStore`] is what the simulator actually
//! reads through.
//!
//! What remains here is the trait that hides all of it, and an in-memory
//! implementation the tests build synthetic terrain with. Nothing is ever
//! uploaded whole: the GPU only sees the small windows the clipmap asks for,
//! which is the boundary that keeps residency independent of how large the
//! dataset grows.

use glam::{IVec2, UVec2};

/// The texel types, re-exported so callers need not know they are shared with
/// the tool that writes the tiles.
pub use terrain_tiles::Srgb8;

use terrain_tiles::Texel;

/// A rectangular window of texels, at some level of detail.
///
/// Implementors own the data however they like -- in memory, memory-mapped, or
/// fetched from disk on demand -- so long as a window can be produced on
/// request. The clipmap deliberately never asks for a whole level.
pub trait RasterSource {
    /// Number of levels, where 0 is the full-resolution raster.
    fn level_count(&self) -> u32;

    /// Copies `size` texels starting at `origin` into `out`, tightly packed.
    ///
    /// Reads outside the raster clamp to the nearest edge texel, so a window
    /// hanging off the edge of the world repeats its border rather than
    /// producing a hole. `out` must hold `size.x * size.y * texel_bytes()`.
    fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]);
}

/// One level of a [`Pyramid`].
#[derive(Clone, Debug)]
pub struct Level<T> {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<T>,
}

impl<T: Texel> Level<T> {
    pub fn new(width: u32, height: u32, texels: Vec<T>) -> Self {
        assert_eq!(
            texels.len(),
            (width as usize) * (height as usize),
            "level data does not match its dimensions"
        );
        Self {
            width,
            height,
            texels,
        }
    }

    fn get(&self, x: u32, y: u32) -> T {
        self.texels[(y as usize) * (self.width as usize) + (x as usize)]
    }
}

/// A raster and every power-of-two reduction of it, held in memory.
///
/// The simulator streams the bulk of its mip chain off disk, where the
/// downloader built it once rather than on every launch. This is what the tests
/// build synthetic terrain with, and what [`Resident`] continues that chain
/// with once the stored levels run out.
#[derive(Clone, Debug)]
pub struct Pyramid<T> {
    levels: Vec<Level<T>>,
}

impl<T: Texel> Pyramid<T> {
    /// Builds the reduction chain from a full-resolution level, down to 1x1.
    pub fn build(base: Level<T>) -> Self {
        let mut levels = vec![base];
        while {
            let last = levels.last().expect("a pyramid always has a base level");
            last.width > 1 || last.height > 1
        } {
            levels.push(reduce(levels.last().expect("just checked")));
        }
        Self { levels }
    }

    /// Size of `level` in texels.
    #[cfg(test)]
    pub fn level_size(&self, level: u32) -> UVec2 {
        let level = &self.levels[(level as usize).min(self.levels.len() - 1)];
        UVec2::new(level.width, level.height)
    }

    #[cfg(test)]
    fn level(&self, level: u32) -> &Level<T> {
        &self.levels[level as usize]
    }
}

#[cfg(test)]
impl Pyramid<f32> {
    /// The lowest and highest value in the full-resolution level.
    pub fn value_range(&self) -> (f32, f32) {
        self.levels[0]
            .texels
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            })
    }
}

/// Halves a level, rounding up so a single odd texel still gets a home.
fn reduce<T: Texel>(fine: &Level<T>) -> Level<T> {
    let width = (fine.width.div_ceil(2)).max(1);
    let height = (fine.height.div_ceil(2)).max(1);

    let mut texels = Vec::with_capacity((width as usize) * (height as usize));
    let mut samples = Vec::with_capacity(4);
    for y in 0..height {
        for x in 0..width {
            samples.clear();
            for dy in 0..2 {
                for dx in 0..2 {
                    let (sx, sy) = (x * 2 + dx, y * 2 + dy);
                    if sx < fine.width && sy < fine.height {
                        let sample = fine.get(sx, sy);
                        // Dropped rather than averaged in, so that one hole
                        // among three real texels does not come out as a value
                        // too low to be ground and too high to be recognised as
                        // a hole. A coarse texel with no real children stays a
                        // hole itself.
                        if !sample.is_nodata() {
                            samples.push(sample);
                        }
                    }
                }
            }
            texels.push(if samples.is_empty() {
                T::NODATA
            } else {
                T::box_filter(&samples)
            });
        }
    }

    Level::new(width, height, texels)
}

impl<T: Texel> RasterSource for Pyramid<T> {
    fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]) {
        let level = &self.levels[(level as usize).min(self.levels.len() - 1)];
        let (max_x, max_y) = (level.width as i32 - 1, level.height as i32 - 1);

        let texels: &mut [T] = bytemuck::cast_slice_mut(
            &mut out[..(size.x as usize) * (size.y as usize) * size_of::<T>()],
        );
        for row in 0..size.y {
            let y = (origin.y + row as i32).clamp(0, max_y) as u32;
            for column in 0..size.x {
                let x = (origin.x + column as i32).clamp(0, max_x) as u32;
                texels[(row as usize) * (size.x as usize) + (column as usize)] = level.get(x, y);
            }
        }
    }
}

/// A source whose coarse end is held in memory instead of streamed.
///
/// The tile pyramid on disk stops wherever the downloader stopped, which is
/// often well short of a level that spans the whole dataset -- the installed
/// 8192-texel raster stores five levels where seven are needed. A clipmap can
/// only build as many levels as it is given, so the outermost ring ends up
/// reaching a couple of kilometres of ground that goes on for eight, and
/// everything past it is sky whether it is drawn or marched.
///
/// This continues the chain rather than accepting where it stopped: it reads the
/// coarsest stored level whole, reduces it to a single texel, and answers for
/// every level from there up. Those levels are small -- the coarsest stored
/// level of that raster is 512 texels square, so a megabyte, and everything
/// above it a third as much again -- which is what makes holding them
/// affordable. Two things fall out of that. The far field can see to the edge of
/// the data from anywhere in it, and the levels whose windows sweep fastest
/// across the ground come off the streaming path altogether.
///
/// Reducing here rather than in the downloader is deliberate. These levels are
/// derived, not measured, so there is nothing to gain by writing them down, and
/// a pyramid already on disk would have to be regenerated to grow them.
pub struct Resident<T> {
    /// Levels finer than `first` are still someone else's problem.
    inner: Box<dyn RasterSource>,
    /// Which level `coarse`'s own level zero stands for.
    first: u32,
    coarse: Pyramid<T>,
}

impl<T: Texel> Resident<T> {
    /// Reads the coarsest level `inner` holds and carries on from it.
    ///
    /// `raster` is the full-resolution size, which is what says how large that
    /// level is; a source knows how many levels it has but not how wide.
    pub fn over(inner: Box<dyn RasterSource>, raster: UVec2) -> Self {
        let first = inner.level_count().saturating_sub(1);
        let size = UVec2::new((raster.x >> first).max(1), (raster.y >> first).max(1));

        let mut texels = vec![T::default(); (size.x as usize) * (size.y as usize)];
        inner.read_rect(
            first,
            IVec2::ZERO,
            size,
            bytemuck::cast_slice_mut(&mut texels),
        );

        Self {
            inner,
            first,
            coarse: Pyramid::build(Level::new(size.x, size.y, texels)),
        }
    }
}

impl<T: Texel> RasterSource for Resident<T> {
    fn level_count(&self) -> u32 {
        // The stored levels below the handover, then the whole chain above it.
        self.first + self.coarse.level_count()
    }

    fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]) {
        if level < self.first {
            self.inner.read_rect(level, origin, size, out);
        } else {
            self.coarse.read_rect(level - self.first, origin, size, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(width: u32, height: u32) -> Level<f32> {
        let texels = (0..width * height).map(|i| i as f32).collect();
        Level::new(width, height, texels)
    }

    #[test]
    fn a_pyramid_halves_until_it_reaches_a_single_texel() {
        let pyramid = Pyramid::build(ramp(8, 4));

        let sizes: Vec<_> = (0..pyramid.level_count())
            .map(|l| pyramid.level_size(l).to_array())
            .collect();
        assert_eq!(sizes, [[8, 4], [4, 2], [2, 1], [1, 1]]);
    }

    #[test]
    fn odd_dimensions_round_up_and_still_reach_one() {
        let pyramid = Pyramid::build(ramp(5, 3));

        let sizes: Vec<_> = (0..pyramid.level_count())
            .map(|l| pyramid.level_size(l).to_array())
            .collect();
        assert_eq!(sizes, [[5, 3], [3, 2], [2, 1], [1, 1]]);
    }

    #[test]
    fn a_coarse_texel_is_the_mean_of_the_four_it_covers() {
        // 2x2 blocks of a row-major ramp: [0 1 / 4 5] averages to 2.5.
        let pyramid = Pyramid::build(ramp(4, 4));
        assert_eq!(pyramid.level(1).get(0, 0), (0.0 + 1.0 + 4.0 + 5.0) / 4.0);
        assert_eq!(
            pyramid.level(1).get(1, 1),
            (10.0 + 11.0 + 14.0 + 15.0) / 4.0
        );
    }

    #[test]
    fn an_odd_edge_averages_only_the_texels_that_exist() {
        // Width 3 leaves the last coarse column covering a single fine column,
        // so it must not be dragged towards zero by a phantom partner.
        let pyramid = Pyramid::build(ramp(3, 2));
        assert_eq!(pyramid.level(1).get(1, 0), (2.0 + 5.0) / 2.0);
    }

    /// A source that stops partway down, the way a stored tile pyramid does.
    ///
    /// Asks are checked rather than clamped: reading a level it does not have is
    /// the mistake worth catching, because clamping would hide it behind a level
    /// that quietly answers for every coarser one.
    struct Shallow {
        whole: Pyramid<f32>,
        levels: u32,
    }

    impl RasterSource for Shallow {
        fn level_count(&self) -> u32 {
            self.levels
        }

        fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]) {
            assert!(
                level < self.levels,
                "asked for level {level} of {}",
                self.levels
            );
            self.whole.read_rect(level, origin, size, out);
        }
    }

    /// The point of [`Resident`]: a pyramid that stops short still ends up with
    /// levels all the way to a single texel, and they are the same levels the
    /// downloader would have written had it kept going.
    ///
    /// Without this the clipmap gets only as many levels as the store happens to
    /// hold, and its outermost ring reaches a fraction of the ground the data
    /// covers -- two kilometres of eight, for the installed pyramid.
    #[test]
    fn the_resident_tail_carries_the_chain_past_where_the_store_stops() {
        let base = ramp(64, 64);
        let stored = 3;
        let resident = Resident::<f32>::over(
            Box::new(Shallow {
                whole: Pyramid::build(base.clone()),
                levels: stored,
            }),
            UVec2::splat(64),
        );

        // Sixty-four texels reduce to one in seven levels however few were kept.
        let whole = Pyramid::build(base);
        assert_eq!(resident.level_count(), 7);
        assert_eq!(resident.level_count(), whole.level_count());

        // Reducing a reduction is reducing the original, so the levels invented
        // here are indistinguishable from the ones that were not written down.
        // The window deliberately overhangs the raster, since that is what a
        // clipmap window does at the coarse end and it is where clamping to the
        // border has to agree as well.
        for level in 0..resident.level_count() {
            let (mut mine, mut theirs) = ([0f32; 100], [0f32; 100]);
            let window = (IVec2::splat(-3), UVec2::splat(10));
            resident.read_rect(
                level,
                window.0,
                window.1,
                bytemuck::cast_slice_mut(&mut mine),
            );
            whole.read_rect(
                level,
                window.0,
                window.1,
                bytemuck::cast_slice_mut(&mut theirs),
            );
            assert_eq!(mine, theirs, "level {level}");
        }
    }

    #[test]
    fn colour_mips_are_filtered_in_linear_space() {
        // As near black as a texel can be without meaning "no imagery here".
        let dark = Srgb8([1, 1, 1, 255]);
        let white = Srgb8([255, 255, 255, 255]);
        let checker = Level::new(2, 2, vec![dark, white, white, dark]);

        let averaged = Pyramid::build(checker).level(1).get(0, 0);

        // Half the light, not half the encoded value: mid-grey is 188, not 128.
        // Averaging in sRGB space here would visibly darken every coastline and
        // snow boundary in the coarse levels.
        assert_eq!(averaged.0[0], 188, "expected a linear-space average");
    }

    /// The rule the whole chain rests on, and the one that is silent when broken:
    /// a hole averaged in with real ground comes out at a plausible-looking
    /// elevation that nothing downstream can tell from measured terrain.
    #[test]
    fn a_hole_is_dropped_from_a_coarse_texel_rather_than_averaged_into_it() {
        const NODATA: f32 = -32767.0;
        let ragged = Level::new(2, 2, vec![NODATA, 10.0, 20.0, 30.0]);
        let coarse = Pyramid::build(ragged).level(1).get(0, 0);
        assert_eq!(
            coarse, 20.0,
            "the hole should not have pulled the mean down"
        );

        // With nothing real underneath it, a coarse texel stays a hole -- and
        // stays recognisable as one.
        let empty = Level::new(2, 2, vec![NODATA; 4]);
        let coarse = Pyramid::build(empty).level(1).get(0, 0);
        assert!(
            coarse < crate::terrain::NODATA_BELOW,
            "an entirely unmeasured texel should still read as nodata, got {coarse}"
        );
    }

    /// Black is how imagery says it has nothing, so it is dropped for the same
    /// reason: a quarter-covered coarse texel should be the colour of the ground
    /// that is there, not that colour darkened towards the gaps.
    #[test]
    fn colours_with_nothing_under_them_do_not_darken_their_neighbours() {
        let nothing = Srgb8([0, 0, 0, 255]);
        let green = Srgb8([40, 160, 60, 255]);
        let ragged = Level::new(2, 2, vec![nothing, nothing, nothing, green]);
        assert_eq!(Pyramid::build(ragged).level(1).get(0, 0), green);

        let empty = Level::new(2, 2, vec![nothing; 4]);
        assert_eq!(Pyramid::build(empty).level(1).get(0, 0), nothing);
    }

    #[test]
    fn reading_past_an_edge_repeats_the_border_texel() {
        let pyramid = Pyramid::build(ramp(4, 4));

        let mut out = vec![0u8; 3 * 3 * size_of::<f32>()];
        pyramid.read_rect(0, IVec2::new(-2, -2), UVec2::new(3, 3), &mut out);

        // Every sample of a window entirely off the top-left corner clamps to
        // the corner texel, so a window hanging off the world has no holes.
        let values: &[f32] = bytemuck::cast_slice(&out);
        assert_eq!(values, &[0.0; 9]);
    }

    #[test]
    fn a_window_reads_the_texels_it_asks_for() {
        let pyramid = Pyramid::build(ramp(4, 4));

        let mut out = vec![0u8; 2 * 2 * size_of::<f32>()];
        pyramid.read_rect(0, IVec2::new(1, 2), UVec2::new(2, 2), &mut out);

        let values: &[f32] = bytemuck::cast_slice(&out);
        assert_eq!(values, &[9.0, 10.0, 13.0, 14.0]);
    }

    #[test]
    fn the_value_range_spans_the_full_resolution_level() {
        let pyramid = Pyramid::build(Level::new(2, 2, vec![-7.5f32, 0.0, 3.0, 12.25]));
        assert_eq!(pyramid.value_range(), (-7.5, 12.25));
    }
}
