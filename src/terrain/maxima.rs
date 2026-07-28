//! The quadtree the raymarched far field skips empty space with.
//!
//! For each clipmap level `l` and each depth `m`, this answers
//!
//! ```text
//! M[l][m][i, j] = max of level l's samples over the closed square
//!                 [i * 2^m, (i + 1) * 2^m] x [j * 2^m, (j + 1) * 2^m]
//! ```
//!
//! in that level's own texel indices, measured from the raster's origin. A ray
//! that stays above `M[l][m][i, j]` across the whole of that square cannot meet
//! the ground inside it, so it skips the square in one step.
//!
//! The square is **closed**, and that is the whole game. A maximum over the
//! samples of a half-open box does not bound the surface *between* the samples,
//! so rays slip through ridges wherever a cell boundary falls on one -- holes
//! scattered through the far field, which is the ugliest available failure. The
//! chain is therefore built from the per-quad maximum
//! `Q[i, j] = max(S[i, j], S[i+1, j], S[i, j+1], S[i+1, j+1])` and reduced from
//! there: a plain two-by-two reduction of `Q` covers exactly the closed square,
//! by induction on `m`.
//!
//! Everything here is anchored to the raster rather than to a window. That is
//! what makes it incremental: a window that moves by a texel exposes a strip and
//! nothing else changes, so the same code that uploads heights uploads this. A
//! chain reduced *inside* a moving window cannot be incremental at all -- every
//! texel's window coordinate changes when the origin does -- and one reduced
//! toroidally lines up only at the first mip, because window origins are snapped
//! to two and never to four.
//!
//! Reductions are read from the source rather than folded texture-to-texture for
//! the same reason: filling mip `m` from mip `m - 1` needs the two windows to
//! start at exactly `2 * origin`, and the slack needed to tolerate the
//! off-by-one doubles at every mip down, reaching a whole window at the bottom.
//!
//! The source is a cache of square blocks, each reduced once when the ground it
//! covers first comes into view. A block is a tile, so building one is one file
//! opened and read in order rather than the scattered rows a strip would touch.

use std::collections::HashMap;

use glam::{IVec2, UVec2};

use crate::terrain::pyramid::RasterSource;

/// One block's reduction chain, from the finest depth held to a single texel.
struct Block {
    /// `chain[m - 1]` is depth `m`: `block >> m` entries square.
    ///
    /// Depth zero -- the per-quad maximum at full resolution -- is deliberately
    /// absent. It is three quarters of the whole chain and it is cheap to
    /// recompute, because a strip of it reads a strip of samples; every coarser
    /// depth reads a widening square instead, which is what makes them worth
    /// keeping.
    chain: Vec<Vec<f32>>,
    /// When this block was last read, for eviction.
    used: u64,
}

impl Block {
    fn bytes(&self) -> usize {
        self.chain
            .iter()
            .map(|level| level.len() * size_of::<f32>())
            .sum()
    }

    /// The maximum over the whole block, closed square included.
    fn top(&self) -> f32 {
        self.chain
            .last()
            .expect("a block always reduces to one texel")[0]
    }
}

/// The maxima over one raster, cached a block at a time.
pub struct Maxima {
    /// Side length of a block, in samples of whichever level it belongs to.
    block: u32,
    /// How many bytes of reduced blocks to keep.
    budget: usize,
    blocks: HashMap<(u32, i32, i32), Block>,
    bytes: usize,
    clock: u64,
    /// Reused between reads so a moving camera allocates nothing.
    scratch: Vec<f32>,
}

impl Maxima {
    /// `block` is the side length blocks are reduced in, and must be a power of
    /// two; `budget` is how many bytes of them to keep resident.
    ///
    /// The budget wants to be at least a window's worth at every level, or the
    /// blocks a single frame touches evict each other and every one of them is
    /// rebuilt on the next.
    pub fn new(block: u32, budget: usize) -> Self {
        assert!(
            block.is_power_of_two(),
            "block {block} is not a power of two"
        );
        Self {
            block,
            budget,
            blocks: HashMap::new(),
            bytes: 0,
            clock: 0,
            scratch: Vec::new(),
        }
    }

    /// How much of the budget is currently spent.
    #[cfg(test)]
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Fills `out` with the ceiling over each cell of a rectangle.
    ///
    /// `origin` and `size` are in cells of depth `mip`, which are `2^mip`
    /// samples of `level` across, measured from the raster's origin.
    pub fn read_rect(
        &mut self,
        heights: &dyn RasterSource,
        level: u32,
        mip: u32,
        origin: IVec2,
        size: UVec2,
        out: &mut [f32],
    ) {
        if size.x == 0 || size.y == 0 {
            return;
        }
        let depth = self.block.trailing_zeros();
        if mip == 0 {
            self.quad_max(heights, level, origin, size, out);
        } else if mip <= depth {
            self.inside_blocks(heights, level, mip, origin, size, out);
        } else {
            self.across_blocks(heights, level, mip - depth, origin, size, out);
        }
    }

    /// Depth zero: the maximum of the four samples around each quad.
    ///
    /// Read straight from the raster rather than from a block, because a strip
    /// of quads is a strip of samples one texel wider -- the same rows the
    /// height upload beside it is reading anyway.
    fn quad_max(
        &mut self,
        heights: &dyn RasterSource,
        level: u32,
        origin: IVec2,
        size: UVec2,
        out: &mut [f32],
    ) {
        let wide = size + UVec2::ONE;
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        scratch.resize((wide.x * wide.y) as usize, 0.0);
        heights.read_rect(
            level,
            origin,
            wide,
            bytemuck::cast_slice_mut(&mut scratch[..]),
        );

        for y in 0..size.y {
            for x in 0..size.x {
                let at = |dx: u32, dy: u32| scratch[((y + dy) * wide.x + x + dx) as usize];
                out[(y * size.x + x) as usize] = at(0, 0).max(at(1, 0)).max(at(0, 1)).max(at(1, 1));
            }
        }
        self.scratch = scratch;
    }

    /// Depths a block holds outright, copied out a block at a time.
    fn inside_blocks(
        &mut self,
        heights: &dyn RasterSource,
        level: u32,
        mip: u32,
        origin: IVec2,
        size: UVec2,
        out: &mut [f32],
    ) {
        // How many entries one block contributes at this depth, which is also
        // the side length of its chain entry.
        let per = (self.block >> mip) as i32;
        let stride = size.x as i32;

        let mut y = 0;
        while y < size.y as i32 {
            let (block_y, in_y) = (
                (origin.y + y).div_euclid(per),
                (origin.y + y).rem_euclid(per),
            );
            let rows = (per - in_y).min(size.y as i32 - y);
            let mut x = 0;
            while x < size.x as i32 {
                let (block_x, in_x) = (
                    (origin.x + x).div_euclid(per),
                    (origin.x + x).rem_euclid(per),
                );
                let columns = (per - in_x).min(stride - x);

                let entry = self.block(heights, level, block_x, block_y);
                let data = &entry.chain[(mip - 1) as usize];
                for row in 0..rows {
                    let from = ((in_y + row) * per + in_x) as usize;
                    let to = ((y + row) * stride + x) as usize;
                    out[to..to + columns as usize]
                        .copy_from_slice(&data[from..from + columns as usize]);
                }
                x += columns;
            }
            y += rows;
        }
    }

    /// Depths coarser than a block, folded from whole blocks.
    ///
    /// `over` is how many blocks a cell spans, as an exponent. The cells line up
    /// with blocks exactly -- a cell of `2^mip` samples starts at a multiple of
    /// `2^mip`, which is a multiple of the block -- so this is a maximum over
    /// each block's own top and nothing has to be reduced again.
    fn across_blocks(
        &mut self,
        heights: &dyn RasterSource,
        level: u32,
        over: u32,
        origin: IVec2,
        size: UVec2,
        out: &mut [f32],
    ) {
        let span = 1i32 << over;
        for y in 0..size.y as i32 {
            for x in 0..size.x as i32 {
                let (cell_x, cell_y) = (origin.x + x, origin.y + y);
                let mut highest = f32::NEG_INFINITY;
                for block_y in cell_y * span..(cell_y + 1) * span {
                    for block_x in cell_x * span..(cell_x + 1) * span {
                        highest = highest.max(self.block(heights, level, block_x, block_y).top());
                    }
                }
                out[(y * size.x as i32 + x) as usize] = highest;
            }
        }
    }

    /// One block's chain, reducing it first if this is its first sighting.
    fn block(&mut self, heights: &dyn RasterSource, level: u32, x: i32, y: i32) -> &Block {
        self.clock += 1;
        let key = (level, x, y);
        if !self.blocks.contains_key(&key) {
            let block = self.reduce(heights, level, x, y);
            self.bytes += block.bytes();
            self.blocks.insert(key, block);
            self.evict(&key);
        }
        let clock = self.clock;
        let block = self.blocks.get_mut(&key).expect("just inserted");
        block.used = clock;
        block
    }

    /// Reads a block's samples and folds them down to a single texel.
    fn reduce(&self, heights: &dyn RasterSource, level: u32, x: i32, y: i32) -> Block {
        // One sample wider on each far side, because a quad's maximum reaches
        // the sample beyond it. Without those the blocks would bound half-open
        // squares and rays would slip between them.
        let side = self.block;
        let wide = side + 1;
        let mut samples = vec![0f32; (wide * wide) as usize];
        heights.read_rect(
            level,
            IVec2::new(x * side as i32, y * side as i32),
            UVec2::splat(wide),
            bytemuck::cast_slice_mut(&mut samples),
        );

        let mut chain: Vec<Vec<f32>> = Vec::new();
        let mut coarse = vec![0f32; (side * side) as usize];
        for j in 0..side {
            for i in 0..side {
                let at = |di: u32, dj: u32| samples[((j + dj) * wide + i + di) as usize];
                coarse[(j * side + i) as usize] =
                    at(0, 0).max(at(1, 0)).max(at(0, 1)).max(at(1, 1));
            }
        }

        let mut fine = coarse;
        let mut span = side;
        while span > 1 {
            let half = span / 2;
            let mut coarse = vec![0f32; (half * half) as usize];
            for j in 0..half {
                for i in 0..half {
                    let at = |di: u32, dj: u32| fine[((2 * j + dj) * span + 2 * i + di) as usize];
                    coarse[(j * half + i) as usize] =
                        at(0, 0).max(at(1, 0)).max(at(0, 1)).max(at(1, 1));
                }
            }
            chain.push(coarse.clone());
            fine = coarse;
            span = half;
        }

        Block { chain, used: 0 }
    }

    /// Drops the least recently read blocks until the budget is met.
    ///
    /// Linear in the number of resident blocks, which is a few hundred at most
    /// and only scanned when the budget is actually exceeded. `keep` is the
    /// block that was just built, which must survive its own arrival.
    fn evict(&mut self, keep: &(u32, i32, i32)) {
        while self.bytes > self.budget {
            let Some((&victim, _)) = self
                .blocks
                .iter()
                .filter(|(key, _)| *key != keep)
                .min_by_key(|(_, block)| block.used)
            else {
                return;
            };
            let block = self.blocks.remove(&victim).expect("just found");
            self.bytes -= block.bytes();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::pyramid::{Level, Pyramid};

    const SIDE: u32 = 64;

    /// Ridged, so that a maximum is a real choice rather than whichever corner
    /// happened to be picked, and so that a bound that misses one sample is
    /// visibly wrong.
    fn terrain() -> Pyramid<f32> {
        let texels = (0..SIDE * SIDE)
            .map(|index| {
                let (x, y) = (index % SIDE, index / SIDE);
                let ridge = ((x * 7 + y * 13) % 17) as f32;
                ridge * ridge * 0.5 - f32::from((x % 11 == 0) as u8) * 40.0
            })
            .collect();
        Pyramid::build(Level::new(SIDE, SIDE, texels))
    }

    fn sample(source: &dyn RasterSource, level: u32, x: i32, y: i32) -> f32 {
        let mut one = [0f32; 1];
        source.read_rect(
            level,
            IVec2::new(x, y),
            UVec2::ONE,
            bytemuck::cast_slice_mut(&mut one),
        );
        one[0]
    }

    fn read(
        maxima: &mut Maxima,
        source: &dyn RasterSource,
        level: u32,
        mip: u32,
        side: u32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; (side * side) as usize];
        maxima.read_rect(
            source,
            level,
            mip,
            IVec2::ZERO,
            UVec2::splat(side),
            &mut out,
        );
        out
    }

    /// The property everything else rests on: no sample inside a cell's closed
    /// square is above the ceiling that cell reports.
    ///
    /// Closed rather than half open. A bound over `[i * 2^m, (i + 1) * 2^m)`
    /// says nothing about the surface between the last sample it covers and the
    /// first one it does not, so a ridge landing exactly on a cell boundary
    /// would be invisible to a ray crossing there.
    #[test]
    fn a_cell_bounds_every_sample_it_closes_over() {
        let source = terrain();
        let mut maxima = Maxima::new(16, 1 << 20);

        for level in 0..3u32 {
            for mip in 0..5u32 {
                let side = (SIDE >> level) >> mip;
                let cells = read(&mut maxima, &source, level, mip, side);
                let step = 1i32 << mip;
                for j in 0..side as i32 {
                    for i in 0..side as i32 {
                        let ceiling = cells[(j * side as i32 + i) as usize];
                        for y in j * step..=(j + 1) * step {
                            for x in i * step..=(i + 1) * step {
                                let height = sample(&source, level, x, y);
                                assert!(
                                    ceiling >= height,
                                    "level {level} mip {mip} cell ({i}, {j}) claims {ceiling} \
                                     but ({x}, {y}) is at {height}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// ... and it is the least such bound, so nothing is skipped that a tighter
    /// pyramid would have descended into.
    #[test]
    fn a_cell_reports_no_more_than_the_ground_under_it() {
        let source = terrain();
        let mut maxima = Maxima::new(16, 1 << 20);

        for mip in 0..5u32 {
            let side = SIDE >> mip;
            let cells = read(&mut maxima, &source, 0, mip, side);
            let step = 1i32 << mip;
            for j in 0..side as i32 {
                for i in 0..side as i32 {
                    let mut highest = f32::NEG_INFINITY;
                    for y in j * step..=(j + 1) * step {
                        for x in i * step..=(i + 1) * step {
                            highest = highest.max(sample(&source, 0, x, y));
                        }
                    }
                    assert_eq!(
                        cells[(j * side as i32 + i) as usize],
                        highest,
                        "mip {mip} cell ({i}, {j}) is looser than the ground under it"
                    );
                }
            }
        }
    }

    /// A cell that spans several blocks is folded from their tops, which is a
    /// different path through the code from one that lives inside a block.
    #[test]
    fn a_cell_wider_than_a_block_still_bounds_it() {
        let source = terrain();
        // Four blocks across the raster, so mips five and six span blocks.
        let mut maxima = Maxima::new(16, 1 << 20);
        for mip in 5..7u32 {
            let side = (SIDE >> mip).max(1);
            let cells = read(&mut maxima, &source, 0, mip, side);
            let step = 1i32 << mip;
            for j in 0..side as i32 {
                for i in 0..side as i32 {
                    let ceiling = cells[(j * side as i32 + i) as usize];
                    for y in j * step..=(j + 1) * step {
                        for x in i * step..=(i + 1) * step {
                            assert!(
                                ceiling >= sample(&source, 0, x, y),
                                "mip {mip} cell ({i}, {j}) does not cover ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Ground nothing is known about must not become a ceiling in its own right.
    ///
    /// A maximum ignores a hole beside real ground, and a cell with nothing but
    /// holes under it comes out at the sentinel -- far below any ray, so the
    /// cell is skipped rather than hit, which is what the leaf test expects.
    #[test]
    fn a_hole_never_lowers_the_ceiling_over_real_ground() {
        const NODATA: f32 = -32767.0;
        let mut texels = vec![NODATA; (SIDE * SIDE) as usize];
        // One real sample, in the middle of the second block along.
        texels[(20 * SIDE + 20) as usize] = 100.0;
        let source = Pyramid::build(Level::new(SIDE, SIDE, texels));
        let mut maxima = Maxima::new(16, 1 << 20);

        for mip in 0..6u32 {
            let side = (SIDE >> mip).max(1);
            let cells = read(&mut maxima, &source, 0, mip, side);
            let step = 1i32 << mip;
            for (index, &ceiling) in cells.iter().enumerate() {
                let (i, j) = (index as i32 % side as i32, index as i32 / side as i32);
                let covers = (i * step..=(i + 1) * step).contains(&20)
                    && (j * step..=(j + 1) * step).contains(&20);
                if covers {
                    assert_eq!(ceiling, 100.0, "mip {mip} cell ({i}, {j}) buried the peak");
                } else {
                    assert!(
                        ceiling < crate::terrain::NODATA_BELOW,
                        "mip {mip} cell ({i}, {j}) invented ground at {ceiling}"
                    );
                }
            }
        }
    }

    /// Counts how many times each level is read, to prove the cache works.
    struct Counted {
        inner: Pyramid<f32>,
        reads: std::cell::Cell<u32>,
    }

    impl RasterSource for Counted {
        fn level_count(&self) -> u32 {
            self.inner.level_count()
        }
        fn read_rect(&self, level: u32, origin: IVec2, size: UVec2, out: &mut [u8]) {
            self.reads.set(self.reads.get() + 1);
            self.inner.read_rect(level, origin, size, out);
        }
    }

    /// A window creeping across the raster must not re-reduce ground it has
    /// already seen, or every frame would pay for a tile it already holds.
    #[test]
    fn ground_already_seen_is_not_reduced_again() {
        let source = Counted {
            inner: terrain(),
            reads: std::cell::Cell::new(0),
        };
        let mut maxima = Maxima::new(16, 1 << 20);
        let mut out = vec![0f32; 8];

        // The first strip pays for the blocks it crosses.
        maxima.read_rect(&source, 0, 1, IVec2::ZERO, UVec2::new(1, 8), &mut out);
        let first = source.reads.get();
        assert!(first > 0, "the first read should have reduced something");

        // Creeping along inside the same blocks pays nothing more.
        for step in 1..8 {
            maxima.read_rect(
                &source,
                0,
                1,
                IVec2::new(step, 0),
                UVec2::new(1, 8),
                &mut out,
            );
        }
        assert_eq!(
            source.reads.get(),
            first,
            "a window moving inside blocks it already holds re-read the raster"
        );
    }

    /// The budget is a ceiling on residency, not a suggestion.
    #[test]
    fn blocks_are_dropped_to_stay_inside_the_budget() {
        let source = terrain();
        // Room for one block's chain and no more.
        let one = Maxima::new(16, usize::MAX);
        let mut maxima = Maxima::new(16, {
            let mut probe = one;
            let mut out = [0f32; 1];
            probe.read_rect(&source, 0, 1, IVec2::ZERO, UVec2::ONE, &mut out);
            probe.bytes()
        });

        let mut out = vec![0f32; 8 * 8];
        for level in 0..3u32 {
            maxima.read_rect(&source, level, 1, IVec2::ZERO, UVec2::splat(8), &mut out);
            assert!(
                maxima.bytes() <= maxima.budget,
                "level {level} left {} bytes resident against a budget of {}",
                maxima.bytes(),
                maxima.budget
            );
        }
    }
}
