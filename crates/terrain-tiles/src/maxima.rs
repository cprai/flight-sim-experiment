//! The rules that build the max pyramid the far field is marched through.
//!
//! Depth `d` of the pyramid answers
//!
//! ```text
//! M[d][i, j] = the greatest, over every level l <= d, of
//!              max of level l's height samples over the closed square
//!              [i * 2^(d-l), (i + 1) * 2^(d-l)] in level l's own texels
//! ```
//!
//! A ray that stays above `M[d][i, j]` across the ground that cell is named
//! after cannot meet any surface the renderer might draw there, so it skips it
//! in one step.
//!
//! **One pyramid serves every clipmap level.** Level `l`'s depth-`m` cell is
//! depth `l + m` of this chain, at the same index: the cell covers
//! `[i * 2^m, (i + 1) * 2^m]` in level `l`'s own texels, which is
//! `[i * 2^(l+m), (i + 1) * 2^(l+m)]` in level-0 texels, and the `l`th term
//! above is exactly the bound that level wants. There is nothing per level to
//! build, and no cell of it is ever rebuilt as the camera moves.
//!
//! **Why a maximum over levels rather than over the raster.** A cell has to
//! bound the surface the march actually intersects, which at the finest depth is
//! the bilinear patch through four of *level `l`'s* height samples. Those are
//! means over `2^l` raster texels each, so the patch at a cell's far corner is
//! fed by ground up to `2^l` texels past it -- and a bound taken over the cell's
//! own raster texels leaves that ground out. A ridge sitting in it would be
//! invisible to a ray crossing there: holes scattered through the far field,
//! which is the ugliest available failure.
//!
//! Widening every cell by a whole cell would cover it, and covers far more
//! besides: at coarse depths that is a bound over four times the ground, and the
//! march pays for it in descents. Taking each level's own samples instead is the
//! tightest a single chain can be. It is exact at `l = 0`, where a texel is a
//! sample rather than a mean and no reach past the square exists at all, and it
//! never asks a coarse level to answer for detail its own surface has already
//! smoothed away.
//!
//! The chain is built from two operations, one of which reads the mip chain
//! beside it:
//!
//! ```text
//! M[0] = quad_max(level-0 samples)
//! M[d] = max(reduce_max(M[d - 1]), quad_max(level-d samples))
//! ```
//!
//! The first argument carries every term for `l < d` forward -- two adjacent
//! closed squares share their boundary, so their union is the whole closed
//! square one depth up -- and the second adds the term for `l = d`, which no
//! reduction of what came before could have known about.
//!
//! Holes need no special case. Nodata is written far below any ground -- see
//! [`crate::NODATA_BELOW`] -- so a maximum ignores a hole beside real ground,
//! and a cell with nothing but holes under it stays at the sentinel, which reads
//! as below every ray and is skipped rather than hit.

/// The greatest of a run of cells, which is never empty.
///
/// The fold that continues the chain past the depths that were written down,
/// and the one that collapses a window's coarsest depth to the single figure a
/// ray clears a whole level with.
pub fn highest(cells: &[f32]) -> f32 {
    cells.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Fills `out` with the maximum of the four values around each cell.
///
/// `values` is `(width + 1) * (height + 1)` in row-major order and `out` is
/// `width * height`: the extra row and column are the reach past the last cell.
/// Applied to samples this closes each cell's square; applied to closed cells it
/// dilates them to two cells wide.
///
/// # Panics
///
/// If either slice is not the length its dimensions call for.
pub fn quad_max(values: &[f32], width: u32, height: u32, out: &mut [f32]) {
    let wide = (width + 1) as usize;
    assert_eq!(
        values.len(),
        wide * (height + 1) as usize,
        "a {width} x {height} quad max reads one value past each edge"
    );
    assert_eq!(out.len(), (width as usize) * (height as usize));

    for y in 0..height as usize {
        for x in 0..width as usize {
            let at = |dx: usize, dy: usize| values[(y + dy) * wide + x + dx];
            out[y * width as usize + x] = at(0, 0).max(at(1, 0)).max(at(0, 1)).max(at(1, 1));
        }
    }
}

/// Fills `out` with the maximum of each 2x2 block of `values`.
///
/// `values` is `width * height`, both even, and `out` is half of each. Carrying
/// a depth up to the next one: the two closed squares under a coarse cell share
/// their boundary, so their union is the coarse cell's own closed square and
/// nothing has to be widened.
///
/// # Panics
///
/// If either slice is not the length its dimensions call for, or if a dimension
/// is odd.
pub fn reduce_max(values: &[f32], width: u32, height: u32, out: &mut [f32]) {
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "{width} x {height} does not halve evenly"
    );
    assert_eq!(values.len(), (width as usize) * (height as usize));
    let (half_width, half_height) = ((width / 2) as usize, (height / 2) as usize);
    assert_eq!(out.len(), half_width * half_height);

    for y in 0..half_height {
        for x in 0..half_width {
            let at = |dx: usize, dy: usize| values[(2 * y + dy) * width as usize + 2 * x + dx];
            out[y * half_width + x] = at(0, 0).max(at(1, 0)).max(at(0, 1)).max(at(1, 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NODATA_BELOW;

    const NODATA: f32 = -32767.0;

    /// Side of the sample raster the tests reduce.
    ///
    /// A power of two, so every level of the mip chain beside the pyramid
    /// halves exactly and no test has to reason about a ragged edge -- which is
    /// the tool's business, not these rules'.
    const SIDE: u32 = 32;

    /// Ridged, so that a maximum is a real choice rather than whichever corner
    /// happened to be picked, and so that a bound missing one value shows.
    fn terrain() -> Vec<f32> {
        (0..SIDE * SIDE)
            .map(|index| {
                let (x, y) = (index % SIDE, index / SIDE);
                let ridge = ((x * 7 + y * 13) % 17) as f32;
                ridge * ridge * 0.5 - f32::from(x.is_multiple_of(11)) * 40.0
            })
            .collect()
    }

    /// The mip chain `terrain-download` writes beside the raster: each level the
    /// mean of the four texels under it, which is what the renderer draws and so
    /// what the pyramid has to bound.
    ///
    /// Holes are dropped from the mean rather than averaged in, as
    /// [`crate::Texel::box_filter`] requires -- one sentinel among three real
    /// metres would otherwise come out far below any ground and nowhere near the
    /// sentinel, and nothing downstream would recognise it as a hole.
    fn mips(samples: &[f32]) -> Vec<(u32, Vec<f32>)> {
        let mut chain = vec![(SIDE, samples.to_vec())];
        while chain.last().expect("never empty").0 > 1 {
            let (span, fine) = chain.last().expect("never empty");
            let (span, half) = (*span, span / 2);
            let mut coarse = vec![0.0; (half * half) as usize];
            for y in 0..half as usize {
                for x in 0..half as usize {
                    let at = |dx: usize, dy: usize| fine[(2 * y + dy) * span as usize + 2 * x + dx];
                    let real: Vec<f32> = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)]
                        .into_iter()
                        .filter(|value| *value > NODATA_BELOW)
                        .collect();
                    coarse[y * half as usize + x] = if real.is_empty() {
                        NODATA
                    } else {
                        real.iter().sum::<f32>() / real.len() as f32
                    };
                }
            }
            chain.push((half, coarse));
        }
        chain
    }

    /// Builds the whole pyramid, as the tool does: the finest depth from the
    /// raster, each one after it from the depth below and that level's own mip.
    ///
    /// The reach past a closed square is what makes each depth one cell narrower
    /// than the mip it takes in. The tool reads that extra cell out of the next
    /// tile along; here the chain simply gives up a cell a depth, which is why
    /// the deepest one checked is well short of a single cell.
    fn chain(mips: &[(u32, Vec<f32>)]) -> Vec<(u32, Vec<f32>)> {
        let mut span = SIDE - 1;
        let mut depth = vec![0.0; (span * span) as usize];
        quad_max(&mips[0].1, span, span, &mut depth);

        let mut chain = vec![(span, depth)];
        for (level, (mip_span, mip)) in mips.iter().enumerate().skip(1) {
            let (last_span, last) = chain.last().expect("never empty");
            // Halving a depth wants an even side, and a depth is odd as often
            // as not, so the last column and row are dropped rather than half
            // covered. The tool loses them to the next tile's overhang instead.
            let carried_span = last_span / 2;
            let mut carried = vec![0.0; (carried_span * carried_span) as usize];
            reduce_max(
                &trim(last, *last_span, carried_span * 2),
                carried_span * 2,
                carried_span * 2,
                &mut carried,
            );

            // ... and this level's own samples, which nothing below could know.
            let own_span = mip_span - 1;
            let mut own = vec![0.0; (own_span * own_span) as usize];
            quad_max(mip, own_span, own_span, &mut own);

            span = carried_span.min(own_span);
            let mut depth = vec![0.0; (span * span) as usize];
            for y in 0..span as usize {
                for x in 0..span as usize {
                    depth[y * span as usize + x] =
                        carried[y * carried_span as usize + x].max(own[y * own_span as usize + x]);
                }
            }
            assert!(level > 0);
            chain.push((span, depth));
            if span == 1 {
                break;
            }
        }
        chain
    }

    /// The top-left `keep` square of a `span` square grid.
    fn trim(values: &[f32], span: u32, keep: u32) -> Vec<f32> {
        let mut out = Vec::with_capacity((keep * keep) as usize);
        for y in 0..keep as usize {
            out.extend_from_slice(&values[y * span as usize..y * span as usize + keep as usize]);
        }
        out
    }

    /// The definition, read straight off the mip chain: the greatest, over every
    /// level at or above this depth, of that level's samples across the closed
    /// square the cell is named after.
    fn defined(mips: &[(u32, Vec<f32>)], depth: u32, i: u32, j: u32) -> f32 {
        let mut highest = f32::NEG_INFINITY;
        for (level, (span, values)) in mips.iter().enumerate().take(depth as usize + 1) {
            let step = 1u32 << (depth - level as u32);
            for y in j * step..=(j + 1) * step {
                for x in i * step..=(i + 1) * step {
                    if x < *span && y < *span {
                        highest = highest.max(values[(y * span + x) as usize]);
                    }
                }
            }
        }
        highest
    }

    /// The chain the tool builds is the definition, at every depth.
    ///
    /// Checked against the mip chain rather than against another reduction, so a
    /// mistake in the recurrence -- carrying without adding this level's own
    /// samples, or adding them a depth late -- has nothing to hide behind.
    #[test]
    fn every_cell_is_the_greatest_bound_the_levels_under_it_ask_for() {
        let samples = terrain();
        let mips = mips(&samples);
        for (depth, (span, cells)) in chain(&mips).iter().enumerate() {
            for j in 0..*span {
                for i in 0..*span {
                    assert_eq!(
                        cells[(j * span + i) as usize],
                        defined(&mips, depth as u32, i, j),
                        "depth {depth} cell ({i}, {j})"
                    );
                }
            }
        }
    }

    /// The property the march depends on: a cell is at or above every height
    /// sample of every clipmap level that reads it, across the closed square it
    /// covers.
    ///
    /// The far corner of that square is the case worth having a test for. A
    /// level-`l` sample is a mean over `2^l` raster texels, so it answers for
    /// ground beyond the cell, and a pyramid reduced from the raster alone comes
    /// out below it -- which is a ray passing through solid ground.
    #[test]
    fn a_cell_bounds_the_samples_of_every_level_that_reads_it() {
        let samples = terrain();
        let mips = mips(&samples);
        let chain = chain(&mips);
        let mut checked = 0;

        // Every way a depth splits into a clipmap level and a quadtree depth.
        for (depth, (span, cells)) in chain.iter().enumerate() {
            for level in 0..=depth {
                let mip = depth - level;
                let (mip_span, values) = &mips[level];
                for j in 0..*span {
                    for i in 0..*span {
                        let ceiling = cells[(j * span + i) as usize];
                        for t_y in j << mip..=(j + 1) << mip {
                            for t_x in i << mip..=(i + 1) << mip {
                                if t_x >= *mip_span || t_y >= *mip_span {
                                    continue;
                                }
                                let height = values[(t_y * mip_span + t_x) as usize];
                                assert!(
                                    ceiling >= height,
                                    "depth {depth} as level {level} mip {mip}: cell ({i}, {j}) \
                                     claims {ceiling} but texel ({t_x}, {t_y}) is at {height}"
                                );
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert!(checked > 10_000, "only {checked} texels were checked");
    }

    /// Ground nothing is known about must not become a ceiling in its own right,
    /// and must not bury the real ground beside it either.
    #[test]
    fn a_hole_never_lowers_the_ceiling_over_real_ground() {
        let mut samples = vec![NODATA; (SIDE * SIDE) as usize];
        // One real sample, off any round number so it lands mid-cell at every
        // depth rather than on a boundary.
        samples[(13 * SIDE + 13) as usize] = 100.0;
        let mips = mips(&samples);

        for (depth, (span, cells)) in chain(&mips).iter().enumerate() {
            // Every cell either sees the one real sample, at its own height, or
            // sees nothing and stays at the sentinel. Anything between the two
            // is a hole averaged into ground: far below any terrain, so it draws
            // as a pit, and nowhere near the sentinel, so nothing downstream
            // recognises it as unmeasured.
            let mut seeing = 0;
            for (index, &ceiling) in cells.iter().enumerate() {
                let (i, j) = (index as u32 % span, index as u32 / span);
                assert!(
                    ceiling == 100.0 || ceiling < NODATA_BELOW,
                    "depth {depth} cell ({i}, {j}) invented ground at {ceiling} out of a hole"
                );
                seeing += u32::from(ceiling == 100.0);
            }
            assert!(
                seeing > 0,
                "depth {depth} lost the peak entirely across {} cells",
                span * span
            );
        }
    }
}
