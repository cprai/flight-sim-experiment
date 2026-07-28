//! Laying out the patches of geometry that make up each clipmap level.
//!
//! Every level is drawn as a square ring one block thick, leaving a hole in the
//! middle that the next finer level fills. Only the finest level drawn has no
//! hole to leave, so it gets a solid centre instead.
//!
//! Along one side a ring reads `block, block, seam, block, block`, where the
//! two-quad seam is what makes the whole side an odd number of quads across.
//! That oddness is deliberate: it leaves the hole exactly one quad larger than
//! the finer level's footprint, and that one-quad gap -- always present, always
//! on a side chosen by where the camera happens to be -- is filled by an
//! L-shaped trim. Without it a level would have to be re-tessellated every time
//! the camera crossed a texel.
//!
//! All coordinates here are integer quad offsets within a level's own grid, and
//! nothing in this module knows about the GPU.

use glam::{IVec2, UVec2};

use crate::terrain::clipmap::ClipmapConfig;

/// The kinds of patch a level is built from.
///
/// Each is a differently-shaped rectangle of quads, so each needs its own
/// indices. Instances are grouped by kind so a whole frame is one draw call per
/// variant.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum PatchKind {
    /// The square that rings are mostly made of.
    Block,
    /// Fills the two-quad seam in the middle of a horizontal run of blocks.
    SeamColumn,
    /// Fills the two-quad seam in the middle of a vertical run of blocks.
    SeamRow,
    /// The upright arm of the L that closes the gap around the finer level.
    TrimColumn,
    /// The flat arm of the same L.
    TrimRow,
    /// Fills the hole of the finest level drawn, which has nothing nested
    /// inside it.
    Centre,
}

impl PatchKind {
    pub const ALL: [PatchKind; 6] = [
        PatchKind::Block,
        PatchKind::SeamColumn,
        PatchKind::SeamRow,
        PatchKind::TrimColumn,
        PatchKind::TrimRow,
        PatchKind::Centre,
    ];

    /// This patch's size in quads.
    pub fn size_quads(self, config: &ClipmapConfig) -> UVec2 {
        let ring = config.ring_quads();
        let hole = config.hole_quads();
        match self {
            PatchKind::Block => UVec2::new(ring, ring),
            PatchKind::SeamColumn => UVec2::new(2, ring),
            PatchKind::SeamRow => UVec2::new(ring, 2),
            PatchKind::TrimColumn => UVec2::new(1, hole),
            // One quad shorter than the hole, because the upright arm already
            // covers the corner they would otherwise share.
            PatchKind::TrimRow => UVec2::new(hole - 1, 1),
            PatchKind::Centre => UVec2::new(hole, hole),
        }
    }
}

/// One rectangle of geometry to draw, positioned within a level's grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Patch {
    pub kind: PatchKind,
    /// Offset of the patch's first vertex within the level's grid.
    pub origin: UVec2,
    /// Which clipmap level, and so which texture array layer, it reads.
    pub level: u32,
}

/// Side length of the shared vertex grid, in vertices.
///
/// Sized for the largest patch. Every other kind indexes a sub-rectangle of the
/// same buffer, so there is only ever one vertex buffer.
pub fn shared_grid_verts(config: &ClipmapConfig) -> u32 {
    config.hole_quads() + 1
}

/// The shared vertex buffer: a grid of integer coordinates.
///
/// Vertices carry nothing but their position in the grid. Height comes from the
/// clipmap texture and the world transform from the level's uniform, so one
/// buffer serves every patch of every level.
pub fn grid_vertices(config: &ClipmapConfig) -> Vec<[u16; 2]> {
    let side = shared_grid_verts(config);
    let mut vertices = Vec::with_capacity((side * side) as usize);
    for y in 0..side {
        for x in 0..side {
            vertices.push([x as u16, y as u16]);
        }
    }
    vertices
}

/// Triangle indices for every patch kind, and where each kind's run starts.
///
/// Returned as one buffer so the whole frame binds a single index buffer and
/// varies only the range it draws from.
pub fn grid_indices(config: &ClipmapConfig) -> (Vec<u16>, Vec<std::ops::Range<u32>>) {
    let side = shared_grid_verts(config);
    let mut indices = Vec::new();
    let mut ranges = Vec::with_capacity(PatchKind::ALL.len());

    for kind in PatchKind::ALL {
        let start = indices.len() as u32;
        let size = kind.size_quads(config);
        for y in 0..size.y {
            for x in 0..size.x {
                let corner = (y * side + x) as u16;
                let (right, below) = (corner + 1, corner + side as u16);
                // Counter-clockwise seen from above, matching the ground plane
                // this replaced. Culling is off, so winding only documents intent.
                indices.extend_from_slice(&[corner, below, below + 1]);
                indices.extend_from_slice(&[corner, below + 1, right]);
            }
        }
        ranges.push(start..indices.len() as u32);
    }

    (indices, ranges)
}

/// Every patch to draw, given where each level's window has landed.
///
/// `origins` holds one grid origin per level, finest first, in each level's
/// own texel coordinates. The trim's orientation is derived from those origins
/// rather than recomputed from the camera, so the geometry cannot disagree with
/// the windows it is drawn against.
///
/// `base` is the finest level worth drawing; anything below it is left out
/// entirely. It is the base rather than level 0 that gets the solid centre,
/// because being the innermost level is what "no finer level to leave a hole
/// for" means -- level 0 is only ever the base when the camera is close enough
/// to the ground to deserve it.
///
/// The result is grouped by kind, ready to upload as an instance buffer.
pub fn patches(config: &ClipmapConfig, origins: &[IVec2], base: u32) -> Vec<Patch> {
    let ring = config.ring_quads();
    let hole = config.hole_quads();
    let grid = config.grid_quads();
    let far = grid - ring;

    let mut patches = Vec::new();
    for level in base..origins.len() as u32 {
        // Where the blocks along one side start, with the seam between the
        // middle pair.
        let runs = [0, ring, 2 * ring + 2, 3 * ring + 2];
        let seam = 2 * ring;

        // Top and bottom edges: a full run of blocks each, plus their seams.
        for y in [0, far] {
            for x in runs {
                patches.push(Patch {
                    kind: PatchKind::Block,
                    origin: UVec2::new(x, y),
                    level,
                });
            }
            patches.push(Patch {
                kind: PatchKind::SeamColumn,
                origin: UVec2::new(seam, y),
                level,
            });
        }

        // Left and right edges: only the two blocks between the corners the
        // horizontal edges already covered.
        for x in [0, far] {
            for y in [runs[1], runs[2]] {
                patches.push(Patch {
                    kind: PatchKind::Block,
                    origin: UVec2::new(x, y),
                    level,
                });
            }
            patches.push(Patch {
                kind: PatchKind::SeamRow,
                origin: UVec2::new(x, seam),
                level,
            });
        }

        match level.checked_sub(1).filter(|_| level > base) {
            // Nothing nested inside the finest level drawn, so fill its hole
            // outright.
            None => patches.push(Patch {
                kind: PatchKind::Centre,
                origin: UVec2::splat(ring),
                level,
            }),
            Some(finer) => {
                // The finer window starts either flush with the hole or one
                // quad into it; the trim goes on whichever side is left over.
                let offset = origins[finer as usize] / 2 - origins[level as usize];
                let flush = (offset - IVec2::splat(ring as i32)).cmpeq(IVec2::ZERO);

                let column_x = if flush.x { ring + hole - 1 } else { ring };
                let row_y = if flush.y { ring + hole - 1 } else { ring };
                // Start the flat arm clear of the upright one so no quad is
                // drawn twice.
                let row_x = if flush.x { ring } else { ring + 1 };

                patches.push(Patch {
                    kind: PatchKind::TrimColumn,
                    origin: UVec2::new(column_x, ring),
                    level,
                });
                patches.push(Patch {
                    kind: PatchKind::TrimRow,
                    origin: UVec2::new(row_x, row_y),
                    level,
                });
            }
        }
    }

    patches.sort_by_key(|patch| patch.kind);
    patches
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::terrain::clipmap::{grid_origin, window_origin};
    use glam::DVec2;

    fn config() -> ClipmapConfig {
        ClipmapConfig {
            block_verts: 8,
            window_texels: 32,
            ..Default::default()
        }
    }

    /// Grid origins for every level, as the renderer would compute them.
    fn origins(config: &ClipmapConfig, camera: DVec2, levels: u32) -> Vec<IVec2> {
        (0..levels)
            .map(|level| grid_origin(config, window_origin(config, level, camera)))
            .collect()
    }

    /// Every quad a set of patches covers at one level.
    fn covered(config: &ClipmapConfig, patches: &[Patch], level: u32) -> HashSet<(u32, u32)> {
        let mut quads = HashSet::new();
        for patch in patches.iter().filter(|p| p.level == level) {
            let size = patch.kind.size_quads(config);
            for y in 0..size.y {
                for x in 0..size.x {
                    let quad = (patch.origin.x + x, patch.origin.y + y);
                    assert!(
                        quads.insert(quad),
                        "{quad:?} covered twice at level {level} by {:?}",
                        patch.kind
                    );
                }
            }
        }
        quads
    }

    /// Where the next finer level's footprint lands on this level's grid.
    fn finer_footprint(
        config: &ClipmapConfig,
        origins: &[IVec2],
        level: u32,
    ) -> HashSet<(u32, u32)> {
        let coarse = origins[level as usize];
        let fine = origins[level as usize - 1];
        let start = fine / 2 - coarse;
        let span = config.grid_quads() / 2;

        let mut quads = HashSet::new();
        for y in 0..span {
            for x in 0..span {
                quads.insert((start.x as u32 + x, start.y as u32 + y));
            }
        }
        quads
    }

    fn whole_grid(config: &ClipmapConfig) -> HashSet<(u32, u32)> {
        let grid = config.grid_quads();
        (0..grid)
            .flat_map(|y| (0..grid).map(move |x| (x, y)))
            .collect()
    }

    #[test]
    fn a_ring_and_the_level_inside_it_tile_the_grid_exactly() {
        let config = config();
        let levels = 4;

        // Quarter-texel steps sweep the camera through every combination of
        // the two parity bits that choose the trim's orientation.
        let mut seen_orientations = HashSet::new();
        for step in 0..32 {
            let camera = DVec2::new(f64::from(step) * 0.25, f64::from(step) * 0.5);
            let origins = origins(&config, camera, levels);
            let patches = patches(&config, &origins, 0);

            for level in 1..levels {
                let ring = covered(&config, &patches, level);
                let finer = finer_footprint(&config, &origins, level);

                assert!(
                    ring.is_disjoint(&finer),
                    "level {level} overlaps the level inside it at {camera}"
                );
                let union: HashSet<_> = ring.union(&finer).copied().collect();
                assert_eq!(
                    union,
                    whole_grid(&config),
                    "level {level} leaves a hole at {camera}"
                );
            }

            let trim = patches
                .iter()
                .find(|p| p.level == 1 && p.kind == PatchKind::TrimColumn)
                .expect("every ring level has a trim");
            seen_orientations.insert((
                trim.origin.x,
                patches
                    .iter()
                    .find(|p| p.level == 1 && p.kind == PatchKind::TrimRow)
                    .expect("every ring level has a trim")
                    .origin
                    .y,
            ));
        }

        assert_eq!(
            seen_orientations.len(),
            4,
            "the sweep should have exercised all four trim orientations"
        );
    }

    #[test]
    fn the_finest_level_is_solid() {
        let config = config();
        let origins = origins(&config, DVec2::new(3.0, 7.0), 3);
        let patches = patches(&config, &origins, 0);

        assert_eq!(covered(&config, &patches, 0), whole_grid(&config));
    }

    #[test]
    fn dropping_the_finest_levels_leaves_the_rest_tiling_the_grid_as_before() {
        // What a camera high above the ground asks for: the levels below the
        // base gone entirely, the base solid in their place, and every level
        // outside it laid out exactly as it would have been.
        let config = config();
        let levels = 5;
        let base = 2;

        for step in 0..32 {
            let camera = DVec2::new(f64::from(step) * 0.25, f64::from(step) * 0.5);
            let origins = origins(&config, camera, levels);
            let patches = patches(&config, &origins, base);

            assert!(
                patches.iter().all(|p| p.level >= base),
                "a level below the base was still drawn at {camera}"
            );
            assert_eq!(
                covered(&config, &patches, base),
                whole_grid(&config),
                "the base level left a hole for a level that is not drawn"
            );

            for level in base + 1..levels {
                let ring = covered(&config, &patches, level);
                let finer = finer_footprint(&config, &origins, level);
                assert!(
                    ring.is_disjoint(&finer),
                    "level {level} overlaps at {camera}"
                );
                let union: HashSet<_> = ring.union(&finer).copied().collect();
                assert_eq!(
                    union,
                    whole_grid(&config),
                    "level {level} leaks at {camera}"
                );
            }
        }
    }

    #[test]
    fn only_the_finest_level_has_a_centre_and_only_the_rest_have_trims() {
        let config = config();
        let origins = origins(&config, DVec2::new(1.0, 1.0), 4);
        let patches = patches(&config, &origins, 0);

        for level in 0..4u32 {
            let kinds: HashSet<_> = patches
                .iter()
                .filter(|p| p.level == level)
                .map(|p| p.kind)
                .collect();
            assert_eq!(kinds.contains(&PatchKind::Centre), level == 0);
            assert_eq!(kinds.contains(&PatchKind::TrimColumn), level > 0);
            assert_eq!(kinds.contains(&PatchKind::TrimRow), level > 0);
        }
    }

    #[test]
    fn every_ring_is_built_from_twelve_blocks_and_four_seams() {
        let config = config();
        let origins = origins(&config, DVec2::ZERO, 3);
        let patches = patches(&config, &origins, 0);

        let count = |level: u32, kind: PatchKind| {
            patches
                .iter()
                .filter(|p| p.level == level && p.kind == kind)
                .count()
        };
        for level in 0..3 {
            assert_eq!(count(level, PatchKind::Block), 12);
            assert_eq!(count(level, PatchKind::SeamColumn), 2);
            assert_eq!(count(level, PatchKind::SeamRow), 2);
        }
    }

    #[test]
    fn patches_are_grouped_so_each_kind_is_one_draw_call() {
        let config = config();
        let origins = origins(&config, DVec2::new(5.0, 9.0), 5);
        let patches = patches(&config, &origins, 0);

        let mut runs: Vec<PatchKind> = Vec::new();
        for patch in &patches {
            if runs.last() != Some(&patch.kind) {
                assert!(
                    !runs.contains(&patch.kind),
                    "{:?} appears in more than one run",
                    patch.kind
                );
                runs.push(patch.kind);
            }
        }
    }

    #[test]
    fn no_patch_reaches_outside_the_shared_vertex_grid() {
        let config = config();
        let side = shared_grid_verts(&config);
        let origins = origins(&config, DVec2::new(2.0, 3.0), 4);

        for patch in patches(&config, &origins, 0) {
            let size = patch.kind.size_quads(&config);
            assert!(size.x < side && size.y < side, "{patch:?} is too large");
            assert!(
                patch.origin.x + size.x <= config.grid_quads()
                    && patch.origin.y + size.y <= config.grid_quads(),
                "{patch:?} runs off the level's grid"
            );
        }
    }

    #[test]
    fn indices_stay_inside_the_vertex_buffer_and_describe_whole_triangles() {
        let config = ClipmapConfig::default();
        let vertices = grid_vertices(&config);
        let (indices, ranges) = grid_indices(&config);

        assert!(
            vertices.len() <= usize::from(u16::MAX) + 1,
            "the grid must stay addressable by 16-bit indices"
        );
        assert_eq!(indices.len() % 3, 0);
        assert!(indices.iter().all(|&i| usize::from(i) < vertices.len()));

        for (kind, range) in PatchKind::ALL.iter().zip(&ranges) {
            let quads = kind.size_quads(&config);
            assert_eq!(
                range.len(),
                (quads.x * quads.y * 6) as usize,
                "{kind:?} has the wrong index count"
            );
        }
    }

    #[test]
    fn the_shared_grid_is_addressable_by_16_bit_indices_at_the_default_size() {
        // The default block size drives the whole triangle budget; if it ever
        // grows past what u16 indices can reach this test says so first.
        let config = ClipmapConfig::default();
        let side = shared_grid_verts(&config) as usize;
        assert!(
            side * side <= usize::from(u16::MAX) + 1,
            "grid is {side}x{side}"
        );
    }
}
