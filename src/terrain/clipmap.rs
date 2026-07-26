//! Where each clipmap level's window sits, and which texels to refresh when it
//! moves.
//!
//! A clipmap keeps a fixed-size window into every mip level, all centred on the
//! camera. Fine levels cover a small area at full detail, coarse levels cover
//! progressively more at progressively less, and together they tile the ground
//! out to the horizon with a texel count that does not depend on how large the
//! dataset is.
//!
//! Windows are addressed toroidally: a window's texels wrap around a
//! fixed-size texture, so moving one texel east costs one column of uploads
//! rather than a full recopy. That makes the arithmetic here -- which column,
//! wrapped where -- the part most worth testing, and it is all integer maths
//! with no GPU types in sight.

use glam::{DVec2, IVec2, UVec2};

/// How large a clipmap's rings are, and how gently they blend into each other.
///
/// This is runtime configuration rather than a constant so that tests can run a
/// small clipmap on a software rasterizer while the application runs a large
/// one.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ClipmapConfig {
    /// Side length of one square block of geometry, in vertices.
    ///
    /// The single knob that scales the triangle count, which grows with its
    /// square. Everything else about a level's size is derived from it.
    pub block_verts: u32,
    /// Fraction of a ring's outer edge over which it blends into the next level.
    pub morph_band: f32,
}

impl Default for ClipmapConfig {
    fn default() -> Self {
        Self {
            block_verts: 64,
            morph_band: 0.25,
        }
    }
}

impl ClipmapConfig {
    /// Radial thickness of one ring, in quads.
    pub const fn ring_quads(&self) -> u32 {
        self.block_verts - 1
    }

    /// Side length of a level's grid, in vertices.
    ///
    /// Four blocks and a two-quad seam wide, which is what makes a ring exactly
    /// one block thick with a hole the next finer level can fill.
    pub const fn grid_verts(&self) -> u32 {
        4 * self.block_verts - 1
    }

    /// Side length of a level's grid, in quads.
    pub const fn grid_quads(&self) -> u32 {
        self.grid_verts() - 1
    }

    /// Side length of a level's clip texture, in texels.
    ///
    /// One more than the grid needs, so that a central difference at the last
    /// vertex still has a neighbour to read. A power of two, so wrapping around
    /// the torus is a bit mask.
    pub const fn window_texels(&self) -> u32 {
        self.grid_verts() + 1
    }

    /// Side length of the hole a ring leaves for the next finer level, in quads.
    pub const fn hole_quads(&self) -> u32 {
        self.grid_quads() - 2 * self.ring_quads()
    }

    /// How many levels are needed to cover a raster of this size.
    ///
    /// The coarsest level has to span the whole raster, otherwise there would be
    /// ground beyond the outermost ring with nothing to draw it. Capped at
    /// `available`, the number of mip levels that actually exist.
    pub fn level_count(&self, raster: UVec2, available: u32) -> u32 {
        let span = f64::from(self.grid_quads());
        let needed = f64::from(raster.max_element()) / span;
        // Level `l` covers `span * 2^l`, so this is the first `l` that reaches
        // across the raster, plus one because levels are counted from zero.
        let levels = needed.log2().ceil().max(0.0) as u32 + 1;
        levels.clamp(1, available.max(1))
    }
}

/// A rectangle of texels, in some level's own texel coordinates.
///
/// Coordinates are signed because a window near the edge of the raster legally
/// hangs off it; reads there clamp to the border rather than failing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub fn is_empty(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    pub fn origin(&self) -> IVec2 {
        IVec2::new(self.x, self.y)
    }

    pub fn size(&self) -> UVec2 {
        UVec2::new(self.width, self.height)
    }
}

/// Where a level's window should sit for a given camera position.
///
/// `camera_texels` is the camera's ground position in the *base* raster's
/// texels; each level divides it down to its own resolution.
///
/// The origin is snapped to an even texel of the level's own lattice. That is
/// what makes consecutive levels nest: halving an even number is exact, so a
/// fine level's vertices always land on coarse-level sample positions and the
/// blend between them has no seam to hide.
pub fn window_origin(config: &ClipmapConfig, level: u32, camera_texels: DVec2) -> IVec2 {
    let texels = camera_texels / f64::from(1u32 << level);
    let step_back = 2 * config.ring_quads() as i32;
    IVec2::new(
        snap_axis(texels.x, step_back),
        snap_axis(texels.y, step_back),
    )
}

fn snap_axis(camera: f64, step_back: i32) -> i32 {
    // Snapping to two rather than one keeps the parity that nesting relies on;
    // stepping back by the ring thickness leaves the camera near the middle of
    // the window, so it has equal room to move in any direction.
    let snapped = 2 * (camera * 0.5).floor() as i32;
    snapped - step_back
}

/// The parts of a window that hold new ground after its origin moves.
///
/// Returns at most two disjoint rectangles whose union is exactly the new
/// window minus the old one: a slab covering the sideways motion, and a slab
/// covering the vertical motion with the first slab's columns already removed.
/// A move of a whole window or more shares nothing with where it was, so the
/// whole window comes back.
pub fn exposed_regions(old: IVec2, new: IVec2, span: u32) -> Vec<Rect> {
    let whole = Rect {
        x: new.x,
        y: new.y,
        width: span,
        height: span,
    };
    let delta = new - old;
    if delta == IVec2::ZERO {
        return Vec::new();
    }
    if delta.x.unsigned_abs() >= span || delta.y.unsigned_abs() >= span {
        return vec![whole];
    }

    let mut regions = Vec::with_capacity(2);

    // The columns uncovered by moving sideways, full height.
    let (kept_x, kept_width) = match delta.x {
        0 => (new.x, span),
        d if d > 0 => {
            regions.push(Rect {
                x: old.x + span as i32,
                y: new.y,
                width: d.unsigned_abs(),
                height: span,
            });
            (new.x, span - d.unsigned_abs())
        }
        d => {
            regions.push(Rect {
                x: new.x,
                y: new.y,
                width: d.unsigned_abs(),
                height: span,
            });
            (old.x, span - d.unsigned_abs())
        }
    };

    // The rows uncovered by moving up or down, narrowed to the columns the
    // first slab did not already carry.
    if delta.y != 0 && kept_width > 0 {
        let y = if delta.y > 0 {
            old.y + span as i32
        } else {
            new.y
        };
        regions.push(Rect {
            x: kept_x,
            y,
            width: kept_width,
            height: delta.y.unsigned_abs(),
        });
    }

    regions.retain(|r| !r.is_empty());
    regions
}

/// Splits a rectangle into pieces that do not straddle the torus seam.
///
/// Texel `t` of a window lives at `t mod span` in the texture, so a rectangle
/// crossing that boundary has to be uploaded as separate pieces. Each result
/// pairs a piece of the original rectangle with the texel it starts at in the
/// texture. A window-sized rectangle yields at most four pieces.
pub fn split_across_seam(rect: Rect, span: u32) -> Vec<(Rect, UVec2)> {
    if rect.is_empty() {
        return Vec::new();
    }

    let columns = split_axis(rect.x, rect.width, span);
    let rows = split_axis(rect.y, rect.height, span);

    let mut pieces = Vec::with_capacity(columns.len() * rows.len());
    for &(y, height, destination_y) in &rows {
        for &(x, width, destination_x) in &columns {
            pieces.push((
                Rect {
                    x,
                    y,
                    width,
                    height,
                },
                UVec2::new(destination_x, destination_y),
            ));
        }
    }
    pieces
}

/// Cuts a run of texels wherever it would wrap past the end of the texture.
///
/// Yields `(start, length, destination)` triples in the original coordinates.
fn split_axis(start: i32, length: u32, span: u32) -> Vec<(i32, u32, u32)> {
    let mut pieces = Vec::with_capacity(2);
    let mut position = start;
    let mut remaining = length;
    while remaining > 0 {
        let destination = position.rem_euclid(span as i32) as u32;
        let run = (span - destination).min(remaining);
        pieces.push((position, run, destination));
        position += run as i32;
        remaining -= run;
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Small enough to enumerate exhaustively in tests.
    fn config() -> ClipmapConfig {
        ClipmapConfig {
            block_verts: 8,
            ..Default::default()
        }
    }

    /// A deterministic stand-in for random sampling, so failures reproduce.
    fn pseudo_random(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    fn camera_positions(count: usize) -> Vec<DVec2> {
        let mut seed = 0x5eed_1234_9abc_def1;
        (0..count)
            .map(|_| {
                let x = (pseudo_random(&mut seed) % 2_000_000) as f64 / 1000.0 - 1000.0;
                let y = (pseudo_random(&mut seed) % 2_000_000) as f64 / 1000.0 - 1000.0;
                DVec2::new(x, y)
            })
            .collect()
    }

    #[test]
    fn a_ring_leaves_a_hole_the_next_level_can_fill() {
        let config = config();
        assert_eq!(config.hole_quads(), 2 * config.block_verts);

        // The finer level's whole footprint, measured on the coarse grid, is
        // exactly one quad short of the hole. That missing quad is the same
        // width whatever the block size, and is what the L-shaped trim fills.
        let footprint = config.grid_quads() / 2;
        assert_eq!(footprint, config.hole_quads() - 1);
    }

    #[test]
    fn a_window_is_one_texel_wider_than_its_grid() {
        let config = config();
        assert_eq!(config.window_texels(), config.grid_verts() + 1);
        // A power of two, so wrapping is a mask rather than a division.
        assert!(config.window_texels().is_power_of_two());
    }

    #[test]
    fn window_origins_land_on_even_texels() {
        let config = config();
        for camera in camera_positions(200) {
            for level in 0..6 {
                let origin = window_origin(&config, level, camera);
                assert_eq!(origin.x % 2, 0, "level {level} at {camera}: {origin}");
                assert_eq!(origin.y % 2, 0, "level {level} at {camera}: {origin}");
            }
        }
    }

    #[test]
    fn each_level_nests_inside_the_next_with_at_most_one_quad_of_slack() {
        let config = config();
        let ring = config.ring_quads() as i32;

        for camera in camera_positions(500) {
            for level in 1..6 {
                let coarse = window_origin(&config, level, camera);
                let fine = window_origin(&config, level - 1, camera);

                for axis in 0..2 {
                    // Where the fine window starts, measured on the coarse grid.
                    let offset = fine[axis] / 2 - coarse[axis];
                    assert!(
                        offset == ring || offset == ring + 1,
                        "level {level} axis {axis} at {camera}: offset {offset}, \
                         expected {ring} or {}",
                        ring + 1
                    );

                    // Which of the two it is depends only on the parity of the
                    // camera's position on this level's lattice, which is what
                    // selects the trim's orientation.
                    let texels = camera[axis] / f64::from(1u32 << level);
                    let parity = (texels.floor() as i32).rem_euclid(2);
                    assert_eq!(offset - ring, parity, "parity should pick the offset");
                }
            }
        }
    }

    #[test]
    fn the_finer_footprint_stays_inside_the_hole() {
        let config = config();
        let ring = config.ring_quads() as i32;
        let hole = config.hole_quads() as i32;
        // A fine level covers half its quads on the coarse grid.
        let footprint = config.grid_quads() as i32 / 2;

        for camera in camera_positions(300) {
            for level in 1..5 {
                let coarse = window_origin(&config, level, camera);
                let fine = window_origin(&config, level - 1, camera);
                for axis in 0..2 {
                    let start = fine[axis] / 2 - coarse[axis];
                    assert!(
                        start >= ring && start + footprint <= ring + hole,
                        "footprint {start}..{} escapes the hole {ring}..{}",
                        start + footprint,
                        ring + hole
                    );
                }
            }
        }
    }

    #[test]
    fn the_camera_stays_near_the_middle_of_every_window() {
        let config = config();
        let half = f64::from(config.grid_quads()) * 0.5;

        for camera in camera_positions(200) {
            for level in 0..6 {
                let origin = window_origin(&config, level, camera);
                let texels = camera / f64::from(1u32 << level);
                for axis in 0..2 {
                    let offset = texels[axis] - f64::from(origin[axis]);
                    // Within a couple of texels of centre; the snap to an even
                    // lattice is the only thing that moves it off.
                    assert!(
                        (offset - half).abs() <= 2.0,
                        "level {level} axis {axis}: camera {offset} from origin, \
                         window centre at {half}"
                    );
                }
            }
        }
    }

    /// The texels a window covers, as absolute coordinates.
    fn window_texels(origin: IVec2, span: u32) -> HashSet<(i32, i32)> {
        let mut set = HashSet::new();
        for y in 0..span as i32 {
            for x in 0..span as i32 {
                set.insert((origin.x + x, origin.y + y));
            }
        }
        set
    }

    #[test]
    fn exposed_regions_cover_exactly_the_newly_visible_texels() {
        let span = 16;
        let mut seed = 0xfeed_face_0000_0001;

        for _ in 0..400 {
            let old = IVec2::new(
                (pseudo_random(&mut seed) % 41) as i32 - 20,
                (pseudo_random(&mut seed) % 41) as i32 - 20,
            );
            let new = old
                + IVec2::new(
                    (pseudo_random(&mut seed) % 41) as i32 - 20,
                    (pseudo_random(&mut seed) % 41) as i32 - 20,
                );

            let expected: HashSet<_> = window_texels(new, span)
                .difference(&window_texels(old, span))
                .copied()
                .collect();

            let regions = exposed_regions(old, new, span);

            let mut covered = HashSet::new();
            for region in &regions {
                for y in 0..region.height as i32 {
                    for x in 0..region.width as i32 {
                        let texel = (region.x + x, region.y + y);
                        assert!(covered.insert(texel), "{texel:?} uploaded twice");
                    }
                }
            }

            assert_eq!(
                covered, expected,
                "moving {old} -> {new} refreshed the wrong texels"
            );
        }
    }

    #[test]
    fn standing_still_uploads_nothing() {
        let origin = IVec2::new(4, -6);
        assert!(exposed_regions(origin, origin, 16).is_empty());
    }

    #[test]
    fn a_jump_of_a_whole_window_refreshes_all_of_it() {
        let regions = exposed_regions(IVec2::ZERO, IVec2::new(16, 0), 16);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].size(), UVec2::splat(16));
    }

    #[test]
    fn a_one_texel_step_uploads_a_single_strip() {
        let regions = exposed_regions(IVec2::ZERO, IVec2::new(1, 0), 16);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].size(), UVec2::new(1, 16));
    }

    #[test]
    fn seam_pieces_tile_the_rectangle_and_land_where_the_torus_wraps() {
        let span = 16;
        let mut seed = 0x0bad_c0de_0000_0001;

        for _ in 0..300 {
            let rect = Rect {
                x: (pseudo_random(&mut seed) % 81) as i32 - 40,
                y: (pseudo_random(&mut seed) % 81) as i32 - 40,
                width: 1 + (pseudo_random(&mut seed) % span as u64) as u32,
                height: 1 + (pseudo_random(&mut seed) % span as u64) as u32,
            };

            let pieces = split_across_seam(rect, span);
            assert!(
                pieces.len() <= 4,
                "{rect:?} split into {} pieces",
                pieces.len()
            );

            let mut covered = HashSet::new();
            for (piece, destination) in &pieces {
                for y in 0..piece.height as i32 {
                    for x in 0..piece.width as i32 {
                        let source = (piece.x + x, piece.y + y);
                        assert!(covered.insert(source), "{source:?} written twice");

                        // Every texel must land where the torus says it should.
                        let expected = (
                            source.0.rem_euclid(span as i32) as u32,
                            source.1.rem_euclid(span as i32) as u32,
                        );
                        assert_eq!(
                            (destination.x + x as u32, destination.y + y as u32),
                            expected,
                            "{source:?} landed in the wrong place"
                        );
                    }
                }
            }

            assert_eq!(
                covered.len(),
                (rect.width * rect.height) as usize,
                "{rect:?} was not fully covered"
            );
            // No piece may run past the edge of the texture.
            for (piece, destination) in &pieces {
                assert!(destination.x + piece.width <= span);
                assert!(destination.y + piece.height <= span);
            }
        }
    }

    #[test]
    fn a_rectangle_clear_of_the_seam_is_left_whole() {
        let pieces = split_across_seam(
            Rect {
                x: 2,
                y: 3,
                width: 4,
                height: 5,
            },
            16,
        );
        assert_eq!(pieces.len(), 1);
        assert_eq!(pieces[0].1, UVec2::new(2, 3));
    }

    #[test]
    fn enough_levels_are_used_to_reach_across_the_raster() {
        let config = config();
        let quads = config.grid_quads();

        for size in [1u32, quads / 2, quads, quads + 1, quads * 8, quads * 8 + 1] {
            let levels = config.level_count(UVec2::splat(size), 32);
            let reach = quads * (1 << (levels - 1));
            assert!(
                reach >= size,
                "{levels} levels reach {reach} texels, raster is {size}"
            );
            // ... and no more levels than that, so nothing is drawn twice over.
            if levels > 1 {
                let smaller = quads * (1 << (levels - 2));
                assert!(smaller < size, "{levels} levels is one more than needed");
            }
        }
    }

    #[test]
    fn level_count_never_exceeds_the_mips_that_exist() {
        let config = config();
        assert_eq!(config.level_count(UVec2::splat(100_000), 4), 4);
        assert_eq!(config.level_count(UVec2::splat(1), 8), 1);
    }
}
