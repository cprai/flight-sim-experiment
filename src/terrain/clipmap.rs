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
    /// Side length of a level's clip texture, in texels.
    ///
    /// At least one texel wider than the grid the mesh draws, so a central
    /// difference at the last vertex still has a neighbour to read, and a power
    /// of two so that wrapping around the torus is a bit mask.
    ///
    /// It is *not* tied to the grid, because the two answer different questions.
    /// The grid says how many triangles a ring costs; the window says how much
    /// ground is resident at each level, which is what decides how fine the
    /// raymarched far field can be at a given distance. Widening the window
    /// costs memory quadratically and no triangles at all, so the mesh keeps its
    /// grid and sits in the middle of whatever window is affordable.
    pub window_texels: u32,
    /// Fraction of a ring's outer edge over which it blends into the next level.
    pub morph_band: f32,
    /// How far out geometry is rasterized, in ring reaches of the base level.
    ///
    /// Ground beyond it is raymarched instead of drawn as triangles, so this is
    /// where the two halves of the renderer meet. One unit is the distance at
    /// which the finest level being drawn would normally hand over to the next
    /// -- the radius of the hole its ring leaves.
    ///
    /// Measured in ring reaches rather than metres so that a single figure
    /// suits a raster of any resolution, and so that the disc widens with
    /// altitude at exactly the rate the levels inside it coarsen: the triangle
    /// count it costs then stays roughly flat whatever the camera is doing.
    /// [`f32::INFINITY`] rasterizes everything and zero raymarches everything,
    /// which is how the two halves are tested against each other.
    pub near_rings: f32,
}

impl Default for ClipmapConfig {
    fn default() -> Self {
        Self {
            block_verts: 64,
            // Exactly one texel wider than the grid, so the mesh fills the
            // window and the margin around it is zero.
            window_texels: 256,
            morph_band: 0.25,
            // Chosen by measurement, not by feel: see the commit that set it.
            // Four reaches sheds a third of the geometry at low altitude while
            // moving the frame by 0.11 of 255; eight sheds only a twelfth, and
            // two starts to show where the ring blend and the march disagree.
            near_rings: 4.0,
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

    /// How many texels of window lie outside the grid, on each side.
    ///
    /// The grid sits in the middle of the window, so that the ground the mesh
    /// draws and the ground the far field marches are both centred on the
    /// camera. Always even, which is what keeps window origins even and the
    /// level-to-level halving exact: `window_texels` and `grid_verts + 1` are
    /// both powers of two, so their difference is a multiple of the smaller.
    pub const fn margin(&self) -> u32 {
        (self.window_texels - self.grid_verts() - 1) / 2
    }

    /// Coarsest depth of the max pyramid, where one texel covers `2^max_mip`
    /// samples on a side.
    ///
    /// Three short of the whole window rather than the window itself, and that
    /// costs the reach below. The pyramid is anchored to the raster, so a cell
    /// at depth `m` starts at a multiple of `2^m` samples and a window whose
    /// origin is not such a multiple straddles one cell more than it has room
    /// for. Stopping three depths short leaves the coarsest cell an eighth of a
    /// window, so an empty annulus still crosses in a handful of steps.
    pub const fn max_mip(&self) -> u32 {
        self.window_texels.trailing_zeros().saturating_sub(3)
    }

    /// Side length of a level's window, in quads.
    ///
    /// How far a ray may travel across a level before it has to be handed over
    /// to the level outside. Short of the window by one coarsest cell, which is
    /// exactly the overhang [`ClipmapConfig::max_mip`] describes: a ray past
    /// this point would want a ceiling the texture has no room for and would
    /// read a wrapped one instead, which is ground somewhere else entirely.
    pub const fn window_quads(&self) -> u32 {
        self.window_texels - (1 << self.max_mip())
    }

    /// Side length of the hole a ring leaves for the next finer level, in quads.
    pub const fn hole_quads(&self) -> u32 {
        self.grid_quads() - 2 * self.ring_quads()
    }

    /// How far out geometry is rasterized, in metres.
    ///
    /// `base` is the finest level being drawn, from [`detail_base`]; its ring
    /// reach is the unit [`ClipmapConfig::near_rings`] counts.
    pub fn near_radius(&self, metres_per_texel: f64, base: u32) -> f64 {
        f64::from(self.near_rings)
            * f64::from(self.hole_quads())
            * 0.5
            * f64::from(1u32 << base)
            * metres_per_texel
    }

    /// How many levels are needed to cover a raster of this size.
    ///
    /// The coarsest level has to reach the whole raster from wherever the camera
    /// is standing, otherwise there is ground with nothing to draw it and the
    /// horizon stops short of the data. Capped at `available`, the number of mip
    /// levels that actually exist.
    ///
    /// That takes a grid *twice* the raster, not one that merely spans it. Every
    /// window is centred on the camera, so half the grid is behind and a level
    /// wide enough to cover the dataset covers only half of it from any given
    /// spot. A camera at one edge has to see to the other. Getting this wrong is
    /// quiet rather than obvious: nothing looks broken standing still, and the
    /// symptom is distant ground arriving as the camera moves towards it.
    pub fn level_count(&self, raster: UVec2, available: u32) -> u32 {
        let reach = f64::from(self.window_quads()) * 0.5;
        let needed = f64::from(raster.max_element()) / reach;
        // Level `l` reaches `reach * 2^l` from the camera, so this is the first
        // `l` that reaches across the raster, plus one because levels are
        // counted from zero.
        let levels = needed.log2().ceil().max(0.0) as u32 + 1;
        levels.clamp(1, available.max(1))
    }
}

/// The finest level worth drawing when the camera is `distance` metres from the
/// ground beneath it, and how far it has already blended into the level outside
/// it.
///
/// Levels are nested rings centred on the camera, so a level's detail is chosen
/// by how far away the ground it covers is: level `l`'s ring starts at
/// `hole_quads / 2 * 2^l` texels out. Horizontally that falls out of the
/// geometry for free, but a camera in the air is that far from the ground
/// directly below it as well, and nothing in the ring layout knows it. Drawing
/// the finest level from ten kilometres up spends full-resolution triangles --
/// and a fine window's worth of tile reads -- on ground that covers a fraction
/// of a pixel.
///
/// So the same rule is applied to the vertical: the level that would serve
/// horizontal distance `distance` is the finest level worth drawing at altitude
/// `distance`, and everything below it is dropped. Combined with the rings, a
/// point at horizontal distance `r` ends up at `max(level_of(r),
/// level_of(distance))`, which is within half a level of the `sqrt(r^2 + d^2)`
/// an exact radial measure would give -- the same square-for-round
/// approximation the rings themselves already make.
///
/// The fractional part is returned rather than rounded away: the caller blends
/// the base level uniformly into the level outside it by that much, so by the
/// time a level is dropped it is already drawing the coarser surface exactly and
/// its disappearance is invisible. It is zero at the coarsest level, which has
/// nothing outside it to blend towards.
pub fn detail_base(
    config: &ClipmapConfig,
    metres_per_texel: f64,
    distance: f64,
    levels: u32,
) -> (u32, f32) {
    // Where level 0 hands over to level 1: the radius of the hole its ring
    // leaves. Half a grid out, the next level has taken over regardless.
    let handover = f64::from(config.hole_quads()) * 0.5 * metres_per_texel;
    let coarsest = levels.saturating_sub(1);

    let t = (distance / handover)
        .max(1.0)
        .log2()
        .clamp(0.0, f64::from(coarsest));
    let base = t.floor();
    if base as u32 >= coarsest {
        (coarsest, 0.0)
    } else {
        (base as u32, (t - base) as f32)
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
///
/// The margin is stepped back on top of the grid's own step, so that the grid
/// lands in the middle of the window rather than in its corner. Both terms are
/// even, so the origin still is.
pub fn window_origin(config: &ClipmapConfig, level: u32, camera_texels: DVec2) -> IVec2 {
    let texels = camera_texels / f64::from(1u32 << level);
    let step_back = 2 * config.ring_quads() as i32 + config.margin() as i32;
    IVec2::new(
        snap_axis(texels.x, step_back),
        snap_axis(texels.y, step_back),
    )
}

/// Where a level's *grid* starts, which is the margin in from its window.
///
/// The mesh is laid out in grid coordinates and the textures are addressed in
/// window ones; this is the one place the two are related, so that neither the
/// patch layout nor the culling has to know how wide a window happens to be.
pub fn grid_origin(config: &ClipmapConfig, window: IVec2) -> IVec2 {
    window + config.margin() as i32
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
            window_texels: 32,
            ..Default::default()
        }
    }

    /// The same clipmap with room around the grid, which is what the far field
    /// is fed from. Everything the grid does must survive the margin.
    fn wide() -> ClipmapConfig {
        ClipmapConfig {
            window_texels: 128,
            ..config()
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

    /// The transform the raymarched far field hands a ray over levels with.
    ///
    /// A point's window position on the level outside is
    /// `w * 0.5 + coarse_offset`, and it has to be *exact*: the march applies it
    /// to both a position and a direction, so any drift would accumulate over
    /// every handoff and pull the ray off the line it started on. It is exact
    /// because window origins are always even, which is what `snap_axis` snaps
    /// to two for, so equality here is the right assertion rather than a
    /// tolerance.
    #[test]
    fn the_coarse_offset_moves_a_point_between_levels_exactly() {
        for config in [config(), wide()] {
            for camera in camera_positions(200) {
                for level in 0..6 {
                    let fine = window_origin(&config, level, camera);
                    let coarse = window_origin(&config, level + 1, camera);
                    let offset = (fine / 2 - coarse).as_dvec2();

                    // Every vertex of the finer window, and the half-texel
                    // positions between them that a ray also lands on.
                    for step in 0..=2 * (config.window_texels - 1) {
                        let w = f64::from(step) * 0.5;
                        // Where this window position sits in the raster, at
                        // each level's own resolution.
                        let texel = f64::from(fine.x) + w;
                        assert_eq!(
                            w * 0.5 + offset.x,
                            texel / 2.0 - f64::from(coarse.x),
                            "level {level} at w {w} from camera {camera}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_near_radius_is_a_ring_reach_of_the_level_it_is_measured_from() {
        let config = ClipmapConfig {
            near_rings: 1.0,
            ..config()
        };
        // One ring reach is where a level's own ring starts, which is exactly
        // the distance `detail_base` hands that level over to the next at.
        let handover = f64::from(config.hole_quads()) * 0.5 * 30.0;
        assert_eq!(config.near_radius(30.0, 0), handover);

        // And it doubles per level, so the disc keeps pace with the ground each
        // level covers as the camera climbs and the base level rises.
        assert_eq!(config.near_radius(30.0, 3), handover * 8.0);

        // The two ends the far field is tested between: nothing rasterized, and
        // nothing raymarched.
        let none = ClipmapConfig {
            near_rings: 0.0,
            ..config
        };
        assert_eq!(none.near_radius(30.0, 4), 0.0);
        let all = ClipmapConfig {
            near_rings: f32::INFINITY,
            ..config
        };
        assert!(all.near_radius(30.0, 0).is_infinite());
    }

    #[test]
    fn a_window_holds_the_grid_with_an_even_margin_around_it() {
        for config in [config(), wide(), ClipmapConfig::default()] {
            // A power of two, so wrapping is a mask rather than a division.
            assert!(config.window_texels.is_power_of_two());
            // Room for the grid and for the extra texel a central difference at
            // its last vertex reads.
            assert!(config.window_texels > config.grid_verts());
            assert_eq!(
                2 * config.margin() + config.grid_verts() + 1,
                config.window_texels,
                "the margin has to be the same on both sides"
            );
            // Evenness is load bearing: an odd margin would make a window origin
            // odd, and halving it on the way to the coarser level would stop
            // being exact.
            assert_eq!(config.margin() % 2, 0, "margin {}", config.margin());
        }
    }

    /// The margin is room around the mesh, not a move of it.
    ///
    /// The vertex stage adds the margin to every grid coordinate before reading
    /// a texture, so the ground a given vertex lands on has to be independent of
    /// how wide the window it sits in happens to be. If it were not, widening
    /// the window to feed the far field would shift the near field sideways.
    #[test]
    fn widening_the_window_does_not_move_the_grid() {
        let (tight, wide) = (config(), wide());
        for camera in camera_positions(200) {
            for level in 0..6 {
                let origin = |c: &ClipmapConfig| grid_origin(c, window_origin(c, level, camera));
                assert_eq!(
                    origin(&tight),
                    origin(&wide),
                    "level {level} at {camera}: the grid moved with the window"
                );
            }
        }
    }

    #[test]
    fn window_origins_land_on_even_texels() {
        for config in [config(), wide()] {
            for camera in camera_positions(200) {
                for level in 0..6 {
                    let origin = window_origin(&config, level, camera);
                    assert_eq!(origin.x % 2, 0, "level {level} at {camera}: {origin}");
                    assert_eq!(origin.y % 2, 0, "level {level} at {camera}: {origin}");
                }
            }
        }
    }

    #[test]
    fn each_level_nests_inside_the_next_with_at_most_one_quad_of_slack() {
        for config in [config(), wide()] {
            // A fine window sits a ring's thickness inside the coarse one, plus
            // half the margin: the margin is stepped back at both levels, and
            // measuring the fine one on the coarse grid halves it.
            let ring = config.ring_quads() as i32 + config.margin() as i32 / 2;

            for camera in camera_positions(500) {
                for level in 1..6 {
                    let coarse = window_origin(&config, level, camera);
                    let fine = window_origin(&config, level - 1, camera);

                    for axis in 0..2 {
                        // Where the fine window starts, on the coarse lattice.
                        let offset = fine[axis] / 2 - coarse[axis];
                        assert!(
                            offset == ring || offset == ring + 1,
                            "level {level} axis {axis} at {camera}: offset {offset}, \
                             expected {ring} or {}",
                            ring + 1
                        );

                        // Which of the two it is depends only on the parity of
                        // the camera's position on this level's lattice, which
                        // is what selects the trim's orientation.
                        let texels = camera[axis] / f64::from(1u32 << level);
                        let parity = (texels.floor() as i32).rem_euclid(2);
                        assert_eq!(offset - ring, parity, "parity should pick the offset");
                    }
                }
            }
        }
    }

    #[test]
    fn the_finer_footprint_stays_inside_the_hole() {
        for config in [config(), wide()] {
            // Measured between grids rather than windows: the hole a ring
            // leaves is a fact about the mesh, and the margin around it is not
            // part of it.
            let ring = config.ring_quads() as i32;
            let hole = config.hole_quads() as i32;
            // A fine level covers half its quads on the coarse grid.
            let footprint = config.grid_quads() as i32 / 2;

            for camera in camera_positions(300) {
                for level in 1..5 {
                    let coarse = grid_origin(&config, window_origin(&config, level, camera));
                    let fine = grid_origin(&config, window_origin(&config, level - 1, camera));
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
    }

    #[test]
    fn the_camera_stays_near_the_middle_of_every_window() {
        for config in [config(), wide()] {
            let half = f64::from(config.window_texels - 1) * 0.5;

            for camera in camera_positions(200) {
                for level in 0..6 {
                    let origin = window_origin(&config, level, camera);
                    let texels = camera / f64::from(1u32 << level);
                    for axis in 0..2 {
                        let offset = texels[axis] - f64::from(origin[axis]);
                        // Within a couple of texels of centre; the snap to an
                        // even lattice is the only thing that moves it off.
                        assert!(
                            (offset - half).abs() <= 2.0,
                            "level {level} axis {axis}: camera {offset} from origin, \
                             window centre at {half}"
                        );
                    }
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

    /// Reach is measured from the camera outwards, not across the grid, because
    /// the camera stands in the middle of every window. A level that spans the
    /// raster covers half of it from any one spot, which looks fine until the
    /// camera moves and the rest of the dataset arrives at the horizon.
    #[test]
    fn enough_levels_are_used_to_reach_across_the_raster() {
        for config in [config(), wide()] {
            let quads = config.window_quads();
            // What the coarsest level reaches from the camera: half its window.
            // The window rather than the grid, because the far field marches to
            // the window's edge and it is the far field that draws the horizon.
            let reach_of = |levels: u32| quads / 2 * (1 << (levels - 1));

            for size in [1u32, quads / 2, quads, quads + 1, quads * 8, quads * 8 + 1] {
                let levels = config.level_count(UVec2::splat(size), 32);
                let reach = reach_of(levels);
                assert!(
                    reach >= size,
                    "{levels} levels reach {reach} texels from the camera, \
                     which has to see {size} to the far edge"
                );
                // ... and no more than that, so nothing is drawn twice over.
                if levels > 1 {
                    let smaller = reach_of(levels - 1);
                    assert!(smaller < size, "{levels} levels is one more than needed");
                }
            }
        }
    }

    #[test]
    fn level_count_never_exceeds_the_mips_that_exist() {
        let config = config();
        assert_eq!(config.level_count(UVec2::splat(100_000), 4), 4);
        assert_eq!(config.level_count(UVec2::splat(1), 8), 1);
    }

    /// The distance at which level 0 hands over to level 1.
    fn handover(config: &ClipmapConfig, metres_per_texel: f64) -> f64 {
        f64::from(config.hole_quads()) * 0.5 * metres_per_texel
    }

    #[test]
    fn every_level_survives_until_the_camera_is_its_own_radius_away() {
        let config = config();
        let reach = handover(&config, 10.0);

        // On the ground and just below the handover, nothing is given up.
        for distance in [0.0, 1.0, reach * 0.5, reach * 0.99] {
            let (base, _) = detail_base(&config, 10.0, distance, 8);
            assert_eq!(base, 0, "level 0 was dropped only {distance} m up");
        }

        // ... and past it, one level goes per doubling, the same schedule the
        // rings follow outwards.
        for level in 0..6u32 {
            let distance = reach * f64::from(1u32 << level);
            let (base, _) = detail_base(&config, 10.0, distance * 1.01, 8);
            assert_eq!(base, level, "wrong level at {distance} m up");
        }
    }

    #[test]
    fn a_level_is_fully_blended_away_before_it_is_dropped() {
        let config = config();
        let reach = handover(&config, 10.0);

        // Approaching a boundary from below, the level being retired is all but
        // entirely blended into the one outside it; crossing it, the level that
        // takes over starts from its own surface. The two describe the same
        // shape, which is what makes the drop invisible.
        let (below, below_morph) = detail_base(&config, 10.0, reach * 1.999, 8);
        let (above, above_morph) = detail_base(&config, 10.0, reach * 2.001, 8);
        assert_eq!((below, above), (0, 1));
        assert!(below_morph > 0.999, "blend stopped short at {below_morph}");
        assert!(above_morph < 0.001, "blend restarted at {above_morph}");
    }

    #[test]
    fn the_coarsest_level_is_never_dropped_and_never_blends_outwards() {
        let config = config();
        let reach = handover(&config, 10.0);

        // There is nothing outside the coarsest level to blend towards, so the
        // ramp has to stop there however high the camera climbs.
        for levels in 1..6u32 {
            for distance in [reach * f64::from(1u32 << levels), 1.0e9] {
                let (base, morph) = detail_base(&config, 10.0, distance, levels);
                assert_eq!(base, levels - 1, "{levels} levels, {distance} m up");
                assert_eq!(morph, 0.0, "the coarsest level blended outwards");
            }
        }
    }

    #[test]
    fn a_finer_raster_gives_up_its_finest_level_sooner() {
        let config = config();
        // The same altitude is more levels' worth of distance over a raster
        // whose texels are smaller, because the level it should be drawn at is
        // measured in texels rather than metres.
        let (coarse, _) = detail_base(&config, 40.0, 5_000.0, 12);
        let (fine, _) = detail_base(&config, 5.0, 5_000.0, 12);
        assert_eq!(fine, coarse + 3, "eight times finer is three levels sooner");
    }
}
