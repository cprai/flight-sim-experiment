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
    /// How many bytes of texture the clipmap may occupy.
    ///
    /// A ceiling on [`ClipmapConfig::window_texels`], applied by
    /// [`ClipmapConfig::fit_window`] once the raster's size is known: the
    /// window the screen asks for is halved until the textures fit. Memory is
    /// quadratic in the window, so one halving is four times less.
    pub memory_budget: usize,
    /// The angle one pixel of the target subtends, in radians.
    ///
    /// The one number both detail rules are written in. How much ground stays
    /// resident at each level -- [`ClipmapConfig::window_for`] -- and how many
    /// of the fine levels are worth keeping at all -- [`detail_base`] -- are
    /// the same question asked at two distances, and neither can be answered
    /// without knowing how large a pixel is. See [`pixel_angle`].
    pub pixel_angle: f64,
    /// How many cells of one level a ray may visit before the march gives up.
    ///
    /// Sixteen, because a window is eight of the coarsest cells across by
    /// construction -- [`ClipmapConfig::max_mip`] stops three depths short of
    /// covering one -- and a ray crossing on the diagonal meets at most twice
    /// that many. A field rather than a constant so that a test can starve it
    /// and see what the march does when it runs out.
    pub march_cells: u32,
}

/// The angle one pixel subtends, for a viewport of this height.
///
/// The horizontal angle is the same: widening the viewport widens the field of
/// view with it rather than stretching the picture, so pixels stay square.
pub fn pixel_angle(viewport_height: u32, fov_y: f64) -> f64 {
    2.0 * (fov_y * 0.5).tan() / f64::from(viewport_height.max(1))
}

/// Widest window ever asked for, whatever the screen would like.
///
/// Not a memory limit -- [`ClipmapConfig::memory_budget`] is that -- but a
/// limit on how much ground the update path is asked to move in one go. A
/// window this wide already holds sixteen million texels a level.
pub const MAX_WINDOW: u32 = 4096;

/// How many texels of a level may cover one pixel, at the worst point of that
/// level's range.
///
/// Level `l` serves from `(window_quads / 2) * 2^(l-1)` texels out to
/// `(window_quads / 2) * 2^l`, and its texel is `2^l` of the base raster's, so
/// across a level a texel covers between `2 / (window * pixel)` and
/// `4 / (window * pixel)` pixels, where `pixel` is the angle one pixel
/// subtends. The ratio swings by exactly two whatever the window is -- that is
/// what a power-of-two level chain costs -- so this names the worse end. One
/// means no texel is ever larger than a pixel; two would allow one to two and
/// costs a quarter of the memory.
const TEXELS_PER_PIXEL: f64 = 1.0;

impl Default for ClipmapConfig {
    fn default() -> Self {
        Self {
            block_verts: 64,
            // Replaced by `window_for` wherever a viewport height is known, and
            // capped by `fit_window` once the raster's size is.
            window_texels: MAX_WINDOW,
            morph_band: 0.25,
            // Chosen by measurement, not by feel: see the commit that set it.
            // Four reaches sheds a third of the geometry at low altitude while
            // moving the frame by 0.11 of 255; eight sheds only a twelfth, and
            // two starts to show where the ring blend and the march disagree.
            near_rings: 4.0,
            // Room for a 4096 window on the raster this flies, which is seven
            // levels of heights, colours and maxima and comes to 1195 MiB.
            // Sized to admit that rather than to any round number, because the
            // window is the one knob worth spending memory on and the sizes it
            // can take are powers of two -- a budget between two of them buys
            // nothing.
            memory_budget: 1600 << 20,
            // 1080p at sixty degrees, replaced wherever a real viewport is
            // known.
            pixel_angle: 2.0 * (30f64.to_radians()).tan() / 1080.0,
            march_cells: 16,
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

    /// The window width that resolves a level's texel to a pixel.
    ///
    /// The far field's sharpness is not a property of how hard it is marched
    /// but of how much ground is resident at each level, and that is the
    /// window. Anything finer than this is detail the screen cannot show;
    /// anything coarser is texels visibly larger than pixels in the distance.
    pub fn window_for(&self) -> u32 {
        let wanted = 4.0 / (TEXELS_PER_PIXEL * self.pixel_angle);
        (wanted.ceil() as u32)
            .next_power_of_two()
            .clamp(self.grid_verts() + 1, MAX_WINDOW)
    }

    /// Bytes of texture a clipmap of this shape occupies.
    ///
    /// Heights and colours are four bytes a texel at full window size; the max
    /// pyramid is two bytes and a real mip chain, so it costs about a third
    /// again over its own base.
    pub fn texture_bytes(&self, levels: u32) -> usize {
        let window = self.window_texels as usize;
        let mut per_level = window * window * (size_of::<f32>() + 4);
        for mip in 0..=self.max_mip() {
            let side = (window >> mip).max(1);
            per_level += side * side * size_of::<u16>();
        }
        per_level * levels as usize
    }

    /// The widest window no wider than this one whose textures fit the budget.
    ///
    /// Halving the window quarters the memory and also, usually, drops a level
    /// -- a narrower window reaches less far, so more of them are needed to
    /// cross the raster -- so the saving is a little under four each time.
    pub fn fit_window(&self, raster: UVec2, available: u32) -> u32 {
        let smallest = self.grid_verts() + 1;
        let mut window = self.window_texels.max(smallest);
        while window > smallest {
            let trial = Self {
                window_texels: window,
                ..*self
            };
            if trial.texture_bytes(trial.level_count(raster, available)) <= self.memory_budget {
                break;
            }
            window /= 2;
        }
        window
    }

    /// How many cells a ray may visit before the far field gives up on it.
    ///
    /// Rays that meet the ground stop when they meet it; this bounds the ones
    /// that do not, which are the ones running along a slope just above the
    /// surface. They are the expensive case in any maximum-mipmap traversal --
    /// too close to skip a cell, too far to hit one -- and on a horizon view a
    /// whole column of pixels can be doing it at once.
    ///
    /// Derived rather than picked, because what a ray legitimately needs scales
    /// with the clipmap. Within one level it crosses at most
    /// [`ClipmapConfig::march_cells`] of the coarsest cells, and each of those
    /// may cost a descent through every depth beneath it; then it hands over to
    /// the next level and does it again.
    ///
    /// That is an underestimate for a ray that hugs the surface the whole way,
    /// which in the worst case visits a cell per texel. Bounding *that* would
    /// mean no bound at all, so the march reports where it had got to rather
    /// than reporting sky when it does run out -- a pixel a little too near
    /// with roughly the right colour, instead of a hole.
    pub const fn march_steps(&self, levels: u32) -> u32 {
        levels * self.march_cells * (self.max_mip() + 1)
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

/// The finest level worth keeping when the camera is `distance` metres from the
/// ground beneath it, and how far it has already blended into the level outside
/// it.
///
/// A level earns its residency while its texels are still smaller than the
/// pixels they land in. The nearest ground on screen is the ground directly
/// below, at `distance`, and that is where the demand is highest: a pixel there
/// covers `distance * pixel_angle` metres, and any level finer than that is
/// detail nothing can show. Everything below it is dropped, which saves both
/// the triangles the mesh would spend on it and -- much more valuable -- the
/// tiles that would have to be read to fill its window.
///
/// The rule this replaced measured altitude in ring reaches: level `l` went
/// once the camera was `hole_quads / 2 * 2^l` texels up, which is where the
/// *rings* hand `l` over to `l + 1` horizontally. That was self-consistent but
/// had nothing to do with what could be seen. On a one-metre raster it gave up
/// level 0 at sixty-four metres of altitude, where a 1080p screen can still
/// resolve a one-metre texel from six hundred; at ten kilometres up it was
/// asking for a hundred-and-twenty-eight-metre texel where sixteen would do.
/// Both errors were invisible while the mesh drew everything, because a level
/// dropped for altitude is one the rings were about to hand over anyway. They
/// stopped being invisible when the far field started reading the same levels
/// out to the edge of the window.
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
    let coarsest = levels.saturating_sub(1);
    // How many base texels a pixel covers at the nearest ground on screen.
    let resolvable = distance * config.pixel_angle / metres_per_texel;

    // Level `l`'s texels are `2^l` base texels across, so this is the level
    // whose texels are about a pixel. Below one the finest level is still not
    // enough, which is the ordinary case near the ground.
    let t = resolvable.max(1.0).log2().clamp(0.0, f64::from(coarsest));
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

    /// The window is sized so that no texel of any level is larger than the
    /// pixel it lands in -- and no larger than it has to be.
    ///
    /// Both halves are the point. Too narrow and the far field is visibly
    /// blocky however hard it is marched, because the detail is simply not
    /// resident; too wide and the memory, which is quadratic in this, is spent
    /// on ground the screen cannot resolve.
    #[test]
    fn a_texel_never_covers_more_than_a_pixel() {
        let fov = 60f64.to_radians();
        let ratio = |window: u32, height: u32| {
            // The coarse end of a level's range, where a texel is largest.
            4.0 / (f64::from(window) * pixel_angle(height, fov))
        };
        let sized = |height: u32| ClipmapConfig {
            pixel_angle: pixel_angle(height, fov),
            ..ClipmapConfig::default()
        };

        for height in [360u32, 480, 720, 1080, 1440, 2160] {
            let config = sized(height);
            let window = config.window_for();
            assert!(window.is_power_of_two(), "{height}p asked for {window}");
            // Past the cap the screen is asking for more than the update path
            // is willing to move, and distant texels start to show.
            assert!(
                ratio(window, height) <= TEXELS_PER_PIXEL || window == MAX_WINDOW,
                "{height}p at {window} texels leaves {:.2} texels a pixel",
                ratio(window, height)
            );
            // Halving it would not do, or the window is wider than it need be.
            assert!(
                window == MAX_WINDOW
                    || window == config.grid_verts() + 1
                    || ratio(window / 2, height) > TEXELS_PER_PIXEL,
                "{height}p could have made do with {}",
                window / 2
            );
        }

        // Twice the pixels wants twice the window, until the cap.
        assert_eq!(sized(1080).window_for(), sized(540).window_for() * 2);
    }

    /// The step budget has to grow with the traversal it bounds, or widening
    /// the window buys detail the march then runs out of steps before reaching.
    #[test]
    fn the_step_budget_grows_with_what_a_ray_has_to_cross() {
        // A window is eight of the coarsest cells across whatever its width,
        // because `max_mip` stops three depths short of covering one. What
        // grows with the window is how many depths there are to descend
        // through, and what grows with the raster is how many levels a ray
        // hands over between.
        for window in [256u32, 1024, 4096] {
            let config = ClipmapConfig {
                window_texels: window,
                ..ClipmapConfig::default()
            };
            assert_eq!(window >> config.max_mip(), 8);
            assert_eq!(
                config.march_steps(7),
                7 * config.march_cells * (config.max_mip() + 1)
            );
        }

        let narrow = ClipmapConfig {
            window_texels: 256,
            ..ClipmapConfig::default()
        };
        let wide = ClipmapConfig {
            window_texels: 4096,
            ..ClipmapConfig::default()
        };
        assert!(
            wide.march_steps(7) > narrow.march_steps(7),
            "a wider window has more depths to descend and needs more steps"
        );
        assert!(
            wide.march_steps(11) > wide.march_steps(7),
            "more levels to hand over between needs more steps"
        );
    }

    #[test]
    fn the_window_shrinks_until_its_textures_fit() {
        let raster = UVec2::splat(100_000);
        let wanted = ClipmapConfig {
            window_texels: 4096,
            ..ClipmapConfig::default()
        };
        let cost = |window: u32| {
            let trial = ClipmapConfig {
                window_texels: window,
                ..wanted
            };
            trial.texture_bytes(trial.level_count(raster, 32))
        };

        // A budget that fits what was asked for leaves it alone.
        let roomy = ClipmapConfig {
            memory_budget: cost(4096),
            ..wanted
        };
        assert_eq!(roomy.fit_window(raster, 32), 4096);

        // One byte short of it drops a halving, and no more than one.
        let tight = ClipmapConfig {
            memory_budget: cost(4096) - 1,
            ..wanted
        };
        assert_eq!(tight.fit_window(raster, 32), 2048);

        // Nothing shrinks below the grid the mesh has to draw, however small
        // the budget: a window narrower than that has no clipmap in it.
        let starved = ClipmapConfig {
            memory_budget: 0,
            ..wanted
        };
        assert_eq!(
            starved.fit_window(raster, 32),
            starved.grid_verts() + 1,
            "the mesh still needs its grid"
        );
    }

    /// Widening the window is what buys far-field detail, and it pays for
    /// itself twice: the same ground is covered by fewer, finer levels.
    #[test]
    fn a_wider_window_reaches_further_with_fewer_levels() {
        let raster = UVec2::splat(114_688);
        let mut previous = u32::MAX;
        for window in [256u32, 512, 1024, 2048, 4096] {
            let config = ClipmapConfig {
                window_texels: window,
                ..ClipmapConfig::default()
            };
            let levels = config.level_count(raster, 32);
            assert!(
                levels <= previous,
                "{window} texels wanted {levels} levels, more than {previous}"
            );
            previous = levels;
            // ... and however few, they still have to reach the far edge.
            let reach = u64::from(config.window_quads() / 2) << (levels - 1);
            assert!(
                reach >= u64::from(raster.x),
                "{levels} levels of {window} reach only {reach}"
            );
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

    /// How far up a level's texels stop being smaller than a pixel.
    ///
    /// Level 0 goes at `handover`, level 1 at twice that, and so on.
    fn handover(config: &ClipmapConfig, metres_per_texel: f64) -> f64 {
        metres_per_texel / config.pixel_angle
    }

    /// A clipmap whose pixels are a fixed, easily-reckoned angle.
    fn seen_at(pixel: f64) -> ClipmapConfig {
        ClipmapConfig {
            pixel_angle: pixel,
            ..config()
        }
    }

    #[test]
    fn every_level_survives_until_its_texels_are_smaller_than_a_pixel() {
        let config = seen_at(0.001);
        let reach = handover(&config, 10.0);
        assert_eq!(reach, 10_000.0, "ten metres at a milliradian");

        // Below the handover the finest level is still not fine enough, so
        // nothing is given up however close to the ground the camera is.
        for distance in [0.0, 1.0, reach * 0.5, reach * 0.99] {
            let (base, _) = detail_base(&config, 10.0, distance, 8);
            assert_eq!(base, 0, "level 0 was dropped only {distance} m up");
        }

        // ... and past it, one level goes per doubling, because each level's
        // texels are twice the last one's.
        for level in 0..6u32 {
            let distance = reach * f64::from(1u32 << level);
            let (base, _) = detail_base(&config, 10.0, distance * 1.01, 8);
            assert_eq!(base, level, "wrong level at {distance} m up");
        }
    }

    /// The rule this replaced answered a different question, and got this one
    /// wrong in both directions.
    ///
    /// It measured altitude in ring reaches, so on a fine raster it gave up
    /// detail the screen could still resolve, and on a coarse one it kept
    /// detail nothing could see. The far field reads whatever `detail_base`
    /// leaves resident, out to the edge of the window, so both now show.
    #[test]
    fn levels_are_kept_by_what_the_screen_can_show_not_by_ring_geometry() {
        // 1080p at sixty degrees over a one-metre raster.
        let config = ClipmapConfig {
            pixel_angle: pixel_angle(1080, 60f64.to_radians()),
            ..ClipmapConfig::default()
        };
        // The old rule's handover, quoted so the comparison below is concrete.
        assert_eq!(f64::from(config.hole_quads()) * 0.5, 64.0);

        // Sixty-four metres up, the old rule gave up level 0. A pixel there
        // covers a tenth of a metre, so it is nowhere near time: a one-metre
        // texel is still the right size until a pixel covers two of them,
        // which is at 1870 m.
        assert_eq!(detail_base(&config, 1.0, 64.0, 12).0, 0);
        assert_eq!(detail_base(&config, 1.0, 1_800.0, 12).0, 0);
        assert_eq!(detail_base(&config, 1.0, 1_900.0, 12).0, 1);

        // Ten kilometres up the old rule asked for level 7, a hundred and
        // twenty-eight metre texels, where a pixel covers about eleven.
        assert_eq!(detail_base(&config, 1.0, 10_000.0, 12).0, 3);
    }

    #[test]
    fn a_level_is_fully_blended_away_before_it_is_dropped() {
        let config = seen_at(0.001);
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
        let config = seen_at(0.001);
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
        let config = seen_at(0.001);
        // The same altitude is more levels' worth of detail over a raster whose
        // texels are smaller, because what is being compared to a pixel is a
        // texel and not a distance.
        let (coarse, _) = detail_base(&config, 40.0, 500_000.0, 12);
        let (fine, _) = detail_base(&config, 5.0, 500_000.0, 12);
        assert_eq!(fine, coarse + 3, "eight times finer is three levels sooner");
    }

    /// A window the budget cut short cannot deliver what the screen asked for,
    /// and nothing downstream should pretend otherwise.
    ///
    /// The two rules are written in the same number, so they cannot disagree:
    /// the finest level kept is always one the window still has room to serve
    /// at the distance it is wanted.
    #[test]
    fn the_finest_level_kept_is_one_the_window_can_still_serve() {
        let raster = UVec2::splat(114_688);
        for height in [360u32, 720, 1080] {
            let mut config = ClipmapConfig {
                pixel_angle: pixel_angle(height, 60f64.to_radians()),
                ..ClipmapConfig::default()
            };
            config.window_texels = config.window_for();
            config.window_texels = config.fit_window(raster, 32);
            let levels = config.level_count(raster, 32);

            // At any altitude, the base level's own window has to reach at
            // least as far as the ground the camera is looking at from it.
            for altitude in [10.0f64, 300.0, 3_000.0, 30_000.0] {
                let (base, _) = detail_base(&config, 1.0, altitude, levels);
                let reach = f64::from(config.window_quads() / 2) * f64::from(1u32 << base);
                assert!(
                    reach >= altitude || base + 1 == levels,
                    "{height}p at {altitude} m keeps level {base}, whose window \
                     reaches only {reach} m"
                );
            }
        }
    }
}
