//! Which tiles of the terrain are resident, and what to load next.
//!
//! Each level keeps a square of **whole tiles** around the camera. A tile is
//! either resident or it is not; nothing is ever partly uploaded, and the
//! camera moving within a tile costs nothing at all. That is the whole point of
//! this module, and it is the one thing the sliding windows it replaced could
//! not do: they re-read a strip of every level for every metre travelled, and
//! an east-west strip touches every row of every tile it crosses -- 7.15 ms a
//! metre against 0.27 ms going north, because tiles are stored one row per
//! strip. Reading a tile whole is 512 consecutive rows of one file, which is
//! what that layout is fast at.
//!
//! ## Addressing
//!
//! A level's texture is [`Residency::tiles_across`] tiles square, a power of
//! two, so a tile at index `t` lives in slot `t mod N` and a texel at index `x`
//! lives at `x & (N * TILE_SIZE - 1)`. Both are pure functions of position on
//! the raster: there is no window origin to carry, and no torus offset, because
//! a slot's address does not depend on where the resident square happens to be.
//! Level `l`'s texel index is exactly half level `l - 1`'s, so handing a ray
//! from one level to the next is a halving with nothing to correct for.
//!
//! ## Moving
//!
//! The resident square moves one tile at a time. Stepping east means loading a
//! new column, and the slots that column writes are the ones the *westmost*
//! column is using -- `t mod N` wraps -- so that column is spoken for as soon as
//! the step begins. While it is in flight the square advertises itself one tile
//! narrower on the trailing side, and a ray out there falls to the next coarser
//! level, which is what it would have done one tile later anyway. That is what
//! buys a smooth swap without a second ring of memory to swap into.
//!
//! Tiles are handed out a few per frame rather than a column at once, so
//! crossing a boundary costs a bounded amount of work rather than a stall.
//!
//! A level that is being filled outright -- its first fill, or a jump too far
//! to walk to -- is a different case, and reads as resident nowhere until it is
//! whole. It has no trailing side to give up: the square it is loading into is
//! the square it is loading over.

use glam::{DVec2, IVec2};
use terrain_tiles::TILE_SIZE;

/// How much ground each level keeps, and how fast it may be swapped.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Residency {
    /// Tiles across one level's texture. A power of two.
    ///
    /// The single knob that trades memory for reach. Eight tiles is 4096
    /// texels a level, which at seven levels and three products is about 1.2
    /// GiB -- and gives the finest level between three and four tiles of reach
    /// in every direction, depending on where in its own tile the camera
    /// stands.
    pub tiles_across: u32,
    /// Side length of one tile, in texels.
    ///
    /// [`TILE_SIZE`] for anything reading a real pyramid, because a tile of
    /// residency being a tile of the store is what makes an upload one file
    /// read in order. Settable only so tests can work at a scale where a raster
    /// is more than a single tile across.
    pub tile_texels: u32,
    /// How many bytes of texture the whole arrangement may occupy.
    pub memory_budget: usize,
    /// The angle one pixel of the target subtends, in radians.
    ///
    /// Decides how many levels are worth keeping: a level earns its residency
    /// while its texels are still smaller than the pixels they land in. See
    /// [`detail_base`].
    pub pixel_angle: f64,
    /// How many tiles may be uploaded in one update.
    ///
    /// The bound on what crossing a tile boundary costs. Low enough that a
    /// crossing is not a stall, high enough that a level keeps up with ordinary
    /// flight; a level that falls behind is not wrong, only coarser than it
    /// could be at its outer edge.
    pub tiles_per_update: u32,
    /// How many texels of one level a ray may cross before the march gives up.
    ///
    /// A field rather than a constant so a test can starve it on purpose and
    /// see what the march does when it runs out.
    pub march_texels: u32,
}

/// The angle one pixel subtends, for a viewport of this height.
///
/// The horizontal angle is the same: widening the viewport widens the field of
/// view with it rather than stretching the picture, so pixels stay square.
pub fn pixel_angle(viewport_height: u32, fov_y: f64) -> f64 {
    2.0 * (fov_y * 0.5).tan() / f64::from(viewport_height.max(1))
}

/// Widest square ever asked for, in tiles, whatever the budget would allow.
///
/// Not a memory limit -- [`Residency::memory_budget`] is that -- but a limit on
/// how much ground one level is asked to hold. Sixteen tiles is already 67
/// million texels a level.
pub const MAX_TILES_ACROSS: u32 = 16;

impl Default for Residency {
    fn default() -> Self {
        Self {
            tiles_across: 8,
            tile_texels: TILE_SIZE,
            // Room for eight tiles a level on the raster this flies, which is
            // seven levels of heights, colours and maxima.
            memory_budget: 1600 << 20,
            // 1080p at sixty degrees, replaced wherever a real viewport is known.
            pixel_angle: 2.0 * (30f64.to_radians()).tan() / 1080.0,
            tiles_per_update: 4,
            march_texels: 512,
        }
    }
}

impl Residency {
    /// Side length of one level's texture, in texels.
    pub const fn texels_across(&self) -> u32 {
        self.tiles_across * self.tile_texels
    }

    /// The mask that wraps a level texel index onto its slot.
    pub const fn texel_mask(&self) -> u32 {
        self.texels_across() - 1
    }

    /// Bytes of texture this shape occupies at `levels` levels.
    ///
    /// Heights are four bytes a texel, material ids four, and the max pyramid
    /// two -- one cell per texel each, because the level array is the quadtree
    /// and no level carries a mip chain of its own.
    ///
    /// On the raster this flies, eight tiles a side over eight levels comes to
    /// 1280 MiB against a 1600 MiB budget. Fourteen bytes a texel would not
    /// fit: [`Residency::fit_tiles`] would halve the square and the finest
    /// level's reach would fall from 1536 texels to 512. Anything added here
    /// should be checked against that, which is what
    /// `eight_tiles_still_fit_the_budget` does.
    pub fn texture_bytes(&self, levels: u32) -> usize {
        let side = self.texels_across() as usize;
        side * side * (size_of::<f32>() + 4 + size_of::<u16>()) * levels as usize
    }

    /// The widest square no wider than this one whose textures fit the budget.
    ///
    /// Halving quarters the memory and also, usually, drops a level -- a
    /// smaller square reaches less far, so more levels are needed to cross the
    /// raster -- so the saving is a little under four each time.
    pub fn fit_tiles(&self, raster: glam::UVec2, available: u32) -> u32 {
        let mut tiles = self.tiles_across.clamp(1, MAX_TILES_ACROSS);
        while tiles > 1 {
            let trial = Self {
                tiles_across: tiles,
                ..*self
            };
            if trial.texture_bytes(trial.level_count(raster, available)) <= self.memory_budget {
                break;
            }
            tiles /= 2;
        }
        tiles
    }

    /// How far a level reaches from the camera, in its own texels.
    ///
    /// The square is centred on the tile the camera stands in, so half of it
    /// lies behind whichever way the camera faces. This is the half that is
    /// guaranteed in every direction, which is one tile short of half the
    /// square: the camera may stand anywhere within its own tile.
    pub const fn reach_texels(&self) -> u32 {
        (self.tiles_across / 2 - 1) * self.tile_texels
    }

    /// How many levels are needed to cover a raster of this size.
    ///
    /// The coarsest level has to reach the whole raster from wherever the
    /// camera is, otherwise there is ground with nothing to draw it and the
    /// horizon stops short of the data. That takes a square *twice* the raster,
    /// not one that merely spans it: every level is centred on the camera, so a
    /// level wide enough to cover the dataset covers only half of it from any
    /// given spot, and a camera at one edge has to see to the other.
    pub fn level_count(&self, raster: glam::UVec2, available: u32) -> u32 {
        let reach = f64::from(self.reach_texels()).max(1.0);
        let needed = f64::from(raster.max_element()) / reach;
        let levels = needed.log2().ceil().max(0.0) as u32 + 1;
        levels.clamp(1, available.max(1))
    }

    /// How many texels a ray may cross in total before the march gives up.
    pub const fn march_steps(&self, levels: u32) -> u32 {
        levels * self.march_texels
    }
}

/// The finest level worth keeping when the camera is `distance` metres above
/// the ground beneath it.
///
/// A level earns its residency while its texels are still smaller than the
/// pixels they land in. The nearest ground on screen is the ground directly
/// below, and that is where the demand is highest: a pixel there covers
/// `distance * pixel_angle` metres, and any level finer than that is detail
/// nothing can show. Everything below it is dropped, which saves the tiles that
/// would otherwise have to be read to fill it.
pub fn detail_base(
    residency: &Residency,
    metres_per_texel: f64,
    distance: f64,
    levels: u32,
) -> u32 {
    let coarsest = levels.saturating_sub(1);
    let resolvable = distance * residency.pixel_angle / metres_per_texel;
    let t = resolvable.max(1.0).log2().clamp(0.0, f64::from(coarsest));
    (t.floor() as u32).min(coarsest)
}

/// A tile of one level, by its index on the raster's tile grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Wanted {
    pub level: u32,
    pub tile: IVec2,
}

/// One level's square of resident tiles, and the step it is part way through.
#[derive(Clone, Debug)]
pub struct LevelResidency {
    /// North-west tile of the square that is fully loaded.
    origin: IVec2,
    /// Whether anything at all has been loaded yet.
    filled: bool,
    /// The step in flight: which way the square is moving, and which of the
    /// new edge's tiles are still to come.
    step: Option<Step>,
}

#[derive(Clone, Debug)]
struct Step {
    /// One of the four unit directions the square is moving in.
    towards: IVec2,
    /// Tiles of the new edge not yet uploaded, nearest the middle first.
    pending: Vec<IVec2>,
}

impl LevelResidency {
    fn new() -> Self {
        Self {
            origin: IVec2::ZERO,
            filled: false,
            step: None,
        }
    }

    /// The tiles a ray may read, as a half-open range of tile indices.
    ///
    /// One narrower than the square on the trailing side while a step is in
    /// flight, because the new edge is being written over the slots the
    /// trailing edge is using -- and empty altogether while a whole square is
    /// being filled, which has no trailing side to give up because every slot
    /// it will use is being written at once.
    pub fn valid(&self, tiles_across: u32) -> (IVec2, IVec2) {
        let across = tiles_across as i32;
        let (mut low, mut high) = (self.origin, self.origin + IVec2::splat(across));
        if let Some(step) = &self.step {
            if step.towards == IVec2::ZERO {
                // A whole square being filled at once, not an edge sliding by
                // one: nothing here is loaded, so nothing here may be read.
                // Every slot the square will use is being written, and until
                // the tile arrives the slot holds either an untouched texture
                // or a tile of somewhere else entirely -- the second of which
                // reads as real ground in the wrong place.
                return (IVec2::ZERO, IVec2::ZERO);
            }
            // Moving east writes the column at `origin.x + across`, whose slot
            // is the one `origin.x` is using; moving west writes `origin.x - 1`,
            // which is the slot of the last column.
            low += step.towards.max(IVec2::ZERO);
            high += step.towards.min(IVec2::ZERO);
        }
        (low, high)
    }
}

/// Which tiles every level holds, and the queue of what to load next.
#[derive(Clone, Debug)]
pub struct TileResidency {
    residency: Residency,
    levels: Vec<LevelResidency>,
}

impl TileResidency {
    pub fn new(residency: Residency, levels: u32) -> Self {
        Self {
            residency,
            levels: vec![LevelResidency::new(); levels as usize],
        }
    }

    pub fn level(&self, level: u32) -> &LevelResidency {
        &self.levels[level as usize]
    }

    /// Whether any level is part way through a step and short of tiles it wants.
    pub fn pending(&self) -> bool {
        self.levels.iter().any(|level| level.step.is_some())
    }

    /// The tile a level's square should start at, for a camera here.
    ///
    /// `camera` is in level-0 texels from the raster's origin. The square is
    /// centred on the tile the camera stands in, so it starts half a square
    /// back from it.
    fn wanted_origin(&self, level: u32, camera: DVec2) -> IVec2 {
        let texels = camera / f64::from(1u32 << level);
        let span = f64::from(self.residency.tile_texels);
        let tile = IVec2::new(
            (texels.x / span).floor() as i32,
            (texels.y / span).floor() as i32,
        );
        tile - IVec2::splat(self.residency.tiles_across as i32 / 2)
    }

    /// Advances every level towards where the camera now is, and returns the
    /// tiles to upload this update.
    ///
    /// `base` is the finest level worth keeping; below it nothing is loaded and
    /// whatever those squares hold is abandoned until the camera comes back
    /// down to them.
    pub fn advance(&mut self, camera: DVec2, base: u32) -> Vec<Wanted> {
        let across = self.residency.tiles_across as i32;
        let mut budget = self.residency.tiles_per_update;
        let mut work = Vec::new();

        // Coarsest first. A ray leaving a fine level hands over to the one
        // outside it, so the outer levels are the ones that must not be
        // missing, and they are also the ones that move least often.
        for level in (base..self.levels.len() as u32).rev() {
            let wanted = self.wanted_origin(level, camera);
            let state = &mut self.levels[level as usize];

            // Nothing shared with where it should be -- a first fill, or a jump
            // far enough that stepping there a tile at a time is pointless.
            let apart = (wanted - state.origin).abs().max_element();
            if !state.filled || apart >= across {
                state.origin = wanted;
                state.filled = true;
                state.step = Some(Step {
                    towards: IVec2::ZERO,
                    pending: square(wanted, across),
                });
            } else if state.step.is_none() && wanted != state.origin {
                // One tile at a time, along whichever axis is further out.
                let difference = wanted - state.origin;
                let towards = if difference.x.abs() >= difference.y.abs() {
                    IVec2::new(difference.x.signum(), 0)
                } else {
                    IVec2::new(0, difference.y.signum())
                };
                state.step = Some(Step {
                    towards,
                    pending: edge(state.origin, across, towards),
                });
            }

            // Hand out as much of the step as the budget allows. A step that
            // finishes moves the square; one that does not is picked up next
            // update, with the square still advertising itself narrower.
            if let Some(step) = &mut state.step {
                while budget > 0 {
                    let Some(tile) = step.pending.pop() else {
                        break;
                    };
                    work.push(Wanted { level, tile });
                    budget -= 1;
                }
                if step.pending.is_empty() {
                    state.origin += step.towards;
                    state.step = None;
                }
            }
        }
        work
    }
}

/// Every tile of a square, nearest the middle last so popping takes those
/// first.
fn square(origin: IVec2, across: i32) -> Vec<IVec2> {
    let middle = origin + IVec2::splat(across / 2);
    let mut tiles: Vec<IVec2> = (0..across)
        .flat_map(|y| (0..across).map(move |x| origin + IVec2::new(x, y)))
        .collect();
    tiles.sort_by_key(|tile| -(*tile - middle).abs().max_element());
    tiles
}

/// The edge a square about to move in `towards` has to load, nearest the middle
/// last.
fn edge(origin: IVec2, across: i32, towards: IVec2) -> Vec<IVec2> {
    // Moving east, the new column is one past the far side; moving west, it is
    // one before the near side.
    let along = |step: i32| {
        if step > 0 { across } else { -1 }
    };
    let mut tiles: Vec<IVec2> = (0..across)
        .map(|at| {
            if towards.x != 0 {
                origin + IVec2::new(along(towards.x), at)
            } else {
                origin + IVec2::new(at, along(towards.y))
            }
        })
        .collect();
    let middle = origin + IVec2::splat(across / 2);
    tiles.sort_by_key(|tile| -(*tile - middle).abs().max_element());
    tiles
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residency() -> Residency {
        Residency {
            tiles_across: 4,
            tiles_per_update: 64,
            ..Default::default()
        }
    }

    /// The whole clipmap shape hangs off how many bytes a texel costs: the
    /// raster this project flies takes 1280 MiB of the 1600 MiB budget at ten
    /// bytes, and fourteen would not fit. Overflowing it does not fail --
    /// [`Residency::fit_tiles`] quietly halves the square, and the finest
    /// level's reach falls from 1536 texels to 512 -- so the next byte anyone
    /// adds should fail here instead.
    #[test]
    fn eight_tiles_still_fit_the_budget() {
        let residency = Residency::default();
        // The installed download: 98304 by 114688 level-0 texels.
        let raster = glam::UVec2::new(98304, 114688);
        let available = 9;
        assert_eq!(
            residency.fit_tiles(raster, available),
            8,
            "{:.0} MiB of texture does not fit {:.0} MiB",
            residency.texture_bytes(residency.level_count(raster, available)) as f64
                / (1 << 20) as f64,
            residency.memory_budget as f64 / (1 << 20) as f64
        );
    }

    /// Texels within a tile of the camera are in the middle of the square, so a
    /// level reaches at least as far behind as ahead.
    #[test]
    fn a_square_is_centred_on_the_tile_the_camera_stands_in() {
        let mut tiles = TileResidency::new(residency(), 1);
        // Just inside tile 3 on both axes.
        let camera = DVec2::new(f64::from(TILE_SIZE) * 3.5, f64::from(TILE_SIZE) * 3.5);
        assert_eq!(tiles.wanted_origin(0, camera), IVec2::new(1, 1));

        tiles.advance(camera, 0);
        let (low, high) = tiles.level(0).valid(4);
        assert!(
            low.x <= 3 && high.x > 3,
            "the camera's own tile is resident"
        );
    }

    /// A level's texel index halves exactly into the level outside it, which is
    /// what lets a ray hand over without a correction term.
    #[test]
    fn a_coarse_square_covers_the_fine_one_it_wraps() {
        let tiles = TileResidency::new(residency(), 3);
        let camera = DVec2::new(9_000.0, 4_000.0);
        for level in 0..2u32 {
            let fine = tiles.wanted_origin(level, camera);
            let coarse = tiles.wanted_origin(level + 1, camera);
            // Both in level-0 texels, so the two ranges are comparable.
            let span = |tile: i32, level: u32| {
                f64::from(tile) * f64::from(TILE_SIZE) * f64::from(1u32 << level)
            };
            let fine_range = (span(fine.x, level), span(fine.x + 4, level));
            let coarse_range = (span(coarse.x, level + 1), span(coarse.x + 4, level + 1));
            assert!(
                coarse_range.0 <= fine_range.0 && coarse_range.1 >= fine_range.1,
                "level {} spans {fine_range:?} but level {} only {coarse_range:?}",
                level,
                level + 1
            );
        }
    }

    /// The first update loads everything; standing still after it loads nothing.
    #[test]
    fn a_camera_that_does_not_move_asks_for_nothing() {
        let mut tiles = TileResidency::new(residency(), 1);
        let camera = DVec2::new(5_000.0, 5_000.0);
        assert_eq!(tiles.advance(camera, 0).len(), 16, "a full square");
        assert!(tiles.advance(camera, 0).is_empty());
        // ... and nor does moving within the same tile.
        assert!(tiles.advance(camera + DVec2::new(10.0, 10.0), 0).is_empty());
    }

    /// Crossing a tile boundary costs one edge, not a square.
    #[test]
    fn crossing_a_boundary_loads_one_column() {
        let mut tiles = TileResidency::new(residency(), 1);
        let span = f64::from(TILE_SIZE);
        let camera = DVec2::new(span * 4.5, span * 4.5);
        tiles.advance(camera, 0);

        let work = tiles.advance(camera + DVec2::new(span, 0.0), 0);
        assert_eq!(work.len(), 4, "one column of a four-tile square");
        assert!(
            work.iter().all(|w| w.tile.x == work[0].tile.x),
            "the column should share an x: {work:?}"
        );
    }

    /// A square being filled outright holds nothing yet, and has to say so.
    ///
    /// Every slot the square will use is spoken for the moment the fill is
    /// queued, and until each tile lands its slot holds either a texture
    /// nothing has written or a tile of somewhere else entirely -- the second
    /// of which reads as real ground in the wrong place rather than as an
    /// obvious hole. Advertising the square before it is whole cost the march
    /// far more than the wrong pixels did: rays descended into a level with no
    /// data in it and crawled, taking 2746 steps each against the 91 the same
    /// camera costs once the fill has finished.
    ///
    /// The narrowing an edge step does is not enough here, because a fill has
    /// no direction to narrow along -- see [`LevelResidency::valid`].
    #[test]
    fn a_square_being_filled_advertises_nothing() {
        let residency = Residency {
            tiles_per_update: 1,
            ..residency()
        };
        let mut tiles = TileResidency::new(residency, 1);
        let span = f64::from(TILE_SIZE);
        let camera = DVec2::new(span * 4.5, span * 4.5);

        // Sixteen tiles at one an update, so fifteen updates leave it short.
        for handed in 1..16 {
            tiles.advance(camera, 0);
            let (low, high) = tiles.level(0).valid(4);
            assert_eq!(
                low, high,
                "{handed} of 16 tiles in, the square is still not readable"
            );
        }
        tiles.advance(camera, 0);
        let (low, high) = tiles.level(0).valid(4);
        assert_eq!(high - low, IVec2::splat(4), "whole once every tile is in");

        // And the same for a jump, which abandons a square that *was* whole:
        // its slots are being written over one at a time, so it stops being
        // readable the moment the refill is queued.
        tiles.advance(DVec2::new(span * 400.5, span * 400.5), 0);
        let (low, high) = tiles.level(0).valid(4);
        assert_eq!(low, high, "a refill is no more readable than a first fill");
    }

    /// While a step is in flight the square advertises itself narrower on the
    /// side whose slots are being overwritten, so nothing reads a half-written
    /// column.
    #[test]
    fn a_step_in_flight_gives_up_the_edge_it_is_overwriting() {
        let residency = Residency {
            tiles_per_update: 1,
            ..residency()
        };
        let mut tiles = TileResidency::new(residency, 1);
        let span = f64::from(TILE_SIZE);
        let camera = DVec2::new(span * 4.5, span * 4.5);
        // Sixteen tiles at one per update.
        for _ in 0..16 {
            tiles.advance(camera, 0);
        }
        let (whole_low, whole_high) = tiles.level(0).valid(4);

        // Step east, leaving three of the four new tiles outstanding.
        let moved = camera + DVec2::new(span, 0.0);
        tiles.advance(moved, 0);
        let (low, high) = tiles.level(0).valid(4);
        assert_eq!(
            (low.x, high.x),
            (whole_low.x + 1, whole_high.x),
            "the westmost column is the one being overwritten"
        );
        assert_eq!((low.y, high.y), (whole_low.y, whole_high.y));

        // Once the column lands the square is whole again, one tile further on.
        for _ in 0..3 {
            tiles.advance(moved, 0);
        }
        let (low, high) = tiles.level(0).valid(4);
        assert_eq!((low.x, high.x), (whole_low.x + 1, whole_high.x + 1));
    }

    /// A jump too far to walk to reloads outright rather than stepping there a
    /// tile at a time.
    #[test]
    fn a_long_jump_refills_instead_of_stepping() {
        let mut tiles = TileResidency::new(residency(), 1);
        let span = f64::from(TILE_SIZE);
        tiles.advance(DVec2::new(span * 4.5, span * 4.5), 0);

        let work = tiles.advance(DVec2::new(span * 400.5, span * 400.5), 0);
        assert_eq!(work.len(), 16, "the whole square, not one edge");
    }

    /// Levels below the base are not loaded at all: the saving that matters at
    /// altitude is the tiles, not the texels.
    #[test]
    fn levels_too_fine_to_draw_are_not_loaded() {
        let mut tiles = TileResidency::new(residency(), 3);
        let camera = DVec2::new(5_000.0, 5_000.0);
        let work = tiles.advance(camera, 1);
        assert!(
            work.iter().all(|wanted| wanted.level >= 1),
            "level 0 should not have been asked for"
        );
    }

    /// The budget is a ceiling on how much one update may cost.
    #[test]
    fn no_update_hands_out_more_than_its_budget() {
        let residency = Residency {
            tiles_per_update: 3,
            ..residency()
        };
        let mut tiles = TileResidency::new(residency, 3);
        let camera = DVec2::new(5_000.0, 5_000.0);
        for _ in 0..40 {
            assert!(tiles.advance(camera, 0).len() <= 3);
        }
    }
}
