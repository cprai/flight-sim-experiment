//! How much of the raster is resident, at what resolution, and what is made up.
//!
//! Two halves. The whole raster is resident from a base level upwards, as a mip
//! chain that never moves and never streams -- a camera crossing the world
//! costs no tile reads at all. Below the base there is no measured ground to
//! hold, so those levels are *generated* into a square of whole tiles around
//! the camera, which does move, a tile at a time, as it does.
//!
//! The window below is the clipmap the chain replaced, kept for exactly the
//! levels that still have to move, and filled by a compute dispatch rather than
//! by a disk read. That is the difference that makes it affordable: a tile is
//! generated once per texel, where a level a metre across cost 1 MB of I/O and
//! a per-row decode.
//!
//! ## What this replaced, and why
//!
//! Each level used to keep a square of whole tiles around the camera, swapped a
//! tile at a time as the camera moved. That bought a 1 m survey at a bounded
//! cost per frame, and it cost 117 GB on disk to hold a resolution only ever
//! read within 1.5 km of the camera -- 7.4 ms mean of tile reads and 3.3 ms of
//! conversion per frame while the camera moved, for detail that was thrown away
//! as soon as it was passed.
//!
//! Storing the survey coarse instead makes the whole thing fit. At 8 m this
//! raster is 12288 x 14336 texels, which is 1.8 GB across the three products
//! with a mip chain -- less ground truth, no streaming, and 60 times less disk.
//! What is lost below the base level is meant to be put back by generating it,
//! which is what makes trading it away reasonable.
//!
//! ## Addressing
//!
//! A resident level `l` is mip `l - base` of each texture, so a texel index *is*
//! its texture coordinate: its mask is all ones and `slot` is the identity. A
//! generated level is a layer of a window `detail_tiles` tiles square, a power
//! of two, so a texel at index `x` lives at `x & (N * tile - 1)` -- a pure
//! function of position, with no window origin to carry.
//!
//! Level `l`'s index is exactly half level `l - 1`'s either way, which is what
//! lets a ray hand over between levels with nothing to correct.
//!
//! ## Two limits decide the base level
//!
//! A texture may not exceed `max_texture_dimension_2d`, which is 16384 on the
//! hardware this was written against and 8192 by WebGPU's own default. That
//! alone rules out holding this raster at 4 m: it would be 24576 x 28672. And
//! the chain has to fit [`Residency::memory_budget`]. [`Residency::fit_base`]
//! coarsens the base until both hold, which is the same job
//! `fit_tiles` did for the square it replaced.

use glam::{DVec2, IVec2, UVec2};

/// Bytes one texel of the four resident products costs between them.
///
/// Heights are `R32Float`, ground cover `R16Uint` -- ids reach 0x080c, so 16
/// bits is ample and the other two were never used -- the max pyramid
/// `R16Float`, and the lift `R16Float` beside it. This is a *restatement* of
/// what `Terrain::new` allocates and nothing enforces the correspondence, which
/// has already gone wrong once: a fourth array was added there and not here, so
/// the clipmap really spent fourteen bytes a texel while this reported ten.
/// Anything added there must be added here.
pub const BYTES_PER_TEXEL: usize = 4 + 2 + 2 + 2;

/// How much of the raster is resident, and how the march is bounded.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Residency {
    /// The finest level held, as a level of the stored pyramid.
    ///
    /// The single knob that trades ground truth for memory. Three is 8 m texels
    /// on a 1 m survey. Coarsened by [`Residency::fit_base`] where the raster
    /// or the budget will not take it; never made finer, because the detail
    /// below it is meant to be generated rather than stored.
    pub resident_base: u32,
    /// How many bytes of texture the whole arrangement may occupy.
    pub memory_budget: usize,
    /// The angle one pixel of the target subtends, in radians.
    ///
    /// Decides the finest level worth descending to: a level earns a descent
    /// while its texels are still smaller than the pixels they land in. See
    /// [`detail_base`].
    pub pixel_angle: f64,
    /// How many texels of one level a ray may cross before the march gives up.
    ///
    /// A field rather than a constant so a test can starve it on purpose and
    /// see what the march does when it runs out.
    pub march_texels: u32,

    /// Tiles across one *generated* level's window. A power of two.
    ///
    /// Levels below the base are not held: there is no measured ground under
    /// them to hold. They are synthesised into a square of whole tiles around
    /// the camera, which moves as it does, and this is how far that square
    /// reaches -- eight tiles of 512 puts level zero at 1536 texels in every
    /// direction, which is what the streamed clipmap reached with the same
    /// arrangement.
    pub detail_tiles: u32,
    /// Side of one generated tile, in texels.
    ///
    /// The unit a window moves and regenerates in. Smaller spreads the cost of
    /// crossing a boundary over more updates and wastes more of each dispatch
    /// on its edges; settable so a test can work at a scale where a raster is
    /// more than a single tile across.
    pub detail_tile_texels: u32,
    /// Metres of relief a generated level adds where the ground is steepest.
    ///
    /// The whole fractal's amplitude, shared out over its octaves, so a level
    /// with one octave carries about two thirds of it and the finest carries
    /// nearly all. Two metres against an eight metre base is about what a box
    /// filter takes off a mountainside; a test sets it to zero to check that
    /// what is left underneath lines up with the base exactly.
    pub detail_relief: f32,
    /// How many tiles may be generated in one update.
    ///
    /// The bound on what crossing a tile boundary costs. A level that falls
    /// behind is not wrong, only coarser at its outer edge until it catches up.
    pub detail_per_update: u32,
}

/// The angle one pixel subtends, for a viewport of this height.
///
/// The horizontal angle is the same: widening the viewport widens the field of
/// view with it rather than stretching the picture, so pixels stay square.
pub fn pixel_angle(viewport_height: u32, fov_y: f64) -> f64 {
    2.0 * (fov_y * 0.5).tan() / f64::from(viewport_height.max(1))
}

impl Default for Residency {
    fn default() -> Self {
        Self {
            // 8 m on the survey this flies, which is the finest whole-raster
            // residency the 16384 texture limit allows for it.
            resident_base: 3,
            // Room for that: 1792 MiB of chain across the three products.
            memory_budget: 2560 << 20,
            // 1080p at sixty degrees, replaced wherever a real viewport is known.
            pixel_angle: 2.0 * (30f64.to_radians()).tan() / 1080.0,
            march_texels: 512,
            detail_tiles: 8,
            detail_tile_texels: 512,
            detail_relief: 2.0,
            detail_per_update: 4,
        }
    }
}

impl Residency {
    /// The size of the base level, in its own texels.
    pub fn base_size(&self, raster: UVec2) -> UVec2 {
        UVec2::new(
            (raster.x >> self.resident_base).max(1),
            (raster.y >> self.resident_base).max(1),
        )
    }

    /// How many mips of a texture this size still halve exactly.
    ///
    /// A max pyramid cell has to bound the four cells under it, and `>> m`
    /// rounding drops a row or column the moment a dimension goes odd -- so the
    /// cell above would bound three quarters of its ground and read as a bound
    /// while not being one. This stops one mip before that happens.
    ///
    /// For the raster this flies, 12288 is `3 * 2^12` and 14336 is `7 * 2^11`,
    /// so the height binds and the chain is twelve mips, ending at 6 x 7 texels
    /// of 16 km. That still covers the whole world in a single march step at the
    /// top, which is all the coarsest level is for.
    pub fn mip_count(size: UVec2) -> u32 {
        1 + size.x.trailing_zeros().min(size.y.trailing_zeros())
    }

    /// Bytes of texture a chain over `size` occupies.
    ///
    /// The chain is a little over four thirds of its base, exactly because each
    /// mip is a quarter of the one under it.
    pub fn texture_bytes(size: UVec2, mips: u32) -> usize {
        (0..mips)
            .map(|mip| {
                let level = UVec2::new((size.x >> mip).max(1), (size.y >> mip).max(1));
                (level.x as usize) * (level.y as usize) * BYTES_PER_TEXEL
            })
            .sum()
    }

    /// The finest base no finer than this one that the products, the device and
    /// the budget will all take.
    ///
    /// Coarsening once quarters the memory and halves each dimension, so this
    /// converges in a couple of steps from anywhere sensible. `available` is how
    /// many levels the source products actually hold: asking for a base past the
    /// end of the stored chain would leave nothing to read.
    ///
    /// `stored` is the other end of the same question, and it is the one thing
    /// here that fails *quietly* if it is not asked. A tile store answers a
    /// request below its own base by repeating texels out of the finest level it
    /// has, so a base two levels finer than what was written would fill sixteen
    /// times the memory with the same ground magnified -- no error, no hole,
    /// just a survey that is not there. Now that the tools write from
    /// [`terrain_tiles::RESIDENT_BASE_LEVEL`] up, that is the ordinary case
    /// rather than a misconfiguration.
    pub fn fit_base(
        &self,
        raster: UVec2,
        stored: u32,
        available: u32,
        max_dimension: u32,
    ) -> u32 {
        let mut base = self.resident_base.max(stored);
        while base + 1 < available {
            let size = Self {
                resident_base: base,
                ..*self
            }
            .base_size(raster);
            let fits_device = size.x <= max_dimension && size.y <= max_dimension;
            let fits_budget = Self::texture_bytes(size, Self::mip_count(size)) <= self.memory_budget;
            if fits_device && fits_budget {
                break;
            }
            base += 1;
        }
        base
    }

    /// How far a generated level reaches from the camera, in its own texels.
    ///
    /// The window is centred on the tile the camera stands in, so half of it
    /// lies behind whichever way the camera faces. This is the half that is
    /// guaranteed in every direction, which is one tile short of half the
    /// square: the camera may stand anywhere within its own tile.
    pub const fn detail_reach(&self) -> u32 {
        (self.detail_tiles / 2 - 1) * self.detail_tile_texels
    }

    /// How many texels a ray may cross in total before the march gives up.
    pub const fn march_steps(&self, levels: u32) -> u32 {
        levels * self.march_texels
    }
}

/// The finest level worth descending to when the camera is `distance` metres
/// above the ground beneath it.
///
/// A level earns a descent while its texels are still smaller than the pixels
/// they land in. The nearest ground on screen is the ground directly below, and
/// that is where the demand is highest: a pixel there covers
/// `distance * pixel_angle` metres, and any level finer than that is detail
/// nothing can show. Everything below it is a descent that costs steps and
/// changes no pixel.
///
/// This used to decide what was *loaded*, which was the larger saving while
/// levels came off disk. Nothing is loaded any more, so what it saves now is
/// march steps -- and it is what will decide how many levels are worth
/// generating once anything is generated.
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

/// A tile of one generated level, by its index on the raster's tile grid.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Wanted {
    pub level: u32,
    pub tile: IVec2,
}

/// One generated level's square of tiles, and the step it is part way through.
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
    ///
    /// A ray outside this falls to the level beyond, which for the coarsest
    /// generated level is the resident chain: there is always something to
    /// fall to, so a window part way through a step costs detail rather than
    /// correctness.
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
        let span = f64::from(self.residency.detail_tile_texels);
        let tile = IVec2::new(
            (texels.x / span).floor() as i32,
            (texels.y / span).floor() as i32,
        );
        tile - IVec2::splat(self.residency.detail_tiles as i32 / 2)
    }

    /// Advances every level towards where the camera now is, and returns the
    /// tiles to generate this update.
    ///
    /// `base` is the finest level worth drawing; below it nothing is generated
    /// and whatever those squares hold is abandoned until the camera comes back
    /// down to them.
    pub fn advance(&mut self, camera: DVec2, base: u32) -> Vec<Wanted> {
        let across = self.residency.detail_tiles as i32;
        let mut budget = self.residency.detail_per_update;
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

    /// The installed download: 98304 by 114688 level-0 texels at 1 m.
    const RASTER: UVec2 = UVec2::new(98304, 114688);

    /// A chain whose coarsest mip has lost a row bounds three quarters of the
    /// ground it claims, which is the one failure the max pyramid exists to
    /// prevent -- and it fails silently, as scattered holes in the far field.
    #[test]
    fn a_chain_stops_before_a_mip_would_drop_a_row() {
        let size = UVec2::new(12288, 14336);
        let mips = Residency::mip_count(size);
        assert_eq!(mips, 12, "12288 is 3 << 12 and 14336 is 7 << 11");
        for mip in 0..mips - 1 {
            let level = UVec2::new(size.x >> mip, size.y >> mip);
            assert!(
                level.x.is_multiple_of(2) && level.y.is_multiple_of(2),
                "mip {mip} is {level} and does not halve evenly"
            );
        }
    }

    /// The base level is chosen by two hard limits, and getting either wrong is
    /// a device error at startup rather than a picture that looks wrong.
    ///
    /// Four metres would be 24576 x 28672, past every texture limit there is;
    /// eight is 12288 x 14336, which the 16384 this was written against takes
    /// and WebGPU's own default 8192 does not.
    #[test]
    fn the_base_coarsens_until_the_device_will_take_it() {
        let residency = Residency {
            resident_base: 2,
            ..Default::default()
        };
        assert_eq!(residency.fit_base(RASTER, 0, 9, 16384), 3, "4 m is too wide");
        assert_eq!(
            residency.fit_base(RASTER, 5, 9, 16384),
            5,
            "a base finer than the products hold would be magnified, not read"
        );
        assert_eq!(
            residency.fit_base(RASTER, 0, 9, 8192),
            4,
            "at the WebGPU default even 8 m is too wide"
        );
    }

    /// The tools store from one level and the renderer holds from another, and
    /// the two numbers are written down in different crates. This is what says
    /// they are the same number.
    ///
    /// Getting it wrong is quiet in both directions. Stored coarser than held
    /// and the base is magnified texels rather than a survey; stored finer and
    /// the extra levels are gigabytes nothing opens.
    #[test]
    fn the_stored_base_is_the_one_the_renderer_holds() {
        assert_eq!(
            Residency::default().resident_base,
            terrain_tiles::RESIDENT_BASE_LEVEL
        );
    }

    /// The budget is the other limit, and this is the number that says whether
    /// the default arrangement fits. A byte added to [`BYTES_PER_TEXEL`] that
    /// pushes it over does not fail -- [`Residency::fit_base`] quietly coarsens
    /// the base and the survey loses half its resolution in each axis -- so it
    /// should fail here instead.
    #[test]
    fn the_default_base_fits_the_default_budget() {
        let residency = Residency::default();
        let size = residency.base_size(RASTER);
        let bytes = Residency::texture_bytes(size, Residency::mip_count(size));
        assert_eq!(size, UVec2::new(12288, 14336));
        assert!(
            bytes <= residency.memory_budget,
            "{:.0} MiB of chain does not fit {:.0} MiB",
            bytes as f64 / (1 << 20) as f64,
            residency.memory_budget as f64 / (1 << 20) as f64
        );
        assert_eq!(residency.fit_base(RASTER, 0, 9, 16384), 3);
    }

    /// A chain is four thirds of its base and no more, which is what makes
    /// holding one affordable at all.
    #[test]
    fn a_chain_costs_a_third_more_than_its_base() {
        let size = UVec2::new(12288, 14336);
        let base = (size.x as usize) * (size.y as usize) * BYTES_PER_TEXEL;
        let whole = Residency::texture_bytes(size, Residency::mip_count(size));
        assert!(whole > base && whole < base * 4 / 3 + base / 100);
    }

    /// Ground a pixel cannot resolve is not worth descending to. The handover
    /// doubles with height, so this is a log and the numbers either side of a
    /// boundary are what pin it.
    #[test]
    fn a_level_is_given_up_when_its_texels_fall_under_a_pixel() {
        let residency = Residency {
            pixel_angle: pixel_angle(1080, 60f64.to_radians()),
            ..Default::default()
        };
        // Eight metre texels: a pixel covers one of them at about 7.5 km.
        let at = |distance| detail_base(&residency, 8.0, distance, 12);
        assert_eq!(at(100.0), 0, "close in, every level is worth having");
        assert_eq!(at(20_000.0), 1);
        assert_eq!(at(40_000.0), 2);
        assert!(at(1.0e9) <= 11, "clamped to the levels that exist");
    }
}
