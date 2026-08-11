//! Hydraulic erosion: the pass that turns fractal ridges into a landscape.
//!
//! Everything recognisable about mountain country is the work of water. Ridges
//! are sharp because the valleys either side were cut, not because the ridge was
//! raised; valleys are V-shaped near their heads and flat-floored near their
//! mouths; every slope drains somewhere, and the somewheres join into a tree.
//! None of that is a property of noise, and no amount of octaves produces it.
//!
//! The model is the particle one: a droplet lands, runs downhill carrying
//! sediment, picks up more where it is moving fast down a steep slope and drops
//! it where it slows, and evaporates. Millions of them, and the drainage network
//! is not designed, it emerges.
//!
//! # Why this pass is tiled rather than sequential
//!
//! A droplet's path depends on the ground the droplet before it left behind.
//! That dependency *is* the algorithm -- it is what makes a channel deepen
//! instead of leaving a million independent scratches -- so for a long time
//! this pass ran one droplet after another on one core, at 216 s of a 713 s
//! run, on the reasoning that sharing cells between threads means either a lock
//! per cell or a race, and a race would make the landscape depend on how the
//! threads interleaved.
//!
//! Both horns of that were avoidable, because the dependency is *local*. A
//! droplet takes at most [`MAX_STEPS`] steps of one cell and spreads its work
//! over a disc of [`BRUSH_RADIUS`], so it can only ever touch ground within
//! [`HALO`] cells of where it landed. Two droplets further apart than that
//! cannot interact however they are ordered -- not "rarely", but never.
//!
//! So the map is cut into tiles of [`TILE`] cells, and tiles far enough apart
//! are run at once. Inside a tile the droplets still run one after another over
//! ground nobody else is touching, so the feedback that makes channels is kept
//! exactly; across tiles there is nothing to keep, because there is no
//! interaction to lose. [`SPACING`] is what keeps concurrent tiles apart, and
//! `the_tiles_do_not_race_each_other` is what checks it.
//!
//! That took the pass from 216.1 s to 6.3 s on 24 cores -- rather more than the
//! core count, because a thread working one 256-cell tile stays inside the
//! cache, where the old single stream jumped about a 44 MB grid at random.
//!
//! The landscape this produces is not the one the single stream produced: the
//! droplets land in a different order and in different places, since each tile
//! draws its own from its own key. It is the same landscape in every sense that
//! was ever specified -- same model, same constants, same statistics -- and the
//! same seed still reproduces it exactly, which is the property that was
//! actually being protected.

use rayon::prelude::*;

use crate::fields::Fields;
use crate::noise::hash;

/// How far a droplet is followed before it is abandoned.
///
/// A droplet that has not evaporated or run off the map in this many steps is
/// circling in a pit, and following it further only polishes that pit.
const MAX_STEPS: u32 = 64;

/// How much of its previous direction a droplet keeps.
///
/// Small, so a droplet mostly follows the ground. What little it keeps is what
/// stops it from turning on a sixpence at every cell boundary, which draws
/// channels that follow the grid axes.
const INERTIA: f32 = 0.06;

/// How much sediment a droplet can hold, per unit of slope, speed and water.
const CAPACITY: f32 = 4.0;

/// The slope a droplet is treated as being on when it is on none.
///
/// Without a floor, a droplet crossing flat ground has zero capacity and drops
/// everything it is carrying in one cell, which builds a wall across every
/// valley floor.
const MIN_SLOPE: f32 = 0.012;

/// The steepest slope a droplet's capacity is allowed to be reckoned from.
///
/// Two, which is about 63 degrees and steeper than any slope the thermal pass
/// leaves standing. This is the first of the three bounds that make the pass
/// stable, and it is the important one -- see [`erode`].
const MAX_SLOPE: f32 = 2.0;

/// The fastest a droplet is allowed to get, in the same units as the slope.
///
/// Water reaches a terminal velocity; a droplet integrating `v^2 += g h` over a
/// kilometre of descent does not.
const MAX_SPEED: f32 = 8.0;

/// The most ground one droplet may move in one step, as a fraction of a cell.
///
/// Five hundredths of a cell -- 0.8 m on the default grid -- which is about
/// twice what a droplet on ordinary ground moves anyway, so it changes nothing
/// about how the landscape is carved and everything about whether the carving
/// terminates.
const MAX_MOVE_SHARE: f32 = 0.05;

/// How quickly a droplet takes up its shortfall, and gives up its excess.
const ERODE_RATE: f32 = 0.35;
const DEPOSIT_RATE: f32 = 0.3;

/// How much of its water a droplet loses per step.
const EVAPORATION: f32 = 0.02;

/// How much speed a droplet gains from a metre of descent.
const GRAVITY: f32 = 0.35;

/// How far, in cells, a droplet's cutting is spread.
///
/// Cutting only the cell a droplet is in digs a one-cell slot with vertical
/// walls, which thermal erosion then has to undo. Spreading it over a small
/// disc cuts a channel with banks, which is what a channel is.
const BRUSH_RADIUS: i64 = 2;

/// How much of the cutting hard rock refuses.
const HARDNESS_RESISTANCE: f32 = 0.75;

/// A `0..1` from a hashed word, using the twenty-four bits an `f32` can hold.
fn unit(bits: u32) -> f32 {
    (bits >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// The disc a droplet's cutting is spread over: offsets and their weights,
/// summing to one.
fn brush() -> Vec<(i64, i64, f32)> {
    let mut cells = Vec::new();
    let mut total = 0.0;
    for dy in -BRUSH_RADIUS..=BRUSH_RADIUS {
        for dx in -BRUSH_RADIUS..=BRUSH_RADIUS {
            let distance = ((dx * dx + dy * dy) as f32).sqrt();
            if distance > BRUSH_RADIUS as f32 {
                continue;
            }
            let weight = 1.0 - distance / (BRUSH_RADIUS as f32 + 1.0);
            cells.push((dx, dy, weight));
            total += weight;
        }
    }
    for cell in &mut cells {
        cell.2 /= total;
    }
    cells
}

/// The ground, while the droplets are running over it.
///
/// Held as bits in atomics rather than as plain floats for one reason: the
/// tiles below run on different threads and Rust cannot be told that their
/// footprints do not overlap. They genuinely do not -- see [`HALO`] -- so no
/// two threads ever touch the same cell and there is nothing to contend for.
/// The atomics buy the compiler's permission, not synchronisation, which is why
/// every access is `Relaxed`: on any machine this runs on, an aligned
/// four-byte relaxed load or store is the same instruction a plain one would
/// have been.
type Ground = [std::sync::atomic::AtomicU32];

fn read(ground: &Ground, index: usize) -> f32 {
    f32::from_bits(ground[index].load(std::sync::atomic::Ordering::Relaxed))
}

fn raise(ground: &Ground, index: usize, delta: f32) {
    let was = read(ground, index);
    ground[index].store(
        (was + delta).to_bits(),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The bilinear height at a position in cells, and its gradient in metres per
/// cell.
fn surface(heights: &Ground, width: usize, rows: usize, x: f32, y: f32) -> (f32, f32, f32) {
    let (cx, cy) = (x.floor() as i64, y.floor() as i64);
    let (fx, fy) = (x - cx as f32, y - cy as f32);
    let at = |dx: i64, dy: i64| {
        let column = (cx + dx).clamp(0, width as i64 - 1) as usize;
        let row = (cy + dy).clamp(0, rows as i64 - 1) as usize;
        read(heights, row * width + column)
    };
    let (h00, h10, h01, h11) = (at(0, 0), at(1, 0), at(0, 1), at(1, 1));
    let height = h00 * (1.0 - fx) * (1.0 - fy)
        + h10 * fx * (1.0 - fy)
        + h01 * (1.0 - fx) * fy
        + h11 * fx * fy;
    let gradient_x = (h10 - h00) * (1.0 - fy) + (h11 - h01) * fy;
    let gradient_y = (h01 - h00) * (1.0 - fx) + (h11 - h10) * fx;
    (height, gradient_x, gradient_y)
}

/// Runs the droplets, cutting the height channel and recording what they left
/// in the deposit channel.
///
/// # The three bounds, and why the pass needs them
///
/// A droplet's cutting is spread over a small disc, and that disc reaches the
/// cell the droplet is about to step into. So a cell that has been cut a little
/// deeper than its neighbours presents a bigger drop to the next droplet that
/// crosses it, which gives that droplet more capacity, which cuts it deeper
/// still. Written without bounds the loop is not merely unstable, it is
/// *exponential*: capacity grows with the slope, and the speed term grows with
/// the square root of the fall on top of that, so past a few hundred metres of
/// depth each visit more than doubles it.
///
/// It is also invisible at small scales. A pit needs to be visited many times
/// to run away, so a grid of a million cells finishes with a plausible
/// landscape and a grid of ten million comes back with a single cell eight
/// thousand kilometres below the sea -- which then presses the whole landscape
/// into the top of its range when it is rescaled, and paints the lot as glacier.
/// The three bounds below are what stop it, and [`MAX_MOVE_SHARE`] is the one
/// that actually closes the loop; the other two only keep the arithmetic
/// sensible.
///
/// None of them changes what the pass does to ordinary ground. A droplet on a
/// one-in-three slope moves about four tenths of a metre in a step, against a
/// cap of eight tenths.
pub fn erode(fields: &mut Fields, seed: u32, per_cell: usize) {
    let metres_per_cell = fields.metres_per_cell;
    let Fields {
        height,
        hardness,
        deposit,
        ..
    } = fields;
    let (width, rows) = (height.width, height.height);
    let rock: &[f32] = &hardness.values;
    let brush = brush();

    let droplets = per_cell * width * rows;
    if droplets == 0 {
        return;
    }
    log::info!("running {droplets} droplets over {width} x {rows} cells");

    // What the ground looked like before, so the deposit channel can be taken
    // as the difference afterwards. Both branches below already move the two
    // channels by exactly the same amount, so carrying a second accumulator
    // through the pass would be carrying the same numbers twice.
    let before = height.values.clone();
    let ground: Vec<std::sync::atomic::AtomicU32> = height
        .values
        .iter()
        .map(|metres| std::sync::atomic::AtomicU32::new(metres.to_bits()))
        .collect();

    for colour in 0..COLOURS {
        let tiles: Vec<(usize, usize)> = (0..rows.div_ceil(TILE))
            .flat_map(|ty| (0..width.div_ceil(TILE)).map(move |tx| (tx, ty)))
            .filter(|(tx, ty)| (tx % SPACING) + SPACING * (ty % SPACING) == colour)
            .collect();
        tiles.par_iter().for_each(|&(tx, ty)| {
            run_tile(
                &ground, rock, &brush, width, rows, metres_per_cell, per_cell, seed, tx, ty,
            );
        });
        log::info!(
            "{}% of the droplets have run",
            (colour + 1) * 100 / COLOURS
        );
    }

    for (index, cell) in ground.iter().enumerate() {
        let after = f32::from_bits(cell.load(std::sync::atomic::Ordering::Relaxed));
        deposit.values[index] += after - before[index];
        height.values[index] = after;
    }
}

/// Cells across the square of ground one thread owns.
///
/// Only the inner square is *seeded* with droplets; a droplet may wander up to
/// [`HALO`] cells beyond it, which is why tiles have to be kept apart rather
/// than merely tiled.
const TILE: usize = 256;

/// How far outside its tile a droplet's writing can reach.
///
/// A droplet takes at most [`MAX_STEPS`] steps of one cell, spreads its cutting
/// over a disc of [`BRUSH_RADIUS`], and its deposits land on the four cells it
/// stands between. Sixty-seven would do; this leaves room for all three to grow
/// a little without silently becoming a data race.
const HALO: usize = 80;

/// How far apart, in tiles, two tiles running at once have to be.
///
/// A tile writes over `TILE + 2 * HALO` cells, so tiles two apart -- 512 cells
/// between their origins against 416 of reach -- cannot overlap. Two on each
/// axis makes four passes, and no two tiles in one pass share a cell.
const SPACING: usize = 2;
const COLOURS: usize = SPACING * SPACING;
const _: () = assert!(TILE * SPACING >= TILE + 2 * HALO);

/// Runs one tile's droplets, in order, over ground nothing else is touching.
#[allow(clippy::too_many_arguments)]
fn run_tile(
    ground: &Ground,
    rock: &[f32],
    brush: &[(i64, i64, f32)],
    width: usize,
    rows: usize,
    metres_per_cell: f32,
    per_cell: usize,
    seed: u32,
    tx: usize,
    ty: usize,
) {
    let (x0, y0) = (tx * TILE, ty * TILE);
    let (tile_width, tile_rows) = (TILE.min(width - x0), TILE.min(rows - y0));
    // As many droplets as this tile's own ground earns, so the total over the
    // map is what a single stream would have been however the tiles fall.
    let droplets = per_cell * tile_width * tile_rows;
    // Its own stream, keyed by where it is rather than by a running count, so
    // that a tile lands the same droplets whatever else is being run beside it.
    let tile_seed = hash(tx as i32, ty as i32, seed);

    for droplet in 0..droplets {
        let word = hash(droplet as i32, 0, tile_seed);
        let other = hash(droplet as i32, 1, tile_seed);
        let mut x = x0 as f32 + unit(word) * (tile_width - 1) as f32;
        let mut y = y0 as f32 + unit(other) * (tile_rows - 1) as f32;
        let (mut dx, mut dy) = (0.0f32, 0.0f32);
        let (mut speed, mut water, mut sediment) = (1.0f32, 1.0f32, 0.0f32);

        for _ in 0..MAX_STEPS {
            let (was, gradient_x, gradient_y) = surface(ground, width, rows, x, y);
            let (cell_x, cell_y) = (x.floor() as i64, y.floor() as i64);
            let (fx, fy) = (x - cell_x as f32, y - cell_y as f32);

            dx = dx * INERTIA - gradient_x * (1.0 - INERTIA);
            dy = dy * INERTIA - gradient_y * (1.0 - INERTIA);
            let length = (dx * dx + dy * dy).sqrt();
            if !length.is_finite() || length <= 1e-6 {
                break;
            }
            (dx, dy) = (dx / length, dy / length);

            let (nx, ny) = (x + dx, y + dy);
            if nx < 0.0 || ny < 0.0 || nx > (width - 1) as f32 || ny > (rows - 1) as f32 {
                break;
            }
            let (now, _, _) = surface(ground, width, rows, nx, ny);
            let fall = was - now;

            // Slope as a rise over run, so the constants above mean the same
            // thing whatever `--sim-metres` the grid is at, and clamped so that
            // a drop into a hole cannot reckon itself a bigger capacity than a
            // slope the landscape could actually hold.
            let slope = (fall / metres_per_cell).clamp(-MAX_SLOPE, MAX_SLOPE);
            let capacity = slope.max(MIN_SLOPE) * speed * water * CAPACITY;
            let most = MAX_MOVE_SHARE * metres_per_cell;

            if sediment > capacity || fall < 0.0 {
                // Uphill, or carrying more than it can: drop the difference
                // where it is, spread over the four cells it stands between.
                // Going uphill it may fill the step it is climbing, but no
                // more -- a droplet cannot build a hill.
                let dropped = if fall < 0.0 {
                    sediment.min(-fall)
                } else {
                    (sediment - capacity) * DEPOSIT_RATE
                }
                .min(most);
                sediment -= dropped;
                let place = |dx: i64, dy: i64, share: f32| {
                    let column = (cell_x + dx).clamp(0, width as i64 - 1) as usize;
                    let row = (cell_y + dy).clamp(0, rows as i64 - 1) as usize;
                    raise(ground, row * width + column, dropped * share);
                };
                place(0, 0, (1.0 - fx) * (1.0 - fy));
                place(1, 0, fx * (1.0 - fy));
                place(0, 1, (1.0 - fx) * fy);
                place(1, 1, fx * fy);
            } else {
                // Room to spare: cut, but never more than the step it is
                // descending, or the droplet would dig itself a pit to sit in.
                let column = cell_x.clamp(0, width as i64 - 1) as usize;
                let row = cell_y.clamp(0, rows as i64 - 1) as usize;
                let resistance = 1.0 - HARDNESS_RESISTANCE * rock[row * width + column];
                // Bounded first, resisted second. The other way round the cap
                // would swallow the hardness whenever it bound -- which on
                // steep ground is most of the time -- and the cliff bands the
                // hardness field exists to draw would quietly stop forming.
                let cut = ((capacity - sediment) * ERODE_RATE)
                    .min(fall.max(0.0))
                    .min(most)
                    * resistance;
                sediment += cut;
                for (dx, dy, share) in brush {
                    let column = (cell_x + dx).clamp(0, width as i64 - 1) as usize;
                    let row = (cell_y + dy).clamp(0, rows as i64 - 1) as usize;
                    raise(ground, row * width + column, -cut * share);
                }
            }

            speed = (speed * speed + fall * GRAVITY)
                .max(0.0)
                .sqrt()
                .min(MAX_SPEED);
            water *= 1.0 - EVAPORATION;
            if water < 0.01 {
                break;
            }
            (x, y) = (nx, ny);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cone, which is the smallest landscape with somewhere for water to go.
    fn cone(side: usize, metres_per_cell: f32) -> Fields {
        let mut fields = Fields::new(
            [
                (side - 1) as f32 * metres_per_cell,
                (side - 1) as f32 * metres_per_cell,
            ],
            metres_per_cell,
        );
        let middle = (side - 1) as f32 * 0.5;
        for row in 0..side {
            for column in 0..side {
                let (dx, dy) = (column as f32 - middle, row as f32 - middle);
                let distance = (dx * dx + dy * dy).sqrt();
                let index = fields.height.index(column, row);
                // A little noise on the flank, so there is something for a
                // channel to start from; a perfectly smooth cone erodes into a
                // perfectly smooth cone.
                fields.height.values[index] =
                    1000.0 - distance * 12.0 + ((column * 7 + row * 13) % 11) as f32 * 0.4;
            }
        }
        fields
    }

    /// The seed is the reproducibility contract for the whole crate, and this
    /// is the one pass whose order could quietly break it.
    #[test]
    fn the_same_seed_erodes_the_same_landscape() {
        let mut first = cone(48, 10.0);
        let mut second = cone(48, 10.0);
        erode(&mut first, 4321, 3);
        erode(&mut second, 4321, 3);
        assert_eq!(first.height.values, second.height.values);
        assert_eq!(first.deposit.values, second.deposit.values);
    }

    /// What the droplets cost on the ground a real run covers.
    ///
    /// The pass was 216.1 s of a 712.7 s run when it was one droplet after
    /// another on one core. Run it with `--ignored --nocapture`.
    ///
    /// `DROPLET_MEASURE_ROUNDS` decides how incised the landscape is first, and
    /// it matters far more than it looks: droplets on raw fractal ground die in
    /// a step or two, and droplets in a cut drainage network run their full
    /// length. A measurement taken at zero rounds says almost nothing about a
    /// real run, which reaches this pass after eighty.
    #[test]
    #[ignore = "a measurement on the full grid, not a check"]
    fn measure_what_the_droplets_cost() {
        let rounds: u32 = std::env::var("DROPLET_MEASURE_ROUNDS")
            .ok()
            .and_then(|rounds| rounds.parse().ok())
            .unwrap_or(80);

        let mut fields = Fields::new([49152.0, 57344.0], 16.0);
        crate::shape::raise(
            &mut fields,
            crate::shape::Relief {
                valley_metres: 700.0,
                peak_metres: 2600.0,
            },
            0,
        );
        crate::thermal::relax(&mut fields, crate::thermal::Settling::Bedrock);
        if rounds > 0 {
            let at = std::time::Instant::now();
            crate::incise::rivers(&mut fields, rounds);
            crate::thermal::relax(&mut fields, crate::thermal::Settling::Bedrock);
            println!("cut {rounds} rounds of valleys first, in {:.1?}", at.elapsed());
        }

        let at = std::time::Instant::now();
        erode(&mut fields, 0, 3);
        println!(
            "the droplets took {:.1?} over {} cells on {} threads",
            at.elapsed(),
            fields.height.values.len(),
            rayon::current_num_threads(),
        );
    }

    /// The same, over enough ground to be more than one tile.
    ///
    /// Every other test here runs on 48 cells, which is one tile and one
    /// thread: none of them can see the tiling at all. This one spans several
    /// tiles on both axes, so the droplets really are running on different
    /// threads over ground that meets. If two tiles running at once could touch
    /// the same cell, the result would depend on how the threads interleaved
    /// and these runs would drift apart -- which is what [`SPACING`] exists to
    /// make impossible, and this is the only thing that checks it.
    ///
    /// Repeated rather than paired: a race that lands the same way twice proves
    /// nothing, and the failure it is looking for is intermittent by nature.
    #[test]
    fn the_tiles_do_not_race_each_other() {
        let side = TILE * 2 + 91;
        let mut first = cone(side, 10.0);
        erode(&mut first, 4321, 1);
        for again in 0..3 {
            let mut next = cone(side, 10.0);
            erode(&mut next, 4321, 1);
            let differing = first
                .height
                .values
                .iter()
                .zip(&next.height.values)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(differing, 0, "run {again} differed in {differing} cells");
        }
    }

    /// Droplets have to land on every part of the map, seams included.
    ///
    /// The tiles are seeded only in their own inner square and the last one on
    /// each axis is a partial tile, so an off-by-one in either would leave a
    /// stripe of ground no droplet ever starts on -- which would draw as a band
    /// of unweathered terrain and pass every other test in this file.
    #[test]
    fn every_tile_of_the_map_is_eroded() {
        // The remainder past the last whole tile has to be wider than `HALO`.
        // Narrower than that and droplets from the tile beside it wander across
        // the whole band, so a band nobody seeds still comes back eroded and
        // this test cannot tell the two apart -- which it could not, at 91.
        let side = TILE * 2 + HALO * 3;
        let before = cone(side, 10.0);
        let mut after = cone(side, 10.0);
        erode(&mut after, 99, 1);

        let width = before.height.width;
        let rows = before.height.height;
        let changed = |cells: &mut dyn Iterator<Item = usize>| {
            let (mut touched, mut total) = (0usize, 0usize);
            for index in cells {
                total += 1;
                if before.height.values[index] != after.height.values[index] {
                    touched += 1;
                }
            }
            touched as f64 / total as f64
        };

        // As a *share* of each band rather than a count, and compared against
        // the other bands rather than against a number picked in advance. A
        // band nobody seeds is not empty -- droplets from the tile beside it
        // wander up to `HALO` cells in, which is enough to pass any threshold
        // low enough to be safe -- so the only honest test is that every band
        // was worked about as hard as the busiest one.
        let mut shares = Vec::new();
        for tx in 0..width.div_ceil(TILE) {
            let columns = tx * TILE..(tx * TILE + TILE).min(width);
            shares.push((
                format!("column {tx}"),
                changed(&mut (0..rows).flat_map(|row| {
                    columns.clone().map(move |column| row * width + column)
                })),
            ));
        }
        for ty in 0..rows.div_ceil(TILE) {
            let band = ty * TILE..(ty * TILE + TILE).min(rows);
            shares.push((
                format!("row {ty}"),
                changed(&mut band.flat_map(|row| (0..width).map(move |column| row * width + column))),
            ));
        }

        let busiest = shares.iter().map(|(_, share)| *share).fold(0.0, f64::max);
        for (band, share) in &shares {
            assert!(
                *share > busiest * 0.5,
                "{band} had {:.1}% of its cells eroded against {:.1}% for the busiest band",
                share * 100.0,
                busiest * 100.0,
            );
        }
    }

    #[test]
    fn a_different_seed_erodes_it_differently() {
        let mut first = cone(48, 10.0);
        let mut second = cone(48, 10.0);
        erode(&mut first, 1, 3);
        erode(&mut second, 2, 3);
        assert_ne!(first.height.values, second.height.values);
    }

    /// Water cuts. If a run came back with the landscape no rougher than it
    /// started, the droplets are not reaching the ground.
    #[test]
    fn the_droplets_cut_the_slope_they_run_down() {
        let before = cone(48, 10.0);
        let mut after = cone(48, 10.0);
        erode(&mut after, 77, 3);

        let moved = before
            .height
            .values
            .iter()
            .zip(&after.height.values)
            .filter(|(was, now)| (*was - *now).abs() > 0.05)
            .count();
        assert!(
            moved > before.height.values.len() / 10,
            "only {moved} of {} cells moved",
            before.height.values.len()
        );
    }

    /// The deposit channel is what the material classifier reads to tell cut
    /// ground from filled ground, so it has to agree with the height the
    /// droplets actually left: everything they took off one place they put on
    /// another, minus whatever ran off the edge of the map.
    ///
    /// Agreement to a millimetre in ten, rather than exactly. The two channels
    /// take the identical increments but hold them at very different
    /// magnitudes -- a height near a thousand metres, a deposit near zero --
    /// so a run of hundreds of `f32` additions rounds them apart by a little,
    /// and it is the height that loses the bits.
    #[test]
    fn the_deposit_channel_tracks_the_height_it_changed() {
        let before = cone(48, 10.0);
        let mut after = cone(48, 10.0);
        erode(&mut after, 91, 3);

        for (index, deposit) in after.deposit.values.iter().enumerate() {
            let moved = after.height.values[index] - before.height.values[index];
            assert!(
                (moved - deposit).abs() < 0.001 * moved.abs().max(1.0),
                "cell {index} moved {moved} m but recorded {deposit} m"
            );
        }
    }

    /// A droplet must not be able to raise ground above what was there: it
    /// carries what it cut, and no more. A landscape that grew would mean the
    /// deposition step was inventing material.
    #[test]
    fn erosion_does_not_raise_the_highest_ground() {
        let before = cone(48, 10.0);
        let mut after = cone(48, 10.0);
        erode(&mut after, 55, 3);
        let (_, was) = before.height.range();
        let (_, now) = after.height.range();
        assert!(now <= was + 1e-3, "the peak rose from {was} to {now}");
    }

    /// The bug this pass was first shipped with, pinned.
    ///
    /// A droplet's cutting reaches the cell it is about to step into, so a cell
    /// cut slightly deep offers the next droplet a bigger drop, more capacity,
    /// and a deeper cut again. Unbounded, that feedback is exponential and a
    /// long enough run leaves one cell thousands of kilometres down -- which
    /// then squeezes the whole landscape into the top of its range when it is
    /// rescaled, and paints every texel of it as glacier.
    ///
    /// The pit is dug deliberately here rather than waited for, because waiting
    /// for it needs a grid of ten million cells and several minutes: the
    /// original bug passed every test in this module.
    #[test]
    fn a_pit_does_not_dig_itself_deeper_without_end() {
        let mut fields = cone(48, 10.0);
        let start = fields.height.range().0;
        // A hole a good deal deeper than any step of the cone, in the middle of
        // the flank where the droplets run.
        for row in 30..34 {
            for column in 30..34 {
                let index = fields.height.index(column, row);
                fields.height.values[index] -= 120.0;
            }
        }
        let dug = fields.height.range().0;
        erode(&mut fields, 8, 3);

        let after = fields.height.range().0;
        assert!(
            after > dug - 25.0,
            "the pit went from {dug} m to {after} m, against a landscape \
             starting at {start} m"
        );
    }

    /// Flat ground has nowhere for a droplet to go, and the run must end rather
    /// than spin or divide by a zero-length direction.
    #[test]
    fn droplets_on_flat_ground_stop_without_changing_it() {
        let mut fields = Fields::new([200.0, 200.0], 10.0);
        fields.height.values.fill(800.0);
        erode(&mut fields, 3, 3);
        assert!(fields.height.values.iter().all(|value| *value == 800.0));
    }

    /// Hard rock has to resist. This is what leaves the cliff bands standing
    /// once the softer beds around them have been cut away.
    #[test]
    fn hard_rock_is_cut_less_than_soft_rock() {
        let cut = |hardness: f32| {
            let before = cone(48, 10.0);
            let mut after = cone(48, 10.0);
            after.hardness.values.fill(hardness);
            erode(&mut after, 12, 3);
            before
                .height
                .values
                .iter()
                .zip(&after.height.values)
                .map(|(was, now)| f64::from((was - now).max(0.0)))
                .sum::<f64>()
        };
        let soft = cut(0.0);
        let hard = cut(1.0);
        assert!(hard < soft * 0.8, "soft lost {soft} m, hard lost {hard} m");
    }
}
