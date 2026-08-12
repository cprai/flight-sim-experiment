//! Where the water goes: filling the hollows, then following the drainage.
//!
//! Droplet erosion leaves a landscape full of small closed hollows -- a droplet
//! that dug slightly too deep at one cell made one -- and a landscape full of
//! hollows has no drainage network, because every trickle stops in the first
//! one it reaches. Two things come out of fixing that, and both are things the
//! rest of the crate needs:
//!
//! * **A filled surface.** Every hollow raised to the level it would spill at,
//!   which is where standing water settles. Lakes are not placed anywhere in
//!   this crate; they are wherever the ground turned out to hold water.
//! * **Drainage area.** How much ground drains through each cell. That single
//!   number is what tells a headwater trickle from a trunk river, and it is
//!   what the material classifier reads to decide where the water, the gravel
//!   bars and the wet meadows are.
//!
//! # Priority flood
//!
//! Both come out of one pass, Barnes' priority flood. Start from the edge of
//! the map, which is where water leaves; repeatedly take the lowest cell
//! reached so far and step to its unvisited neighbours, raising each to at
//! least the level of the cell that reached it. A cell in a hollow is therefore
//! reached over the hollow's rim and raised to the rim's height -- which is
//! exactly its spill level -- and a cell on open ground is reached from below
//! and left alone.
//!
//! The order cells come out in is the second thing this pass produces, and it
//! is worth as much as the surface. It is non-decreasing in the filled height,
//! so walking it backwards visits every cell before the cell it drains to, and
//! drainage area accumulates in one linear sweep with no iteration to converge
//! and no possibility of a cycle -- even across the dead flat surface of a
//! filled lake, where a steepest-descent rule has no answer at all.
//!
//! # Why the area is shared and the receiver is not
//!
//! There are two different questions here and they want two different answers.
//!
//! *Which way is downhill from this cell* has one answer, and [`Drainage`]
//! records it in `drains_to`. Incision needs it to be one answer: the implicit
//! stream-power sweep is a walk over a **tree**, and a tree is what makes it
//! unconditionally stable and linear in the cell count.
//!
//! *How much ground drains through this cell* must not be answered that way,
//! and answering it that way was a real and very visible bug. Sending every
//! cell's whole area to a single steepest neighbour is the D8 rule, and on a
//! planar hillside D8 has only eight directions to choose between: neighbouring
//! cells pick between two nearly equal directions, their flow lines run
//! parallel without ever merging, and which line a cell lands on is decided by
//! roughness far below the scale of any real valley. On a plane forty cells
//! across, with twenty centimetres of noise on a two-metre-per-cell fall,
//! neighbouring cells that are alike in every physical respect came out
//! carrying drainage differing fifteenfold. Incision cuts in proportion to the
//! square root of that, so the landscape arrives corrugated with grooves one
//! cell wide -- a corduroy visible from the air across the whole map. Over a
//! 16 km box the incision pass was multiplying the power at the
//! two-to-three-cell scale by four hundred.
//!
//! So area is shared instead, over *every* neighbour below the cell, in
//! proportion to how steeply each falls away. That is Freeman's multiple flow
//! direction rule (Freeman 1991, "Calculating catchment area with divergent
//! flow based on a regular grid"), and it is the standard answer to exactly
//! this artifact. Water spreads on a divergent hillside, as it does in reality,
//! and converges only where the ground actually converges -- so a valley still
//! collects a valley's worth of drainage and a smooth slope no longer pretends
//! to have a river running down one column of it.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::fields::Fields;

/// The eight neighbours of a cell.
const NEIGHBOURS: [(i64, i64); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

/// How sharply the sharing favours the steepest way down.
///
/// Freeman's value, and the reason it is a little above one rather than well
/// above it. At one the share is proportional to the fall, which is the
/// gentlest spreading there is; as the exponent grows the steepest neighbour
/// takes more and more, and in the limit the rule *is* D8 again, along with the
/// corrugation D8 causes. The temptation is to raise it on the theory that a
/// river wants to stay in one channel, and it is the wrong instinct: the
/// channels a landscape should have are the ones the ground converges into, and
/// a rule that concentrates water on its own makes them wherever it likes.
const SPREAD: f32 = 1.1;

/// How far in front of itself the accumulation asks the hardware to fetch.
///
/// The sweep is a random walk over a 220 MB working set and f27c6a2 measured it
/// three quarters memory rather than arithmetic, so what it is waiting for is
/// main memory rather than any instruction. Because `order` is a finished list,
/// the addresses it will want are known thousands of iterations early -- which
/// is the case a prefetch exists for, and the reason this is worth more here
/// than any amount of arithmetic.
///
/// Shared with `incise`, whose stream-power sweep walks the same order in the
/// other direction over the same scattered grids and wants the same lead.
///
/// Measured rather than picked; see `measure_how_far_ahead_to_fetch`. Three is
/// the middle of a flat 2-to-4 plateau, chosen over the single fastest sample
/// because the three are within noise of each other and the plateau is the real
/// finding.
///
/// **Three cells, not fifty**, and that is worth knowing rather than filing as
/// a tuning constant. Hiding a main-memory round trip needs tens of iterations
/// of lead, and out past eight the win shrinks steadily until by 256 it is
/// almost gone. So this is not latency hiding: it is getting a couple of
/// iterations' worth of gathers into flight at once, and further ahead only
/// means the line is evicted before the sweep arrives. Prefetching only the
/// cell's own line rather than the rows either side is worth 2% against 15%,
/// which says the same thing from the other side -- what the sweep waits for is
/// the eight-neighbour gather across three rows, not the cell it started from.
pub const AHEAD: usize = 3;

/// Asks for a cache line without waiting for it, if the index is real.
///
/// The only `unsafe` in this crate, and about as small as the keyword gets: a
/// prefetch has no architectural effect whatever -- it cannot fault, cannot
/// change a value, and a wrong address costs a wasted fetch rather than a bug.
/// The bounds check above it is not for soundness but so that the address is
/// one the walk will really want. Everything the sweep computes is identical
/// with this compiled in or out, which the fingerprint gate is what proves.
#[inline(always)]
pub fn prefetch<T>(values: &[T], index: usize) {
    if index >= values.len() {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: `index` is in bounds, so the pointer is inside the allocation.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
            values.as_ptr().add(index) as *const i8,
        );
    }
}

/// A total order on `f32` that matches the numeric one, as bits.
///
/// The heap holds one `u64` per cell rather than a pair, so that it is a plain
/// integer comparison and the tie-break falls to the cell index in the low
/// half -- which makes the pop order, and so the whole drainage network,
/// depend on nothing but the landscape.
fn ordered(value: f32) -> u32 {
    let bits = value.to_bits();
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits | 0x8000_0000
    }
}

/// The drainage of a landscape: where the water stands, where it goes, and how
/// much of the map goes with it.
///
/// Everything below is per cell of the grid it was computed from.
#[derive(Default)]
pub struct Drainage {
    /// The surface standing water settles to.
    pub filled: Vec<f32>,
    /// Which cell each one drains into, or `u32::MAX` for a cell that leaves
    /// the map.
    pub drains_to: Vec<u32>,
    /// How many cells drain through each one, counting itself.
    pub area: Vec<f32>,
    /// The order the flood reached the cells in, which is non-decreasing in
    /// [`Drainage::filled`].
    ///
    /// Every cell appears after the cell it drains into, so one sweep forward
    /// is downstream-to-upstream and one sweep backward is the other way. Both
    /// directions are used: the incision needs the first and the accumulation
    /// needed the second.
    pub order: Vec<u32>,
}

/// How many buckets the flood's queue spreads the height range over.
///
/// The trade is the whole design in one number: coarse buckets mean a big
/// active heap and the log(n) creeping back, fine ones mean more empty buckets
/// to step over and more heapify calls. Measured over the shipped grid rather
/// than reasoned about, holding the round count fixed because a faster flood
/// otherwise fits more of the later, costlier rounds into the same wall clock
/// and flatters itself:
///
/// ```text
/// bits   buckets   flood, ms/round
///   10      1024        762
///   12      4096        694.7
///   14     16384        620.1
///   16     65536        584.2
///   18    262144        595.5
/// ```
///
/// Shallow enough either side of sixteen that the exact choice hardly matters;
/// what does matter is not being at ten, where the active heap is back to
/// tens of thousands of cells.
const BUCKET_BITS: u32 = 16;
const BUCKETS: usize = 1 << BUCKET_BITS;

/// The flood's queue: buckets by height, with a heap for the one being drained.
///
/// A `BinaryHeap` over eleven million cells is an 88 MB array, and both ends of
/// it chase scattered lines: pushes here land near the current minimum, so they
/// sift a long way up. Bucketing makes a push an append into one of a handful
/// of hot vectors and a pop a sift inside something small enough to stay in
/// cache, and takes the flood from O(n log n) to amortised O(n + BUCKETS).
///
/// # Why this pops in exactly the order the plain heap did
///
/// Two properties, and the second is the one that is easy to get wrong.
///
/// **The queue is monotone.** Every push has `filled[n] = max(heights[n],
/// filled[current])`, so no cell is ever pushed lower than the cell it was
/// reached from, and nothing can land in a bucket the cursor has already gone
/// past. That is what lets the cursor move forward only.
///
/// **The active bucket still needs a heap.** A push *can* land in the bucket
/// being drained with a key below the one just popped -- equal `filled`, lower
/// index -- so the pop order within a bucket is not the order the cells arrived
/// in, and draining a sorted bucket in one go would be wrong. Keeping the
/// active bucket as a small heap on the full key reproduces the global heap's
/// answer element for element, which is what
/// `the_bucket_queue_floods_exactly_as_a_plain_heap_does` holds.
///
/// Cells wait as bare indices and the key is rebuilt when the bucket becomes
/// active, which halves what the queue moves about: `filled[i]` is written once
/// when a cell is first seen and never changes after, so rebuilding gives the
/// same key that would have been stored.
#[derive(Default)]
struct Queue {
    /// Cells waiting, by bucket. Empty at the end of every flood -- the cursor
    /// drains each one it passes -- so a new flood only resets the cursor.
    waiting: Vec<Vec<u32>>,
    /// The bucket being drained, ordered on the full key.
    active: BinaryHeap<Reverse<u64>>,
    /// Which bucket that is. Only ever moves forward.
    at: usize,
    /// How many cells are still in `waiting`, so the flood knows it is done
    /// without scanning to the end.
    parked: usize,
    /// `ordered` of the lowest ground, and how far to shift the difference. The
    /// span comes from the height range because the filled surface never leaves
    /// it: filling only ever raises ground towards a rim that is itself ground.
    base: u32,
    shift: u32,
    /// The grid width, so a bucket's cells can be asked for a row either side.
    stride: usize,
}

impl Queue {
    fn reset(&mut self, low: f32, high: f32, stride: usize) {
        if self.waiting.len() != BUCKETS {
            self.waiting.resize_with(BUCKETS, Vec::new);
        }
        self.active.clear();
        self.at = 0;
        self.parked = 0;
        self.stride = stride;
        self.base = ordered(low);
        // Spread whatever range this landscape has over the buckets, rather
        // than bucketing on the top bits of the float itself. Ground between
        // 700 m and 2600 m occupies a two-hundredth of the `ordered` space, so
        // raw top bits would leave all but a few hundred buckets empty however
        // many there were.
        let span = ordered(high).saturating_sub(self.base);
        self.shift = (32 - span.leading_zeros()).saturating_sub(BUCKET_BITS);
    }

    fn bucket(&self, value: f32) -> usize {
        ((ordered(value).saturating_sub(self.base) >> self.shift) as usize).min(BUCKETS - 1)
    }

    fn push(&mut self, index: usize, value: f32) {
        let bucket = self.bucket(value);
        debug_assert!(
            bucket >= self.at,
            "a push below the active bucket: the queue is not monotone after all"
        );
        if bucket <= self.at {
            self.active
                .push(Reverse((u64::from(ordered(value)) << 32) | index as u64));
        } else {
            self.waiting[bucket].push(index as u32);
            self.parked += 1;
        }
    }

    fn pop(&mut self, filled: &[f32], heights: &[f32]) -> Option<usize> {
        loop {
            if let Some(Reverse(packed)) = self.active.pop() {
                return Some((packed & 0xffff_ffff) as usize);
            }
            if self.parked == 0 {
                return None;
            }
            self.at += 1;
            if self.at >= BUCKETS {
                return None;
            }
            let Queue {
                waiting,
                active,
                at,
                parked,
                stride,
                ..
            } = self;
            let bucket = &mut waiting[*at];
            *parked -= bucket.len();
            active.extend(bucket.drain(..).map(|index| {
                let index = index as usize;
                for line in [index.wrapping_sub(*stride), index, index + *stride] {
                    prefetch(filled, line);
                    prefetch(heights, line);
                }
                Reverse((u64::from(ordered(filled[index])) << 32) | index as u64)
            }));
        }
    }
}

/// The buffers a drainage needs, kept between calls.
///
/// `incise::rivers` calls the drainage ninety-two times over one landscape, and
/// each call was allocating about 230 MB of grids and growing an eleven-million
/// element heap from nothing by doubling. None of that is per-round work: the
/// grid does not change size, so the buffers can be filled in once and then
/// written over. What it costs instead is the first-touch page faults on a
/// quarter of a gigabyte, ninety-two times over.
///
/// The same bargain `creep::settle` already takes with the caller's scratch,
/// and for the same reason -- see `incise::rivers`.
#[derive(Default)]
pub struct Scratch {
    /// Which cell reached each one. The flood's own tree, kept because it is
    /// the only thing that can route across a lake's flat surface. Never
    /// escapes: the receivers read it and nothing else does.
    reached_by: Vec<u32>,
    queue: Queue,
    drainage: Drainage,
}

impl Scratch {
    /// Hands the last drainage out, leaving the scratch empty.
    ///
    /// For the callers that want one drainage and no loop -- `route`, and every
    /// test -- so that reusing buffers stays an option a caller takes rather
    /// than an obligation the type imposes.
    fn take(&mut self) -> Drainage {
        std::mem::take(&mut self.drainage)
    }
}

/// Fills the hollows and works out where the water goes.
///
/// Three phases, and they are timed separately at `debug` because they are not
/// equally amenable to being sped up: the flood is a heap and inherently
/// ordered, the receivers are one independent decision per cell, and the
/// accumulation is a sweep that has to follow the flood's order. Which of them
/// dominates decides where any effort is worth spending, and that was worth
/// measuring rather than assuming.
pub fn drainage(fields: &Fields) -> Drainage {
    let mut scratch = Scratch::default();
    scratch.drainage(fields);
    scratch.take()
}

impl Scratch {
    /// Fills the hollows and works out where the water goes, into the buffers
    /// this scratch is already holding.
    ///
    /// See [`drainage`] for what the three phases are and why they are timed
    /// apart.
    pub fn drainage(&mut self, fields: &Fields) -> &Drainage {
        let (width, rows) = (fields.width(), fields.rows());
        let count = width * rows;
        let heights: &[f32] = &fields.height.values;
        let started = std::time::Instant::now();

        // Everything is sized once and then reused for the life of the scratch.
        // Which of these need clearing and which do not is not a judgement
        // call: a buffer that is written for every cell before it is read needs
        // nothing, and one that is not needs everything. That distinction is
        // what `a_reused_scratch_does_not_remember_the_last_landscape` exists
        // to hold, because getting it wrong leaves a stale value from another
        // landscape somewhere in the middle of this one.
        let out = &mut self.drainage;
        // Infinity means "the flood has not reached this cell", which is what a
        // separate `seen` array used to say. Folding the two removes an 11 MB
        // random-access grid from the hottest loop in the crate, and moves the
        // has-this-been-reached test onto the very cache line the code is about
        // to write the answer into. Nothing survives as infinity: the flood
        // starts from the whole rim of a rectangular grid, so every cell is
        // reached, and `every_cell_is_reached_by_the_flood` holds that.
        out.filled.resize(count, f32::INFINITY);
        out.filled.fill(f32::INFINITY);
        out.drains_to.resize(count, u32::MAX);
        out.area.resize(count, 1.0);
        out.area.fill(1.0);
        out.order.clear();
        out.order.reserve(count);
        self.reached_by.resize(count, u32::MAX);
        let (low, high) = fields.height.range();
        self.queue.reset(low, high, width);

        let (filled, drains_to, area, order) = (
            &mut out.filled,
            &mut out.drains_to,
            &mut out.area,
            &mut out.order,
        );
        let (reached_by, queue) = (&mut self.reached_by, &mut self.queue);

        // The edge of the map is where water leaves, so that is where the flood
        // starts. A landscape with no outlet at all would otherwise fill to its
        // own rim and drown.
        for row in 0..rows {
            for column in 0..width {
                let edge = row == 0 || column == 0 || row == rows - 1 || column == width - 1;
                if !edge {
                    continue;
                }
                let index = row * width + column;
                filled[index] = heights[index];
                // Written here rather than left to a full reset of the array.
                // A seed cell is the one kind the flood never reaches from
                // anywhere, so this is the only place its "reached from nowhere"
                // can be said -- and on a reused scratch, not saying it would
                // leave the previous landscape's answer on the rim, where it
                // decides which way a lake spills.
                reached_by[index] = u32::MAX;
                queue.push(index, filled[index]);
            }
        }

        while let Some(index) = queue.pop(filled, heights) {
            order.push(index as u32);
            let (column, row) = ((index % width) as i64, (index / width) as i64);
            for (dx, dy) in NEIGHBOURS {
                let (nx, ny) = (column + dx, row + dy);
                if nx < 0 || ny < 0 || nx >= width as i64 || ny >= rows as i64 {
                    continue;
                }
                let neighbour = ny as usize * width + nx as usize;
                if filled[neighbour].is_finite() {
                    continue;
                }
                filled[neighbour] = heights[neighbour].max(filled[index]);
                reached_by[neighbour] = index as u32;
                queue.push(neighbour, filled[neighbour]);
            }
        }

        log::debug!("drainage: the flood took {:.2?}", started.elapsed());
        let at = std::time::Instant::now();

        // Where each cell drains to: the steepest neighbour strictly below it
        // on the filled surface, or -- on the flat of a lake, where there is no
        // such neighbour -- whichever cell the flood reached it from, which is
        // by construction on the way to the spill point.
        // One row per task. Every cell's answer depends only on the finished
        // filled surface and never on another cell's answer, so this is the one
        // phase of the three that parallelises by simply being asked to: rayon
        // splitting the rows cannot change a single value, only how long they
        // take to arrive.
        let reached_by: &[u32] = reached_by;
        let filled: &[f32] = filled;
        drains_to
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(row, drains_to)| {
                for (column, drains_to) in drains_to.iter_mut().enumerate() {
                    let index = row * width + column;
                    let here = filled[index];
                    let (mut best, mut steepest) = (u32::MAX, 0.0f32);
                    for (dx, dy) in NEIGHBOURS {
                        let (nx, ny) = (column as i64 + dx, row as i64 + dy);
                        if nx < 0 || ny < 0 || nx >= width as i64 || ny >= rows as i64 {
                            continue;
                        }
                        let neighbour = ny as usize * width + nx as usize;
                        let reach = if dx != 0 && dy != 0 {
                            std::f32::consts::SQRT_2
                        } else {
                            1.0
                        };
                        let fall = (here - filled[neighbour]) / reach;
                        if fall > steepest {
                            steepest = fall;
                            best = neighbour as u32;
                        }
                    }
                    *drains_to = if best != u32::MAX {
                        best
                    } else {
                        reached_by[index]
                    };
                }
            });

        log::debug!("drainage: the receivers took {:.2?}", at.elapsed());
        let at = std::time::Instant::now();

        accumulate::<AHEAD>(width, rows, filled, order, drains_to, area, |fall| {
            fall.powf(SPREAD)
        });

        log::debug!("drainage: the accumulation took {:.2?}", at.elapsed());

        &self.drainage
    }
}

/// Shares every cell's drainage among the neighbours below it, in one backwards
/// sweep of the flood's order.
///
/// The sweep is exact in a single pass: a strictly lower neighbour was popped
/// strictly earlier, so it is still to come in this direction, and by the time a
/// cell is reached every cell that could give to it already has. That is also
/// what makes it strictly serial -- a cell reads its own area and adds into up
/// to eight arbitrary others.
///
/// `share` is the spreading rule, `fall.powf(SPREAD)` in a run. It is a
/// parameter rather than a call so that a measurement can time this sweep with
/// the exponent swapped out and still be timing *this* sweep. A measurement
/// that reimplements its subject stops measuring it the moment either one moves
/// -- see 581ddd0, where a rewritten emit loop reported a hundredth of the
/// truth. Being generic, it monomorphises to the same code a direct call would.
///
/// `AHEAD` is how many cells in front of itself the sweep asks the hardware to
/// fetch; zero compiles the prefetching away entirely. See [`AHEAD`] for what it
/// is set to and why that number was measured rather than picked.
fn accumulate<const AHEAD: usize>(
    width: usize,
    rows: usize,
    filled: &[f32],
    order: &[u32],
    drains_to: &[u32],
    area: &mut [f32],
    mut share: impl FnMut(f32) -> f32,
) {
    let mut falls = [0.0f32; 8];
    for (position, index) in order.iter().enumerate().rev() {
        // Where this sweep is going is known long before it gets there --
        // `order` is a finished list -- which is the one thing a random-access
        // walk needs to stop being latency-bound. The rows either side are
        // asked for as well as the cell itself, because the eight neighbours it
        // is about to touch straddle three rows and only the middle one shares
        // a cache line with the cell.
        if AHEAD > 0 && position >= AHEAD {
            let soon = order[position - AHEAD] as usize;
            for line in [soon.wrapping_sub(width), soon, soon + width] {
                prefetch(filled, line);
                prefetch(area, line);
            }
        }

        let index = *index as usize;
        let (column, row) = ((index % width) as i64, (index / width) as i64);
        let here = filled[index];

        let mut total = 0.0f32;
        for (slot, (dx, dy)) in NEIGHBOURS.iter().enumerate() {
            falls[slot] = 0.0;
            let (nx, ny) = (column + dx, row + dy);
            if nx < 0 || ny < 0 || nx >= width as i64 || ny >= rows as i64 {
                continue;
            }
            let neighbour = ny as usize * width + nx as usize;
            let reach = if *dx != 0 && *dy != 0 {
                std::f32::consts::SQRT_2
            } else {
                1.0
            };
            let fall = (here - filled[neighbour]) / reach;
            if fall > 0.0 {
                // The share each neighbour takes: how steeply it falls, raised
                // to the spreading exponent, times how much of this cell's
                // outline faces it. A diagonal neighbour is offered the corner
                // rather than a whole side, which is `1 / sqrt(2)` of one --
                // without that the four corners would between them take more
                // than the four sides and the water would drift onto the
                // diagonals.
                falls[slot] = share(fall) / reach;
                total += falls[slot];
            }
        }

        if total <= 0.0 {
            // Nothing below: the flat of a filled lake, or a cell on the rim of
            // the map. The flood's own tree is the only thing that knows the
            // way to the spill point from here.
            let into = drains_to[index];
            if into != u32::MAX {
                area[into as usize] += area[index];
            }
            continue;
        }

        let sending = area[index];
        for (slot, (dx, dy)) in NEIGHBOURS.iter().enumerate() {
            if falls[slot] <= 0.0 {
                continue;
            }
            let neighbour = (row + dy) as usize * width + (column + dx) as usize;
            area[neighbour] += sending * falls[slot] / total;
        }
    }
}

/// Fills the hollows and writes the water channels the rest of the crate reads.
pub fn route(fields: &mut Fields) {
    let drainage = drainage(fields);
    fields.flow.values = drainage.area.iter().map(|cells| cells.log2()).collect();
    fields.filled.values = drainage.filled;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bowl inside a plain that falls away to the east, so there is both a
    /// hollow to fill and somewhere for the water to leave.
    fn bowl(side: usize) -> Fields {
        let mut fields = Fields::new([(side - 1) as f32 * 10.0, (side - 1) as f32 * 10.0], 10.0);
        let middle = (side - 1) as f32 * 0.5;
        for row in 0..side {
            for column in 0..side {
                let index = fields.height.index(column, row);
                let (dx, dy) = (column as f32 - middle, row as f32 - middle);
                let distance = (dx * dx + dy * dy).sqrt();
                let plain = 900.0 - column as f32 * 0.5;
                fields.height.values[index] = if distance < middle * 0.4 {
                    plain - (middle * 0.4 - distance) * 3.0
                } else {
                    plain
                };
            }
        }
        fields
    }

    /// A scratch that has already seen one landscape must answer about the next
    /// one exactly as a fresh scratch would.
    ///
    /// This is the whole risk in reusing the buffers, and it is a quiet one. A
    /// grid that is written for every cell before it is read needs no clearing;
    /// one that is not needs all of it, and the difference is invisible until a
    /// value from the *previous* landscape survives in the middle of this one.
    /// `reached_by` is the trap: only cells the flood arrives at are written,
    /// so the seed cells on the rim keep whatever was there -- and those are
    /// exactly the cells that decide which way a lake spills.
    ///
    /// The grids are deliberately **different sizes**, and that is the whole
    /// design of the test rather than incidental. Written the obvious way --
    /// two landscapes of the same size -- it passes with the `reached_by` seed
    /// deleted, because every landscape of a given size has the same rim, so
    /// those cells hold `u32::MAX` from the first `resize` and never stop. It
    /// takes a *smaller* second grid, where a cell that was interior before is
    /// now on the rim carrying a real receiver from the last landscape, to make
    /// the bug observable at all.
    ///
    /// All four products, against a fresh run, because each reset that is
    /// missing fails somewhere different. Sabotaged three ways, all caught:
    ///
    /// - dropping the `reached_by` seed fails on `drains_to`, and on nothing
    ///   else -- the surface is right and only the lake routing is wrong, which
    ///   is why comparing `filled` alone would not have done
    /// - dropping `seen.fill(false)` fails on `filled`: the previous run left
    ///   every cell seen, so the flood reaches nothing past the rim and most of
    ///   the surface is simply the last landscape's, truncated
    /// - dropping `area.fill(1.0)` fails on `area`, every cell inflated by
    ///   whatever drained through it last time
    #[test]
    fn a_reused_scratch_does_not_remember_the_last_landscape() {
        let first = bowl(41);
        let second = bowl(21);

        let mut scratch = Scratch::default();
        scratch.drainage(&first);
        let reused = scratch.drainage(&second);
        let fresh = drainage(&second);

        assert_eq!(reused.filled, fresh.filled, "filled");
        assert_eq!(reused.order, fresh.order, "order");
        assert_eq!(reused.drains_to, fresh.drains_to, "drains_to");
        assert_eq!(reused.area, fresh.area, "area");
    }

    /// The flood as it was before the bucket queue: one global heap over every
    /// cell, popped lowest first.
    ///
    /// Kept as the oracle rather than deleted, in the pattern `flood.rs` uses
    /// for the GPU fill and `a_plane_drains_evenly...` uses for D8. The bucket
    /// queue's entire claim is that it pops in exactly this order, and a claim
    /// of *exactness* can only be checked against the thing it is exact to.
    #[cfg(test)]
    fn flood_with_one_global_heap(fields: &Fields) -> (Vec<f32>, Vec<u32>, Vec<u32>) {
        let (width, rows) = (fields.width(), fields.rows());
        let count = width * rows;
        let heights = &fields.height.values;

        let mut filled = vec![0f32; count];
        let mut reached_by = vec![u32::MAX; count];
        let mut seen = vec![false; count];
        let mut order = Vec::with_capacity(count);
        let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::new();

        for row in 0..rows {
            for column in 0..width {
                if row != 0 && column != 0 && row != rows - 1 && column != width - 1 {
                    continue;
                }
                let index = row * width + column;
                filled[index] = heights[index];
                seen[index] = true;
                heap.push(Reverse(
                    (u64::from(ordered(filled[index])) << 32) | index as u64,
                ));
            }
        }
        while let Some(Reverse(packed)) = heap.pop() {
            let index = (packed & 0xffff_ffff) as usize;
            order.push(index as u32);
            let (column, row) = ((index % width) as i64, (index / width) as i64);
            for (dx, dy) in NEIGHBOURS {
                let (nx, ny) = (column + dx, row + dy);
                if nx < 0 || ny < 0 || nx >= width as i64 || ny >= rows as i64 {
                    continue;
                }
                let neighbour = ny as usize * width + nx as usize;
                if seen[neighbour] {
                    continue;
                }
                seen[neighbour] = true;
                filled[neighbour] = heights[neighbour].max(filled[index]);
                reached_by[neighbour] = index as u32;
                heap.push(Reverse(
                    (u64::from(ordered(filled[neighbour])) << 32) | neighbour as u64,
                ));
            }
        }
        (filled, reached_by, order)
    }

    /// The bucket queue has to pop in the same sequence the global heap did --
    /// not a similar one, the same one.
    ///
    /// `order` is the strict check and the surface is the loose one. Two floods
    /// can leave identical `filled` values while visiting the cells in
    /// different sequences, and the sequence is what the accumulation and the
    /// stream-power sweep both walk, so comparing the surface alone would let a
    /// reordering through that changes the landscape everywhere downstream.
    ///
    /// Three landscapes, because the queue's correctness is about where cells
    /// land relative to bucket boundaries: a bowl with a real hollow, a plane
    /// with no hollow at all, and dead flat ground where every cell shares one
    /// height and so every cell lands in one bucket -- which is the case that
    /// makes the active heap do all the work and the bucketing none of it.
    ///
    /// Sabotaged by inverting the order *within* the active bucket while
    /// leaving the order *between* buckets alone -- which is precisely the
    /// property the design claims to need, and it fails on `order` for the
    /// bowl. Two earlier attempts at a sabotage were no such thing and are
    /// worth recording so they are not tried again: pre-sorting a bucket before
    /// pushing it into the heap changes nothing, because a heap re-orders
    /// whatever it is given; and dropping the parked cells fails, but for
    /// losing them rather than for mis-ordering them.
    #[test]
    fn the_bucket_queue_floods_exactly_as_a_plain_heap_does() {
        let flat = {
            let mut fields = Fields::new([400.0, 400.0], 10.0);
            fields.height.values.fill(850.0);
            fields
        };
        let plane = {
            let mut fields = Fields::new([400.0, 400.0], 10.0);
            let width = fields.width();
            for (index, height) in fields.height.values.iter_mut().enumerate() {
                *height = 900.0 - (index % width) as f32 * 0.5;
            }
            fields
        };

        for (name, fields) in [("bowl", bowl(41)), ("plane", plane), ("flat", flat)] {
            let (filled, reached_by, order) = flood_with_one_global_heap(&fields);
            let mine = drainage(&fields);

            assert_eq!(mine.order, order, "{name}: the pop order");
            assert_eq!(mine.filled, filled, "{name}: the filled surface");
            // Not returned by the drainage, but it is what routes a lake, so
            // check it through the receivers it decides.
            let _ = reached_by;
            assert!(!order.is_empty(), "{name}: the flood reached nothing");
        }
    }

    /// Every cell has to come out of the flood with a real height on it.
    ///
    /// The filled surface doubles as the flood's own record of what it has
    /// reached -- infinity means "not yet" -- so a cell the flood somehow
    /// missed would leave an infinity behind rather than a wrong number, and
    /// then quietly poison the receivers, the accumulation and every height
    /// downstream of them. It cannot happen on a rectangular grid seeded from
    /// its whole rim, which is exactly the kind of reasoning worth pinning to a
    /// test rather than to a comment.
    ///
    /// Sabotaged by a fencepost in the queue -- `drain(1..)` instead of
    /// `drain(..)` when a bucket becomes active, losing one cell per bucket --
    /// which leaves infinities behind and fails here.
    ///
    /// Seeding only the northern edge is *not* a sabotage of this and was tried
    /// first: a rectangular grid is connected, so the flood still reaches every
    /// cell from any non-empty seed set. What this catches is a queue that
    /// loses cells, not a rim that is too small.
    #[test]
    fn every_cell_is_reached_by_the_flood() {
        for fields in [bowl(41), bowl(9)] {
            let drainage = drainage(&fields);
            assert_eq!(drainage.order.len(), fields.height.values.len());
            let missed = drainage
                .filled
                .iter()
                .filter(|height| !height.is_finite())
                .count();
            assert_eq!(missed, 0, "cells the flood never reached");
        }
    }

    /// The property everything downstream rests on: after routing, no cell is
    /// lower than every one of its neighbours. A single surviving pit is a
    /// place the drainage network simply stops.
    #[test]
    fn the_filled_surface_has_no_hollow_left_in_it() {
        let mut fields = bowl(41);
        route(&mut fields);
        let (width, rows) = (fields.width(), fields.rows());
        for row in 1..rows - 1 {
            for column in 1..width - 1 {
                let here = fields.filled.values[row * width + column];
                let lowest = NEIGHBOURS
                    .iter()
                    .map(|(dx, dy)| {
                        fields.filled.values
                            [(row as i64 + dy) as usize * width + (column as i64 + dx) as usize]
                    })
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    here >= lowest - 1e-4,
                    "({column}, {row}) sits at {here}, under every neighbour's {lowest}"
                );
            }
        }
    }

    /// A lake is flat. It is the one thing everybody can see about water, and
    /// the filled surface is what draws it.
    #[test]
    fn a_hollow_fills_to_one_level_and_never_below_the_ground() {
        let mut fields = bowl(41);
        route(&mut fields);

        let mut levels: Vec<f32> = Vec::new();
        for (index, filled) in fields.filled.values.iter().enumerate() {
            let ground = fields.height.values[index];
            assert!(
                *filled >= ground - 1e-4,
                "cell {index} filled to {filled}, under its own ground at {ground}"
            );
            if *filled - ground > 1.0 {
                levels.push(*filled);
            }
        }
        assert!(!levels.is_empty(), "the bowl did not fill at all");
        let (low, high) = levels
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), v| {
                (l.min(*v), h.max(*v))
            });
        assert!(high - low < 1e-2, "the lake surface spans {low} to {high}");
    }

    /// Ground that already drains must be left exactly where it is, or every
    /// open slope in the landscape would be quietly raised.
    #[test]
    fn ground_that_already_drains_is_not_raised() {
        let mut fields = Fields::new([400.0, 400.0], 10.0);
        for row in 0..fields.rows() {
            for column in 0..fields.width() {
                let index = fields.height.index(column, row);
                fields.height.values[index] = 900.0 - column as f32 * 2.0;
            }
        }
        route(&mut fields);
        for (filled, ground) in fields.filled.values.iter().zip(&fields.height.values) {
            assert_eq!(filled, ground);
        }
    }

    /// Drainage area is the classifier's whole notion of a river. It has to
    /// grow downstream, and it has to add up: the outlets between them must
    /// account for every cell on the map.
    #[test]
    fn drainage_gathers_downstream_and_accounts_for_every_cell() {
        let mut fields = bowl(41);
        route(&mut fields);
        let count = fields.width() * fields.rows();

        let (low, high) = fields.flow.range();
        assert_eq!(low, 0.0, "a cell with nothing above it drains one cell");
        assert!(
            high > (count as f32).log2() * 0.25,
            "the largest drainage is 2^{high} cells out of {count}"
        );

        // Every cell drains through some edge cell, so the edge accounts for
        // the map. Counted in cells rather than in the stored logarithm.
        let (width, rows) = (fields.width(), fields.rows());
        let mut through_the_edge = 0.0f64;
        for row in 0..rows {
            for column in 0..width {
                if row == 0 || column == 0 || row == rows - 1 || column == width - 1 {
                    through_the_edge += f64::from(fields.flow.values[row * width + column].exp2());
                }
            }
        }
        assert!(
            through_the_edge >= count as f64,
            "{through_the_edge} cells left through the edge, out of {count}"
        );
    }

    /// The bug this module's sharing rule exists to prevent, stated as
    /// directly as it can be.
    ///
    /// A plane tilted so that it falls between two of the eight neighbour
    /// directions is the worst case for D8 and the reason for everything above:
    /// with one receiver each, cells pick whichever of the two is nearer, their
    /// flow lines run parallel without ever merging, and the area they carry
    /// ends up differing by orders of magnitude between columns that are
    /// identical in every physical respect. Incision then cuts in proportion to
    /// its square root and the hillside comes out corrugated.
    ///
    /// Nothing about this ground distinguishes one cell from the next, so
    /// nothing about the drainage may either.
    ///
    /// Stated against the rule it replaced rather than against a bare number,
    /// by accumulating the same landscape both ways. That is worth the extra
    /// dozen lines: a threshold on its own would still pass if the sharing were
    /// ever quietly weakened back towards a single receiver, whereas this
    /// asserts the contrast that is the whole point -- and records, for anyone
    /// reading it later, exactly how badly the old rule behaved.
    #[test]
    fn a_plane_drains_evenly_rather_than_in_parallel_lines() {
        let side = 81;
        let mut fields = Fields::new([(side - 1) as f32 * 10.0, (side - 1) as f32 * 10.0], 10.0);
        for row in 0..side {
            for column in 0..side {
                let index = fields.height.index(column, row);
                // Falling south-south-east, so that neither an axis nor a
                // diagonal is the answer and D8 has to keep choosing between
                // two directions that are very nearly as steep as each other.
                // The centimetre of noise on top is what makes it choose
                // differently in different places, and a centimetre is all it
                // takes -- which is the point. A dead plane is the one case D8
                // handles: every cell picks the same direction, so the lines
                // are parallel and carry the same area as each other, and the
                // problem hides completely.
                fields.height.values[index] = 900.0 - row as f32 * 2.0 - column as f32 * 0.7
                    + crate::noise::gradient(column as f32 / 3.0, row as f32 / 3.0, 77) * 0.2;
            }
        }
        let drainage = drainage(&fields);

        // The single-receiver accumulation this module used to do, over the
        // same flood and the same receivers.
        let mut d8 = vec![1.0f32; drainage.area.len()];
        for index in drainage.order.iter().rev() {
            let index = *index as usize;
            let into = drainage.drains_to[index];
            if into != u32::MAX {
                d8[into as usize] += d8[index];
            }
        }

        // Across the middle, well inside the edges, where every cell has the
        // same ground above it as its neighbours.
        let row = side / 2;
        let spread = |area: &[f32]| {
            let (low, high) = (20..side - 20)
                .map(|column| area[row * side + column])
                .fold((f32::INFINITY, 0.0f32), |(l, h), a| (l.min(a), h.max(a)));
            high / low
        };
        assert!(
            spread(&d8) > 10.0,
            "the single-receiver rule spread this plane's drainage by only {}, \
             so this landscape no longer demonstrates the problem",
            spread(&d8)
        );
        assert!(
            spread(&drainage.area) < 1.5,
            "identical cells of a plane carry drainage differing by {}",
            spread(&drainage.area)
        );
    }

    /// ... and the other half of it, which is what stops the cure from being
    /// worse than the disease. Spreading the water must not stop it collecting:
    /// ground that converges has to end up carrying far more than ground that
    /// does not, or there are no rivers to cut anything with.
    #[test]
    fn a_valley_still_collects_far_more_than_the_slope_beside_it() {
        let side = 81;
        let mut fields = Fields::new([(side - 1) as f32 * 10.0, (side - 1) as f32 * 10.0], 10.0);
        let middle = (side - 1) as f32 * 0.5;
        for row in 0..side {
            for column in 0..side {
                let index = fields.height.index(column, row);
                let across = (column as f32 - middle).abs().min(20.0);
                fields.height.values[index] = 900.0 - row as f32 * 2.0 + across * 1.5;
            }
        }
        let drainage = drainage(&fields);
        let row = side - 12;
        let floor = drainage.area[row * side + side / 2];
        let flank = drainage.area[row * side + side - 12];
        assert!(
            floor > flank * 20.0,
            "the valley floor carries {floor} against the flank's {flank}"
        );
    }

    /// The pop order is what makes a single accumulation sweep exact. If it
    /// ever stopped being non-decreasing, cells would drain into cells that had
    /// already been counted and the network would lose whole tributaries.
    #[test]
    fn every_cell_drains_into_one_no_higher_than_itself() {
        let mut fields = bowl(41);
        route(&mut fields);
        let (width, rows) = (fields.width(), fields.rows());
        for row in 0..rows {
            for column in 0..width {
                let index = row * width + column;
                let here = fields.filled.values[index];
                // Re-derived rather than kept, because the check is about the
                // surface the routing left rather than about the bookkeeping.
                let lowest = NEIGHBOURS
                    .iter()
                    .filter_map(|(dx, dy)| {
                        let (nx, ny) = (column as i64 + dx, row as i64 + dy);
                        (nx >= 0 && ny >= 0 && nx < width as i64 && ny < rows as i64)
                            .then(|| fields.filled.values[ny as usize * width + nx as usize])
                    })
                    .fold(f32::INFINITY, f32::min);
                assert!(lowest <= here + 1e-4, "({column}, {row}) has nowhere to go");
            }
        }
    }
}

#[cfg(test)]
mod measure {
    use super::*;

    /// One run of the real sweep, timed, with what it changed against a control.
    ///
    /// Generic rather than taking a function pointer, and that is not a
    /// stylistic choice: through a `fn(f32) -> f32` every variant below would
    /// compile to the same indirect call and they would all measure the same,
    /// which is exactly the shape of a measurement that quietly answers a
    /// different question than the one asked.
    fn sweep<const AHEAD: usize>(
        label: &str,
        fields: &Fields,
        drainage: &Drainage,
        control: Option<&[f32]>,
        share: impl FnMut(f32) -> f32 + Copy,
    ) -> Vec<f32> {
        const REPEATS: usize = 3;
        let (width, rows) = (fields.width(), fields.rows());
        let mut area = Vec::new();
        let mut best = std::time::Duration::MAX;
        for _ in 0..REPEATS {
            area = vec![1.0f32; width * rows];
            let at = std::time::Instant::now();
            accumulate::<AHEAD>(
                width,
                rows,
                &drainage.filled,
                &drainage.order,
                &drainage.drains_to,
                &mut area,
                share,
            );
            best = best.min(at.elapsed());
        }

        match control {
            None => println!("  {label:38} {best:>8.1?}"),
            Some(control) => {
                let mut differing = 0usize;
                let (mut worst_relative, mut worst_log) = (0.0f32, 0.0f32);
                for (mine, theirs) in area.iter().zip(control) {
                    if mine != theirs {
                        differing += 1;
                    }
                    worst_relative =
                        worst_relative.max((mine - theirs).abs() / theirs.abs().max(1e-9));
                    // What the classifier actually reads: `route` stores the
                    // log of the area, so a tenfold change in a headwater
                    // trickle matters far less than this ratio suggests.
                    worst_log = worst_log.max((mine.log2() - theirs.log2()).abs());
                }
                let share_differing = differing as f64 * 100.0 / area.len() as f64;
                println!(
                    "  {label:38} {best:>8.1?}   {share_differing:5.1}% differ, \
                     worst {worst_relative:.3} relative, {worst_log:.4} in log2",
                );
            }
        }
        area
    }

    /// What the spreading exponent costs, and what a cheaper one would change.
    ///
    /// The accumulation is 62 s of a 203 s run and two hypotheses fit its
    /// 61 ns per cell equally well. Either it is arithmetic-bound -- three or
    /// four `powf` calls a cell at 8-20 ns each -- or it is latency-bound on
    /// about seven random cache lines a cell over a 220 MB working set, at
    /// which point the `powf` hides underneath the misses and removing it buys
    /// nothing. Those lead to completely different work, so the number decides
    /// the plan rather than decorating it.
    ///
    /// The last row is the floor: the same sweep, the same scatter, no
    /// transcendental at all. If the exponent is free, that row is not much
    /// faster than the first and the loop is memory-bound.
    ///
    /// The second row is the candidate. `x^1.125 = x * x^(1/8)` is three
    /// correctly-rounded `sqrt` instructions -- exact, deterministic, no table,
    /// and it transcribes to WGSL -- but 1.125 is not 1.1, so it is a change to
    /// the landscape and the columns beside the timing are what that change
    /// costs.
    ///
    /// Run with `--release ... -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement on the full grid, not a check"]
    fn measure_what_the_spreading_exponent_costs() {
        let fields = crate::fields::shipped_grid();
        println!(
            "grid {} x {} = {} cells",
            fields.width(),
            fields.rows(),
            fields.width() * fields.rows()
        );

        // The real flood, so the sweep walks the order and the surface a run
        // actually hands it rather than something with a tidier shape.
        let drainage = drainage(&fields);

        // Counted in its own untimed run, so the counter does not sit in the
        // loop being measured.
        let mut calls = 0u64;
        let mut counted = vec![1.0f32; fields.width() * fields.rows()];
        accumulate::<AHEAD>(
            fields.width(),
            fields.rows(),
            &drainage.filled,
            &drainage.order,
            &drainage.drains_to,
            &mut counted,
            |fall| {
                calls += 1;
                fall.powf(SPREAD)
            },
        );
        println!(
            "the exponent is evaluated {calls} times a sweep, {:.2} per cell",
            calls as f64 / counted.len() as f64
        );

        println!("  rule                                       best   against the rule today");
        let control = sweep::<AHEAD>("powf(1.1), the rule today", &fields, &drainage, None, |f| {
            f.powf(SPREAD)
        });
        sweep::<AHEAD>(
            "f * f^(1/8), i.e. 1.125, three sqrts",
            &fields,
            &drainage,
            Some(&control),
            |f| f * f.sqrt().sqrt().sqrt(),
        );
        sweep::<AHEAD>(
            "f, i.e. 1.0, the cheapest exponent",
            &fields,
            &drainage,
            Some(&control),
            |f| f,
        );
        sweep::<AHEAD>(
            "1.0, no transcendental at all",
            &fields,
            &drainage,
            Some(&control),
            |_| 1.0,
        );
    }

    /// How far in front of itself the sweep should ask for its cache lines.
    ///
    /// f27c6a2 measured the accumulation three quarters memory and only a
    /// quarter arithmetic, so this is where its time actually is. `order` is a
    /// finished list, which means the addresses are knowable arbitrarily far
    /// ahead -- the only question is how far ahead is useful, and that is a
    /// property of this machine's memory system rather than of the code, so it
    /// is measured rather than reasoned about.
    ///
    /// Zero is the control: the same function with the prefetching compiled
    /// out.
    ///
    /// Run with `--release ... -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement on the full grid, not a check"]
    fn measure_how_far_ahead_to_fetch() {
        let fields = crate::fields::shipped_grid();
        let drainage = drainage(&fields);
        let rule = |f: f32| f.powf(SPREAD);

        println!("  cells ahead                                best");
        let control = sweep::<0>("0, no prefetch at all", &fields, &drainage, None, rule);
        sweep::<1>("1", &fields, &drainage, Some(&control), rule);
        sweep::<2>("2", &fields, &drainage, Some(&control), rule);
        sweep::<3>("3", &fields, &drainage, Some(&control), rule);
        sweep::<4>("4", &fields, &drainage, Some(&control), rule);
        sweep::<6>("6", &fields, &drainage, Some(&control), rule);
        sweep::<8>("8", &fields, &drainage, Some(&control), rule);
        sweep::<12>("12", &fields, &drainage, Some(&control), rule);
        sweep::<16>("16", &fields, &drainage, Some(&control), rule);
        sweep::<32>("32", &fields, &drainage, Some(&control), rule);
    }

    /// Whether a tiled, parallel flood is worth the rest of Barnes.
    ///
    /// The gate this exists to answer: the fill is the cheap half, and the pop
    /// order and the lake-routing tree are the expensive half. If the cheap
    /// half does not clear about three times, the expensive half is not worth
    /// designing and the flood stays as it is.
    ///
    /// Exactness is checked first and is not negotiable -- the minimax surface
    /// has one answer however it is reached, so a tiled flood that disagrees
    /// with the heap is wrong rather than approximate.
    ///
    /// Run with `--release ... -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement on the full grid, not a check"]
    fn measure_whether_a_tiled_flood_is_worth_it() {
        let fields = crate::fields::shipped_grid();
        println!(
            "grid {} x {}, {} threads",
            fields.width(),
            fields.rows(),
            rayon::current_num_threads()
        );

        // The phase timings inside `drainage` are logged rather than returned,
        // and the comparison has to be against the flood phase alone: timing
        // the whole `drainage` call would put the receivers and the
        // accumulation on the serial side of the ratio and flatter the tiles by
        // about three times.
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Debug)
            .is_test(false)
            .try_init();
        let heap = drainage(&fields).filled;

        for tile in [64usize, 128, 256, 512] {
            let at = std::time::Instant::now();
            let (mine, sweeps) = super::tiled::fill(&fields, tile);
            let elapsed = at.elapsed();
            let differing = mine.iter().zip(&heap).filter(|(a, b)| a != b).count();
            println!(
                "  {tile:>4}-cell tiles, {sweeps:>2} sweeps   {elapsed:>9.1?}   \
                 {differing} of {} cells differ",
                mine.len(),
            );
        }
    }

    /// How deep the drainage network is, and how wide each of its levels is.
    ///
    /// The evidence for or against parallelising the accumulation by depth.
    /// Cells of equal depth never donate to each other, so a level can run
    /// across all cores -- but only if there are levels worth handing to
    /// twenty-four threads, and only if working out the depths is cheaper than
    /// the sweep it is meant to parallelise. Both halves of that are printed
    /// here rather than argued.
    ///
    /// Run with `--release ... -- --ignored --nocapture`.
    #[test]
    #[ignore = "a measurement on the full grid, not a check"]
    fn measure_how_deep_the_drainage_network_is() {
        let fields = crate::fields::shipped_grid();
        let (width, rows) = (fields.width(), fields.rows());
        let drainage = drainage(&fields);

        // The same backwards sweep the accumulation does, with the arithmetic
        // replaced by a max -- which is the point: this is what a level
        // decomposition would have to pay before it parallelised anything.
        let at = std::time::Instant::now();
        let mut depth = vec![0u32; width * rows];
        for index in drainage.order.iter().rev() {
            let index = *index as usize;
            let (column, row) = ((index % width) as i64, (index / width) as i64);
            let here = drainage.filled[index];
            let mine = depth[index];
            let mut fell = false;
            for (dx, dy) in NEIGHBOURS.iter() {
                let (nx, ny) = (column + dx, row + dy);
                if nx < 0 || ny < 0 || nx >= width as i64 || ny >= rows as i64 {
                    continue;
                }
                let neighbour = ny as usize * width + nx as usize;
                if drainage.filled[neighbour] < here {
                    depth[neighbour] = depth[neighbour].max(mine + 1);
                    fell = true;
                }
            }
            if !fell {
                let into = drainage.drains_to[index];
                if into != u32::MAX {
                    depth[into as usize] = depth[into as usize].max(mine + 1);
                }
            }
        }
        let pre_pass = at.elapsed();

        let levels = *depth.iter().max().expect("a non-empty grid") as usize + 1;
        let mut population = vec![0usize; levels];
        for at in &depth {
            population[*at as usize] += 1;
        }
        let thin = population
            .iter()
            .filter(|cells| **cells < rayon::current_num_threads())
            .count();
        let mut sorted = population.clone();
        sorted.sort_unstable();
        let at_percentile = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];

        println!("{levels} levels over {} cells", depth.len());
        println!(
            "cells per level: 50th {}, 90th {}, 99th {}, largest {}",
            at_percentile(0.5),
            at_percentile(0.9),
            at_percentile(0.99),
            sorted[sorted.len() - 1],
        );
        println!(
            "{thin} levels ({:.1}%) hold fewer cells than this machine has threads ({})",
            thin as f64 * 100.0 / levels as f64,
            rayon::current_num_threads(),
        );
        println!(
            "working the depths out took {pre_pass:.1?}, which is what a level \
             decomposition pays before it parallelises anything"
        );
    }
}

/// A prototype of the one thing left that could put every core on the flood.
///
/// The flood is the largest single item in a run and it is the only phase with
/// no parallel form: a priority queue is a total order and a total order is one
/// thread. Barnes' answer (Barnes 2016, "Parallel Priority-Flood depression
/// filling for trillion cell digital elevation models") is to decompose the
/// grid into tiles, flood each independently, and reconcile them.
///
/// This is the fill and nothing else, which is deliberate: a run needs the pop
/// order and the lake-routing tree as much as it needs the surface, and neither
/// falls out of a tiled flood. That is the expensive half of adopting Barnes
/// and it is only worth designing if the cheap half wins first, so this exists
/// to answer whether it does.
///
/// # What this is, and what it is not
///
/// It is the **iterative** tiled fill: sweep every tile, repeat until nothing
/// moves. It is not Barnes' algorithm, which floods each tile once, solves a
/// small graph over the tile borders, and floods once more -- two passes and a
/// reconciliation rather than however many sweeps the geometry demands. The
/// difference is the whole result below: information crosses one tile per
/// sweep, so a grid twelve tiles across wants about eight sweeps, and doing
/// eight times the work to get twenty-four times the parallelism leaves very
/// little over.
///
/// Kept because the numbers it produces are the ones a real attempt at Barnes
/// would need as its starting point, and because it establishes the part that
/// was not obvious: a tiled fill can be *exact*, to the cell, against the heap.
#[cfg(test)]
mod tiled {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// The filled surface computed one tile at a time, across every core.
    ///
    /// The surface is a minimax: a cell's filled height is the lowest, over
    /// every path out to the rim of the map, of the highest ground on that
    /// path. That has a fixed point which does not care how it is reached --
    /// which is what makes a tiled version able to agree with the heap exactly
    /// rather than approximately.
    ///
    /// Each tile floods inwards from its own halo, lowering cells it can
    /// improve and leaving the rest, and a sweep is done when no tile lowered
    /// anything. Values only ever fall, from infinity towards the answer, so it
    /// terminates.
    ///
    /// Tiles are coloured two apart on both axes, exactly as `hydraulic` does,
    /// so that no two tiles running at once can touch the same cell: a tile
    /// writes only its own interior and reads only one ring beyond it.
    /// `AtomicU32` for the same reason it is used there -- the compiler's
    /// permission, not synchronisation.
    pub fn fill(fields: &Fields, tile: usize) -> (Vec<f32>, usize) {
        let (width, rows) = (fields.width(), fields.rows());
        let heights = &fields.height.values;

        let filled: Vec<AtomicU32> = (0..width * rows)
            .map(|index| {
                let (column, row) = (index % width, index / width);
                let rim = row == 0 || column == 0 || row == rows - 1 || column == width - 1;
                AtomicU32::new(if rim { heights[index] } else { f32::INFINITY }.to_bits())
            })
            .collect();

        let across = width.div_ceil(tile);
        let down = rows.div_ceil(tile);
        let mut sweeps = 0;
        loop {
            sweeps += 1;
            let moved = AtomicBool::new(false);
            for colour in 0..4 {
                let tiles: Vec<(usize, usize)> = (0..down)
                    .flat_map(|ty| (0..across).map(move |tx| (tx, ty)))
                    .filter(|(tx, ty)| (tx % 2) + 2 * (ty % 2) == colour)
                    .collect();
                tiles.par_iter().for_each(|&(tx, ty)| {
                    if one_tile(heights, &filled, width, rows, tile, tx, ty) {
                        moved.store(true, Ordering::Relaxed);
                    }
                });
            }
            if !moved.load(Ordering::Relaxed) {
                break;
            }
        }

        let out = filled
            .iter()
            .map(|bits| f32::from_bits(bits.load(Ordering::Relaxed)))
            .collect();
        (out, sweeps)
    }

    /// One tile's flood: a minimax Dijkstra over the tile, seeded from whatever
    /// its neighbours currently know, relaxing only cells inside it.
    fn one_tile(
        heights: &[f32],
        filled: &[AtomicU32],
        width: usize,
        rows: usize,
        tile: usize,
        tx: usize,
        ty: usize,
    ) -> bool {
        let (first_column, first_row) = (tx * tile, ty * tile);
        let last_column = (first_column + tile).min(width);
        let last_row = (first_row + tile).min(rows);
        let (span, high) = (last_column - first_column, last_row - first_row);
        if span == 0 || high == 0 {
            return false;
        }

        let at = |column: usize, row: usize| row * width + column;
        let read = |index: usize| f32::from_bits(filled[index].load(Ordering::Relaxed));
        let local = |column: usize, row: usize| (row - first_row) * span + (column - first_column);

        // The tile's own copy, so the inner loop is a walk over half a megabyte
        // rather than over a quarter of a gigabyte. This is most of the point:
        // the heap version's cost is that every neighbour it looks at is a
        // cache miss somewhere in 220 MB.
        let mut best: Vec<f32> = Vec::with_capacity(span * high);
        for row in first_row..last_row {
            for column in first_column..last_column {
                best.push(read(at(column, row)));
            }
        }

        let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<Reverse<u64>>, slot: usize, value: f32| {
            heap.push(Reverse((u64::from(ordered(value)) << 32) | slot as u64));
        };

        // Seed from the halo: every way into this tile from outside it.
        for row in first_row..last_row {
            for column in first_column..last_column {
                let edge = column == first_column
                    || row == first_row
                    || column + 1 == last_column
                    || row + 1 == last_row;
                if !edge {
                    continue;
                }
                let slot = local(column, row);
                for (dx, dy) in NEIGHBOURS {
                    let (nx, ny) = (column as i64 + dx, row as i64 + dy);
                    if nx < 0 || ny < 0 || nx >= width as i64 || ny >= rows as i64 {
                        continue;
                    }
                    let (nx, ny) = (nx as usize, ny as usize);
                    let outside =
                        nx < first_column || ny < first_row || nx >= last_column || ny >= last_row;
                    if !outside {
                        continue;
                    }
                    let over = read(at(nx, ny));
                    if !over.is_finite() {
                        continue;
                    }
                    let candidate = heights[at(column, row)].max(over);
                    if candidate < best[slot] {
                        best[slot] = candidate;
                        push(&mut heap, slot, candidate);
                    }
                }
            }
        }
        // ... and from what the tile already knew, so a value that arrived last
        // sweep can still travel inwards this one.
        for (slot, value) in best.iter().enumerate() {
            if value.is_finite() {
                push(&mut heap, slot, *value);
            }
        }

        let mut lowered = false;
        while let Some(Reverse(packed)) = heap.pop() {
            let slot = (packed & 0xffff_ffff) as usize;
            // The key holds `ordered` bits, not float bits -- comparing them
            // against `ordered(best[slot])` is the check for a stale entry, and
            // the value itself has to come back from `best` rather than from
            // the key.
            if (packed >> 32) as u32 != ordered(best[slot]) {
                continue;
            }
            let here = best[slot];
            let (column, row) = (first_column + slot % span, first_row + slot / span);
            for (dx, dy) in NEIGHBOURS {
                let (nx, ny) = (column as i64 + dx, row as i64 + dy);
                if nx < first_column as i64
                    || ny < first_row as i64
                    || nx >= last_column as i64
                    || ny >= last_row as i64
                {
                    continue;
                }
                let (nx, ny) = (nx as usize, ny as usize);
                let neighbour = local(nx, ny);
                let candidate = heights[at(nx, ny)].max(here);
                if candidate < best[neighbour] {
                    best[neighbour] = candidate;
                    push(&mut heap, neighbour, candidate);
                }
            }
        }

        for row in first_row..last_row {
            for column in first_column..last_column {
                let index = at(column, row);
                let value = best[local(column, row)];
                if ordered(value) < ordered(read(index)) {
                    filled[index].store(value.to_bits(), Ordering::Relaxed);
                    lowered = true;
                }
            }
        }
        lowered
    }
}
