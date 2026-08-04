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

use std::cmp::Reverse;
use std::collections::BinaryHeap;

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

/// Fills the hollows and works out where the water goes.
pub fn drainage(fields: &Fields) -> Drainage {
    let (width, rows) = (fields.width(), fields.rows());
    let count = width * rows;
    let heights: &[f32] = &fields.height.values;

    let mut filled = vec![0f32; count];
    // Which cell reached each one. The flood's own tree, kept because it is the
    // only thing that can route across a lake's flat surface.
    let mut reached_by = vec![u32::MAX; count];
    let mut seen = vec![false; count];
    let mut order: Vec<u32> = Vec::with_capacity(count);
    let mut heap: BinaryHeap<Reverse<u64>> = BinaryHeap::new();

    // The edge of the map is where water leaves, so that is where the flood
    // starts. A landscape with no outlet at all would otherwise fill to its own
    // rim and drown.
    for row in 0..rows {
        for column in 0..width {
            let edge = row == 0 || column == 0 || row == rows - 1 || column == width - 1;
            if !edge {
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

    // Where each cell drains to: the steepest neighbour strictly below it on
    // the filled surface, or -- on the flat of a lake, where there is no such
    // neighbour -- whichever cell the flood reached it from, which is by
    // construction on the way to the spill point.
    let mut drains_to = vec![u32::MAX; count];
    for row in 0..rows {
        for column in 0..width {
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
            drains_to[index] = if best != u32::MAX {
                best
            } else {
                reached_by[index]
            };
        }
    }

    // One backwards sweep of the pop order. Every cell is visited before the
    // cell it drains to, so a single pass is exact.
    let mut area = vec![1.0f32; count];
    for index in order.iter().rev() {
        let index = *index as usize;
        let into = drains_to[index];
        if into != u32::MAX {
            area[into as usize] += area[index];
        }
    }

    Drainage {
        filled,
        drains_to,
        area,
        order,
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
