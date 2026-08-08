//! The trees standing on a texel: how high they reach, and how much of it they
//! cover.
//!
//! Nothing here runs while a frame is being drawn. `terrain-generate` calls
//! [`baked`] once per texel of every level of the elevation and ground-cover
//! products, and the renderer then meets a tree by meeting the ground -- the
//! crowns are *in* the heights it was going to trace anyway, and the forest
//! colour is in the material it was going to read anyway.
//!
//! That is a deliberate reversal. An earlier version kept the elevation bare and
//! grew the crowns in the march, sphere-tracing a distance field per ray. It drew
//! well and it cost about three quarters of the frame -- and not for the reason
//! it looked: shortening the walk barely helped, because the expense was
//! *entering* the wooded path at all, a cover lookup and nine hashes at every
//! leaf texel a ray crossed, which at the finest level is once a metre through a
//! stand. Sampling the same field offline pays once per texel instead of once per
//! ray-texel, and the fine levels are where those two counts diverge most.
//!
//! This is still a *fake*, and knowing that is what keeps it cheap. It does not
//! model trees; it raises a height field where the ground cover says trees grow,
//! in a shape that reads as treetops from the air.
//!
//! # Where the trees are is the caller's question, not this crate's
//!
//! [`Cover`] is the whole of the input: how closed a stand is, and how well it is
//! growing. Nothing here decides where a forest is or reads a raster to find out.
//! That belongs to whoever is writing a product, because only they know what
//! their landscape is made of -- the generator asks its own classifier about the
//! very point it is writing, so a stand's edge lands exactly where the classifier
//! put it rather than on some stored grid it would have to interpolate back off.
//!
//! Trees are all one kind. Leaf type decided a crown's colour and shape while the
//! forest was read off material ids, and neither survived contact with a canopy
//! seen from a kilometre up: every stand read as one green, and the difference
//! between a spire and a dome is a couple of pixels. What is left is one crown
//! shape whose height comes from [`Cover::health`], which says far more about
//! what a forest looks like than its species does.
//!
//! # The stand has to look unplanted
//!
//! An earlier version put one crown at the middle of every cell of the lattice,
//! barely jittered, so that a single hash could answer -- and it drew a visible
//! square grid across every forest, which is the one thing a fake canopy cannot
//! afford to do. Irregularity comes from three things and it needs all three:
//! trunks jitter across the *whole* of their cell, cells stand empty in
//! proportion to the density, and crowns are wide enough to overlap their
//! neighbours. Overlapping is what costs the single lookup -- a crown reaching
//! out of its own cell has to be found from the cells beside it, so a point is
//! answered from the three-by-three around it.
//!
//! [`clump`] then multiplies the density by a slow field, so a stand thins and
//! thickens inside itself rather than being uniform out to its edge.
//!
//! # Band limiting is now the whole of the design
//!
//! The opposite of what this file used to say. While the march grew the crowns it
//! evaluated one function at every level, deliberately, so that ground did not
//! change height as the camera closed on it. A baked product cannot do that: a
//! texel is a texel, and what it should hold depends entirely on how much ground
//! it covers. [`baked`] takes the texel size for exactly that reason, and there
//! are two ways to get it wrong.
//!
//! Sample too coarsely and a crown's apex is missed. A cone that rises eight
//! metres per metre of ground clips hard -- a point sample at one metre lost
//! about three metres of peak and at two metres about six, so a stand *shrank* as
//! it crossed the ring where those two levels meet. That was the pop. [`baked`]
//! samples every texel finely enough to find the apex, so the clipping is about
//! equal at every level and nothing shrinks with distance.
//!
//! Average too honestly and a distant forest reads as a green-painted hillside.
//! The area mean of this canopy is about five metres where the crowns are twenty,
//! because coverage is under a half and a cone averages a third of its peak over
//! its own footprint. That mean is the right answer for a view looking straight
//! down and the wrong one for the grazing rays that are nearly all of a far
//! field. So a texel leans towards the tallest crown in it; see [`SILHOUETTE`].
//!
//! # Two answers, one walk
//!
//! [`Baked`] carries the height and the crown share together, because the height
//! goes into the elevation product and the share decides whether the ground-cover
//! product says `Canopy` -- and the nine hashes a crown costs are the expensive
//! part. Taking both at once also means a texel cannot be drawn as a tree and
//! painted as a meadow.

/// How far apart the cells that may hold a trunk are, in metres.
///
/// Not the spacing of the trees: cells stand empty in proportion to the
/// density, and the rest put their trunk anywhere inside themselves, so the
/// gaps between trunks run from touching to several cells. Seven metres is what
/// leaves a closed interior stand once those two have had their way with it.
pub const SPACING: f32 = 7.0;

/// The shortest and tallest a full-health crown gets, in metres.
///
/// The spread between them is per-tree rather than per-stand: the hash decides
/// where in the range each crown falls, so a stand is a range of ages rather
/// than one tree repeated. [`Cover::health`] then scales the whole range, which
/// is what makes a sheltered valley stand tall and a wind-scoured one stunted.
pub const SHORTEST: f32 = 15.0;
pub const TALLEST: f32 = 28.0;

/// How far a crown reaches from its trunk, in metres, at full size.
///
/// Two things pull on it. Wider crowns overlap more, which is what makes a
/// closed canopy rather than a field of separate spikes. But only the nine cells
/// around a point are searched, so a crown wider than half a cell could reach a
/// point from outside them and be missed -- a hole in the canopy that no test of
/// one cell could find. Half a cell is where those meet, and the assertion at
/// the foot of this file is what keeps it there.
pub const RADIUS: f32 = 3.5;

/// The crown's profile: `0` a straight-sided cone, `1` a dome.
///
/// Nearly a cone. Conifers are what this landscape is, and a cone silhouette is
/// most of what says "evergreen" at the distance any of this is seen from; the
/// little roundness knocks the apex off, which a perfect cone reads badly at.
pub const ROUNDNESS: f32 = 0.15;

/// Where the understorey sits, as a share of [`SHORTEST`].
///
/// A floor under the crowns, for the ground between them. Without it a stand
/// reads as spikes standing on bare earth. Scaled by density as well as by
/// health, so a clearing does not draw as a raised plate.
pub const FLOOR: f32 = 0.35;

/// How wide a band of density a crown grows in over, as a share of the whole.
///
/// A crown does not appear the moment the density passes its cell's threshold;
/// it grows in over this much density, in height and in width together. That is
/// what a real stand edge looks like -- smaller trees, not fewer full-sized
/// ones -- and it is also what hides the seam that would otherwise show.
///
/// The seam is worth spelling out, because it is the reason this is not a plain
/// comparison. A caller evaluates the cover once per texel and hands the same
/// value for every sample inside it, so two neighbouring texels can disagree by
/// a little about the density at one lattice cell that straddles them. With a
/// hard threshold that disagreement is a tree that exists on one side of the
/// texel wall and not the other -- a crown sliced vertically down the middle.
/// With a band it is a tree a few centimetres shorter on one side, which nothing
/// can see.
const EDGE: f32 = 0.30;

/// The fixed seed the lattice is drawn from.
///
/// A constant rather than a parameter because nothing that crosses a pipeline
/// boundary depends on it. The crowns go into the stored heights, and the max
/// pyramid is reduced from those heights afterwards, so a pyramid bounds
/// whatever this happened to grow without ever knowing the seed.
const SEED: u32 = 0x54_72_65_65;

/// Feature size of both noise fields, in metres.
///
/// About five crown spacings, which is the scale a stand has to thin and
/// thicken at to read as a forest rather than as one density painted over a
/// region. Much shorter and it fights the lattice; much longer and a whole
/// hillside is uniformly thick or uniformly thin.
const NOISE_WAVELENGTH: f32 = 34.0;

/// What [`clump`] multiplies a sampled density by, at its thinnest and thickest.
///
/// The thick end is the one that has to be argued for, because it is what makes
/// a forest read as a forest. A cell holds a full-sized tree only once the
/// density is [`EDGE`] past what that cell wants, and the wants run up to one,
/// so a *completely* closed lattice takes a density of `1 + EDGE`. Anything less
/// and the thickest part of the deepest stand still has gaps in it -- which is
/// what the old ceiling of `1.55` bought, at four fifths of a stand: `1.24`, a
/// little short of closing, everywhere, forever. At `1.9` a stand stored at
/// seven tenths closes where the field peaks, so the density a classifier writes
/// is the share of the *landscape* that is deep forest rather than a level the
/// canopy is capped at.
///
/// The thin end is what keeps the glades. It is a multiplier and not a floor, so
/// it thins a light stand towards nothing and a heavy one only to the middle of
/// its range.
const CLUMP_THINNEST: f32 = 0.6;
const CLUMP_THICKEST: f32 = 1.9;

/// The cover a point takes, once the stored field has been sampled and the
/// noise applied.
///
/// A stand, not a tree. The density says what share of the crown lattice holds
/// one and the health scales how tall they all get, and between them they are
/// the whole of what a caller has to know about a forest.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Cover {
    pub density: f32,
    pub health: f32,
}

/// Wellons' `lowbias32`.
///
/// The same mixer `crates/terrain-generate/src/noise.rs` uses, restated here
/// rather than shared so this crate depends on nothing. Its avalanche is
/// measured rather than assumed, which is what stops neighbouring cells of the
/// lattice from drawing correlated trunks and putting a grain of rows across a
/// stand.
fn mix(mut bits: u32) -> u32 {
    bits ^= bits >> 16;
    bits = bits.wrapping_mul(0x7feb_352d);
    bits ^= bits >> 15;
    bits = bits.wrapping_mul(0x846c_a68b);
    bits ^= bits >> 16;
    bits
}

/// One repeatable random word for a cell of a lattice.
///
/// The two coordinates go in one at a time, each through its own mixer.
/// Combining them first is cheaper and wrong in a way that shows: they would
/// then meet only through a single xor, whole diagonals of the lattice would
/// collide, and the forest would grow in stripes.
fn hash(x: i32, y: i32, seed: u32) -> u32 {
    let mut bits = seed.wrapping_mul(0x9e37_79b1);
    bits = mix(bits ^ (x as u32).wrapping_mul(0x3504_f333));
    bits = mix(bits ^ (y as u32).wrapping_mul(0xf1bb_cdcb));
    bits
}

/// Linear interpolation, exact at both ends.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Hermite ease, flat at both ends.
fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

/// Three independent smooth fields, each `0` to `1`, from one lattice.
///
/// One lattice and one set of four hashes for all three, split out of the same
/// words. That is three fields for the price of one, and they are independent
/// because the mixer's avalanche already made every output bit depend on every
/// input bit -- the same argument the crown's own fields rest on.
fn fields(x: f32, y: f32) -> [f32; 3] {
    let u = x / NOISE_WAVELENGTH;
    let v = y / NOISE_WAVELENGTH;
    let cell_x = u.floor();
    let cell_y = v.floor();
    let fx = fade(u - cell_x);
    let fy = fade(v - cell_y);
    let (cell_x, cell_y) = (cell_x as i32, cell_y as i32);

    let mut out = [0f32; 3];
    let corner = [
        hash(cell_x, cell_y, SEED),
        hash(cell_x + 1, cell_y, SEED),
        hash(cell_x, cell_y + 1, SEED),
        hash(cell_x + 1, cell_y + 1, SEED),
    ];
    for (field, slot) in out.iter_mut().enumerate() {
        let take = |bits: u32| ((bits >> (10 * field as u32)) & 0x3ff) as f32 * (1.0 / 1023.0);
        *slot = lerp(
            lerp(take(corner[0]), take(corner[1]), fx),
            lerp(take(corner[2]), take(corner[3]), fx),
            fy,
        );
    }
    out
}

/// What to multiply a sampled density by, so a stand thins and thickens inside
/// itself rather than being uniform out to its edge.
///
/// Reaches well above one: a stand stored at seven tenths closes completely in
/// its thickest parts, which is what a stand does. Density above one is harmless
/// -- every cell holds a tree and there is nothing further to add -- and it
/// never touches the ceiling, which depends on health alone.
pub fn clump(x: f32, y: f32) -> f32 {
    lerp(CLUMP_THINNEST, CLUMP_THICKEST, fields(x, y)[2])
}

/// How high the ground under a stand is drawn, in metres above the earth.
///
/// A floor under the crowns, closing the gaps between them so a stand does not
/// read as spikes standing on bare ground. It is scaled by the density as well
/// as by the health, so it fades out into a clearing rather than ending at one.
///
/// It is *not* canopy, and the difference is what [`Baked::share`] counts: a
/// sample standing on the understorey is forest floor, a few metres up, and it
/// wants the floor's own colour. Calling it canopy is what turned every wooded
/// slope into one flat green, because the gaps between the crowns are most of
/// what is visible from above.
fn understorey(cover: &Cover) -> f32 {
    FLOOR * SHORTEST * cover.health * cover.density.min(1.0)
}

/// The canopy over a point, in metres above the ground under it.
///
/// `x` and `y` are metres on the raster's own grid; which corner they are
/// measured from does not matter, so long as one caller measures them the same
/// way twice.
///
/// Nine cells are searched rather than one because crowns overlap: a tree wide
/// enough to close a canopy reaches out of its own cell, so a point has to ask
/// the three-by-three around it. Those nine hashes are the expensive part of
/// this crate, and the reason [`baked`] takes both its answers from one walk.
fn nearby(x: f32, y: f32, cover: &Cover) -> f32 {
    let cell_x = (x / SPACING).floor() as i32;
    let cell_y = (y / SPACING).floor() as i32;

    if cover.health <= 0.0 || cover.density <= 0.0 {
        return 0.0;
    }
    let mut found = understorey(cover);

    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = cell_x + dx;
            let cy = cell_y + dy;
            let bits = hash(cx, cy, SEED);

            // Four fields out of one word. Splitting a hash beats taking four
            // of them: a crown costs two mixers rather than eight, and the
            // fields are independent because the mixer's avalanche already made
            // every output bit depend on every input bit.
            let jitter_x = (bits & 0x3ff) as f32 * (1.0 / 1024.0);
            let jitter_y = ((bits >> 10) & 0x3ff) as f32 * (1.0 / 1024.0);
            let size = ((bits >> 20) & 0x3f) as f32 * (1.0 / 64.0);
            let wants = ((bits >> 26) & 0x3f) as f32 * (1.0 / 64.0);

            // How much of a tree this cell grows, if any. See `EDGE`.
            let grow = fade(((cover.density - wants) / EDGE).clamp(0.0, 1.0));
            if grow <= 0.0 {
                continue;
            }

            // Anywhere in its own cell, which is what stops the stand drawing
            // as a grid.
            let trunk_x = (cx as f32 + jitter_x) * SPACING;
            let trunk_y = (cy as f32 + jitter_y) * SPACING;
            // A short tree is a narrow one. Tying the two together stops the
            // stand looking like one tree scaled up and down, and keeps every
            // radius under `RADIUS`, which the clamp above relies on.
            let scale = grow * (0.72 + 0.28 * size);
            let radius = (RADIUS * scale).max(1.0 / 1024.0);
            let height = cover.health * lerp(SHORTEST, TALLEST, size) * grow;

            let offset_x = x - trunk_x;
            let offset_y = y - trunk_y;
            let u = (offset_x * offset_x + offset_y * offset_y).sqrt() / radius;
            if u < 1.0 {
                let cone = 1.0 - u;
                let dome = (1.0 - u * u).max(0.0).sqrt();
                found = found.max(height * lerp(cone, dome, ROUNDNESS));
            }
        }
    }
    found
}

/// The canopy above the ground at one point, in metres.
///
/// The field itself, with no texel in the question. [`baked`] is what a texel
/// carries and it is the only caller outside the tests -- this is here so the
/// tests can ask about the field rather than about a block of it.
#[cfg(test)]
fn canopy(x: f32, y: f32, cover: &Cover) -> f32 {
    nearby(x, y, cover)
}

/// What one texel of a stored product carries, once the crowns in it have been
/// looked at.
///
/// Two answers out of one walk, because both callers ask about the same block of
/// ground and the nine hashes a crown costs are the expensive part. The heights
/// take [`Baked::height`] and the ground cover takes [`Baked::share`], so a texel
/// cannot end up drawn as a tree and painted as a meadow.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Baked {
    /// The canopy this texel stands for, in metres above the earth under it.
    pub height: f32,
    /// How much of the texel lies under some crown, from zero to one.
    pub share: f32,
}

/// What a texel of this size carries, given the cover on it.
///
/// This is the whole of the canopy now. `terrain-generate` calls it once per
/// texel of every level of both products, and nothing at run time grows a tree at
/// all: a ray meets a crown by meeting the ground, and a pixel is painted as
/// forest because the material under it says so. What that buys is the march's
/// hottest loop -- the walk it replaced was re-entered at every leaf texel a ray
/// crossed, once a metre through a stand.
///
/// # Why the answer is a conditional mean
///
/// The obvious two candidates both fail, and they fail in opposite directions.
///
/// The plain average of the block is the honest downsample and it is *stable*:
/// averaging is scale-invariant, so a stand keeps the same height however coarse
/// the texels holding it get, and no ring shows a step. But the average of this
/// canopy is about five metres where the crowns are twenty, because coverage is
/// under a half and a cone averages a third of its peak over its own footprint.
/// It draws a distant forest as a green-painted hillside. That is what it did.
///
/// Leaning from the average towards the tallest crown in the block fixes the
/// look and breaks the stability, because a maximum is *not* scale-invariant --
/// a wider block has more chances to land on an apex, so it climbs with the texel
/// while the mean stays put. Measured, a closed stand went 7.2, 8.1, 10.3, 15.0,
/// 18.7 m across texel sizes 1 to 16 m: an eleven-metre climb, a step at every
/// ring, and forest that grows as you fly away from it.
///
/// What satisfies both is an order statistic. The mean of the tallest
/// [`SILHOUETTE`] of the block is scale-invariant for the same reason the plain
/// mean is -- it is an average, just of a selected part of the same distribution
/// -- while sitting up among the crowns rather than down among the gaps. It is
/// the height a grazing ray meets, which is what nearly every ray reaching a
/// coarse texel is, and it is the same height at every level.
///
/// # Why the sampling is what it is
///
/// A crown is a cone rising eight metres for every metre of ground, so a sample
/// that misses the apex misses the height badly. Point sampling clipped a peak by
/// about three metres at a one-metre texel and six at a two-metre one, and that
/// *difference* was the other half of the pop: a stand shrank as it crossed the
/// ring between those levels. [`samples_across`] keeps the spacing under a
/// quarter of a crown radius at every level, so the clipping is about equal
/// everywhere and nothing shrinks with distance.
pub fn baked(x: f32, y: f32, cover: &Cover, texel_metres: f32) -> Baked {
    let across = samples_across(texel_metres);
    let step = texel_metres / across as f32;
    // Sample centres, so the block is symmetric about the texel and a texel twice
    // the size of its neighbour covers the same ground its four children did.
    let first = 0.5 * step - 0.5 * texel_metres;
    let floor = understorey(cover);
    let mut heights = Vec::with_capacity((across * across) as usize);
    let mut under = 0u32;
    for row in 0..across {
        for column in 0..across {
            let here = nearby(
                x + first + column as f32 * step,
                y + first + row as f32 * step,
                cover,
            );
            heights.push(here);
            // A sample is under a crown when it stands on something taller than
            // the forest floor.
            under += u32::from(here > floor);
        }
    }
    let samples = heights.len();
    // Descending, so the tallest share is a prefix. `total_cmp` rather than
    // `partial_cmp`: these are heights and none of them is a NaN, but sorting is
    // not the place to find that out.
    heights.sort_unstable_by(|a, b| b.total_cmp(a));
    let taken = ((samples as f32 * SILHOUETTE).ceil() as usize).clamp(1, samples);
    let tall: f32 = heights[..taken].iter().sum::<f32>() / taken as f32;
    Baked {
        height: tall,
        share: under as f32 / samples as f32,
    }
}

/// What share of a block, taken from the tallest end, [`baked`] averages.
///
/// A seventh. The whole block would be the plain average, which draws a forest
/// twenty metres short; a single sample would be the maximum, which is not
/// scale-invariant and puts a step at every ring. Between those two the choice is
/// a trade, and the measurement says it is a slack one: a closed stand comes out
/// at 7.2, 8.1, 10.0, 12.3, 13.6 m across texel sizes 1 to 16 m, and every one of
/// those handovers is about half a pixel at the distance its ring sits at. Buying
/// four more metres of distant canopy costs a step nobody can see, so it is
/// bought. See `a_stand_holds_its_height_as_the_texels_coarsen`.
///
/// It is also what decides how a stand thins. A sparse scattering has few tall
/// samples, so its tallest quarter reaches down into the gaps and the texel sits
/// low; a closed stand's tallest quarter is all crown. That is the behaviour a
/// plain maximum loses, by turning a handful of trees into a plateau.
const SILHOUETTE: f32 = 0.15;

/// How many samples across a texel [`baked`] takes.
///
/// A quarter of a crown radius apart, with a floor of four however fine the texel
/// is. Both halves of that serve the same end: a crown apex has to be found, or
/// the height is clipped by an amount that depends on the texel size, and a
/// clipping that varies by level is a stand that changes height at every ring.
///
/// Capped at thirty-two because the answer stops moving. A conditional mean over
/// a thousand samples does not care about one more, and a two-hundred-metre texel
/// is drawn from four hundred kilometres away.
fn samples_across(texel_metres: f32) -> u32 {
    let wanted = (texel_metres / (0.25 * RADIUS)).ceil();
    (wanted as u32).clamp(4, 32)
}

/// The crown share at which a texel is painted as canopy rather than as the
/// ground cover under it.
///
/// A quarter, and deliberately well under a half. The share is a *plan view* one
/// and it saturates at about `0.36`: round crowns on a square lattice cannot
/// close a plan view however many of them there are, so a threshold of a half
/// would paint every stand on the landscape as bare ground. Against that
/// ceiling a quarter is high -- it wants a density around `0.8`, which after
/// [`clump`] is the thicker parts of a stand the classifier calls wooded rather
/// than the whole of it.
///
/// High on purpose, because what a viewer sees is not the plan view. Most rays
/// that reach a stand are grazing ones, and a grazing ray meets crown flanks
/// rather than the gaps between them, so it sees far more crown than the share
/// suggests; painting on the plan-view half would spread canopy colour across
/// ground that reads as forest floor from every angle it is actually seen at.
///
/// A judgement call, and the one thing here a render can overrule.
pub const PAINTED: f32 = 0.25;

/// A crown reaches at most half a cell, or the nine cells [`nearby`] searches
/// would not be all the cells that can cover a point -- and a crown missed
/// because its trunk sat two cells away is a hole in the canopy that nothing
/// downstream could tell from a clearing.
const _: () = assert!(2.0 * RADIUS <= SPACING);

/// The understorey is under the trees, not over them.
const _: () = assert!(FLOOR * SHORTEST < TALLEST);

/// The thickest part of a stand stored at seven tenths holds a tree in every
/// cell of the lattice, which is the claim [`clump`] is documented by and the
/// reason a forest can read as closed at all.
const _: () = assert!(0.7 * CLUMP_THICKEST >= 1.0 + EDGE);

/// The thin end thins and does not erase: a stand's glades are gaps in a forest,
/// not holes in the product.
const _: () = assert!(CLUMP_THINNEST > 0.0 && CLUMP_THINNEST < 1.0);

#[cfg(test)]
mod tests {
    use super::*;

    /// A closed stand in good health, which is what most of a forest is.
    const CLOSED: Cover = Cover {
        density: 0.9,
        health: 1.0,
    };

    /// Walks a wide patch of ground, finely enough to catch a crown between two
    /// samples.
    fn over_a_stand(mut visit: impl FnMut(f32, f32)) {
        // Strides that do not divide the spacing, so the samples do not walk
        // the lattice in step with it and measure one tree over and over.
        for i in 0..1000 {
            for j in 0..1000 {
                visit(i as f32 * 0.37, j as f32 * 0.41);
            }
        }
    }

    /// Integrates the crown field over a block of ground at one density.
    ///
    /// Returns the average canopy height per unit of health, and the share of
    /// the ground standing under a crown rather than on the forest floor. The
    /// health is fixed at one because it factors out of both.
    ///
    /// This is the independent statement of what [`baked`] answers by sampling a
    /// texel, and the strides are chosen not to divide [`SPACING`] so the walk
    /// drifts across the lattice instead of measuring the same place in every
    /// cell.
    fn integrate(density: f32, steps: u32, stride: (f32, f32), from: (f32, f32)) -> (f32, f32) {
        let cover = Cover {
            density,
            health: 1.0,
        };
        let floor = understorey(&cover);
        let mut height = 0f64;
        let mut under = 0u64;
        for i in 0..steps {
            for j in 0..steps {
                let x = from.0 + i as f32 * stride.0;
                let y = from.1 + j as f32 * stride.1;
                let found = nearby(x, y, &cover);
                height += f64::from(found);
                under += u64::from(found > floor);
            }
        }
        let samples = f64::from(steps) * f64::from(steps);
        ((height / samples) as f32, (under as f64 / samples) as f32)
    }

    /// What [`baked`] reports a coarse texel's crown share to be is the share the
    /// field actually has.
    ///
    /// This is the number that decides whether a texel of the ground-cover
    /// product says `Canopy`, so getting it wrong paints hillsides the wrong
    /// colour and nothing reports it. Checked against an independent integral
    /// over the same ground rather than against a remembered table.
    #[test]
    fn a_coarse_texel_reports_the_share_the_field_has() {
        for density in [0.2f32, 0.45, 0.7, 0.95, 1.3] {
            let cover = Cover {
                density,
                health: 1.0,
            };
            let (mut sum, mut count) = (0.0f64, 0u32);
            for i in 0..60 {
                for j in 0..60 {
                    sum += f64::from(baked(i as f32 * 64.0, j as f32 * 64.0, &cover, 64.0).share);
                    count += 1;
                }
            }
            let got = (sum / f64::from(count)) as f32;
            let want = integrate(density, 900, (0.31, 0.29), (0.0, 0.0)).1;
            assert!(
                (got - want).abs() < 0.02,
                "a density of {density} bakes a share of {got:.3} where the field \
                 measures {want:.3}",
            );
        }
    }

    /// The share is what [`PAINTED`] is compared against, so it has to reach past
    /// it for a stand anyone would call a wood, and stay under it for scattered
    /// trees. A threshold nothing crosses would paint every forest as bare
    /// ground; one everything crosses would paint every meadow as forest.
    #[test]
    fn the_paint_threshold_falls_inside_the_range_the_share_covers() {
        let share = |density| integrate(density, 700, (0.31, 0.29), (0.0, 0.0)).1;
        assert!(
            share(0.2) < PAINTED,
            "scattered trees at 0.2 already cover {:.3}, past the {PAINTED} paint \
             threshold",
            share(0.2),
        );
        assert!(
            share(0.9) > PAINTED,
            "a closed stand only covers {:.3}, under the {PAINTED} paint threshold",
            share(0.9),
        );
    }

    /// A stand does not visibly change height as the texels holding it coarsen.
    ///
    /// This is the pop-in invariant, stated where it can be measured. The levels
    /// are concentric rings around the camera, so ground crosses from one texel
    /// size to the next as the aircraft moves; if a stand's height moved with it,
    /// a forest would grow or shrink at that ring. It used to shrink, badly --
    /// point sampling clipped a crown's apex by about three metres at one metre
    /// and six at two, and a cone rising eight metres per metre of ground makes
    /// that a large number.
    ///
    /// **Visibly** is the whole of the test. The height does climb with the texel
    /// -- an order statistic of a one-metre block is nearly that block's own
    /// value, while over sixteen metres it is a real statistic of the stand, and
    /// nothing but a plain average escapes that. What matters is not the total
    /// climb but each step measured against the distance its own ring sits at,
    /// because a ring twice as far away subtends half the angle.
    #[test]
    fn a_stand_holds_its_height_as_the_texels_coarsen() {
        // Both restated rather than imported: this crate depends on nothing, and
        // these belong to the renderer. A level reaches `Residency::reach_texels`
        // of its own texels, so the ring where texels of `t` metres hand over to
        // `2 * t` sits at `REACH * t` metres; `PIXEL` is `Residency::pixel_angle`
        // at 1080p and sixty degrees.
        const REACH: f32 = 1536.0;
        const PIXEL: f32 = 1.069e-3;

        let average = |texel: f32| {
            let (mut sum, mut count) = (0.0f64, 0u32);
            for i in 0..120 {
                for j in 0..120 {
                    // Strides that do not divide the texel, so the blocks land
                    // all over the lattice rather than in step with it.
                    sum +=
                        f64::from(baked(i as f32 * 37.0, j as f32 * 41.0, &CLOSED, texel).height);
                    count += 1;
                }
            }
            (sum / f64::from(count)) as f32
        };

        let sizes = [1.0f32, 2.0, 4.0, 8.0, 16.0];
        let heights: Vec<f32> = sizes.iter().map(|t| average(*t)).collect();
        for (index, pair) in heights.windows(2).enumerate() {
            let texel = sizes[index];
            let step = (pair[1] - pair[0]).abs();
            // How many metres one pixel covers where these two levels meet.
            let pixel = REACH * texel * PIXEL;
            assert!(
                step < pixel,
                "handing {texel} m texels over to {} m moves a closed stand by \
                 {step:.2} m at {:.0} m, where a pixel is {pixel:.2} m -- \
                 {:.1} pixels of step. Heights were {heights:?}",
                texel * 2.0,
                REACH * texel,
                step / pixel,
            );
        }
        // And it climbs rather than wandering, so no ring can hand a stand back
        // down again.
        assert!(
            heights.windows(2).all(|pair| pair[1] >= pair[0]),
            "a stand's height is not monotone in the texel size: {heights:?}",
        );
    }

    /// A texel many crowns wide stands nearer the treetops than the average.
    ///
    /// The whole reason [`SILHOUETTE`] exists. A coarse texel that took the
    /// average would draw distant forest twenty metres short -- as a green
    /// hillside rather than as trees -- because the average is what a view
    /// looking straight down sees and the far field is nearly all grazing rays.
    #[test]
    fn a_coarse_texel_stands_near_the_treetops() {
        let (mut sum, mut count) = (0.0f64, 0u32);
        for i in 0..200 {
            for j in 0..200 {
                sum += f64::from(baked(i as f32 * 31.0, j as f32 * 29.0, &CLOSED, 32.0).height);
                count += 1;
            }
        }
        let got = (sum / f64::from(count)) as f32;
        // The average of the same field over the same ground: what a texel would
        // carry with no lean at all, and what it used to carry.
        let flat = integrate(CLOSED.density, 700, (0.53, 0.47), (0.0, 0.0)).0 * CLOSED.health;
        assert!(
            got > 2.0 * flat,
            "a 32 m texel of closed forest bakes {got:.2} m, not much over the \
             {flat:.2} m the same ground averages -- the lean is not doing its job",
        );
        let peak = TALLEST * CLOSED.health;
        assert!(
            got <= peak,
            "a 32 m texel bakes {got:.2} m, over the {peak:.2} m tallest crown",
        );
    }

    /// The invariant the max pyramid rests on, asked of the baked heights.
    ///
    /// [`baked`] blends an average and a maximum over samples of [`canopy`], so
    /// it cannot exceed what [`canopy`] can -- but the pyramid is what a ray
    /// tests against before it looks at any height, and the baked levels are
    /// where the ray meets trees by meeting the ground, so this is worth stating
    /// rather than deriving.
    #[test]
    fn nothing_baked_stands_above_the_ceiling_over_it() {
        for health in [0.0f32, 0.2, 0.5, 1.0] {
            for density in [0.0f32, 0.3, 0.9, 1.0, 1.3, 1.9] {
                let cover = Cover { density, health };
                let bound = TALLEST * health;
                for texel in [1.0f32, 2.0, 8.0, 32.0, 128.0] {
                    for step in 0..2000 {
                        let (x, y) = (step as f32 * 0.37, step as f32 * -0.41 + 13.0);
                        let got = baked(x, y, &cover, texel).height;
                        assert!(
                            got <= bound,
                            "a {texel} m texel of {density} at {health} baked {got} m \
                             through a ceiling of {bound} m",
                        );
                    }
                }
            }
        }
    }

    /// More density is more of the ground under a crown, up to the point every
    /// cell holds a full-sized tree and there is nothing left to add. A share
    /// that dipped anywhere would paint a thicker stand as bare ground while the
    /// thinner one beside it drew as forest.
    #[test]
    fn the_share_never_falls_as_the_density_rises() {
        let mut last = 0.0f32;
        for step in 0..=60 {
            let density = step as f32 * 0.025;
            let cover = Cover {
                density,
                health: 1.0,
            };
            let (mut sum, mut count) = (0.0f64, 0u32);
            for i in 0..24 {
                for j in 0..24 {
                    sum += f64::from(baked(i as f32 * 64.0, j as f32 * 64.0, &cover, 64.0).share);
                    count += 1;
                }
            }
            let now = (sum / f64::from(count)) as f32;
            assert!(
                now >= last - 0.005,
                "at a density of {density} the share fell to {now} from {last}",
            );
            last = last.max(now);
        }
    }

    /// Bare ground grows nothing, at any texel size. A clearing baked as a
    /// plateau would be a step in the middle of a meadow.
    #[test]
    fn nothing_grows_where_nothing_grows() {
        let bare = Cover {
            density: 0.0,
            health: 0.0,
        };
        let dead = Cover {
            density: 1.0,
            health: 0.0,
        };
        for texel in [1.0f32, 4.0, 64.0] {
            for cover in [&bare, &dead] {
                assert_eq!(baked(123.0, -456.0, cover, texel).height, 0.0);
                assert_eq!(baked(123.0, -456.0, cover, texel).share, 0.0);
            }
        }
    }

    /// The invariant the whole max pyramid rests on. A ceiling below the surface
    /// it is supposed to bound is a ray passing through a tree.
    #[test]
    fn the_ceiling_is_never_below_the_canopy_under_it() {
        for health in [0.2f32, 0.5, 0.75, 1.0] {
            for density in [0.1f32, 0.5, 0.9, 1.0, 1.55] {
                let cover = Cover { density, health };
                // The roof over a stand of one cover is that cover's health,
                // whatever its neighbours hold.
                let bound = TALLEST * health;
                over_a_stand(|x, y| {
                    let got = canopy(x, y, &cover);
                    assert!(
                        got <= bound,
                        "a canopy of {density} at {health} reached {got} m at ({x}, {y}), \
                         over a ceiling of {bound} m"
                    );
                });
            }
        }
    }

    /// Ground the product says holds nothing grows nothing, whatever else is
    /// set: the density and the health are each a veto.
    #[test]
    fn ground_with_no_trees_on_it_carries_no_canopy() {
        for cover in [
            Cover {
                density: 0.0,
                health: 0.0,
            },
            Cover {
                density: 0.0,
                health: 1.0,
            },
            Cover {
                density: 1.0,
                health: 0.0,
            },
        ] {
            over_a_stand(|x, y| {
                assert_eq!(canopy(x, y, &cover), 0.0, "{cover:?} grew something");
            });
        }
    }

    /// Density is what it says it is: more of it means more trees on the same
    /// ground, monotonically, or a stand's edge would not read as an edge.
    #[test]
    fn a_denser_stand_holds_more_trees() {
        let mut previous = 0.0;
        for density in [0.0f32, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let cover = Cover {
                density,
                health: 1.0,
            };
            let mut covered = 0u32;
            let mut total = 0u32;
            over_a_stand(|x, y| {
                total += 1;
                if nearby(x, y, &cover) > understorey(&cover) {
                    covered += 1;
                }
            });
            let share = f64::from(covered) / f64::from(total);
            assert!(
                share >= previous,
                "a stand of {density} covers {share:.3} of the ground where a thinner one \
                 covered {previous:.3}"
            );
            previous = share;
        }
        // Crowns are round and the lattice is square, so even with a tree in
        // every cell they cover barely a third of the ground in plan view --
        // which is what the understorey is for, and why [`PAINTED`] is well
        // under a half.
        assert!(
            previous > 0.28,
            "even a full stand covers only {previous:.3} of its ground"
        );
    }

    /// Health is the height of the forest, and it has to scale the whole of it
    /// -- the crowns and the floor under them alike -- or a sick stand would
    /// draw as full-height trees standing on a lowered floor.
    #[test]
    fn health_scales_the_whole_canopy() {
        let (mut sick, mut well) = (0.0f64, 0.0f64);
        over_a_stand(|x, y| {
            sick += f64::from(canopy(
                x,
                y,
                &Cover {
                    density: 0.9,
                    health: 0.5,
                },
            ));
            well += f64::from(canopy(x, y, &CLOSED));
        });
        assert!(
            (sick / well - 0.5).abs() < 1e-3,
            "half health averaged {:.4} of full health",
            sick / well
        );
    }

    /// Crowns are objects, so the field has to fall away between them -- and it
    /// has to reach a full-height tree somewhere, or the tallest tree the
    /// pyramid makes room for is room wasted on every cell of the forest.
    #[test]
    fn the_canopy_is_lumpy_and_reaches_the_trees_it_claims() {
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        over_a_stand(|x, y| {
            let got = canopy(x, y, &CLOSED);
            low = low.min(got);
            high = high.max(got);
        });
        assert_eq!(
            low,
            understorey(&CLOSED),
            "the stand never came down to the understorey"
        );
        assert!(
            high > 0.95 * TALLEST,
            "the stand never reached a full-height tree: {high} m of {TALLEST}"
        );
    }

    /// [`baked`] tells a crown from the floor under it by comparing the field
    /// against the understorey, so the floor has to *be* the floor: nothing
    /// anywhere may come out below it, and a good deal has to come out above.
    ///
    /// Too low and the gaps between the trunks paint as canopy, which is a
    /// wooded slope drawn as one flat green. Too high and the crowns paint as
    /// ground.
    #[test]
    fn the_understorey_is_the_floor_of_the_field_and_not_all_of_it() {
        for cover in [
            CLOSED,
            Cover {
                density: 0.3,
                health: 0.17,
            },
            Cover {
                density: 1.0,
                health: 1.0,
            },
        ] {
            let floor = understorey(&cover);
            let (mut under, mut over) = (0u32, 0u32);
            over_a_stand(|x, y| {
                let got = canopy(x, y, &cover);
                assert!(
                    got >= floor,
                    "{cover:?} came down to {got} m under a floor of {floor} m at ({x}, {y})"
                );
                if got > floor {
                    over += 1;
                } else {
                    under += 1;
                }
            });
            // A sparse stand really does spend most of its ground on the
            // floor -- a twentieth of a million samples is the krummholz
            // case, and it is the one this most has to get right.
            assert!(
                over > 20_000,
                "{cover:?} only reached above its floor {over} times"
            );
            assert!(
                under > 100_000,
                "{cover:?} only sat on its floor {under} times"
            );
        }
    }

    /// The stand must not draw as a grid, which is what it did when a trunk sat
    /// at the middle of every cell.
    ///
    /// Measured as the canopy over the lattice's own lines against the canopy
    /// over its cell middles. A planted stand is tall in the middles and short
    /// on the lines; an unplanted one cannot tell them apart, because a trunk is
    /// as likely to sit on a line as anywhere else.
    #[test]
    fn the_trunks_do_not_line_up_with_the_lattice() {
        let (mut middles, mut lines) = (0.0f64, 0.0f64);
        let mut count = 0u32;
        for i in 0..200 {
            for j in 0..200 {
                let (cx, cy) = (i as f32 * SPACING, j as f32 * SPACING);
                middles += f64::from(canopy(cx + SPACING * 0.5, cy + SPACING * 0.5, &CLOSED));
                lines += f64::from(canopy(cx, cy, &CLOSED));
                count += 1;
            }
        }
        let (middles, lines) = (middles / f64::from(count), lines / f64::from(count));
        assert!(
            (middles - lines).abs() < 0.1 * f64::from(TALLEST),
            "the stand averages {middles:.2} m over its cell middles and {lines:.2} m over \
             its lattice lines, which is a planted grid"
        );
    }

    /// The lattice runs through the origin and on into negative ground, and a
    /// hash of a negative cell must be as good as one of a positive cell. A seam
    /// on the axes would draw as a straight line of identical trees running the
    /// width of the raster.
    #[test]
    fn the_forest_crosses_the_origin_without_a_seam() {
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in -600..600 {
            for j in -600..600 {
                let got = canopy(i as f32 * 0.37, j as f32 * 0.41, &CLOSED);
                low = low.min(got);
                high = high.max(got);
            }
        }
        assert!(
            high - low > 0.4 * TALLEST,
            "the forest around the origin spread only {} m",
            high - low
        );
    }

    /// Tiles and levels are generated independently, so the only
    /// thing that makes them join is that the canopy at a position does not
    /// depend on who asked.
    #[test]
    fn the_canopy_at_a_position_does_not_depend_on_who_asks() {
        for x in [0.0f32, -7.0, 511.5, 512.5, 8191.5, 49151.5] {
            let first = canopy(x, 1024.5, &CLOSED);
            let second = canopy(x, 1024.5, &CLOSED);
            assert_eq!(first, second, "at x = {x}");
        }
    }

    /// The clumping has to be smooth, or the thinning and thickening it shapes a
    /// stand with would break into the lattice it is drawn on -- which is the
    /// artefact it exists to hide.
    #[test]
    fn the_noise_field_does_not_jump_between_neighbouring_points() {
        for i in 0..4000 {
            let (x, y) = (i as f32 * 0.31 - 600.0, i as f32 * -0.17 + 90.0);
            let step = 0.05;
            assert!(
                (clump(x, y) - clump(x + step, y)).abs() < 0.05,
                "the clumping jumped at ({x}, {y})"
            );
        }
    }

    /// What the thick end of the clumping is *for*: ground the classifier calls
    /// mostly wooded has to draw as a closed canopy where the field peaks, or a
    /// forest is an open woodland everywhere and no density can say otherwise.
    ///
    /// Measured as the share of the ground inside a crown, against the same
    /// stand with no clumping on it. The ceiling is the lattice's, not the
    /// density's -- round crowns on a square grid leave corners uncovered
    /// however many of them there are -- so this asks for a good deal more than
    /// the unclumped stand rather than for all of the ground.
    #[test]
    fn the_thickest_clumping_closes_a_stand_that_is_mostly_wooded() {
        let covered = |density: f32| {
            let cover = Cover {
                density,
                health: 1.0,
            };
            let (mut inside, mut total) = (0u32, 0u32);
            over_a_stand(|x, y| {
                total += 1;
                if nearby(x, y, &cover) > understorey(&cover) {
                    inside += 1;
                }
            });
            f64::from(inside) / f64::from(total)
        };
        let stand = 0.7;
        let (plain, thickest) = (covered(stand), covered(stand * CLUMP_THICKEST));
        assert!(
            thickest > 0.35,
            "a stand of {stand} at its thickest still covers only {thickest:.3} of its ground"
        );
        assert!(
            thickest > plain + 0.12,
            "clumping a stand of {stand} took it from {plain:.3} to {thickest:.3}, which is not \
             the difference between an open woodland and a closed one"
        );
    }

    /// The clumping has to reach above one somewhere, or a stand can never
    /// close, and it must never go negative, or it would erase one.
    #[test]
    fn the_clumping_thickens_as_well_as_thins() {
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..2000 {
            for j in 0..200 {
                let got = clump(i as f32 * 0.53 - 500.0, j as f32 * 1.7 - 170.0);
                low = low.min(got);
                high = high.max(got);
            }
        }
        assert!(low > 0.0, "the clumping fell to {low}");
        assert!(high > 1.0, "the clumping never reached one: {high}");
    }
}
