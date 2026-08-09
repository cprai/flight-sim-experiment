//! The stones lying on a texel: how high they stand, and how much of it they
//! hide.
//!
//! The canopy's method, applied to rock. `terrain-generate` calls [`baked`] once
//! per texel of every level of the elevation and ground-cover products, the
//! stones go into the heights the renderer was going to march anyway, and a ray
//! meets a boulder by meeting the ground. Nothing here runs while a frame is
//! being drawn, and `terrain-canopy`'s module doc argues at length for why --
//! the same argument, so it is not restated.
//!
//! # Where the stones are is the caller's question, not this crate's
//!
//! [`Scatter`] is the whole of the input: how much of the ground holds a
//! boulder, how much holds rubble, and how big the stones are. Nothing here
//! decides where a talus slope is or reads a raster to find out. That belongs to
//! whoever is writing a product, because only they know what their landscape is
//! made of -- and the generator asks its own classifier about the very point it
//! is writing, so a boulder field's edge lands exactly where the classifier put
//! it rather than on some stored grid it would have to interpolate back off.
//!
//! # Two sizes, because one is not a landscape
//!
//! A mountainside is strewn at two scales that do not blend into each other: the
//! blocks that came off the cliff, metres across and tens of metres apart, and
//! the rubble they broke down into, which covers everything. Drawing one size
//! with a wide spread of radii gives neither -- it gives a field of medium
//! stones -- so there are two lattices and the assertion at the foot of this
//! file keeps them apart.
//!
//! Neither is small. A texel of the base level is a metre of ground, so nothing
//! narrower than two metres can be drawn at all; [`RUBBLE_RADIUS`] is set so the
//! smaller class is a couple of texels across and no finer. Genuine gravel would
//! be a constant lift of a few centimetres, which is invisible, and the roughness
//! it stands for is already in the texture octaves `terrain-generate`'s `detail`
//! sums into every steep slope.
//!
//! # A stone is not a tree, in three ways
//!
//! Most of this is `terrain-canopy` with different numbers, and the places it is
//! not are the places worth reading.
//!
//! **A stone is a dome.** A crown is very nearly a cone, because a conifer
//! silhouette is what says "evergreen" from a kilometre up. A boulder is a
//! squashed ellipsoid, so [`ROUNDNESS`] sits at the other end of the same
//! interpolation.
//!
//! **There is no understorey.** A stand of trees is a *raised surface* with a
//! forest floor a few metres up, and `terrain-canopy` puts a floor under its
//! crowns so that a wood does not draw as spikes standing on bare earth. A
//! boulder field is stones lying on ground you can see between: the gaps are the
//! ground, at the ground's own height, and a floor here would draw every talus
//! slope as a raised plate with lumps on it.
//!
//! **The aggregate barely leans.** [`SILHOUETTE`] is the fraction of a texel,
//! taken from the tall end, that [`baked`] averages, and `terrain-canopy` sets it
//! low -- a seventh -- because the honest area mean draws a distant forest as a
//! green-painted hillside twenty metres short of its own treetops. That failure
//! mode does not exist here. A distant boulder field *should* read as a rocky
//! slope, which is what it is, and a dome averages two thirds of its peak over
//! its own footprint where a cone averages a third. So the lean is small, and
//! what it buys is the grazing ray: nearly every ray reaching a coarse texel
//! meets stone flanks rather than the gaps between them.
//!
//! # Two answers, one walk
//!
//! [`Baked`] carries the height and both shares together, because the eighteen
//! hashes a sample costs are the expensive part and all three callers ask about
//! the same block of ground. Taking them at once also means a texel cannot be
//! raised as a boulder and painted as a meadow.

/// How far apart the cells that may hold a boulder are, in metres.
///
/// Not the spacing of the boulders: cells stand empty in proportion to the
/// density, and the rest put their stone anywhere inside themselves. Twenty-four
/// metres is what leaves a field that reads as scattered blocks at the densities
/// a classifier can reach, rather than as a cobbled street.
pub const BOULDER_SPACING: f32 = 24.0;

/// How far apart the cells that may hold a piece of rubble are, in metres.
pub const RUBBLE_SPACING: f32 = 3.0;

/// How far a stone of each class reaches from its middle, in metres, at full
/// size.
///
/// Bounded above by half the spacing, and for the reason `terrain-canopy`'s
/// `RADIUS` is: only the nine cells around a point are searched, so a stone
/// wider than half a cell could reach a point from outside them and be missed --
/// a hole in a boulder that no test of one cell could find. The assertions at
/// the foot of this file are what keep both under it.
///
/// The rubble's radius is also the finest thing this crate draws, so it is what
/// [`samples_across`] measures a texel against.
pub const BOULDER_RADIUS: f32 = 5.0;
pub const RUBBLE_RADIUS: f32 = 1.2;

/// The shortest and tallest a stone of each class stands, in metres, at full
/// size.
///
/// The two ranges do not meet, which is the whole point of having two: a
/// landscape strewn at one scale with a wide spread reads as a field of medium
/// stones rather than as blocks lying in rubble. [`Scatter::stature`] scales both
/// ranges together, so a gravel bar's cobbles and a valley floor's erratics come
/// off the same two lattices.
pub const BOULDER_SHORTEST: f32 = 2.5;
pub const BOULDER_TALLEST: f32 = 9.0;
pub const RUBBLE_SHORTEST: f32 = 0.4;
pub const RUBBLE_TALLEST: f32 = 1.6;

/// The stone's profile: `0` a straight-sided cone, `1` a dome.
///
/// Nearly a dome, which is the opposite end of the interpolation from a crown.
/// A boulder is a lump of rock that has been rolled, dropped and weathered, and
/// what makes it read as one from the air is a rounded top with a shoulder that
/// falls away quickly -- not the point a cone comes to.
pub const ROUNDNESS: f32 = 0.9;

/// How wide a band of density a stone grows in over, as a share of the whole.
///
/// Borrowed wholesale from `terrain-canopy`'s `EDGE`, and it is here for the
/// second of the two reasons stated there rather than the first. A caller
/// evaluates the scatter once per texel and hands the same value to every sample
/// inside it, so two neighbouring texels can disagree by a little about the
/// density at one lattice cell that straddles them. With a hard threshold that
/// disagreement is a boulder that exists on one side of the texel wall and not
/// the other -- a stone sliced vertically down the middle. With a band it is a
/// stone a centimetre shorter on one side, which nothing can see.
const EDGE: f32 = 0.30;

/// The fixed seeds the two lattices are drawn from.
///
/// Constants rather than parameters because nothing that crosses a pipeline
/// boundary depends on them: the stones go into the stored heights and the max
/// pyramid is reduced from those heights afterwards, so a ceiling bounds
/// whatever this happened to scatter without ever knowing the seed.
///
/// Two of them, because the lattices would otherwise draw the same words. Cell
/// `(3, 4)` of the boulder grid and cell `(3, 4)` of the rubble grid are
/// different ground but the same pair of integers, and one seed would put the
/// same jitter and the same grade in both -- the rubble would be an eight-times
/// scale model of the boulders lying on it.
const BOULDER_SEED: u32 = 0x52_6f_63_6b;
const RUBBLE_SEED: u32 = 0x53_63_72_65;

/// Feature size of the field that decides where the boulders are, in metres.
///
/// About eight boulder spacings. This is the number that makes a boulder *field*
/// a field: a talus fan, a moraine ridge or a rockfall run-out is a couple of
/// hundred metres of ground with stones all over it, next to ground with none.
const FIELD_WAVELENGTH: f32 = 200.0;

/// Where the boulder field starts and where it is full, as positions in the
/// noise the field is cut out of.
///
/// The gate is what separates this from `terrain-canopy`'s `clump`, which
/// multiplies a stand by something between two thirds and twice and never
/// reaches zero. A forest thins and thickens; a boulder field is *there or not
/// there*, and ground between two fields has no stones on it at all rather than
/// a few. Everything under [`FIELD_EDGE`] is bare, and the run up to
/// [`FIELD_FULL`] is short so that a field has an edge rather than a gradient.
const FIELD_EDGE: f32 = 0.42;
const FIELD_FULL: f32 = 0.78;

/// What a full boulder field multiplies its sampled density by.
///
/// Above one, for `terrain-canopy`'s reason: a cell holds a full-sized stone
/// only once the density is [`EDGE`] past what that cell wants, and the wants
/// run up to one, so a completely closed lattice takes a density of `1 + EDGE`.
/// A classifier that writes a half means "half of this hillside is boulder
/// field", not "the boulder field on this hillside is capped at half".
const FIELD_THICKEST: f32 = 2.2;

/// Feature size of the field that thins and thickens the rubble, and what it
/// multiplies a sampled density by at its thinnest and thickest.
///
/// Rubble genuinely is `clump`'s case rather than the gate's: a talus slope is
/// covered everywhere and it is patchier in some places than others. Shorter
/// than the boulder field's wavelength so the two do not draw the same blobs on
/// top of each other.
const STREW_WAVELENGTH: f32 = 60.0;
const STREW_THINNEST: f32 = 0.55;
const STREW_THICKEST: f32 = 1.7;

/// What share of a block, taken from the tallest end, [`baked`] averages.
///
/// The one number here that was set twice. The argument for a *high* share is
/// real -- it is in this module's doc, and it is that the plain mean is the most
/// scale-stable answer there is and that rock can nearly afford it, because a
/// distant boulder field ought to read as the rocky slope it is rather than as
/// the green paint a distant forest read as. Two fifths was tried on that
/// argument and it is wrong, for a reason the canopy already knew:
///
/// **The share has to sit under the coverage.** A boulder field cannot cover
/// much of its ground. Round stones of [`BOULDER_RADIUS`] on a lattice of
/// [`BOULDER_SPACING`] close at about an eighth of the plan view even with one
/// in every cell, because that is what the ratio of those two numbers is, and a
/// field of blocks a couple of stone-widths apart is what a boulder field *is*.
/// Averaging the tallest two fifths of such a block therefore averages one part
/// stone to three parts bare ground and reports a five-metre erratic as sixty
/// centimetres of nothing in particular.
///
/// A twelfth is under the eighth the boulders can reach, so the selected samples
/// are mostly stone at every level. The rubble class, which closes about a third
/// of its ground, is well served by the same number. What it costs is the climb
/// across levels that a lower share always costs, and
/// `a_boulder_field_holds_its_height_as_the_texels_coarsen` is what says the
/// climb is under a pixel at every ring it happens at.
const SILHOUETTE: f32 = 0.08;

/// The boulder share at which a texel is painted as boulder rather than as the
/// ground under it, and the total stone share at which it is painted as rubble.
///
/// Both are plan-view shares, and [`BOULDERED`] is the small one for the reason
/// [`SILHOUETTE`] is: a full boulder field covers about an eighth of its ground
/// and nothing can make it cover more, so a threshold anywhere near
/// `terrain-canopy::PAINTED`'s quarter would paint every boulder field on the
/// landscape as meadow. Against that ceiling a fourteenth is high -- it wants a
/// field most of the way closed -- which is the same place the canopy's quarter
/// sits against its own ceiling of about a third.
///
/// Rubble closes far more of its ground, so [`STREWN`] can be a plain share.
///
/// Boulder is tested first by the caller, so a texel with both gets the coarser
/// answer, which is the one a viewer can actually resolve.
pub const BOULDERED: f32 = 0.07;
pub const STREWN: f32 = 0.30;

/// What is scattered on a patch of ground, once the stored fields have been
/// sampled and the noise applied.
///
/// Two densities and a stature. The densities are the shares of each lattice
/// that hold a stone; the stature scales the height range of both, and is what
/// makes a gravel bar's cobbles and a valley floor's erratics the same two
/// lattices rather than two more sets of constants.
///
/// It is called a stature rather than a size because it is not one: it scales
/// how tall the stones stand and leaves their footprints alone. That is
/// deliberate, and it is what having two lattices already bought. *The classes
/// are the sizes* -- boulders are the big stones and rubble is the small ones,
/// each with a spacing and a reach that go together -- so a third number
/// shrinking a boulder's footprint would only blur the line between them.
/// Within a class the hash still varies both together, so a low stone is a
/// narrow one; see `class`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Scatter {
    pub boulders: f32,
    pub rubble: f32,
    pub stature: f32,
}

impl Scatter {
    /// Ground with nothing lying on it.
    pub const NONE: Scatter = Scatter {
        boulders: 0.0,
        rubble: 0.0,
        stature: 0.0,
    };
}

/// Wellons' `lowbias32`.
///
/// The same mixer `terrain-canopy` and `terrain-generate`'s `noise` use,
/// restated here rather than shared so this crate depends on nothing. Its
/// avalanche is measured rather than assumed, which is what stops neighbouring
/// cells of a lattice from drawing correlated stones and putting a grain of rows
/// across a field.
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
/// collide, and the stones would lie in stripes.
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

/// Hermite ease between two edges, clamped outside them.
fn ramp(edge0: f32, edge1: f32, t: f32) -> f32 {
    fade(((t - edge0) / (edge1 - edge0)).clamp(0.0, 1.0))
}

/// Three independent smooth fields, each `0` to `1`, from one lattice.
///
/// `terrain-canopy`'s, with the wavelength passed in rather than fixed, because
/// there are two fields here at two scales. One lattice and one set of four
/// hashes gives all three, split out of the same words: they are independent
/// because the mixer's avalanche already made every output bit depend on every
/// input bit.
fn fields(x: f32, y: f32, wavelength: f32) -> [f32; 3] {
    let u = x / wavelength;
    let v = y / wavelength;
    let cell_x = u.floor();
    let cell_y = v.floor();
    let fx = fade(u - cell_x);
    let fy = fade(v - cell_y);
    let (cell_x, cell_y) = (cell_x as i32, cell_y as i32);

    let mut out = [0f32; 3];
    let corner = [
        hash(cell_x, cell_y, BOULDER_SEED),
        hash(cell_x + 1, cell_y, BOULDER_SEED),
        hash(cell_x, cell_y + 1, BOULDER_SEED),
        hash(cell_x + 1, cell_y + 1, BOULDER_SEED),
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

/// What to multiply a sampled boulder density by, so that a field is a field.
///
/// Gated rather than scaled, which is the difference between this and
/// `terrain-canopy::clump`. Most ground comes back at zero and has no boulders
/// on it at any density; the ground inside a patch comes back above one and
/// closes the lattice. A classifier writing one number per texel would otherwise
/// draw an even sprinkle of stones over whole mountainsides, which is the one
/// thing a fake boulder field cannot afford to look like.
pub fn field(x: f32, y: f32) -> f32 {
    FIELD_THICKEST * ramp(FIELD_EDGE, FIELD_FULL, fields(x, y, FIELD_WAVELENGTH)[0])
}

/// What to multiply a sampled rubble density by, so a strewn slope thins and
/// thickens inside itself rather than being uniform out to its edge.
///
/// A multiplier and not a gate: rubble covers a talus slope everywhere, and what
/// varies is how thickly. Never reaches zero, so a slope the classifier called
/// strewn has stone on all of it.
pub fn strew(x: f32, y: f32) -> f32 {
    lerp(
        STREW_THINNEST,
        STREW_THICKEST,
        fields(x, y, STREW_WAVELENGTH)[1],
    )
}

/// How high the stones of one class stand over a point, in metres.
///
/// One lattice, nine cells. Nine rather than one because stones overlap: a
/// boulder wide enough to lie against its neighbours reaches out of its own
/// cell, so a point has to ask the three-by-three around it. Those nine hashes,
/// twice over, are the whole cost of this crate and the reason [`baked`] takes
/// all three of its answers from one walk.
#[allow(clippy::too_many_arguments)]
fn class(
    x: f32,
    y: f32,
    density: f32,
    stature: f32,
    spacing: f32,
    radius: f32,
    shortest: f32,
    tallest: f32,
    seed: u32,
) -> f32 {
    if density <= 0.0 || stature <= 0.0 {
        return 0.0;
    }
    let cell_x = (x / spacing).floor() as i32;
    let cell_y = (y / spacing).floor() as i32;
    let mut found = 0.0f32;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = cell_x + dx;
            let cy = cell_y + dy;
            let bits = hash(cx, cy, seed);

            // Four fields out of one word. Splitting a hash beats taking four of
            // them: a stone costs two mixers rather than eight, and the fields
            // are independent because the mixer's avalanche already made every
            // output bit depend on every input bit.
            let jitter_x = (bits & 0x3ff) as f32 * (1.0 / 1024.0);
            let jitter_y = ((bits >> 10) & 0x3ff) as f32 * (1.0 / 1024.0);
            let grade = ((bits >> 20) & 0x3f) as f32 * (1.0 / 64.0);
            let wants = ((bits >> 26) & 0x3f) as f32 * (1.0 / 64.0);

            // How much of a stone this cell holds, if any. See `EDGE`.
            let grow = fade(((density - wants) / EDGE).clamp(0.0, 1.0));
            if grow <= 0.0 {
                continue;
            }

            // Anywhere in its own cell, which is what stops the field drawing as
            // a grid.
            let middle_x = (cx as f32 + jitter_x) * spacing;
            let middle_y = (cy as f32 + jitter_y) * spacing;
            // A low stone is a narrow one. Tying the two together stops the field
            // looking like one boulder scaled up and down, and keeps every radius
            // under `radius`, which the nine-cell search relies on.
            let scale = grow * (0.72 + 0.28 * grade);
            let reach = (radius * scale).max(1.0 / 1024.0);
            let height = stature * lerp(shortest, tallest, grade) * grow;

            let offset_x = x - middle_x;
            let offset_y = y - middle_y;
            let u = (offset_x * offset_x + offset_y * offset_y).sqrt() / reach;
            if u < 1.0 {
                let cone = 1.0 - u;
                let dome = (1.0 - u * u).max(0.0).sqrt();
                found = found.max(height * lerp(cone, dome, ROUNDNESS));
            }
        }
    }
    found
}

/// How high each class of stone stands over one point, in metres above the
/// ground under it.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Stone {
    boulder: f32,
    rubble: f32,
}

impl Stone {
    /// The surface a ray meets: whichever class is higher here.
    fn top(&self) -> f32 {
        self.boulder.max(self.rubble)
    }
}

/// The stones over a point, one walk of each lattice.
///
/// `x` and `y` are metres on the raster's own grid; which corner they are
/// measured from does not matter, so long as one caller measures them the same
/// way twice.
fn nearby(x: f32, y: f32, scatter: &Scatter) -> Stone {
    Stone {
        boulder: class(
            x,
            y,
            scatter.boulders,
            scatter.stature,
            BOULDER_SPACING,
            BOULDER_RADIUS,
            BOULDER_SHORTEST,
            BOULDER_TALLEST,
            BOULDER_SEED,
        ),
        rubble: class(
            x,
            y,
            scatter.rubble,
            scatter.stature,
            RUBBLE_SPACING,
            RUBBLE_RADIUS,
            RUBBLE_SHORTEST,
            RUBBLE_TALLEST,
            RUBBLE_SEED,
        ),
    }
}

/// The stone standing above the ground at one point, in metres.
///
/// The field itself, with no texel in the question. [`baked`] is what a texel
/// carries and it is the only caller outside the tests -- this is here so the
/// tests can ask about the field rather than about a block of it.
#[cfg(test)]
fn stone(x: f32, y: f32, scatter: &Scatter) -> f32 {
    nearby(x, y, scatter).top()
}

/// What one texel of a stored product carries, once the stones on it have been
/// looked at.
///
/// Three answers out of one walk, because all three callers ask about the same
/// block of ground and the eighteen hashes a sample costs are the expensive
/// part. The heights take [`Baked::height`] and the ground cover takes the two
/// shares, so a texel cannot be raised as a boulder and painted as a meadow.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Baked {
    /// How high the stones on this texel stand, in metres above the earth under
    /// them.
    pub height: f32,
    /// How much of the texel lies under a boulder, from zero to one.
    pub boulders: f32,
    /// How much lies under a stone of either class, from zero to one.
    ///
    /// Never below [`Baked::boulders`], because a boulder is a stone.
    pub covered: f32,
}

/// What a texel of this size carries, given what is scattered on it.
///
/// This is the whole of the rock now. `terrain-generate` calls it once per texel
/// of every level of both products, and nothing at run time scatters a stone at
/// all: a ray meets a boulder by meeting the ground, and a pixel is painted as
/// stone because the material under it says so.
///
/// # Why the answer is a conditional mean
///
/// `terrain-canopy::baked` sets this out in full and the shape of the argument
/// is the same, so only the difference is worth restating. A plain block average
/// is scale-invariant -- it is what keeps a stand the same height however coarse
/// the texels holding it get -- while a maximum climbs with the texel, because a
/// wider block has more chances to land on a peak. The mean of the tallest
/// [`SILHOUETTE`] of the block is an average, so it is stable for the same reason
/// the plain mean is, while sitting up among the stones rather than down in the
/// gaps between them.
///
/// Where rock parts company with canopy is how far the lean has to go. A cone
/// averages a third of its peak over its own footprint and a stand covers barely
/// a third of its ground, so the honest average of a forest is a quarter of its
/// own height and drawing it is what made distant woods read as green paint. A
/// dome averages two thirds, and a boulder field that reads as a rocky slope
/// reads as what it is. So the lean here is small and the level-to-level
/// stability that buys is large.
///
/// # Why the sampling is what it is
///
/// A sample that misses the top of a stone misses its height, and by an amount
/// that depends on the texel size -- which is a field that changes height at
/// every clipmap ring, and is exactly the pop the canopy was rebuilt to remove.
/// [`samples_across`] keeps the sample spacing under a quarter of the *smaller*
/// class's radius, so the clipping is about equal at every level and nothing
/// shrinks with distance.
pub fn baked(x: f32, y: f32, scatter: &Scatter, texel_metres: f32) -> Baked {
    let across = samples_across(texel_metres);
    let step = texel_metres / across as f32;
    // Sample centres, so the block is symmetric about the texel and a texel twice
    // the size of its neighbour covers the same ground its four children did.
    let first = 0.5 * step - 0.5 * texel_metres;
    let mut heights = Vec::with_capacity((across * across) as usize);
    let (mut under_boulder, mut under_stone) = (0u32, 0u32);
    for row in 0..across {
        for column in 0..across {
            let here = nearby(
                x + first + column as f32 * step,
                y + first + row as f32 * step,
                scatter,
            );
            let top = here.top();
            heights.push(top);
            // There is no floor to compare against: the ground between the
            // stones is the ground, so anything above it is a stone.
            under_boulder += u32::from(here.boulder > 0.0);
            under_stone += u32::from(top > 0.0);
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
        boulders: under_boulder as f32 / samples as f32,
        covered: under_stone as f32 / samples as f32,
    }
}

/// How many samples across a texel [`baked`] takes.
///
/// A quarter of the *rubble's* radius apart, because the finer class is the one
/// that gets missed, with a floor of four however fine the texel is.
///
/// Capped at thirty-two, as the canopy's is, and here the cap does bind: a
/// sixteen-metre texel wants fifty-four samples across and gets thirty-two, so
/// the rubble's tops are clipped a little more at the coarse levels than at the
/// fine ones. What makes that affordable is that the rubble's whole amplitude is
/// [`RUBBLE_TALLEST`] -- the step it can produce is a fraction of a metre where
/// the canopy's was a fraction of thirty. `a_boulder_field_holds_its_height_as_
/// the_texels_coarsen` is what decides whether it is affordable, measured in
/// pixels at the ring each handover happens at; if it ever stops being, the
/// remedies in order are raising this cap and lowering `RUBBLE_TALLEST`.
fn samples_across(texel_metres: f32) -> u32 {
    let wanted = (texel_metres / (0.25 * RUBBLE_RADIUS)).ceil();
    (wanted as u32).clamp(4, 32)
}

/// A stone reaches at most half a cell, or the nine cells [`class`] searches
/// would not be all the cells that can cover a point -- and a stone missed
/// because its middle sat two cells away is a hole in a boulder that nothing
/// downstream could tell from bare ground.
const _: () = assert!(2.0 * BOULDER_RADIUS <= BOULDER_SPACING);
const _: () = assert!(2.0 * RUBBLE_RADIUS <= RUBBLE_SPACING);

/// The two classes are two sizes rather than one blurred range. Ranges that
/// overlapped would draw a field of medium stones, which is what having two
/// lattices is for avoiding.
const _: () = assert!(RUBBLE_TALLEST < BOULDER_SHORTEST);
const _: () = assert!(RUBBLE_RADIUS < BOULDER_RADIUS);

/// The thickest a boulder field gets holds a stone in every cell of the lattice,
/// which is the claim [`field`] is documented by, and the thin end of the
/// rubble's strewing thins without erasing.
const _: () = assert!(FIELD_THICKEST >= 1.0 + EDGE);
const _: () = assert!(STREW_THINNEST > 0.0 && STREW_THINNEST < 1.0);

#[cfg(test)]
mod tests {
    use super::*;

    /// A talus slope in a boulder field: rubble over most of it, blocks lying in
    /// the rubble. What most of the strewn ground on this landscape is.
    const TALUS: Scatter = Scatter {
        boulders: 0.55,
        rubble: 0.85,
        stature: 1.0,
    };

    /// Boulders alone, at the density a full field reaches.
    const FIELD: Scatter = Scatter {
        boulders: 0.9,
        rubble: 0.0,
        stature: 1.0,
    };

    /// Walks a wide patch of ground, finely enough to catch a piece of rubble
    /// between two samples.
    fn over_a_slope(mut visit: impl FnMut(f32, f32)) {
        // Strides that do not divide either spacing, so the samples do not walk
        // a lattice in step with it and measure one stone over and over.
        for i in 0..900 {
            for j in 0..900 {
                visit(i as f32 * 0.37, j as f32 * 0.41);
            }
        }
    }

    /// Integrates the stone field over a block of ground.
    ///
    /// Returns the average height, the share of the ground under a boulder and
    /// the share under a stone of either class. This is the independent
    /// statement of what [`baked`] answers by sampling a texel, and the strides
    /// are chosen not to divide either spacing so the walk drifts across both
    /// lattices instead of measuring the same place in every cell.
    fn integrate(
        scatter: &Scatter,
        steps: u32,
        stride: (f32, f32),
        from: (f32, f32),
    ) -> (f32, f32, f32) {
        let mut height = 0f64;
        let (mut boulders, mut covered) = (0u64, 0u64);
        for i in 0..steps {
            for j in 0..steps {
                let x = from.0 + i as f32 * stride.0;
                let y = from.1 + j as f32 * stride.1;
                let found = nearby(x, y, scatter);
                height += f64::from(found.top());
                boulders += u64::from(found.boulder > 0.0);
                covered += u64::from(found.top() > 0.0);
            }
        }
        let samples = f64::from(steps) * f64::from(steps);
        (
            (height / samples) as f32,
            (boulders as f64 / samples) as f32,
            (covered as f64 / samples) as f32,
        )
    }

    /// The average of [`baked`] over a wide patch, at one texel size.
    fn average(scatter: &Scatter, texel: f32, pick: impl Fn(&Baked) -> f32) -> f32 {
        let (mut sum, mut count) = (0.0f64, 0u32);
        for i in 0..90 {
            for j in 0..90 {
                // Strides that do not divide the texel, so the blocks land all
                // over the lattices rather than in step with them.
                let got = baked(i as f32 * 37.0, j as f32 * 41.0, scatter, texel);
                sum += f64::from(pick(&got));
                count += 1;
            }
        }
        (sum / f64::from(count)) as f32
    }

    /// What [`baked`] reports a coarse texel's stone shares to be are the shares
    /// the field actually has.
    ///
    /// These are the numbers that decide whether a texel of the ground-cover
    /// product says `Boulder` or `Rubble`, so getting them wrong paints
    /// hillsides the wrong colour and nothing reports it. Checked against an
    /// independent integral over the same ground rather than against a
    /// remembered table.
    #[test]
    fn a_coarse_texel_reports_the_shares_the_field_has() {
        for scatter in [
            TALUS,
            FIELD,
            Scatter {
                boulders: 0.2,
                rubble: 0.4,
                stature: 0.7,
            },
            Scatter {
                boulders: 0.0,
                rubble: 1.2,
                stature: 1.0,
            },
        ] {
            let (_, boulders, covered) = integrate(&scatter, 900, (0.31, 0.29), (0.0, 0.0));
            let baked_boulders = average(&scatter, 64.0, |b| b.boulders);
            let baked_covered = average(&scatter, 64.0, |b| b.covered);
            assert!(
                (baked_boulders - boulders).abs() < 0.03,
                "{scatter:?} bakes a boulder share of {baked_boulders:.3} where the \
                 field measures {boulders:.3}",
            );
            assert!(
                (baked_covered - covered).abs() < 0.03,
                "{scatter:?} bakes a stone share of {baked_covered:.3} where the \
                 field measures {covered:.3}",
            );
        }
    }

    /// A boulder field does not visibly change height as the texels holding it
    /// coarsen.
    ///
    /// The pop-in invariant, stated where it can be measured, and the test that
    /// decides both [`SILHOUETTE`] and the cap in [`samples_across`]. The levels
    /// are concentric rings around the camera, so ground crosses from one texel
    /// size to the next as the aircraft moves; if a field's height moved with it,
    /// a slope would rise or fall at that ring.
    ///
    /// **Visibly** is the whole of the test, exactly as it is for the canopy. The
    /// height does climb with the texel -- an order statistic of a one-metre
    /// block is nearly that block's own value, while over sixteen metres it is a
    /// real statistic of the field, and nothing but a plain average escapes that.
    /// What matters is each step measured against the distance its own ring sits
    /// at, because a ring twice as far away subtends half the angle.
    #[test]
    fn a_boulder_field_holds_its_height_as_the_texels_coarsen() {
        // Both restated rather than imported: this crate depends on nothing, and
        // these belong to the renderer. A level reaches `Residency::reach_texels`
        // of its own texels, so the ring where texels of `t` metres hand over to
        // `2 * t` sits at `REACH * t` metres; `PIXEL` is `Residency::pixel_angle`
        // at 1080p and sixty degrees.
        const REACH: f32 = 1536.0;
        const PIXEL: f32 = 1.069e-3;

        let sizes = [1.0f32, 2.0, 4.0, 8.0, 16.0];
        for scatter in [TALUS, FIELD] {
            let heights: Vec<f32> = sizes
                .iter()
                .map(|t| average(&scatter, *t, |b| b.height))
                .collect();
            for (index, pair) in heights.windows(2).enumerate() {
                let texel = sizes[index];
                let step = (pair[1] - pair[0]).abs();
                // How many metres one pixel covers where these two levels meet.
                let pixel = REACH * texel * PIXEL;
                assert!(
                    step < pixel,
                    "handing {texel} m texels over to {} m moves {scatter:?} by \
                     {step:.2} m at {:.0} m, where a pixel is {pixel:.2} m -- \
                     {:.1} pixels of step. Heights were {heights:?}",
                    texel * 2.0,
                    REACH * texel,
                    step / pixel,
                );
            }
            // And it climbs rather than wandering, so no ring can hand a field
            // back down again.
            assert!(
                heights.windows(2).all(|pair| pair[1] >= pair[0]),
                "{scatter:?} is not monotone in the texel size: {heights:?}",
            );
        }
    }

    /// A texel many stones wide stands nearer their tops than the gaps between
    /// them, which is what [`SILHOUETTE`] exists for -- and it never stands above
    /// the tallest stone there is, which is what the max pyramid rests on.
    #[test]
    fn a_coarse_texel_stands_among_the_stones() {
        let got = average(&FIELD, 32.0, |b| b.height);
        let flat = integrate(&FIELD, 700, (0.53, 0.47), (0.0, 0.0)).0;
        assert!(
            got > 1.3 * flat,
            "a 32 m texel of boulder field bakes {got:.2} m against the {flat:.2} m \
             the same ground averages -- the lean is not doing its job",
        );
        assert!(
            got <= BOULDER_TALLEST,
            "a 32 m texel bakes {got:.2} m, over the {BOULDER_TALLEST} m tallest stone",
        );
    }

    /// The invariant the max pyramid rests on, asked of the baked heights. The
    /// pyramid is reduced from these heights, so a ceiling bounds them for free
    /// -- but it is what a ray tests against before it looks at any height, so
    /// this is worth stating rather than deriving.
    #[test]
    fn nothing_baked_stands_above_the_tallest_stone_there_is() {
        for stature in [0.0f32, 0.3, 0.7, 1.0] {
            for density in [0.0f32, 0.3, 0.9, 1.0, 1.5, 2.2] {
                let scatter = Scatter {
                    boulders: density,
                    rubble: density,
                    stature,
                };
                let bound = BOULDER_TALLEST * stature;
                for texel in [1.0f32, 2.0, 8.0, 32.0, 128.0] {
                    for step in 0..1500 {
                        let (x, y) = (step as f32 * 0.37, step as f32 * -0.41 + 13.0);
                        let got = baked(x, y, &scatter, texel).height;
                        assert!(
                            got <= bound,
                            "a {texel} m texel of {density} at a stature of {stature} baked {got} m \
                             through a ceiling of {bound} m",
                        );
                    }
                }
            }
        }
    }

    /// Bare ground carries nothing, at any texel size, and each of the three
    /// numbers is a veto on its own. A clearing baked as a plateau would be a
    /// step in the middle of a meadow.
    #[test]
    fn nothing_lies_where_nothing_is_scattered() {
        for scatter in [
            Scatter::NONE,
            Scatter {
                boulders: 1.0,
                rubble: 1.0,
                stature: 0.0,
            },
            Scatter {
                boulders: 0.0,
                rubble: 0.0,
                stature: 1.0,
            },
        ] {
            for texel in [1.0f32, 4.0, 64.0] {
                let got = baked(123.0, -456.0, &scatter, texel);
                assert_eq!(got.height, 0.0, "{scatter:?} at {texel} m");
                assert_eq!(got.boulders, 0.0, "{scatter:?} at {texel} m");
                assert_eq!(got.covered, 0.0, "{scatter:?} at {texel} m");
            }
            over_a_slope(|x, y| {
                assert_eq!(stone(x, y, &scatter), 0.0, "{scatter:?} grew something");
            });
        }
    }

    /// More density is more of the ground under a stone, monotonically, up to
    /// the point every cell of the lattice holds one. A share that dipped
    /// anywhere would paint a thicker field as bare ground while the thinner one
    /// beside it drew as stone.
    #[test]
    fn the_shares_never_fall_as_the_densities_rise() {
        let (mut boulders, mut covered) = (0.0f32, 0.0f32);
        for step in 0..=40 {
            let density = step as f32 * 0.055;
            let scatter = Scatter {
                boulders: density,
                rubble: density,
                stature: 1.0,
            };
            let (mut boulder_sum, mut covered_sum, mut count) = (0.0f64, 0.0f64, 0u32);
            for i in 0..20 {
                for j in 0..20 {
                    let got = baked(i as f32 * 64.0, j as f32 * 64.0, &scatter, 64.0);
                    boulder_sum += f64::from(got.boulders);
                    covered_sum += f64::from(got.covered);
                    count += 1;
                }
            }
            let now_boulders = (boulder_sum / f64::from(count)) as f32;
            let now_covered = (covered_sum / f64::from(count)) as f32;
            assert!(
                now_boulders >= boulders - 0.005,
                "at a density of {density} the boulder share fell to {now_boulders} \
                 from {boulders}",
            );
            assert!(
                now_covered >= covered - 0.005,
                "at a density of {density} the stone share fell to {now_covered} \
                 from {covered}",
            );
            assert!(
                now_covered >= now_boulders - 1e-6,
                "at a density of {density} more ground is under a boulder \
                 ({now_boulders}) than under a stone ({now_covered})",
            );
            boulders = boulders.max(now_boulders);
            covered = covered.max(now_covered);
        }
    }

    /// The paint thresholds have to be crossable by ground anyone would call
    /// strewn and uncrossable by ground nobody would. A threshold nothing
    /// crosses would paint every talus slope as meadow; one everything crosses
    /// would paint every meadow as talus.
    #[test]
    fn the_paint_thresholds_fall_inside_the_ranges_the_shares_cover() {
        let scattered = |boulders, rubble| {
            let scatter = Scatter {
                boulders,
                rubble,
                stature: 1.0,
            };
            let (_, under_boulder, under_stone) =
                integrate(&scatter, 700, (0.31, 0.29), (0.0, 0.0));
            (under_boulder, under_stone)
        };

        let (sparse, _) = scattered(0.15, 0.0);
        assert!(
            sparse < BOULDERED,
            "a scattering of 0.15 already covers {sparse:.3}, past the {BOULDERED} \
             boulder threshold",
        );
        let (full, _) = scattered(FIELD_THICKEST * 0.7, 0.0);
        assert!(
            full > BOULDERED,
            "a full boulder field only covers {full:.3}, under the {BOULDERED} threshold",
        );

        let (_, thin) = scattered(0.0, 0.2);
        assert!(
            thin < STREWN,
            "rubble at 0.2 already covers {thin:.3}, past the {STREWN} threshold",
        );
        let (_, thick) = scattered(0.0, STREW_THICKEST * 0.8);
        assert!(
            thick > STREWN,
            "a thickly strewn slope only covers {thick:.3}, under the {STREWN} threshold",
        );
    }

    /// The field must not draw as a grid, which is what it would do with a stone
    /// at the middle of every cell.
    ///
    /// Measured as the stone over each lattice's own lines against the stone over
    /// its cell middles. A planted field is tall in the middles and short on the
    /// lines; a scattered one cannot tell them apart, because a stone is as
    /// likely to sit on a line as anywhere else.
    #[test]
    fn the_stones_do_not_line_up_with_the_lattices() {
        for (spacing, scatter, tallest) in [
            (BOULDER_SPACING, FIELD, BOULDER_TALLEST),
            (
                RUBBLE_SPACING,
                Scatter {
                    boulders: 0.0,
                    rubble: 0.9,
                    stature: 1.0,
                },
                RUBBLE_TALLEST,
            ),
        ] {
            let (mut middles, mut lines) = (0.0f64, 0.0f64);
            let mut count = 0u32;
            for i in 0..200 {
                for j in 0..200 {
                    let (cx, cy) = (i as f32 * spacing, j as f32 * spacing);
                    middles += f64::from(stone(cx + spacing * 0.5, cy + spacing * 0.5, &scatter));
                    lines += f64::from(stone(cx, cy, &scatter));
                    count += 1;
                }
            }
            let (middles, lines) = (middles / f64::from(count), lines / f64::from(count));
            assert!(
                (middles - lines).abs() < 0.1 * f64::from(tallest),
                "a field on a {spacing} m lattice averages {middles:.2} m over its cell \
                 middles and {lines:.2} m over its lattice lines, which is a grid"
            );
        }
    }

    /// The lattices run through the origin and on into negative ground, and a
    /// hash of a negative cell must be as good as one of a positive cell. A seam
    /// on the axes would draw as a straight line of identical stones running the
    /// width of the raster.
    #[test]
    fn the_field_crosses_the_origin_without_a_seam() {
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in -600..600 {
            for j in -600..600 {
                let got = stone(i as f32 * 0.37, j as f32 * 0.41, &TALUS);
                low = low.min(got);
                high = high.max(got);
            }
        }
        assert_eq!(low, 0.0, "the ground around the origin was never bare");
        assert!(
            high > 0.7 * BOULDER_TALLEST,
            "the field around the origin only reached {high} m",
        );
    }

    /// Tiles and levels are generated independently, so the only thing that makes
    /// them join is that the stone at a position does not depend on who asked.
    #[test]
    fn the_stone_at_a_position_does_not_depend_on_who_asks() {
        for x in [0.0f32, -7.0, 511.5, 512.5, 8191.5, 49151.5] {
            assert_eq!(
                stone(x, 1024.5, &TALUS),
                stone(x, 1024.5, &TALUS),
                "at x = {x}"
            );
        }
    }

    /// What the gate is *for*. A boulder field has to be a field: ground with
    /// stones all over it next to ground with none, rather than an even sprinkle
    /// across the whole mountainside.
    #[test]
    fn a_boulder_field_is_patchy_rather_than_even() {
        let (mut bare, mut full, mut total) = (0u32, 0u32, 0u32);
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..600 {
            for j in 0..600 {
                let got = field(i as f32 * 3.7, j as f32 * 4.1);
                low = low.min(got);
                high = high.max(got);
                total += 1;
                bare += u32::from(got <= 0.0);
                full += u32::from(got >= 1.0 + EDGE);
            }
        }
        assert_eq!(low, 0.0, "the field never came down to bare ground");
        assert!(
            high >= 1.0 + EDGE,
            "the field only reached {high}, so no lattice cell ever closes",
        );
        let bare = f64::from(bare) / f64::from(total);
        let full = f64::from(full) / f64::from(total);
        assert!(
            bare > 0.2,
            "only {bare:.3} of the ground is outside a boulder field, which is a \
             sprinkle rather than a field",
        );
        assert!(
            full > 0.05,
            "only {full:.3} of the ground is inside a full field",
        );
    }

    /// The strewing thins and thickens without ever erasing: a slope the
    /// classifier called strewn has stone on all of it.
    #[test]
    fn the_strewing_thickens_as_well_as_thins() {
        let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
        for i in 0..2000 {
            for j in 0..200 {
                let got = strew(i as f32 * 0.53 - 500.0, j as f32 * 1.7 - 170.0);
                low = low.min(got);
                high = high.max(got);
            }
        }
        assert!(low > 0.0, "the strewing fell to {low}");
        assert!(high > 1.0, "the strewing never reached one: {high}");
    }

    /// Both noise fields have to be smooth, or the patchiness they shape a slope
    /// with would break into the lattice it is drawn on -- which is the artefact
    /// they exist to hide.
    #[test]
    fn the_noise_fields_do_not_jump_between_neighbouring_points() {
        for i in 0..4000 {
            let (x, y) = (i as f32 * 0.31 - 600.0, i as f32 * -0.17 + 90.0);
            let step = 0.05;
            assert!(
                (field(x, y) - field(x + step, y)).abs() < 0.05,
                "the boulder field jumped at ({x}, {y})"
            );
            assert!(
                (strew(x, y) - strew(x + step, y)).abs() < 0.05,
                "the strewing jumped at ({x}, {y})"
            );
        }
    }

    /// The two classes are two sizes on the ground as well as in the constants:
    /// a slope of rubble and a field of boulders must not bake to the same
    /// height, or there was no reason to walk two lattices.
    ///
    /// Measured at the tall end rather than on the average, and the difference
    /// between those two is the point of the boulder lattice. Boulders are
    /// twenty-four metres apart, so most eight-metre texels of a full field hold
    /// none at all and bake to nothing while the few that hold one stand right
    /// up; rubble is three metres apart, so every texel holds some and they all
    /// bake to much the same. Averaging over positions therefore compares how
    /// *often* against how *tall* and finds the two classes equal, which is true
    /// and is not what this is asking.
    #[test]
    fn the_two_classes_are_two_different_sizes_of_stone() {
        let tallest = |scatter: &Scatter| {
            let mut high = f32::NEG_INFINITY;
            for i in 0..90 {
                for j in 0..90 {
                    high = high.max(baked(i as f32 * 37.0, j as f32 * 41.0, scatter, 8.0).height);
                }
            }
            high
        };
        let rubble = tallest(&Scatter {
            boulders: 0.0,
            rubble: 0.9,
            stature: 1.0,
        });
        let boulders = tallest(&FIELD);
        assert!(
            boulders > 2.0 * rubble,
            "the tallest texel of a boulder field bakes {boulders:.2} m against \
             rubble's {rubble:.2} m, which is one size of stone rather than two",
        );
    }
}
