// Filling the hollows, in parallel, by relaxation instead of by a heap.
//
// The CPU pass this replaces walks a priority queue: take the lowest cell
// reached so far, step to its neighbours, raise each to the level it was
// reached at. That is inherently serial -- the next cell to take depends on
// every cell taken so far -- and it is what costs `flow::drainage` its two
// seconds a round on one core.
//
// The same surface has a fixed-point description that needs no order at all
// (Planchon and Darboux, 2001):
//
//     w[i] = max(h[i], min over the eight neighbours of w[n])
//
// with the cells on the edge of the map pinned at their own height, because the
// edge is where water leaves. Started from a surface *above* the answer and
// relaxed downwards, this converges to exactly the flood's filled surface: a
// cell in a hollow settles at the lowest level it can still reach the edge
// through, which is its spill level, and a cell on open ground settles at its
// own height.
//
// The direction matters and is not symmetric. From above, every step is a
// decrease and the limit is the fill. From *below* the iteration converges to
// something else entirely -- a basin would rise one lattice step per sweep and
// stop wherever it was when the sweeps ran out -- so every seed here has to be
// an upper bound on the answer, and `cs_seed_warm` is written to guarantee one.
//
// Information moves one cell per iteration, so a cold start costs about as many
// iterations as the grid is wide. That is affordable once. What makes the
// eighty rounds of incision affordable is that filling is monotone in the
// ground: lower the terrain and the fill can only fall, so the previous round's
// answer is already an upper bound on this one's.

struct Params {
    width: u32,
    rows: u32,
    // How much any cell may have risen since the surface in `previous` was
    // computed. Added to the warm seed so that it stays an upper bound even
    // though creep can lift ground as well as move it sideways.
    lift: f32,
    // How far along each of the eight directions one iteration looks.
    //
    // One is the plain stencil, and it is why this pass was slow: a cell can
    // only learn about ground it is next to, so a surface settles at one cell an
    // iteration and the grid is three thousand cells across. See `cs_fill`.
    reach: u32,
}

// Higher than any ground and finite, so `min` and `max` stay ordinary
// arithmetic. An actual infinity would work too and would make a stray NaN out
// of `inf - inf` the moment anyone subtracted two of these.
const ABOVE_EVERYTHING: f32 = 3.0e30;
const BELOW_EVERYTHING: f32 = -3.0e30;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> height: array<f32>;
@group(0) @binding(2) var<storage, read> previous: array<f32>;
@group(0) @binding(3) var<storage, read_write> next: array<f32>;
@group(0) @binding(4) var<storage, read_write> changed: atomic<u32>;

fn is_edge(x: u32, y: u32) -> bool {
    return x == 0u || y == 0u || x + 1u == params.width || y + 1u == params.rows;
}

// A surface far above the landscape, which is the only safe cold start.
@compute @workgroup_size(8, 8)
fn cs_seed_cold(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.rows {
        return;
    }
    let index = id.y * params.width + id.x;
    if is_edge(id.x, id.y) {
        next[index] = height[index];
    } else {
        next[index] = ABOVE_EVERYTHING;
    }
}

// The previous round's answer, lifted by however much ground rose since.
//
// Filling is monotone: if the ground only ever fell, last round's filled
// surface is already above this round's and can be relaxed straight down from.
// Ground does not only ever fall -- creep moves material and can raise a cell --
// so the seed carries `lift`, the largest rise anywhere on the grid this round.
// Raising every cell by the worst case keeps the bound honest for a cost of a
// few extra iterations, where trusting the previous surface unadjusted would
// quietly relax from below a basin's true level and leave it unfilled.
@compute @workgroup_size(8, 8)
fn cs_seed_warm(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.rows {
        return;
    }
    let index = id.y * params.width + id.x;
    if is_edge(id.x, id.y) {
        next[index] = height[index];
    } else {
        next[index] = max(previous[index] + params.lift, height[index]);
    }
}

// How the tiled iteration below divides the grid.
//
// A workgroup loads a patch, relaxes it in workgroup memory for as many
// iterations as its halo is deep, and writes back only the middle. The halo is
// what makes that legal: after `k` iterations only cells further than `k` from
// the patch edge can still be trusted, because everything nearer has been
// reading a margin that was never itself updated. Writing back only the middle
// throws exactly the untrustworthy part away.
//
// The trade is redundancy against distance. A patch of 48 that keeps 32 does
// 2.25 cells of work for every cell it settles, and buys 8 iterations for one
// trip through memory instead of eight. Bigger halos buy more distance and cost
// more overlap; this pair fits in workgroup memory with room for the heights
// beside it, which is what decides it.
const TILE: i32 = 48;
const HALO: i32 = 8;
const PATCH: i32 = TILE + 2 * HALO;
const PATCH_SHIFT: u32 = 6u;
const PATCH_CELLS: i32 = PATCH * PATCH;
const THREADS: i32 = 256;

// Two surfaces to ping-pong between, and the ground under them. Single-buffered
// would halve this and make every iteration a race between invocations reading
// a cell and the one writing it -- which for a monotone descent would probably
// converge anyway, and would be undefined behaviour that happened to work.
var<workgroup> tile: array<f32, 8192>;
var<workgroup> ground: array<f32, 4096>;

// The surface at a global cell, as the patch loader sees it.
//
// Outside the grid reads as a wall rather than as a hole: the only cells that
// can see past the edge are the grid's own edge cells, which are pinned to their
// own height and do not care what is beyond them, and a wall cannot make a fill
// come out too low if one ever did.
fn surface_at(x: i32, y: i32) -> f32 {
    if x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.rows) {
        return ABOVE_EVERYTHING;
    }
    return previous[u32(y * i32(params.width) + x)];
}

fn ground_at(x: i32, y: i32) -> f32 {
    if x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.rows) {
        return ABOVE_EVERYTHING;
    }
    return height[u32(y * i32(params.width) + x)];
}

@compute @workgroup_size(16, 16)
fn cs_fill_tiled(
    @builtin(workgroup_id)group: vec3<u32>,
    @builtin(local_invocation_index) local: u32,
) {
    let origin = vec2<i32>(
        i32(group.x) * TILE - HALO,
        i32(group.y) * TILE - HALO,
    );

    for (var slot = i32(local); slot < PATCH_CELLS; slot += THREADS) {
        let x = origin.x + (slot & (PATCH - 1));
        let y = origin.y + (slot >> PATCH_SHIFT);
        tile[slot] = surface_at(x, y);
        ground[slot] = ground_at(x, y);
    }
    workgroupBarrier();

    var live = 0;
    for (var iteration = 0; iteration < HALO; iteration += 1) {
        let read = live * PATCH_CELLS;
        let write = (1 - live) * PATCH_CELLS;
        for (var slot = i32(local); slot < PATCH_CELLS; slot += THREADS) {
            let px = slot & (PATCH - 1);
            let py = slot >> PATCH_SHIFT;
            let x = origin.x + px;
            let y = origin.y + py;

            // The rim of the patch has no neighbours of its own to read, and
            // the edge of the grid is pinned. Both keep what they hold.
            let rim = px == 0 || py == 0 || px == PATCH - 1 || py == PATCH - 1;
            if rim || x <= 0 || y <= 0 || x + 1 >= i32(params.width) || y + 1 >= i32(params.rows) {
                tile[write + slot] = tile[read + slot];
                continue;
            }

            var lowest = ABOVE_EVERYTHING;
            for (var dy = -1; dy <= 1; dy += 1) {
                for (var dx = -1; dx <= 1; dx += 1) {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    lowest = min(lowest, tile[read + slot + dy * PATCH + dx]);
                }
            }
            tile[write + slot] = max(ground[slot], lowest);
        }
        workgroupBarrier();
        live = 1 - live;
    }

    // Only the middle is trustworthy, and only the middle is written.
    let settled = live * PATCH_CELLS;
    for (var slot = i32(local); slot < PATCH_CELLS; slot += THREADS) {
        let px = slot & (PATCH - 1);
        let py = slot >> PATCH_SHIFT;
        if px < HALO || py < HALO || px >= HALO + TILE || py >= HALO + TILE {
            continue;
        }
        let x = origin.x + px;
        let y = origin.y + py;
        if x >= i32(params.width) || y >= i32(params.rows) {
            continue;
        }
        let index = u32(y * i32(params.width) + x);
        let value = tile[settled + slot];
        next[index] = value;
        if value != previous[index] {
            atomicOr(&changed, 1u);
        }
    }
}

@compute @workgroup_size(8, 8)
fn cs_fill(@builtin(global_invocation_id)id: vec3<u32>) {
    if id.x >= params.width || id.y >= params.rows {
        return;
    }
    let index = id.y * params.width + id.x;
    let ground = height[index];

    // Pinned, every iteration. The edge is the boundary condition the whole
    // relaxation is measured against, and a landscape with no outlet would
    // otherwise fill to its own rim and drown.
    if is_edge(id.x, id.y) {
        next[index] = ground;
        return;
    }

    let x = i32(id.x);
    let y = i32(id.y);
    let width = i32(params.width);
    let rows = i32(params.rows);
    var lowest = ABOVE_EVERYTHING;

    // Eight directions, matching the flood it replaces. Four-connected would
    // fill differently: a saddle that spills diagonally is a way out to one
    // rule and a wall to the other.
    //
    // Each direction is followed for `reach` cells rather than one, which is
    // what makes this affordable. Expanding the fixed point twice shows why it
    // is allowed:
    //
    //     w[i] = max(h[i], min over j of max(h[j], min over k of w[k]))
    //
    // -- so a two-step path through `j` to `k` offers `max(h[j], w[k])`, and a
    // longer one offers the highest ground it crossed on the way against the
    // surface at its far end. Every straight run below is such a path, so every
    // value it offers is one the cell could really reach the edge through: the
    // relaxation still only ever descends towards the same fixed point, it just
    // learns about ground `reach` cells away in one iteration instead of
    // crawling there one cell at a time.
    for (var dy = -1; dy <= 1; dy += 1) {
        for (var dx = -1; dx <= 1; dx += 1) {
            if dx == 0 && dy == 0 {
                continue;
            }
            // The highest ground crossed so far, which starts below everything
            // because the first step crosses nothing: a neighbour offers its own
            // surface and no barrier, exactly as the one-cell stencil did.
            var barrier = BELOW_EVERYTHING;
            for (var step = 1; step <= i32(params.reach); step += 1) {
                let nx = x + dx * step;
                let ny = y + dy * step;
                if nx < 0 || ny < 0 || nx >= width || ny >= rows {
                    break;
                }
                let neighbour = u32(ny * width + nx);
                lowest = min(lowest, max(barrier, previous[neighbour]));
                // Past this cell, it is ground the path has to cross rather than
                // ground the path ends on.
                barrier = max(barrier, height[neighbour]);
                // Nothing further along this ray can offer less than the wall
                // already crossed, so a ray that has climbed above the best
                // answer so far has nothing left to say.
                if barrier >= lowest {
                    break;
                }
            }
        }
    }

    let settled = max(ground, lowest);
    next[index] = settled;
    // One flag for the whole grid rather than a count: the loop only asks
    // whether anything is still moving, and a count would need an atomic add
    // per cell to answer a question nobody has.
    if settled != previous[index] {
        atomicOr(&changed, 1u);
    }
}
