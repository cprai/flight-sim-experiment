// The wind, solved once around the mountains and then left alone.
//
// A coarse box over the whole raster, uniform in x and z and in level layers up
// to `TOP_METRES`, holding a velocity per cell. The ground is an obstacle in it:
// a cell whose centre is under the terrain is solid, the flow may not pass
// through its faces, and what comes out is air that piles up on a windward
// slope, accelerates over a ridge and sinks into the lee behind it.
//
// Every kernel here runs at load and none of them runs again. That is what lets
// the solve be as long as it needs to be -- three hundred steps of half a minute
// each, two and a half hours of weather -- and what makes the result a texture
// the frame reads rather than a simulation the frame carries.
//
// The method is Stam's stable fluids: advect the velocity backwards along
// itself, then project out whatever divergence that left. Semi-Lagrangian
// advection is unconditionally stable, so the step is bounded by how far the
// answer should move in one and not by the cell size -- see Stam, "Stable
// Fluids", SIGGRAPH 1999 section 3.2. The projection is a Poisson solve, run
// here as red-black Gauss-Seidel over a storage buffer: it converges about
// twice as fast per sweep as Jacobi and needs no second buffer to ping-pong
// against, because a cell of one colour only ever reads cells of the other.
//
// The grid is collocated -- one velocity at each cell centre rather than one per
// face -- which a staggered grid would beat on paper. It is not worth it here.
// A staggered grid's advantage is that it cannot support the checkerboard
// pressure mode, and a checkerboard at this cell size is a five-hundred-metre
// wobble in a field that is read through a trilinear fetch and then used to
// displace cloud by kilometres. What it would cost is three separate face
// arrays, three sets of boundary conditions, and an interpolation at every
// sample site.

// The grid. Must match `CELLS`, `TOP_METRES` and the rest in `src/air.rs`;
// there is no preprocessor here, so these are a second copy and there is a test
// comparing the two as text.
const CELLS_X: u32 = 160u;
const CELLS_Y: u32 = 20u;
const CELLS_Z: u32 = 160u;

// How high the solved air reaches, in metres.
//
// Seven kilometres, which covers the low and middle cloud decks. Nothing above
// this is solved: cirrus sits in air the mountains do not reach and takes the
// bulk drift alone, so simulating it would be a third of the grid spent on a
// deflection that is not there.
const TOP_METRES: f32 = 7000.0;

// Metres per layer: 350, against 300-720 across. Deliberately near-cubic.
//
// The tempting thing is to distribute the layers the way the aerial-perspective
// volume distributes its slices, thick at the top and thin at the bottom, so
// that the air near the ground is resolved best. That is right for a table
// addressed along a view ray and wrong for a Poisson solve: a square root over
// this range would put the first layer nine metres thick against six hundred
// across, an aspect ratio near seventy to one, and a Poisson problem on cells
// that anisotropic is badly conditioned -- Gauss-Seidel crawls and the no-flux
// boundary throws off spurious vertical gradients.
const CELL_Y: f32 = TOP_METRES / f32(CELLS_Y);

// How many steps a parcel is followed back along its own streamline in.
//
// A loop bound, so it is a constant rather than a uniform member like the knobs
// below; how *long* it follows for is `knobs.w`.
const DRIFT_STEPS: u32 = 20u;

// What the bake was asked for.
//
// Everything here that is a tunable number rather than a shape lives in the
// uniform rather than in a second `const` beside its twin in `src/air.rs`. Two
// copies of a number is two things to keep in step, and a knob that is only
// ever read here has no reason to be one of them -- only the grid's shape has
// to be known on both sides, because that is what decides how big a texture is
// and how many workgroups cover it.
struct Air {
    // World x and z of the grid's low corner, then the metres one cell spans
    // across each of those axes. The vertical is not here: it is `CELL_Y`,
    // which is fixed, where the horizontal follows whatever raster was loaded.
    bounds: vec4<f32>,
    // The wind aloft, as a velocity in metres per second, and in `w` the step
    // this dispatch covers in seconds.
    aloft: vec4<f32>,
    // The roughness length of the ground and the height the wind has reached
    // its free-stream speed by, both in metres; then how long the flow takes to
    // be pulled back towards that wind, and how far back a parcel is followed,
    // both in seconds.
    knobs: vec4<f32>,
};

@group(1) @binding(0) var<uniform> air: Air;

// The velocity being read. Never the one being written: a texture cannot be
// bound writable and sampled in the same pass, so the bake keeps two and hands
// them to the kernels in the two orders it needs. See `src/sky.rs`, which found
// the same wall from the other side.
@group(2) @binding(0) var velocity: texture_3d<f32>;
@group(2) @binding(1) var air_sampler: sampler;

// The terrain, one height per column of the grid, filled from the coarse mirror
// the terrain already keeps on the CPU.
//
// A storage buffer rather than a texture because it is read at arbitrary
// positions and `R32Float` is not filterable without a device feature -- the
// trap `LUT_FORMAT` in `src/sky.rs` documents. Bilinear by hand off a buffer
// costs four loads and no feature, and the field is a hundred kilobytes.
@group(3) @binding(0) var<storage, read> ground: array<f32>;
// Read and written by the same dispatch, which a buffer allows and a texture
// does not. That is the whole reason the Poisson solve lives in buffers.
@group(3) @binding(1) var<storage, read_write> pressure: array<f32>;
@group(3) @binding(2) var<storage, read_write> divergence: array<f32>;
@group(3) @binding(3) var out_velocity: texture_storage_3d<rgba16float, write>;
@group(3) @binding(4) var out_drift: texture_storage_3d<rgba16float, write>;

// Where a cell sits in the buffers: x fastest, then the layer, then z.
fn cell_index(at: vec3<u32>) -> u32 {
    return at.x + CELLS_X * (at.y + CELLS_Y * at.z);
}

// Where a column sits in the ground field.
fn column_index(x: u32, z: u32) -> u32 {
    return x + CELLS_X * z;
}

// The world position a cell's centre stands at.
fn cell_centre(at: vec3<u32>) -> vec3<f32> {
    return vec3<f32>(
        air.bounds.x + (f32(at.x) + 0.5) * air.bounds.z,
        (f32(at.y) + 0.5) * CELL_Y,
        air.bounds.y + (f32(at.z) + 0.5) * air.bounds.w,
    );
}

// Where a world position sits in the grid, as a fraction of it in each axis.
fn grid_uvw(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        (p.x - air.bounds.x) / (air.bounds.z * f32(CELLS_X)),
        p.y / TOP_METRES,
        (p.z - air.bounds.y) / (air.bounds.w * f32(CELLS_Z)),
    );
}

// The wind at a world position, between cells.
//
// Clamped rather than wrapped: past the edge of the grid the nearest cell is
// the best answer there is, and it is the free stream anyway.
fn wind_at(p: vec3<f32>) -> vec3<f32> {
    let uvw = clamp(grid_uvw(p), vec3<f32>(0.0), vec3<f32>(1.0));
    return textureSampleLevel(velocity, air_sampler, uvw, 0.0).xyz;
}

// The ground under a world position, bilinear across the coarse field.
//
// Bilinear rather than nearest because this decides which cells are solid, and
// a nearest lookup would make the obstacle a staircase on the grid's own
// lattice -- which is exactly the artefact a coarse grid is most likely to show
// and the one hardest to tell from a bug.
fn ground_at(x: f32, z: f32) -> f32 {
    let across = vec2<f32>(
        (x - air.bounds.x) / air.bounds.z - 0.5,
        (z - air.bounds.y) / air.bounds.w - 0.5,
    );
    let low = floor(across);
    let f = across - low;
    let i = vec2<i32>(low);
    let high = vec2<i32>(i32(CELLS_X) - 1, i32(CELLS_Z) - 1);
    let a = clamp(i, vec2<i32>(0), high);
    let b = clamp(i + vec2<i32>(1, 0), vec2<i32>(0), high);
    let c = clamp(i + vec2<i32>(0, 1), vec2<i32>(0), high);
    let d = clamp(i + vec2<i32>(1, 1), vec2<i32>(0), high);
    let top = mix(
        ground[column_index(u32(a.x), u32(a.y))],
        ground[column_index(u32(b.x), u32(b.y))],
        f.x,
    );
    let bottom = mix(
        ground[column_index(u32(c.x), u32(c.y))],
        ground[column_index(u32(d.x), u32(d.y))],
        f.x,
    );
    return mix(top, bottom, f.y);
}

// Whether the ground fills a world position.
//
// A cell is solid or it is not; there is no fraction. A fractional obstacle
// would let a little flow through every mountain, and the two properties worth
// being able to state about this field -- that solid cells hold still and that
// fluid cells are divergence free -- are both statements about a hard boundary.
fn is_solid(p: vec3<f32>) -> bool {
    return p.y < ground_at(p.x, p.z);
}

fn in_grid(at: vec3<i32>) -> bool {
    return all(at >= vec3<i32>(0))
        && at.x < i32(CELLS_X)
        && at.y < i32(CELLS_Y)
        && at.z < i32(CELLS_Z);
}

// The wind aloft, brought down to the ground through the boundary layer.
fn target_at(p: vec3<f32>) -> vec3<f32> {
    let roughness = air.knobs.x;
    let above = max(p.y - ground_at(p.x, p.z), 0.0);
    let profile = clamp(
        log(1.0 + above / roughness) / log(1.0 + air.knobs.y / roughness),
        0.0,
        1.0,
    );
    return air.aloft.xyz * profile;
}

// Carries the velocity along itself and pulls it back towards the free stream.
//
// The midpoint rather than a straight step back: a first-order trace turns a
// shear into a spiral over three hundred steps, where the midpoint keeps a
// rotating field rotating. It costs one extra fetch on a kernel that is not
// what this bake spends its time in.
@compute @workgroup_size(4, 4, 4)
fn cs_air_advect(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= CELLS_X || id.y >= CELLS_Y || id.z >= CELLS_Z {
        return;
    }
    let here = cell_centre(id);
    if is_solid(here) {
        textureStore(out_velocity, vec3<i32>(id), vec4<f32>(0.0));
        return;
    }

    let dt = air.aloft.w;
    let moving = wind_at(here);
    let midpoint = wind_at(here - moving * dt * 0.5);
    var carried = wind_at(here - midpoint * dt);

    // What drives the whole field: without it it would decay to nothing, and
    // with it too strong the terrain would never get to shape anything.
    // Exponential rather than a fixed fraction, so the pull is the same per
    // second whatever the step is set to.
    let alpha = 1.0 - exp(-dt / air.knobs.z);
    carried = mix(carried, target_at(here), alpha);

    textureStore(out_velocity, vec3<i32>(id), vec4<f32>(carried, 0.0));
}

// The velocity of a neighbour, for a difference across a face.
//
// A solid neighbour and a neighbour off the edge both answer with the cell
// asking, so the difference across that face is zero and no flux is counted
// there. That is the same no-flux condition the pressure solve applies from the
// other side, and the two have to agree or the projection would work to cancel
// a flux the divergence never saw.
fn neighbour_wind(at: vec3<i32>, here: vec3<f32>) -> vec3<f32> {
    if !in_grid(at) {
        return here;
    }
    let p = cell_centre(vec3<u32>(at));
    if is_solid(p) {
        return here;
    }
    return textureLoad(velocity, at, 0).xyz;
}

@compute @workgroup_size(4, 4, 4)
fn cs_air_divergence(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= CELLS_X || id.y >= CELLS_Y || id.z >= CELLS_Z {
        return;
    }
    let index = cell_index(id);
    let centre = cell_centre(id);
    if is_solid(centre) {
        divergence[index] = 0.0;
        pressure[index] = 0.0;
        return;
    }

    let at = vec3<i32>(id);
    let here = textureLoad(velocity, at, 0).xyz;
    let east = neighbour_wind(at + vec3<i32>(1, 0, 0), here);
    let west = neighbour_wind(at - vec3<i32>(1, 0, 0), here);
    let above = neighbour_wind(at + vec3<i32>(0, 1, 0), here);
    let below = neighbour_wind(at - vec3<i32>(0, 1, 0), here);
    let far = neighbour_wind(at + vec3<i32>(0, 0, 1), here);
    let near = neighbour_wind(at - vec3<i32>(0, 0, 1), here);

    divergence[index] = (east.x - west.x) / (2.0 * air.bounds.z)
        + (above.y - below.y) / (2.0 * CELL_Y)
        + (far.z - near.z) / (2.0 * air.bounds.w);
}

// One neighbour's contribution to the Poisson stencil.
//
// Returns the pressure to weight and the weight to give it. A solid neighbour
// contributes neither, which is the Neumann condition -- no pressure gradient
// across a wall, because no flow crosses it. A neighbour off the edge
// contributes zero pressure at full weight, which is Dirichlet and lets air
// leave and enter freely: the grid is a window onto a larger sky, not a box.
fn stencil(at: vec3<i32>, span: f32) -> vec2<f32> {
    let weight = 1.0 / (span * span);
    if !in_grid(at) {
        // The floor is the one edge that is not open: under the bottom layer is
        // more ground, not more sky.
        if at.y < 0 {
            return vec2<f32>(0.0, 0.0);
        }
        return vec2<f32>(0.0, weight);
    }
    if is_solid(cell_centre(vec3<u32>(at))) {
        return vec2<f32>(0.0, 0.0);
    }
    return vec2<f32>(pressure[cell_index(vec3<u32>(at))] * weight, weight);
}

// One Gauss-Seidel sweep over the cells of one colour.
//
// Red and black are separate entry points rather than one kernel taking the
// colour, because the colour would have to arrive in the uniform and the
// uniform would then have to be rewritten between two dispatches of the same
// pass -- which `queue.write_buffer` orders at submit, not where it was called,
// so both sweeps would read whichever value was written last.
fn relax(id: vec3<u32>, colour: u32) {
    if id.x >= CELLS_X || id.y >= CELLS_Y || id.z >= CELLS_Z {
        return;
    }
    if ((id.x + id.y + id.z) & 1u) != colour {
        return;
    }
    let index = cell_index(id);
    if is_solid(cell_centre(id)) {
        pressure[index] = 0.0;
        return;
    }

    let at = vec3<i32>(id);
    var sum = 0.0;
    var weight = 0.0;
    let axes = array<vec2<f32>, 6>(
        stencil(at + vec3<i32>(1, 0, 0), air.bounds.z),
        stencil(at - vec3<i32>(1, 0, 0), air.bounds.z),
        stencil(at + vec3<i32>(0, 1, 0), CELL_Y),
        stencil(at - vec3<i32>(0, 1, 0), CELL_Y),
        stencil(at + vec3<i32>(0, 0, 1), air.bounds.w),
        stencil(at - vec3<i32>(0, 0, 1), air.bounds.w),
    );
    for (var i = 0u; i < 6u; i += 1u) {
        sum += axes[i].x;
        weight += axes[i].y;
    }

    // A cell walled in on all six sides has no equation to satisfy: nothing can
    // flow in or out of it, so any pressure will do and zero is the one that
    // disturbs its neighbours least.
    if weight <= 0.0 {
        pressure[index] = 0.0;
        return;
    }
    pressure[index] = (sum - divergence[index]) / weight;
}

@compute @workgroup_size(4, 4, 4)
fn cs_air_red(@builtin(global_invocation_id) id: vec3<u32>) {
    relax(id, 0u);
}

@compute @workgroup_size(4, 4, 4)
fn cs_air_black(@builtin(global_invocation_id) id: vec3<u32>) {
    relax(id, 1u);
}

// The pressure to use across one face when taking the gradient.
//
// The same two rules the stencil applies, in the same order, so that the field
// the projection subtracts is the gradient of the field the solve produced.
fn wall_pressure(at: vec3<i32>, here: f32) -> f32 {
    if !in_grid(at) {
        if at.y < 0 {
            return here;
        }
        return 0.0;
    }
    if is_solid(cell_centre(vec3<u32>(at))) {
        return here;
    }
    return pressure[cell_index(vec3<u32>(at))];
}

// Takes the divergence back out: what is left is the flow that fits.
@compute @workgroup_size(4, 4, 4)
fn cs_air_project(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= CELLS_X || id.y >= CELLS_Y || id.z >= CELLS_Z {
        return;
    }
    let at = vec3<i32>(id);
    if is_solid(cell_centre(id)) {
        textureStore(out_velocity, at, vec4<f32>(0.0));
        return;
    }

    let here = pressure[cell_index(id)];
    let gradient = vec3<f32>(
        (wall_pressure(at + vec3<i32>(1, 0, 0), here)
            - wall_pressure(at - vec3<i32>(1, 0, 0), here))
            / (2.0 * air.bounds.z),
        (wall_pressure(at + vec3<i32>(0, 1, 0), here)
            - wall_pressure(at - vec3<i32>(0, 1, 0), here))
            / (2.0 * CELL_Y),
        (wall_pressure(at + vec3<i32>(0, 0, 1), here)
            - wall_pressure(at - vec3<i32>(0, 0, 1), here))
            / (2.0 * air.bounds.w),
    );
    let flowing = textureLoad(velocity, at, 0).xyz - gradient;
    textureStore(out_velocity, at, vec4<f32>(flowing, 0.0));
}

// How far the air arriving at each cell has strayed from the bulk drift, and
// how far it has climbed to get there.
//
// The frame samples the cloud field at `x - mean * t - drift(x)`. The first term
// is the whole sky sliding downwind, exact and free of any diffusion because it
// is an offset rather than an advection. This is the second: what the terrain
// did to that, accumulated backwards along the streamline that ends here. Cloud
// then stretches through a valley and piles against a slope without anything
// being advected at run time at all.
//
// `w` is the rise -- how much higher this parcel is than it was `knobs.w` ago.
// Air that has just been lifted is air that has just cooled, which is where
// cloud forms, so this is the term that puts a cap on a windward slope and takes
// it off again in the lee.
//
// The window is fixed and short rather than an integral from the start of time,
// which is what keeps the offset a perturbation: a steady field integrated
// forever would stretch a cloud without bound.
@compute @workgroup_size(4, 4, 4)
fn cs_air_drift(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= CELLS_X || id.y >= CELLS_Y || id.z >= CELLS_Z {
        return;
    }
    let start = cell_centre(id);
    var p = start;
    var strayed = vec3<f32>(0.0);
    let step = air.knobs.w / f32(DRIFT_STEPS);
    for (var i = 0u; i < DRIFT_STEPS; i += 1u) {
        let moving = wind_at(p);
        strayed += (moving - air.aloft.xyz) * step;
        p -= moving * step;
    }
    textureStore(out_drift, vec3<i32>(id), vec4<f32>(strayed, start.y - p.y));
}
