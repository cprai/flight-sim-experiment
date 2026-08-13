// Shading: the G-buffer in, the image out.
//
// The geometry pass has already done the expensive part -- every pixel holds
// the material id, world position and surface normal of the ground its ray
// met, or depth zero where it met nothing. This pass is one fetch, one table
// lookup and one dot product per pixel: the table gives the material's colour
// and the normal decides how much of the sun reaches it. Real material texture
// belongs here later, which is the point of the split -- it will not touch the
// march.

// Must match `PALETTE_SIZE` in `src/palette.rs`: one slot per id up to the
// last assigned category block.
const PALETTE_SIZE: u32 = 2304u;

// Missing data. Ids at or past the table's end can only come from a corrupt
// tile -- in-range unassigned ids are magenta in the table itself.
const MAGENTA: vec4<f32> = vec4<f32>(1.0, 0.0, 1.0, 1.0);

// Must match `MATERIAL_MASK` in `src/terrain.wgsl`.
const MATERIAL_MASK: u32 = 0xffffu;

// The planet and the tables' shapes. Must match `src/sky.rs` and
// `src/sky.wgsl`; see the module doc of `src/sky.rs` for what pinning a round
// atmosphere under a flat world costs.
const GROUND_RADIUS: f32 = 6360000.0;
const TOP_RADIUS: f32 = 6460000.0;
const TRANSMITTANCE_WIDTH: u32 = 256u;
const TRANSMITTANCE_HEIGHT: u32 = 64u;
const MULTISCATTER_SIZE: u32 = 32u;
const SKYVIEW_HEIGHT: u32 = 108u;
const AERIAL_SLICES: u32 = 64u;
const AERIAL_FAR: f32 = 100000.0;

const PI: f32 = 3.14159265358979;

// The light volumes' shape. Must match `LIGHT_ACROSS` and `LIGHT_SLICES` in
// `src/cloud.rs`.
const LIGHT_ACROSS: u32 = 192u;
const LIGHT_SLICES: u32 = 48u;

// How much of the sky a cloud blocks still arrives, having bounced. Must match
// `BOUNCED` in `src/cloud_march.wgsl`; see there for what it stands in for.
const BOUNCED: f32 = 0.4;

// How much display brightness a unit of radiance is worth.
//
// Everything upstream is in units of the sun's irradiance at the top of the
// atmosphere, where level ground of albedo `a` under the reference 45-degree
// sun comes out at about `0.209 a`. The number that follows is the one that
// puts that back where the old two-constant light left it -- `0.35 + 0.65 cos
// 45`, or `0.81 a` -- so the change of model is a change of *behaviour* and not
// a change of overall brightness. Must match `EXPOSURE` in `src/sky.rs`.
const EXPOSURE: f32 = 5.0;

// The radiance that maps to white. Must match `WHITE` in `src/sky.rs`.
//
// 1.6 in unexposed units, which is far above sunlit snow at about 0.3 and far
// below the sun's own disc. So the disc clips to white and nothing else in the
// frame does.
const WHITE: f32 = 8.0;

struct Camera {
    view_proj: mat4x4<f32>,
    was_view_proj: mat4x4<f32>,
    position: vec4<f32>,
    ray_right: vec4<f32>,
    ray_up: vec4<f32>,
    ray_forward: vec4<f32>,
};

struct Palette {
    colours: array<vec4<f32>, 2304>,
};

// What the world is lit by. Mirrors `SkyUniform` in `src/sky.rs`.
//
// `sun` is the unit vector pointing at the sun from the ground; `w` is unused
// padding.
struct Sky {
    sun: vec4<f32>,
    eye: vec4<f32>,
    up: vec4<f32>,
    sun_tangent: vec4<f32>,
};

// Where the light volumes stand and how their columns lean. Must match `Light`
// in `src/cloud_march.wgsl` and `LightUniform` in `src/cloud.rs`.
struct Light {
    origin: vec4<f32>,
    walk: vec4<f32>,
};

// Group 0 is the camera, as it is for every pipeline in this program: the
// shading needs a world position now, and rebuilds it from the depth and the
// same ray basis the march walked.
@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var<uniform> sky: Sky;

// The scattering tables and the sampler that reads between their texels. Built
// once at load; see `src/sky.wgsl`.
@group(2) @binding(0) var lut_sampler: sampler;
@group(2) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(2) @binding(2) var multiscatter_lut: texture_2d<f32>;
// Wrapping in `u`; see `skyview_u` in `src/sky.wgsl`.
@group(2) @binding(3) var skyview_sampler: sampler;
@group(2) @binding(4) var skyview_lut: texture_2d<f32>;
// The air in front of every part of the frame, sliced by distance: the
// in-scattered light and, separately, what survives of what was behind it.
@group(2) @binding(5) var aerial_scatter_lut: texture_3d<f32>;
@group(2) @binding(6) var aerial_transmit_lut: texture_3d<f32>;

@group(3) @binding(0) var<uniform> palette: Palette;
// A material id in the low sixteen bits and where inside its pixel the ground
// sits in the rest; only the id is wanted here. See `MATERIAL_MASK` in
// `src/terrain.wgsl` for what the other half is for.
@group(3) @binding(1) var material: texture_2d<u32>;
@group(3) @binding(3) var depth: texture_2d<f32>;
@group(3) @binding(4) var normal: texture_2d<f32>;
// What the cloud march left, at half this pass's resolution: scattered light
// and, in `w`, how much of what is behind it survives. Beside it, how far along
// the ray that cloud was, in metres.
//
// Loaded rather than sampled, and declared unfilterable so that nothing can
// quietly start sampling them: the upsample below is bilateral, and a hardware
// bilinear across a mountain's silhouette blends cloud that was marched past the
// mountain into pixels the mountain covers. That is the halo this pass exists to
// avoid.
@group(3) @binding(2) var cloud_colour: texture_2d<f32>;
@group(3) @binding(5) var cloud_along: texture_2d<f32>;
// How much of the sun reaches each point of a coarse world grid, and how much
// of the sky does. The same two volumes the cloud march lights itself from, read
// at the ground instead: a cloud's shadow on a mountain and a cloud's shadow on
// another cloud are the same fact, looked up in the same place.
//
// Sampled rather than loaded, through the tables' own clamped sampler -- these
// clamp too, and what is wanted between their texels is exactly what a linear
// fetch gives. They are not maxima; see `cloud_colour` above for the buffers
// that are.
@group(3) @binding(6) var sun_light: texture_3d<f32>;
@group(3) @binding(7) var sky_light: texture_3d<f32>;
@group(3) @binding(8) var<uniform> light: Light;

// The ray through a point on the screen, before it is normalised.
//
// Must match `ndc_of` and `ray_raw_at` in `src/terrain.wgsl`, character for
// character. There is no preprocessor here and no `#include`, so this is a
// second copy of the march's own arithmetic, and it has to be the same
// arithmetic: a last-bit difference would put the air on a slightly different
// ray from the ground it is colouring, which stands still in a still frame and
// shimmers in a moving one. There is a test comparing the two as text.
//
// The march takes the viewport from its own uniform where this takes it from
// the depth buffer's dimensions. They are always the same number -- both come
// from the one viewport `Scene::resize` hands out -- but that is an invariant
// rather than a proof, so it is written down here rather than assumed.
//
// Unnormalised, with its component along the view axis exactly one, which is
// what makes `distance_at` a multiply rather than a divide by a length.
fn ray_raw_at(screen: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(depth));
    let ndc = vec2<f32>(
        screen.x / size.x * 2.0 - 1.0,
        1.0 - screen.y / size.y * 2.0,
    );
    return camera.ray_right.xyz * ndc.x + camera.ray_up.xyz * ndc.y + camera.ray_forward.xyz;
}

// How far along `ray_raw_at` a reversed-Z depth puts a point. Must match
// `distance_at` in `src/terrain.wgsl`; `ray_forward.w` carries the near plane.
fn distance_at(d: f32) -> f32 {
    return camera.ray_forward.w / d;
}

// The half-texel correction. Must match `to_texture` in `src/sky.wgsl`.
fn to_texture(x: f32, n: f32) -> f32 {
    return 0.5 / n + x * (1.0 - 1.0 / n);
}

fn top_distance(r: f32, mu: f32) -> f32 {
    let discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
    return max(-r * mu + sqrt(max(discriminant, 0.0)), 0.0);
}

// Must match `transmittance_uv` in `src/sky.wgsl`.
fn transmittance_uv(r: f32, mu: f32) -> vec2<f32> {
    let horizon = sqrt(max(r * r - GROUND_RADIUS * GROUND_RADIUS, 0.0));
    let atmosphere = sqrt(max(TOP_RADIUS * TOP_RADIUS - GROUND_RADIUS * GROUND_RADIUS, 0.0));
    let distance = top_distance(r, mu);
    let shortest = TOP_RADIUS - r;
    let longest = horizon + atmosphere;
    let span = max(longest - shortest, 1e-6);
    return vec2<f32>(
        to_texture(clamp((distance - shortest) / span, 0.0, 1.0), f32(TRANSMITTANCE_WIDTH)),
        to_texture(clamp(horizon / atmosphere, 0.0, 1.0), f32(TRANSMITTANCE_HEIGHT)),
    );
}

fn sample_transmittance(r: f32, mu: f32) -> vec3<f32> {
    return textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(r, mu), 0.0).rgb;
}

// Must match `multiscatter_uv` in `src/sky.wgsl`.
fn sample_multiscatter(r: f32, mu_s: f32) -> vec3<f32> {
    let altitude = clamp((r - GROUND_RADIUS) / (TOP_RADIUS - GROUND_RADIUS), 0.0, 1.0);
    let uv = vec2<f32>(
        to_texture(clamp(mu_s * 0.5 + 0.5, 0.0, 1.0), f32(MULTISCATTER_SIZE)),
        to_texture(altitude, f32(MULTISCATTER_SIZE)),
    );
    return textureSampleLevel(multiscatter_lut, lut_sampler, uv, 0.0).rgb;
}

// Where a world point sits in a light volume. Must match `light_uvw` in
// `src/cloud_march.wgsl`; there is a test comparing the four functions here as
// text against that file, because a shadow the ground draws in a different place
// from the one the cloud draws it is a cloud floating free of its own shadow.
fn light_uvw(p: vec3<f32>, shear: vec2<f32>) -> vec3<f32> {
    let climbed = p.y - light.origin.z;
    let flat = p.xz - light.origin.xy - climbed * shear;
    return vec3<f32>(
        flat / (light.origin.w * f32(LIGHT_ACROSS)),
        climbed / (light.walk.w * f32(LIGHT_SLICES)),
    );
}

// How far outside a light volume a coordinate has strayed, as a share of it.
// Must match `beyond_light` in `src/cloud_march.wgsl`.
fn beyond_light(uvw: vec3<f32>) -> f32 {
    let out = abs(uvw.xy - vec2<f32>(0.5)) * 2.0;
    return max(out.x, out.y);
}

// Must match `sun_reaching` in `src/cloud_march.wgsl`, but for the sampler:
// there the fields tile and are read through a repeating one, here everything
// else in the group is a table and the clamped one is already to hand.
fn sun_reaching(p: vec3<f32>) -> f32 {
    let uvw = light_uvw(p, light.walk.xy);
    let blocked = textureSampleLevel(sun_light, lut_sampler, uvw, 0.0).r;
    return 1.0 - blocked * (1.0 - smoothstep(0.9, 1.0, beyond_light(uvw)));
}

// Must match `sky_reaching` in `src/cloud_march.wgsl`, likewise.
fn sky_reaching(p: vec3<f32>) -> f32 {
    let uvw = light_uvw(p, vec2<f32>(0.0));
    let blocked = textureSampleLevel(sky_light, lut_sampler, uvw, 0.0).r;
    return 1.0 - blocked * (1.0 - smoothstep(0.9, 1.0, beyond_light(uvw)));
}

// Must match `sky_share` in `src/cloud_march.wgsl`.
fn sky_share(reaching: f32) -> f32 {
    return 1.0 - (1.0 - BOUNCED) * (1.0 - reaching);
}

// The zenith angle of the horizon. Must match `horizon_zenith` in
// `src/sky.wgsl`.
fn horizon_zenith(r: f32) -> f32 {
    return PI - asin(clamp(GROUND_RADIUS / max(r, GROUND_RADIUS), -1.0, 1.0));
}

// Must match `skyview_v` in `src/sky.wgsl`.
fn skyview_v(r: f32, zenith: f32) -> f32 {
    let horizon = horizon_zenith(r);
    if zenith < horizon {
        return 0.5 * (1.0 - sqrt(max(1.0 - zenith / horizon, 0.0)));
    }
    return 0.5 + 0.5 * sqrt(max((zenith - horizon) / (PI - horizon), 0.0));
}

// The sky in a direction, from the table. Must match `sample_skyview` in
// `src/sky.wgsl`, including that it reads below the horizon as well as above:
// the table marches air and never adds what the ground reflects, so its lower
// half is the haze a downward ray crosses, which is what a world whose terrain
// simply stops should show.
fn sample_skyview(direction: vec3<f32>) -> vec3<f32> {
    let up = sky.up.xyz;
    let forward = sky.sun_tangent.xyz;
    let side = cross(up, forward);
    let flat = normalize(direction - up * dot(up, direction));
    // No half-texel correction across: that axis wraps. See `src/sky.wgsl`.
    let u = 0.5 + atan2(dot(side, flat), dot(forward, flat)) * (0.5 / PI);
    let zenith = acos(clamp(dot(up, direction), -1.0, 1.0));
    let v = to_texture(skyview_v(sky.eye.w, zenith), f32(SKYVIEW_HEIGHT));
    return textureSampleLevel(skyview_lut, skyview_sampler, vec2<f32>(u, v), 0.0).rgb;
}

// Where a view-axis distance sits down the aerial-perspective volume. Must
// match `aerial_w` and `aerial_z` in `src/sky.wgsl`.
fn aerial_depth(along: f32) -> f32 {
    let w = sqrt(clamp(along, 0.0, AERIAL_FAR) / AERIAL_FAR);
    return w - 0.5 / f32(AERIAL_SLICES);
}

// Half the sun's apparent width, in radians: 0.5334 degrees across, which is
// what it is from this planet. Must match `SUN_ANGULAR_RADIUS` in `src/sky.rs`.
const SUN_ANGULAR_RADIUS: f32 = 0.004654;

// One over the disc's solid angle, `2 pi (1 - cos r)` = 6.805e-5 steradians.
//
// Everything else here is a radiance and the sun is given as an irradiance, so
// this is what turns one into the other: spreading the whole of the sun's light
// over the small patch of sky it actually occupies. Must match
// `SUN_DISC_RADIANCE` in `src/sky.rs`.
const SUN_DISC_RADIANCE: f32 = 14696.0;

// The sun itself, where the ray points at it.
//
// Only ever called on a pixel whose ray found no ground, which is what makes
// terrain occlude the disc: there is no separate visibility test and none is
// wanted, because the march has already answered that question exactly.
fn sun_disc(direction: vec3<f32>) -> vec3<f32> {
    let angle = acos(clamp(dot(direction, sky.sun.xyz), -1.0, 1.0));
    // One pixel of feather, so the edge is a curve rather than a staircase.
    // The disc is about six pixels across at 720 rows and a 60-degree field,
    // which is small enough that a hard cut is plainly stepped.
    let feather = sky.up.w;
    let edge = 1.0 - smoothstep(
        SUN_ANGULAR_RADIUS - feather,
        SUN_ANGULAR_RADIUS + feather,
        angle,
    );
    if edge <= 0.0 {
        return vec3<f32>(0.0);
    }

    // Limb darkening: the disc is a sphere of gas seen through more of its own
    // atmosphere at the edges than at the middle, so it is not flat. Invisible
    // while the sun is high -- everything here is thousands of times over the
    // white point and clips alike -- and the whole of the shape of it once the
    // air has taken the disc down near that point, which is the only time
    // anyone looks straight at one.
    let across = clamp(angle / SUN_ANGULAR_RADIUS, 0.0, 1.0);
    let limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - across * across, 0.0)));

    // Through the same air the sky in front of it went through, which is what
    // reddens the disc as it sets and what makes it vanish on its own once it
    // is down. No branch for "the sun has set": the table already answers that.
    let mu = dot(sky.up.xyz, direction);
    return vec3<f32>(SUN_DISC_RADIANCE * edge * limb) * sample_transmittance(sky.eye.w, mu);
}

// Radiance to a displayable colour: expose, then roll the top off.
//
// Extended Reinhard, per channel, and chosen over a fitted curve like ACES for
// one reason -- it inverts in closed form, so a test can predict a byte from a
// radiance instead of restating the shader. It is the identity to first order
// near zero, so dark ground still scales linearly with the light falling on it;
// it passes exactly through one at `WHITE`, so the white point is a number
// rather than a fitted constant; and it is monotone, so nothing ever crosses
// over anything else.
//
// In linear space, because the render target is sRGB and re-encodes on write --
// the same arrangement the palette has always relied on.
fn tonemap(radiance: vec3<f32>) -> vec3<f32> {
    let x = radiance * EXPOSURE;
    return saturate(x * (1.0 + x / (WHITE * WHITE)) / (1.0 + x));
}

// How much of the sky's own distance a block and a pixel may disagree by before
// the block stops speaking for the pixel.
//
// A hundredth of `AERIAL_FAR`, so a kilometre of view-axis distance. Below that
// two taps are treated as describing the same place, which is what keeps a
// smooth slope from being upsampled through one tap at a time; above it a tap
// falls away as the reciprocal of how far out it is. A sky tap beside a ground
// pixel two kilometres off is down by a hundred, which is the case this is set
// for.
const BILATERAL_BIAS: f32 = 0.01;

// The cloud in front of one pixel, and where it was.
struct Cloud {
    // Scattered light, and what survives of everything behind it.
    lit: vec4<f32>,
    // The view-axis distance to put that light at, or zero where there is none.
    along: f32,
};

// How far the cloud march let a block of four pixels run, along the view axis.
//
// The same rule the march itself applies -- the farthest of the four, and
// unbounded if any of them found no ground -- so this reproduces the reach that
// produced the tap rather than guessing at it. In view-axis units where the
// march works in metres, which costs nothing: this is only ever compared
// against another number in the same units, never used to rebuild a position.
fn block_reach(block: vec2<i32>) -> f32 {
    let full = vec2<i32>(textureDimensions(depth));
    var reach = 0.0;
    for (var j = 0; j < 2; j += 1) {
        for (var i = 0; i < 2; i += 1) {
            let d = textureLoad(depth, min(block + vec2<i32>(i, j), full - 1), 0).r;
            if d == 0.0 {
                reach = AERIAL_FAR;
            } else {
                reach = max(reach, min(distance_at(d), AERIAL_FAR));
            }
        }
    }
    return reach;
}

// The half-resolution cloud buffer, read at a full-resolution pixel.
//
// A bilinear weight per tap, divided by how far that tap's block reached from
// what this pixel is showing. Across the open sky and across a continuous slope
// every tap agrees and this is an ordinary bilinear; at a silhouette the taps
// that marched past the ridge fall away by two orders of magnitude and the pixel
// is reconstructed from the ones that stopped where it did.
//
// The distance comes from the single best tap and is never blended, which is not
// fussiness: a blended distance addresses the aerial-perspective volume at a
// place no cloud is, and between a cloud at two kilometres and one at forty the
// average is a haze belonging to neither.
fn cloud_at(pixel: vec2<i32>, along: f32) -> Cloud {
    // Half-resolution texel `h` stands for the point where its block's four
    // pixels meet, which is full-resolution coordinate `2h + 1`; see
    // `cs_cloud_march`. So a pixel centre at `p + 0.5` sits at `(p - 0.5) / 2`
    // in the half-resolution grid.
    let at = (vec2<f32>(pixel) - 0.5) * 0.5;
    let corner = floor(at);
    let f = at - corner;
    let last = vec2<i32>(textureDimensions(cloud_colour)) - 1;

    var lit = vec4<f32>(0.0);
    var total = 0.0;
    var best = -1.0;
    var found = 0.0;
    var empty = true;
    for (var j = 0; j < 2; j += 1) {
        for (var i = 0; i < 2; i += 1) {
            let tap = clamp(vec2<i32>(corner) + vec2<i32>(i, j), vec2<i32>(0), last);
            let share = mix(1.0 - f.x, f.x, f32(i)) * mix(1.0 - f.y, f.y, f32(j));
            let apart = abs(block_reach(tap * 2) - along) / AERIAL_FAR;
            let weight = share / (apart + BILATERAL_BIAS);
            let marched = textureLoad(cloud_colour, tap, 0);
            empty = empty && marched.w >= 1.0;
            lit = lit + marched * weight;
            total = total + weight;
            if weight > best {
                best = weight;
                found = textureLoad(cloud_along, tap, 0).r;
            }
        }
    }

    var out: Cloud;
    // Read off the taps rather than off the average of them. Four taps that all
    // say "nothing here" average to a transmittance of one in exact
    // arithmetic and to one and a hair either side of it in this one: the
    // weights are summed twice, once into `total` and once through the alpha
    // channel, and nothing obliges a compiler to associate the two sums the same
    // way. Dividing then leaves a last-bit error, which is enough to take a
    // pixel down the compositing path and shift its colour by a count. It shows
    // as a fine magenta static over the whole frame when the two paths are
    // painted apart -- which is how this was found, since the difference from a
    // clear sky is otherwise a handful of counts nobody would see.
    if empty {
        out.lit = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
        out.lit = lit / max(total, 1e-6);
    }
    // The buffer holds metres and the tables are addressed along the view axis,
    // which is more than a metre per unit anywhere but the middle of the frame.
    out.along = found / max(length(ray_raw_at(vec2<f32>(pixel) + 0.5)), 1e-6);
    return out;
}

// The cloud put in front of what was already there.
//
// Everything the frame had is radiance measured at the eye, so it already
// carries the air between the eye and it. Putting something halfway along that
// path means undoing the near half of the air, inserting the cloud, and putting
// the near half back:
//
//   beyond = (background - haze(t)) / through(t)
//   final  = haze(t) + through(t) * (scattered + transmitted * beyond)
//
// which collapses to `background` exactly when the cloud is absent, and which
// is why a cloud reddens at sunset and dims the sun's disc for nothing extra --
// the transmittance multiplies the whole background, disc included.
//
// The clamp at zero is not decoration. The sky-view table and the aerial volume
// are built by different marches over the same air and disagree by a hair near
// the horizon, where `background` can come out a shade under the haze that is
// supposed to be part of it.
fn composite(background: vec3<f32>, screen: vec2<f32>, cloud: Cloud) -> vec3<f32> {
    // A pixel with nothing in front of it takes the path it always took, to the
    // last bit. That is what lets a clear sky be checked against the frames this
    // pass drew before there were clouds in it.
    if cloud.lit.w >= 1.0 {
        return background;
    }
    let uvw = vec3<f32>(
        screen / vec2<f32>(textureDimensions(depth)),
        aerial_depth(cloud.along),
    );
    let haze = textureSampleLevel(aerial_scatter_lut, lut_sampler, uvw, 0.0).rgb;
    let through = max(
        textureSampleLevel(aerial_transmit_lut, lut_sampler, uvw, 0.0).rgb,
        vec3<f32>(1e-4),
    );
    let beyond = max((background - haze) / through, vec3<f32>(0.0));
    return haze + through * (cloud.lit.rgb + cloud.lit.w * beyond);
}

@vertex
fn vs_shade(@builtin(vertex_index)index: u32) -> @builtin(position) vec4<f32> {
    // The same oversized triangle the geometry pass draws.
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 1.0, 1.0);
}

@fragment
fn fs_shade(@builtin(position)clip: vec4<f32>) -> @location(0) vec4<f32> {
    // The fragment coordinate is the pixel centre, so truncation is the index.
    let pixel = vec2<i32>(clip.xy);

    // The march writes zero depth where its ray found no ground, and the
    // reversed-infinite projection cannot write zero for any finite hit, so
    // this test is exact: nothing is there and it is sky.
    //
    // Recomputed here every frame rather than stored, which is what keeps a
    // moving sun honest: the reprojection carries sky pixels between frames as
    // a fact about a *direction*, and a direction is all this needs.
    if textureLoad(depth, pixel, 0).r == 0.0 {
        let towards = normalize(ray_raw_at(clip.xy));
        let sky = sample_skyview(towards) + sun_disc(towards);
        // A sky ray reaches nothing, so as far as the cloud upsample is
        // concerned it is showing whatever is at the end of the table.
        let cloud = cloud_at(pixel, AERIAL_FAR);
        return vec4<f32>(tonemap(composite(sky, clip.xy, cloud)), 1.0);
    }

    let id = textureLoad(material, pixel, 0).r & MATERIAL_MASK;
    var albedo = MAGENTA.rgb;
    if id < PALETTE_SIZE {
        albedo = palette.colours[id].rgb;
    }

    // Where this pixel's ground actually is, rebuilt from the depth and the
    // same ray the march walked. The G-buffer stores no world position on
    // purpose -- a depth says the same thing in four bytes rather than sixteen
    // -- and this is the rebuild `src/deferred.rs` has always said a haze would
    // want. The sub-pixel offset is not applied: it moves the point by under
    // half a pixel, and the air at that scale is the same air.
    let raw = ray_raw_at(clip.xy);
    let ground = camera.position.xyz + raw * distance_at(textureLoad(depth, pixel, 0).r);
    // Into the planet's frame, whose centre sits one ground radius below the
    // world's origin. See the module doc of `src/sky.rs`.
    let centred = ground + vec3<f32>(0.0, GROUND_RADIUS, 0.0);
    let radius = max(length(centred), GROUND_RADIUS);
    let up = centred / radius;
    let sun_mu = dot(up, sky.sun.xyz);

    let surface = textureLoad(normal, pixel, 0).xyz;

    // Direct sun: what a surface at this angle collects, times what survives
    // the air between this patch of ground and space. That second factor is the
    // whole of what the old `SUNLIGHT` constant was standing in for, and it is
    // where a low sun turns orange -- the same table that reddens the sky
    // reddens what the sky is lighting.
    //
    // ... and by whatever cloud stands between this patch of ground and the
    // sun, which is the same volume the cloud march shades itself from, read at
    // the ground instead. There is no shadow map and there is no second
    // traversal: a cloud's shadow on a mountain and a cloud's shadow on another
    // cloud are one fact, and it was integrated once.
    //
    // Still no shadow from the terrain itself. A slope facing the sun is bright
    // whatever ridge stands between it and the sun, so the relief that brings
    // out is local and a mountain does not yet darken the valley behind it.
    let direct = sample_transmittance(radius, sun_mu)
        * max(dot(surface, sky.sun.xyz), 0.0)
        * sun_reaching(ground);

    // ... and the sky itself, which is what `AMBIENT` was standing in for. The
    // multiple-scattering table already *is* the sky's isotropic radiance, so
    // the irradiance onto a surface facing straight up is `pi` times it. The
    // wrap term takes that down for a slope that faces the ground rather than
    // the dome -- a cliff sees half a sky, not a whole one. Not a fifth table:
    // a proper irradiance table would be a whole extra precompute for a
    // difference the haze in front of it will shortly dominate.
    //
    // Dimmed by the cloud overhead as well, and that half is not optional:
    // taking the sun away and leaving the sky at full strength gives an
    // overcast landscape that is bright and flat rather than dim, which looks
    // less like weather than like the sun having been switched off.
    let ambient = sample_multiscatter(radius, sun_mu)
        * PI
        * (0.5 + 0.5 * dot(surface, up))
        * sky_share(sky_reaching(ground));

    // A Lambertian surface scatters what lands on it over a hemisphere, which
    // is the `1 / pi`. The palette is stored linearised and is an albedo now
    // rather than a finished colour -- what fraction of each wavelength the
    // ground sends back -- and everything above is in units of the sun's own
    // irradiance, so this line is where radiance is made.
    let leaving = albedo * (direct + ambient) / PI;

    // ... and this is what the air between here and there does to it: takes
    // some of it out, and puts its own scattered light in front. Addressed by
    // the same view-axis distance the depth already encodes, at this pixel's
    // own place in the frame, so there is no normalise and no length -- the
    // froxel column standing over this pixel marched the very ray the march
    // did.
    //
    // The viewport comes from the depth buffer's dimensions, which is the same
    // number the froxel volume's own basis was built from.
    let along = distance_at(textureLoad(depth, pixel, 0).r);
    let uvw = vec3<f32>(
        clip.xy / vec2<f32>(textureDimensions(depth)),
        aerial_depth(along),
    );
    let haze = textureSampleLevel(aerial_scatter_lut, lut_sampler, uvw, 0.0).rgb;
    let through = textureSampleLevel(aerial_transmit_lut, lut_sampler, uvw, 0.0).rgb;
    // ... and then whatever cloud stands between the eye and all of that. The
    // ground, the air in front of it and the cloud in front of that are three
    // layers of one integral, composited in that order.
    let cloud = cloud_at(pixel, along);
    return vec4<f32>(
        tonemap(composite(leaving * through + haze, clip.xy, cloud)),
        1.0,
    );
}
