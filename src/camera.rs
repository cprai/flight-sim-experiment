use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};

/// A perspective camera placed and oriented in world space.
///
/// World space is right-handed and Y-up: +X points east, +Y up, and -Z north.
/// At the identity orientation the camera looks along its local -Z (north) with
/// local +X right and +Y up, which is the usual OpenGL/glam convention.
///
/// Orientation is a quaternion rather than stored Euler angles so that an
/// aircraft can later be rolled and pitched through any attitude without
/// gimbal lock; [`Camera::from_yaw_pitch_roll`] builds one from the
/// aviation-style angles that are easier to write down.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub orientation: Quat,
    /// Vertical field of view, in radians.
    pub fov_y: f32,
    /// Viewport width divided by height.
    pub aspect: f32,
    /// Distance to the near plane. There is no far plane; see [`Camera::projection`].
    pub z_near: f32,
}

/// Vertical field of view, in degrees.
///
/// Fixed rather than configurable, and named here rather than buried in
/// [`Camera::new`], because the clipmap sizes its windows from the angle one
/// pixel subtends -- which is this over the viewport's height.
pub const FOV_Y_DEGREES: f32 = 60.0;

impl Camera {
    pub fn new(position: Vec3, orientation: Quat, aspect: f32) -> Self {
        Self {
            position,
            orientation,
            fov_y: FOV_Y_DEGREES.to_radians(),
            aspect,
            z_near: 1.0,
        }
    }

    /// A starting viewpoint that takes in a whole terrain of the given size.
    ///
    /// Sits over the middle of the terrain's southern edge looking north, high
    /// enough above its tallest point that the far edge stays comfortably
    /// inside the frustum. Everything is derived from the terrain that was
    /// loaded, so this places the camera sensibly whatever the data covers.
    pub fn overlooking(extent: Vec2, highest: f32, aspect: f32) -> Self {
        // A fraction of the terrain's own size, so the viewpoint scales with it
        // rather than being tuned to one dataset.
        const CLEARANCE: f32 = 0.08;

        let position = Vec3::new(
            0.0,
            highest + CLEARANCE * extent.max_element(),
            extent.y * 0.5,
        );
        // Aim at the far edge of the data rather than at a fixed angle. A fixed
        // pitch frames one dataset and misses every other: it puts the centre of
        // the view a fixed multiple of the eye height away, which was most of the
        // way across a raster covering ninety kilometres and several times past
        // the end of one covering eight. How much ground a run holds is now
        // whatever box was downloaded, so the angle has to follow it.
        let pitch = -(position.y / extent.y.max(f32::EPSILON)).atan();
        Self::new(position, Self::from_yaw_pitch_roll(0.0, pitch, 0.0), aspect)
    }

    /// Builds an orientation from aviation-style angles, in radians.
    ///
    /// Each angle is positive in the direction a pilot would call positive: yaw
    /// turns the nose right (east), pitch raises the nose, and roll drops the
    /// right wing. Yaw and roll run opposite to a right-handed rotation about
    /// the matching world axis, hence their negations; pitch already agrees.
    ///
    /// The angles are applied yaw, then pitch, then roll, each about the axes
    /// left by the previous one, so yaw is a compass heading and pitch is
    /// measured from the horizon regardless of heading.
    pub fn from_yaw_pitch_roll(yaw: f32, pitch: f32, roll: f32) -> Quat {
        Quat::from_euler(EulerRot::YXZ, -yaw, pitch, -roll)
    }

    /// World space -> view space.
    pub fn view(&self) -> Mat4 {
        // The camera's model matrix is `translate(position) * rotate(orientation)`;
        // the view matrix is its inverse, which for a rigid transform is just the
        // conjugate rotation applied after undoing the translation.
        Mat4::from_quat(self.orientation.inverse()) * Mat4::from_translation(-self.position)
    }

    /// View space -> clip space, in wgpu's NDC convention: Y-up, depth in 0..1.
    ///
    /// Depth is reversed -- the near plane maps to 1 and infinity to 0 -- and there
    /// is no far plane at all. Terrain spans tens of kilometres with kilometres of
    /// relief, and a conventional projection cannot resolve that: depth precision
    /// falls off as the square of distance, leaving hundreds of metres of
    /// quantization at the far end of the view. Reversing the range makes the stored
    /// value proportional to `1/z`, which is exactly where a float depth buffer's
    /// exponent puts its resolution, so precision stays roughly constant with
    /// distance. Pair this with a `Depth32Float` buffer and a `Greater` compare.
    pub fn projection(&self) -> Mat4 {
        glam::camera::rh::proj::directx::perspective_infinite_reverse(
            self.fov_y,
            self.aspect,
            self.z_near,
        )
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection() * self.view()
    }

    /// Right, up, and forward vectors that turn a screen position into a ray.
    ///
    /// A pixel at normalized device coordinates `(x, y)` looks along
    /// `x * right + y * up + forward`, before normalizing. The two lateral
    /// vectors carry the field of view and the aspect ratio, so a shader that
    /// needs a world-space ray per pixel -- the raymarched far field -- needs
    /// neither the projection matrix nor its inverse, and pays a couple of
    /// multiply-adds instead of a matrix product and a perspective divide.
    ///
    /// Lengths are those of the near plane at unit distance, so the returned
    /// forward vector is a unit vector and the others are scaled relative to it.
    /// Nothing here depends on the depth range, so the reversed, infinite
    /// projection needs no special handling.
    pub fn ray_basis(&self) -> [Vec3; 3] {
        let half_height = (self.fov_y * 0.5).tan();
        [
            self.orientation * Vec3::X * half_height * self.aspect,
            self.orientation * Vec3::Y * half_height,
            self.orientation * Vec3::NEG_Z,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Projects a world point and returns its normalized device coordinates.
    fn project(camera: &Camera, point: Vec3) -> Vec3 {
        let clip = camera.view_projection() * point.extend(1.0);
        clip.truncate() / clip.w
    }

    fn camera_at_origin() -> Camera {
        Camera::new(Vec3::ZERO, Quat::IDENTITY, 16.0 / 9.0)
    }

    #[test]
    fn identity_orientation_looks_north() {
        let view = camera_at_origin().view();
        // The view matrix maps world axes onto the camera's own: looking north
        // down -Z puts world -Z at view -Z, world +X on the right, +Y up.
        assert!(
            view.transform_vector3(Vec3::NEG_Z)
                .abs_diff_eq(Vec3::NEG_Z, 1e-6)
        );
        assert!(view.transform_vector3(Vec3::X).abs_diff_eq(Vec3::X, 1e-6));
        assert!(view.transform_vector3(Vec3::Y).abs_diff_eq(Vec3::Y, 1e-6));
    }

    #[test]
    fn yaw_pitch_roll_follow_aviation_sign_conventions() {
        let yawed = Camera::from_yaw_pitch_roll(90f32.to_radians(), 0.0, 0.0);
        // Nose right of north is east, i.e. +X.
        assert!((yawed * Vec3::NEG_Z).abs_diff_eq(Vec3::X, 1e-6));

        let pitched = Camera::from_yaw_pitch_roll(0.0, 90f32.to_radians(), 0.0);
        // Nose up.
        assert!((pitched * Vec3::NEG_Z).abs_diff_eq(Vec3::Y, 1e-6));

        let rolled = Camera::from_yaw_pitch_roll(0.0, 0.0, 90f32.to_radians());
        // Right wing dropped, so the right vector now points at the ground.
        assert!((rolled * Vec3::X).abs_diff_eq(Vec3::NEG_Y, 1e-6));
    }

    #[test]
    fn point_straight_ahead_projects_to_the_center() {
        let camera = camera_at_origin();
        let ndc = project(&camera, Vec3::new(0.0, 0.0, -100.0));
        assert!(ndc.x.abs() < 1e-5, "expected centered, got {ndc}");
        assert!(ndc.y.abs() < 1e-5, "expected centered, got {ndc}");
        assert!(
            (0.0..=1.0).contains(&ndc.z),
            "expected in depth range, got {ndc}"
        );
    }

    #[test]
    fn depth_is_reversed_with_the_near_plane_at_one() {
        let camera = camera_at_origin();

        // The near plane sits at the top of the range, not the bottom.
        let at_near = project(&camera, Vec3::new(0.0, 0.0, -camera.z_near));
        assert!(
            (at_near.z - 1.0).abs() < 1e-5,
            "near plane should map to 1, got {at_near}"
        );

        // ... so nearer geometry has the *greater* depth value, which is what the
        // `Greater` depth compare in the pipeline relies on.
        let near = project(&camera, Vec3::new(0.0, 0.0, -100.0)).z;
        let far = project(&camera, Vec3::new(0.0, 0.0, -10_000.0)).z;
        assert!(near > far, "expected reversed depth, got {near} then {far}");
    }

    #[test]
    fn very_distant_geometry_is_not_clipped() {
        let camera = camera_at_origin();
        // No far plane, so terrain hundreds of kilometres out still projects into
        // the depth range instead of vanishing at an arbitrary cutoff.
        let ndc = project(&camera, Vec3::new(0.0, 0.0, -200_000.0));
        assert!(
            (0.0..=1.0).contains(&ndc.z),
            "distant point should survive clipping, got {ndc}"
        );
        assert!(
            ndc.z > 0.0,
            "depth should not have collapsed to 0, got {ndc}"
        );
    }

    #[test]
    fn pitching_down_moves_the_horizon_up_the_screen() {
        let mut camera = camera_at_origin();
        let far_ahead = Vec3::new(0.0, 0.0, -1000.0);

        let level = project(&camera, far_ahead).y;
        camera.orientation = Camera::from_yaw_pitch_roll(0.0, -15f32.to_radians(), 0.0);
        let tilted = project(&camera, far_ahead).y;

        assert!(
            tilted > level,
            "looking down should raise distant geometry: {level} -> {tilted}"
        );
    }

    #[test]
    fn the_ray_through_the_centre_of_the_screen_is_the_forward_axis() {
        let camera = Camera::new(
            Vec3::new(10.0, 20.0, 30.0),
            Camera::from_yaw_pitch_roll(0.7, -0.3, 0.2),
            16.0 / 9.0,
        );
        let [.., forward] = camera.ray_basis();
        // Normalized device coordinates (0, 0) contribute neither lateral
        // vector, so the middle of the screen looks straight down the barrel.
        assert!(forward.abs_diff_eq(camera.orientation * Vec3::NEG_Z, 1e-6));
        assert!((forward.length() - 1.0).abs() < 1e-6, "{forward}");
    }

    #[test]
    fn rays_agree_with_the_projection_they_have_to_invert() {
        let camera = Camera::new(
            Vec3::new(-40.0, 120.0, 55.0),
            Camera::from_yaw_pitch_roll(-1.1, 0.4, -0.25),
            16.0 / 9.0,
        );
        let [right, up, forward] = camera.ray_basis();

        // The corners are where an error in the field of view or the aspect
        // shows up; the centre would pass with both wrong.
        for (x, y) in [
            (0.0, 0.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
            (0.35, -0.8),
        ] {
            let ray = (right * x + up * y + forward).normalize();
            // Any distance along the ray lands on the same pixel, so projecting
            // one point back is a complete check of the direction.
            let ndc = project(&camera, camera.position + ray * 5_000.0);
            assert!(
                (ndc.x - x).abs() < 1e-4 && (ndc.y - y).abs() < 1e-4,
                "ray through ({x}, {y}) projects back to {ndc}"
            );
        }
    }

    #[test]
    fn ground_ahead_of_a_raised_camera_projects_below_the_center() {
        let camera = Camera::new(Vec3::new(0.0, 50.0, 0.0), Quat::IDENTITY, 16.0 / 9.0);
        let ndc = project(&camera, Vec3::new(0.0, 0.0, -200.0));
        assert!(ndc.y < 0.0, "ground should be in the lower half, got {ndc}");
    }
}
