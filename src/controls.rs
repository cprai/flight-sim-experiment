use glam::{EulerRot, Vec3};
use winit::keyboard::KeyCode;

use crate::camera::Camera;

/// Base translation speed, in metres per second.
///
/// Terrain now spans tens of kilometres, so this is set to a cruising speed
/// rather than a walking one; anything slower makes crossing the map a chore.
const MOVE_SPEED: f32 = 200.0;

/// Factor applied to [`MOVE_SPEED`] while shift is held. Rotation is
/// deliberately left alone: the boost is for covering ground, and speeding the
/// mouse up at the same time would make the view impossible to aim.
///
/// Fast enough to reposition across a whole dataset in under a minute, which is
/// what you want when inspecting how the terrain resolves at different ranges.
const BOOST_FACTOR: f32 = 10.0;

/// Rotation per unit of raw mouse movement, in radians.
const LOOK_SENSITIVITY: f32 = 0.0025;

/// How close to straight up or down the camera may pitch, in radians.
///
/// Stopping just short keeps the forward vector from becoming parallel to the
/// world up axis, where the heading it implies is undefined and the view snaps
/// as it passes through.
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.001;

/// Longest frame the controller integrates over, in seconds.
///
/// A stalled frame -- a resize, a shader compile, the window being dragged --
/// would otherwise translate its whole wall-clock gap into one step and fling
/// the camera across the world.
const MAX_STEP: f32 = 0.1;

/// Which movement keys are currently down.
#[derive(Clone, Copy, Debug, Default)]
struct Held {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
}

/// Free-flying camera controls: WASD to move, drag with the left mouse button
/// to look, shift to move faster.
///
/// The controller owns the camera's orientation, holding heading and pitch as
/// angles and rebuilding the quaternion each frame rather than accumulating
/// incremental rotations onto it. Accumulating would let floating-point error
/// creep in as roll -- yaw and pitch do not commute, so repeated small
/// rotations about the moving axes tilt the horizon -- and there would be no
/// single value to clamp the pitch against.
#[derive(Clone, Copy, Debug, Default)]
pub struct FlyController {
    yaw: f32,
    pitch: f32,
    held: Held,
    boost: bool,
    looking: bool,
}

impl FlyController {
    /// Starts from the camera's current heading and pitch, so taking control
    /// does not swing the view.
    ///
    /// Any roll the camera carries is dropped, since the controller only ever
    /// produces level orientations.
    pub fn new(camera: &Camera) -> Self {
        // Inverse of `Camera::from_yaw_pitch_roll`, which builds the quaternion
        // as `Quat::from_euler(EulerRot::YXZ, -yaw, pitch, -roll)`.
        let (y, pitch, _roll) = camera.orientation.to_euler(EulerRot::YXZ);
        Self {
            yaw: -y,
            pitch: pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT),
            ..Self::default()
        }
    }

    /// Records a movement key going down or coming up. Other keys are ignored.
    ///
    /// Keys are identified by physical position rather than by the character
    /// they produce, so the WASD cluster stays under the same fingers on AZERTY
    /// and Dvorak layouts.
    pub fn key(&mut self, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::KeyW => self.held.forward = pressed,
            KeyCode::KeyS => self.held.back = pressed,
            KeyCode::KeyA => self.held.left = pressed,
            KeyCode::KeyD => self.held.right = pressed,
            _ => {}
        }
    }

    /// Sets whether the move speed is boosted, i.e. whether shift is held.
    pub fn set_boost(&mut self, boost: bool) {
        self.boost = boost;
    }

    /// Sets whether the look drag is active, i.e. whether the left mouse button
    /// is held.
    pub fn set_looking(&mut self, looking: bool) {
        self.looking = looking;
    }

    /// Applies raw mouse movement, in the device's own units, with +x right and
    /// +y down. Ignored unless the drag is active.
    pub fn mouse_motion(&mut self, dx: f32, dy: f32) {
        if !self.looking {
            return;
        }
        // Pushing the mouse away from you looks up, the unflipped convention.
        self.yaw = (self.yaw + dx * LOOK_SENSITIVITY).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch - dy * LOOK_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Forgets every held key and button.
    ///
    /// Call when the window loses focus: the key-up that ends a press goes to
    /// whichever window has focus when it happens, so without this a camera
    /// left moving during an alt-tab keeps moving forever.
    pub fn release_all(&mut self) {
        *self = Self {
            yaw: self.yaw,
            pitch: self.pitch,
            ..Self::default()
        };
    }

    /// Advances the camera by one frame of `dt` seconds.
    pub fn update(&self, camera: &mut Camera, dt: f32) {
        camera.orientation = Camera::from_yaw_pitch_roll(self.yaw, self.pitch, 0.0);

        // Movement is relative to where the camera points, including pitch, so
        // the view can be flown up and down without a separate altitude key.
        let mut direction = Vec3::ZERO;
        if self.held.forward {
            direction += Vec3::NEG_Z;
        }
        if self.held.back {
            direction += Vec3::Z;
        }
        if self.held.left {
            direction += Vec3::NEG_X;
        }
        if self.held.right {
            direction += Vec3::X;
        }
        if direction == Vec3::ZERO {
            return;
        }

        let speed = if self.boost {
            MOVE_SPEED * BOOST_FACTOR
        } else {
            MOVE_SPEED
        };
        // Normalizing keeps a diagonal from being faster than a straight line.
        camera.position += camera.orientation * direction.normalize() * speed * dt.min(MAX_STEP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// A frame long enough to measure but short enough to escape the stall clamp.
    const STEP: f32 = MAX_STEP;
    /// How far one [`STEP`] of unboosted movement covers.
    const DISTANCE: f32 = MOVE_SPEED * STEP;

    fn camera() -> Camera {
        Camera::new(Vec3::ZERO, Quat::IDENTITY, 16.0 / 9.0)
    }

    /// A controller with the look drag already active.
    fn dragging() -> FlyController {
        let mut controls = FlyController::default();
        controls.set_looking(true);
        controls
    }

    /// Where the camera ends up after one step of the given input.
    fn travel(controls: &FlyController) -> Vec3 {
        let mut camera = camera();
        controls.update(&mut camera, STEP);
        camera.position
    }

    #[test]
    fn w_and_s_move_along_the_view_axis() {
        let mut controls = FlyController::default();
        controls.key(KeyCode::KeyW, true);
        assert!(travel(&controls).abs_diff_eq(Vec3::new(0.0, 0.0, -DISTANCE), 1e-3));

        controls.key(KeyCode::KeyW, false);
        controls.key(KeyCode::KeyS, true);
        assert!(travel(&controls).abs_diff_eq(Vec3::new(0.0, 0.0, DISTANCE), 1e-3));
    }

    #[test]
    fn a_and_d_strafe() {
        let mut controls = FlyController::default();
        controls.key(KeyCode::KeyD, true);
        assert!(travel(&controls).abs_diff_eq(Vec3::new(DISTANCE, 0.0, 0.0), 1e-3));

        controls.key(KeyCode::KeyD, false);
        controls.key(KeyCode::KeyA, true);
        assert!(travel(&controls).abs_diff_eq(Vec3::new(-DISTANCE, 0.0, 0.0), 1e-3));
    }

    #[test]
    fn movement_follows_the_camera_after_looking_around() {
        let mut controls = dragging();
        // A quarter turn to the right, so forward becomes east.
        controls.mouse_motion(std::f32::consts::FRAC_PI_2 / LOOK_SENSITIVITY, 0.0);
        controls.key(KeyCode::KeyW, true);
        assert!(travel(&controls).abs_diff_eq(Vec3::new(DISTANCE, 0.0, 0.0), 1e-2));
    }

    #[test]
    fn a_diagonal_is_no_faster_than_a_straight_line() {
        let mut controls = FlyController::default();
        controls.key(KeyCode::KeyW, true);
        controls.key(KeyCode::KeyD, true);
        let distance = travel(&controls).length();
        assert!(
            (distance - DISTANCE).abs() < 1e-3,
            "expected {DISTANCE} m, travelled {distance} m"
        );
    }

    #[test]
    fn shift_speeds_up_movement_but_not_rotation() {
        let mut normal = dragging();
        normal.key(KeyCode::KeyW, true);
        let mut boosted = normal;
        boosted.set_boost(true);

        let (slow, fast) = (travel(&normal).length(), travel(&boosted).length());
        assert!(
            (fast - slow * BOOST_FACTOR).abs() < 1e-2,
            "boost should scale movement {BOOST_FACTOR}x: {slow} -> {fast}"
        );

        normal.mouse_motion(100.0, -60.0);
        boosted.mouse_motion(100.0, -60.0);
        assert_eq!(
            (boosted.yaw, boosted.pitch),
            (normal.yaw, normal.pitch),
            "boost must not change how far the mouse turns the camera"
        );
    }

    #[test]
    fn the_mouse_only_turns_the_camera_while_the_button_is_held() {
        let mut controls = FlyController::default();
        controls.mouse_motion(200.0, 200.0);
        assert_eq!((controls.yaw, controls.pitch), (0.0, 0.0));

        controls.set_looking(true);
        controls.mouse_motion(200.0, 0.0);
        assert!(controls.yaw > 0.0, "a held drag should turn the camera");

        let turned = controls.yaw;
        controls.set_looking(false);
        controls.mouse_motion(200.0, 0.0);
        assert_eq!(controls.yaw, turned, "releasing should stop the turning");
    }

    #[test]
    fn dragging_right_and_up_looks_right_and_up() {
        let mut controls = dragging();
        controls.mouse_motion(100.0, -100.0);

        let mut camera = camera();
        controls.update(&mut camera, STEP);
        let forward = camera.orientation * Vec3::NEG_Z;
        assert!(forward.x > 0.0, "should be looking east, got {forward}");
        assert!(forward.y > 0.0, "should be looking up, got {forward}");
    }

    #[test]
    fn looking_around_never_rolls_the_camera() {
        let mut controls = dragging();
        let mut camera = camera();
        // Sweep well past the poles and right around the compass.
        for _ in 0..50 {
            controls.mouse_motion(137.0, -211.0);
            controls.update(&mut camera, STEP);
            let right = camera.orientation * Vec3::X;
            assert!(
                right.y.abs() < 1e-5,
                "horizon should stay level, right vector is {right}"
            );
        }
    }

    #[test]
    fn pitch_stops_short_of_vertical() {
        let mut controls = dragging();
        controls.mouse_motion(0.0, -1e6);
        assert!(controls.pitch < std::f32::consts::FRAC_PI_2);
        assert!((controls.pitch - PITCH_LIMIT).abs() < 1e-6);

        controls.mouse_motion(0.0, 2e6);
        assert!((controls.pitch + PITCH_LIMIT).abs() < 1e-6);
    }

    #[test]
    fn a_stalled_frame_does_not_fling_the_camera() {
        let mut controls = FlyController::default();
        controls.key(KeyCode::KeyW, true);

        let mut camera = camera();
        controls.update(&mut camera, 30.0);
        assert!(
            camera.position.length() <= MOVE_SPEED * MAX_STEP + 1e-3,
            "a 30 s gap should clamp to one {MAX_STEP} s step, moved {}",
            camera.position.length()
        );
    }

    #[test]
    fn losing_focus_releases_held_input() {
        let mut controls = dragging();
        controls.key(KeyCode::KeyW, true);
        controls.set_boost(true);
        controls.mouse_motion(400.0, 0.0);
        let aim = (controls.yaw, controls.pitch);

        controls.release_all();

        assert_eq!(travel(&controls), Vec3::ZERO, "keys should be released");
        assert_eq!((controls.yaw, controls.pitch), aim, "the view should hold");
        controls.mouse_motion(400.0, 0.0);
        assert_eq!((controls.yaw, controls.pitch), aim, "the drag should end");
    }

    #[test]
    fn taking_over_a_camera_keeps_it_aimed_where_it_was() {
        let mut camera = camera();
        camera.orientation =
            Camera::from_yaw_pitch_roll(200f32.to_radians(), -12f32.to_radians(), 0.0);
        let before = camera.orientation * Vec3::NEG_Z;

        let controls = FlyController::new(&camera);
        controls.update(&mut camera, STEP);

        let after = camera.orientation * Vec3::NEG_Z;
        assert!(
            after.abs_diff_eq(before, 1e-5),
            "view swung on takeover: {before} -> {after}"
        );
    }
}
