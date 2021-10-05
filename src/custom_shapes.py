from pyrep.objects import Shape, Dummy, Object
from pyrep.robots.arms.arm import Arm
from pyrep.const import ObjectType, TextureMappingMode
import numpy as np


deg = np.rad2deg
rad = np.deg2rad


class Head(Shape):
    def __init__(self, name_or_handle):
        super().__init__(name_or_handle)
        joints = self.get_objects_in_tree(
            object_type=ObjectType.JOINT
        )
        shapes = self.get_objects_in_tree(
            object_type=ObjectType.SHAPE
        )
        self.left_pan_joint = next(
            s for s in joints
            if s.get_name().startswith("vs_eye_pan_left")
        )
        self.left_tilt_joint = next(
            s for s in joints
            if s.get_name().startswith("vs_eye_tilt_left")
        )
        self.left_cyclo_joint = next(
            s for s in joints
            if s.get_name().startswith("vs_eye_cyclo_left")
        )
        self.eyeball_left = next(
            s for s in shapes
            if s.get_name().startswith("eyeball_left")
        )
        self.right_pan_joint = next(
            s for s in joints
            if s.get_name().startswith("vs_eye_pan_right")
        )
        self.right_tilt_joint = next(
            s for s in joints
            if s.get_name().startswith("vs_eye_tilt_right")
        )
        self.right_cyclo_joint = next(
            s for s in joints
            if s.get_name().startswith("vs_eye_cyclo_right")
        )
        self.eyeball_right = next(
            s for s in shapes
            if s.get_name().startswith("eyeball_right")
        )
        self._tilt_position = 0.0
        self._pan_position = 0.0
        self._vergence_position = 0.0
        self._cyclo_position = 0.0
        self._tilt_velocity = 0.0
        self._pan_velocity = 0.0
        self._vergence_velocity = 0.0
        self._cyclo_velocity = 0.0
        self._tilt_acceleration = 0.0
        self._pan_acceleration = 0.0
        self._vergence_acceleration = 0.0
        self._cyclo_acceleration = 0.0

    def get_eye_position(self, eye):
        if eye == 'left':
            return self.left_pan_joint.get_position()
        elif eye == 'right':
            return self.right_pan_joint.get_position()
        else:
            raise ValueError("Incorrect eye name {} must be either left or right".format(eye))

    def get_eye_orientation(self, eye):
        return [-np.pi / 2, 0.0, np.pi]
        # if eye == 'left':
        #     return self.eyeball_left.get_orientation() + np.array([np.pi / 2, 0, 0])
        # elif eye == 'right':
        #     return self.eyeball_right.get_orientation() + np.array([np.pi / 2, 0, 0])
        # else:
        #     raise ValueError("Incorrect eye name {} must be either left or right".format(eye))

    def get_eye_parent(self, eye):
        if eye == 'left':
            return self.eyeball_left
        elif eye == 'right':
            return self.eyeball_right
        else:
            raise ValueError("Incorrect eye name {} must be either left or right".format(eye))

    def get_joints_positions(self):
        return np.array([
            self._tilt_position,
            self._pan_position,
            self._vergence_position,
            self._cyclo_position,
        ])

    def get_joints_velocities(self):
        return np.array([
            self._tilt_velocity,
            self._pan_velocity,
            self._vergence_velocity,
            self._cyclo_velocity,
        ])

    def set_joints_positions(self, tilt, pan, vergence, cyclo):
        self._tilt_position = tilt
        self._pan_position = pan
        self._vergence_position = vergence
        self._cyclo_position = cyclo
        self._tilt_velocity = 0
        self._pan_velocity = 0
        self._vergence_velocity = 0
        self._cyclo_velocity = 0
        tilt, pan, vergence, cyclo = rad(tilt), rad(pan), rad(vergence), rad(cyclo)
        self.left_pan_joint.set_joint_position(pan + vergence / 2)
        self.left_tilt_joint.set_joint_position(tilt)
        self.left_cyclo_joint.set_joint_position(-cyclo / 2)
        self.right_pan_joint.set_joint_position(pan - vergence / 2)
        self.right_tilt_joint.set_joint_position(tilt)
        self.right_cyclo_joint.set_joint_position(cyclo / 2)

    def set_joints_velocities(self, tilt, pan, vergence, cyclo):
        self._tilt_position += tilt
        self._pan_position += pan
        self._vergence_position += vergence
        self._cyclo_position += cyclo
        self._tilt_velocity = tilt
        self._pan_velocity = pan
        self._vergence_velocity = vergence
        self._cyclo_velocity = cyclo
        tilt, pan, vergence, cyclo = rad(self._tilt_position), rad(self._pan_position), rad(self._vergence_position), rad(self._cyclo_position)
        self.left_pan_joint.set_joint_position(pan + vergence / 2)
        self.left_tilt_joint.set_joint_position(tilt)
        self.left_cyclo_joint.set_joint_position(-cyclo / 2)
        self.right_pan_joint.set_joint_position(pan - vergence / 2)
        self.right_tilt_joint.set_joint_position(tilt)
        self.right_cyclo_joint.set_joint_position(cyclo / 2)

    def set_action(self, tilt_acceleration, pan_acceleration, vergence_velocity, cyclo_velocity):
        self._tilt_acceleration = tilt_acceleration
        self._pan_acceleration = pan_acceleration
        self._tilt_velocity += tilt_acceleration
        self._pan_velocity += pan_acceleration
        self.set_joints_velocities(
            self._tilt_velocity,
            self._pan_velocity,
            vergence_velocity,
            cyclo_velocity,
        )

    def reset(self, tilt=0.0, pan=0.0, vergence=0.0, cyclo=0.0):
        self.set_joints_positions(tilt, pan, vergence, cyclo)

    def set_control_loop_enabled(self, bool):
        self.left_pan_joint.set_control_loop_enabled(bool)
        self.left_tilt_joint.set_control_loop_enabled(bool)
        self.right_pan_joint.set_control_loop_enabled(bool)
        self.right_tilt_joint.set_control_loop_enabled(bool)

    def set_motor_enabled(self, bool):
        self.left_pan_joint.set_motor_enabled(bool)
        self.left_tilt_joint.set_motor_enabled(bool)
        self.right_pan_joint.set_motor_enabled(bool)
        self.right_tilt_joint.set_motor_enabled(bool)

    def set_motor_locked_at_zero_velocity(self, bool):
        self.left_pan_joint.set_motor_locked_at_zero_velocity(bool)
        self.left_tilt_joint.set_motor_locked_at_zero_velocity(bool)
        self.right_pan_joint.set_motor_locked_at_zero_velocity(bool)
        self.right_tilt_joint.set_motor_locked_at_zero_velocity(bool)


class Screen(Shape):
    def __init__(self, textures_list, size=1.5):
        vertices = np.array([
            [-1.0, -1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [-1.0,  1.0,  1.0],
            [-1.0, -1.0,  1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0],
            [ 1.0,  1.0, -1.0],
            [ 1.0, -1.0, -1.0],
        ]) * np.array([[size / 2, 0.0025, size / 2]])
        indices = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [2, 3, 4],
            [2, 4, 5],
            [1, 2, 5],
            [1, 5, 6],
            [4, 6, 7],
            [4, 5, 6],
            [0, 3, 4],
            [0, 4, 7],
            [0, 1, 6],
            [0, 6, 7],
        ])
        mesh = Shape.create_mesh(list(vertices.flatten()), list(indices.flatten()))
        super().__init__(mesh.get_name())
        self._handle = self.get_handle()
        self.size = size
        self.set_position((0, 1, 0))

    def set_texture(self, index=None):
        if index is None:
            index = np.random.randint(len(self.textures_list))
        super().set_texture(
            self.textures_list[index],
            TextureMappingMode.CUBE,
            interpolate=False,
            uv_scaling=[self.size, self.size],
        )


class UniformMotionScreen(Screen):
    def __init__(self, textures_list, size=1.5,
            min_distance=0.5, max_distance=5.0,
            max_depth_speed=0.03, max_speed_in_deg=1.125):
        super().__init__(textures_list, size=size)
        self.textures_list = textures_list
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_depth_speed = max_depth_speed
        self.max_speed_in_rad = rad(max_speed_in_deg)
        self.episode_reset()

    def episode_reset(self, start_distance=None, depth_speed=None,
            angular_speed=None, direction=None, texture_id=None, preinit=False):
        if start_distance is None:
            self.start_distance = np.random.uniform(self.min_distance, self.max_distance)
        else:
            self.start_distance = start_distance
        self.distance = self.start_distance
        if depth_speed is None:
            self.depth_speed = np.random.uniform(-self.max_depth_speed, self.max_depth_speed)
        else:
            self.depth_speed = depth_speed
        if angular_speed is None:
            self.angular_speed = np.random.uniform(0.0, self.max_speed_in_rad)
        else:
            self.angular_speed = rad(angular_speed)
        if direction is None:
            self.direction = np.random.uniform(0, 2 * np.pi)
        else:
            self.direction = direction
        self.set_texture(texture_id)
        self.set_episode_iteration(-1 if preinit else 0)

    def set_episode_iteration(self, it):
        self._episode_iteration = it
        self.set_position(self.position)

    def move(self):
        self.set_episode_iteration(self._episode_iteration + 1)

    def _get_position(self):
        cos_speed = np.cos(self._episode_iteration * self.angular_speed)
        sin_speed = np.sin(self._episode_iteration * self.angular_speed)
        cos_dir = np.cos(self.direction)
        sin_dir = np.sin(self.direction)
        self.distance = self.start_distance + (self._episode_iteration * self.depth_speed)
        x = self.distance * sin_speed * cos_dir
        y = self.distance * cos_speed
        z = self.distance * sin_speed * sin_dir
        return np.array([x, y, z])

    def _get_orientation(self):
        tilt_speed, pan_speed = self.tilt_pan_speed
        tilt, pan = tilt_speed * self._episode_iteration, pan_speed * self._episode_iteration
        x = rad(90 + tilt)
        y = rad(-pan)
        z = 0
        return np.array([x, y, z])

    def _get_tilt_pan_speed(self):
        return np.array([
            deg(self.angular_speed * np.sin(self.direction)),
            deg(self.angular_speed * np.cos(self.direction))
        ])

    orientation = property(_get_orientation)
    position = property(_get_position)
    tilt_pan_speed = property(_get_tilt_pan_speed)
