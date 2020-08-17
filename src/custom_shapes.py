from pyrep.objects import Shape, Dummy, Object
from pyrep.robots.arms.arm import Arm
from pyrep.const import ObjectType


class Head(Dummy):
    def __init__(self, name_or_handle):
        super().__init__(name_or_handle)
        joints = self.get_objects_in_tree(
            object_type=ObjectType.JOINT
        )
        print(joints)
        self.left_pan_joint = next(
            s for s in proximity_sensors
            if s.get_name().startswith("???")
        )
        self.left_tilt_joint = next(
            s for s in proximity_sensors
            if s.get_name().startswith("???")
        )
        self.right_pan_joint = next(
            s for s in proximity_sensors
            if s.get_name().startswith("???")
        )
        self.right_tilt_joint = next(
            s for s in proximity_sensors
            if s.get_name().startswith("???")
        )

    def get_eye_position(self, eye):
        pass

    def get_eye_orientation(self, eye):
        pass


class Screen(object):
    pass
