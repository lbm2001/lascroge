# mock_pyrobotdesign.py
import numpy as np
from enum import Enum

class JointType(Enum):
    NONE = 0
    FIXED = 1
    HINGE = 2
    FREE = 3

class LinkShape(Enum):
    NONE = 0
    CAPSULE = 1

class JointControlMode(Enum):
    POSITION = 0

class Quaterniond:
    def __init__(self, w, x, y, z):
        self.w, self.x, self.y, self.z = w, x, y, z

class Link:
    def __init__(self, parent, joint_type, joint_pos, joint_rot, joint_axis,
                 shape, length, radius, density, friction,
                 joint_kp, joint_kd, joint_torque, joint_control_mode,
                 color, joint_color, label, joint_label):
        self.parent = parent
        self.joint_type = joint_type
        self.joint_pos = joint_pos
        self.joint_rot = joint_rot
        self.joint_axis = joint_axis
        self.shape = shape
        self.length = length
        self.radius = radius
        self.density = density
        self.friction = friction
        self.joint_kp = joint_kp
        self.joint_kd = joint_kd
        self.joint_torque = joint_torque
        self.joint_control_mode = joint_control_mode
        self.color = color
        self.joint_color = joint_color
        self.label = label
        self.joint_label = joint_label

class Robot:
    def __init__(self):
        self.links = []