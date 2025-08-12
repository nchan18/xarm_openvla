# <--- Keep your imports as-is --->
import argparse
import datetime
import pygame
import math
import threading
import time
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import h5py
from scipy.spatial.transform import Rotation as R
from oculus_reader.reader import OculusReader
from abc import ABC, abstractmethod
from rlcpy.Node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from scipy.spatial.transform import Rotation as R
import multiprocessing
import subprocess
import threading


def run_terminal_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, executable="/bin/bash", encoding="utf8"
    )

    return process


def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()

    return thread


def run_multiprocessed_command(command, args=()):
    process = multiprocessing.Process(target=command, args=args)
    process.start()

    return process


### Conversions ###
def quat_to_euler(quat, degrees=False):
    euler = R.from_quat(quat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_quat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_quat()


def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rmat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


def rmat_to_quat(rot_mat, degrees=False):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat


def quat_to_rmat(quat, degrees=False):
    return R.from_quat(quat, degrees=degrees).as_matrix()


### Subtractions ###
def quat_diff(target, source):
    result = R.from_quat(target) * R.from_quat(source).inv()
    return result.as_quat()


def angle_diff(target, source, degrees=False):
    target_rot = R.from_euler("xyz", target, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    result = target_rot * source_rot.inv()
    return result.as_euler("xyz")


def pose_diff(target, source, degrees=False):
    lin_diff = np.array(target[:3]) - np.array(source[:3])
    rot_diff = angle_diff(target[3:6], source[3:6], degrees=degrees)
    result = np.concatenate([lin_diff, rot_diff])
    return result


### Additions ###
def add_quats(delta, source):
    result = R.from_quat(delta) * R.from_quat(source)
    return result.as_quat()


def add_angles(delta, source, degrees=False):
    delta_rot = R.from_euler("xyz", delta, degrees=degrees)
    source_rot = R.from_euler("xyz", source, degrees=degrees)
    new_rot = delta_rot * source_rot
    return new_rot.as_euler("xyz", degrees=degrees)


def add_poses(delta, source, degrees=False):
    lin_sum = np.array(delta[:3]) + np.array(source[:3])
    rot_sum = add_angles(delta[3:6], source[3:6], degrees=degrees)
    result = np.concatenate([lin_sum, rot_sum])
    return result


### MISC ###
def change_pose_frame(pose, frame, degrees=False):
    R_frame = euler_to_rmat(frame[3:6], degrees=degrees)
    R_pose = euler_to_rmat(pose[3:6], degrees=degrees)
    t_frame, t_pose = frame[:3], pose[:3]
    euler_new = rmat_to_euler(R_frame @ R_pose, degrees=degrees)
    t_new = R_frame @ t_pose + t_frame
    result = np.concatenate([t_new, euler_new])
    return result

# Helper Functions
def euler_to_quat(euler_angles):
    return R.from_euler('xyz', euler_angles, degrees=False).as_quat()

def run_threaded_command(target_fn):
    thread = threading.Thread(target=target_fn, daemon=True)
    thread.start()

def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X

# Robot Arm Environment
class xArm7GripperEnv:
    def __init__(self, robot_ip="192.168.1.198", arm_speed=1000, gripper_speed=500, task_name="default_task",
                 max_lin_speed_mm=50, max_rot_speed_deg=5, workspace_half_extent_mm=250):
        """
        max_lin_speed_mm: maximum linear speed (mm/s) applied to vc_set_cartesian_velocity
        max_rot_speed_deg: maximum rotational speed (deg/s)
        workspace_half_extent_mm: half the side length of cube workspace in mm (0.25m -> 250 mm)
        """
        self.robot_ip = robot_ip
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.task_name = task_name
        self.arm_pos = None
        self.arm_rot = None
        self.previous_arm_pos = None
        self.gripper_pos = None
        self.wait = False

        # Limits & workspace
        self.max_lin_speed_mm = max_lin_speed_mm
        self.max_rot_speed_deg = max_rot_speed_deg
        self.workspace_half_extent_mm = workspace_half_extent_mm

        self.arm = XArmAPI(self.robot_ip)
        # Set a sensible starting pose (x,y,z,roll,pitch,yaw) in mm/degrees
        self.arm_starting_pose = (300, 0, 300, 180, 0, 0)
        tcp_offset = [0, 0, 100, 0, 0, 0]
        self.arm.set_tcp_offset(tcp_offset)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(self.gripper_speed)

        # Workspace center is the starting pose position
        self.workspace_center = np.array(self.arm_starting_pose[:3], dtype=float)
        self.workspace_min = self.workspace_center - self.workspace_half_extent_mm
        self.workspace_max = self.workspace_center + self.workspace_half_extent_mm

        self.move_to_starting_position()
        self.update_arm_state()

    def move_to_starting_position(self):
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.move_position(self.arm_starting_pose)
        self.gripper_open()
        self.arm.set_mode(5)
        self.arm.set_state(0)

    def update_arm_state(self):
        _, arm_pos = self.arm.get_position(is_radian=False)
        self.previous_arm_pos = self.arm_pos
        self.arm_pos = tuple(arm_pos[:3])  # mm
        self.arm_rot = tuple(arm_pos[3:])  # deg
        _, gripper_pos = self.arm.get_gripper_position()
        self.gripper_pos = gripper_pos

    def _clamp_to_workspace(self, pos_mm):
        """Clamp an absolute position (x,y,z in mm) to the defined workspace box."""
        pos = np.array(pos_mm, dtype=float)
        clamped = np.minimum(np.maximum(pos, self.workspace_min), self.workspace_max)
        return tuple(clamped.tolist())

    def move_position(self, action):
        """
        action: absolute pose (x,y,z,roll,pitch,yaw) in mm and deg
        Clamp to workspace before issuing.
        """
        # Clamp linear components
        x, y, z = action[0], action[1], action[2]
        x_c, y_c, z_c = self._clamp_to_workspace((x, y, z))
        clamped_action = (x_c, y_c, z_c, action[3], action[4], action[5])
        self.arm.set_position(x=clamped_action[0], y=clamped_action[1], z=clamped_action[2],
                              roll=clamped_action[3], pitch=clamped_action[4], yaw=clamped_action[5],
                              relative=False, wait=True, speed=100)

    def move(self, action, dt=0.05):
        """
        action: list or array length 6 [vx(mm/s), vy(mm/s), vz(mm/s), vroll(deg/s), vpitch(deg/s), vyaw(deg/s)]
        This function will clip velocities to max limits and zero components that would push the
        arm outside the workspace (taking dt into account).
        """

        print(action)
        if np.allclose(action, [0, 0, 0, 0, 0, 0]):
            return

        # Clip linear velocities and rotational velocities
        lin = np.array(action[:3], dtype=float)
        rot = np.array(action[3:], dtype=float)

        # Clip magnitudes per axis by the signed max (keep sign)
        lin = np.clip(lin, -self.max_lin_speed_mm, self.max_lin_speed_mm)
        rot = np.clip(rot, -self.max_rot_speed_deg, self.max_rot_speed_deg)

        # Predict next absolute position (approx) and zero components that would leave workspace
        if self.arm_pos is None:
            self.update_arm_state()
        current_pos = np.array(self.arm_pos, dtype=float)
        pred_pos = current_pos + lin * dt  # mm

        # For each axis, if predicted pos outside workspace, zero that axis velocity
        for i in range(3):
            if pred_pos[i] < self.workspace_min[i] or pred_pos[i] > self.workspace_max[i]:
                lin[i] = 0.0

        safe_action = np.concatenate([lin, rot]).tolist()

        if any(abs(v) > 1e-6 for v in safe_action):
            print(f"[xArm7GripperEnv] applying safe velocity (mm/s,deg/s): {safe_action}")
            self.arm.vc_set_cartesian_velocity(safe_action)

    def gripper_open(self):
        self.gripper_state = 1
        self.arm.set_gripper_position(850, wait=self.wait)

    def gripper_close(self):
        self.gripper_state = 0
        self.arm.set_gripper_position(0, wait=self.wait)


class SimController(BaseController):
    """
    Dummy controller for simulation.  It ignores all hardware inputs and simply
    forwards whatever velocity it receives from the real controller to a ROS2
    topic.
    """

    def __init__(self, ros_topic="/xarm/velocity_cmd", node_name="xarm_sim_controller"):
        if rclpy is None:
            raise RuntimeError("rclpy not available – cannot run in sim mode.")
        # Spin up the ROS2 node in a background thread so it does not block
        self._ros_node = _SimRosNode(ros_topic, node_name)
        threading.Thread(target=self._ros_node.spin, daemon=True).start()

    def get_action(self, state_dict):
        """
        In simulation mode we don't need to read any hardware – the real
        controller (PlayStation or VR) already produced a velocity vector.
        We simply return that same vector so that ControlSystem can publish it.
        """
        # The base class expects us to return (velocity, gripper_command).
        # We'll just forward whatever the underlying controller generated.
        return state_dict["last_velocity"], state_dict.get("gripper_cmd", "none")


class _SimRosNode(Node):
    def __init__(self, topic_name, node_name):
        super().__init__(node_name)
        self.publisher_ = self.create_publisher(Float64MultiArray, topic_name, 10)
        self.latest_cmd = None
        # Spin in a separate thread so the node can be stopped cleanly.
        self._stop_event = threading.Event()

    def spin(self):
        """Spin loop that publishes every 50 ms."""
        while not rclpy.ok() or not self._stop_event.is_set():
            if self.latest_cmd is not None:
                msg = Float64MultiArray(data=self.latest_cmd)
                self.publisher_.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)

    def stop(self):
        self._stop_event.set()
        rclpy.shutdown()

# VR Policy for VR Controller
class VRPolicy:
    def __init__(
        self,
        right_controller: bool = True,
        max_lin_vel: float = 1,
        max_rot_vel: float = 1,
        max_gripper_vel: float = 1,
        spatial_coeff: float = 1,
        pos_action_gain: float = 5,
        rot_action_gain: float = 2,
        gripper_action_gain: float = 3,
        rmat_reorder: list = [-2, -1, -3, 4],
    ):
        self.oculus_reader = OculusReader()
        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self.reset_orientation = True
        self.reset_state()

        # Start State Listening Thread #
        run_threaded_command(self._update_internal_state)

    def reset_state(self):
        self._state = {
            "poses": {},
            "buttons": {"A": False, "B": False, "X": False, "Y": False},
            "movement_enabled": False,
            "controller_on": True,
        }
        self.update_sensor = True
        self.reset_origin = True
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state = None

    def _update_internal_state(self, num_wait_sec=5, hz=50):
        last_read_time = time.time()
        while True:
            # Regulate Read Frequency #
            time.sleep(1 / hz)

            # Read Controller
            time_since_read = time.time() - last_read_time
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            self._state["controller_on"] = time_since_read < num_wait_sec
            if poses == {}:
                continue

            # Determine Control Pipeline #
            toggled = self._state["movement_enabled"] != buttons[self.controller_id.upper() + "G"]
            self.update_sensor = self.update_sensor or buttons[self.controller_id.upper() + "G"]
            self.reset_orientation = self.reset_orientation or buttons[self.controller_id.upper() + "J"]
            self.reset_origin = self.reset_origin or toggled

            # Save Info #
            self._state["poses"] = poses
            self._state["buttons"] = buttons
            self._state["movement_enabled"] = buttons[self.controller_id.upper() + "G"]
            self._state["controller_on"] = True
            last_read_time = time.time()

            # Update Definition Of "Forward" #
            stop_updating = self._state["buttons"][self.controller_id.upper() + "J"] or self._state["movement_enabled"]
            if self.reset_orientation:
                rot_mat = np.asarray(self._state["poses"][self.controller_id])
                if stop_updating:
                    self.reset_orientation = False
                # try to invert the rotation matrix, if not possible, then just use the identity matrix
                try:
                    rot_mat = np.linalg.inv(rot_mat)
                except:
                    print(f"exception for rot mat: {rot_mat}")
                    rot_mat = np.eye(4)
                    self.reset_orientation = True
                self.vr_to_global_mat = rot_mat

    def _process_reading(self):
        rot_mat = np.asarray(self._state["poses"][self.controller_id])
        rot_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat
        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        vr_gripper = self._state["buttons"]["rightTrig" if self.controller_id == "r" else "leftTrig"][0]

        self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": vr_gripper}

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        """Scales down the linear and angular magnitudes of the action"""
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm(gripper_vel)
        if lin_vel_norm > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def _calculate_action(self, state_dict, include_info=False):
        # Read Sensor #
        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False

        # Read Observation
        robot_pos = np.array(state_dict["cartesian_position"][:3])
        robot_euler = state_dict["cartesian_position"][3:]
        robot_quat = euler_to_quat(robot_euler)
        robot_gripper = state_dict["gripper_position"]

        # Reset Origin On Release #
        if self.reset_origin:
            self.robot_origin = {"pos": robot_pos, "quat": robot_quat}
            self.vr_origin = {"pos": self.vr_state["pos"], "quat": self.vr_state["quat"]}
            self.reset_origin = False

        # Calculate Positional Action #
        robot_pos_offset = robot_pos - self.robot_origin["pos"]
        target_pos_offset = self.vr_state["pos"] - self.vr_origin["pos"]
        pos_action = target_pos_offset - robot_pos_offset

        # Calculate Euler Action #
        robot_quat_offset = quat_diff(robot_quat, self.robot_origin["quat"])
        target_quat_offset = quat_diff(self.vr_state["quat"], self.vr_origin["quat"])
        quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        euler_action = quat_to_euler(quat_action)

        # Calculate Gripper Action #
        gripper_action = (self.vr_state["gripper"] * 1.5) - robot_gripper

        # Calculate Desired Pose #
        target_pos = pos_action + robot_pos
        target_euler = add_angles(euler_action, robot_euler)
        target_cartesian = np.concatenate([target_pos, target_euler])
        target_gripper = self.vr_state["gripper"]

        # Scale Appropriately #
        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        gripper_action *= self.gripper_action_gain
        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action, gripper_action)

        # Prepare Return Values #
        info_dict = {"target_cartesian_position": target_cartesian, "target_gripper_position": target_gripper}
        action = np.concatenate([lin_vel, rot_vel, [gripper_vel]])
        action = action.clip(-1, 1)

        # Return #
        if include_info:
            return action, info_dict
        else:
            return action

    def get_info(self):
        return {
            "success": self._state["buttons"]["A"] if self.controller_id == 'r' else self._state["buttons"]["X"],
            "failure": self._state["buttons"]["B"] if self.controller_id == 'r' else self._state["buttons"]["Y"],
            "movement_enabled": self._state["movement_enabled"],
            "controller_on": self._state["controller_on"],
        }

    def forward(self, obs_dict, include_info=False):
        if self._state["poses"] == {}:
            action = np.zeros(7)
            if include_info:
                return action, {}
            else:
                return action
        return self._calculate_action(obs_dict["robot_state"], include_info=include_info)

# Controller Base Class
class BaseController(ABC):
    @abstractmethod
    def get_action(self, state_dict):
        pass

# PlayStation Controller (unchanged except clearer units comment)
class PlayStationController(BaseController):
    def __init__(self, pos_step_size=50, rot_step_size=15, threshold=0.2):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.pos_step_size = pos_step_size  # interpreted as mm/s
        self.rot_step_size = rot_step_size  # deg/s
        self.threshold = threshold
        self.gripper_state = 1  # Assume open initially

    def get_action(self, state_dict):
        pygame.event.pump()
        LeftJoystickY = self.joystick.get_axis(1)
        LeftJoystickX = self.joystick.get_axis(0)
        RightJoystickY = self.joystick.get_axis(4)
        DPadX, DPadY = self.joystick.get_hat(0)
        LeftTrigger = self.joystick.get_axis(2)
        RightTrigger = self.joystick.get_axis(5)
        X = self.joystick.get_button(0)  # Open gripper
        SQUARE = self.joystick.get_button(3)  # Close gripper

        velocity = [0.0] * 6
        # Using joysticks to command velocities in mm/s & deg/s
        if abs(LeftJoystickY) > self.threshold:
            velocity[0] = (-LeftJoystickY) * self.pos_step_size  # invert sign to make natural forward
        if abs(LeftJoystickX) > self.threshold:
            velocity[1] = (LeftJoystickX) * self.pos_step_size
        if abs(RightJoystickY) > self.threshold:
            velocity[2] = (-RightJoystickY) * self.pos_step_size
        if DPadX != 0:
            velocity[3] = -self.rot_step_size if DPadX > 0 else self.rot_step_size
        if DPadY != 0:
            velocity[4] = -self.rot_step_size if DPadY > 0 else self.rot_step_size
        if LeftTrigger > 0.5:
            velocity[5] = self.rot_step_size
        if RightTrigger > 0.5:
            velocity[5] = -self.rot_step_size

        gripper_command = 'none'
        if X:
            gripper_command = 'open'
            self.gripper_state = 1
        elif SQUARE:
            gripper_command = 'close'
            self.gripper_state = 0

        return velocity, gripper_command

# VR Controller wrapper: converts VRPolicy outputs -> robot-friendly units (mm/s, deg/s)
class VRController(BaseController):
    def __init__(self):
        self.vr_policy = VRPolicy()
        # Note: VRPolicy._calculate_action returns lin_vel in m/s; convert below.

    def get_action(self, state_dict):
        lin_vel_m, rot_vel, gripper_command = self.vr_policy._calculate_action(state_dict)
        # Convert meters/s -> millimeters/s for robot API
        lin_vel_mm = lin_vel_m * 1000.0
        # rot_vel assumed to be in deg/s already by policy; if radians, convert accordingly
        velocity = np.concatenate([lin_vel_mm, rot_vel])
        return velocity.tolist(), gripper_command

class RealSenseRecorder:
    def __init__(self, device_serials, camera_names, frame_size=(640, 480), base_save_dir="."):
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.image_dirs = [os.path.join(base_save_dir, f"images_{name}_{self.time_stamp}") for name in camera_names]
        self.image_paths = [[] for _ in camera_names]
        self.frame_idx = 0
        self.frame_size = frame_size
        self.camera_names = camera_names
        self.pipelines = []

        for dir in self.image_dirs:
            os.makedirs(dir, exist_ok=True)

        for serial in device_serials:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, 30)
            config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, 30)
            try:
                pipeline.start(config)
                time.sleep(1)  # Wait for camera to stabilize
                for _ in range(30):
                    pipeline.wait_for_frames()
                self.pipelines.append(pipeline)
            except Exception as e:
                print(f"Failed to initialize RealSense camera {serial}: {e}")

    def record_frame(self, action_idx=None):
        for i, pipeline in enumerate(self.pipelines):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                print(f"Warning: Failed to get frames from RealSense camera {self.camera_names[i]}")
                continue

            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Skip if frame is too dark (likely blank or corrupted)
            if color_image.mean() < 10:
                print(f"Skipping frame {self.frame_idx} from camera {self.camera_names[i]} due to low brightness")
                continue

            depth_image = np.asanyarray(depth_frame.get_data())

            resized_color_image = cv2.resize(color_image, self.frame_size)
            if action_idx is not None:
                resized_color_image = cv2.putText(resized_color_image, str(action_idx), (10, 40),
                                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            color_path = os.path.join(self.image_dirs[i], f"color_frame_{self.frame_idx:05d}.jpg")
            depth_path = os.path.join(self.image_dirs[i], f"depth_frame_{self.frame_idx:05d}.png")

            cv2.imwrite(color_path, resized_color_image)
            cv2.imwrite(depth_path, depth_image)

            self.image_paths[i].append({"color": color_path, "depth": depth_path})

        self.frame_idx += 1

    def stop(self):
        for pipeline in self.pipelines:
            pipeline.stop()

# Webcam Recorder
class WebcamRecorder:
    def __init__(self, device_ids, camera_names, frame_size=(640, 480), base_save_dir="."):
        self.cap = [cv2.VideoCapture(cam) for cam in device_ids]
        self.frame_size = frame_size
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")

        self.image_dirs = [os.path.join(base_save_dir, f"images_{name}_{self.time_stamp}") for name in camera_names]
        self.image_paths = [[] for _ in camera_names]
        self.frame_idx = 0
        self.camera_names = camera_names

        for dir in self.image_dirs:
            os.makedirs(dir, exist_ok=True)

    def record_frame(self, action_idx=None):
        for i, cap in enumerate(self.cap):
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"Warning: Failed to read from webcam {self.camera_names[i]}")
                continue
            
            if frame.mean() < 10:
                print(f"Skipping frame {self.frame_idx} from webcam {self.camera_names[i]} due to low brightness")
                continue

            resized = cv2.resize(frame, self.frame_size)
            if action_idx is not None:
                resized = cv2.putText(resized, str(action_idx), (10, 40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            color_path = os.path.join(self.image_dirs[i], f"color_frame_{self.frame_idx:05d}.jpg")
            cv2.imwrite(color_path, resized)

            self.image_paths[i].append({"color": color_path, "depth": None})

        self.frame_idx += 1

    def stop(self):
        for cap in self.cap:
            cap.release()

# Camera Manager
class CameraManager:
    def __init__(self, realsense_ids, webcam_ids, realsense_names, webcam_names, frame_size=(640, 480), base_save_dir="."):
        self.recorders = []
        self.image_paths = []
        self.camera_names = realsense_names + webcam_names

        if realsense_ids:
            rs_recorder = RealSenseRecorder(realsense_ids, realsense_names, frame_size, base_save_dir=base_save_dir)
            self.recorders.append(rs_recorder)
            self.image_paths.extend(rs_recorder.image_paths)

        if webcam_ids:
            wc_recorder = WebcamRecorder(webcam_ids, webcam_names, frame_size, base_save_dir=base_save_dir)
            self.recorders.append(wc_recorder)
            self.image_paths.extend(wc_recorder.image_paths)

        if not self.recorders:
            raise ValueError("No cameras specified")

    def record_frame(self, action_idx=None):
        for recorder in self.recorders:
            recorder.record_frame(action_idx=action_idx)

    def stop(self):
        for recorder in self.recorders:
            recorder.stop()

# HDF5 Recorder
class HDF5Recorder:
    def __init__(self, save_dir, task_name, camera_names):
        self.task_name = task_name.replace(" ", "_")
        self.save_dir = save_dir
        self.camera_names = camera_names
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.save_dir, f"{self.task_name}_{timestamp}.hdf5")
        self.h5file = h5py.File(self.filename, "w")
        self.action_data = []
        self.timestamps = []
        self.image_paths = []

    def record_step(self, action, image_paths):
        self.action_data.append(np.array(action, dtype=np.float32))
        self.timestamps.append(time.time())
        self.image_paths.append(image_paths)

    def save(self):
        self.h5file.create_dataset("actions", data=np.array(self.action_data), dtype=np.float32)
        self.h5file.create_dataset("timestamps", data=np.array(self.timestamps), dtype=np.float64)
        dt = h5py.string_dtype(encoding='utf-8')
        for i, name in enumerate(self.camera_names):
            color_paths = [p[i]["color"] for p in self.image_paths if len(p) > i]
            depth_paths = [p[i]["depth"] or "" for p in self.image_paths if len(p) > i]
            self.h5file.create_dataset(f"color_image_paths_{name}", data=np.array(color_paths, dtype=dt))
            self.h5file.create_dataset(f"depth_image_paths_{name}", data=np.array(depth_paths, dtype=dt))
        self.h5file.close()
        print(f"[HDF5Recorder] Saved dataset to {self.filename}")

# Main Control System
class ControlSystem:
    def __init__(self, config):
        self.config = config
        self.arm = xArm7GripperEnv(
            robot_ip=config["robot_ip"],
            arm_speed=config["arm_speed"],
            gripper_speed=config["gripper_speed"],
            task_name=config["task_name"],
            max_lin_speed_mm=config.get("max_lin_speed_mm", 50),
            max_rot_speed_deg=config.get("max_rot_speed_deg", 5),
            workspace_half_extent_mm=int(config.get("workspace_half_extent_m", 0.25) * 1000)
        )
        if config["controller_type"] == "playstation":
            self.controller = PlayStationController(
                pos_step_size=config["pos_step_size"],
                rot_step_size=config["rot_step_size"]
            )
        elif config["controller_type"] == "vr":
            self.controller = VRController()
        else:
            raise ValueError("Unknown controller_type")

        if config["mode"] == "sim":
            self.sim_ros_node = _SimRosNode(config["ros2_topic"], config["ros2_node_name"])
            threading.Thread(target=self.sim_ros_node.spin, daemon=True).start()

        self.camera_manager = CameraManager(
            realsense_ids=config["realsense_ids"],
            webcam_ids=config["webcam_ids"],
            realsense_names=config["realsense_names"],
            webcam_names=config["webcam_names"],
            base_save_dir=config["save_dir"]
        )
        self.data_recorder = HDF5Recorder(
            save_dir=config["save_dir"],
            task_name=config["task_name"],
            camera_names=config["realsense_names"] + config["webcam_names"]
        )
    

    def run(self, num_steps=500):
        dt = 0.05  # seconds per step (same as original sleep)
        for step in range(num_steps):
            self.arm.update_arm_state()
            state_dict = {
                "cartesian_position": list(self.arm.arm_pos) + list(self.arm.arm_rot),
                "gripper_position": self.arm.gripper_pos,
            }
            velocity, gripper_command = self.controller.get_action(state_dict)

            self.arm.move(safe_velocity, dt=dt)

            if gripper_command == 'open':
                self.arm.gripper_open()
            elif gripper_command == 'close':
                self.arm.gripper_close()

            self.camera_manager.record_frame(action_idx=step)
            image_paths = [path[-1] for path in self.camera_manager.image_paths]
            gripper_value = {'open': 1.0, 'close': 0.0, 'none': -1.0}[gripper_command]
            self.data_recorder.record_step(action=safe_velocity + [gripper_value], image_paths=image_paths)
            time.sleep(dt)
        self.data_recorder.save()
        self.camera_manager.stop()
    

# Command-Line Argument Parsing (kept for completeness, but we will auto-run by default)
def parse_args():
    parser = argparse.ArgumentParser(description="xArm Control with VR or PlayStation Controller")
    parser.add_argument("--controller_type", type=str, default="playstation", choices=["playstation", "vr"],
                        help="Type of controller to use")
    parser.add_argument("--task_name", type=str, default="pick up carrot and place in bowl", help="Name of the task")
    parser.add_argument("--save_dir", type=str, default=os.path.expanduser("~/workspace/xarm-dataset"), help="Directory to save data")
    parser.add_argument("--robot_ip", type=str, default="192.168.1.198", help="Robot IP address")
    parser.add_argument("--arm_speed", type=int, default=1000, help="Arm speed")
    parser.add_argument("--gripper_speed", type=int, default=2000, help="Gripper speed")
    parser.add_argument("--pos_step_size", type=int, default=50, help="Position step size (PlayStation only)")
    parser.add_argument("--rot_step_size", type=int, default=15, help="Rotation step size (PlayStation only)")
    parser.add_argument("--realsense_ids", type=str, nargs="*", default=[], help="RealSense camera serial numbers")
    parser.add_argument("--webcam_ids", type=int, nargs="*", default=[4, 10], help="Webcam device IDs")
    parser.add_argument("--realsense_names", type=str, nargs="*", default=["rs1", "rs2"],
                        help="Names for RealSense cameras")
    parser.add_argument("--webcam_names", type=str, nargs="*", default=["wristcam", "exo1"],
                        help="Names for webcams")
    parser.add_argument("--num_steps", type=int, default=500, help="Number of steps to run")
    parser.add_argument(
    "--mode",
    type=str,
    default="real",
    choices=["real", "sim"],
    help="Run against a real robot or publish commands to ROS2 for simulation."
    )
    parser.add_argument(
        "--ros2_topic",
        type=str,
        default="/xarm/velocity_cmd",
        help="ROS2 topic to publish velocity commands when in sim mode."
    )
    parser.add_argument(
        "--ros2_node_name",
        type=str,
        default="xarm_sim_controller",
        help="Name of the ROS2 node that publishes simulation commands."
    )
    args = parser.parse_args()

    # Validate camera names
    if len(args.realsense_names) != len(args.realsense_ids):
        args.realsense_names = [f"rs{i+1}" for i in range(len(args.realsense_ids))]
    if len(args.webcam_names) != len(args.webcam_ids):
        args.webcam_names = [f"wc{i+1}" for i in range(len(args.webcam_ids))]

    # Prepare save directory
    trajectory_num = 0
    base_path = os.path.join(args.save_dir, args.task_name.replace(" ", "_"))
    while os.path.isdir(os.path.join(base_path, f"trajectory{trajectory_num}")):
        trajectory_num += 1
    args.save_dir = os.path.join(base_path, f"trajectory{trajectory_num}")

    config = {
        "controller_type": args.controller_type,
        "task_name": args.task_name,
        "save_dir": args.save_dir,
        "robot_ip": args.robot_ip,
        "arm_speed": args.arm_speed,
        "gripper_speed": args.gripper_speed,
        "pos_step_size": args.pos_step_size,
        "rot_step_size": args.rot_step_size,
        "realsense_ids": args.realsense_ids,
        "webcam_ids": args.webcam_ids,
        "realsense_names": args.realsense_names,
        "webcam_names": args.webcam_names,
        "num_steps": args.num_steps
    }
    return config

def default_config():
    # Auto-run friendly defaults (you can edit these values directly)
    save_dir = os.path.expanduser("~/workspace/xarm-dataset")
    task_name = "pick up carrot and place in bowl"
    base_path = os.path.join(save_dir, task_name.replace(" ", "_"))
    trajectory_num = 0
    while os.path.isdir(os.path.join(base_path, f"trajectory{trajectory_num}")):
        trajectory_num += 1
    save_dir = os.path.join(base_path, f"trajectory{trajectory_num}")

    config = {
        "controller_type": "vr",  # or "vr"
        "task_name": task_name,
        "save_dir": save_dir,
        "robot_ip": "192.168.1.198",
        "arm_speed": 1000,
        "gripper_speed": 2000,
        "pos_step_size": 20,  # mm/s default for playstation (small)
        "rot_step_size": 5,   # deg/s (small)
        "realsense_ids": [],
        "webcam_ids": [4, 10],
        "realsense_names": ["rs1", "rs2"],
        "webcam_names": ["wristcam", "exo1"],
        "num_steps": 500,
        # safety + workspace parameters:
        "max_lin_speed_mm": 250,  # VERY slow linear speed (50 mm/s)
        "max_rot_speed_deg": 30,  # VERY slow rotational speed (5 deg/s)
        "workspace_half_extent_m": 0.25,  # half-length in meters (0.25m -> 0.5 m cube)
        "mode": "real",          # ← new key, values: "real" or "sim"
        "ros2_topic": "/xarm/velocity_cmd",   # topic name used when mode == "sim"
        "ros2_node_name": "xarm_sim_controller",

    }
    return config

if __name__ == "__main__":
    # Use default_config() to auto-run without terminal args
    config = default_config()

    # If you still want to use argparse, you can uncomment below:
    # config = parse_args()

    control_system = ControlSystem(config)
    control_system.run(num_steps=config["num_steps"])
