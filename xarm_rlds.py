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
import numpy as np
from scipy.spatial.transform import Rotation as R
import multiprocessing
import subprocess
import threading
from queue import Queue, Empty

# Low-latency rates
VR_UPDATE_HZ = 120
ROBOT_UPDATE_HZ = 200

def run_terminal_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True, executable="/bin/bash", encoding="utf8"
    )
    return process


def run_threaded_command(target_fn, args=(), daemon=True):
    thread = threading.Thread(target=target_fn, args=args, daemon=daemon)
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


def rmat_to_quat(rot_mat):
    """
    Convert a rotation matrix to a quaternion [x, y, z, w].
    Validates matrix before conversion.
    """
    rot_mat = np.array(rot_mat, dtype=float)
    
    # Check for invalid or zero matrix
    if rot_mat.shape != (3, 3) or not np.isfinite(rot_mat).all() or np.allclose(rot_mat, 0):
        # Return a neutral quaternion
        return np.array([0, 0, 0, 1], dtype=float)
    
    # Also check determinant to avoid left-handed frames
    if np.linalg.det(rot_mat) <= 0:
        return np.array([0, 0, 0, 1], dtype=float)
    
    return R.from_matrix(rot_mat).as_quat()


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
    return result.as_euler("xyz", degrees=degrees)


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
        self.robot_ip = robot_ip
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.task_name = task_name
        self.arm_pos = None
        self.arm_rot = None
        self.previous_arm_pos = None
        self.gripper_pos = None
        self.wait = False

        # Gripper latch target: None => no target, 1 => open target, 0 => close target
        self.gripper_target = None
        self.gripper_tolerance = 5.0  # hardware units tolerance

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
        # set initial gripper target to open
        self.gripper_target = 1
        self.apply_gripper_target()
        self.arm.set_mode(5)
        self.arm.set_state(0)

    def update_arm_state(self):
        _, arm_pos = self.arm.get_position(is_radian=False)
        self.previous_arm_pos = self.arm_pos
        self.arm_pos = tuple(arm_pos[:3])  # mm
        self.arm_rot = tuple(arm_pos[3:])  # deg
        _, gripper_pos = self.arm.get_gripper_position()
        self.gripper_pos = gripper_pos
        # If we have a gripper target, keep applying until reached
        if self.gripper_target is not None:
            self.apply_gripper_target()

    def _clamp_to_workspace(self, pos_mm):
        pos = np.array(pos_mm, dtype=float)
        clamped = np.minimum(np.maximum(pos, self.workspace_min), self.workspace_max)
        return tuple(clamped.tolist())

    def move_position(self, action):
        x, y, z = action[0], action[1], action[2]
        x_c, y_c, z_c = self._clamp_to_workspace((x, y, z))
        clamped_action = (x_c, y_c, z_c, action[3], action[4], action[5])
        self.arm.set_position(x=clamped_action[0], y=clamped_action[1], z=clamped_action[2],
                              roll=clamped_action[3], pitch=clamped_action[4], yaw=clamped_action[5],
                              relative=False, wait=True, speed=100)

    def move(self, action, dt=0.005):
        if action is None or np.allclose(action, [0, 0, 0, 0, 0, 0]):
            return

        lin = np.array(action[:3], dtype=float)
        rot = np.array(action[3:], dtype=float)

        lin = np.clip(lin, -self.max_lin_speed_mm, self.max_lin_speed_mm)
        rot = np.clip(rot, -self.max_rot_speed_deg, self.max_rot_speed_deg)

        if self.arm_pos is None:
            self.update_arm_state()
        current_pos = np.array(self.arm_pos, dtype=float)
        pred_pos = current_pos + lin * dt  # mm

        # for i in range(3):
        #     if pred_pos[i] < self.workspace_min[i] or pred_pos[i] > self.workspace_max[i]:
        #         lin[i] = 0.0

        safe_action = np.concatenate([lin, rot]).tolist()

        if any(abs(v) > 1e-6 for v in safe_action):
            # non-blocking velocity API call
            self.arm.vc_set_cartesian_velocity(safe_action)

    def apply_gripper_target(self):
        """
        Sends gripper set commands repeatedly until the hardware reports it's at target.
        gripper hardware values: 0 (closed) .. 850 (open) used in earlier code.
        self.gripper_target values: 1 -> open (850), 0 -> closed (0)
        """
        if self.gripper_target is None:
            return

        target_val_hw = 850 if self.gripper_target == 1 else 0
        try:
            # Always issue non-blocking set until it's within tolerance
            self.arm.set_gripper_position(target_val_hw, wait=False)
            # If gripper_pos is available, check if reached
            if self.gripper_pos is not None:
                # gripper_pos might be reported as a tuple or scalar; handle both
                cur = self.gripper_pos[0] if isinstance(self.gripper_pos, (list, tuple)) else self.gripper_pos
                if abs(cur - target_val_hw) <= self.gripper_tolerance:
                    # target reached; clear target so we don't continually send redundant commands
                    self.gripper_target = None
        except Exception as e:
            print(f"[xArm7GripperEnv] apply_gripper_target error: {e}")

    def gripper_open(self):
        # set a persistent open target
        self.gripper_target = 1
        self.apply_gripper_target()

    def gripper_close(self):
        # set a persistent close target
        self.gripper_target = 0
        self.apply_gripper_target()


# VR Policy (produces vr_state into queue and keeps a latest fallback)
class VRPolicy:
    def __init__(self,
                 vr_queue: Queue = None,
                 right_controller: bool = True,
                 spatial_coeff: float = 1,
                 rmat_reorder: list = [-2, -1, -3, 4],
                 update_hz: int = VR_UPDATE_HZ):
        self.oculus_reader = OculusReader()
        self.vr_to_global_mat = np.eye(4)
        self.spatial_coeff = spatial_coeff
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self._state = {"poses": {}, "buttons": {}, "controller_on": False}
        self.vr_queue = vr_queue
        self.latest_vr_state = None
        self.update_hz = update_hz

        run_threaded_command(self._update_internal_state)

    def _update_internal_state(self):
        last_read_time = time.time()
        hz = self.update_hz or VR_UPDATE_HZ
        while True:
            time.sleep(1.0 / hz)
            try:
                poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            except Exception as e:
                # don't flood logs
                print(f"[VRPolicy] OculusReader error: {e}")
                continue

            now = time.time()
            if poses is not None and poses != {}:
                self._state["poses"] = poses
                last_read_time = now
            if buttons is not None and buttons != {}:
                self._state["buttons"].update(buttons)

            self._state["controller_on"] = (now - last_read_time) < 5.0

            ok = self._process_reading()
            if not ok:
                continue

            # store latest fallback
            self.latest_vr_state = self.vr_state

            # try to push to queue (fast-path). if queue full, discard oldest then push newest
            if self.vr_queue is not None:
                try:
                    if self.vr_queue.full():
                        try:
                            self.vr_queue.get_nowait()
                        except Exception:
                            pass
                    self.vr_queue.put_nowait(self.vr_state)
                except Exception:
                    pass

    def _process_reading(self):
        if not self._state["poses"]:
            return False

        rot_mat = None
        if self.controller_id in self._state["poses"]:
            rot_mat = np.asarray(self._state["poses"][self.controller_id])
        else:
            try:
                any_key = next(iter(self._state["poses"].keys()))
                rot_mat = np.asarray(self._state["poses"][any_key])
            except Exception:
                return False

        try:
            rot_mat = self.global_to_env_mat @ self.vr_to_global_mat @ rot_mat
        except Exception as e:
            print(f"[VRPolicy] error applying transform matrices: {e}")
            return False

        vr_pos = self.spatial_coeff * rot_mat[:3, 3]
        vr_quat = rmat_to_quat(rot_mat[:3, :3])
        trig_key = "rightTrig" if self.controller_id == "r" else "leftTrig"
        trig_val = self._state["buttons"].get(trig_key, (0.0,))
        if isinstance(trig_val, (list, tuple)):
            trig = trig_val[0]
        else:
            trig = float(trig_val) if trig_val is not None else 0.0

        self.vr_state = {"pos": vr_pos, "quat": vr_quat, "gripper": trig}
        return True


# Controller Base Class
class BaseController(ABC):
    @abstractmethod
    def get_action(self, state_dict):
        pass


# PlayStation Controller (unchanged)
class PlayStationController(BaseController):
    def __init__(self, pos_step_size=50, rot_step_size=15, threshold=0.2):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.pos_step_size = pos_step_size  # mm/s
        self.rot_step_size = rot_step_size  # deg/s
        self.threshold = threshold
        self.gripper_state = 1

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
        if abs(LeftJoystickY) > self.threshold:
            velocity[0] = (-LeftJoystickY) * self.pos_step_size
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
        elif SQUARE:
            gripper_command = 'close'

        return velocity, gripper_command


# VR Controller wrapper: uses queue fast-path and latest_vr_state fallback; computes actions
class VRController(BaseController):
    def __init__(self,
                 vr_queue: Queue = None,
                 right_controller: bool = True,
                 max_lin_vel: float = 0.3,
                 max_rot_vel: float = 0.1,
                 max_gripper_vel: float = 1,
                 spatial_coeff: float = 1,
                 pos_action_gain: float = 5,
                 rot_action_gain: float = 2,
                 gripper_action_gain: float = 3,
                 rmat_reorder: list = [-2, -1, -3, 4]):
        self.vr_queue = vr_queue or Queue(maxsize=1)
        self.vr_policy = VRPolicy(vr_queue=self.vr_queue, right_controller=right_controller,
                                  spatial_coeff=spatial_coeff, rmat_reorder=rmat_reorder)

        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.max_gripper_vel = max_gripper_vel
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.gripper_action_gain = gripper_action_gain

        self.reset_origin = True
        self.robot_origin = None
        self.vr_origin = None
        self.vr_state_last = None

    def _drain_queue_get_latest(self):
        latest = None
        try:
            while True:
                item = self.vr_queue.get_nowait()
                latest = item
        except Empty:
            pass
        return latest

    def _get_vr_state(self):
        # try queue (fastest path)
        latest_from_queue = self._drain_queue_get_latest()
        if latest_from_queue is not None:
            self.vr_state_last = latest_from_queue
            return latest_from_queue
        # fallback to VRPolicy.latest_vr_state
        if hasattr(self.vr_policy, "latest_vr_state") and self.vr_policy.latest_vr_state is not None:
            self.vr_state_last = self.vr_policy.latest_vr_state
            return self.vr_state_last
        return None

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel=0.0):
        lin_vel_norm = np.linalg.norm(lin_vel)
        rot_vel_norm = np.linalg.norm(rot_vel)
        gripper_vel_norm = np.linalg.norm([gripper_vel])
        if lin_vel_norm > self.max_lin_vel and lin_vel_norm > 0:
            lin_vel = lin_vel * self.max_lin_vel / lin_vel_norm
        if rot_vel_norm > self.max_rot_vel and rot_vel_norm > 0:
            rot_vel = rot_vel * self.max_rot_vel / rot_vel_norm
        if gripper_vel_norm > self.max_gripper_vel and gripper_vel_norm > 0:
            gripper_vel = gripper_vel * self.max_gripper_vel / gripper_vel_norm
        return lin_vel, rot_vel, gripper_vel

    def get_action(self, state_dict):
        vr_state = self._get_vr_state()
        if vr_state is None:
            return [0.0] * 6, 'none'

        robot_pos = np.array(state_dict["cartesian_position"][:3], dtype=float) / 1000.0  # mm -> m
        robot_euler_deg = state_dict["cartesian_position"][3:]
        robot_quat = euler_to_quat(robot_euler_deg, degrees=True)

        # reset origins as before
        if self.reset_origin or self.robot_origin is None or self.vr_origin is None:
            self.robot_origin = {"pos": robot_pos.copy(), "quat": robot_quat.copy()}
            self.vr_origin = {"pos": vr_state["pos"].copy(), "quat": vr_state["quat"].copy()}
            self.reset_origin = False

        # compute pos action in meters
        robot_pos_offset = robot_pos - self.robot_origin["pos"]
        target_pos_offset = vr_state["pos"] - self.vr_origin["pos"]
        pos_action = target_pos_offset - robot_pos_offset  # meters

        robot_quat_offset = quat_diff(robot_quat, self.robot_origin["quat"])
        target_quat_offset = quat_diff(vr_state["quat"], self.vr_origin["quat"])
        quat_action = quat_diff(target_quat_offset, robot_quat_offset)
        euler_action_rad = quat_to_euler(quat_action)            # radians
        euler_action = np.degrees(euler_action_rad)  

        pos_action *= self.pos_action_gain
        euler_action *= self.rot_action_gain
        # gripper_action *= self.gripper_action_gain

        lin_vel, rot_vel, gripper_vel = self._limit_velocity(pos_action, euler_action)

        action = np.concatenate([lin_vel, rot_vel, [gripper_vel]])
        action = action.clip(-1, 1)

        # convert linear m/s -> mm/s for robot API
        for i in range(3):
            if abs(action[i]) < 0.03:
                action[i] = 0.0
        lin_vel_mm = (lin_vel * 1000.0).tolist()
        rot_vel_list = rot_vel.tolist()                          # degrees/sec, ready for API
        velocity = lin_vel_mm + rot_vel_list

        rg = state_dict["gripper_position"]
        rg = rg[0] if isinstance(rg, (list, tuple)) else float(rg)
        robot_gripper_norm = np.clip(rg / 850.0, 0.0, 1.0)

        # Simple proportional action in [−1,1] before gains/limits
        gripper_action = (vr_state["gripper"]) - robot_gripper_norm

        if vr_state["gripper"] > 0.75: gripper_command = 'open'
        elif vr_state["gripper"] < 0.25: gripper_command = 'close'
        else: gripper_command = 'none'

        return velocity, gripper_command


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
            print(serial)
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, 30)
            config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, 30)
            try:
                pipeline.start(config)
                time.sleep(1)
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


# Webcam Recorder (unchanged)
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


# Camera Manager (unchanged)
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


# HDF5 Recorder (unchanged)
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
        # shared queue (fast-path)
        self.vr_queue = Queue(maxsize=1)

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
            self.controller = VRController(vr_queue=self.vr_queue)
        else:
            raise ValueError("Unknown controller_type")

        if config.get("mode", "real") == "sim":
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
        dt = 1.0 / ROBOT_UPDATE_HZ
        step =0
        while True:
            loop_start = time.time()

            self.arm.update_arm_state()
            state_dict = {
                "cartesian_position": list(self.arm.arm_pos) + list(self.arm.arm_rot),
                "gripper_position": self.arm.gripper_pos,
            }

            velocity, gripper_command = self.controller.get_action(state_dict)
            print(velocity)
            # Send velocity (non-blocking)
            self.arm.move(velocity, dt=dt)

            # Gripper handling: use latch behavior (only change target on explicit 'open' or 'close')
            if gripper_command == 'open':
                self.arm.gripper_open()
            elif gripper_command == 'close':
                self.arm.gripper_close()
            # if 'none' do nothing (preserve previous target until reached)

            # Cameras (left as-is)
            self.camera_manager.record_frame(action_idx=step)
            image_paths = [path[-1] for path in self.camera_manager.image_paths]

            gripper_value = {'open': 1.0, 'close': 0.0, 'none': -1.0}[gripper_command]
            self.data_recorder.record_step(action=velocity + [gripper_value], image_paths=image_paths)

            elapsed = time.time() - loop_start
            to_sleep = dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            
            step +=1

        self.data_recorder.save()
        self.camera_manager.stop()


# Arg parsing and defaults (unchanged)
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

    if len(args.realsense_names) != len(args.realsense_ids):
        args.realsense_names = [f"rs{i+1}" for i in range(len(args.realsense_ids))]
    if len(args.webcam_names) != len(args.webcam_ids):
        args.webcam_names = [f"wc{i+1}" for i in range(len(args.webcam_ids))]

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
    save_dir = os.path.expanduser("~/workspace/xarm-dataset")
    task_name = "pick up carrot and place in bowl"
    base_path = os.path.join(save_dir, task_name.replace(" ", "_"))
    trajectory_num = 0
    while os.path.isdir(os.path.join(base_path, f"trajectory{trajectory_num}")):
        trajectory_num += 1
    save_dir = os.path.join(base_path, f"trajectory{trajectory_num}")

    config = {
        "controller_type": "vr",
        "task_name": task_name,
        "save_dir": save_dir,
        "robot_ip": "192.168.1.198",
        "arm_speed": 1000,
        "gripper_speed": 2000,
        "pos_step_size": 20,
        "rot_step_size": 5,
        "realsense_ids": ["341522301282","334622071624"],
        "webcam_ids": [],
        "realsense_names": ["wristcam", "exo1"],
        "webcam_names": [],
        "num_steps": 500,
        "max_lin_speed_mm": 250,
        "max_rot_speed_deg": 30,
        "workspace_half_extent_m": 0.25,
        "mode": "real",
        "ros2_topic": "/xarm/velocity_cmd",
        "ros2_node_name": "xarm_sim_controller",
    }
    return config


if __name__ == "__main__":
    config = default_config()
    # config = parse_args()  # uncomment to use CLI
    control_system = ControlSystem(config)
    control_system.run(num_steps=config["num_steps"])
