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


# VR Policy for VR Controller
class VRPolicy:
    def __init__(self, right_controller=True, max_lin_vel=0.05, max_rot_vel=0.1, spatial_coeff=1, pos_action_gain=5, rot_action_gain=2, rmat_reorder=[-2, -1, -3, 4]):
        """
        NOTE: VRPolicy works in meters for position deltas. We will convert to mm before issuing to the robot.
        max_lin_vel: max linear velocity (m/s) in VR policy; will be converted to mm/s downstream.
        """
        # self.oculus_reader = OculusReader()  # keep original usage in your environment
        self.oculus_reader = OculusReader()  # placeholder if OculusReader not available in this environment
        self.vr_to_global_mat = np.eye(4)
        self.max_lin_vel = max_lin_vel
        self.max_rot_vel = max_rot_vel
        self.spatial_coeff = spatial_coeff
        self.pos_action_gain = pos_action_gain
        self.rot_action_gain = rot_action_gain
        self.global_to_env_mat = vec_to_reorder_mat(rmat_reorder)
        self.controller_id = "r" if right_controller else "l"
        self.reset_orientation = True
        self.reset_state()
        run_threaded_command(self._update_internal_state)  # disabled unless OculusReader present

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
            time.sleep(1 / hz)
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            if not poses:
                self._state["controller_on"] = (time.time() - last_read_time) < num_wait_sec
                continue
            last_read_time = time.time()
            self._state["controller_on"] = True

            toggled = self._state["movement_enabled"] != buttons[self.controller_id.upper() + "G"]
            self.update_sensor |= buttons[self.controller_id.upper() + "G"]
            self.reset_orientation |= buttons[self.controller_id.upper() + "J"]
            self.reset_origin |= toggled
            self._state["poses"] = poses
            self._state["buttons"] = buttons
            self._state["movement_enabled"] = buttons[self.controller_id.upper() + "G"]

            # Always update current VR pose
            if self.controller_id in poses:
                rot_mat = np.asarray(poses[self.controller_id])
                self.vr_state = {"pos": rot_mat[:3, 3], "rot": rot_mat[:3, :3]}

            # Only reset orientation if flagged
            if self.reset_orientation and self.controller_id in poses:
                rot_mat = np.asarray(poses[self.controller_id])
                rot_mat = np.linalg.inv(rot_mat)
                self.vr_to_global_mat = rot_mat
                self.reset_orientation = False

    def _limit_velocity(self, lin_vel_m, rot_vel):
        # lin_vel_m is in meters per second here; clip to max_lin_vel (m/s)
        norm = np.linalg.norm(lin_vel_m)
        if norm > self.max_lin_vel and norm > 0:
            lin_vel_m = lin_vel_m * (self.max_lin_vel / norm)
        rnorm = np.linalg.norm(rot_vel)
        if rnorm > self.max_rot_vel and rnorm > 0:
            rot_vel = rot_vel * (self.max_rot_vel / rnorm)
        return lin_vel_m, rot_vel

    def _calculate_action(self, state_dict):
        if self.vr_state is None or not self._state.get("poses"):
            return np.zeros(3), np.zeros(3), 'none'

        current_pos = np.array(self.vr_state["pos"])
        current_rot = np.array(self.vr_state["rot"])

        now = time.time()
        if not hasattr(self, "_last_time"):
            self._last_time = now
            self._last_pos = current_pos
            self._last_rot = current_rot
            return np.zeros(3), np.zeros(3), 'none'

        dt = now - self._last_time
        if dt <= 0: dt = 1e-6

        # Linear velocity in m/s (difference / dt)
        lin_vel_m = (current_pos - self._last_pos) * self.spatial_coeff / dt

        # Rotational velocity in rad/s
        rot_delta_mat = current_rot @ self._last_rot.T
        rotvec = R.from_matrix(rot_delta_mat).as_rotvec()  # radians
        rot_vel = rotvec / dt

        # Update memory
        self._last_time = now
        self._last_pos = current_pos
        self._last_rot = current_rot

        # Apply limits
        lin_vel_m, rot_vel = self._limit_velocity(lin_vel_m, rot_vel)

        # Gripper commands
        gripper_command = 'none'
        if self._state["buttons"].get("X", False):
            gripper_command = 'open'
        elif self._state["buttons"].get("Y", False):
            gripper_command = 'close'

        return lin_vel_m, np.degrees(rot_vel), gripper_command  

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

            # Ensure velocity is length 6
            velocity = list(velocity)[:6] + [0] * (6 - len(velocity))

            # Clip final velocities to env maxs (safety)
            # linear units expected in mm/s here
            lin = np.clip(np.array(velocity[:3], dtype=float),
                          -self.arm.max_lin_speed_mm, self.arm.max_lin_speed_mm)
            rot = np.clip(np.array(velocity[3:6], dtype=float),
                          -self.arm.max_rot_speed_deg, self.arm.max_rot_speed_deg)
            safe_velocity = np.concatenate([lin, rot]).tolist()

            # Let the env check workspace crossing and apply velocities
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
    }
    return config

if __name__ == "__main__":
    # Use default_config() to auto-run without terminal args
    config = default_config()

    # If you still want to use argparse, you can uncomment below:
    # config = parse_args()

    control_system = ControlSystem(config)
    control_system.run(num_steps=config["num_steps"])
