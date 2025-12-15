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
import sys
from scipy.spatial.transform import Rotation as R
from oculus_reader.reader import OculusReader
import numpy as np
from scipy.spatial.transform import Rotation as R

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

class xArm7GripperEnv:
    def __init__(self, robot_ip="192.168.1.198", arm_speed=1000, gripper_speed=500, pos_step_size=50, rot_step_size=5, grip_size=100, task_name=" "):
        self.robot_ip = robot_ip
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.pos_step_size = pos_step_size
        self.rot_step_size = rot_step_size
        self.task_name = task_name
        self.grip_size = grip_size
        self.arm_pos = None
        self.arm_rot = None
        self.previous_arm_pos = None
        self.gripper_pos = None
        self.wait = False
        self.save_video = False
        self.arm = XArmAPI(self.robot_ip)
        self.arm_starting_pose = (300, 0, 300, 180, 0, 0)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(self.gripper_speed)
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
        self.arm_pos = tuple(arm_pos[:3])
        self.arm_rot = tuple(arm_pos[3:])
        _, gripper_pos = self.arm.get_gripper_position()
        self.gripper_pos = gripper_pos

    def move_position(self, action):
        self.arm.set_position(x=action[0], y=action[1], z=action[2],
                              roll=action[3], pitch=action[4], yaw=action[5],
                              relative=False, wait=True, speed=100)

    def move(self, action):
        if action != [0, 0, 0, 0, 0, 0]:
            print(action)
        self.arm.vc_set_cartesian_velocity(action)

    def gripper_open(self):
        self.gripper_state = 1
        self.arm.set_gripper_position(850, wait=self.wait)

    def gripper_close(self):
        self.gripper_state = 0
        self.arm.set_gripper_position(0, wait=self.wait)

    def clean_errors(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

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
        

class RealSenseVideoRecorder:
    def __init__(self, frame_size=(640, 480), depth_stream=True, rgb_stream=True):
        self.pipeline = rs.pipeline()
        config = rs.config()
        if depth_stream:
            config.enable_stream(rs.stream.depth, frame_size[0], frame_size[1], rs.format.z16, 30)
        if rgb_stream:
            config.enable_stream(rs.stream.color, frame_size[0], frame_size[1], rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.frame_size = frame_size
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.image_dir = "images_" + self.time_stamp
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_path = []
        self.frame_idx = 0

    def record_frame(self, action_idx=None):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            print("Warning: Failed to get frames from RealSense")
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        resized_color_image = cv2.resize(color_image, self.frame_size)
        if action_idx is not None:
            resized_color_image = cv2.putText(resized_color_image, str(action_idx), (10, 40),
                                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        color_path = os.path.join(self.image_dir, f"color_frame_{self.frame_idx:05d}.jpg")
        depth_path = os.path.join(self.image_dir, f"depth_frame_{self.frame_idx:05d}.png")
        cv2.imwrite(color_path, resized_color_image)
        cv2.imwrite(depth_path, depth_image)
        self.image_path.append({"color": color_path, "depth": depth_path})
        self.frame_idx += 1

    def reset(self):
        self.frame_idx = 0
        self.image_path = []

    def stop(self):
        self.pipeline.stop()

class HDF5Recorder:
    def __init__(self, save_dir="data_logs", task_name="task"):
        self.task_name = task_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.save_dir, f"{task_name}_{timestamp}.hdf5")
        self.h5file = h5py.File(self.filename, "w")
        self.action_data = []
        self.image_paths = []
        self.timestamps = []

    def record_step(self, action, gripper_pos, color_path, depth_path):
        self.action_data.append(np.array(action, dtype=np.float32))
        self.image_paths.append((color_path, depth_path))
        self.timestamps.append(time.time())

    def save(self):
        self.h5file.create_dataset("actions", data=np.array(self.action_data), dtype=np.float32)
        self.h5file.create_dataset("timestamps", data=np.array(self.timestamps), dtype=np.float64)
        dt = h5py.string_dtype(encoding='utf-8')
        color_paths = [pair[0] for pair in self.image_paths]
        depth_paths = [pair[1] for pair in self.image_paths]
        self.h5file.create_dataset("color_image_paths", data=np.array(color_paths, dtype=dt))
        self.h5file.create_dataset("depth_image_paths", data=np.array(depth_paths, dtype=dt))
        self.h5file.close()
        print(f"[HDF5Recorder] Saved dataset to {self.filename}")

class VRControlSystem:
    def __init__(self, config):
        self.config = config
        task_name = config.get("task_name", "default_task")
        save_dir = config.get("save_dir", "data_logs")
        arm_speed = config.get("arm_speed", 1000)
        gripper_speed = config.get("gripper_speed", 500)
        robot_ip = config.get("robot_ip", "192.168.1.198")
        self.robot = xArm7GripperEnv(robot_ip=robot_ip, arm_speed=arm_speed,
                                     gripper_speed=gripper_speed, task_name=task_name)
        self.oculus_reader = OculusReader()
        self.vr_policy = VRPolicy()
        self.video_recorder = RealSenseVideoRecorder()
        self.hdf5_recorder = HDF5Recorder(task_name=task_name, save_dir=save_dir)
        self.step = 0
        self.current_pos = [0,0,0]
        self.current_rot = [0,0,0]

    def run(self):
        while True:
            # action = self._calculate_action(obs_dict["robot_state"], include_info=include_info)
            transformations, buttons = self.oculus_reader.get_transformations_and_buttons()
            if 'r' not in transformations:
                continue

            right_controller_pose = transformations['r']
            right_controller_pose = np.vstack(right_controller_pose)

            rotation_matrix = right_controller_pose[:3, :3]

            # Convert to rotation object
            rot = R.from_matrix(rotation_matrix)

            # Get Euler angles in XYZ order (roll, pitch, yaw), in radians
            euler_xyz = rot.as_euler('xyz', degrees=False)

            # Convert to degrees (optional)
            euler_xyz_deg = np.degrees(euler_xyz)
            pos = [i * 1000 for i in euler_xyz]
            rot = [i * 180/math.pi for i in euler_xyz_deg]

            delta_pos = result_list = [a - b for a, b in zip(self.current_pos, pos)]
            delta_pos = [i * 50 for i in delta_pos]
            delta_rot = result_list = [a - b for a, b in zip(self.current_rot, rot)]
            
            action_vec = delta_pos + delta_rot
            print(f"Action:{action_vec}")
            self.robot.move(action_vec)
            self.current_pos = pos
            self.current_rot = rot
            self.video_recorder.record_frame(action_idx=self.step)
            last_image = self.video_recorder.image_path[-1]
            self.hdf5_recorder.record_step(action=action_vec,
                                           gripper_pos=self.robot.gripper_pos,
                                           color_path=last_image["color"],
                                           depth_path=last_image["depth"])
            time.sleep(0.05)
            self.step += 1
        
        self.hdf5_recorder.save()
        self.video_recorder.stop()

if __name__ == "__main__":
    config = {
        "task_name": "demo_task",
        "save_dir": "data_logs",
        "store_images": True,
        "store_raw_images": False,
    }
    control_system = VRControlSystem(config=config)
    control_system.run()