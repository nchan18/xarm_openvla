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
        right_controller=True,
        max_lin_vel=1,
        max_rot_vel=1,
        max_gripper_vel=1,
        spatial_coeff=1,
        pos_action_gain=5,
        rot_action_gain=2,
        gripper_action_gain=3,
        rmat_reorder=[-2, -1, -3, 4],
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
            time.sleep(1 / hz)
            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            if not poses:
                self._state["controller_on"] = (time.time() - last_read_time) < num_wait_sec
                print("[VR] No poses received from OculusReader.")
                continue
            last_read_time = time.time()
            self._state["controller_on"] = True
            print("[VR] OculusReader returned poses:", poses.keys())

            toggled = self._state["movement_enabled"] != buttons[self.controller_id.upper() + "G"]
            self.update_sensor |= buttons[self.controller_id.upper() + "G"]
            self.reset_orientation |= buttons[self.controller_id.upper() + "J"]
            self.reset_origin |= toggled
            self._state["poses"] = poses
            self._state["buttons"] = buttons
            self._state["movement_enabled"] = buttons[self.controller_id.upper() + "G"]
            self._state["controller_on"] = True
            last_read_time = time.time()

            stop_updating = self._state["buttons"][self.controller_id.upper() + "J"] or self._state["movement_enabled"]
            if self.reset_orientation:
                print("[DEBUG] Received VR pose for controller:", self.controller_id)
                print("[DEBUG] VR pose matrix:\n", self._state["poses"].get(self.controller_id, "Not Found"))
                try:
                    rot_mat = np.asarray(self._state["poses"][self.controller_id])
                    self.vr_state = {
                        "pos": rot_mat[:3, 3],
                        "rot": rot_mat[:3, :3]
                    }
                    rot_mat = np.linalg.inv(rot_mat)
                    self.vr_to_global_mat = rot_mat
                    if stop_updating:
                        self.reset_orientation = False
                except Exception as e:
                    print(f"exception for rot mat: {e}")
                    self.reset_orientation = True
                    self.vr_to_global_mat = np.eye(4)

    def _process_reading(self):
        if not self._state.get("poses"):
            print("Warning: poses dictionary is empty or missing.")
            return
        if self.controller_id not in self._state["poses"]:
            print(f"Warning: controller_id '{self.controller_id}' not found in poses yet.")
            return
        rot_mat = np.asarray(self._state["poses"][self.controller_id])
        print(f"Rotation matrix for controller '{self.controller_id}':\n{rot_mat}")

    def _limit_velocity(self, lin_vel, rot_vel, gripper_vel):
        if np.linalg.norm(lin_vel) > self.max_lin_vel:
            lin_vel = lin_vel * self.max_lin_vel / np.linalg.norm(lin_vel)
        if np.linalg.norm(rot_vel) > self.max_rot_vel:
            rot_vel = rot_vel * self.max_rot_vel / np.linalg.norm(rot_vel)
        if np.linalg.norm(gripper_vel) > self.max_gripper_vel:
            gripper_vel = gripper_vel * self.max_gripper_vel / np.linalg.norm(gripper_vel)
        return lin_vel, rot_vel, gripper_vel

    def _calculate_action(self, state_dict, include_info=False):
        if self.vr_state is None:
            print("Warning: vr_state is None — skipping action calculation.")
            return np.zeros(3), np.zeros(3), np.zeros(1)

        if self.update_sensor:
            self._process_reading()
            self.update_sensor = False

        robot_pos = np.array(state_dict["cartesian_position"][:3])
        robot_euler = state_dict["cartesian_position"][3:]
        robot_quat = euler_to_quat(robot_euler)
        robot_gripper = state_dict["gripper_position"]

        if self.reset_origin:
            self.vr_origin = np.copy(self.vr_state["pos"])
            self.robot_origin = np.copy(robot_pos)
            self.reset_origin = False

        delta = self.vr_state["pos"] - self.vr_origin
        lin_vel = delta * self.pos_action_gain
        lin_vel, rot_vel, gripper_vel = self._limit_velocity(lin_vel, np.zeros(3), np.array([robot_gripper]))

        if include_info:
            return lin_vel, rot_vel, gripper_vel, robot_pos, robot_euler
        return lin_vel, rot_vel, gripper_vel

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
        self.vr_policy = VRPolicy()
        self.video_recorder = RealSenseVideoRecorder()
        self.hdf5_recorder = HDF5Recorder(task_name=task_name, save_dir=save_dir)

    def run(self, num_steps=100):
        for step in range(num_steps):
            lin_vel, rot_vel, gripper_vel = self.vr_policy._calculate_action({
                "cartesian_position": list(self.robot.arm_pos) + list(self.robot.arm_rot),
                "gripper_position": self.robot.gripper_pos,
            })
            pos = [i * 1000 for i in lin_vel]
            rot = [i * 180/math.pi for i in rot_vel]
            action_vec = pos + rot
            
            self.robot.move(action_vec)
            self.video_recorder.record_frame(action_idx=step)
            last_image = self.video_recorder.image_path[-1]
            self.hdf5_recorder.record_step(action=action_vec,
                                           gripper_pos=self.robot.gripper_pos,
                                           color_path=last_image["color"],
                                           depth_path=last_image["depth"])
            time.sleep(0.05)
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
    control_system.run(num_steps=500)