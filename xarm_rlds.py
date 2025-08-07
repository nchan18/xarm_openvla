import argparse
import datetime
import pygame
import math
import threading
import cv2
import time, os
# from evdev import InputDevice, categorize, ecodes
from xarm.wrapper import XArmAPI
import h5py
import numpy as np

class xArm7GripperEnv:
    def __init__(self, robot_ip="192.168.1.198", arm_speed=1000, gripper_speed=500, pos_step_size=50, rot_step_size=5, grip_size=100,task_name=" "):
        self.robot_ip = robot_ip
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.pos_step_size = pos_step_size
        self.rot_step_size = rot_step_size
        self.task_name = task_name
        self.grip_size = grip_size
        self.arm_pos: tuple = None
        self.previous_arm_pos: tuple = None
        self.arm_rot: tuple = None
        self.previous_arm_pos: tuple = None
        self.gripper_pos: int = None
        self.wait = False
        self.save_video = False
        self.arm = XArmAPI(self.robot_ip)
        self.arm_starting_pose = (300, 0, 170, 180, 0, 0)
        self.arm.set_mode(0) # set_position
        self.arm.set_state(0)
        code = self.arm.set_gripper_mode(0)
        print("set gripper mode: location mode, code={}".format(code))
        code = self.arm.set_gripper_enable(True)
        print("set gripper enable, code={}".format(code))
        code = self.arm.set_gripper_speed(self.gripper_speed)
        print("set gripper speed, code={}".format(code))
        self.move_to_starting_position()

        self.update_arm_state()

    def move_to_starting_position(self):
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.move_position(self.arm_starting_pose)
        self.gripper_open()
        self.arm.set_mode(5) # set catesian velocity
        self.arm.set_state(0)

    def update_arm_state(self):
        _, arm_pos = self.arm.get_position(is_radian=False)
        self.previous_arm_pos = self.arm_pos
        self.previous_arm_rot = self.arm_rot
        self.arm_pos = tuple(arm_pos[:3])
        self.arm_rot = tuple(arm_pos[3:])
        _, gripper_pos = self.arm.get_gripper_position()
        self.gripper_pos = gripper_pos

    def move_position(self,action):
        self.arm.set_position(x=action[0],y=action[1],z=action[2],roll=action[3],pitch=action[4],yaw=action[5], relative=False, wait=True, speed=100)

    def move(self,action):
        if not action == [0,0,0,0,0,0]:
            print(action)

        self.arm.vc_set_cartesian_velocity(action)
        # wait = True
        # self.arm.set_position(x=action[0],y=action[1],z=action[2],roll=action[3],pitch=action[4],yaw=action[5], relative=True, wait=wait, speed=self.arm_speed)

    def gripper_open(self):
        self.gripper_state=1
        self.arm.set_gripper_position(850, wait=self.wait)

    def gripper_close(self):
        self.gripper_state=0
        self.arm.set_gripper_position(0, wait=self.wait)

    def clean_errors(self):
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

class PlayStationController(xArm7GripperEnv):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(
        self,
        *args,
        #input_device="/dev/input/event15",
        save_path="",
        task_name="",
        webcam=[],
        webcam_name = [],
        webcam_crop = [],
        flip_view=True,
        **kwargs,
        
    ):
        super().__init__(*args, **kwargs)
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No joystick detected")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        #save_path = datetime.datetime.now().strftime(f"{self.save_root}/actions_%Y-%m-%d_%H-%M-%S") + ".csv"
        self.task_name = task_name.replace(" ", "_")
        
        self.webcam = webcam
        self.webcam_name = webcam_name
        self.webcam_crop = webcam_crop
        self.flip_view = flip_view
        self.create_new_save_file()
        self.setup_action_save()
        self.setup_camera_recorder()
        self.LeftJoystickY = 0.0
        self.LeftJoystickX = 0.0
        self.RightJoystickY = 0.0
        self.RightJoystickX = 0.0
        self.LeftTrigger = 0.0
        self.RightTrigger = 0.0
        self.LeftBumper = 0.0
        self.RightBumper = 0.0
        self.X = 0.0
        self.SQUARE = 0.0
        self.TRIANGLE = 0.0
        self.O = 0.0
        self.LeftThumb = 0.0
        self.RightThumb = 0.0
        self.Back = 0.0
        self.Start = 0.0
        self.LeftDPad = 0.0
        self.RightDPad = 0.0
        self.UpDPad = 0.0
        self.DownDPad = 0.0
        self.threshold = 0.2
        

    def setup_action_save(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.hdf5_path = os.path.join(self.save_actions, f"actions_{timestamp}.hdf5")
        self.hdf5_file = h5py.File(self.hdf5_path, "w")
        self.action_idx = 0

        self.hdf5_file.create_dataset("abs_pos", shape=(0, 3), maxshape=(None, 3), dtype="f")
        self.hdf5_file.create_dataset("abs_rot", shape=(0, 3), maxshape=(None, 3), dtype="f")
        self.hdf5_file.create_dataset("gripper", shape=(0,), maxshape=(None,), dtype="f")
        self.hdf5_file.create_dataset("rel_action", shape=(0, 6), maxshape=(None, 6), dtype="f")
        self.hdf5_file.create_dataset("abs_jointstate", shape=(0, 7), maxshape=(None, 7), dtype="f")
        for cam_name in self.webcam_name:
            self.hdf5_file.create_dataset(f"image_path_{cam_name}", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
        
    def setup_camera_recorder(self):
        self.video_recorder = VideoRecorder(
            image_dir=self.save_video,
            frame_size=(1280, 800),
            webcam=self.webcam,
            webcam_name=self.webcam_name,
            webcam_crop=self.webcam_crop
        )
    def save_action(self):
        self.update_arm_state()
        abs_pos = np.array([i / 1000 for i in self.arm_pos], dtype=np.float32)
        abs_rot = np.array([i * math.pi / 180 for i in self.arm_rot], dtype=np.float32)
        gripper = np.array(int(self.gripper_pos / 600.0), dtype=np.float32)
        rel_action_pos = tuple(a - b for a, b in zip(self.arm_pos, self.previous_arm_pos))
        rel_action_rot = tuple(a - b for a, b in zip(self.arm_rot, self.previous_arm_rot))
        rel_action = rel_action_pos + rel_action_rot
        rel_action = np.array(rel_action, dtype=np.float32)
        abs_jointstate = np.array(self.arm.get_joint_states()[1][0], dtype=np.float32)

        for key, data in zip(["abs_pos", "abs_rot", "gripper", "rel_action", "abs_jointstate"], [abs_pos, abs_rot, gripper, rel_action, abs_jointstate]):
            dset = self.hdf5_file[key]
            new_shape = (self.action_idx + 1,) + dset.shape[1:]
            dset.resize(new_shape)
            dset[self.action_idx] = data

        # Save frame and path
        self.video_recorder.record_frame(self.action_idx)
        for i in range(len(self.webcam_name)):
            image_path_dset1 = self.hdf5_file[f"image_path_{self.webcam_name[i]}"]
            image_path_dset1.resize(self.action_idx + 1, axis=0)
            image_path_dset1[self.action_idx] = self.video_recorder.image_path[i][-1]

        self.action_idx += 1    

    def reset_save_file(self):
        self.hdf5_file.close()
        self.setup_action_save()
    
    def create_new_save_file(self):
        trajectory_num = 0
        os.makedirs(os.path.join(save_path, self.task_name), exist_ok=True)
        while os.path.isdir(os.path.join(save_path, self.task_name, f"trajectory{trajectory_num}")):
            trajectory_num += 1
        self.save_actions = os.path.join(save_path, self.task_name, f"trajectory{trajectory_num}", "action")
        self.save_video = []
        for cam_name in self.webcam_name:
            self.save_video.append(os.path.join(save_path, self.task_name, f"trajectory{trajectory_num}", "video",cam_name))
            os.makedirs(self.save_video[-1], exist_ok=True)
        os.makedirs(self.save_actions, exist_ok=True)
        
            
    def controller_listen(self):
        joystick = self.joystick
        # while True:
        pygame.event.pump()

        # Read joystick inputs
        LeftJoystickY = joystick.get_axis(1)
        LeftJoystickX = joystick.get_axis(0)
        RightJoystickY = joystick.get_axis(4)
        LeftTrigger = joystick.get_axis(2)
        RightTrigger = joystick.get_axis(5)
        X = joystick.get_button(0)
        SQUARE = joystick.get_button(3)
        TRIANGLE = joystick.get_button(2)
        CIRCLE = joystick.get_button(1)
        DPadX, DPadY = joystick.get_hat(0)
        action = [0,0,0,0,0,0]

        if abs(LeftJoystickY) > self.threshold:
            if LeftJoystickY > 0:
                action[0] = -self.pos_step_size
                print("x_minus")
            else:
                action[0] = self.pos_step_size
                print("x_plus")

        if abs(LeftJoystickX) > self.threshold:
            if LeftJoystickX > 0:
                action[1] = -self.pos_step_size
                print("y_minus")
            else:
                action[1] = self.pos_step_size
                print("y_plus")

        Z_LOWER_LIMIT = 35
        if abs(RightJoystickY) > self.threshold:
            if RightJoystickY > 0:
                if self.arm_pos[2] - self.pos_step_size > Z_LOWER_LIMIT:
                    action[2] = -self.pos_step_size
                    print("z_minus")
                else:
                    print("Z limit reached")
            else:
                action[2] = self.pos_step_size
                print("z_plus")

        if abs(DPadX) > self.threshold:
            if DPadX > 0:
                action[3] = -self.rot_step_size
                print("a_minus")
            else:
                action[3] = self.rot_step_size
                print("a_plus")

        if abs(DPadY) > self.threshold:
            if DPadY > 0:
                action[4] = -self.rot_step_size
                print("b_minus")
            else:
                action[4] = self.rot_step_size
                print("b_plus")

        if LeftTrigger > 0.5:
            action[5] = self.pos_step_size
            print("c_plus")

        if RightTrigger > 0.5:
            action[5] = -self.pos_step_size
            print("c_minus")

        if X:
            self.gripper_open()
            print("gripper_open")

        if SQUARE:
            self.gripper_close()
            print("gripper_close")

        if TRIANGLE:
            self.move_to_starting_position()
        
        if CIRCLE:
            self.reset_save_file()
            self.create_new_save_file()
            self.setup_action_save()


        self.move(action)
        self.save_action()

class VideoRecorder:
    def __init__(self, image_dir= [], frame_size=(1280, 800), webcam=[], webcam_name=[], webcam_crop=[[0,0,0,0]]):
        self.cap = []
        self.webcam = webcam
        self.webcam_name = webcam_name
        self.webcam_crop = webcam_crop
        for cam in (webcam):
            self.cap.append(cv2.VideoCapture(cam))

        for cap in self.cap:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            for i in range(30):
                ret, frm = cap.read()
            
            print('cam init finished')

        self.frame_size = frame_size
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.image_dir = []
        self.image_path = []
        for dir in image_dir:
            self.image_dir.append(dir)
            os.makedirs(dir, exist_ok=True)
            self.image_path.append([])
        self.frame_idx = 0


    def record_frame(self, action_idx=None):
        frames = []
        for cam in self.cap:
            ret, frame = cam.read()            
            if not ret or frame is None:
                print("Warning: Failed to read from one of the webcams.")
                return  # Exit early to avoid crashing
            frames.append(frame)
            

        # if not frame or not ret2:``
        #     print("Warning: Failed to capture from one or both cameras")
        #     return

        def process_frame(frame, idx):
            # target_width = 640
            # target_height = 630
            # h, w, _ = frame.shape
            # x_start = 50 + (w - target_width) // 2 - 60
            # y_start = (h - target_height) // 2 + 50
            cropped = frame[self.webcam_crop[idx][3]:self.webcam_crop[idx][3] + self.webcam_crop[idx][1],self.webcam_crop[idx][2]:self.webcam_crop[idx][2] + self.webcam_crop[idx][0]]
            print(f"cropped:{cropped.shape} frame{frame.shape}")
            resized = cv2.resize(cropped, (480, 480), interpolation=cv2.INTER_LINEAR)
            if action_idx is not None:
                resized = cv2.putText(resized, str(action_idx), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            fname = f"{self.webcam_name[idx]}_frame_{self.frame_idx:05d}.jpg"
            path = os.path.join(self.image_dir[idx], fname)
            cv2.imwrite(path, resized)
            return path
        
        for i in range(len(frames)):
            path = process_frame(frames[i],i)
            self.image_path[i].append(path)

        self.frame_idx += 1

    def reset(self):
        self.frame_idx = 0
        for i in range(len(self.image_path)):
            self.image_path[i] = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--frame_size", type=int, nargs=2, default=(512, 512))
    return parser.parse_args()

# args = parse_args()
save_path = "/home/skapse/workspace/xarm-dataset"
contorller_args = {
    "robot_ip": "192.168.1.198",
    "save_path": "/home/skapse/workspace/xarm-dataset",
    "arm_speed": 1000,
    "gripper_speed": 2000,
    "pos_step_size": 50,
    "rot_step_size": 15,
    "grip_size": 1000,
    "webcam": [4,10],
    "webcam_crop": [[720,720,280,0],[640,640,310,80]],
    "webcam_name": ["wristcam","exo1"],
    "flip_view": False,
    "task_name": "pick up carrot and place in bowl"
}
arm = PlayStationController(**contorller_args)

try:
    time_after_loop = time.process_time()
    frequency = 1/15.0
    while True:
        time_before_loop = time.process_time()
        if time_before_loop - time_after_loop >= frequency:
            arm.controller_listen()
            time_after_loop = time.process_time()
except KeyboardInterrupt:
    print("KeyboardInterrupt received. Stopping.")
except Exception as e:
    print("Error:", e)
finally:
    if arm.save_actions:
        print("Saving and closing HDF5 file...")
        arm.hdf5_file.flush()
        arm.hdf5_file.close()