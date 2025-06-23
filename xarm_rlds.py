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
    def __init__(self, robot_ip="192.168.1.198", arm_speed=1000, gripper_speed=1000, pos_step_size=50, rot_step_size=5, grip_size=100,task_name=" "):
        self.robot_ip = robot_ip
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.pos_step_size = pos_step_size
        self.rot_step_size = rot_step_size
        self.task_name = task_name
        self.grip_size = grip_size
        self.arm_pos: tuple = None
        self.arm_rot: tuple = None
        self.gripper_pos: int = None
        self.wait = False
        self.save_video = False
        self.arm = XArmAPI(self.robot_ip)
        self.arm_starting_pose = (300, 0, 300, 180, 0, 0)
        self.arm.set_mode(0) # set_position
        self.arm.set_state(0)
        code = self.arm.set_gripper_mode(0)
        print("set gripper mode: location mode, code={}".format(code))
        code = self.arm.set_gripper_enable(True)
        print("set gripper enable, code={}".format(code))
        code = self.arm.set_gripper_speed(self.gripper_speed)
        print("set gripper speed, code={}".format(code))
        self.move_position(self.arm_starting_pose)
        self.arm.set_mode(5) # set catesian velocity
        self.arm.set_state(0)


        self.update_arm_state()

    def update_arm_state(self):
        _, arm_pos = self.arm.get_position(is_radian=False)
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
        delta = self.grip_size
        self.arm.set_gripper_position(850, wait=self.wait)

    def gripper_close(self):
        delta = self.grip_size
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
        webcam=0,
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
        trajectory_num = 0
        os.makedirs(os.path.join(save_path, self.task_name), exist_ok=True)
        while os.path.isdir(os.path.join(save_path, self.task_name, f"trajectory{trajectory_num}")):
            trajectory_num += 1
        self.save_actions = os.path.join(save_path, self.task_name, f"trajectory{trajectory_num}", "action")
        self.save_video = os.path.join(save_path, self.task_name, f"trajectory{trajectory_num}", "video")
        os.makedirs(self.save_actions, exist_ok=True)
        os.makedirs(self.save_video, exist_ok=True)
        self.webcam = webcam
        self.flip_view = flip_view
        if self.save_action:
            self.setup_action_save()
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

        #self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        #self._monitor_thread.daemon = True
        # self.lock = threading.Lock()
        #self._monitor_thread.start()

    # def setup_action_save(self):
    #     with open(self.action_save_path, "w") as f:
    #         f.write("abs_pos, abs_rot, gripper, rel_pos, rel_rot\n")
    #     print(f"Saving actions to {self.action_save_path}")

    # def save_action(self, rel_action):
    #     self.update_arm_state()
    #     rel_pos = rel_action[0:3]
    #     rel_rot = rel_action[3:5]
    #     arm_pos_meters = [i / 1000 for i in self.arm_pos]
    #     arm_rot_rad = [i * math.pi/180.0 for i in self.arm_rot]
    #     gripper_pos_norm = self.gripper_pos/850.0
    #     rel_pos_meters = [i / 1000 for i in self.arm_pos]
    #     rel_rot_rad = [i * math.pi/180.0 for i in rel_rot]
    #     with open(self.action_save_path, "a") as f:
    #         f.write(f"{arm_pos_meters}, {arm_rot_rad}, {gripper_pos_norm}, {rel_pos_meters}, {rel_rot_rad}\n")
    #     if self.save_video:
    #         self.video_recorder.record_frame(action=arm_pos_meters) # for writing action on frame - Dominick

    # def reset_save_file(self):
    #     save_path = datetime.datetime.now().strftime(f"{self.save_root}/actions_%Y-%m-%d_%H-%M-%S") + ".csv"
    #     self.action_save_path = save_path
    #     self.setup_action_save()
    #     if self.save_video:
    #         self.video_recorder.reset()

    def setup_action_save(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.hdf5_path = os.path.join(self.save_actions, f"actions_{timestamp}.hdf5")
        self.hdf5_file = h5py.File(self.hdf5_path, "w")
        self.action_idx = 0

        self.hdf5_file.create_dataset("abs_pos", shape=(0, 3), maxshape=(None, 3), dtype="f")
        self.hdf5_file.create_dataset("abs_rot", shape=(0, 3), maxshape=(None, 3), dtype="f")
        self.hdf5_file.create_dataset("gripper", shape=(0,), maxshape=(None,), dtype="f")
        self.hdf5_file.create_dataset("rel_action", shape=(0, 6), maxshape=(None, 6), dtype="f")
        self.hdf5_file.create_dataset("image_path", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))

        if self.save_video:
            self.video_recorder = VideoRecorder(self.save_video, webcam=self.webcam, frame_size=(1280, 800))


    def save_action(self, rel_action):
        self.update_arm_state()
        abs_pos = np.array([i / 1000 for i in self.arm_pos], dtype=np.float32)
        abs_rot = np.array([i * math.pi / 180 for i in self.arm_rot], dtype=np.float32)
        gripper = np.array(self.gripper_pos / 850.0, dtype=np.float32)
        rel_action = np.array(rel_action, dtype=np.float32)

        for key, data in zip(["abs_pos", "abs_rot", "gripper", "rel_action"], [abs_pos, abs_rot, gripper, rel_action]):
            dset = self.hdf5_file[key]
            new_shape = (self.action_idx + 1,) + dset.shape[1:]
            dset.resize(new_shape)
            dset[self.action_idx] = data

        # Save frame and path
        if self.save_video:
            self.video_recorder.record_frame(self.action_idx)
            image_path_dset = self.hdf5_file["image_path"]
            image_path_dset.resize(self.action_idx + 1, axis=0)
            image_path_dset[self.action_idx] = self.video_recorder.image_paths[-1]

        self.action_idx += 1    

    def reset_save_file(self):
        self.hdf5_file.close()
        self.setup_action_save()
        if self.save_video:
            self.video_recorder.reset()
            
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

        if abs(RightJoystickY) > self.threshold:
            if RightJoystickY > 0:
                action[2] = -self.pos_step_size
                print("z_minus")
            else:
                action[2] = self.pos_step_size
                print("z_plus")

        if abs(DPadX) > self.threshold:
            if DPadX > 0:
                action[3] = -self.pos_step_size
                print("a_minus")
            else:
                action[3] = self.pos_step_size
                print("a_plus")

        if abs(DPadY) > self.threshold:
            if DPadY > 0:
                action[4] = -self.pos_step_size
                print("b_minus")
            else:
                action[4] = self.pos_step_size
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

        self.move(action)
        self.save_action(tuple(action))

class VideoRecorder:
    def __init__(self, image_dir, frame_size=(1280, 800), webcam=0):
        self.cap = cv2.VideoCapture(webcam)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame_size = frame_size
        self.time_stamp = time.strftime("%Y%m%d-%H%M%S")
        print(self.time_stamp)
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)
        self.frame_idx = 0

        # Store image paths
        self.image_paths = []

    def record_frame(self, action_idx=None):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, self.frame_size)
            filename = f"frame_{self.frame_idx:05d}.jpg"
            filepath = os.path.join(self.image_dir,filename)
            print(filepath)

            # Optionally draw info
            if action_idx is not None:
                frame = cv2.putText(frame, str(action_idx), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            # Save image
            cv2.imwrite(filepath, frame)
            self.image_paths.append(filepath)
            self.frame_idx += 1
        else:
            print("Warning: Failed to capture frame")

    def close(self):
        self.cap.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--frame_size", type=int, nargs=2, default=(512, 512))
    return parser.parse_args()

# args = parse_args()
save_path = "/workspace/xarm-dataset"
contorller_args = {
    "robot_ip": "192.168.1.198",
    "save_path": "/workspace/xarm-dataset",
    "arm_speed": 1000,
    "gripper_speed": 10000,
    "pos_step_size": 50,
    "rot_step_size": 5,
    "grip_size": 1000,
    "webcam": 4 ,
    "flip_view": False,
    "task_name": "place the blue box on plate"
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
