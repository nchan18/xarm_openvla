import argparse
import datetime
import pygame
import math
import threading
import cv2
import time, os
# from evdev import InputDevice, categorize, ecodes
from xarm.wrapper import XArmAPI


class xArm7GripperEnv:
    def __init__(self, robot_ip="192.168.1.198", arm_speed=1000, gripper_speed=1000, pos_step_size=50, rot_step_size=5, grip_size=100):
        self.robot_ip = robot_ip
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.pos_step_size = pos_step_size
        self.rot_step_size = rot_step_size
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
        save_actions="",
        save_video="",
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

        self.save_actions = len(save_actions) > 0
        self.save_root = save_actions
        save_path = datetime.datetime.now().strftime(f"{self.save_root}/actions_%Y-%m-%d_%H-%M-%S") + ".csv"
        self.action_save_path = save_path
        if self.save_actions:
            self.setup_action_save()
        self.save_video = len(save_video) > 0
        self.webcam = webcam
        self.flip_view = flip_view
        self.video_recorder = VideoRecorder(save_video, webcam=webcam, fps=5.0) if self.save_video else None
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

    def setup_action_save(self):
        with open(self.action_save_path, "w") as f:
            f.write("abs_pos, abs_rot, gripper, rel_pos, rel_rot\n")
        print(f"Saving actions to {self.action_save_path}")

    def save_action(self, rel_action):
        self.update_arm_state()
        rel_pos = rel_action[0:3]
        rel_rot = rel_action[3:5]
        arm_pos_meters = [i / 1000 for i in self.arm_pos]
        arm_rot_rad = [i * math.pi/180.0 for i in self.arm_rot]
        gripper_pos_norm = self.gripper_pos/850.0
        rel_pos_meters = [i / 1000 for i in self.arm_pos]
        rel_rot_rad = [i * math.pi/180.0 for i in rel_rot]
        with open(self.action_save_path, "a") as f:
            f.write(f"{arm_pos_meters}, {arm_rot_rad}, {gripper_pos_norm}, {rel_pos_meters}, {rel_rot_rad}\n")
        if self.save_video:
            self.video_recorder.record_frame(action=arm_pos_meters) # for writing action on frame - Dominick

    def reset_save_file(self):
        save_path = datetime.datetime.now().strftime(f"{self.save_root}/actions_%Y-%m-%d_%H-%M-%S") + ".csv"
        self.action_save_path = save_path
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
    def __init__(self, video_dir, frame_size=(1280,800), fps=3.0, webcam=0, idle_frames=50):
        self.dir = video_dir
        self.video_path = datetime.datetime.now().strftime(f"{video_dir}/video_%Y-%m-%d_%H-%M-%S.mp4")
        self.frame_size = frame_size
        self.fps = fps
        self.webcam = webcam
        self.idle_frames = idle_frames
        self.setup()

    def setup(self):
        self.cap = cv2.VideoCapture(self.webcam)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])

        # for i in range(self.idle_frames): # give some time for the camera to adjust to lighting
        #     ret, frame = self.cap.read()
        print('ready')

        self.writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.frame_size)

    def record_frame(self, action=None):
        ret, frame = self.cap.read()

        if ret:
            frame = cv2.resize(frame, self.frame_size)
            frame = cv2.putText(frame, str(action), (5,200), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
            self.writer.write(frame)

            # Below is hardcoded for frame writing
            # im_num = len(os.listdir('/home/nchandr8/Desktop/temp_images/'))
            # cv2.imwrite(f'/home/nchandr8/Desktop/temp_images/{im_num}.jpg', frame)
        else:
            print("Warning: Failed to capture frame")

    def reset(self):
        self.close()
        self.video_path = datetime.datetime.now().strftime(f"{self.dir}/video_%Y-%m-%d_%H-%M-%S.mp4")
        self.setup()

    def close(self):
        self.cap.release()
        self.writer.release()


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
    "arm_speed": 1000,
    "gripper_speed": 10000,
    "pos_step_size": 50,
    "rot_step_size": 5,
    "grip_size": 1000,
    "save_actions": f"{save_path}/actions",
    "save_video": f"{save_path}/videos",
    "webcam": 4 ,
    "flip_view": False,
}
arm = PlayStationController(**contorller_args)

time_after_loop = time.process_time() # initalisation
frequency = 1/15.0
while True:
    time_before_loop = time.process_time()
    if time_before_loop - time_after_loop >= frequency:
        try:
            arm.controller_listen()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
        time_after_loop = time.process_time()
