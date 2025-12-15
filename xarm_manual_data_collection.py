import time
from xarm.wrapper import XArmAPI
import cv2
from enum import Enum

class arm_modes(Enum):
    POSITION_CONTROL = 0
    JOINT_TEACHING = 2


#Arm setup
arm = XArmAPI('192.168.1.198')
time.sleep(0.5)
arm.set_mode(2)
arm.set_state(0)
if not arm.arm._enable_report or arm.arm._stream_type != 'socket':
    arm.arm.get_err_warn_code()
    arm.arm.get_state()
if arm.arm._warn_code != 0:
    arm.arm.clean_warn()
if arm.arm._error_code != 0:
    arm.arm.clean_error()
    arm.arm.motion_enable(enable=True, servo_id=8)
    arm.arm.set_state(0)
if not arm.arm._is_ready:
    arm.arm.motion_enable(enable=True, servo_id=8)
    arm.arm.set_state(state=0)

code = arm.set_gripper_mode(0)

code = arm.set_gripper_enable(True)

code = arm.set_gripper_speed(5000)

arm.start_record_trajectory()
print("recording trajectory, press ctrl-c to stop")
try:
    while True:
        pass
except KeyboardInterrupt:
    pass


arm.stop_record_trajectory()
print(arm.get_record_seconds())
arm.save_record_trajectory(filename='test1')
print("Trajectory saved")

print("Playing back trajectory")
code = arm.set_gripper_mode(2)
arm.playback_trajectory(filename='test1', times=1, wait=True)

print("Playback done")

