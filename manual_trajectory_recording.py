import time
from xarm.wrapper import XArmAPI

#arm setup
arm = XArmAPI('192.168.1.198')
time.sleep(0.5)
arm.set_mode(0)
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

#trajectory recording
code = arm.set_mode(2)
code = arm.set_state(0)

arm.set_gripper_enable(True)
while(True):
    print("Open")
    arm.set_gripper_position(300)
    time.sleep(2)
    print("Close")
    arm.set_gripper_position(800)
    time.sleep(2)

# print(code)

# arm.start_record_trajectory()

# print("start moving that thang")
# arm.motion_enable(7)

# time.sleep(5)
# print(arm.get_record_seconds())
# arm.stop_record_trajectory(filename='test2')

# time.sleep(2)

# print("Playing back trajectory")

# arm.set_mode(0)
# arm.set_state(0)
# arm.load_trajectory(filename='test2')
# arm.playback_trajectory()

