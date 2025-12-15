import time
from xarm.wrapper import XArmAPI

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

# #move to pick the object
# arm.set_gripper_position(850, wait=True)
# arm.set_position(*[300, 0, 300, 180, 0, 0], wait=True)
# arm.set_position(*[300, 0, 200, 180, 0, 0], wait=True)
# arm.set_gripper_position(0, wait=True)

# #set tcp payload
# arm.set_tcp_load(0.3, [0, 0, 30])
# arm.set_state(0)

# #move to place the object
# arm.set_position(*[300, 0, 300, 180, 0, 0], wait=True)
# arm.set_position(*[300, -150, 300, 180, 0, 0], wait=True)
# arm.set_position(*[300, -150, 200, 180, 0, 0], wait=True)
# arm.set_gripper_position(850, wait=True)
# arm.set_tcp_load(0, [0, 0, 30])
# arm.set_state(0)
# arm.set_position(*[300, -150, 300, 180, 0, 0], wait=True)


code = arm.set_gripper_mode(0)

code = arm.set_gripper_enable(True)

code = arm.set_gripper_speed(5000)

# arm.set_servo_angle(servo_id=7, angle=-90, is_radian=False)
target_joints = [-25.8,0.7,20.3,2.9,174.6,91.4,175.5]

code = arm.set_servo_angle(angle=target_joints, speed=20, is_radian=False, wait=True)
if code != 0:
    print(f"⚠️ Failed to set joint angles, error code: {code}")

time.sleep(1)

arm.set_gripper_position(600, wait=True)

arm.set_gripper_position(300, wait=True)
