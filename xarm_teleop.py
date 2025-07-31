from inputs import get_gamepad
import math
import threading
import time
from xarm.wrapper import XArmAPI



class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()


    def read(self): # return the buttons/triggers that you care about in this methode
        x = self.LeftJoystickX
        y = self.LeftJoystickY
        a = self.A
        b = self.X # b=1, x=2
        rb = self.RightBumper
        return [x, y, a, b, rb]


    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_X':
                    self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RY':
                    self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_RX':
                    self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
                elif event.code == 'BTN_TL':
                    self.LeftBumper = event.state
                elif event.code == 'BTN_TR':
                    self.RightBumper = event.state
                elif event.code == 'BTN_SOUTH':
                    self.A = event.state
                elif event.code == 'BTN_NORTH':
                    self.Y = event.state #previously switched with X
                elif event.code == 'BTN_WEST':
                    self.X = event.state #previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.B = event.state
                elif event.code == 'BTN_THUMBL':
                    self.LeftThumb = event.state
                elif event.code == 'BTN_THUMBR':
                    self.RightThumb = event.state
                elif event.code == 'BTN_SELECT':
                    self.Back = event.state
                elif event.code == 'BTN_START':
                    self.Start = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY1':
                    self.LeftDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY2':
                    self.RightDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY3':
                    self.UpDPad = event.state
                elif event.code == 'BTN_TRIGGER_HAPPY4':
                    self.DownDPad = event.state




if __name__ == '__main__':
    joy = XboxController()

    step_size = 25 # mm
    step_size_rot = 5 #degrees
    threshold = 0.2
    current_pos = [300,0,300]
    current_rot = [180,0,0]
    current_gripper_pos = 850
    
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

    arm.set_position(*[300, 0, 300, 180, 0, 0], wait=True)
    code = arm.set_gripper_mode(0)

    code = arm.set_gripper_enable(True)

    code = arm.set_gripper_speed(5000)
    arm.set_gripper_position(850, wait=True)

    while True:
        if abs(joy.LeftJoystickY) > threshold:
            current_pos[0]+= joy.LeftJoystickY * step_size
        if abs(joy.LeftJoystickX) > threshold:
            current_pos[1]+= joy.LeftJoystickX * step_size
        if abs(joy.RightJoystickY) > threshold:
            current_pos[2]+= joy.RightJoystickY * step_size
        if abs(joy.RightTrigger-joy.LeftTrigger) > threshold:
            current_rot[0]+= (joy.RightTrigger-joy.LeftTrigger) * step_size_rot
        if abs(joy.UpDPad) > threshold or abs(joy.DownDPad) > threshold :
            current_rot[1]+= (joy.UpDPad - joy.DownDPad) * step_size_rot
        current_rot[2]+= (joy.RightBumper-joy.LeftBumper) * step_size_rot
        current_gripper_pos += (joy.A-joy.X) * step_size*10
        arm.set_position(*[current_pos[0],current_pos[1],current_pos[2],current_rot[0],current_rot[1],current_rot[2]], wait=False)
        # arm.set_position(300,100,200,180,0,0, wait=False)
        arm.set_gripper_position(current_gripper_pos, wait=False)