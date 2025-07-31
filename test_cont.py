import pygame
import time
import math
import threading
from xarm.wrapper import XArmAPI
pygame.init()

# This is a simple class that will help us print to the screen.
# It has nothing to do with the joysticks, just outputting the
# information.

class PlayStationController(object):
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
        self.X = 0
        self.SQUARE = 0
        self.TRIANGLE = 0
        self.O = 0
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


    def _monitor_controller(self):
        while True:
            for event in pygame.event.get():
                joysticks = {}
                # Handle hotplugging
                if event.type == pygame.JOYDEVICEADDED:
                    # This event will be generated when the program starts for every
                    # joystick, filling up the list without needing to create them manually.
                    joy = pygame.joystick.Joystick(event.device_index)
                    joysticks[joy.get_instance_id()] = joy
                if event.type == pygame.JOYDEVICEREMOVED:
                    del joysticks[event.instance_id]

                self.X = int(joy.get_button(0))
                self.SQUARE = int(joy.get_button(3))
                self.TRIANGLE = int(joy.get_button(2))
                self.O = int(joy.get_button(1))
                self.LeftBumper = int(joy.get_button(4))
                self.RightBumper = int(joy.get_button(5))
                self.LeftJoystickX = joy.get_axis(0)
                self.LeftJoystickY = joy.get_axis(1)
                self.RightJoystickX = joy.get_axis(3)
                self.RightJoystickY = joy.get_axis(4)
                self.LeftTrigger = joy.get_axis(2)
                self.RightTrigger = joy.get_axis(5)

                dpad = joy.get_hat(0)
                if dpad [0] == -1 :
                    self.LeftDPad = 1
                    self.RightDPad = 0
                elif dpad [0] == 1 :
                    self.LeftDPad = 0
                    self.RightDPad = 1
                else :
                    self.LeftDPad = 0
                    self.RightDPad = 0

                if dpad [1] == -1 :
                    self.UpDPad = 1
                    self.DownDPad = 0
                elif dpad [1] == 1 :
                    self.UpDPad = 0
                    self.DownDPad = 1
                else :
                    self.UpDPad = 0
                    self.DownDPad = 0



def main():
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

    
    joy = PlayStationController()
    while True:
        if abs(joy.LeftJoystickY) > threshold:
            current_pos[0]+= joy.LeftJoystickY * step_size *-1
        if abs(joy.LeftJoystickX) > threshold:
            current_pos[1]+= joy.LeftJoystickX * step_size *-1
        if abs(joy.RightJoystickY) > threshold:
            current_pos[2]+= joy.RightJoystickY * step_size *-1
        if abs(joy.LeftDPad-joy.RightDPad) > threshold:
            current_rot[0]+= (joy.LeftDPad-joy.RightDPad) * step_size_rot
        if abs(joy.UpDPad) > threshold or abs(joy.DownDPad) > threshold :
            current_rot[1]+= (joy.UpDPad - joy.DownDPad) * step_size_rot
        current_rot[2]+= (joy.LeftTrigger-joy.RightTrigger) * step_size_rot
        current_gripper_pos += (joy.X-joy.SQUARE) * step_size*40
        arm.set_position(*[current_pos[0],current_pos[1],current_pos[2],current_rot[0],current_rot[1],current_rot[2]], wait=False)
        # arm.set_position(300,100,200,180,0,0, wait=False)
        arm.set_gripper_position(current_gripper_pos, wait=False)
    # Set the width and height of the screen (width, height), and name the window.


if __name__ == "__main__":
    main()
    # If you forget this line, the program will 'hang'
    # on exit if running from IDLE.
    pygame.quit()