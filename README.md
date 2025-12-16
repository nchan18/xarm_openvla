# xarm_openvla

## xarm_rlds.py — usage and controller reference

This README explains how to run and use `xarm_rlds.py` to teleoperate the xArm7 with either a PlayStation controller (via pygame) or a VR controller (via an OculusReader). It also documents the controller button/axis mappings and recommended setup steps.

### Quick summary
- Script: `xarm_rlds.py` — main control loop that reads a controller (PlayStation or VR), sends velocity/position/gripper commands to an xArm7, and records camera frames and HDF5 dataset entries.
- Two controller modes:
  - `playstation` — standard gamepad using `pygame.joystick`.
  - `vr` — Oculus/VR controller using `oculus_reader.reader.OculusReader`.

### Important safety notes
- This script talks to a real robot by default (xArm IP is configured in the script). Keep an emergency stop or power cut within reach. Use `mode: sim` in config to avoid sending commands to real hardware.
- Verify the `robot_ip` and network connectivity before running.
- Ensure cameras are safe and scene is clear before enabling robot movement.

### Dependencies
Make sure the following Python packages are installed (your environment may already provide some):

- python3
- numpy
- scipy
- opencv-python (cv2)
- pygame
- h5py
- pyrealsense2 (only if using RealSense cameras)
- xArm SDK (`xarm` package providing `xarm.wrapper.XArmAPI`)
- oculus_reader (provides `OculusReader`) — project-local or pip package depending on your setup

Install with pip (example):

```bash
pip install numpy scipy opencv-python pygame h5py
# pyrealsense2, xarm SDK and oculus_reader may require platform-specific install steps
```

### How to run

By default the script uses the internal `default_config()` defined in the file and runs in real mode. To run the file directly:

```bash
python3 xarm_rlds.py
```

If you prefer to use command-line arguments, uncomment the `config = parse_args()` line in the `if __name__ == "__main__":` block and run with options. Example (after enabling CLI):

```bash
python3 xarm_rlds.py --controller_type playstation --robot_ip 192.168.1.198 --save_dir ~/workspace/xarm-dataset
```

Notes:
- `--controller_type` choices: `playstation` or `vr`.
- `save_dir` is where dataset/hdf5 and camera images will be saved.
- `mode` can be set to `sim` to avoid sending commands to the real arm (requires ROS2 sim node configuration).

### Changing the task (task_name & task_type)

There are two distinct task-related settings in `xarm_rlds.py` you may want to change:

- `task_name` — a human-readable name used when saving datasets. It appears in the default save path and HDF5 filename.
- `task_type` — alters starting poses and behavior. Current code supports at least `"pick_and_place"` and `"pouring"` (the latter enables special pouring logic in the VR controller).

How to change the task:

1. Edit `default_config()` inside `xarm_rlds.py` (quickest for development):

   - Change the `task_name` string to a descriptive name (e.g., `"stack_cups"` or `"place_block"`).
   - Change `task_type` to either `"pick_and_place"` or `"pouring"` depending on the behavior you want.

   Example:

   ```python
   config = {
       "controller_type": "playstation",
       "task_name": "stack_the_cups",
       "task_type": "pick_and_place",
       ...
   }
   ```

2. Use command-line arguments (recommended if you enable CLI parsing):

   - Uncomment `config = parse_args()` in the `__main__` block. Then run:

   ```bash
   python3 xarm_rlds.py --task_name "stack the cups" --task_type pick_and_place
   ```

   - Supported `--task_type` values: `pick_and_place`, `pouring`.

3. Adjust task-specific starting poses and behavior (optional, advanced):

   - `xArm7GripperEnv` defines `arm_starting_pose_pick_and_place` and `arm_starting_pose_pouring`. If you need a different start pose for a task, edit these tuples inside the class. Example:

   ```python
   self.arm_starting_pose_pick_and_place = (x, y, z, roll, pitch, yaw)
   ```

   - For complex task changes (new task types), you may also need to add logic where `task_type` is checked (e.g., in `move_to_starting_position()` or in controller code) to implement task-specific behavior.

Save path and naming notes:

- The script builds a save path using `save_dir` and `task_name`. By default `default_config()` will also create a unique `trajectoryN` folder if previous runs exist. Changing `task_name` will change the folder and HDF5 file naming.

### High-level runtime behavior
- The `ControlSystem` instantiates the `xArm7GripperEnv`, the chosen controller wrapper, camera manager(s), and an `HDF5Recorder`.
- In a loop (robot update rate ~15 Hz):
  - Read/update current robot/cartesian pose and gripper position.
  - Call controller.get_action(state_dict) -> returns (velocity, gripper_command, pour_flag).
  - Send velocity to the robot with arm.vc_set_cartesian_velocity(...) and handle gripper latch commands.
  - Record camera frames and append a dataset record.

### PlayStation controller mapping (pygame)

The `PlayStationController` class (pygame) reads axes/buttons and converts them to a 6-DOF velocity + gripper command.

- Axes
  - Left joystick vertical (axis 1): controls X linear velocity (forward/back). A nonzero axis moves along robot X. The code uses negative sign to map forward/back correctly.
  - Left joystick horizontal (axis 0): controls Y linear velocity (left/right).
  - Right joystick vertical (axis 4): controls Z linear velocity (up/down).
  - D-Pad (hat 0) X (left/right): toggles rotation axis 0 (roll) by +/- `rot_step_size` degrees/s.
  - D-Pad (hat 0) Y (up/down): toggles rotation axis 1 (pitch) by +/- `rot_step_size` degrees/s.
  - Left trigger (axis 2) pressed > 0.5: rotates yaw positively by `rot_step_size` deg/s.
  - Right trigger (axis 5) pressed > 0.5: rotates yaw negatively by `rot_step_size` deg/s.

- Buttons
  - X (pygame button 0): Open gripper -> `gripper_command = 'open'`.
  - SQUARE (pygame button 3): Close gripper -> `gripper_command = 'close'`.

Notes:
- `pos_step_size` (mm/s) and `rot_step_size` (deg/s) are configurable in the `default_config()` or via CLI (when enabled).
- A joystick deadzone threshold (`threshold`) is used to ignore small axis noise.

### VR controller mapping (OculusReader)

The `VRController` uses `OculusReader()` producing transformation matrices and a `buttons` mapping. The code expects certain button keys and uses them as follows.

Primary concepts:
- The VR pipeline computes a VR controller pose (3D position + quaternion) transformed into the robot/environment frame. The difference between the VR pose and a stored `vr_origin` is used to compute a target offset.
- When movement is enabled and the origin is set, the code runs two PID controllers to convert pose error into linear and angular velocities. These velocities are scaled and sent to the robot.

Buttons and their effects (as implemented in the code):
- movement toggle (controller-specific): controller ID + `G` (e.g., `RG` or `LG`) — toggles `movement_enabled`. When movement is enabled the controller drives the robot.
- reset orientation (controller-specific): controller ID + `J` (e.g., `RJ` or `LJ`) — when pressed, the controller's current orientation is used to redefine "forward"; this affects the VR-to-global transform.
- Triggers (`rightTrig` or `leftTrig`): used for gripper control. The numeric trigger value is read and interpreted:
  - trigger < 0.1 -> `close`
  - trigger > 0.9 -> `open`
  - otherwise -> `none`
- Face buttons `A`, `B`, `X`, `Y` (global keys in VR controller state):
  - `A` (when using the right controller): zeros rotational velocity (sets rot_vel_list = [0,0,0]).
  - `B` (right controller): zeros linear velocity (sets lin_vel_mm = [0,0,0]).
  - `X` (any controller): when `task_type == 'pouring'` this sets `pour = True` and zeroes velocities (used to start pouring action).
  - `Y` (any controller): when `task_type == 'pouring'` this sets `pour = False` and zeroes velocities (used to stop pouring action).

Other useful behaviors and notes:
- Reset origin: when the script detects `reset_origin` it stores the current robot pose and the current VR controller pose as the origin pair; subsequent motions are offsets from that origin. `reset_origin` is toggled when the movement button is pressed/released.
- Reset orientation: pressing the `J` button will reset how the VR orientation maps into the robot frame (useful to re-zero yaw/forward direction).
- `get_info()` returns a small status dict showing which face buttons are active and whether movement is currently enabled.

Be aware: the actual dictionary keys coming from your `OculusReader` implementation may have slightly different names. The code expects keys like `A`, `B`, `X`, `Y`, `rightTrig`, `leftTrig`, and controller-specific `RG`/`RJ` or `LG`/`LJ` for the movement/reset mapping. If your `OculusReader.get_transformations_and_buttons()` returns different names, update the code to match those names.

### Tuning and configuration
- PID controllers for VR are created inside `VRController` with initial gains. Modify `pos_pid` and `rot_pid` parameters there if motion is too slow or oscillatory.
- `max_lin_vel`, `max_rot_vel`, and `max_gripper_vel` in `VRController` limit commanded velocities.
- Workspace bounds and starting poses are defined in `xArm7GripperEnv` and can be adjusted to fit your robot setup.

### Troubleshooting
- No joystick detected for PlayStation mode: ensure your controller is paired/connected and visible to the OS; test with `jstest` or a small pygame joystick listing script.
- VR mode not responding: ensure `OculusReader` is running properly and returns pose/button data. Check the `oculus_reader` module and run its sample reader if available.
- Robot does not move: verify `robot_ip`, network routing, and xArm SDK connectivity. Check for exceptions printed to the console.

### Example workflow

1. Connect PlayStation controller or start your VR runtime.
2. Edit `default_config()` to change `robot_ip`, `save_dir`, `controller_type`, or desired camera IDs (or enable CLI parsing).
3. Run `python3 xarm_rlds.py`.
4. Use the controller mappings above to drive the arm and open/close the gripper. Press movement toggle/zeroing buttons to calibrate origin and orientation as needed.

### Where to look next in the code
- `xarm_rlds.py` contains the following key classes and functions:
  - `xArm7GripperEnv` — robot wrapper and gripper latch logic
  - `PlayStationController` — pygame-based controller mapping
  - `VRController` — OculusReader-based controller mapping and PID controllers
  - `CameraManager`, `RealSenseRecorder`, `WebcamRecorder` — camera recording utilities
  - `HDF5Recorder` — dataset creation and saving logic

---
Last updated: December 15, 2025
