from xarm.wrapper import XArmAPI
import h5py
import numpy as np
import time
import os

class xArm7GripperEnv:
    def __init__(self, robot_ip="192.168.1.198", arm_speed=1000, gripper_speed=2000, task_name="default_task", task_type="pick_and_place",
                 max_lin_speed_mm=50, max_rot_speed_deg=15, workspace_half_extent_mm=250):
        self.robot_ip = robot_ip
        self.arm_speed = arm_speed
        self.gripper_speed = gripper_speed
        self.task_name = task_name
        self.task_type = task_type
        self.arm_pos = None
        self.arm_rot = None
        self.previous_arm_pos = None
        self.gripper_pos = None
        self.wait = False

        # Gripper latch target: None => no target, 1 => open target, 0 => close target
        self.gripper_target = None
        self.gripper_tolerance = 5.0  # hardware units tolerance

        # Limits & workspace
        self.max_lin_speed_mm = max_lin_speed_mm
        self.max_rot_speed_deg = max_rot_speed_deg
        self.workspace_half_extent_mm = workspace_half_extent_mm

        self.arm = XArmAPI(self.robot_ip)
        # Set a sensible starting pose (x,y,z,roll,pitch,yaw) in mm/degrees
        self.arm_starting_pose_pick_and_place = (300, 0, 200, 180, 0, 0)
        self.arm_starting_pose_pouring = (486,-11,166,53,-83,128)
        tcp_offset = [0, 0, 240, 0, 0, 0]
        self.arm.set_tcp_offset(tcp_offset)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(self.gripper_speed)

        # Workspace center is the starting pose position
        self.workspace_center = np.array(self.arm_starting_pose_pick_and_place[:3], dtype=float)
        self.workspace_min = self.workspace_center - self.workspace_half_extent_mm
        self.workspace_max = self.workspace_center + self.workspace_half_extent_mm

        self.move_to_starting_position()
        self.update_arm_state()

    def move_to_starting_position(self):
        self.arm.set_mode(0)
        self.arm.set_state(0)
        if self.task_type == "pick_and_place":
            self.move_position(self.arm_starting_pose_pick_and_place)
        if self.task_type == "pouring":
            # self.arm.set_servo_angle_j([-25.8,0.7,20.3,2.9,174.6,91.4],speed=20,is_radian=False)
            target_joints = [-25.8,0.7,20.3,2.9,174.6,91.4,175.5]
            code = self.arm.set_servo_angle(angle=target_joints, speed=20, is_radian=False, wait=True)
        # set initial gripper target to open
        self.gripper_target = 1
        self.apply_gripper_target()
        self.arm.set_mode(5)
        self.arm.set_state(0)

    def update_arm_state(self):
        _, arm_pos = self.arm.get_position(is_radian=False)
        self.previous_arm_pos = self.arm_pos
        self.arm_pos = tuple(arm_pos[:3])  # mm
        self.arm_rot = tuple(arm_pos[3:])  # deg
        _, gripper_pos = self.arm.get_gripper_position()
        self.gripper_pos = gripper_pos
        # If we have a gripper target, keep applying until reached
        if self.gripper_target is not None:
            self.apply_gripper_target()

    def _clamp_to_workspace(self, pos_mm):
        pos = np.array(pos_mm, dtype=float)
        clamped = np.minimum(np.maximum(pos, self.workspace_min), self.workspace_max)
        return tuple(clamped.tolist())

    def move_position(self, action):
        x, y, z = action[0], action[1], action[2]
        x_c, y_c, z_c = self._clamp_to_workspace((x, y, z))
        clamped_action = (x_c, y_c, z_c, action[3], action[4], action[5])
        self.arm.set_position(x=clamped_action[0], y=clamped_action[1], z=clamped_action[2],
                              roll=clamped_action[3], pitch=clamped_action[4], yaw=clamped_action[5],
                              relative=False, wait=True, speed=100)

    def move(self, action, dt=0.005):
        if action is None or np.allclose(action, [0, 0, 0, 0, 0, 0]):
            self.arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])

        lin = np.array(action[:3], dtype=float)
        rot = np.array(action[3:], dtype=float)

        lin = np.clip(lin, -self.max_lin_speed_mm, self.max_lin_speed_mm)
        rot = np.clip(rot, -self.max_rot_speed_deg, self.max_rot_speed_deg)

        if self.arm_pos is None:
            self.update_arm_state()
        current_pos = np.array(self.arm_pos, dtype=float)
        pred_pos = current_pos + lin * dt  # mm

        # for i in range(3):
        #     if pred_pos[i] < self.workspace_min[i] or pred_pos[i] > self.workspace_max[i]:
        #         lin[i] = 0.0

        safe_action = np.concatenate([lin, rot]).tolist()
        # print(safe_action)
        if any(abs(v) > 1e-6 for v in safe_action):
            # non-blocking velocity API call
            self.arm.vc_set_cartesian_velocity(safe_action)
            # self.arm.vc_set_cartesian_velocity(safe_action[:3]+[0,0,0])

    def move_joint(self, id, angle):
        self.arm.set_mode(0)
        self.arm.set_state(0)
        print("setting joint", id, "to angle", angle)
        self.arm.set_servo_angle(servo_id=id,angle=angle,is_radian=False, speed=20, wait=True)
        self.arm.set_mode(5)
        self.arm.set_state(0)
    
    def apply_gripper_target(self):
        """
        Sends gripper set commands repeatedly until the hardware reports it's at target.
        gripper hardware values: 0 (closed) .. 850 (open) used in earlier code.
        self.gripper_target values: 1 -> open (850), 0 -> closed (0)
        """
        if self.gripper_target is None:
            return

        target_val_hw = 850 if self.gripper_target == 1 else 0
        try:
            # Always issue non-blocking set until it's within tolerance
            self.arm.set_gripper_position(target_val_hw, wait=False)
            # If gripper_pos is available, check if reached
            if self.gripper_pos is not None:
                # gripper_pos might be reported as a tuple or scalar; handle both
                cur = self.gripper_pos[0] if isinstance(self.gripper_pos, (list, tuple)) else self.gripper_pos
                if abs(cur - target_val_hw) <= self.gripper_tolerance:
                    # target reached; clear target so we don't continually send redundant commands
                    self.gripper_target = None
        except Exception as e:
            print(f"[xArm7GripperEnv] apply_gripper_target error: {e}")

    def gripper_open(self):
        # set a persistent open target
        self.gripper_target = 1
        self.apply_gripper_target()
        print("opened gripper")

    def gripper_close(self):
        # set a persistent close target
        self.gripper_target = 0
        self.apply_gripper_target()
        print("closed gripper")

def hdf5_to_absolute_position(hdf5_file_path, output_file_path):
    with h5py.File(hdf5_file_path, 'r') as h5_file:
        actions = h5_file['actions'][:]
        depth_images = h5_file['depth_image_paths_exo1'][:]
        color_images = h5_file['color_image_paths_exo1'][:]
        time_step = h5_file['timestamps'][:]

    h5file = h5py.File(output_file_path, "w")
    
    env = xArm7GripperEnv()
    absolute_pos = []
    absolute_rot = []
    joint_angles = []
    # Move each action with cycle time of 1/15 seconds
    prev_gripper_action = 1.0
    cycle_time = 1.0 / 15.0
    for x , action in enumerate(actions):
        # print(f"Processing step {x+1}/{len(actions)}")
        loop_start = time.time()
        env.move(action[:6], dt=cycle_time)
        
        if action[6] == 1.0:
            env.gripper_open()
            prev_gripper_action = 1.0
        elif action[6] == 0.0:
            env.gripper_close()
            prev_gripper_action = 0.0

        if action[6] == -1.0:
            action[6] = prev_gripper_action
     

        env.update_arm_state()
        absolute_pos.append(env.arm_pos)
        absolute_rot.append(env.arm_rot)
        _, joints = env.arm.get_servo_angle(is_radian=False)
        joint_angles.append(joints)
        elapsed = time.time() - loop_start
        to_sleep = cycle_time - elapsed
        if to_sleep > 0:
            # print(f"Processing step {x+1}/{len(actions)}")
            time.sleep(to_sleep)
    h5file.create_dataset("abs_jointstate",data=joint_angles, dtype=np.float32)
    h5file.create_dataset("abs_pos",data=np.array(absolute_pos), dtype=np.float32)
    h5file.create_dataset("abs_rot",data=np.array(absolute_rot), dtype=np.float32)
    h5file.create_dataset("gripper",data=np.array(actions[:, 6]), dtype=np.float32)
    h5file.create_dataset("color_image_path_exo1",data=color_images, dtype=h5py.string_dtype(encoding='utf-8'))
    h5file.create_dataset("depth_image_path_exo1",data=depth_images, dtype=h5py.string_dtype(encoding='utf-8'))
    h5file.create_dataset("timestamps",data=time_step, dtype=np.float32)

    np.savez(output_file_path, absolute_pos=np.array(absolute_pos), absolute_rot=np.array(absolute_rot), joint_angles=np.array(joint_angles))
    
    print(f"Saved absolute positions to {output_file_path}")
if __name__ == "__main__":
    for i in range(0,30):
        parent_dir = f"../xarm-dataset/rotate_carrot_180_degrees/trajectory{i}/"
        #find the hdf5 file in the parent_dir
        hdf5_file_path = None
        for file in os.listdir(parent_dir):
            if file.endswith(".hdf5") and file.startswith("rotate_"):
                hdf5_file_path = os.path.join(parent_dir, file)
            else :
                continue    
        # hdf5_file_path = f"../xarm-dataset/fold_the_towel_twice/trajectory{i}/fold_the_towel_twice_20251119_204543.hdf5"
        output_file_path = f"../xarm-dataset/rotate_carrot_180_degrees/trajectory{i}/updated_actions_1_0.hdf5"
        hdf5_to_absolute_position(hdf5_file_path, output_file_path)
