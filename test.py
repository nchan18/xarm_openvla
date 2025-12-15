import numpy as np
from scipy.spatial.transform import Rotation as R

def rmat_to_euler(rot_mat, degrees=False):
    euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=degrees)
    return euler


def euler_to_rmat(euler, degrees=False):
    return R.from_euler("xyz", euler, degrees=degrees).as_matrix()


def change_pose_frame(pose, frame, degrees=False):
    R_frame = euler_to_rmat(frame[3:6], degrees=degrees)
    R_pose = euler_to_rmat(pose[3:6], degrees=degrees)
    t_frame, t_pose = frame[:3], pose[:3]
    euler_new = rmat_to_euler(R_frame @ R_pose, degrees=degrees)
    t_new = R_frame @ t_pose + t_frame
    result = np.concatenate([t_new, euler_new])
    return result

R_invert_x = np.array([
    [-1,  0,  0],  # X is flipped
    [ 0,  0,  1],  # Y is unchanged
    [ 0,  1,  0]   # Z is unchanged
])


euler_invert_x = R.from_matrix(R_invert_x).as_euler('xyz', degrees=True)
frame_invert_x = np.array([0, 0, 0, *euler_invert_x])  # no translation
print(frame_invert_x)
pose = np.array([1, 2, 3, 0, 0, 0])  # example pose
new_pose = change_pose_frame(pose, frame_invert_x, degrees=True)
print(new_pose)