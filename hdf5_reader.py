
import h5py
import matplotlib as plt
import cv2
with h5py.File("/workspace/xarm-dataset/actions/actions_2025-06-23_16-24-55.hdf5", "r") as f:
    for k in f:
        print(f"{k}: shape = {f[k].shape}")
        print(f"{k}: sample = {f[k][:1]}")
frame = f["video_frames"][0]  # first frame
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()
