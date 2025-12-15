import os
import cv2
from PIL import Image

def preprocess_frame(frame, cam_id):
    cam_id = cam_id - 1
    webcam_crop = [[720,720,280,0], [380,370,120,60]]

    cropped = frame[
        webcam_crop[cam_id][3] : webcam_crop[cam_id][3] + webcam_crop[cam_id][1],
        webcam_crop[cam_id][2] : webcam_crop[cam_id][2] + webcam_crop[cam_id][0],
    ]

    return cropped


# ------------------------
# PROCESS THE ENTIRE DATASET
# ------------------------

dataset_root = "/home/skapse/workspace/xarm-dataset/move_near"
cam_id = 2   # your chosen camera id

# Iterate over all trajectory folders
for traj in os.listdir(dataset_root):

    traj_path = os.path.join(dataset_root, traj)
    if not os.path.isdir(traj_path):
        continue

    # Find folders starting with "images_"
    for folder in os.listdir(traj_path):
        if folder.startswith("images_"):
            
            images_folder = os.path.join(traj_path, folder)
            output_folder = images_folder + "_processed"

            os.makedirs(output_folder, exist_ok=True)

            print(f"\n➡️ Processing {images_folder}")

            for filename in os.listdir(images_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

                    in_path = os.path.join(images_folder, filename)
                    out_path = os.path.join(output_folder, filename)

                    img = cv2.imread(in_path)
                    if img is None:
                        print(f"⚠️ Could not read {in_path}")
                        continue

                    processed = preprocess_frame(img, cam_id)
                    cv2.imwrite(out_path, processed)
                    print(f"   ✔ Saved: {out_path}")

print("\n✅ ALL TRAJECTORIES PROCESSED SUCCESSFULLY!")
