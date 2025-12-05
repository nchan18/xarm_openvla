import os
import shutil

ROOT = "/home/skapse/workspace/xarm-dataset"
OUT_DIR = os.path.join(ROOT, "movenear")

os.makedirs(OUT_DIR, exist_ok=True)

task_folders = sorted([f for f in os.listdir(ROOT) if f.startswith("move_")])

global_idx = 0

for task in task_folders:
    task_path = os.path.join(ROOT, task)

    # Extract object names: move_banana_to_cucumber -> banana_to_cucumber
    object_names = task.replace("move_", "")

    # Get trajectory folders
    trajs = sorted([t for t in os.listdir(task_path) if t.startswith("trajectory")])

    for traj in trajs:
        src = os.path.join(task_path, traj)

        # Build output folder name
        # Use both object_names and original trajectory name
        new_name = f"trajectory{global_idx:01d}"
        dst = os.path.join(OUT_DIR, new_name)
        shutil.copytree(src, dst)
        print(f"Copied {src} -> {dst}")
        global_idx += 1

print(global_idx)
print(":white_check_mark: Done merging with object names + trajectory + global index!")