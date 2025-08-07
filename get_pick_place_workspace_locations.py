import os
import random
import shutil

random.seed(1000)
locations = list(range(1, 10))  # Locations 1 to 9
vegetables  = ["carrot" , "eggplant" , "tomato" , "banana" , "corn", "green pepper" , "strawberry" , "orange" , "grape" , "green apple"]
pick_place_pairs = []

while len(pick_place_pairs) < 10:
    pick = random.choice(locations)
    place = random.choice(locations)
    if pick != place:
        pair = (pick, place)
        if pair not in pick_place_pairs:
            pick_place_pairs.append(pair)

random.shuffle(vegetables)

# Output the results
for i, ((pick, place), obj) in enumerate(zip(pick_place_pairs, vegetables), 1):
    print(f"Trial {i}: Pick {obj} from location {pick}, place at location {place}")


# # Paths
# input_base_path = '/workspace/xarm-dataset'         
# output_path = '/workspace/xarm-dataset/videos'          

# # Ensure output directory exists
# os.makedirs(output_path, exist_ok=True)

# obj = 'eggplant'
# task_name = f'place_the_{obj}_in_the_bowl'
# # List all trajectory folders
# task_path = os.path.join(input_base_path, task_name)

# trajectory_folders = sorted([f for f in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, f))])

# # For each trajectory folder, sample one image
# for traj_folder in trajectory_folders:
#     traj_path = os.path.join(task_path, traj_folder)
#     traj_path = os.path.join(traj_path, 'video/exo1')
#     print(traj_path)
#     images = [img for img in sorted(os.listdir(traj_path)) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
#     if not images:
#         print(f"No images found in {traj_folder}")
#         continue
    
#     # Sample one image (random, or change to images[0] for first image)
#     selected_image = images[0]

#     # Source and destination paths
#     src_image_path = os.path.join(traj_path, selected_image)
#     dst_image_path = os.path.join(output_path, f"{traj_folder}_{selected_image}")
    
#     # Copy image
#     shutil.copy(src_image_path, dst_image_path)
#     print(f"Copied {selected_image} from {traj_folder} to sampled_images")

# print("✅ Done. Sample images saved for visual inspection.")
