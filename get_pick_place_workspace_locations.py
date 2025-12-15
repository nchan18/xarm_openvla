import os
import random
import shutil

# random.seed(1000)
# locations = list(range(1, 10))  # Locations 1 to 9
# rotations = [30,45,60,90,120]
# vegetables  = ["carrot" , "eggplant" , "tomato" , "banana" , "corn", "green pepper" , "strawberry" , "orange" , "grape" , "green apple"]
# pick_place_pairs = []

# while len(pick_place_pairs) < 20:
#     pick = random.choice(locations)
#     place = random.choice(locations)
#     rot = random.choice(rotations)
#     if pick != place:
#         pair = (pick, place, rot)
#         if pair not in pick_place_pairs:
#             pick_place_pairs.append(pair)

# random.shuffle(vegetables)

# # Output the results
# for i, ((pick, place,rot), obj) in enumerate(zip(pick_place_pairs, vegetables*2), 1):
#     print(f"Trial {i}: Pick {obj} from location {pick}, place at location {place}, rotation degree {rot}")



random.seed(1000)

all_objects = ["corn", "banana", "lemon"]
all_locations = list(range(1, 10))
num_trials = 20

trials = []

for trial_num in range(1, num_trials + 1):
    # Step 1: Pick 2 out of 3 objects
    selected_objects = random.sample(all_objects, 2)
    
    # Step 2: Pick 3 shared workspace locations
    shared_locations = random.sample(all_locations, 3)
    
    # Step 3: Assign pick/place to each object (from shared locations)
    trial_data = []
    for obj in selected_objects:
        while True:
            pick, place = random.sample(shared_locations, 2)
            if pick != place:
                break
        trial_data.append({
            "object": obj,
            "pick": pick,
            "place": place
        })
    
    trials.append({
        "trial": trial_num,
        "shared_locations": shared_locations,
        "objects": trial_data
    })


for trial in trials:
    print(f"\n Trial {trial['trial']} — Shared locations: {trial['shared_locations']}")
    for obj in trial["objects"]:
        print(f"  - {obj['object']}: Pick from {obj['pick']} → Place at {obj['place']}")

