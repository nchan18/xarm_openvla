import random
import json

# Constants
fruit_types = [
    "eggpalnt", "banana", "grape", "carrot", "green_apple",
    "pepper", "tomatoe", "strawberry", "orange", "corn"
]
bowl_locations = list(range(9))     # 0-8
fruit_locations = list(range(11))   # 0-10

# Result container
all_configs = []

# Generate 10 configurations per fruit type
for fruit in fruit_types:
    configs = set()  # use set to avoid accidental duplicates
    while len(configs) < 10:
        bowl_loc = random.choice(bowl_locations)
        fruit_loc = random.choice(fruit_locations)
        if bowl_loc != fruit_loc:
            configs.add((bowl_loc, fruit_loc))
    
    # Add to final list
    for bowl_loc, fruit_loc in configs:
        all_configs.append({
            # "fruit_type": fruit,
            "bowl_location": bowl_loc+1,
            "fruit_location": fruit_loc+1
        })

# Shuffle final list (optional)
random.shuffle(all_configs)

all_configs.sort(key=lambda x: x["fruit_type"])

# Print output
for config in all_configs:
    print(config)