import itertools
import random

def generate_positions(num_fruit_places=11, num_bowl_places=9, seed=None):
    """
    Generate all combinations of fruit placement and bowl placement,
    then randomize the order.

    Args:
        num_fruit_places (int): Number of places fruit can be placed (indexed 1..num_fruit_places).
        num_bowl_places (int): Number of places bowl can be placed (indexed 1..num_bowl_places).
        seed (int or None): If given, seed for reproducibility of randomization.

    Returns:
        List of tuples: Each tuple is (fruit_position, bowl_position).
    """
    # Create ranges or lists of places
    fruit_places = list(range(1, num_fruit_places + 1))
    bowl_places = list(range(1, num_bowl_places + 1))

    # Generate all combinations
    all_combinations = list(itertools.product(fruit_places, bowl_places))
    # Now randomize them
    if seed is not None:
        random.seed(seed)
    random.shuffle(all_combinations)
    return all_combinations

def main():
    combos = generate_positions()
    # For example, print them
    for i, (fruit, bowl) in enumerate(combos, start=1):
        print(f"Option {i}: Fruit at {fruit}, Bowl at {bowl}")

    # If you want only one random combo:
    random_combo = random.choice(combos)
    print("\nRandom single choice:", random_combo)

if __name__ == "__main__":
    main()
