import random
from collections import Counter

def integer_to_binary(num, num_bits):
    
    binary_str = bin(num)[2:]
    return binary_str.zfill(num_bits)

def single_point_crossover(parent1, parent2):
   
    # Determine the length of the parents' genomes
    genome_length = min(len(parent1), len(parent2))
    
    # Roll the dice 10 times for each parent
    parent1_rolls = [random.randint(0, genome_length - 1) for _ in range(10)]
    parent2_rolls = [random.randint(0, genome_length - 1) for _ in range(10)]
    
    # Calculate the probabilities of each bit being chosen as the crossover point for each parent
    parent1_counts = Counter(parent1_rolls)
    parent2_counts = Counter(parent2_rolls)
    print(parent1_counts)
    # Determine the crossover point for each parent based on the highest probability
    parent1_crossover_point = max(parent1_counts, key=parent1_counts.get)
    parent2_crossover_point = max(parent2_counts, key=parent2_counts.get)
    
    # Perform crossover
    child1 = parent1[:parent1_crossover_point] + parent2[parent1_crossover_point:]
    child2 = parent2[:parent2_crossover_point] + parent1[parent2_crossover_point:]
    
    # Convert offspring to binary representation
    child1_binary = int(''.join(map(str, child1)), 2)
    child2_binary = int(''.join(map(str, child2)), 2)
    
    return child1, child2, child1_binary, child2_binary

# Example usage
parent1_decimal = 20
parent2_decimal = 400

# Convert decimal inputs to 12-bit binary representations
parent1_binary = integer_to_binary(parent1_decimal, 12)
parent2_binary = integer_to_binary(parent2_decimal, 12)

# Convert binary inputs to lists of integers
parent1 = [int(bit) for bit in parent1_binary]
parent2 = [int(bit) for bit in parent2_binary]

# Perform crossover
child1, child2, child1_decimal, child2_decimal = single_point_crossover(parent1, parent2)

# Print results
print("Parent 1 (binary):", parent1)
print("Parent 2 (binary):", parent2)
print("Child 1 (binary):", child1)
print("Child 2 (binary):", child2)
print("Parent 1 (decimal):", parent1_decimal)
print("Parent 2 (decimal):", parent2_decimal)
print("Child 1 (decimal):", child1_decimal)
print("Child 2 (decimal):",child2_decimal)
