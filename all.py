# %% [markdown]
# # Color Image

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# %%
img=cv2.imread('133518501384460247.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')  
plt.show()

# %%
m,n,_=img.shape
print(m,n)

# %%
low_ql_img=img*0.01
plt.imshow(low_ql_img.astype(np.uint8))
plt.axis('off') 
plt.show()

# %%
threshold=np.sum(low_ql_img)/(m*n*3)
threshold

# %%
sr=np.zeros((m,n,3),dtype=np.float64)

# %%
sigma=0.5
for _ in range(50):
    noisy=0.5*(np.random.randn(m,n,3)) + low_ql_img
    modified_img=np.where(noisy>threshold,255,0)
    sr+=modified_img

# %%
plt.imshow(noisy)
plt.axis('off') 
plt.show()

# %%
sr=sr/50

# %%
# Display the final results
plt.imshow(sr.astype(np.uint8))
plt.axis('off')
plt.show()

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read the color image
ref_img = cv2.imread('133518501384460247.jpg')
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct display with Matplotlib
plt.imshow(ref_img)
plt.show()

# Get the dimensions of the image
m, n, _ = ref_img.shape
print(m, n)

# Scale down the quality of the image
low_q_img = ref_img * 0.01
plt.imshow(low_q_img.astype(np.uint8))  # Convert to uint8 for display
plt.show()

# Compute the threshold
threshold = np.sum(low_q_img) / (m * n * 3)  # Sum over all channels

# Initialize variables
sr = np.zeros((m, n, 3), dtype=np.float64)
sigma = 0.5
mse = []

# Apply super-resolution to each channel
for _ in range(50):
    noisy = sigma * (np.random.randn(m, n, 3)) + low_q_img
    modified_img = np.where(noisy > threshold, 255, 0)
    sr += modified_img
    e = np.mean((ref_img - modified_img) ** 2)
    mse.append(e)

# Normalize the super-resolved image
sr = sr / 50
mse1 = np.mean((ref_img - sr) ** 2)
print(mse1)

# Display the final results
plt.imshow(sr.astype(np.uint8))
plt.show()

# Plot the Mean Squared Error over iterations
plt.plot(range(1, 51), mse)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Mean Squared Error over iterations')
plt.show()


# %% [markdown]
# # Contrast degraded video

# %%
import cv2
import numpy as np

# Function to degrade the contrast of an image
def degrade_contrast(image, alpha=0.5, beta=0):
    """
    Degrades the contrast of an image by scaling pixel values.
    alpha < 1 decreases contrast, alpha > 1 increases contrast.
    beta adjusts brightness.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Read the input video
input_video_path = 'video1.mp4'
output_video_path_contrast_degraded = 'contrast_degraded_video.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_contrast_degraded = cv2.VideoWriter(output_video_path_contrast_degraded, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Degrade the contrast of the frame
    degraded_frame = degrade_contrast(frame, alpha=0.5, beta=0)
    # Write the frame to the output video
    out_contrast_degraded.write(degraded_frame)

cap.release()
out_contrast_degraded.release()


# %%
import matplotlib.pyplot as plt

# Function to apply super-resolution technique to a single frame
def super_resolve_frame(low_q_img, ref_img, iterations=50, sigma=0.5):
    m, n, _ = low_q_img.shape
    threshold = np.sum(low_q_img) / (m * n * 3)  # Sum over all channels
    sr = np.zeros((m, n, 3), dtype=np.float64)
    mse = []
    
    for _ in range(iterations):
        noisy = sigma * (np.random.randn(m, n, 3)) + low_q_img
        modified_img = np.where(noisy > threshold, 255, 0)
        sr += modified_img
        e = np.mean((ref_img - modified_img) ** 2)
        mse.append(e)
    
    sr = sr / iterations
    mse1 = np.mean((ref_img - sr) ** 2)
    print(mse1)
    
    return sr.astype(np.uint8), mse

# Read the degraded video
input_video_path = 'contrast_degraded_video.mp4'
output_video_path_sr = 'super_resolved_video.mp4'
cap = cv2.VideoCapture(input_video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_sr = cv2.VideoWriter(output_video_path_sr, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Super-resolve the frame
    ref_frame = frame  # In practice, you might use a different reference frame
    low_q_frame = frame * 0.01
    sr_frame, mse = super_resolve_frame(low_q_frame, ref_frame)
    # Write the frame to the output video
    out_sr.write(sr_frame)

cap.release()
out_sr.release()

# %% [markdown]
# # Coin Flipping 

# %%
import numpy as np

def coin_flip_experiment(num_flips, prob_heads):
    return np.random.choice([0, 1], size=num_flips, p=[1 - prob_heads, prob_heads])


# %%
import cv2

def degrade_contrast(image, prob_heads):
    degraded_image = image.copy()
    mask = np.random.choice([0, 1], size=image.shape[:2], p=[1 - prob_heads, prob_heads])
    degraded_image[mask == 1] = (degraded_image[mask == 1] * 0.5).astype(np.uint8)  # Degrade half of the pixels
    return degraded_image


# %%
def super_resolve(low_q_img, ref_img, iterations=50, sigma=0.5):
    m, n, _ = low_q_img.shape
    threshold = np.sum(low_q_img) / (m * n * 3)  # Sum over all channels
    sr = np.zeros((m, n, 3), dtype=np.float64)
    mse = []

    for _ in range(iterations):
        noisy = sigma * (np.random.randn(m, n, 3)) + low_q_img
        modified_img = np.where(noisy > threshold, 255, 0)
        sr += modified_img
        e = np.mean((ref_img - modified_img) ** 2)
        mse.append(e)

    sr = sr / iterations
    mse1 = np.mean((ref_img - sr) ** 2)
    print(mse1)

    return sr.astype(np.uint8), mse

# %%
import matplotlib.pyplot as plt

# Read the input image
ref_img = cv2.imread("nature.jpg")
ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)  # Convert to RGB for correct display with Matplotlib
plt.imshow(ref_img)
plt.show()

# User inputs
num_flips = 10000
prob_heads = 0.3  # Probability of heads (user-defined)

# Perform coin flipping experiment
coin_flips = coin_flip_experiment(num_flips, prob_heads)

# Degrade the contrast of the image
low_q_img = degrade_contrast(ref_img, prob_heads)
plt.imshow(low_q_img)
plt.show()

# Apply super-resolution
sr_img, mse = super_resolve(low_q_img, ref_img)
plt.imshow(sr_img)
plt.show()

# Plot the Mean Squared Error over iterations
plt.plot(range(1, 51), mse)
plt.xlabel('Iteration')
plt.ylabel('MSE')
plt.title('Mean Squared Error over iterations')
plt.show()

# %% [markdown]
# # GSA - Force Computation

# %%
import numpy as np

# Function to calculate the Euclidean distance between two agents
def euclidean_distance(agent1, agent2):
    return np.linalg.norm(agent1 - agent2)

# Function to calculate the mass of each agent based on fitness
def calculate_mass(fitness):
    best = np.min(fitness)
    worst = np.max(fitness)
    mass = (fitness - worst) / (best - worst + 1e-10)  # Avoid division by zero
    return mass / np.sum(mass)

# Function to calculate the gravitational constant at time t
def gravitational_constant(G0, alpha, t):
    return G0 * np.exp(-alpha * t)

# Function to calculate the total force acting on each agent
def total_force(agents, fitness, G, epsilon=1e-10):
    num_agents, num_dimensions = agents.shape
    mass = calculate_mass(fitness)
    total_force = np.zeros((num_agents, num_dimensions))

    for i in range(num_agents):
        for j in range(num_agents):
            if i != j:
                distance = euclidean_distance(agents[i], agents[j])
                for k in range(num_dimensions):
                    force = G * (mass[i] * mass[j]) / (distance + epsilon) * (agents[j][k] - agents[i][k])
                    total_force[i][k] += np.random.rand() * force

    return total_force

# Example usage
num_agents = 5
num_dimensions = 3
agents = np.random.rand(num_agents, num_dimensions)
fitness = np.random.rand(num_agents)

G0 = 100  # Initial gravitational constant
alpha = 20  # Gravitational constant decay rate
t = 1  # Time step

G = gravitational_constant(G0, alpha, t)
forces = total_force(agents, fitness, G)

print("Agents' positions:\n", agents)
print("Fitness values:\n", fitness)
print("Total forces on each agent:\n", forces)
print(np.sum(forces))


# %% [markdown]
# # Genetic - Crossover

# %%
import numpy as np

# Dice rolling function
def roll_dice():
    return np.random.randint(1, 7)

# Initialize population
def initialize_population(pop_size, chromosome_length):
    return np.random.randint(2, size=(pop_size, chromosome_length))

# Evaluate fitness
def evaluate_fitness(population):
    # Example fitness function: sum of bits (maximize the number of 1s)
    return np.sum(population, axis=1)

# Selection: roulette wheel selection
def select_parents(population, fitness):
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    parents_indices = np.random.choice(np.arange(len(population)), size=len(population), p=probabilities)
    return population[parents_indices]

# Crossover
def crossover(parents):
    offspring = np.empty_like(parents)
    for i in range(0, len(parents), 2):
        if i+1 < len(parents):
            parent1, parent2 = parents[i], parents[i+1]
            # Roll the dice to determine the crossover point
            crossover_point = roll_dice() % len(parent1)  # Ensure the crossover point is within bounds
            offspring[i, :crossover_point] = parent1[:crossover_point]
            offspring[i, crossover_point:] = parent2[crossover_point:]
            offspring[i+1, :crossover_point] = parent2[:crossover_point]
            offspring[i+1, crossover_point:] = parent1[crossover_point:]
        else:
            offspring[i] = parents[i]
    return offspring

# Mutation
def mutate(offspring, mutation_rate=0.01):
    mutation_mask = np.random.rand(*offspring.shape) < mutation_rate
    offspring[mutation_mask] = 1 - offspring[mutation_mask]
    return offspring

# Main GA function
def genetic_algorithm(pop_size, chromosome_length, max_generations, mutation_rate=0.01):
    # Initialize population
    population = initialize_population(pop_size, chromosome_length)
    
    for generation in range(max_generations):
        # Evaluate fitness
        fitness = evaluate_fitness(population)
        # Select parents
        parents = select_parents(population, fitness)
        # Perform crossover
        offspring = crossover(parents)
        # Perform mutation
        offspring = mutate(offspring, mutation_rate)
        # Update population
        population = offspring
        
        # Print the best fitness in the current generation
        best_fitness = np.max(fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    
    # Return the best solution
    best_index = np.argmax(fitness)
    return population[best_index], np.max(fitness)

# Example usage
pop_size = 10
chromosome_length = 8
max_generations = 20
mutation_rate = 0.01

best_solution, best_fitness = genetic_algorithm(pop_size, chromosome_length, max_generations, mutation_rate)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)

# %% [markdown]
# # Genetic - Mutation

# %%
import numpy as np

# Dice rolling function
def roll_dice():
    return np.random.randint(1, 7)

# Initialize population
def initialize_population(pop_size, chromosome_length):
    return np.random.randint(2, size=(pop_size, chromosome_length))

# Evaluate fitness
def evaluate_fitness(population):
    # Example fitness function: sum of bits (maximize the number of 1s)
    return np.sum(population, axis=1)

# Selection: roulette wheel selection
def select_parents(population, fitness):
    total_fitness = np.sum(fitness)
    probabilities = fitness / total_fitness
    parents_indices = np.random.choice(np.arange(len(population)), size=len(population), p=probabilities)
    return population[parents_indices]

# Crossover
def crossover(parents):
    offspring = np.empty_like(parents)
    for i in range(0, len(parents), 2):
        if (i + 1) < len(parents):
            parent1, parent2 = parents[i], parents[i + 1]
            crossover_point = roll_dice() % len(parent1)  # Ensure the crossover point is within bounds
            offspring[i, :crossover_point] = parent1[:crossover_point]
            offspring[i, crossover_point:] = parent2[crossover_point:]
            offspring[i + 1, :crossover_point] = parent2[:crossover_point]
            offspring[i + 1, crossover_point:] = parent1[crossover_point]
        else:
            offspring[i] = parents[i]
    return offspring

# Hamming distance calculation
def hamming_distance(parent, child):
    return np.sum(parent != child)

# Mutation with max Hamming distance selection
def mutate_with_max_hamming_distance(offspring, num_mutations=5, mutation_rate=0.01):
    mutated_offspring = np.empty_like(offspring)
    for i in range(len(offspring)):
        parent = offspring[i]
        max_hamming_dist = -1
        best_mutation = parent
        for _ in range(num_mutations):
            mutation_mask = np.random.rand(*parent.shape) < mutation_rate
            child = np.copy(parent)
            child[mutation_mask] = 1 - child[mutation_mask]
            hamming_dist = hamming_distance(parent, child)
            if hamming_dist > max_hamming_dist:
                max_hamming_dist = hamming_dist
                best_mutation = child
        mutated_offspring[i] = best_mutation
    return mutated_offspring

# Main GA function
def genetic_algorithm(pop_size, chromosome_length, max_generations, num_mutations=5, mutation_rate=0.01):
    # Initialize population
    population = initialize_population(pop_size, chromosome_length)
    
    for generation in range(max_generations):
        # Evaluate fitness
        fitness = evaluate_fitness(population)
        # Select parents
        parents = select_parents(population, fitness)
        # Perform crossover
        offspring = crossover(parents)
        # Perform mutation with max Hamming distance selection
        offspring = mutate_with_max_hamming_distance(offspring, num_mutations, mutation_rate)
        # Update population
        population = offspring
        
        # Print the best fitness in the current generation
        best_fitness = np.max(fitness)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
    
    # Return the best solution
    best_index = np.argmax(fitness)
    return population[best_index], np.max(fitness)

# Example usage
pop_size = 10
chromosome_length = 8
max_generations = 20
mutation_rate = 0.01

best_solution, best_fitness = genetic_algorithm(pop_size, chromosome_length, max_generations, mutation_rate=mutation_rate)
print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)

# %%



