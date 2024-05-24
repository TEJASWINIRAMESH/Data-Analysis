import numpy as np

# Example fitness function (to be minimized)
def fitness_function(position):
    return np.sum(position**2)

def initialize_positions(num_particles, dimensions, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (num_particles, dimensions))

def initialize_velocities(num_particles, dimensions):
    return np.zeros((num_particles, dimensions))

def compute_fitness(positions):
    return np.array([fitness_function(pos) for pos in positions])

def compute_masses(fitness):
    worst = np.max(fitness)
    best = np.min(fitness)
    masses = (fitness - worst) / (best - worst + 1e-10)
    masses = masses / np.sum(masses)  # Normalize masses
    return masses

def compute_total_force(positions, masses, G, epsilon=1e-10):
    num_particles = len(positions)
    dimensions = positions.shape[1]
    
    total_forces = np.zeros_like(positions)
    
    for i in range(num_particles):
        force = np.zeros(dimensions)
        
        for j in range(num_particles):
            if i != j:
                r = positions[j] - positions[i]
                distance = np.linalg.norm(r) + epsilon
                force_direction = r / distance
                force_magnitude = G * (masses[i] * masses[j]) / (distance**2)
                force += force_magnitude * force_direction
        
        total_forces[i] = force
    
    return total_forces

def update_particles(positions, velocities, forces, masses, dt=1.0):
    num_particles = len(positions)
    
    for i in range(num_particles):
        acceleration = forces[i] / (masses[i] + 1e-10)  # Avoid division by zero
        velocities[i] += acceleration * dt
        positions[i] += velocities[i] * dt
    
    return positions, velocities

# Example usage
num_particles = 5
dimensions = 3
lower_bound = -10
upper_bound = 10
G = 1

positions = initialize_positions(num_particles, dimensions, lower_bound, upper_bound)
velocities = initialize_velocities(num_particles, dimensions)

# Optimization loop
for iteration in range(100):  # Number of iterations
    fitness = compute_fitness(positions)
    masses = compute_masses(fitness)
    total_forces = compute_total_force(positions, masses, G)
    positions, velocities = update_particles(positions, velocities, total_forces, masses)

print("Final Positions:")
print(positions)
print("Final Velocities:")
print(velocities)
