import numpy as np

def initialize_positions(num_particles, dimensions, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (num_particles, dimensions))

def compute_total_force(positions, masses, G, epsilon=1e-10):
    num_particles = len(positions)
    dimensions = positions.shape[1]  # Assuming positions are in a 2D array
    
    # Initialize total force array
    total_forces = np.zeros_like(positions)
    
    for i in range(num_particles):
        force = np.zeros(dimensions)
        
        for j in range(num_particles):
            if i != j:
                r = positions[j] - positions[i]
                distance = np.linalg.norm(r) + epsilon
                force_direction = r / distance  # Unit vector pointing from particle i to j
                force_magnitude = G * (masses[i] * masses[j]) / (distance**2)
                force += force_magnitude * force_direction
        
        total_forces[i] = force
    
    return total_forces

# Example usage
num_particles = 5  # Number of particles
dimensions = 3     # Dimensionality of the problem
lower_bound = -10  # Lower bound for each dimension
upper_bound = 10   # Upper bound for each dimension

positions = initialize_positions(num_particles, dimensions, lower_bound, upper_bound)
masses = np.random.rand(num_particles) * 10  # Random masses for each particle
G = 1  # Gravitational constant for simplicity in this example

total_forces = compute_total_force(positions, masses, G)
print("Initial Positions:")
print(positions)
print("Total Forces:")
print(total_forces)
