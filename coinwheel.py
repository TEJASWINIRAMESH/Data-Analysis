import random

class CoinFlippingGA:
    def __init__(self, population_size, chromosome_length, mutation_rate, head_probability):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.head_probability = head_probability
        self.population = []

    def initialize_population(self):
        self.population = [[random.choice([0, 1]) for _ in range(self.chromosome_length)] for _ in range(self.population_size)]

    def flip_coin(self):
        return random.random() < self.head_probability

    def crossover(self, parent1, parent2):
        child = []
        heads_child = 0
        tails_child = 0
        for bit1, bit2 in zip(parent1, parent2):
            if self.flip_coin():  # Flip a biased coin based on head probability
                child.append(bit1)
                heads_child += 1
            else:
                child.append(bit2)
                tails_child += 1
        return child, heads_child, tails_child

    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def fitness(self, individual):
        return sum(individual)

    def evolve(self):
        new_population = []
        for _ in range(self.population_size):
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
            child, heads_child, tails_child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
            print("Parent 1: {}, Parent 2: {}, Child: {}, Heads: {}, Tails: {}".format(parent1, parent2, child, heads_child, tails_child))
        self.population = new_population

    def run(self, generations):
        self.initialize_population()
        for gen in range(generations):
            print("Generation:", gen + 1)
            self.evolve()
            best_individual = max(self.population, key=self.fitness)
            #print("Best Individual - {}, Fitness - {}".format(best_individual, self.fitness(best_individual)))
        return self.population

# Example usage
population_size = 6
chromosome_length = 8
mutation_rate = 0.01
head_probability = float(input("Enter the probability of head: "))
generations = 3

ga = CoinFlippingGA(population_size, chromosome_length, mutation_rate, head_probability)
final_population = ga.run(generations)
print("Final population:")
for individual in final_population:
    print(individual)
