import random
import math
import csv
import time

import matplotlib.pyplot as plt

GENERATIONS = 1000
POPULATION = 64

MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.95

class City:
    # Representing cities as a class as it's a more readable method for storing their data
    def __init__(self, id, x_coord,y_coord):
        self.x=x_coord
        self.y=y_coord
        self.id=id

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def __repr__(self):
        return f"{self.id}"

class Agent:
    def __init__(self, order):
        self.order=order
        self.get_fitness()
        self.check_correctness()

    def get_order(self):
        return self.order

    def set_order(self, order):
        self.order=order

    def mutate(self):
        # swaps the placement of two random cities
        idx1, idx2 = random.sample(range(1, len(self.order)-1), 2)
        self.order[idx1], self.order[idx2] = self.order[idx2], self.order[idx1]
        self.get_fitness()

    def get_fitness(self):
        # Negative sums distances travelled
        self.fitness=0
        for x in range(0, len(self.order) - 1):
            self.fitness += get_distance(self.order[x], self.order[x+1])
        
    def check_correctness(self):
        # Helper function used to sanity check crossover functions. Disabled when going for performance
        if (len(self.order)== (len(set(self.order))+1) and self.order[0] == self.order[-1]):
            return
        else:
            raise IncorrectOrderException


class Population:
    def __init__(self, population_size, cities):
        self.best_fitness = 50000
        self.agents = []
        for _ in range(0,population_size):
            self.agents.append(Agent(get_random_order(cities)))

    def roulette_wheel(self):
        # weighted selection by roulette
        weights = [agent.fitness**2.5 for agent in self.agents]
        selected_agent = random.choices(self.agents, weights=weights, k=1)[0]
        return selected_agent

    def tournament(self):
        # take the best agent from a tournament of a subpopulation
        tournament = random.sample(self.agents, 16)    
        best_agent = max(tournament, key=lambda agent: agent.fitness)
        return best_agent

    def cull_population(self, population_size):
        # remove less fit agents from the population
        while (len(self.agents) > population_size):
            self.agents.remove(self.get_worst())
        return        

    def mutate_population(self, current_generation, mutation_rate):
        # trigger a change of some kind within an agent
        for agent in self.agents:
            if random.random() < mutation_rate*(current_generation/GENERATIONS):
                agent.mutate()  
        return

    def crossover_population(self):
        # cause one agent to influence another in some way
        self.agents.append(Agent(order=self.get_crossover(self.get_best(), self.tournament())))
        return

    def get_crossover(self, agent_a, agent_b):
        # randomly choose a point, and take a section before or after that point from one parent
        # cities not added to the child are then appended in the order they appear in the second parent
        crossover_point = random.randint(1, len(agent_a.get_order()) - 2)
        if random.random() > 0.5:
            helper = agent_b
            agent_b = agent_a
            agent_a = helper

        if random.random() > 0.5:
            child_order = agent_a.get_order()[:crossover_point] 
            remaining_cities = [city for city in agent_b.get_order()[:-1] if city not in child_order]
            child_order = child_order + remaining_cities
            child_order.append(child_order[0])
        else:
            child_order = agent_a.get_order()[crossover_point:] 
            remaining_cities = [city for city in agent_b.get_order()[:-1] if city not in child_order]
            child_order = remaining_cities + child_order 
            child_order.append(child_order[0])

        return child_order

    def get_best(self):
        best = min(self.agents, key=lambda agent: agent.fitness)
        self.best_fitness = best.fitness
        return best

    def get_worst(self):
        return max(self.agents, key=lambda agent: agent.fitness)

    def get_fitnesses(self):
        for agent in self.agents:
            if agent.fitness == None:
                agent.get_fitness()


def get_random_order(cities):
    # Randomises order and ensures salesman returns home in the end
    order=random.sample(cities, len(cities))
    order.append(order[0])
    return order


def get_distance(city_a, city_b):
    # Standard Euclidean distance calculation
    x1 = float(city_a.get_x())
    x2 = float(city_b.get_x())
    y1 = float(city_a.get_y())
    y2 = float(city_b.get_y())
    distance_squared = ((x1-x2)**2) + ((y1-y2)**2)
    distance = math.sqrt(distance_squared)
    return distance 

def import_cities(file_name):
    cities = []

    with open(file_name, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            cities.append(City(row[0], row[1], row[2]))

    return cities


def main():
    start_time = time.time()

    seed = random.random()
    random.seed(seed)
    print(f"Random seed: {seed}")

    cities = import_cities("./berlin52.tsp.3c.csv")
    myPopulation = Population(population_size=POPULATION, cities=cities)
    avg_fitness_per_gen = []
    best_fitness = {
        "fitness": 999999999,
        "mutation": "n/a",
        "crossover": "n/a",
        "order": "n/a"
    }

    # The values in brackets here can be adjusted to use grid search or to test various parameters
    for i in [19]:
        for j in [1]:
            for _ in range(0,GENERATIONS):
                myPopulation.cull_population(POPULATION)
                while random.random() < (i/20.0):
                    myPopulation.crossover_population()
                myPopulation.mutate_population(_, (j/20.0))

                # Saves a record of the best agent and its parameters
                best_agent = myPopulation.get_best()
                if best_fitness["fitness"] > best_agent.fitness:
                    best_fitness["fitness"] = best_agent.fitness
                    best_fitness["mutation"] = j
                    best_fitness["crossover"] = i
                    best_fitness["order"] = best_agent.get_order()

                avg_fitness = sum(agent.fitness for agent in myPopulation.agents) / len(myPopulation.agents)
                avg_fitness_per_gen.append(avg_fitness)

    end_time = time.time()

    print("done simulating")
    myPopulation.get_fitnesses()
    bestAgent = myPopulation.get_best()
    print(bestAgent.get_order())
    print(bestAgent.fitness)
    print(f"Best achieved fitness: {best_fitness}")

    print(f"Elapsed time: {end_time - start_time} seconds")

    plt.plot(range(GENERATIONS), avg_fitness_per_gen, label="Average Fitness")
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness Over Generations')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()