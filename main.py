import random
import math
import csv

import matplotlib.pyplot as plt

GENERATIONS = 10000
POPULATION = 64

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
        weights = [agent.fitness**2.5 for agent in self.agents]
        selected_agent = random.choices(self.agents, weights=weights, k=1)[0]
        return selected_agent

    def tournament(self):
        tournament = random.sample(self.agents, 20)    
        best_agent = max(tournament, key=lambda agent: agent.fitness)
        return best_agent

    def cull_population(self, population_size):
        # remove less fit agents from the population
        while (len(self.agents) > population_size):
            self.agents.remove(self.get_worst())
        return        

    def mutate_population(self):
        # trigger a change of some kind within an agent
        for agent in self.agents:
            while random.random() < min(0.2, (7500/agent.fitness)**2):
                agent.mutate()  
        return

    def crossover_population(self):
        # cause one agent to influence another in some way
        self.agents.append(Agent(order=self.get_crossover(self.get_best(), self.tournament())))
        return

    def get_crossover(self, agent_a, agent_b):

        crossover_point = random.randint(1, len(agent_a.get_order()) - 2)

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
    cities = import_cities("./berlin52.tsp.3c.csv")
    myPopulation = Population(population_size=POPULATION, cities=cities)
    avg_fitness_per_gen = []


    for _ in range(0,GENERATIONS):
        myPopulation.cull_population(POPULATION)
        myPopulation.mutate_population()
        for _ in range(20):
            myPopulation.crossover_population()

        avg_fitness = sum(agent.fitness for agent in myPopulation.agents) / len(myPopulation.agents)
        avg_fitness_per_gen.append(avg_fitness)


    print("done simulating")
    myPopulation.get_fitnesses()
    bestAgent = myPopulation.get_best()
    print(bestAgent.get_order())
    print(bestAgent.fitness)

    plt.plot(range(GENERATIONS), avg_fitness_per_gen, label="Average Fitness")
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness Over Generations')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()