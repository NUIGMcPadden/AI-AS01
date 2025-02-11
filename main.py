import random
import math
import csv

GENERATIONS = 3
POPULATION = 3

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
    def __init__(self, order=None):
        self.order=order
        self.fitness=None

    def get_order(self):
        return self.order

    def set_order(self, order):
        self.order=order

    def get_fitness(self):
        # Negative sums distances travelled
        self.fitness=0
        for x in range(0, len(self.order) - 1):
            self.fitness -= get_distance(self.order[x], self.order[x+1])
        
    def check_correctness(self):
        # Helper function used to sanity check crossover functions. Disabled when going for performance
        if (len(self.order)== (len(set(self.order))+1) and self.order[0] == self.order[-1]):
            return
        else:
            raise IncorrectOrderException


class Population:
    def __init__(self, population_size, cities):
        self.agents = []
        for _ in range(0,population_size):
            self.agents.append(Agent(get_random_order(cities)))

    def cull_population(self):
        # remove less fit agents from the population
        return        

    def mutate_population(self):
        # trigger a change of some kind within an agent
        return

    def crossover_population(self):
        # cause one agent to influence another in some way
        return

    def get_best(self):
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

    for _ in range(0,GENERATIONS):
        myPopulation.get_fitnesses()
        myPopulation.mutate_population()
        myPopulation.crossover_population()
        myPopulation.cull_population()

    print("done simulating")
    myPopulation.get_fitnesses()
    bestAgent = myPopulation.get_best()
    print(bestAgent.get_order())
    print(bestAgent.fitness)

if __name__ == "__main__":
    main()