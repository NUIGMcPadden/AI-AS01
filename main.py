GENERATIONS = X
POPULATION = Y

class City:
    # Representing cities as a class as it's a more readable method for storing their data
    def init(x_coord,y_coord):
        self.x=x_coord
        self.y=y_coord

    def get_x():
        return self.x

    def get_y():
        return self.y

class Agent:
    def init(order=None):
        self.order=order
        self.fitness=None

    def get_order():
        return self.order

    def set_order(order):
        self.order=order

    def get_fitness():
        # Negative sums distances travelled
        fitness=0
        for x in range(0, len(self.order) - 1):
            fitness -= get_distance(city[x], city[x+1])
        return fitness

    def check_correctness():
        # Helper function used to sanity check crossover functions. Disabled when going for performance
        if (len(self.order)== (len(set(self.order))+1) and self.order[0] == self.order[-1]):
            return
        else:
            raise IncorrectOrderException


class Population:
    def init(population_size,cities):
        self.agents = []
        for _ in range(0,population_size):
            self.agents+=Agent(get_random_order(cities))

    def cull_population():
        # remove less fit agents from the population
        return        

    def mutate_population():
        # trigger a change of some kind within an agent
        return

    def crossover_population():
        # cause one agent to influence another in some way
        return

    def get_best():
        return max(agents, key=lambda agent: agent.fitness)

    def get_fitnesses():
        for agent in self.agents:
            _ = agent.get_fintess


def get_random_order(cities):
    # Randomises order and ensures salesman returns home in the end
    order=random.sample(cities)
    order.add(order[0])
    return order


def get_distance(city_a, city_b):
    # Standard Euclidean distance calculation
    distance_squared = (city_a.get_x()**2) + (city_b.get_y()**2)
    distance = math.sqrt(distance_squared)
    return distance 

def import_cities(file_name):
    cities = []

    with open(file_name, "r") as file
        for row in file:
            cities.add(City(row[1],row[2]))

    return cities


def main():
    cities = import_cities("./berlin52.tsp.3.csv")
    myPopulation = Population(population_size=POPULATION,cities=cities)

    for _ in range(0,len(GENERATIONS)):
        myPopulation.mutate_population()
        myPopulation.crossover_population()
        myPopulation.cull_population()
        myPopulation.get_fitnesses()

    print(myPopulation.get_best().get_order())

if __name__ == "__main__":
    main()