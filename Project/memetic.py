import cv2
import numpy as np
import random
import sys

NEIGHBOURS_NUM = 7
POPULATION_SIZE = 50
ITERATIONS = 10
CROSSOVER_PROBABILITY = 0.05


def fitness(target, colors):
    res = 0
    used_colors = [0 for _ in colors]
    for pixel in target:
        color_index = np.argmin([np.linalg.norm(pixel / 255 - color / 255) for color in colors])
        res += np.linalg.norm(pixel / 255 - colors[color_index] / 255)
        used_colors[color_index] = 1
    mult_factor = (len(colors) - sum(used_colors) + 1)
    return res * mult_factor


def crossover(first_parent, second_parent):
    max_number_of_replacement = len(first_parent) // 2
    number_of_replacement = 0
    first_child, second_child = [], []
    for i in range(len(first_parent)):
        if random.random() <= 0.5 and number_of_replacement < max_number_of_replacement:
            number_of_replacement += 1
            first_child.append(second_parent[i])
            second_child.append(first_parent[i])
        else:
            first_child.append(first_parent[i])
            second_child.append(second_parent[i])
    return first_child, second_child


def get_neighbour_color(color, range):
    blue = max(0, min(255, color[0] + random.randint(-range, range)))
    green = max(0, min(255, color[1] + random.randint(-range, range)))
    red = max(0, min(255, color[2] + random.randint(-range, range)))
    return np.array([blue, green, red], dtype=np.uint8)


def get_neighbour(colors, range=10):
    return np.array(list(map(lambda x: get_neighbour_color(x, range), colors)))


class Individual:
    def __init__(self, colors):
        self.colors = np.array(colors, dtype=np.uint8)
        self.neighbours = [get_neighbour(colors) for _ in range(NEIGHBOURS_NUM)]
        self.best_neighbour = None
        self.fitness = None

    def calculate_fitness(self, target):
        self.fitness = fitness(target, self.colors)
        self.best_neighbour = self.colors
        for colors in self.neighbours:
            neighbour_fitness = fitness(target, colors)
            if neighbour_fitness < self.fitness:
                self.fitness = neighbour_fitness
                self.best_neighbour = colors


def create_fitted_image(image, colors):
    width, height = image.shape[0], image.shape[1]
    target = image.reshape(-1, image.shape[-1])
    assign_func = lambda pixel: colors[np.argmin([np.linalg.norm(pixel / 255 - color / 255) for color in colors])]
    result = np.array(list(map(assign_func, target)), dtype=np.uint8)
    return result.reshape(width, height, 3)


def memetic(linear_image, clusters):
    population = [
        Individual([[random.randint(0, 255) for _ in range(3)] for __ in range(clusters)])
        for _ in range(POPULATION_SIZE)
    ]
    for individual in population:
        individual.calculate_fitness(linear_image)

    for iteration in range(ITERATIONS):
        print(iteration, ": ", population[0].fitness, end="\r")
        new_generation = []
        for i in range(POPULATION_SIZE):
            for j in range(i + 1, POPULATION_SIZE):
                if random.random() < CROSSOVER_PROBABILITY:
                    first_child, second_child = crossover(population[i].best_neighbour, population[j].best_neighbour)
                    new_generation.append(Individual(first_child))
                    new_generation.append(Individual(second_child))
        for individual in new_generation:
            individual.calculate_fitness(linear_image)

        population.extend(new_generation)
        population.sort(key=lambda x: x.fitness)
        population = population[:POPULATION_SIZE]

    return population[0].best_neighbour, population[0].fitness


def main():
    image_name = sys.argv[1]
    clusters = int(sys.argv[2])
    size = 32

    image = cv2.imread(image_name)
    resized_image = cv2.resize(image, (size, size))
    linear_image = resized_image.reshape(-1, resized_image.shape[-1])

    best_result, best_fitness = memetic(linear_image, clusters)

    image_id = image_name.split('/')[1].split('.')[0]
    output_file = "output/%s_%s_%d.txt" % (image_id, "MEMETIC", clusters)
    print("\n", image_id, best_result, best_fitness)
    print(image_id,
          clusters, *[" ".join(map(str, color)) for color in best_result],
          sep="\n",
          file=open(output_file, 'w+'))
    fitted_image = create_fitted_image(image, np.array(best_result))
    cv2.imwrite("./output/%s_%s_%d.tif" % (image_id, "MEMETIC", clusters), fitted_image)


if __name__ == '__main__':
    main()
