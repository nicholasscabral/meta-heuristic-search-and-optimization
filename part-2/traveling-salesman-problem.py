import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def generate_points(N):
    x_partition = np.random.uniform(-10, 10, size=(N, 3))
    y_partition = np.random.uniform(0, 20, size=(N, 3))
    z_partition = np.random.uniform(-20, 0, size=(N, 3))
    w_partition = np.random.uniform(0, 20, size=(N, 3))

    x1 = np.array([[20, -20, -20]])
    x1 = np.tile(x1, (N, 1))
    x_partition = x_partition + x1

    x1 = np.array([[-20, 20, 20]])
    x1 = np.tile(x1, (N, 1))
    y_partition = y_partition + x1

    x1 = np.array([[-20, 20, -20]])
    x1 = np.tile(x1, (N, 1))
    z_partition = z_partition + x1

    x1 = np.array([[20, 20, -20]])
    x1 = np.tile(x1, (N, 1))
    w_partition = w_partition + x1

    return np.concatenate((x_partition, y_partition, z_partition, w_partition), axis=0)


def fitness(route, points):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += distance(points[route[i]], points[route[i + 1]])
    return total_distance


def tournament_selection(population, fitness_values, tournament_size=5):
    selected_indices = np.random.choice(
        len(population), size=tournament_size, replace=False
    )
    tournament_fitness = fitness_values[selected_indices]
    winner_index = selected_indices[np.argmin(tournament_fitness)]
    return population[winner_index]


def crossover(parent1, parent2):
    crossover_point1 = np.random.randint(1, len(parent1))
    crossover_point2 = np.random.randint(crossover_point1, len(parent1))
    child = np.zeros_like(parent1)

    child[crossover_point1:crossover_point2] = parent1[
        crossover_point1:crossover_point2
    ]
    remaining_indices = np.setdiff1d(parent2, child[crossover_point1:crossover_point2])
    child[:crossover_point1] = remaining_indices[:crossover_point1]
    child[crossover_point2:] = remaining_indices[crossover_point1:]

    return child


def mutate(route):
    mutate_point1 = np.random.randint(0, len(route))
    mutate_point2 = np.random.randint(0, len(route))

    route[mutate_point1], route[mutate_point2] = (
        route[mutate_point2],
        route[mutate_point1],
    )

    return route


def genetic_algorithm(
    points, population_size, generations, mutation_prob, crossover_prob
):
    population = np.array(
        [np.random.permutation(len(points)) for _ in range(population_size)]
    )
    best_route = None
    best_fitness = float("inf")

    best_fitness_per_generation = []

    for generation in range(generations):
        print(generation, max_generations)
        fitness_values = np.array([fitness(route, points) for route in population])
        best_fitness_per_generation.append(np.min(fitness_values))

        if np.min(fitness_values) < best_fitness:
            best_route = population[np.argmin(fitness_values)]
            best_fitness = np.min(fitness_values)

        new_population = []

        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values)

            if np.random.rand() < crossover_prob:
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if np.random.rand() < mutation_prob:
                child1 = mutate(child1)
            if np.random.rand() < mutation_prob:
                child2 = mutate(child2)

            new_population.extend([child1, child2])

        population = np.array(new_population)

    return best_route, best_fitness, best_fitness_per_generation


def plot_initial_state(points):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="blue", marker="o")
    ax.set_title("Estado Inicial")


def plot_best_route(points, best_route, best_fitness):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="blue", marker="o")
    ax.plot(
        points[best_route, 0],
        points[best_route, 1],
        points[best_route, 2],
        c="red",
        linewidth=2,
        marker="o",
    )
    ax.set_title("Melhor Caminho")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.text2D(
        0.05,
        0.95,
        f"Melhor Aptidão: {best_fitness:.2f}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.show()


def plot_best_fitness_per_generation(best_fitness_per_generation):
    plt.figure()
    plt.plot(
        range(11, len(best_fitness_per_generation) + 1),
        best_fitness_per_generation[10:],
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.title("Melhores aptidões por geração")
    plt.xlabel("Geração")
    plt.ylabel("Melhor aptidão")
    plt.show()


def calculate_mode_of_100_runs(
    points, pop_size, max_generations, mutation_prob, crossover_prob
):
    i = 1
    all_best_fitness = []
    for _ in range(100):
        _, best_fitness, _ = genetic_algorithm(
            points, pop_size, max_generations, mutation_prob, crossover_prob
        )
        all_best_fitness.append(best_fitness)
        print(f"{i}/100")
        i = i + 1

    print(all_best_fitness)
    mode_fitness, mode_count = mode(all_best_fitness)

    plt.figure()
    plt.plot(
        range(1, 101),
        all_best_fitness,
        marker="o",
        linestyle="-",
        color="blue",
    )
    plt.axhline(
        mode_fitness,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"Moda dos resultados: {mode_fitness:.2f} ({mode_count} ocorrências)",
    )
    plt.title("Melhores aptidões para 100 execuções")
    plt.xlabel("Execução")
    plt.ylabel("Melhor aptidão")
    plt.legend()
    plt.show()


# Parameters
num_points = 50
pop_size = 100
max_generations = 1000
mutation_prob = 0.01
crossover_prob = 0.9

# Generate points
points = generate_points(num_points)

# Run the genetic algorithm
best_route, best_fitness, best_fitness_per_generation = genetic_algorithm(
    points, pop_size, max_generations, mutation_prob, crossover_prob
)

print(best_fitness)
plot_initial_state(points)
plot_best_route(points, best_route, best_fitness)
plot_best_fitness_per_generation(best_fitness_per_generation)
calculate_mode_of_100_runs(
    points, pop_size, max_generations, mutation_prob, crossover_prob
)
