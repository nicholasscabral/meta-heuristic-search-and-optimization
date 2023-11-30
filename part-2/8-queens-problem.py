import numpy as np
import matplotlib.pyplot as plt

# Definição do problema das 8 rainhas
num_queens = 8

# Req 1. Definição da quantidade N de indivíduos em uma população e quantidade máxima de gerações.
pop_size = 100
max_generations = 1000

mutation_prob = 0.01
crossover_prob = 0.9


# Req 2. Projeto do operador de seleção baseado no método da roleta.
def roulette_wheel_selection(fitness_values):
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness
    selected_indices = np.random.choice(
        len(fitness_values), size=len(fitness_values), p=probabilities
    )
    return selected_indices  # Requisito 2


# Função de Aptidão
def fitness_function(chromosome):
    attacking_pairs = 28 - count_attacking_pairs(chromosome)
    return attacking_pairs


# Contagem de pares atacantes em um cromossomo
def count_attacking_pairs(chromosome):
    count = 0
    for i in range(num_queens - 1):
        for j in range(i + 1, num_queens):
            if chromosome[i] == chromosome[j] or abs(
                chromosome[i] - chromosome[j]
            ) == abs(i - j):
                count += 1
    return count


# 3. Escolha do operador de recombinação de um ponto ou recombinação de dois pontos.
def one_point_crossover(parent1, parent2):
    crossover_point = np.random.randint(1, num_queens)
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def two_point_crossover(parent1, parent2):
    crossover_points = np.sort(np.random.choice(num_queens, 2, replace=False))
    child1 = np.concatenate(
        (
            parent1[: crossover_points[0]],
            parent2[crossover_points[0] : crossover_points[1]],
            parent1[crossover_points[1] :],
        )
    )
    child2 = np.concatenate(
        (
            parent2[: crossover_points[0]],
            parent1[crossover_points[0] : crossover_points[1]],
            parent2[crossover_points[1] :],
        )
    )
    return child1, child2  # Requisito 3


# 4. Aplicação da mutação com probabilidade de 1%
def mutate(child, mutation_prob):
    if np.random.rand() < mutation_prob:
        mutation_point = np.random.randint(num_queens)
        new_value = np.random.randint(num_queens)
        child[mutation_point] = new_value
    return child  # Requisito 4


# 5. Condição de parada do algoritmo
def stop_condition(generation, best_fitness):
    print(generation, best_fitness)
    return generation >= max_generations or best_fitness == 28  # Requisito 5


# Algoritmo Genético
def genetic_algorithm(pop_size, num_queens, mutation_prob, crossover_prob):
    population = np.array([np.random.permutation(num_queens) for _ in range(pop_size)])

    for generation in range(max_generations):
        fitness_values = np.array(
            [fitness_function(chromosome) for chromosome in population]
        )

        best_index = np.argmax(fitness_values)
        best_chromosome = population[best_index].copy()
        best_fitness = fitness_function(best_chromosome)

        if stop_condition(generation, best_fitness):
            break

        selected_indices = roulette_wheel_selection(fitness_values)
        new_population = [population[idx].copy() for idx in selected_indices]

        for i in range(0, pop_size - 1, 2):
            parent1 = new_population[i]
            parent2 = new_population[i + 1]

            # Escolha do operador de recombinação
            if np.random.rand() < crossover_prob:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1, child2 = two_point_crossover(parent1, parent2)

            # Aplicação da mutação
            child1 = mutate(child1, mutation_prob)
            child2 = mutate(child2, mutation_prob)

            new_population.extend([child1, child2])

        population = np.array(new_population)

    return list(map(lambda x: x + 1, best_chromosome)), 28 - best_fitness


def plot_chessboard_with_queens(arrangement):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(8))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)  # Remove as marcas dos eixos

    # Preencher os quadrados do tabuleiro com bordas pretas
    for i in range(num_queens):
        for j in range(num_queens):
            color = "white" if (i + j) % 2 == 0 else "black"
            ax.fill_between([i, i + 1], j, j + 1, color=color, edgecolor="black")

    # Adicionar rainhas ao tabuleiro
    for col, row in enumerate(arrangement):
        ax.text(
            col + 0.5,
            num_queens - row + 0.5,
            "♛",
            fontsize=30,
            ha="center",
            va="center",
            color="red",
        )

    plt.title("Tabuleiro de xadrez com melhor arranjo de rainhas")
    plt.show()


# Execução do Algoritmo Genético
best_solution, best_fitness = genetic_algorithm(
    pop_size=pop_size,
    num_queens=num_queens,
    mutation_prob=mutation_prob,
    crossover_prob=crossover_prob,
)

# Resultados
print("Melhor Arranjo de Rainhas:", best_solution)
print(f"Melhor Aptidão: {best_fitness} pares atacantes")
plot_chessboard_with_queens(best_solution)
