import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def objective_function(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1/2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

def is_valid(x1, x2, xl1, xu1, xl2, xu2):
    return xl1 <= x1 <= xu1 and xl2 <= x2 <= xu2

def find_optimal_sigma(x, xl1, xu1, xl2, xu2, sigma_min, sigma_max, objective_function):
    best_value = float('inf')
    best_sigma = None
    
    for sigma_candidate in np.linspace(sigma_min, sigma_max, 100):
        candidate_solution, candidate_value = local_random_search_iteration(x, xl1, xu1, xl2, xu2, sigma_candidate, objective_function)
        if candidate_value < best_value:
            best_value = candidate_value
            best_sigma = sigma_candidate
    
    return best_sigma

def local_random_search_iteration(initial_solution, xl1, xu1, xl2, xu2, sigma, objective_function):
    current_solution = initial_solution
    current_value = objective_function(*current_solution)
    
    perturbation = np.random.normal(0, sigma, size=2)  # Perturbação aleatória
    candidate_solution = current_solution + perturbation
    
    if is_valid(candidate_solution[0], candidate_solution[1], xl1, xu1, xl2, xu2):
        candidate_value = objective_function(*candidate_solution)
        
        if candidate_value < current_value:
            current_solution = candidate_solution
            current_value = candidate_value
    
    return current_solution, current_value

def local_random_search(xl1, xu1, xl2, xu2, sigma_min, sigma_max, num_iterations_per_round, objective_function):
    # Inicialização do melhor candidato
    xbest = np.random.uniform(low=[xl1, xl2], high=[xu1, xu2])
    fbest = objective_function(*xbest)
    
    # Encontrar o melhor valor de sigma
    best_sigma = find_optimal_sigma(xbest, xl1, xu1, xl2, xu2, sigma_min, sigma_max, objective_function)
    
    # Executar a busca local aleatória com o melhor valor de sigma
    for _ in range(num_iterations_per_round):
        xbest, fbest = local_random_search_iteration(xbest, xl1, xu1, xl2, xu2, best_sigma, objective_function)
    
    return xbest, fbest, best_sigma

#desvio padrão
sigma_min = 0.01
sigma_max = 1.0
# Executar 100 rodadas do algoritmo
num_rounds = 100
num_iterations_per_round = 1000
xl1, xu1 = -200.0, 0  # Limites para x1
xl2, xu2 = -200.0, 0   # Limites para x2

best_solutions = []
best_values = []

for _ in range(num_rounds):
    # Escolha uma solução inicial aleatória dentro dos limites
    initial_solution = np.random.uniform(low=[xl1, xl2], high=[xu1, xu2])
    
    # Executar a busca local aleatória
    best_solution, best_value, _ = local_random_search(xl1, xu1, xl2, xu2, sigma_min, sigma_max, num_iterations_per_round, objective_function)
    
    best_solutions.append(best_solution)
    best_values.append(best_value)

# Encontrar a melhor solução entre todas as rodadas
index_of_best_solution = np.argmin(best_values)
overall_best_solution = best_solutions[index_of_best_solution]
overall_best_value = best_values[index_of_best_solution]

# Criar um gráfico 3D para visualizar a função e as melhores soluções
x1_range = np.linspace(xl1, xu1, 100)
x2_range = np.linspace(xl2, xu2, 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
f_values = objective_function(x1_mesh, x2_mesh)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, f_values, cmap='viridis', alpha=0.8)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Valor da Função Objetivo')
ax.set_title('Função Objetivo em Projeção 3D')

best_solutions = np.array(best_solutions)
ax.scatter(best_solutions[:, 0], best_solutions[:, 1], best_values, c='red', marker='x', label='Melhores Soluções')

ax.scatter(overall_best_solution[0], overall_best_solution[1], overall_best_value, c='blue', marker='*', s=200, label='Melhor Solução Geral')
ax.legend()

plt.show()