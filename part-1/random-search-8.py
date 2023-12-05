import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def objective_function(x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1/2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

def is_valid(x1, x2, xl, xu):
    return xl <= x1 <= xu and xl <= x2 <= xu

def random_search(Nmax, xl, xu):
    x1_best, x2_best = np.random.uniform(low=xl, high=xu, size=2)
    fbest = objective_function(x1_best, x2_best)
    sigma = find_optimal_sigma([x1_best, x2_best], xl, xu)
    
    for i in range(Nmax):
        n = np.random.normal(0, sigma, size=2)
        x1_cand, x2_cand = x1_best + n[0], x2_best + n[1]
        
        if is_valid(x1_cand, x2_cand, xl, xu):
            fcand = objective_function(x1_cand, x2_cand)
            
            if fcand < fbest:
                x1_best, x2_best = x1_cand, x2_cand
                fbest = fcand
    
    return [x1_best, x2_best], fbest, sigma

def find_optimal_sigma(x, xl, xu):
    sigma_values = np.linspace(0.01, 1.0, 100)
    best_value = float('inf')
    best_sigma = sigma_values[0]  # Defina um valor padrão
    
    for sigma in sigma_values:
        n = np.random.normal(0, sigma, size=2)
        x1_cand, x2_cand = x[0] + n[0], x[1] + n[1]
        
        if is_valid(x1_cand, x2_cand, xl, xu):
            fcand = objective_function(x1_cand, x2_cand)
            
            if fcand < best_value:
                best_value = fcand
                best_sigma = sigma
    
    return best_sigma

# Executar 100 rodadas do algoritmo
num_rounds = 100
Nmax_per_round = 1000
xl = -200.0
xu = 0.0

best_solutions = []
best_values = []
best_sigmas = []

for _ in range(num_rounds):
    best_solution, best_value, best_sigma = random_search(Nmax_per_round, xl, xu)
    best_solutions.append(best_solution)
    best_values.append(best_value)
    best_sigmas.append(best_sigma)

# Encontrar a melhor solução entre todas as rodadas
index_of_best_solution = np.argmin(best_values)
overall_best_solution = best_solutions[index_of_best_solution]
overall_best_value = best_values[index_of_best_solution]
min_value_optimal_solution = min(best_sigmas)
print("Menor valor que encontra a solução ótima:", min_value_optimal_solution)

# Criar um gráfico 3D para visualizar a função e as melhores soluções
x1_range = np.linspace(xl, xu, 100)
x2_range = np.linspace(xl, xu, 100)
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
