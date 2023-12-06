import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def objective_function(x):
    return x[0]**2 + x[1]**2

def is_valid(x, xl, xu):
    return xl <= x[0] <= xu and xl <= x[1] <= xu

def local_random_search(initial_solution, xl1, xu1, xl2, xu2, num_iterations):
    current_solution = initial_solution
    current_value = objective_function(current_solution)
    
    for i in range(num_iterations):
        perturbation = np.random.normal(0, 0.1, size=2)  # Perturbação aleatória
        candidate_solution = current_solution + perturbation
        
        if is_valid(candidate_solution, xl1, xu1) and is_valid(candidate_solution, xl2, xu2):
            candidate_value = objective_function(candidate_solution)
            
            if candidate_value < current_value:
                current_solution = candidate_solution
                current_value = candidate_value
    
    return current_solution, current_value

# Executar 100 rodadas do algoritmo
num_rounds = 100
num_iterations_per_round = 1000
xl1, xu1 = -100.0, 100.0  # Limites para x1
xl2, xu2 = -100.0, 100.0    # Limites para x2

best_solutions = []
best_values = []

for _ in range(num_rounds):
    # Escolha uma solução inicial aleatória dentro dos limites
    initial_solution = np.random.uniform(low=[xl1, xl2], high=[xu1, xu2])
    
    # Executar a busca local aleatória
    best_solution, best_value = local_random_search(initial_solution, xl1, xu1, xl2, xu2, num_iterations_per_round)
    
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
f_values = objective_function([x1_mesh, x2_mesh])

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


#adapte esse codigo para as seguintes funcoes objetivo: 
# 
#3. -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)
#4. (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)
#5. (x1 - 1)**2 + 100 * (x2 - x1**2)**2
#7. (-np.sin(x1) * (np.sin((x1**2) / np.pi))**2 * 1e-10) + (-np.sin(x2) * (np.sin((2 * x2**2) / np.pi))**2 * 1e-10)
#8. -(x2 + 47) * np.sin(np.sqrt(np.abs(x1/2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))