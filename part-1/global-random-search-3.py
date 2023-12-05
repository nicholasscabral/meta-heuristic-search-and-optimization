import numpy as np
import matplotlib.pyplot as plt

def is_valid(x1, x2, xl, xu):
    return xl <= x1 <= xu and xl <= x2 <= xu

def global_random_search(Nmax, xl, xu, f):
    xbest = np.random.uniform(low=xl, high=xu, size=(2,))
    fbest = f(xbest[0], xbest[1])
    sigma = find_optimal_sigma(xbest, xl, xu, f)
    i = 0
    
    while i < Nmax:
        candidate_solution = np.random.uniform(low=xl, high=xu, size=(2,))
        
        if is_valid(candidate_solution[0], candidate_solution[1], xl, xu):
            candidate_value = f(candidate_solution[0], candidate_solution[1])
            
            if candidate_value < fbest:
                xbest = candidate_solution
                fbest = candidate_value
        
        i += 1
    
    return xbest, fbest, sigma

def objective_function(x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)

def find_optimal_sigma(x, xl, xu, f):
    sigma_values = np.linspace(0.01, 1.0, 100)
    best_value = float('inf')
    best_sigma = None
    
    for sigma in sigma_values:
        n = np.random.normal(0, sigma, size=2)
        xcand = x + n
        
        if is_valid(xcand[0], xcand[1], xl, xu):
            fcand = f(xcand[0], xcand[1])
            
            if fcand < best_value:
                best_value = fcand
                best_sigma = sigma
    
    return best_sigma

# Parâmetros
num_rounds = 100  # Número de rodadas
iterations_per_round = 1000  # Número de iterações por rodada
Nmax = iterations_per_round
xl = -8
xu = 8

# Listas para rastrear resultados
best_solutions = []
best_values = []
best_sigmas = []

# Executar o algoritmo para 100 rodadas
for round_num in range(num_rounds):
    best_solution, best_value, best_sigma = global_random_search(Nmax, xl, xu, objective_function)
    best_solutions.append(best_solution)
    best_values.append(best_value)
    best_sigmas.append(best_sigma)

# Encontrar a melhor solução global
global_best_index = np.argmin(best_values)
global_best_solution = best_solutions[global_best_index]
global_best_value = best_values[global_best_index]
global_best_sigma = best_sigmas[global_best_index]

# Imprimir resultados
print("Melhor solução global encontrada:", global_best_solution)
print("Melhor valor global encontrado:", global_best_value)
print("Melhor valor de sigma encontrado:", global_best_sigma)

best_solutions = np.array(best_solutions)

# Plotar o gráfico
x1_values = np.linspace(xl, xu, 100)
x2_values = np.linspace(xl, xu, 100)
X1, X2 = np.meshgrid(x1_values, x2_values)
Y = objective_function(X1, X2)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.8, label='Função Objetivo')
ax.scatter(best_solutions[:, 0], best_solutions[:, 1], best_values, marker='o', color='red', label='Melhores Soluções de Cada Rodada')
ax.scatter(global_best_solution[0], global_best_solution[1], global_best_value, marker='X', color='green', s=100, label='Melhor Solução Global')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Função Objetivo')
ax.set_title('Superfície da Função Objetivo e Soluções')

plt.show()
