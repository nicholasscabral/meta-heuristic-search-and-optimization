import numpy as np
import matplotlib.pyplot as plt

def is_valid(x1, x2, xl1, xu1, xl2, xu2):
    return xl1 <= x1 <= xu1 and xl2 <= x2 <= xu2

def global_random_search(Nmax, xl1, xu1, xl2, xu2, f):
    xbest = np.random.uniform(low=[xl1, xl2], high=[xu1, xu2])
    fbest = f(xbest[0], xbest[1])
    sigma = find_optimal_sigma(xbest, xl1, xu1, xl2, xu2, f)
    i = 0
    
    while i < Nmax:
        candidate_solution = np.random.uniform(low=[xl1, xl2], high=[xu1, xu2])
        
        if is_valid(candidate_solution[0], candidate_solution[1], xl1, xu1, xl2, xu2):
            candidate_value = f(candidate_solution[0], candidate_solution[1])
            
            if candidate_value < fbest:
                xbest = candidate_solution
                fbest = candidate_value
        
        i += 1
    
    return xbest, fbest, sigma

def objective_function(x1, x2):
    return (x1 - 1)**2 + 100 * (x2 - x1**2)**2

def find_optimal_sigma(x, xl1, xu1, xl2, xu2, f):
    sigma_values = np.linspace(0.01, 1.0, 100)
    best_value = float('inf')
    best_sigma = None
    
    for sigma in sigma_values:
        n = np.random.normal(0, sigma, size=2)
        xcand = x + n
        
        if is_valid(xcand[0], xcand[1], xl1, xu1, xl2, xu2,):
            fcand = f(xcand[0], xcand[1])
            
            if fcand < best_value:
                best_value = fcand
                best_sigma = sigma
    
    return best_sigma

# Restante do código (a função objective_function, find_optimal_sigma, parâmetros, execução do algoritmo, etc.)

# Parâmetros
num_rounds = 100  # Número de rodadas
iterations_per_round = 1000  # Número de iterações por rodada
Nmax = iterations_per_round
xl1, xu1 = -2.0, 2.0  # Limites para x1
xl2, xu2 = -1.0, 3.0    # Limites para x2

# Listas para rastrear resultados
best_solutions = []
best_values = []
best_sigmas = []

# Executar o algoritmo para 100 rodadas
for round_num in range(num_rounds):
    best_solution, best_value, best_sigma = global_random_search(Nmax, xl1, xu1, xl2, xu2, objective_function)
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
x1_values = np.linspace(xl1, xu1, 100)
x2_values = np.linspace(xl2, xu2, 100)
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
