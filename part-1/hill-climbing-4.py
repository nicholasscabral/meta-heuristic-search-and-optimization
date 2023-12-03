import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def candidato(x, epsilon):
    return x + np.random.uniform(low=-epsilon, high=epsilon)

def restricao_caixa(x, limite_inferior, limite_superior):
    return np.maximum(limite_inferior, np.minimum(x, limite_superior))

def funcao_objetivo(x1, x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)

def hill_climbing(limite_inferior, limite_superior, epsilon, maxit):
    x1_best = np.random.uniform(low=limite_inferior, high=limite_superior)
    x2_best = np.random.uniform(low=limite_inferior, high=limite_superior)
    fbest = funcao_objetivo(x1_best, x2_best)

    resultados = []

    for _ in range(maxit):
        x1_candidate = restricao_caixa(candidato(x1_best, epsilon), limite_inferior, limite_superior)
        x2_candidate = restricao_caixa(candidato(x2_best, epsilon), limite_inferior, limite_superior)
        f_candidate = funcao_objetivo(x1_candidate, x2_candidate)

        if f_candidate < fbest:
            x1_best, x2_best = x1_candidate, x2_candidate
            fbest = f_candidate

        resultados.append((x1_best, x2_best, fbest))

    return resultados

# Substitua os valores apropriados
limite_inferior = -5.12
limite_superior = 5.12
epsilon = 0.1
maxit = 1000
rodadas = 100

# Executa o algoritmo Hill Climbing em 100 rodadas
resultados_por_rodada = []

for _ in range(rodadas):
    resultados = hill_climbing(limite_inferior, limite_superior, epsilon, maxit)
    resultados_por_rodada.append(resultados)

# Encontra o ótimo global entre as rodadas
resultados_globais = [min(resultados, key=lambda x: x[2]) for resultados in resultados_por_rodada]
resultado_otimo_global = min(resultados_globais, key=lambda x: x[2])

# Criação do gráfico da função
x1_vals = np.linspace(limite_inferior, limite_superior, 100)
x2_vals = np.linspace(limite_inferior, limite_superior, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Y = funcao_objetivo(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='viridis')

# Plota os ótimos de cada rodada de forma mais transparente
for resultado in resultados_globais:
    ax.scatter(resultado[0], resultado[1], resultado[2], marker='o', s=50, linewidth=1, color='green', alpha=0.2)

# Plota o ótimo global de forma mais destacada
ax.scatter(resultado_otimo_global[0], resultado_otimo_global[1], resultado_otimo_global[2], marker='x', s=200, linewidth=3, color='red', label='Ótimo Global')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('f(x1, x2) - Hill Climbing (Mínimo) - Ótimos por Rodada e Ótimo Global')
ax.legend()

plt.tight_layout()
plt.show()
