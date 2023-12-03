import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def candidato(x, epsilon):
    return x + np.random.uniform(low=-epsilon, high=epsilon)

def restricao_caixa(x, limite_inferior, limite_superior):
    return np.maximum(limite_inferior, np.minimum(x, limite_superior))

def funcao_objetivo(x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))

def hill_climbing(limite_inferior, limite_superior, epsilon, maxit):
    x1_best = np.random.uniform(low=limite_inferior, high=limite_superior)
    x2_best = np.random.uniform(low=limite_inferior, high=limite_superior)
    fbest = funcao_objetivo(x1_best, x2_best)

    resultados = []

    for _ in range(maxit):
        x1_candidate = restricao_caixa(candidato(x1_best, epsilon), limite_inferior, limite_superior)
        x2_candidate = restricao_caixa(candidato(x2_best, epsilon), limite_inferior, limite_superior)
        f_candidate = funcao_objetivo(x1_candidate, x2_candidate)

        if f_candidate > fbest:
            x1_best, x2_best = x1_candidate, x2_candidate
            fbest = f_candidate

        resultados.append((x1_best, x2_best, fbest))

    return resultados

# Substitua os valores apropriados
limite_inferior = -2
limite_superior = 4
epsilon = 0.1
maxit = 1000
rodadas = 100

# Executa o algoritmo Hill Climbing em 100 rodadas
resultados_por_rodada = []

for _ in range(rodadas):
    resultados = hill_climbing(limite_inferior, limite_superior, epsilon, maxit)
    resultados_por_rodada.append(resultados)

# Encontra o ótimo global entre as rodadas
resultados_globais = [max(resultados, key=lambda x: x[2]) for resultados in resultados_por_rodada]
resultado_otimo_global = max(resultados_globais, key=lambda x: x[2])

# Criação do gráfico da função
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plota os ótimos de cada rodada com marcador maior e uma cor diferente
for resultado_rodada in resultados_globais:
    ax.scatter(resultado_rodada[0], resultado_rodada[1], resultado_rodada[2], marker='x', s=100, linewidth=3, color='blue', label='Ótimo da Rodada')

# Plota o ótimo global com uma cor diferente e marcador maior
ax.scatter(resultado_otimo_global[0], resultado_otimo_global[1], resultado_otimo_global[2], marker='x', s=200, linewidth=3, color='red', label='Ótimo Global')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('f(x1, x2) - Hill Climbing (Maximização) - Ótimos por Rodada e Ótimo Global')
ax.legend()

plt.tight_layout()
plt.show()
