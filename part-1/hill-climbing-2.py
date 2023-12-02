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
todos_resultados = []

for _ in range(rodadas):
    resultados = hill_climbing(limite_inferior, limite_superior, epsilon, maxit)
    todos_resultados.extend(resultados)

# Encontra o resultado ótimo global
resultado_otimo_global = max(todos_resultados, key=lambda x: x[2])

# Criação do gráfico
x1 = np.linspace(-2, 4, 1000)
X1, X2 = np.meshgrid(x1, x1)
Y = funcao_objetivo(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')

# Extrai as coordenadas de x, y e z dos resultados
x_points = [result[0] for result in todos_resultados]
y_points = [result[1] for result in todos_resultados]
z_points = [result[2] for result in todos_resultados]

# Plota todos os pontos encontrados durante as 1000 iterações de cada rodada
ax.scatter(x_points, y_points, z_points, c='green', marker='o', alpha=0.1, label='Pontos das Iterações')

# Plota os ótimos encontrados em cada rodada com um marcador maior e uma cor diferente
for i in range(0, len(todos_resultados), maxit):
    resultado_otimo_rodada = max(todos_resultados[i:i+maxit], key=lambda x: x[2])
    ax.scatter(resultado_otimo_rodada[0], resultado_otimo_rodada[1], resultado_otimo_rodada[2], marker='x', s=100, linewidth=3, color='blue', label='Ótimo da Rodada' if i == 0 else "")

# Plota a solução ótima global com uma cor diferente e marcador maior
ax.scatter(resultado_otimo_global[0], resultado_otimo_global[1], resultado_otimo_global[2], marker='x', s=200, linewidth=3, color='red', label='Ótimo Global')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('f(x1, x2) - Hill Climbing (Maximização) - Iterações e Ótimos')
ax.legend()

plt.tight_layout()
plt.show()
