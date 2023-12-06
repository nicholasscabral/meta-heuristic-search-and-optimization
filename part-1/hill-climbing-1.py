import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def candidato(x, epsilon):
    return x + np.random.uniform(low=-epsilon, high=epsilon)

def restricao_caixa(x, limite_inferior, limite_superior):
    return np.maximum(limite_inferior, np.minimum(x, limite_superior))

def funcao_objetivo(x1, x2):
    return x1**2 + x2**2

def hill_climbing(limite_inferior, limite_superior, epsilon, maxit):
    x1_best = np.random.uniform(low=limite_inferior, high=limite_superior)
    x2_best = np.random.uniform(low=limite_inferior, high=limite_superior)
    fbest = funcao_objetivo(x1_best, x2_best)

    for _ in range(maxit):
        x1_candidate = restricao_caixa(candidato(x1_best, epsilon), limite_inferior, limite_superior)
        x2_candidate = restricao_caixa(candidato(x2_best, epsilon), limite_inferior, limite_superior)
        f_candidate = funcao_objetivo(x1_candidate, x2_candidate)

        if f_candidate < fbest:
            x1_best, x2_best = x1_candidate, x2_candidate
            fbest = f_candidate

    return x1_best, x2_best, fbest

# Substitua os valores apropriados
limite_inferior = -100
limite_superior = 100
epsilon = 0.1
maxit = 1000
rodadas = 100

# Executa o algoritmo Hill Climbing em 100 rodadas
resultados_otimos_globais = []

for _ in range(rodadas):
    resultado_otimo_global = hill_climbing(limite_inferior, limite_superior, epsilon, maxit)
    #print(resultado_otimo_global)
    resultados_otimos_globais.append(resultado_otimo_global)

# Encontra o resultado ótimo global entre todas as rodadas
resultado_otimo_global = min(resultados_otimos_globais, key=lambda x: x[2])
print(resultado_otimo_global)

# Criação do gráfico
x1 = np.linspace(-100, 100, 1000)
X1, X2 = np.meshgrid(x1, x1)
Y = funcao_objetivo(X1, X2)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')

#for resultado_otimo in resultados_otimos_globais:
#    ax.scatter(resultado_otimo[0], resultado_otimo[1], resultado_otimo[2], marker='x', s=90, linewidth=3, color='green')

# Destaca o resultado ótimo global com uma cor diferente
ax.scatter(resultado_otimo_global[0], resultado_otimo_global[1], resultado_otimo_global[2], marker='x', s=90, linewidth=3, color='black', label='Ótimo Global')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('f(x1, x2) - Hill Climbing')
ax.legend()

plt.tight_layout()
plt.show()
