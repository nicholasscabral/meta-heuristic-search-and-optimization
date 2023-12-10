import numpy as np
import matplotlib.pyplot as plt
from problemas import definirlimites
from statistics import mode

#Função para candidato no Hill Climbing
def candidato(x, epsilon):
    """
    Gera um candidato próximo ao valor atual x.
    """
    return x + np.random.uniform(low=-epsilon, high=epsilon)

#Função para verificar se está dentro dos limites dos domínios das variáveis
#Fazer para limites diferentes para cada variável
def restricao_caixa(x, limite_inferior, limite_superior):
    """
    Aplica a restrição de caixa aos valores de x.
    """
    return np.maximum(limite_inferior, np.minimum(x, limite_superior))

def anneal_schedule(T):
    alpha = 0.99
    return alpha * T

#ENCONTRAR ÓTIMO GLOBAL
def encontrar_otimo_global(tipo_problema, resultados_por_rodada):
    """
    Encontra o ótimo global entre todas as rodadas.
    """
    if (tipo_problema == "minimizacao"):
        resultados_globais = [min(resultados, key=lambda x: x[2]) for resultados in resultados_por_rodada]
        resultado_otimo_global = min(resultados_globais, key=lambda x: x[2])
    else:
        resultados_globais = [max(resultados, key=lambda x: x[2]) for resultados in resultados_por_rodada]
        resultado_otimo_global = max(resultados_globais, key=lambda x: x[2])

    return resultado_otimo_global, resultados_globais

def plotar_grafico(limites, funcao_objetivo, tipo_problema, resultados_globais, resultado_otimo_global, algoritmo, nome):
    """
    Cria um gráfico 3D da função objetivo com destaque para os ótimos globais.
    """
    limite_inferior_x1, limite_superior_x1, limite_inferior_x2, limite_superior_x2 = definirlimites(limites)

    x1_vals = np.linspace(limite_inferior_x1, limite_superior_x1, 100)
    x2_vals = np.linspace(limite_inferior_x2, limite_superior_x2, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Y = funcao_objetivo(X1, X2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='viridis')

    # Plota os ótimos de cada rodada de forma mais transparente
    for resultado in resultados_globais:
        ax.scatter(resultado[0], resultado[1], resultado[2], marker='o', s=30, linewidth=1, color='green', alpha=0.2)

    # Plota o ótimo global de forma mais destacada
    ax.scatter(resultado_otimo_global[0], resultado_otimo_global[1], resultado_otimo_global[2], marker='*', s=200, linewidth=3, color='black', label='Ótimo Global')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title(f'{nome} para Problemas de {tipo_problema} - Ótimos por Rodada e Ótimo Global')
    ax.legend()

    plt.tight_layout()
    plt.show()