import numpy as np
from problemas import problema1, problema2, problema3, problema4, problema5, problema6, problema7, problema8
from algoritmos import hill_climbing, local_random_search, global_random_search, simulated_annealing
from funcoes import encontrar_otimo_global, plotar_grafico

def executar_algoritmo(problema, algoritmo):
    # Substitua os valores apropriados
    limites = problema["dominios"]
    funcao_objetivo = problema["funcao_objetivo"]
    tipo_problema = problema["tipo_problema"]
    rodadas = 100

    # Executa o algoritmo Hill Climbing
    resultados_por_rodada = [algoritmo(limites, funcao_objetivo, tipo_problema) for _ in range(rodadas)]

    # Encontra o ótimo global entre as rodadas
    resultado_otimo_global, resultados_globais = encontrar_otimo_global(tipo_problema, resultados_por_rodada)

    # Criação do gráfico da função
    plotar_grafico(limites, funcao_objetivo, tipo_problema, resultados_globais, resultado_otimo_global, algoritmo)


problema_1 = {
    "dominios": [(-100, 100), (-100, 100)],
    "funcao_objetivo": problema1,
    "tipo_problema": "minimizacao",
}

problema_2 = {
    "dominios": [(-2, 4), (-2, 4)],
    "funcao_objetivo": problema2,
    "tipo_problema": "maximizacao",
}

problema_3 = {
    "dominios": [(-8, 8), (-8, 8)],
    "funcao_objetivo": problema3,
    "tipo_problema": "minimizacao",
}

problema_4 = {
    "dominios": [(-5.12, 5.12), (-5.12, 5.12)],
    "funcao_objetivo": problema4,
    "tipo_problema": "minimizacao",
}

problema_5 = {
    "dominios": [(-2, 2), (-1, 3)],
    "funcao_objetivo": problema5,
    "tipo_problema": "minimizacao",
}

problema_6 = {
    "dominios": [(-1, 3), (-1, 3)],
    "funcao_objetivo": problema6,
    "tipo_problema": "maximizacao",
}

problema_7 = {
    "dominios": [(0, np.pi), (0, np.pi)],
    "funcao_objetivo": problema7,
    "tipo_problema": "minimizacao",
}

problema_8 = {
    "dominios": [(-200, 20), (-200, 20)],
    "funcao_objetivo": problema8,
    "tipo_problema": "minimizacao",
}

#EXECUÇÕES DO HILL CLIMBING
executar_algoritmo(problema_6, hill_climbing)
#executar_algoritmo(problema_2)
#executar_algoritmo(problema_3)
#executar_algoritmo(problema_4)
#executar_algoritmo(problema_5)
#executar_algoritmo(problema_6)
#executar_algoritmo(problema_7)
#executar_algoritmo(problema_8)

#EXECUÇÕES DO LOCAL RANDOM SEARCH



