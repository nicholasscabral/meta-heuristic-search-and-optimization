import numpy as np
from problemas import problema1, problema2, problema3, problema4, problema5, problema6, problema7, problema8
from algoritmos import hill_climbing, local_random_search, global_random_search, simulated_annealing
from funcoes import encontrar_otimo_global, plotar_grafico
from prettytable import PrettyTable

def executar_algoritmo(problema, algoritmo, nome):
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
    plotar_grafico(limites, funcao_objetivo, tipo_problema, resultados_globais, resultado_otimo_global, algoritmo, nome)

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

titulo_algoritmos = ["Hill Climbing", "Local Random Search", "Global Random Search", "Simulated Annealing"]

#EXECUÇÕES DO PROBLEMA 1
#moda_hill1 = executar_algoritmo(problema_1, hill_climbing, titulo_algoritmos[0])
#moda_local1 = executar_algoritmo(problema_1, local_random_search, titulo_algoritmos[1])
#moda_global1 = executar_algoritmo(problema_1, global_random_search, titulo_algoritmos[2])
#moda_annealing1 = executar_algoritmo(problema_1, simulated_annealing, titulo_algoritmos[3])

#EXECUÇÕES DO PROBLEMA 2
#moda_hill2 = executar_algoritmo(problema_2, hill_climbing, titulo_algoritmos[0])
#moda_local2 = executar_algoritmo(problema_2, local_random_search, titulo_algoritmos[1])
#moda_global2 = executar_algoritmo(problema_2, global_random_search, titulo_algoritmos[2])
#moda_annealing2 = executar_algoritmo(problema_2, simulated_annealing, titulo_algoritmos[3])

#EXECUÇÕES DO PROBLEMA 3
#moda_hill3 = executar_algoritmo(problema_3, hill_climbing, titulo_algoritmos[0])
#moda_local3 = executar_algoritmo(problema_3, local_random_search, titulo_algoritmos[1])
#moda_global3 = executar_algoritmo(problema_3, global_random_search, titulo_algoritmos[2])
moda_annealing3 = executar_algoritmo(problema_3, simulated_annealing, titulo_algoritmos[3])

#EXECUÇÕES DO PROBLEMA 4
#moda_hill4 = executar_algoritmo(problema_4, hill_climbing, titulo_algoritmos[0])
#moda_local4 = executar_algoritmo(problema_4, local_random_search, titulo_algoritmos[1])
#moda_global4 = executar_algoritmo(problema_4, global_random_search, titulo_algoritmos[2])
moda_annealing4 = executar_algoritmo(problema_4, simulated_annealing, titulo_algoritmos[3])

#EXECUÇÕES DO PROBLEMA 5
#moda_hill5 = executar_algoritmo(problema_5, hill_climbing, titulo_algoritmos[0])
#moda_local5 = executar_algoritmo(problema_5, local_random_search, titulo_algoritmos[1])
#moda_global5 = executar_algoritmo(problema_5, global_random_search, titulo_algoritmos[2])
moda_annealing5 = executar_algoritmo(problema_5, simulated_annealing, titulo_algoritmos[3])

#EXECUÇÕES DO PROBLEMA 6
#moda_hill6 = executar_algoritmo(problema_6, hill_climbing, titulo_algoritmos[0])
#moda_local6 = executar_algoritmo(problema_6, local_random_search, titulo_algoritmos[1])
#moda_global6 = executar_algoritmo(problema_6, global_random_search, titulo_algoritmos[2])
moda_annealing6 = executar_algoritmo(problema_6, simulated_annealing, titulo_algoritmos[3])

#EXECUÇÕES DO PROBLEMA 7
#moda_hill7 = executar_algoritmo(problema_7, hill_climbing, titulo_algoritmos[0])
#moda_local7 = executar_algoritmo(problema_7, local_random_search, titulo_algoritmos[1])
#moda_global7 = executar_algoritmo(problema_7, global_random_search, titulo_algoritmos[2])
moda_annealing7 = executar_algoritmo(problema_7, simulated_annealing, titulo_algoritmos[3])

#EXECUÇÕES DO PROBLEMA 8
#moda_hill8 = executar_algoritmo(problema_8, hill_climbing, titulo_algoritmos[0])
#moda_local8 = executar_algoritmo(problema_8, local_random_search, titulo_algoritmos[1])
#moda_global8 = executar_algoritmo(problema_8, global_random_search, titulo_algoritmos[2])
moda_annealing8 = executar_algoritmo(problema_8, simulated_annealing, titulo_algoritmos[3])




