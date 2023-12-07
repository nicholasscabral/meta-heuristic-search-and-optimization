import numpy as np
import random
from funcoes import candidato, restricao_caixa, anneal_schedule
from problemas import definirlimites

def hill_climbing(limites, funcao_objetivo, tipo_problema, epsilon=0.1, maxit=1000):
    """
    Executa o algoritmo Hill Climbing para maximizar a função objetivo.
    """
    limite_inferior_x1, limite_superior_x1, limite_inferior_x2, limite_superior_x2 = definirlimites(limites)

    x1_best = np.random.uniform(low=limite_inferior_x1, high=limite_superior_x1)
    x2_best = np.random.uniform(low=limite_inferior_x2, high=limite_superior_x2)
    fbest = funcao_objetivo(x1_best, x2_best)

    resultados = []
    i = 0
    t = 0

    for _ in range(maxit): 
        while t < 1000:
            melhoria = False
            x1_candidate = restricao_caixa(candidato(x1_best, epsilon), limite_inferior_x1, limite_superior_x1)
            x2_candidate = restricao_caixa(candidato(x2_best, epsilon), limite_inferior_x2, limite_superior_x2)
            f_candidate = funcao_objetivo(x1_candidate, x2_candidate)

            if (tipo_problema == "maximizacao"):
                if f_candidate > fbest:
                    x1_best, x2_best = x1_candidate, x2_candidate
                    fbest = f_candidate
                    melhoria = True
                    break
            else:
                if f_candidate < fbest:
                    x1_best, x2_best = x1_candidate, x2_candidate
                    fbest = f_candidate
                    melhoria = True
                    break

            if melhoria:
                t = 0
            else:
                t += 1
            
            i += 1

            resultados.append((x1_best, x2_best, fbest))

    return resultados

def local_random_search(limites, funcao_objetivo, tipo_problema, epsilon=0.1, maxit=1000):
    limite_inferior_x1, limite_superior_x1, limite_inferior_x2, limite_superior_x2 = definirlimites(limites)

    x1_best = np.random.uniform(low=limite_inferior_x1, high=limite_superior_x1)
    x2_best = np.random.uniform(low=limite_inferior_x2, high=limite_superior_x2)
    fbest = funcao_objetivo(x1_best, x2_best)

    resultados = []
    i = 0
    t = 0

    for _ in range(maxit): 
        while t < 1000:
            melhoria = False
            # Gera um perturbação aleatória
            #sigma = round(random.uniform(0, 1))
            perturbacao = np.random.normal(0, 0.1, size=2)
            #print(perturbacao)
            x1_best_perturbado = x1_best + perturbacao[0]
            x2_best_perturbado = x2_best + perturbacao[1]

            x1_candidate = restricao_caixa(candidato(x1_best_perturbado, epsilon), limite_inferior_x1, limite_superior_x1)
            x2_candidate = restricao_caixa(candidato(x2_best_perturbado, epsilon), limite_inferior_x2, limite_superior_x2)

            f_candidate = funcao_objetivo(x1_candidate, x2_candidate)

            if (tipo_problema == "maximizacao"):
                if f_candidate > fbest:
                    x1_best, x2_best = x1_candidate, x2_candidate
                    fbest = f_candidate
                    melhoria = True
                    break
            else:
                if f_candidate < fbest:
                    x1_best, x2_best = x1_candidate, x2_candidate
                    fbest = f_candidate
                    melhoria = True
                    break

            if melhoria:
                t = 0
            else:
                t += 1
            
            i += 1

            resultados.append((x1_best, x2_best, fbest))

    return resultados

def global_random_search(limites, funcao_objetivo, tipo_problema, maxit=1000):
    limite_inferior_x1, limite_superior_x1, limite_inferior_x2, limite_superior_x2 = definirlimites(limites)

    x1_best = np.random.uniform(low=limite_inferior_x1, high=limite_superior_x1)
    x2_best = np.random.uniform(low=limite_inferior_x2, high=limite_superior_x2)
    fbest = funcao_objetivo(x1_best, x2_best)

    resultados = []
    i = 0
    t = 0

    for _ in range(maxit): 
        while t < 1000:
            melhoria = False
            x1_candidate = np.random.uniform(low=limite_inferior_x1, high=limite_superior_x1)
            x2_candidate = np.random.uniform(low=limite_inferior_x2, high=limite_superior_x2)

            f_candidate = funcao_objetivo(x1_candidate, x2_candidate)

            if (tipo_problema == "maximizacao"):
                if f_candidate > fbest:
                    x1_best, x2_best = x1_candidate, x2_candidate
                    fbest = f_candidate
                    melhoria = True
                    break
            else:
                if f_candidate < fbest:
                    x1_best, x2_best = x1_candidate, x2_candidate
                    fbest = f_candidate
                    melhoria = True
                    break

            if melhoria:
                t = 0
            else:
                t += 1
            
            i += 1

            resultados.append((x1_best, x2_best, fbest))

    return resultados

def simulated_annealing(limites, funcao_objetivo, tipo_problema, epsilon=0.1, maxit=1000, T0=100):
    limite_inferior_x1, limite_superior_x1, limite_inferior_x2, limite_superior_x2 = definirlimites(limites)

    x1_best = np.random.uniform(low=limite_inferior_x1, high=limite_superior_x1)
    x2_best = np.random.uniform(low=limite_inferior_x2, high=limite_superior_x2)
    fbest = funcao_objetivo(x1_best, x2_best)

    resultados = []
    i = 0
    t = 0
    T = T0

    for _ in range(maxit): 
        while t < 1000:
            melhoria = False
            # Gera um perturbação aleatória
            #sigma = round(random.uniform(0, 1))
            perturbacao = np.random.normal(0, 1, size=2)
            #print(perturbacao)
            x1_best_perturbado = x1_best + perturbacao[0]
            x2_best_perturbado = x2_best + perturbacao[1]

            x1_candidate = restricao_caixa(candidato(x1_best_perturbado, epsilon), limite_inferior_x1, limite_superior_x1)
            x2_candidate = restricao_caixa(candidato(x2_best_perturbado, epsilon), limite_inferior_x2, limite_superior_x2)

            f_candidate = funcao_objetivo(x1_candidate, x2_candidate)

            if (tipo_problema == "maximizacao" and f_candidate > fbest) or (tipo_problema == "minimizacao" and f_candidate < fbest):
                x1_best, x2_best = x1_candidate, x2_candidate
                fbest = f_candidate
                melhoria = True
                break

            elif np.random.rand() < np.exp(-(f_candidate - fbest) / T):
                x1_best, x2_best = x1_candidate, x2_candidate
                fbest = f_candidate

        i += 1

        if melhoria:
            t = 0
        else:
            t += 1

        T = anneal_schedule(T)

        resultados.append((x1_best, x2_best, fbest))

    return resultados