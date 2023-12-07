import numpy as np

def problema1 (x1, x2):
    return x1**2 + x2**2

def problema2 (x1, x2):
    return np.exp(-(x1**2 + x2**2)) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))

def problema3 (x1, x2):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)

def problema4 (x1, x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)

def problema5 (x1, x2):
    return (x1 - 1)**2 + 100 * (x2 - x1**2)**2

def problema6 (x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

def problema7 (x1, x2):
    termo1 = -np.sin(x1) * (np.sin((x1**2) / np.pi))**2 * 1e-10
    termo2 = -np.sin(x2) * (np.sin((2 * x2**2) / np.pi))**2 * 1e-10
    return termo1 + termo2

def problema8 (x1, x2):
    return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1/2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

def definirlimites (dominios):
    limiteInferiorX1 = dominios[0][0]
    limiteSuperiorX1 = dominios[0][1]
    limiteInferiorX2 = dominios[1][0]
    limiteSuperiorX2 = dominios[1][1]
    return limiteInferiorX1, limiteSuperiorX1, limiteInferiorX2, limiteSuperiorX2