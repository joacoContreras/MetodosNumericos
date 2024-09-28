import numpy as np
import time

# Definición del sistema
n = 50
A = np.zeros((n, n))
B = np.full(n, 5.0)

# Llenar la matriz A según la descripción del problema
for i in range(n):
    A[i, i] = 12
    if i > 0:
        A[i, i-1] = -2
    if i < n-1:
        A[i, i+1] = 1

# Función para calcular el ancho de banda de la matriz A
def calcular_ancho_banda(A):
    n = A.shape[0]
    ancho_banda = 0
    for i in range(n):
        for j in range(n):
            if A[i, j] != 0:
                ancho_banda = max(ancho_banda, abs(j - i))
    return ancho_banda

# Método de Gauss-Seidel optimizado para matrices con ancho de banda
def gauss_seidel_optimizado(A, B, X0, ancho_banda, tolera, iteramax):
    n = A.shape[0]
    X = np.copy(X0)
    diferencia = np.ones(n, dtype=float)
    errado = 2 * tolera

    itera = 0
    while not (errado <= tolera or itera > iteramax):
        for i in range(n):
            suma = 0 
            for j in range(max(0, i-ancho_banda), min(n, i+ancho_banda+1)):
                if i != j:
                    suma -= A[i, j] * X[j]
            nuevo = (B[i] + suma) / A[i, i]
            diferencia[i] = np.abs(nuevo - X[i])
            X[i] = nuevo
        errado = np.max(diferencia)
        itera += 1
    
    return X, itera

# Configuración inicial
X0 = np.zeros(n)
tolera = 1e-5
iteramax = 1000

# Calcular el ancho de banda
ancho_banda = calcular_ancho_banda(A)
print(f"Ancho de banda de la matriz A: {ancho_banda}")

# Resolver el sistema sin optimización (considerando toda la matriz)
start = time.time()
X_sin_opt, itera_sin_opt = gauss_seidel_optimizado(A, B, X0, n-1, tolera, iteramax)
end = time.time()
print("\nSin optimización (considerando todos los elementos de la matriz):")
print('Respuesta X: ', X_sin_opt)
print('Iteraciones: ', itera_sin_opt)
print(f'Tiempo de ejecución: {end - start:.5f} segundos')

# Resolver el sistema con optimización (utilizando solo la banda diferente de cero)
start = time.time()
X_opt, itera_opt = gauss_seidel_optimizado(A, B, X0, ancho_banda, tolera, iteramax)
end = time.time()
print("\nCon optimización (utilizando solo la banda diferente de cero):")
print('Respuesta X: ', X_opt)
print('Iteraciones: ', itera_opt)
print(f'Tiempo de ejecución: {end - start:.5f} segundos')
