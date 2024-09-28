import numpy as np

def generar_matriz_sistema(n):
    A = np.zeros((n, n))
    b = np.zeros(n)
    
    for i in range(n):
        if i == 0:
            A[i, i] = 2
            A[i, i + 1] = 1
            b[i] = 4.5
        elif i == n - 1:
            A[i, i] = 2
            A[i, i - 1] = 1
            b[i] = 4.5
        else:
            A[i, i] = 2
            A[i, i - 1] = 1
            A[i, i + 1] = 1
            b[i] = 6
    
    return A, b

# Generar la matriz 100x100 y el vector b
A, b = generar_matriz_sistema(100)

def gauss_seidel(A, b, tol=1e-8, max_iter=10000):
    n = len(b)
    x = np.zeros(n)
    for iter in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, iter + 1
        x = x_new
    raise Exception("No convergió en el número máximo de iteraciones")

# Resolver el sistema usando Gauss-Seidel
solucion, iteraciones = gauss_seidel(A, b)

print("Solución del sistema:", solucion)
print("Número de iteraciones:", iteraciones)

#El método de Gauss-Seidel es una buena opción aquí porque la matriz es 
# diagonalmente dominante, lo que garantiza la convergencia.