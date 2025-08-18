import numpy as np
import pprint

print('MÉTODO DE JACOBI\n')
print("Este método iterativo calcula la solución de un sistema de ecuaciones tomando un vector inicial.")

# Verifica si la matriz es diagonalmente dominante
def es_diagonal_dominante(a):
    n = a.shape[0]
    for i in range(n):
        suma_fila = sum(abs(a[i, j]) for j in range(n) if i != j)
        if abs(a[i, i]) <= suma_fila:
            return False
    return True

# Una iteración del método de Jacobi
def jacobi(a, b, x):
    n = len(x)
    nuevo_x = np.zeros_like(x)
    for i in range(n):
        suma = sum(a[i, j] * x[j] for j in range(n) if j != i)
        nuevo_x[i] = (b[i] - suma) / a[i, i]
    return nuevo_x

# Método iterativo completo
def jacobim(a, b, x0, tol, max_iter):
    x = x0.copy()
    for k in range(max_iter):
        x_nuevo = jacobi(a, b, x)
        # Norma infinito para medir error
        d = np.linalg.norm(x_nuevo - x, np.inf)
        if d < tol:
            return x_nuevo, k + 1
        x = x_nuevo
    return x, max_iter

# Matriz A y vector b
A = np.array([[2, -1, 3, -4], 
              [1, 0, -2, 2], 
              [3, -4, -1, 1], 
              [-1, -1, 3, 1]], dtype=float)

b = np.array([5, 3, -4, 10], dtype=float)

# Vector inicial (1D, no columna)
x0 = np.zeros(len(b))

# Número de iteraciones
maxite = 1000

# Verificación de diagonal dominante
if not es_diagonal_dominante(A):
    print("Advertencia: La matriz no es diagonalmente dominante. El método de Jacobi puede no converger.")
else:
    print("La matriz es diagonalmente dominante, el método de Jacobi debería converger.")

print("\nMatriz A:")
pprint.pprint(A)
print("\nVector b:")
pprint.pprint(b)
print("")

# Ejecución del método
x, k = jacobim(A, b, x0, 1.e-14, maxite)

if k == maxite:
    print("\nEl método diverge o no converge para la cota de error pedida")
else:
    print("\nEl vector 'x' es:")
    print(x)
    print("\nEl número de iteraciones es:", k)
