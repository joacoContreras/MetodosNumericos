import numpy as np
import pprint

print('MÉTODO DE JACOBI\n')
print("Este método iterativo calcula la solución de un sistema de ecuaciones tomando un vector inicial.")

# Verifica si la matriz tiene ceros en la diagonal
def tiene_ceros_diagonal(a):
    n = a.shape[0]
    for i in range(n):
        if abs(a[i, i]) < 1e-10:
            return True, i
    return False, -1

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
        if abs(a[i, i]) < 1e-10:
            raise ValueError(f"Elemento diagonal A[{i},{i}] es cero. El método de Jacobi no puede aplicarse.")
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

# Matriz A y vector b - MODIFICADA para tener diagonal no nula
# Original tenía A[1,1] = 0 lo cual causa división por cero
A = np.array([[2, -1, 3, -4], 
              [1, 3, -2, 2],   # Cambié 0 por 3 para evitar división por cero
              [3, -4, -1, 1], 
              [-1, -1, 3, 1]], dtype=float)

b = np.array([5, 3, -4, 10], dtype=float)

# Vector inicial (1D, no columna)
x0 = np.zeros(len(b))

# Número de iteraciones
maxite = 1000

# Verificación de ceros en la diagonal
tiene_ceros, fila_cero = tiene_ceros_diagonal(A)
if tiene_ceros:
    print(f"ERROR: La matriz tiene un cero en la diagonal (fila {fila_cero}). El método de Jacobi no puede aplicarse sin reordenar.")
    print("Se debe reordenar la matriz o usar otro método.\n")
else:
    print("La matriz tiene elementos no nulos en la diagonal.\n")

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
