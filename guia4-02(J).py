import numpy as np
import pprint

print('MÉTODO DE JACOBI', end="\n\n")
print("Este método iterativo te cálcula la solución de un sistema de ecuaciones tomando un vector inicial.")

# Función que verifica si una matriz es diagonalmente dominante
def es_diagonal_dominante(a):
    n = a.shape[0]
    for i in range(n):
        suma_fila = sum(abs(a[i, j]) for j in range(n) if i != j)
        if abs(a[i, i]) <= suma_fila:
            return False
    return True

def jacobi(a, b, x): 
    n = len(x) 
    t = x.copy()
    for i in range(n): 
        s = 0
        for j in range(n): 
            if i != j:
                s += a[i, j] * t[j]
        x[i] = (b[i] - s) / a[i, i]
    return x

def jacobim(a, b, x, e, m): 
    n = len(x)  
    t = x.copy()
    for k in range(m): 
        x = jacobi(a, b, x)
        d = np.linalg.norm(np.array(x) - np.array(t), np.inf)
        if d < e:
            return [x, k] 
        else:
            t = x.copy() 
    return [[], m]

# Matriz a usar
A = np.array([[3,1,1],
              [2,6,1],
              [1,1,4]], float)

# Vector Solución
b = np.array([[5], [9], [6]], float)

# Vector de Inicio
x = np.array([[0], [0], [0]], float)

# Número de iteraciones
maxite = 1000

# Verificación de la diagonal dominante
if not es_diagonal_dominante(A):
    print("Advertencia: La matriz no es diagonalmente dominante. El método de Jacobi puede no converger.")
else:
    print("La matriz es diagonalmente dominante, el método de Jacobi debería converger.")

print("\nMatriz A:")
pprint.pprint(A)
print("\nVector b:")
pprint.pprint(b)
print("")

# X es la solución y k las iteraciones
[x, k] = jacobim(A, b, x, 1e-4, maxite)

if k == maxite:
    print("\nEl método diverge o no converge para la cota de error pedido")
else: 
    print("\nEl vector 'x' es:")
    print(x)
    print("\nEl número de iteraciones es: " + str(k + 1))
