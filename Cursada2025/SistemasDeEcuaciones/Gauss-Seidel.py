import numpy as np

# INGRESO
A = np.array([[5, 1, -2],
              [1, -4, 2],
              [0, 2, 5]], dtype=float)

B = np.array([16, -19, -14], dtype=float)
X0 = np.zeros_like(B)

tolera = 1e-4
iteramax = 1000

# Verificación de diagonal dominante
def es_diagonal_dominante(A):
    n = A.shape[0]
    for i in range(n):
        suma_fila = sum(abs(A[i, j]) for j in range(n) if i != j)
        if abs(A[i, i]) <= suma_fila:
            return False
    return True

if not es_diagonal_dominante(A):
    print("Advertencia: La matriz no es diagonalmente dominante. El método de Gauss-Seidel puede no converger.")

# Gauss-Seidel
n = A.shape[0]
X = X0.copy()
diferencia = np.ones(n, dtype=float)
errado = 2 * tolera
itera = 0

while errado > tolera and itera < iteramax:
    for i in range(n):
        suma = 0
        for j in range(n):
            if i != j:
                suma += A[i, j] * X[j]
        nuevo = (B[i] - suma) / A[i, i]
        diferencia[i] = abs(nuevo - X[i])
        X[i] = nuevo
    errado = np.max(diferencia)
    itera += 1

# Respuesta
if itera == iteramax:
    print("No converge en el número máximo de iteraciones")
else:
    print("Respuesta X:")
    print(X)
    print("Verificar A.X = B:")
    print(np.dot(A, X))
    print("Iteraciones:", itera)
    print("Error estimado:", errado)
