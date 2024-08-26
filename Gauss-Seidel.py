import numpy as np

# INGRESO
A = np.array([[3. , -0.1, -0.2],
              [0.1,  7  , -0.3],
              [0.3, -0.2, 10  ]])

B = np.array([7.85,-19.3,71.4])

X0  = np.array([0.,0.,0.])

tolera = 0.00001
iteramax = 100

# PROCEDIMIENTO

# Verificación de la diagonal dominante
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
tamano = np.shape(A)
n = tamano[0]
m = tamano[1]
#  valores iniciales
X = np.copy(X0)
diferencia = np.ones(n, dtype=float)
errado = 2 * tolera

itera = 0
while not(errado <= tolera or itera > iteramax):
    # por fila
    for i in range(0, n, 1):
        # por columna
        suma = 0 
        for j in range(0, m, 1):
            # excepto diagonal de A
            if (i != j): 
                suma -= A[i, j] * X[j]
        
        nuevo = (B[i] + suma) / A[i, i]
        diferencia[i] = np.abs(nuevo - X[i])
        X[i] = nuevo
    errado = np.max(diferencia)
    itera += 1

# Respuesta X en columna
X = np.transpose([X])

# revisa si NO converge
if (itera > iteramax):
    X = 0

# revisa respuesta
verifica = np.dot(A, X)

# SALIDA
print('Respuesta X: ')
print(X)
print('Verificar A.X=B: ')
print(verifica)
