import numpy as np

# INGRESO
A = np.array([[3, 1, 1],
              [2, 6, 1],
              [1, 1, 4]])

B = np.array([5, 9, 6])

X0  = np.array([0., 0., 0.])

tolera = 1e-11
iteramax = 100

# PROCEDIMIENTO
#Gauss - Seidel modificado
#Este cambio hace que el método de Gauss-Seidel verifique 
# su convergencia en función de si la solución cumple la 
# ecuación 𝐴𝑥𝑘 = b 
# dentro de la tolerancia establecida. 
# Este criterio es más riguroso que simplemente verificar la 
# diferencia entre iteraciones consecutivas.

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

itera = 0
residual = np.linalg.norm(np.dot(A, X) - B)

while residual > tolera and itera < iteramax:
    # Por fila
    for i in range(n):
        suma = 0 
        for j in range(m):
            if i != j: 
                suma -= A[i, j] * X[j]
        
        nuevo = (B[i] + suma) / A[i, i]
        X[i] = nuevo
    
    # Calcular el residual
    residual = np.linalg.norm(np.dot(A, X) - B)
    itera += 1

# Respuesta X en columna
X = np.transpose([X])

# Revisa si NO converge
if itera >= iteramax:
    print("El método no converge después del número máximo de iteraciones.")
else:
    print('Respuesta X: ')
    print(X)
    print('Verificar A.X=B: ')
    print(np.dot(A, X))
    print('Iteraciones: ')
    print(itera)
