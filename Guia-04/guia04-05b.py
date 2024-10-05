import numpy as np

# Matriz A y vector b de ejemplo
A = np.array([[3, -2, 1, 0, 0, 0], 
              [-2, 4, -2, 1, 0, 0], 
              [1, -2, 4, -2, 1, 0],
              [0, 1, -2, 4, -2, 1],
              [0,0,1, -2, 4, -2],
              [0,0,0, 1, -2, 3]
              ])

b = np.array([10, -8, 10, 10, -8, 10])

def get_base(A):
    # Crea una matriz base de ceros del mismo tamaño que A
    return np.zeros_like(A)

def get_U(A):
    # Extrae la matriz U (componentes por encima de la diagonal principal)
    U = get_base(A)
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            U[i, j] = -A[i, j]
    return U

def get_D(A):
    # Extrae la matriz diagonal D
    D = get_base(A)
    np.fill_diagonal(D, np.diag(A))
    return D

def get_L(A):
    # Extrae la matriz L (componentes por debajo de la diagonal principal)
    L = get_base(A)
    n = len(A)
    for i in range(1, n):
        for j in range(i):
            L[i, j] = -A[i, j]
    return L

def get_B(D, L, U, w):
    # Calcula la matriz B del método de relajación
    return np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)

def get_f(D, L, w, b):
    # Calcula el vector f del método de relajación
    return w * np.linalg.inv(D - w * L) @ b

def roll(B, f, x0):
    # Realiza una iteración del método de relajación
    return B @ x0 + f

def main(A, b, x0, e, w):
    U = get_U(A)
    D = get_D(A)
    L = get_L(A)
    B = get_B(D, L, U, w)
    f = get_f(D, L, w, b)

    n = 0
    x_old = x0
    x_new = roll(B, f, x_old)
    
    while np.linalg.norm(x_new - x_old, ord=np.inf) > e:
        x_old = x_new
        x_new = roll(B, f, x_old)
        n += 1
        
    return x_new, n

# Valores iniciales
x0 = np.array([0,0,0,0,0,0])
e = 1e-10 # Tolerancia
w = 1.0    # Factor de relajación

# Ejecuta el método de relajación
sol, iteraciones = main(A, b, x0, e, w)
print("Solución:", sol)
print("Iteraciones:", iteraciones)
