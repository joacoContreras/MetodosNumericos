import numpy as np

# Matriz A y vector b de ejemplo
A = np.array([[10.0, 3.0, 1.0], 
              [2.0, -10.0, 3.0], 
              [1.0, 3.0, 10.0]])

b = np.array([14.0, -5.0, 14.0])

def get_base(A):
    return np.zeros_like(A, dtype=float)

def get_U(A):
    U = get_base(A)
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            U[i, j] = -A[i, j]
    return U

def get_D(A):
    D = get_base(A)
    for i in range(len(A)):
        D[i, i] = A[i, i]
    return D

def get_L(A):
    L = get_base(A)
    n = len(A)
    for i in range(1, n):
        for j in range(i):
            L[i, j] = -A[i, j]
    return L

def get_B(D, L, U, w):
    return np.linalg.inv(D - w * L) @ ((1 - w) * D + w * U)

def get_f(D, L, w, b):
    return w * np.linalg.inv(D - w * L) @ b

def roll(B, f, x0):
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
x0 = np.array([1.0, 1.0, 2.0])
e = 1e-5
w = 1.0  # Jacobi si w=1, Gauss-Seidel si w=1, SOR si w != 1

# Ejecuta el método
sol, iteraciones = main(A, b, x0, e, w)
print("Solución:", sol)
print("Iteraciones:", iteraciones)
