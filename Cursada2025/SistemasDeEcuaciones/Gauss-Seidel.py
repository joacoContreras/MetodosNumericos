import numpy as np

A = np.array([[5, 1, -2],
              [1, -4, 2],
              [0, 2, 5]], dtype=float)

b = np.array([16, -19, -14], dtype=float)
x0 = np.zeros_like(b)

tol = 1e-4
maxiter = 1000

def es_diagonal_dominante(A):
    n = A.shape[0]
    for i in range(n):
        suma = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) <= suma:
            return False
    return True

def gauss_seidel(A, b, x0, tol=1e-8, maxiter=100, verbose=False):
    n = A.shape[0]
    x = x0.astype(float).copy()
    diff = np.full(n, np.inf)
    iter_count = 0

    while True:
        iter_count += 1
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i, j] * x[j]   # usa valores ya actualizados (Gauss-Seidel)
            new = (b[i] - s) / A[i, i]
            diff[i] = abs(new - x[i])
            x[i] = new

        err = np.max(diff)
        if verbose:
            print(f"Iter {iter_count:2d}: x = {x}, err = {err:.6e}")

        if err < tol or iter_count >= maxiter:
            break

    return x, iter_count, err

if not es_diagonal_dominante(A):
    print("Advertencia: la matriz no es diagonalmente dominante; Gauss-Seidel puede no converger.")

x_sol, iters, err = gauss_seidel(A, b, x0, tol=tol, maxiter=maxiter, verbose=True)

print("\nResultado final:")
print("x =", x_sol)
print("A·x =", A.dot(x_sol))
print("b    =", b)
print("Iteraciones:", iters)
print("Error estimado (inf-norma de la diferencia):", err)
print("Residual ||A x - b||_inf:", np.linalg.norm(A.dot(x_sol) - b, np.inf))
