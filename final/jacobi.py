"""Método de Jacobi para resolver un sistema lineal A*x = b.

Ejecutar: python final/jacobi.py
Uso general:
    x, iteraciones, error = jacobi(A, b, x0, tol=1e-3, mostrar=False)

Cambiar A, b, x0, tol y max_iter al final para resolver otro ejercicio.
A debe ser cuadrada y no tener ceros en su diagonal. La convergencia depende
del sistema: está garantizada para todo x0 si el radio espectral es menor que 1.
No se reordenan las ecuaciones automáticamente.

Criterio de parada: max(abs(x_nuevo - x_anterior)) < tol.
Este error es el cambio entre iteraciones, no el error real de la solución.
"""

import numpy as np


def jacobi(A, b, x0=None, tol=1e-3, max_iter=1000, mostrar=True):
    """Devuelve aproximación, iteraciones y último cambio en norma infinito.

    Si x0 es None, comienza con ceros. Si no alcanza la tolerancia,
    lanza RuntimeError en lugar de presentar el último vector como solución.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A debe ser una matriz cuadrada no vacía.")
    n = len(A)
    if b.shape != (n,):
        raise ValueError("b debe ser una lista con un término por ecuación.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("Los datos deben ser números finitos.")
    if np.any(np.diag(A) == 0):
        raise ValueError("Hay ceros en la diagonal: reordenar las ecuaciones antes de usar Jacobi.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("La tolerancia debe ser positiva y finita.")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter debe ser un entero positivo.")

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float, copy=True)
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        raise ValueError("x0 debe contener un valor inicial finito por incógnita.")

    if mostrar:
        # Matriz de iteración T: x_nuevo = T*x_anterior + D^(-1)*b.
        T = -A / np.diag(A)[:, None]
        np.fill_diagonal(T, 0)
        radio = np.max(np.abs(np.linalg.eigvals(T)))
        print(f"Radio espectral de Jacobi = {radio:.6f} (converge para todo x0 si es < 1)")
        print(f"Parada: max|x_nuevo - x_anterior| < {tol:g}")
        encabezado = "   k" + "".join(f"{'x' + str(i + 1):>13}" for i in range(n))
        print(encabezado + f"{'Cambio max':>14}{'Residuo max':>14}")
        residuo = np.max(np.abs(A @ x - b))
        print(f"{0:4d}" + "".join(f"{xi:13.6f}" for xi in x)
              + f"{'-':>14}{residuo:14.6e}")

    for k in range(1, max_iter + 1):
        x_nuevo = np.zeros(n)
        for i in range(n):
            # Todos los componentes se calculan con la iteración ANTERIOR.
            # x_i^(k+1) = (b_i - sum(A_ij*x_j^k, j != i)) / A_ii.
            suma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_nuevo[i] = (b[i] - suma) / A[i, i]

        if not np.all(np.isfinite(x_nuevo)):
            raise RuntimeError("La iteración produjo valores no finitos; no se obtuvo convergencia.")
        error = np.max(np.abs(x_nuevo - x))
        residuo = np.max(np.abs(A @ x_nuevo - b))

        if mostrar:
            print(f"{k:4d}" + "".join(f"{xi:13.6f}" for xi in x_nuevo)
                  + f"{error:14.6e}{residuo:14.6e}")

        if error < tol:
            return x_nuevo, k, error

        x = x_nuevo

    raise RuntimeError(f"No se alcanzó la tolerancia en {max_iter} iteraciones. "
                       f"Último cambio: {error:.6e}; residuo: {residuo:.6e}.")


if __name__ == "__main__":
    np.set_printoptions(precision=9, suppress=True, linewidth=120)

    # PROBLEMA 3. Cada columna corresponde a x, y, z, respectivamente.
    A = np.array([[3, 1, -1],
                  [2, -4, 5],
                  [-1, 2, -3]], dtype=float)
    b = np.array([1, 2, -1], dtype=float)
    x0 = np.zeros(len(b))  # El enunciado pide (0, 0, 0).
    tol = 1e-3
    max_iter = 1000

    print("MÉTODO DE JACOBI")
    print("Matriz A:")
    print(A)
    print("Vector b:", b)
    print("Vector inicial:", x0)
    print(f"Tolerancia: {tol:g}; máximo de iteraciones: {max_iter}\n")

    solucion, iteraciones, error = jacobi(A, b, x0, tol, max_iter)

    print("\nVector aproximado (x1, x2, ..., xn):")
    print(solucion)
    print(f"Iteraciones = {iteraciones}")
    print(f"Cambio máximo entre las últimas iteraciones = {error:.9e}")
    print("Residuo A*x - b:", A @ solucion - b)
    print(f"Norma infinito del residuo = {np.max(np.abs(A @ solucion - b)):.9e}")
    # Con este criterio: 28 iteraciones, x aproximadamente
    # (0.427740778, -0.276117084, 0.007326900).
