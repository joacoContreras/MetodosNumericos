"""Eliminación de Gauss con pivoteo parcial y sustitución hacia atrás.

Ejecutar: python final/gauss.py
Uso general: solucion = gauss(A, b, mostrar=False)

Para otro sistema, cambiar A y b al final del archivo.
Cada fila de A contiene los coeficientes de una ecuación, en el mismo orden
de incógnitas. b contiene los términos independientes.
Admite cualquier cantidad de incógnitas si A es cuadrada y no singular.
"""

import numpy as np


def gauss(A, b, mostrar=True):
    """Resuelve A*x = b sin modificar los datos originales."""
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)

    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("A debe ser una matriz cuadrada no vacía.")
    n = len(A)
    if b.shape != (n,):
        raise ValueError("b debe ser una lista con un término por ecuación.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("Los datos deben ser números finitos.")

    # Umbral según la escala de A para detectar pivotes numéricamente nulos.
    umbral = np.finfo(float).eps * n * np.max(np.abs(A))

    if mostrar:
        print("Matriz aumentada inicial [A | b]:")
        print(np.column_stack((A, b)))

    # 1) ELIMINACIÓN: convertir A en una matriz triangular superior.
    for k in range(n):
        # Pivoteo parcial: buscar el mayor valor absoluto de la columna,
        # entre la fila actual y las que están debajo.
        pivote = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[pivote, k]) <= umbral:
            raise ValueError("Sistema singular o numéricamente singular: no se puede continuar.")

        if pivote != k:
            A[[k, pivote]] = A[[pivote, k]]
            b[[k, pivote]] = b[[pivote, k]]
            if mostrar:
                print(f"\nIntercambio F{k + 1} <-> F{pivote + 1}")
                print(np.column_stack((A, b)))

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]
            A[i, k] = 0.0
            if mostrar:
                print(f"\nF{i + 1} <- F{i + 1} - ({factor:.6f})*F{k + 1}")
                print(np.column_stack((A, b)))

    # 2) SUSTITUCIÓN HACIA ATRÁS: empezar por la última ecuación.
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = sum(A[i, j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - suma) / A[i, i]
        if mostrar:
            print(f"x{i + 1} = {x[i]:.9f}")

    return x


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    # PROBLEMA 3. Orden de incógnitas: x, y, z.
    #  3x +  y -  z =  1
    #  2x - 4y + 5z =  2
    #  -x + 2y - 3z = -1
    A = np.array([[3, 1, -1],
                  [2, -4, 5],
                  [-1, 2, -3]], dtype=float)
    b = np.array([1, 2, -1], dtype=float)

    print("MÉTODO DE ELIMINACIÓN DE GAUSS")
    solucion = gauss(A, b)

    print("\nVector solución (x1, x2, ..., xn):")
    print(solucion)
    print("Residuo A*x - b:")
    print(A @ solucion - b)
    print(f"Norma infinito del residuo = {np.max(np.abs(A @ solucion - b)):.3e}")
    # Para este sistema: (x, y, z) = (3/7, -2/7, 0).
