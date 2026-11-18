"""Ajuste de una recta por mínimos cuadrados con ecuaciones normales.

Ejecutar: python final/minimos_cuadrados.py

Uso general:
    a0, a1 = minimos_cuadrados(x, y)
    La recta ajustada es y = a0 + a1*x.

Para ajustar y = a1*x, usar por_el_origen=True.
Este archivo y regresion_lineal.py minimizan la misma suma de cuadrados.
Aquí se arma y resuelve el sistema de ecuaciones del método explícitamente.
"""

import numpy as np


def minimos_cuadrados(x, y, por_el_origen=False):
    """Devuelve a0 y a1 minimizando sum((y - a0 - a1*x)**2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Ingresar dos listas de igual longitud, con al menos 2 datos.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Los datos deben ser números finitos.")
    if np.all(x == x[0]):
        raise ValueError("Ingresar al menos dos valores de x distintos.")

    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x**2)
    sxy = np.sum(x * y)

    if por_el_origen:
        # Con a0 = 0 queda una sola ecuación: sum(x**2)*a1 = sum(x*y).
        matriz = np.array([[sxx]])
        termino = np.array([sxy])
        a1 = np.linalg.solve(matriz, termino)[0]
        a0 = 0.0
    else:
        # Al derivar la suma de cuadrados respecto de a0 y a1:
        # n*a0      + sum(x)*a1    = sum(y)
        # sum(x)*a0 + sum(x**2)*a1 = sum(x*y)
        matriz = np.array([[n, sx], [sx, sxx]])
        termino = np.array([sy, sxy])
        a0, a1 = np.linalg.solve(matriz, termino)

    return a0, a1


if __name__ == "__main__":
    # DATOS DEL INCISO C. Para otro problema cambiar datos y transformación.
    t = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    Q = np.array([0, 2.1, 3.8, 5.1, 5.9, 6.3])
    A = 6.5

    # Linealización: X = t, Y = ln(1 - Q/A), entonces Y = -B*X.
    # Se requiere 1 - Q/A > 0 para calcular el logaritmo natural.
    X = t
    Y = np.log(1 - Q / A)

    # Para una recta general, pasar x, y sin transformar y usar False.
    a0, a1 = minimos_cuadrados(X, Y, por_el_origen=True)
    B = -a1

    # Correlación de Pearson entre los datos transformados.
    r = np.corrcoef(X, Y)[0, 1]
    suma_cuadrados = np.sum((Y - (a0 + a1 * X))**2)

    print(f"sum(X**2) = {np.sum(X**2):.6f}")
    print(f"sum(X*Y)  = {np.sum(X * Y):.6f}")
    print(f"\nRecta: Y = {a0:.6f} + ({a1:.6f})*X")
    print(f"B = {B:.6f} s^(-1)")
    print(f"r de Pearson entre X e Y = {r:.6f}")
    print(f"Suma de cuadrados de residuos en Y = {suma_cuadrados:.6f}")
    print(f"Modelo: Q(t) = {A}*(1 - exp(-{B:.6f}*t))")
    # Resultado: B = 0.612545 s^(-1), r = -0.982377.
    # No confundir r**2 con el R² centrado del ajuste forzado al origen.
