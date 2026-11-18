"""Regresión lineal mediante las fórmulas de mínimos cuadrados.

Ejecutar: python final/regresion_lineal.py

Uso general:
    a0, a1, r = regresion_lineal(x, y)
    La recta ajustada es y = a0 + a1*x.

Si el modelo exige pasar por (0, 0), usar por_el_origen=True.
La regresión lineal también usa mínimos cuadrados: no son métodos opuestos.
En minimos_cuadrados.py se resuelve el mismo ajuste con ecuaciones normales.
"""

import numpy as np


def regresion_lineal(x, y, por_el_origen=False):
    """Devuelve ordenada a0, pendiente a1 y correlación de Pearson r."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Ingresar dos listas de igual longitud, con al menos 2 datos.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Los datos deben ser números finitos.")
    if np.all(x == x[0]):
        raise ValueError("Se necesitan valores de x distintos para calcular r.")

    n = len(x)
    sx = np.sum(x)
    sy = np.sum(y)
    sxx = np.sum(x**2)
    sxy = np.sum(x * y)

    if por_el_origen:
        # Minimizar S = sum((y - a1*x)**2), con a0 fijo en cero.
        a0 = 0.0
        a1 = sxy / sxx
    else:
        # Minimizar S = sum((y - a0 - a1*x)**2).
        a1 = (n * sxy - sx * sy) / (n * sxx - sx**2)
        a0 = (sy - a1 * sx) / n

    # Pearson mide la relación lineal entre los datos x e y.
    dx = x - np.mean(x)
    dy = y - np.mean(y)
    denominador = np.sqrt(np.sum(dx**2) * np.sum(dy**2))
    r = np.sum(dx * dy) / denominador if denominador > 0 else np.nan
    # Si y es constante, r no está definido (nan).
    return a0, a1, r


if __name__ == "__main__":
    # DATOS DEL INCISO C: cambiar estas listas y A para otro capacitor.
    t = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    Q = np.array([0, 2.1, 3.8, 5.1, 5.9, 6.3])
    A = 6.5

    # Q(t) = A*(1 - exp(-B*t)) se transforma en Y = -B*X.
    # np.log es el logaritmo natural. Se requiere 1 - Q/A > 0.
    X = t
    Y = np.log(1 - Q / A)

    # Para este modelo a0 = 0, porque A está fijado por el enunciado.
    # Para una recta general y = a0 + a1*x, usar False y los datos originales.
    a0, a1, r = regresion_lineal(X, Y, por_el_origen=True)
    B = -a1

    print("Datos transformados:")
    print("     X           Y")
    for xi, yi in zip(X, Y):
        print(f"{xi:6.1f}  {yi:10.6f}")

    print(f"\nRecta: Y = {a0:.6f} + ({a1:.6f})*X")
    print(f"B = {B:.6f} s^(-1)")
    print(f"r de Pearson entre X e Y = {r:.6f}")
    print(f"Modelo: Q(t) = {A}*(1 - exp(-{B:.6f}*t))")
    # Resultado: B = 0.612545 s^(-1), r = -0.982377.
    # Con por_el_origen=False: a0 = 0.272505 y B = 0.686865.
    # Esa segunda opción permite una ordenada distinta de cero.
