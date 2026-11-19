"""Método del trapecio compuesto: con una tabla o con una función.

Ejecutar desde la raíz del proyecto: python final/trapecio.py

Uso general:
    integral = trapecio_tabla(x, y)       # y contiene los valores de f(x).
    integral = trapecio_funcion(f, a, b, n)  # n es la cantidad de subintervalos.

Para este ejercicio se integra Q(t)**2 entre 0 y 5 de ambas maneras.
Los datos y parámetros que se pueden cambiar están al final del archivo.
"""

import numpy as np


def trapecio_tabla(x, y):
    """Integra una tabla de f(x). Admite pasos distintos entre puntos."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("Ingresar dos listas de igual longitud, con al menos 2 puntos.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Los datos deben ser números finitos.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Los valores de x deben estar en orden creciente, sin repetir.")

    integral = 0.0
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        # Área de cada trapecio = base * promedio de las dos alturas.
        integral += h * (y[i] + y[i + 1]) / 2

    return integral


def trapecio_funcion(f, a, b, n):
    """Integra f entre a y b usando n subintervalos de igual ancho."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n debe ser un entero positivo.")
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        raise ValueError("Los extremos deben ser finitos y cumplir a < b.")

    # n subintervalos necesitan n + 1 puntos, incluyendo ambos extremos.
    x = np.linspace(a, b, n + 1)
    y = [f(xi) for xi in x]
    return trapecio_tabla(x, y)


if __name__ == "__main__":
    # DATOS DEL PROBLEMA. Para otro ejercicio cambiar esta sección.
    t = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    Q = np.array([0, 2.1, 3.8, 5.1, 5.9, 6.3])
    A = 6.5
    B = 0.6125454529809076  # Valor del inciso c, con Y = -B*X y ordenada cero.
    a = 0.0
    b = 5.0
    n = 10

    # 1) CON LA TABLA: las alturas son Q**2, porque eso pide la integral.
    # Se usan todos los puntos medidos: 6 puntos forman 5 subintervalos.
    energia_tabla = trapecio_tabla(t, Q**2)

    # 2) CON LA FUNCIÓN: definir el integrando, incluyendo el cuadrado.
    # Para otro problema reemplazar esta función por la que se desea integrar.
    def potencia(t):
        carga = A * (1 - np.exp(-B * t))
        return carga**2

    energia_modelo = trapecio_funcion(potencia, a, b, n)

    # SALIDA: tablas y resultados para la captura de la consola.
    print("MÉTODO DEL TRAPECIO COMPUESTO")
    print(f"Integral de Q(t)^2 entre {a:g} y {b:g}")
    print(f"Modelo: Q(t) = {A}*(1 - exp(-{B:.9f}*t))")

    print("\n1) USANDO LOS DATOS DE LA TABLA")
    print(f"Subintervalos: {len(t) - 1}")
    print("  t (s)       Q(t)       Q(t)^2")
    for ti, qi in zip(t, Q):
        print(f"{ti:7.2f}  {qi:10.6f}  {qi**2:10.6f}")
    print(f"E con la tabla = {energia_tabla:.6f}")

    print("\n2) USANDO LA FUNCIÓN Q(t)")
    print(f"A = {A}, B = {B:.9f}")
    print(f"Subintervalos: {n}, puntos: {n + 1}, h = {(b - a) / n:.6f}")
    print("  t (s)       Q(t)       Q(t)^2")
    for ti in np.linspace(a, b, n + 1):
        qi = A * (1 - np.exp(-B * ti))
        print(f"{ti:7.2f}  {qi:10.6f}  {potencia(ti):10.6f}")
    print(f"E con la función = {energia_modelo:.6f}")
