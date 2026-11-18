"""Genera una tabla con splines naturales e integra con trapecios y Simpson.

Ejecución: python final/splines_tabla_equiespaciada.py
Requiere NumPy. Se supone S''(x_inicial) = S''(x_final) = 0.
"""

import numpy as np


def calcular_splines_naturales(fuente_datos):
    """Devuelve los nodos y los coeficientes de cada polinomio cúbico.

    En [x[i], x[i+1]] se usa:
        S_i(x) = a[i] + b[i]*t + c[i]*t**2 + d[i]*t**3,
        con t = x - x[i].
    """
    datos = np.asarray(fuente_datos, dtype=float)
    if datos.ndim != 2 or datos.shape[1] != 2 or len(datos) < 2:
        raise ValueError("Se necesitan al menos dos pares [x, f(x)].")
    if not np.all(np.isfinite(datos)):
        raise ValueError("Los datos deben ser números finitos.")

    datos = datos[np.argsort(datos[:, 0])]
    x, y = datos[:, 0], datos[:, 1]
    pasos = np.diff(x)  # Separaciones originales; no tienen que ser iguales.
    if np.any(pasos <= 0):
        raise ValueError("Los valores de x deben ser distintos.")

    # Las incógnitas c_i son la mitad de las segundas derivadas en los nodos.
    cantidad = len(x)
    matriz = np.zeros((cantidad, cantidad))
    termino = np.zeros(cantidad)
    matriz[0, 0] = matriz[-1, -1] = 1.0  # c_0 = c_n = 0.

    for i in range(1, cantidad - 1):
        matriz[i, i - 1] = pasos[i - 1]
        matriz[i, i] = 2 * (pasos[i - 1] + pasos[i])
        matriz[i, i + 1] = pasos[i]
        termino[i] = 3 * (
            (y[i + 1] - y[i]) / pasos[i]
            - (y[i] - y[i - 1]) / pasos[i - 1]
        )

    c_nodos = np.linalg.solve(matriz, termino)
    a = y[:-1].copy()
    b = np.diff(y) / pasos - pasos * (2 * c_nodos[:-1] + c_nodos[1:]) / 3
    c = c_nodos[:-1]
    d = np.diff(c_nodos) / (3 * pasos)
    return x, a, b, c, d


def generar_tabla(fuente_datos, h=0.1, inicio=None, fin=None):
    """Interpola en [inicio, fin], incluyendo ambos extremos.

    Si no se indican límites, usa el primer y el último nodo originales.
    Para incluir ambos extremos sin alterar h, la longitud del intervalo
    debe ser un múltiplo entero de h. Devuelve filas [x, S(x)].
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("El paso h debe ser positivo y finito.")

    x, a, b, c, d = calcular_splines_naturales(fuente_datos)
    inicio = x[0] if inicio is None else float(inicio)
    fin = x[-1] if fin is None else float(fin)
    if not np.isfinite(inicio) or not np.isfinite(fin) or inicio >= fin:
        raise ValueError("Los límites deben ser finitos y cumplir inicio < fin.")
    if inicio < x[0] or fin > x[-1]:
        raise ValueError(
            f"El intervalo debe estar dentro de [{x[0]}, {x[-1]}]."
        )

    cantidad_pasos = (fin - inicio) / h
    n = round(cantidad_pasos)
    if n < 1 or not np.isclose(cantidad_pasos, n, rtol=0, atol=1e-10):
        raise ValueError("La longitud del intervalo debe ser múltiplo de h.")

    # Evita acumular errores mediante sumas sucesivas de h.
    x_tabla = inicio + h * np.arange(n + 1)
    x_tabla[-1] = fin

    # Cada punto se evalúa con el polinomio de su intervalo original.
    indices = np.searchsorted(x, x_tabla, side="right") - 1
    indices = np.clip(indices, 0, len(a) - 1)
    t = x_tabla - x[indices]
    y_tabla = a[indices] + t * (b[indices] + t * (c[indices] + t * d[indices]))
    return np.column_stack((x_tabla, y_tabla))


def integrar_trapecios(fuente_datos, a, b, h=0.1):
    """Aproxima la integral del spline con la regla del trapecio compuesto.

    Requiere a < b dentro del dominio de los datos y n = (b-a)/h entero
    y positivo. Admite tanto cantidades pares como impares de subintervalos.
    """
    tabla = generar_tabla(fuente_datos, h=h, inicio=a, fin=b)
    valores = tabla[:, 1]
    # Integral ≈ h/2 * [y_0 + 2*(y_1 + ... + y_(n-1)) + y_n].
    return float(h / 2 * (valores[0] + 2 * np.sum(valores[1:-1]) + valores[-1]))


def integrar_simpson(fuente_datos, a, b, h=0.1):
    """Aproxima la integral del spline en [a, b] con Simpson 1/3 compuesto.

    Requiere a < b dentro del dominio de los datos y n = (b-a)/h entero,
    positivo y par. Los valores de la integración se obtienen del spline,
    aunque a y b no coincidan con nodos de la tabla original.
    """
    tabla = generar_tabla(fuente_datos, h=h, inicio=a, fin=b)
    n = len(tabla) - 1
    if n % 2 != 0:
        raise ValueError(
            f"Simpson 1/3 requiere una cantidad par de subintervalos; n = {n}. "
            "Modificá h o los límites de integración."
        )

    valores = tabla[:, 1]
    # Integral ≈ h/3 * [y_0 + y_n + 4*(y_1+y_3+...) + 2*(y_2+y_4+...)].
    suma_impares = np.sum(valores[1:-1:2])
    suma_pares = np.sum(valores[2:-1:2])
    return float(
        h / 3 * (valores[0] + valores[-1] + 4 * suma_impares + 2 * suma_pares)
    )


if __name__ == "__main__":
    fuente_datos = [
        [1.0, 5.381],
        [1.3, 8.672],
        [1.4, 9.592],
        [1.45, 9.988],
        [1.6, 10.853],
        [1.72, 11.146],
        [1.8, 11.136],
        [1.93, 10.780],
        [2.0, 10.435],
    ]
    h = 0.1
    # Límites de integración: modificá a y b según la consigna.
    # Deben estar dentro de [1.0, 2.0] y (b-a)/h debe ser entero.
    # Simpson requiere además que esa cantidad de subintervalos sea par.
    a = 1.0
    b = 2.0
    tabla = generar_tabla(fuente_datos, h)

    print("Tabla obtenida con splines cúbicas naturales")
    print(f"Paso h = {h}")
    print(f"{'x':>8} | {'S(x)':>12}")
    print("-" * 24)
    for xi, yi in tabla:
        print(f"{xi:8.2f} | {yi:12.6f}")

    print("\nIntegración numérica de S(x)")
    print(f"Intervalo: [{a}, {b}], h = {h}")
    metodos = [
        ("Trapecio compuesto", integrar_trapecios),
        ("Simpson 1/3 compuesto", integrar_simpson),
    ]
    resultados = {}
    for nombre, metodo in metodos:
        try:
            integral = metodo(fuente_datos, a, b, h)
        except ValueError as error:
            print(f"{nombre}: no se pudo calcular. {error}")
        else:
            resultados[nombre] = integral
            print(f"{nombre}: {integral:.8f}")

    if len(resultados) == 2:
        diferencia = abs(
            resultados["Simpson 1/3 compuesto"] - resultados["Trapecio compuesto"]
        )
        # Esta comparación no representa el error respecto de la integral real.
        print(f"Diferencia absoluta entre los métodos: {diferencia:.8f}")
