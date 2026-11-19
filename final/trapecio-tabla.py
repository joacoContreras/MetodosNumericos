"""Problema 2: integral de x*f'(x) mediante el trapecio compuesto.

Ejecutar desde la raíz del proyecto:
    python final/trapecio-tabla.py

Las derivadas se aproximan con tres puntos (segundo orden), incluso
cuando los pasos son desiguales. No se agrega el punto x=1.75 que no
figura en el enunciado. Solo requiere la biblioteca estándar de Python.
"""

import math


# DATOS DEL EJERCICIO: para otra tabla modificar estos pares [x, f(x)].
FUENTE_DATOS = [
    [0.00, 1.000],
    [0.25, 1.384],
    [0.50, 1.849],
    [0.75, 2.417],
    [1.00, 3.118],
    [1.25, 3.990],
    [1.50, 5.082],
    [2.00, 8.189],
]


def validar_tabla(tabla):
    """Valida al menos tres pares finitos, con x estrictamente creciente."""
    datos = []
    for fila in tabla:
        if len(fila) != 2:
            raise ValueError('Cada fila debe contener exactamente [x, f(x)].')
        x, y = map(float, fila)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Los datos deben ser números finitos.')
        datos.append((x, y))
    if len(datos) < 3:
        raise ValueError('Se necesitan al menos tres puntos para derivar con segundo orden.')
    if any(datos[i + 1][0] <= datos[i][0] for i in range(len(datos) - 1)):
        raise ValueError('Los valores de x deben estar en orden creciente, sin repetir.')
    return datos


def derivadas_segundo_orden(tabla):
    """Devuelve f'(x_i) y los coeficientes de la fórmula usada en cada nodo.

    Se deriva el interpolante cuadrático local de tres puntos:
        f'(x_i) ≈ suma_j w_j*f(x_j)
        w_j = (2*x_i - x_k - x_l) / ((x_j-x_k)*(x_j-x_l))
    donde j, k, l son los tres índices distintos del tramo local.

    En los extremos se usan los primeros/últimos tres puntos; en el
    interior, el anterior, el actual y el siguiente. Para paso uniforme
    se obtienen las fórmulas progresiva, centrada y regresiva habituales.
    """
    datos = validar_tabla(tabla)
    derivadas = []
    formulas = []
    for i, (xi, _) in enumerate(datos):
        inicio = min(max(i - 1, 0), len(datos) - 3)
        indices = range(inicio, inicio + 3)
        terminos = []
        for j in indices:
            k, l = [indice for indice in indices if indice != j]
            xj, xk, xl = datos[j][0], datos[k][0], datos[l][0]
            peso = (2 * xi - xk - xl) / ((xj - xk) * (xj - xl))
            terminos.append((j, peso))
        derivadas.append(math.fsum(peso * datos[j][1] for j, peso in terminos))
        formulas.append(terminos)
    return derivadas, formulas


def trapecio_tabla(x, y):
    """Integra valores tabulados y_i con el ancho real de cada intervalo."""
    x, y = list(map(float, x)), list(map(float, y))
    if len(x) != len(y) or len(x) < 2:
        raise ValueError('Ingresar listas de igual longitud, con al menos dos puntos.')
    if not all(math.isfinite(valor) for valor in x + y):
        raise ValueError('Los datos deben ser números finitos.')
    if any(x[i + 1] <= x[i] for i in range(len(x) - 1)):
        raise ValueError('Los valores de x deben estar en orden creciente, sin repetir.')
    return math.fsum(
        (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
        for i in range(len(x) - 1)
    )


def main():
    datos = validar_tabla(FUENTE_DATOS)
    derivadas, formulas = derivadas_segundo_orden(datos)
    x = [xi for xi, _ in datos]
    integrando = [xi * derivada for xi, derivada in zip(x, derivadas)]
    integral = trapecio_tabla(x, integrando)

    print('PROBLEMA 2: TRAPECIO COMPUESTO CON DATOS TABULADOS')
    print(f'I = integral de x*f\'(x) entre {x[0]:g} y {x[-1]:g}')
    print('\nDERIVADAS CON ESQUEMAS DE SEGUNDO ORDEN')
    print('Se deriva un interpolante cuadrático local de tres puntos.')
    print('En cada fórmula, f[j] es el valor de la fila j (índices desde 0).')
    print('Los coeficientes mostrados están redondeados; el cálculo no se redondea.')
    print('{:>2} {:>6} {:>9} {:>12} {:>12}  Fórmula para f\'(x)'.format(
        'i', 'x', 'f(x)', "f'(x)", "x*f'(x)",
    ))
    for i, ((xi, yi), derivada, gi, terminos) in enumerate(
        zip(datos, derivadas, integrando, formulas)
    ):
        formula = ' '.join(f'{peso:+.8g}*f[{j}]' for j, peso in terminos if peso != 0)
        print(f'{i:2d} {xi:6.2f} {yi:9.3f} {derivada:12.6f} {gi:12.6f}  {formula}')

    print('\nTRAPECIO COMPUESTO: g(x) = x*f\'(x)')
    print('I ≈ suma_i (x[i+1]-x[i]) * (g[i]+g[i+1]) / 2')
    print(f"{'Intervalo':>15} {'Paso':>8} {'Área del trapecio':>20}")
    for i in range(len(x) - 1):
        paso = x[i + 1] - x[i]
        area = paso * (integrando[i] + integrando[i + 1]) / 2
        intervalo = f'[{x[i]:.2f}, {x[i + 1]:.2f}]'
        print(f'{intervalo:>15} {paso:8.2f} {area:20.8f}')
    print(f'\nI ≈ {integral:.8f}')


if __name__ == '__main__':
    main()
