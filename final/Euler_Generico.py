"""Euler explícito para ecuaciones escalares y sistemas de primer orden.

Uso:
    python final/Euler_Generico.py
    python final/Euler_Generico.py --h 0.1
    python final/Euler_Generico.py --h 0.1 --detalles
    python final/Euler_Generico.py --h 0.1 --no-grafica

Entrada: editar la sección DATOS DEL PROBLEMA y elegir h al ejecutar.
  * funcion(x, Y) devuelve las derivadas del estado Y.
  * Para una EDO escalar, Y0 y la derivada son números.
  * Para un sistema, Y0 y las derivadas son vectores en el mismo orden.
  * Una EDO de orden superior se convierte primero en un sistema de primer orden.
  * SOLUCION_EXACTA debe devolver la misma forma que Y0; usar None si se desconoce.

Fórmula: Y_(n+1) = Y_n + h_n * funcion(x_n, Y_n).
Todas las componentes se actualizan con los valores del mismo estado anterior.
Euler explícito es de primer orden: para soluciones suaves y pasos adecuados,
el error global es O(h). La solución analítica sirve para medir el error,
no interviene en las iteraciones numéricas.

La tabla incluye todos los nodos, valores calculados, valores exactos y errores
absolutos por componente. La gráfica compara las soluciones en los mismos ejes:
línea continua para la exacta; círculos y línea discontinua para Euler.
Requiere NumPy y, para mostrar la gráfica, Matplotlib.
"""

import argparse
import math

import numpy as np


# ======================== DATOS DEL PROBLEMA ========================
# y'' + 2*y' + 5*y = 0; y(0) = 0; y'(0) = 2; 0 <= x <= 1.
# Definimos v = y', de modo que Y = [y, v].
# Euler: y_nuevo = y + h*v; v_nuevo = v + h*(-2*v - 5*y).
# Ambos cálculos utilizan los valores anteriores de y y v.
def funcion(x, Y):
    """Devuelve [y', v'] en el mismo orden que las condiciones iniciales."""
    y, v = Y
    return np.array([v, -2 * v - 5 * y])


def solucion_exacta(x):
    """Devuelve [y(x), v(x)]; v es la derivada de y."""
    y = np.exp(-x) * np.sin(2 * x)
    v = np.exp(-x) * (2 * np.cos(2 * x) - np.sin(2 * x))
    return np.array([y, v])


X0 = 0.0
Y0 = [0.0, 2.0]             # [y(X0), v(X0)]. Para una EDO escalar: un número.
X_FINAL = 1.0
H = 0.1                     # Paso sugerido al pedirlo por teclado.
NOMBRES_VARIABLES = ['y', 'v']  # Un nombre por componente de Y0; o None.
SOLUCION_EXACTA = solucion_exacta  # None si no se conoce.
# ===================================================================


def _estado(valor, forma, contexto):
    """Comprueba dimensiones y valores sin permitir broadcasting accidental."""
    arreglo = np.asarray(valor, dtype=float)
    if arreglo.shape != forma:
        raise ValueError(
            f'{contexto}: se esperaba forma {forma} y se recibió {arreglo.shape}.'
        )
    if not np.all(np.isfinite(arreglo)):
        raise ValueError(f'{contexto}: se obtuvo NaN o infinito.')
    return float(arreglo) if forma == () else arreglo


def _malla(x0, x_final, h, n_pasos):
    """Respeta h y acorta solo el último paso; permite integrar hacia atrás."""
    if h is not None and n_pasos is not None:
        raise ValueError('Indicar solo h o n_pasos, no ambos.')
    if h is not None:
        h = float(h)
        if not math.isfinite(h) or h <= 0:
            raise ValueError('h debe ser un número finito mayor que cero.')
    else:
        if n_pasos is None:
            n_pasos = 10
        if (isinstance(n_pasos, (bool, np.bool_))
                or not isinstance(n_pasos, (int, np.integer)) or n_pasos <= 0):
            raise ValueError('n_pasos debe ser un entero positivo.')

    if x0 == x_final:
        return np.array([x0])
    distancia = abs(x_final - x0)
    if not math.isfinite(distancia):
        raise ValueError('La longitud del intervalo debe ser finita.')
    direccion = 1 if x_final > x0 else -1
    if h is not None and h < distancia and x0 + direccion * h == x0:
        raise ValueError('El paso no permite avanzar con precisión float.')
    if h is None:
        x = np.linspace(x0, x_final, n_pasos + 1)
    else:
        cociente = distancia / h
        if not math.isfinite(cociente):
            raise ValueError('h es demasiado pequeño para este intervalo.')
        cercano = round(cociente)
        # Evita un paso adicional causado solo por redondeo, por ejemplo 3.000...4.
        if math.isclose(cociente, cercano, rel_tol=0, abs_tol=1e-12):
            n = max(1, cercano)
        else:
            n = max(1, math.ceil(cociente))
        x = x0 + direccion * h * np.arange(n + 1, dtype=float)
        x[-1] = x_final

    if not np.all(np.isfinite(x)) or np.any(direccion * np.diff(x) <= 0):
        raise ValueError('El paso no permite avanzar con precisión float.')
    return x


def euler(f, x0, y0, x_final, h=None, *, n_pasos=None,
          retornar_detalles=False):
    """Integra Y' = f(x, Y) con Euler explícito.

    Devuelve (x, y), o (x, y, detalles) si retornar_detalles=True.
    Para una EDO escalar y tiene forma (n+1,); para m variables, (n+1, m).
    Elegir h > 0 o un n_pasos entero positivo. Si no se indica ninguno,
    se utilizan diez pasos. La dirección se deduce de x0 y x_final.
    Con h se mantiene el paso solicitado, acortando solo el último si hace falta.
    """
    x0, x_final = float(x0), float(x_final)
    if not math.isfinite(x0) or not math.isfinite(x_final):
        raise ValueError('Los extremos deben ser números finitos.')
    inicial = np.asarray(y0, dtype=float)
    if inicial.ndim > 1 or inicial.size == 0:
        raise ValueError('y0 debe ser un número o un vector no vacío.')
    forma = inicial.shape
    _estado(inicial, forma, 'Condición inicial')
    x = _malla(x0, x_final, h, n_pasos)
    y = np.empty((len(x),) + forma, dtype=float)
    y[0] = inicial
    detalles = []

    for i in range(len(x) - 1):
        xi, yi = x[i], y[i]
        paso = x[i + 1] - xi
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            pendiente = _estado(f(xi, yi), forma, f'Derivada en x={xi:.12g}')
            # Se calculan TODAS las componentes a partir del estado anterior.
            siguiente = yi + paso * pendiente
        y[i + 1] = _estado(siguiente, forma, f'Solución en x={x[i + 1]:.12g}')
        if retornar_detalles:
            detalles.append({
                'paso': i, 'h': paso, 'xi': xi, 'yi': yi.copy(),
                'pendiente': np.copy(pendiente),
                'x_next': x[i + 1], 'y_next': y[i + 1].copy(),
            })
    return (x, y, detalles) if retornar_detalles else (x, y)


metodo_euler = euler


def _datos_resultados(x, y, sol_exacta, nombres_variables):
    """Normaliza resultados y evalúa la referencia por punto y componente."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or x.size == 0 or y.ndim not in (1, 2)
            or len(x) != len(y) or y.size == 0):
        raise ValueError('x e y deben contener la misma cantidad de puntos.')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Los resultados deben ser finitos.')
    forma = y.shape[1:]
    valores = y.reshape(len(x), -1)
    cantidad = valores.shape[1]
    if nombres_variables is None:
        nombres = ['y'] if cantidad == 1 else [f'Y{j + 1}' for j in range(cantidad)]
    else:
        nombres = list(nombres_variables)
        if len(nombres) != cantidad:
            raise ValueError('Se necesita un nombre por variable.')
    exactos = None
    if sol_exacta is not None:
        exactos = np.array([
            _estado(sol_exacta(xi), forma, f'Solución exacta en x={xi:.12g}')
            for xi in x
        ]).reshape(valores.shape)
    return x, valores, exactos, nombres


def imprimir_tabla(x, y, sol_exacta=None, nombres_variables=None):
    """Muestra todos los nodos y el error absoluto real de cada componente."""
    x, valores, exactos, nombres = _datos_resultados(
        x, y, sol_exacta, nombres_variables
    )
    titulos = ['Paso', 'x']
    for nombre in nombres:
        titulos.append(f'{nombre} Euler')
        if exactos is not None:
            titulos.extend([f'{nombre} exacta', f'Error {nombre}'])
    anchos = [max(16, len(titulo)) for titulo in titulos]
    anchos[0] = max(5, len(str(len(x) - 1)))
    print('\nTABLA DE RESULTADOS — EULER EXPLÍCITO')
    if exactos is None:
        print('Sin solución exacta: no se puede calcular el error real.')
    else:
        print('Error absoluto por componente = |valor exacto - valor Euler|.')
    print(' | '.join(t.rjust(a) for t, a in zip(titulos, anchos)))
    print('-' * (sum(anchos) + 3 * (len(anchos) - 1)))
    for i, xi in enumerate(x):
        fila = [str(i), f'{xi:.10g}']
        for j in range(len(nombres)):
            fila.append(f'{valores[i, j]:.10f}')
            if exactos is not None:
                fila.extend([
                    f'{exactos[i, j]:.10f}',
                    f'{abs(exactos[i, j] - valores[i, j]):.8e}',
                ])
        print(' | '.join(t.rjust(a) for t, a in zip(fila, anchos)))


def imprimir_detalles(detalles):
    """Muestra el estado anterior, la pendiente y el resultado de cada paso."""
    def texto(valor):
        return np.array2string(np.asarray(valor), precision=10, separator=', ')

    for detalle in detalles:
        print(f"\nPaso {detalle['paso'] + 1}: "
              f"x={detalle['xi']:.10g} -> {detalle['x_next']:.10g}; "
              f"h={detalle['h']:.10g}")
        print(f"  Y anterior = {texto(detalle['yi'])}")
        print(f"  Pendiente F(x, Y) = {texto(detalle['pendiente'])}")
        print(f"  Y siguiente = {texto(detalle['y_next'])}")


def graficar_solucion(x, y, sol_exacta=None, titulo='Solución con Euler',
                      nombres_variables=None, x_label='x', y_label='Y(x)'):
    """Compara cada variable con su solución exacta, cuando está disponible."""
    import matplotlib.pyplot as plt

    x, valores, _, nombres = _datos_resultados(x, y, sol_exacta, nombres_variables)
    plt.figure(figsize=(9, 5))
    densos = np.linspace(x[0], x[-1], 300)
    if sol_exacta is not None:
        forma = () if np.asarray(y).ndim == 1 else (valores.shape[1],)
        referencias = np.array([
            _estado(sol_exacta(xi), forma, 'Solución exacta para la gráfica')
            for xi in densos
        ]).reshape(len(densos), -1)
    for j, nombre in enumerate(nombres):
        linea, = plt.plot(x, valores[:, j], 'o--', label=f'{nombre} Euler',
                          markerfacecolor='none', markersize=6, zorder=3)
        if sol_exacta is not None:
            plt.plot(densos, referencias[:, j], color=linea.get_color(),
                     label=f'{nombre} exacta')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(titulo)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h', type=float, help='Magnitud positiva del paso.')
    parser.add_argument('--detalles', action='store_true', help='Mostrar pendiente y actualización por paso.')
    parser.add_argument('--grafica', action=argparse.BooleanOptionalAction, default=True,
                        help='Mostrar la comparación gráfica (activada por defecto).')
    args = parser.parse_args()
    try:
        h = args.h
        if h is None:
            entrada = input(f'Ingresá h [Enter para {H}]: ').strip()
            h = float(entrada.replace(',', '.')) if entrada else H
        resultado = euler(funcion, X0, Y0, X_FINAL, h=h,
                        retornar_detalles=args.detalles)
        x, y = resultado[:2]
        print(f'\nEuler desde x={X0} hasta x={X_FINAL}; h solicitado={h}.')
        print(f'Condición inicial: Y({X0}) = {Y0}.')
        print('Se mantiene h; solo se acorta el último paso si hace falta.')
        imprimir_tabla(x, y, SOLUCION_EXACTA, NOMBRES_VARIABLES)
        if args.detalles:
            imprimir_detalles(resultado[2])
        if args.grafica:
            graficar_solucion(x, y, SOLUCION_EXACTA,
                              nombres_variables=NOMBRES_VARIABLES)
    except (ValueError, ArithmeticError, ImportError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
