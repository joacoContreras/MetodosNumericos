"""Runge-Kutta clásico de cuarto orden para y' = f(x, y).

Ejecutar: python final/runge_kutta.py
O indicar el paso directamente: python final/runge_kutta.py --h 0.1

Editar f, solucion_exacta y los datos iniciales para resolver otro ejercicio.
El error absoluto real requiere conocer la solución exacta del mismo problema
inicial. Si no se conoce, establecer SOLUCION_EXACTA = None.
Solo se utiliza la biblioteca estándar de Python.
"""

import argparse
import math


# ======================= DATOS DEL EJERCICIO =======================
# Ejemplo: y' = y - x**2 + 1, con y(0) = 0.5, hasta x = 2.
def f(x, y):
    """Lado derecho de la ecuación diferencial y' = f(x, y)."""
    return y - x**2 + 1


def solucion_exacta(x):
    """Solución analítica del ejemplo, que cumple y(0) = 0.5."""
    return (x + 1)**2 - 0.5 * math.exp(x)


X0 = 0.0
Y0 = 0.5
X_FINAL = 2.0
SOLUCION_EXACTA = solucion_exacta  # Usar None si no se conoce.
# ==================================================================


def runge_kutta_4(funcion, x0, y0, x_final, h, exacta=None):
    """Devuelve filas (x, y_RK4, y_exacta, error_absoluto), incluida la inicial.

    Integra hacia adelante con h > 0. Mantiene el paso solicitado salvo
    el último, que se acorta si es necesario para llegar a x_final.
    Sin solución exacta, las dos últimas columnas contienen None.
    """
    x0, y0, x_final, h = map(float, (x0, y0, x_final, h))
    if not all(math.isfinite(v) for v in (x0, y0, x_final, h)):
        raise ValueError('Los datos iniciales y h deben ser números finitos.')
    if h <= 0:
        raise ValueError('El paso h debe ser mayor que cero.')
    if x_final < x0:
        raise ValueError('X_FINAL debe ser mayor o igual que X0.')
    if not math.isfinite(x_final - x0):
        raise ValueError('La longitud del intervalo debe ser finita.')

    def crear_fila(x, y):
        y_exacta = float(exacta(x)) if exacta is not None else None
        error = abs(y_exacta - y) if y_exacta is not None else None
        if y_exacta is not None and not all(
            math.isfinite(v) for v in (y_exacta, error)
        ):
            raise ValueError(f'La solución exacta o el error no es finito en x={x}.')
        return x, y, y_exacta, error

    x, y = x0, y0
    tabla = [crear_fila(x, y)]
    n = 0

    while x < x_final:
        restante = x_final - x
        if restante <= h or math.isclose(restante, h, rel_tol=1e-12):
            x_siguiente = x_final
        else:
            # Construir la malla desde x0 evita acumular sumas de h.
            x_siguiente = min(x0 + (n + 1) * h, x_final)
        paso = x_siguiente - x
        if paso <= 0:
            raise ValueError('h es demasiado pequeño para avanzar con precisión float.')

        # Las k son pendientes; el factor paso se aplica en la actualización.
        k1 = funcion(x, y)
        k2 = funcion(x + paso / 2, y + paso * k1 / 2)
        k3 = funcion(x + paso / 2, y + paso * k2 / 2)
        k4 = funcion(x_siguiente, y + paso * k3)

        y = y + paso * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not math.isfinite(y):
            raise ValueError(f'RK4 produjo un valor no finito en x={x_siguiente}.')
        x = x_siguiente
        n += 1
        tabla.append(crear_fila(x, y))

    return tabla


def imprimir_tabla(tabla):
    """Muestra todos los puntos; el error se expresa en notación científica."""
    print(f"\n{'i':>5} | {'x':>14} | {'y (RK4)':>18} | {'y exacta':>18} | {'Error absoluto':>16}")
    print('-' * 83)
    for i, (x, y, y_exacta, error) in enumerate(tabla):
        texto_exacta = f'{y_exacta:.10f}' if y_exacta is not None else 'N/D'
        texto_error = f'{error:.8e}' if error is not None else 'N/D'
        print(f'{i:5d} | {x:14.8f} | {y:18.10f} | {texto_exacta:>18} | {texto_error:>16}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h', type=float, help='Tamaño de paso positivo.')
    args = parser.parse_args()
    try:
        h = args.h
        if h is None:
            h = float(input('Ingresá el paso h (por ejemplo, 0.1): ').replace(',', '.'))
        tabla = runge_kutta_4(f, X0, Y0, X_FINAL, h, SOLUCION_EXACTA)
    except (ValueError, ArithmeticError) as error:
        parser.exit(1, f'Error: {error}\n')

    print(f'\nRK4 clásico: y({X0}) = {Y0}; x final = {X_FINAL}; h = {h}')
    print('El último paso se acorta si h no divide el intervalo.')
    if SOLUCION_EXACTA is None:
        print('Sin solución exacta: no se puede calcular el error real (N/D).')
    else:
        print('Error absoluto = |y exacta(x) - y RK4(x)|.')
    imprimir_tabla(tabla)


if __name__ == '__main__':
    main()
