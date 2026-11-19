"""Problema 3, leyendo la ecuación como (dy/dt)/y - (4*y - t**2) = 0.

Uso:
    python final/problema3_euler_rk4.py
    python final/problema3_euler_rk4.py --detalles

Al despejar: y' = 4*y**2 - t**2*y, y(0) = 1.
Con u = 1/y se obtiene u' - t**2*u = -4, u(0) = 1, y por tanto:

    y(t) = exp(-t**3/3) / (1 - 4*integral_0^t exp(-s**3/3) ds).

El denominador se anula antes de t=0.3. El PVI no tiene solución hasta
t=1, 2 o 3. Se usa Simpson compuesto y bisección para ubicar ese polo;
Euler y RK4 se aplican a la EDO original solamente antes del polo.
La localización del polo es auxiliar: ambos métodos mantienen h=0.1.
"""

import argparse
import math

from Euler_Generico import euler, imprimir_detalles
from RK4_Generico import rk4, imprimir_detalles_k


H = 0.1


def funcion(t, y):
    return 4 * y**2 - t**2 * y


def integral_factor(t):
    """Aproxima la integral de exp(-s**3/3) entre 0 y t con Simpson."""
    n = 1000  # Simpson compuesto requiere una cantidad par de subintervalos.
    paso = t / n
    suma = 1.0 + math.exp(-t**3 / 3)
    for j in range(1, n):
        s = j * paso
        suma += (4 if j % 2 else 2) * math.exp(-s**3 / 3)
    return paso * suma / 3


def denominador(t):
    return 1 - 4 * integral_factor(t)


def tiempo_singularidad():
    """Busca el único cero positivo del denominador dentro de (0.25, 0.3)."""
    a, b = 0.25, 0.3
    while b - a > 1e-12:
        medio = (a + b) / 2
        if denominador(medio) > 0:
            a = medio
        else:
            b = medio
    return (a + b) / 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detalles', action='store_true',
                        help='Mostrar cada paso de Euler y las pendientes RK4.')
    args = parser.parse_args()

    t_polo = tiempo_singularidad()
    print("EDO: y' = 4*y**2 - t**2*y; y(0) = 1; h = 0.1")
    print(f'Singularidad positiva: t* ≈ {t_polo:.10f}')
    print('La solución del PVI solo existe para 0 <= t < t*.')

    # Último nodo de la malla h=0.1 que pertenece al intervalo de existencia.
    ultimo_nodo = math.floor(t_polo / H)
    t_final = ultimo_nodo * H
    t_e, y_e, detalles_e = euler(
        funcion, 0.0, 1.0, t_final, h=H, retornar_detalles=True
    )
    _, y_r, detalles_r = rk4(
        funcion, 0.0, 1.0, t_final, h=H, retornar_detalles=True
    )

    print('\nNodos anteriores a la singularidad:')
    print(f'{"t":>5} | {"Euler":>14} | {"RK4":>14} | {"e_rel (con signo)":>18}')
    for t, ye, yr in zip(t_e, y_e, y_r):
        error = (ye - yr) / yr  # Fórmula del enunciado, sin valor absoluto.
        print(f'{t:5.1f} | {ye:14.10f} | {yr:14.10f} | {error:18.10f}')

    print('\nResultados solicitados por el enunciado:')
    for t in (1, 2, 3):
        print(f't={t}: y(t) no existe para este PVI; error relativo no definido.')
    print('Los valores iterados después del polo no aproximan esta solución.')

    if args.detalles:
        print('\nDETALLES DE EULER')
        imprimir_detalles(detalles_e)
        print('\nDETALLES DE RK4 (k1, ..., k4 son pendientes)')
        imprimir_detalles_k(detalles_r)


if __name__ == '__main__':
    main()
