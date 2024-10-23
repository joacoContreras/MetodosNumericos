# Integración: Regla Simpson 1/3
import math
import numpy as np

def funcion_a_integrar(x):
    """Función a integrar."""
    return math.pow(math.e,math.sqrt(1+x)) * math.log(1+2*math.pow(x,2))

def regla_simpson_13(func, a, b, tramos):
    """Calcula la integral usando la regla de Simpson 1/3."""
    if tramos % 2 != 0:
        raise ValueError("El número de tramos debe ser par para la regla de Simpson 1/3.")

    h = (b - a) / tramos
    xi = np.linspace(a, b, tramos + 1)
    delta_areas = (h / 3) * np.sum([func(xi[i]) + 4 * func(xi[i + 1]) + func(xi[i + 2]) for i in range(0, tramos, 2)])
    return delta_areas

# INGRESO
a = 0
b = 1
tramos = 6

# PROCEDIMIENTO
area = regla_simpson_13(funcion_a_integrar, a, b, tramos)

# SALIDA
print(f'Número de tramos: {tramos}')
print(f'Integral con regla de Simpson 1/3: {area}')