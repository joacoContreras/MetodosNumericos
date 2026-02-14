import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid, fixed_quad, quad

# --- Definición de la Función y Datos ---

def f(x):
    """Función original del problema."""
    return np.exp(x) * np.sin(x)

# Datos de la tabla original (Problema 1a)
x_tabla = np.array([0.00, 0.20, 0.43, 0.55, 0.85, 0.90, 1.00])
# Si quieres usar los valores numéricos de la tabla, descomenta la línea siguiente:
# y_tabla = np.array([-0.1483, -0.0400, 0.1818, 0.4493, 0.7001])
y_tabla = f(x_tabla)


print("### Problema 1: Spline Cúbico ###")

# (a) Crear el spline (natural por defecto)
spline = CubicSpline(x_tabla, y_tabla, bc_type='natural')

x_nuevos = np.linspace(0.0, 1.0, 5)
y_nuevos_exactos = f(x_nuevos)
y_nuevos_spline = spline(x_nuevos)
error_absoluto = np.abs(y_nuevos_exactos - y_nuevos_spline)

# Mostrar la tabla de resultados
print("Tabla de Puntos Equiespaciados (b):")
print("-----------------------------------------------------------------")
print("|  x  | Valor Exacto f(x) | Valor Spline S(x) | Error Absoluto |")
print("-----------------------------------------------------------------")
for i in range(len(x_nuevos)):
    print(f"| {x_nuevos[i]:.2f} | {y_nuevos_exactos[i]:17.6f} | {y_nuevos_spline[i]:17.6f} | {error_absoluto[i]:14.6e} |")
print("-----------------------------------------------------------------")

