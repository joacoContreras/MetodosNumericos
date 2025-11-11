import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid, fixed_quad, quad

# --- Definición de la Función y Datos ---

def f(x):
    """Función original del problema."""
    return np.log(x**2 + 1) - np.sin(x)

# Datos de la tabla original (Problema 1a)
x_tabla = np.array([1.00, 1.20, 1.45, 1.70, 2.00])
# Si quieres usar los valores numéricos de la tabla, descomenta la línea siguiente:
# y_tabla = np.array([-0.1483, -0.0400, 0.1818, 0.4493, 0.7001])
# Actualmente usamos f(x_tabla) para mayor consistencia:
y_tabla = f(x_tabla)

# --- Apartados (a) y (b): Spline Cúbico y Nueva Tabla ---

print("### Problema 1: Spline Cúbico ###")

# (a) Crear el spline (natural por defecto)
spline = CubicSpline(x_tabla, y_tabla, bc_type='natural')

# (b) Crear nueva tabla con 5 puntos equiespaciados entre 1.0 y 2.0
x_nuevos = np.linspace(1.0, 2.0, 5)
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

# Graficar f(x) y el spline
x_plot = np.linspace(1.0, 2.0, 200)
y_plot_exacto = f(x_plot)
y_plot_spline = spline(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot_exacto, 'b-', label='f(x) (Exacta)', linewidth=2)
plt.plot(x_plot, y_plot_spline, 'r--', label='Spline Cubico S(x)', linewidth=2)
plt.plot(x_tabla, y_tabla, 'ko', label='Puntos de datos originales')
plt.plot(x_nuevos, y_nuevos_spline, 'gx', label='Puntos equiespaciados (Spline)', markersize=8)
plt.title('Función vs. Interpolación con Spline Cubico')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# --- Apartado (c): Derivación Numérica ---

print("\n### Problema 1: Derivación Numérica ###")

# Espaciado
h = x_nuevos[1] - x_nuevos[0]

# Derivada del spline en los puntos nuevos (más precisa que np.gradient sobre valores)
derivadas_spline = spline(x_nuevos, 1)

# Derivada exacta de f
def f_prima(x):
    return (2*x) / (x**2 + 1) - np.cos(x)

derivadas_exactas = f_prima(x_nuevos)

# Mostrar la tabla de derivadas
print("\nTabla de Derivadas (c):")
print("---------------------------------------------------------------")
print("|  x  | Derivada Exacta f'(x) | Derivada Numérica (Spline) |")
print("---------------------------------------------------------------")
for i in range(len(x_nuevos)):
    print(f"| {x_nuevos[i]:.2f} | {derivadas_exactas[i]:21.6f} | {derivadas_spline[i]:26.6f} |")
print("---------------------------------------------------------------")

# --- Apartados (d), (e), (f): Integración Numérica ---

print("\n### Problema 1: Integración Numérica ###")

# Definimos la nueva función a integrar
def g(x):
    return f(x) / x

# (d) Método del Trapecio usando valores de la spline para f(x)
g_valores_spline = spline(x_nuevos) / x_nuevos
integral_trapecio = trapezoid(g_valores_spline, x_nuevos)
print(f"(d) Integral por Regla del Trapecio (con datos spline): {integral_trapecio:.8f}")

# (d) "Valor Exacto" con quad
integral_exacta, error_estimado_quad = quad(g, 1.0, 2.0)
print(f"(d) Integral 'Exacta' (usando scipy.integrate.quad): {integral_exacta:.8f}")

# (d) Error absoluto
error_abs_trapecio = np.abs(integral_exacta - integral_trapecio)
print(f"(d) Error Absoluto (Trapecio vs Quad): {error_abs_trapecio:.4e}")

# --- (e) y (f) Cuadratura de Gauss ---
integral_gauss_2, _ = fixed_quad(g, 1.0, 2.0, n=2)
integral_gauss_6, _ = fixed_quad(g, 1.0, 2.0, n=6)

# Evitar división por cero al calcular errores porcentuales
if integral_exacta != 0:
    error_perc_gauss_2 = np.abs((integral_exacta - integral_gauss_2) / integral_exacta) * 100
    error_perc_gauss_6 = np.abs((integral_exacta - integral_gauss_6) / integral_exacta) * 100
else:
    error_perc_gauss_2 = np.nan
    error_perc_gauss_6 = np.nan

print(f"\n(e) Integral con Gauss (n=2): {integral_gauss_2:.8f} (Error porcentual: {error_perc_gauss_2:.6f}%)")
print(f"(f) Integral con Gauss (n=6): {integral_gauss_6:.8f} (Error porcentual: {error_perc_gauss_6:.6f}%)")