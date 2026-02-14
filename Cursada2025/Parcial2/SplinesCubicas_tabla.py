import numpy as np
from scipy.interpolate import CubicSpline

# --- Datos (los mismos de antes) ---
x_tabla = np.array([0.00, 0.20, 0.43, 0.55, 0.85, 0.90, 1.00])
def f(x):
    return np.exp(x) - np.sin(x)
y_tabla = f(x_tabla)
spline = CubicSpline(x_tabla, y_tabla, bc_type='natural')

# -------------------------------------------------------------
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# AQUÍ CAMBIAS EL ÍNDICE DEL INTERVALO QUE QUIERES VER
# 0 = [1.00, 1.20]
# 1 = [1.20, 1.45]
# 2 = [1.45, 1.70]
# 3 = [1.70, 2.00]

indice_intervalo = 5 

# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# -------------------------------------------------------------

# --- El código se adapta automáticamente ---

# Validar que el índice es correcto
if indice_intervalo < 0 or indice_intervalo >= len(x_tabla) - 1:
    print(f"Error: El índice {indice_intervalo} está fuera de rango. Debe ser entre 0 y {len(x_tabla) - 2}.")
else:
    # 1. Extraer los coeficientes para ese intervalo
    coeficientes = spline.c[:, indice_intervalo]
    
    a = coeficientes[0] # Cúbico
    b = coeficientes[1] # Cuadrático
    c = coeficientes[2] # Lineal
    d = coeficientes[3] # Constante
    
    # 2. Obtener el punto de inicio 'x_i' de ese intervalo
    x_i = x_tabla[indice_intervalo]
    x_f = x_tabla[indice_intervalo + 1]

    print(f"### Polinomio del Subintervalo {indice_intervalo+1}: [{x_i}, {x_f}] ###")
    print(f"El polinomio tiene la forma: S(x) = a(x - {x_i})^3 + b(x - {x_i})^2 + c(x - {x_i}) + d\n")
    print(f"Coeficientes:")
    print(f"  a = {a:.8f}")
    print(f"  b = {b:.8f}")
    print(f"  c = {c:.8f}")
    print(f"  d = {d:.8f}\n")

    print("--- Ecuación ---")
    print(f"S_{indice_intervalo+1}(x) = {a:.6f}(x - {x_i})^3 + {b:.6f}(x - {x_i})^2 + {c:.6f}(x - {x_i}) + {d:.6f}")
    print("----------------")