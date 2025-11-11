import numpy as np
import matplotlib.pyplot as plt

# --- Definiciones Base (EDO y Solución Exacta) ---

def f_edo(x, y):
    """Define la EDO: dy/dx = f(x, y)"""
    # Manejar el caso y=0 para evitar división por cero, aunque
    # con y(0)=1 no debería ocurrir en este intervalo.
    if y == 0:
        return np.inf
    return (x * np.exp(x)) / y

def sol_exacta(x):
    """Solución analítica exacta (inciso b)."""
    return np.sqrt(2 * (x - 1) * np.exp(x) + 3)

# --- (a) Implementación del Método Propuesto ---

def solve_metodo_propuesto(f, x0, y0, h, x_final):
    """Resuelve la EDO con el método de 2 pasos del examen."""
    n_steps = int((x_final - x0) / h)
    x = np.linspace(x0, x_final, n_steps + 1)
    y = np.zeros(n_steps + 1)
    
    # Condición inicial (i=0)
    y[0] = y0
    
    # (i=1) ARRANQUE con Euler
    y[1] = y[0] + h * f(x[0], y[0])
    
    # (i=2 en adelante) Bucle principal de 2 pasos
    for i in range(1, n_steps):
        f_i = f(x[i], y[i])
        f_i_menos_1 = f(x[i-1], y[i-1])
        
        y[i+1] = y[i] + h * (2 * f_i - f_i_menos_1)
        
    return x, y

# --- (d) Implementación del Método RK4 ---

def solve_rk4(f, x0, y0, h, x_final):
    """Resuelve una EDO usando el método estándar RK4."""
    n_steps = int((x_final - x0) / h)
    x = np.linspace(x0, x_final, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0

    for i in range(n_steps):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * k1)
        k3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * k2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x, y

# --- (c), (d), (e) Ejecución y Resultados ---

print("### Problema 2: Solución de EDOs ###")

# Parámetros del problema
x0 = 0.0
y0 = 1.0
x_final = 1.0
h = 0.1 # Paso pedido en el inciso (c)

# 1. Resolver con Método Propuesto (c)
x_prop, y_prop = solve_metodo_propuesto(f_edo, x0, y0, h, x_final)

# 2. Resolver con RK4 (d)
x_rk4, y_rk4 = solve_rk4(f_edo, x0, y0, h, x_final)

# 3. Calcular Solución Exacta (b)
# Usamos x_prop como referencia de puntos, es igual a x_rk4
x_exact = x_prop
y_exact = sol_exacta(x_exact)

# (c) y (d) Tabla de resultados y errores
print(f"\nResultados para h = {h}:")
print("------------------------------------------------------------------------------------------------")
print("  x  | Sol. Exacta | Sol. Propuesta (c) | Error Abs. (c) |    Sol. RK4 (d)    | Error Abs. (d) |")
print("------------------------------------------------------------------------------------------------")

# Índices para x=0.5 y x=1.0
idx_05 = int(0.5 / h) # 5
idx_10 = int(1.0 / h) # 10

# Fila para x = 0.5
val_ex_05 = y_exact[idx_05]
val_pr_05 = y_prop[idx_05]
val_rk_05 = y_rk4[idx_05]
err_pr_05 = np.abs(val_ex_05 - val_pr_05)
err_rk_05 = np.abs(val_ex_05 - val_rk_05)
print(f" {x_exact[idx_05]:.1f} |  {val_ex_05:.8f} |     {val_pr_05:.8f} |    {err_pr_05:10.4e} | {val_rk_05:.8f} |   {err_rk_05:10.4e} |")

# Fila para x = 1.0
val_ex_10 = y_exact[idx_10]
val_pr_10 = y_prop[idx_10]
val_rk_10 = y_rk4[idx_10]
err_pr_10 = np.abs(val_ex_10 - val_pr_10)
err_rk_10 = np.abs(val_ex_10 - val_rk_10)
print(f" {x_exact[idx_10]:.1f} |  {val_ex_10:.8f} |     {val_pr_10:.8f} |    {err_pr_10:10.4e} | {val_rk_10:.8f} |   {err_rk_10:10.4e} |")
print("------------------------------------------------------------------------------------------------")

# (e) Graficar las tres soluciones
plt.figure(figsize=(12, 7))
plt.plot(x_exact, y_exact, 'b-', label='Solución Exacta (b)', linewidth=3)
plt.plot(x_prop, y_prop, 'r--o', label='Método Propuesto (a, c)', markersize=5)
plt.plot(x_rk4, y_rk4, 'g:x', label='Método RK4 (d)', markersize=5)
plt.title('Comparación de Métodos de Solución de EDOs', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('y(x)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()