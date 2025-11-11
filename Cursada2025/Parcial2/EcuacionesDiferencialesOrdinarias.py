import numpy as np
import math

def f(x: float, y: float) -> float:
    """
    Define la Ecuación Diferencial Ordinaria dy/dx = f(x, y).
    Usamos el ejemplo de los apuntes[cite: 1129].
    """
    return -2 * x * y

def metodo_euler(f, x0: float, y0: float, x_final: float, n_pasos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve una EDO usando el Método de Euler.
    """
    h = (x_final - x0) / n_pasos
    x = np.linspace(x0, x_final, n_pasos + 1)
    y = np.zeros(n_pasos + 1)
    y[0] = y0
    
    for i in range(n_pasos):
        # Fórmula: y_i+1 = y_i + h * f(x_i, y_i)
        y[i+1] = y[i] + h * f(x[i], y[i])
        
    return x, y

def metodo_heun(f, x0: float, y0: float, x_final: float, n_pasos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve una EDO usando el Método de Heun (Predictor-Corrector).
    """
    h = (x_final - x0) / n_pasos
    x = np.linspace(x0, x_final, n_pasos + 1)
    y = np.zeros(n_pasos + 1)
    y[0] = y0
    
    for i in range(n_pasos):
        # 1. Predictor (Euler) 
        k1 = f(x[i], y[i])
        y_predictor = y[i] + h * k1
        
        # 2. Corrector (Promedio) 
        k2 = f(x[i+1], y_predictor)
        y[i+1] = y[i] + (h / 2.0) * (k1 + k2)
        
    return x, y

def metodo_punto_medio(f, x0: float, y0: float, x_final: float, n_pasos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve una EDO usando el Método del Punto Medio.
    """
    h = (x_final - x0) / n_pasos
    x = np.linspace(x0, x_final, n_pasos + 1)
    y = np.zeros(n_pasos + 1)
    y[0] = y0
    
    for i in range(n_pasos):
        # 1. Estimar el punto medio (x_m, y_m) [cite: 1187-1188]
        x_medio = x[i] + h / 2.0
        y_medio = y[i] + (h / 2.0) * f(x[i], y[i])
        
        # 2. Paso final usando la pendiente media 
        pendiente_media = f(x_medio, y_medio)
        y[i+1] = y[i] + h * pendiente_media
        
    return x, y

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # Parámetros del problema (de los apuntes [cite: 1128-1129])
    x0 = 0.0
    y0 = 1.0
    x_final = 1.0
    n = 5  # 5 subintervalos -> h = 0.2

    # Solución exacta para comparar [cite: 1114]
    y_exacta_func = lambda x: math.exp(-x**2)
    
    # Resolver con los tres métodos
    x_euler, y_euler = metodo_euler(f, x0, y0, x_final, n)
    x_heun, y_heun = metodo_heun(f, x0, y0, x_final, n)
    x_pm, y_pm = metodo_punto_medio(f, x0, y0, x_final, n)
    
    # Imprimir la tabla de resultados
    print("--- Comparativa de Métodos EDO (Euler, Heun, Punto Medio) ---")
    print(f"Resolviendo dy/dx = -2xy, y(0)=1, h={ (x_final-x0)/n }")
    print("-" * 75)
    print(f"{'x':<5} | {'y (Euler)':<12} | {'y (Heun)':<12} | {'y (P. Medio)':<12} | {'y (Exacta)':<12}")
    print("-" * 75)
    
    for i in range(n + 1):
        x_val = x_euler[i]
        y_ex = y_exacta_func(x_val)
        print(f"{x_val:<5.1f} | {y_euler[i]:<12.6f} | {y_heun[i]:<12.6f} | {y_pm[i]:<12.6f} | {y_ex:<12.6f}")

    # Imprimir el resultado final (como en el apunte [cite: 1142, 1143])
    print("-" * 75)
    print(f"y_euler(1.0) = {y_euler[-1]:.6f} (Error: {abs(y_euler[-1]-y_exacta_func(1.0)):.6f})")
    print(f"y_heun(1.0) = {y_heun[-1]:.6f} (Error: {abs(y_heun[-1]-y_exacta_func(1.0)):.6f})")
    print(f"y_pm(1.0)   = {y_pm[-1]:.6f} (Error: {abs(y_pm[-1]-y_exacta_func(1.0)):.6f})")
    print(f"y_exacta(1.0) = {y_exacta_func(1.0):.6f}")