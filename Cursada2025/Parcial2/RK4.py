import numpy as np
import math

def f(x: float, y: float) -> float:
    """
    Define la Ecuación Diferencial Ordinaria dy/dx = f(x, y).
    Usamos el ejemplo de los apuntes.
    """
    return -2 * x * y
def metodo_rk4(f, x0: float, y0: float, x_final: float, n_pasos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve una EDO usando el Método de Runge-Kutta de 4to orden (Clásico).
    """
    if n_pasos <= 0:
        raise ValueError("n_pasos debe ser un entero positivo mayor que 0")
    h = (x_final - x0) / n_pasos
    x = np.linspace(x0, x_final, n_pasos + 1)
    y = np.empty(n_pasos + 1, dtype=float)
    y[0] = float(y0)
    
    for i in range(n_pasos):
        # Fórmulas de RK4 Clásico
        k1 = f(x[i], y[i])
        
        x_arg2 = x[i] + h / 2.0
        y_arg2 = y[i] + (h / 2.0) * k1
        k2 = f(x_arg2, y_arg2)
        
        y_arg3 = y[i] + (h / 2.0) * k2
        k3 = f(x_arg2, y_arg3)
        
        x_arg4 = x[i] + h
        y_arg4 = y[i] + h * k3
        k4 = f(x_arg4, y_arg4)
        
        y[i+1] = y[i] + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
    return x, y

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # Parámetros del problema (de los apuntes)
    x0 = 0.0
    y0 = 1.0
    x_final = 1.0
    n = 5  # 5 subintervalos -> h = 0.2

    # Solución exacta para comparar
    y_exacta_func = lambda x: np.exp(-x**2)
    
    # Resolver con RK4
    x_rk4, y_rk4 = metodo_rk4(f, x0, y0, x_final, n)
    
    # Imprimir la tabla de resultados
    print("--- Método de Runge-Kutta 4 (RK4) ---")
    print(f"Resolviendo dy/dx = -2xy, y(0)=1, h={ (x_final-x0)/n }")
    print("-" * 50)
    print(f"{'x':<5} | {'y (RK4)':<15} | {'y (Exacta)':<15}")
    print("-" * 50)
    
    for i in range(n + 1):
        x_val = x_rk4[i]
        y_ex = y_exacta_func(x_val)
        print(f"{x_val:<5.1f} | {y_rk4[i]:<15.8f} | {y_ex:<15.8f}")

    # Imprimir el resultado final (como en el apunte)
    print("-" * 50)
    print(f"y_rk4(1.0)    = {y_rk4[-1]:.8f}")
    print(f"y_exacta(1.0) = {y_exacta_func(1.0):.8f}")
    print(f"Error (RK4):  {abs(y_rk4[-1]-y_exacta_func(1.0)):.2E}")
    
    # A modo de comparación, veamos el error de Euler con h=0.2
    # y_euler(1.0) = 0.39938 (del apunte)
    y_euler_1_0 = 0.39938
    print(f"Error (Euler): {abs(y_euler_1_0 - y_exacta_func(1.0)):.2E}")
    print("\n¡Nota la enorme diferencia en la precisión!")