import numpy as np
import math
from typing import List, Callable

# --- Solvers Genéricos para Sistemas de EDOs ---

def resolver_sistema_euler(
    funciones: List[Callable], 
    x0: float, 
    y0_vec: List[float], 
    x_final: float, 
    n_pasos: int
) -> tuple[np.ndarray, List[np.ndarray]]:
    """
    Resuelve un sistema de EDOs de primer orden usando el método de Euler.
    """
    num_ecuaciones = len(funciones)
    if n_pasos <= 0:
        raise ValueError("n_pasos debe ser un entero positivo mayor que 0")
    h = (x_final - x0) / n_pasos

    x = np.linspace(x0, x_final, n_pasos + 1)
    # Crear una lista de arrays de numpy para los resultados (asegurando float)
    y_resultados = [np.zeros(n_pasos + 1, dtype=float) for _ in range(num_ecuaciones)]

    # Asignar valores iniciales
    for i in range(num_ecuaciones):
        y_resultados[i][0] = float(y0_vec[i])

    # Bucle del método de Euler
    for i in range(n_pasos):
        # Obtener el vector de y actual [y1, y2, ...]
        y_vals = [y_res[i] for y_res in y_resultados]
        
        # Calcular el siguiente y para cada ecuación
        for j in range(num_ecuaciones):
            # y_j[i+1] = y_j[i] + h * f_j(x[i], y1[i], y2[i], ...)
            y_resultados[j][i + 1] = y_resultados[j][i] + h * funciones[j](x[i], *y_vals)
            
    return x, y_resultados

def resolver_sistema_rk4(
    funciones: List[Callable], 
    x0: float, 
    y0_vec: List[float], 
    x_final: float, 
    n_pasos: int
) -> tuple[np.ndarray, List[np.ndarray]]:
    """
    Resuelve un sistema de EDOs de primer orden usando el método RK4.
    """
    num_ecuaciones = len(funciones)
    if n_pasos <= 0:
        raise ValueError("n_pasos debe ser un entero positivo mayor que 0")
    h = (x_final - x0) / n_pasos

    x = np.linspace(x0, x_final, n_pasos + 1)
    y_resultados = [np.zeros(n_pasos + 1, dtype=float) for _ in range(num_ecuaciones)]

    for i in range(num_ecuaciones):
        y_resultados[i][0] = float(y0_vec[i])

    # Arrays temporales para los 'k' de cada ecuación
    k1 = np.zeros(num_ecuaciones, dtype=float)
    k2 = np.zeros(num_ecuaciones, dtype=float)
    k3 = np.zeros(num_ecuaciones, dtype=float)
    k4 = np.zeros(num_ecuaciones, dtype=float)

    for i in range(n_pasos):
        # Obtener el vector de y actual [y1, y2, ...]
        y_vals = [y_res[i] for y_res in y_resultados]
        
        # 1. Calcular todos los k1
        for j in range(num_ecuaciones):
            k1[j] = funciones[j](x[i], *y_vals)
        
        # 2. Calcular todos los k2
        y_arg_k2 = [y_vals[j] + 0.5 * h * k1[j] for j in range(num_ecuaciones)]
        for j in range(num_ecuaciones):
            k2[j] = funciones[j](x[i] + 0.5 * h, *y_arg_k2)
            
        # 3. Calcular todos los k3
        y_arg_k3 = [y_vals[j] + 0.5 * h * k2[j] for j in range(num_ecuaciones)]
        for j in range(num_ecuaciones):
            k3[j] = funciones[j](x[i] + 0.5 * h, *y_arg_k3)
            
        # 4. Calcular todos los k4
        y_arg_k4 = [y_vals[j] + h * k3[j] for j in range(num_ecuaciones)]
        for j in range(num_ecuaciones):
            k4[j] = funciones[j](x[i] + h, *y_arg_k4)
            
        # 5. Combinar para el paso final de cada y
        for j in range(num_ecuaciones):
            y_resultados[j][i + 1] = y_resultados[j][i] + (h / 6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])

    return x, y_resultados

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---

if __name__ == "__main__":
    
    # 1. Definir el sistema de EDOs
    # Problema: y'' = 4cos(x) - y
    # Sistema:
    # y1 = y  -> dy1/dx = y2
    # y2 = y' -> dy2/dx = 4cos(x) - y1
    
    def f1(x, y1, y2):
        return y2
    
    def f2(x, y1, y2):
        return 4 * math.cos(x) - y1
        
    funciones_sistema = [f1, f2]

    # 2. Condiciones iniciales
    # y(0) = 0  -> y1(0) = 0
    # y'(0) = 0 -> y2(0) = 0
    cond_iniciales = [0.0, 0.0]
    
    # 3. Parámetros
    x_inicial = 0.0
    x_final = 2.0
    n_pasos = 20 # h = 0.1
    
    # Solución exacta para comparar: y(x) = 2x*sin(x)
    sol_exacta_y1 = lambda x: 2 * x * math.sin(x)
    
    # 4. Resolver y mostrar
    x_rk4, y_rk4 = resolver_sistema_rk4(funciones_sistema, x_inicial, cond_iniciales, x_final, n_pasos)
    
    print("--- Sistema de EDOs / EDO de Orden Superior con RK4 ---")
    print("-" * 65)
    print(f"{'x':<5} | {'y1 (Numérico)':<15} | {'y (Exacto)':<15} | {'Error':<15}")
    print("-" * 65)

    # Evita división por cero al decidir cada cuántos pasos imprimir
    print_step = max(1, n_pasos // 10)

    for i in range(n_pasos + 1):
        x_val = x_rk4[i]
        y_num = y_rk4[0][i]  # y_rk4[0] es el vector de y1
        y_exa = sol_exacta_y1(x_val)
        error = abs(y_num - y_exa)
        
        # Imprimir solo algunos pasos para claridad
        if i % print_step == 0 or i == n_pasos:
             print(f"{x_val:<5.2f} | {y_num:<15.8f} | {y_exa:<15.8f} | {error:<15.2E}")