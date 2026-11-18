import numpy as np
import math
import matplotlib.pyplot as plt

def f(x: float, y: float) -> float:
    """
    Define la Ecuación Diferencial Ordinaria dy/dx = f(x, y).
    Ejemplo de los apuntes: dy/dx = -2xy
    """
    return -2.0 * x * y

def metodo_rk4(
    f, 
    x0: float, 
    y0: float, 
    x_final: float, 
    paso_o_n = None, 
    *, 
    h: float = None, 
    n_pasos: int = None, 
    retornar_detalles: bool = False
) -> tuple:
    """
    Resuelve una EDO dy/dx = f(x, y) con condición inicial y(x0) = y0
    usando el Método de Runge-Kutta de 4to orden (Clásico).

    Args:
        f: Función f(x, y) de la EDO dy/dx = f(x, y).
        x0: Valor inicial de x (condición inicial).
        y0: Valor inicial de y, tal que y(x0) = y0.
        x_final: PUNTO DE CÁLCULO (valor final de x donde se desea evaluar la solución).
        paso_o_n: Puede ser el paso h (float, ej: 0.2) o la cantidad de pasos (int, ej: 5).
        h: Tamaño de paso (keyword opcional).
        n_pasos: Cantidad de pasos (keyword opcional).
        retornar_detalles: Si es True, retorna (x, y, detalles) con k1, k2, k3, k4 de cada paso.

    Returns:
        (x, y) o (x, y, detalles)
    """
    # 1. Determinar h y n_pasos según los argumentos provistos
    if paso_o_n is not None:
        if isinstance(paso_o_n, int) and n_pasos is None and h is None:
            n_pasos = paso_o_n
        elif isinstance(paso_o_n, float) and h is None and n_pasos is None:
            h = paso_o_n
        elif n_pasos is None and h is None:
            if paso_o_n == int(paso_o_n) and paso_o_n >= 1:
                n_pasos = int(paso_o_n)
            else:
                h = float(paso_o_n)

    # Si se especificó h pero no n_pasos
    if h is not None and n_pasos is None:
        if h <= 0:
            raise ValueError("El tamaño de paso h debe ser mayor que 0.")
        n_pasos = int(round(abs(x_final - x0) / h))
        if n_pasos == 0:
            n_pasos = 1
    # Si se especificó n_pasos pero no h
    elif n_pasos is not None and h is None:
        if n_pasos <= 0:
            raise ValueError("n_pasos debe ser un entero positivo mayor que 0.")
        n_pasos = int(n_pasos)
    elif h is None and n_pasos is None:
        n_pasos = 10  # Valor por defecto

    # 2. Paso exacto y partición de x
    h_efectivo = (x_final - x0) / n_pasos
    x = np.linspace(x0, x_final, n_pasos + 1)
    y = np.empty(n_pasos + 1, dtype=float)
    y[0] = float(y0)

    detalles = []

    # 3. Bucle de RK4 clásico
    for i in range(n_pasos):
        xi = x[i]
        yi = y[i]

        # Pendientes intermedias
        k1 = f(xi, yi)
        k2 = f(xi + 0.5 * h_efectivo, yi + 0.5 * h_efectivo * k1)
        k3 = f(xi + 0.5 * h_efectivo, yi + 0.5 * h_efectivo * k2)
        k4 = f(xi + h_efectivo, yi + h_efectivo * k3)

        # Promedio ponderado de pendientes (orden 4)
        pendiente_promedio = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        y[i + 1] = yi + h_efectivo * pendiente_promedio

        detalles.append({
            'paso': i,
            'xi': xi,
            'yi': yi,
            'k1': k1,
            'k2': k2,
            'k3': k3,
            'k4': k4,
            'x_next': x[i + 1],
            'y_next': y[i + 1]
        })

    if retornar_detalles:
        return x, y, detalles
    return x, y


# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # -------------------------------------------------------------
    # PARÁMETROS DEL PROBLEMA
    # -------------------------------------------------------------
    x0 = 0.0          # Punto inicial (condición inicial)
    y0 = 1.0          # Valor inicial: y(x0) = y0
    x_final = 1.0     # PUNTO DE CÁLCULO: valor de x donde queremos hallar y(x)
    h = 0.2           # Tamaño de paso (o alternativamente n_pasos = 5)

    # Solución exacta para comparar (si se conoce)
    # Para dy/dx = -2xy con y(0) = 1, la solución analítica es y(x) = exp(-x^2)
    y_exacta_func = lambda x: np.exp(-x**2)

    # -------------------------------------------------------------
    # RESOLVER CON RK4
    # -------------------------------------------------------------
    x_rk4, y_rk4, detalles = metodo_rk4(
        f, x0, y0, x_final, h=h, retornar_detalles=True
    )
    n_total = len(x_rk4) - 1
    h_real = (x_final - x0) / n_total

    # -------------------------------------------------------------
    # PRESENTAR RESULTADOS
    # -------------------------------------------------------------
    print("--- Método de Runge-Kutta de 4to Orden (RK4 Clásico) ---")
    print(f"EDO: dy/dx = -2xy, con condición inicial y({x0}) = {y0}")
    print(f"Punto de cálculo: x_final = {x_final}")
    print(f"Paso: h = {h_real:.4f} | Cantidad de pasos: n = {n_total}")
    print("=" * 72)
    print(f"{'Paso':<5} | {'x':<7} | {'y (RK4)':<15} | {'y (Exacta)':<15} | {'Error Absoluto':<15}")
    print("-" * 72)

    for i in range(len(x_rk4)):
        x_val = x_rk4[i]
        y_calc = y_rk4[i]
        if y_exacta_func is not None:
            y_ex = y_exacta_func(x_val)
            err = abs(y_calc - y_ex)
            print(f"{i:<5} | {x_val:<7.4f} | {y_calc:<15.8f} | {y_ex:<15.8f} | {err:<15.2E}")
        else:
            print(f"{i:<5} | {x_val:<7.4f} | {y_calc:<15.8f} | {'N/A':<15} | {'N/A':<15}")
    print("=" * 72)

    # Tabla detallada de pendientes intermedias (útil para exámenes)
    print("\n--- Pendientes intermedias (k1, k2, k3, k4) por paso ---")
    print("-" * 75)
    print(f"{'Paso':<5} | {'k1':<12} | {'k2':<12} | {'k3':<12} | {'k4':<12} | {'y_{i+1}':<14}")
    print("-" * 75)
    for d in detalles:
        print(f"{d['paso']:<5} | {d['k1']:<12.6f} | {d['k2']:<12.6f} | {d['k3']:<12.6f} | {d['k4']:<12.6f} | {d['y_next']:<14.8f}")
    print("-" * 75)

    # Resultado final en el punto de cálculo
    print(f"\n--- Resultado Final en el Punto de Cálculo (x = {x_final}) ---")
    print(f"y_rk4({x_final})    = {y_rk4[-1]:.8f}")
    if y_exacta_func is not None:
        y_ex_final = y_exacta_func(x_final)
        error_final = abs(y_rk4[-1] - y_ex_final)
        print(f"y_exacta({x_final}) = {y_ex_final:.8f}")
        print(f"Error (RK4):        {error_final:.2E}")

    # Comparativa referencial con Euler
    y_euler_apunte = 0.39938
    if y_exacta_func is not None:
        print(f"\nComparación con Euler (h={h_real}): Error Euler = {abs(y_euler_apunte - y_exacta_func(x_final)):.2E}")
        print(f"-> RK4 es aproximadamente {abs(y_euler_apunte - y_exacta_func(x_final)) / error_final:.0f} veces más preciso.")

    # -------------------------------------------------------------
    # GRÁFICA COMPARATIVA
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    if y_exacta_func is not None:
        x_denso = np.linspace(x0, x_final, 200)
        plt.plot(x_denso, y_exacta_func(x_denso), 'b-', label='Solución Exacta $y = e^{-x^2}$', linewidth=2)
    plt.plot(x_rk4, y_rk4, 'ro--', label=f'RK4 (h={h_real:.2f})', markersize=6)
    plt.axvline(x_final, color='gray', linestyle=':', label=f'Punto de cálculo $x={x_final}$')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Método de Runge-Kutta 4 (RK4 Clásico): $dy/dx = -2xy$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
