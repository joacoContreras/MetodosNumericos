"""
================================================================================
MÉTODO DE RUNGE-KUTTA DE 4TO ORDEN (RK4 CLÁSICO) - SOLVER GENÉRICO
================================================================================
Este archivo contiene una implementación genérica y flexible del método RK4
capaz de resolver:
  1. Cualquier Ecuación Diferencial Ordinaria (EDO) de 1er orden: dy/dx = f(x, y)
  2. Cualquier Sistema de EDOs de 1er orden: dY/dx = F(x, Y)
  3. EDOs de orden superior (2do, 3er orden, etc.) convirtiéndolas en sistemas.

CONCEPTOS CLAVE:
  - x0: Punto inicial (donde se conoce la condición inicial).
  - y0: Condición inicial y(x0) = y0 (puede ser un número o una lista para sistemas).
  - x_final: PUNTO DE CÁLCULO (valor de x donde se desea evaluar/conocer la solución).
  - h: Tamaño de paso entre cada iteración.
  - n_pasos: Número de subintervalos en los que se divide [x0, x_final].
    (Se puede especificar 'h' o 'n_pasos', el código calcula el otro automáticamente).
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, List, Tuple, Dict, Any, Optional


def rk4(
    f: Callable,
    x0: float,
    y0: Union[float, List[float], np.ndarray],
    x_final: float,
    paso_o_n: Optional[Union[float, int]] = None,
    *,
    h: Optional[float] = None,
    n_pasos: Optional[int] = None,
    retornar_detalles: bool = False
) -> Tuple:
    """
    Resuelve numéricamente dy/dx = f(x, y) desde x0 hasta x_final usando RK4 clásico.

    Args:
        f: Función f(x, y) para escalares, o F(x, Y) para sistemas de EDOs.
        x0: Valor inicial de la variable independiente x.
        y0: Condición inicial (número float para EDO simple, o lista/array para sistemas).
        x_final: PUNTO DE CÁLCULO (valor objetivo de x).
        paso_o_n: Puede pasarse el paso 'h' (float) o el número de pasos 'n' (int) posicionalmente.
        h: Tamaño de paso deseado (keyword opcional).
        n_pasos: Número entero de pasos deseado (keyword opcional).
        retornar_detalles: Si es True, retorna (x, y, detalles_k) con los coeficientes k1..k4.

    Returns:
        (x, y) si retornar_detalles es False.
        (x, y, detalles_k) si retornar_detalles es True.
    """
    # --------------------------------------------------------------------------
    # 1. Determinación y validación del paso h y del número de pasos n_pasos
    # --------------------------------------------------------------------------
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

    if h is not None and n_pasos is None:
        if h <= 0:
            raise ValueError("El paso 'h' debe ser estrictamente positivo.")
        n_pasos = int(round(abs(x_final - x0) / h))
        if n_pasos == 0:
            n_pasos = 1
    elif n_pasos is not None and h is None:
        if n_pasos <= 0:
            raise ValueError("El número de pasos 'n_pasos' debe ser un entero positivo.")
        n_pasos = int(n_pasos)
    elif h is None and n_pasos is None:
        n_pasos = 10  # Valor por defecto si no se especifica ninguno

    # Paso efectivo exacto
    h_efectivo = (x_final - x0) / n_pasos
    x = np.linspace(x0, x_final, n_pasos + 1)

    # --------------------------------------------------------------------------
    # 2. Detección de EDO escalar vs. Sistema vectorial
    # --------------------------------------------------------------------------
    y0_arr = np.asarray(y0, dtype=float)
    es_escalar = (y0_arr.ndim == 0)

    if es_escalar:
        y = np.empty(n_pasos + 1, dtype=float)
        y[0] = float(y0)
    else:
        dimension = len(y0_arr)
        y = np.empty((n_pasos + 1, dimension), dtype=float)
        y[0] = y0_arr

    detalles_k = []

    # --------------------------------------------------------------------------
    # 3. Bucle de integración RK4 de 4to orden
    # --------------------------------------------------------------------------
    for i in range(n_pasos):
        xi = x[i]
        yi = y[i]

        if es_escalar:
            # Caso escalar: operaciones numéricas simples
            k1 = float(f(xi, yi))
            k2 = float(f(xi + 0.5 * h_efectivo, yi + 0.5 * h_efectivo * k1))
            k3 = float(f(xi + 0.5 * h_efectivo, yi + 0.5 * h_efectivo * k2))
            k4 = float(f(xi + h_efectivo, yi + h_efectivo * k3))

            pendiente_promedio = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            y[i + 1] = yi + h_efectivo * pendiente_promedio
        else:
            # Caso vectorial: compatible con numpy arrays
            k1 = np.asarray(f(xi, yi), dtype=float)
            k2 = np.asarray(f(xi + 0.5 * h_efectivo, yi + 0.5 * h_efectivo * k1), dtype=float)
            k3 = np.asarray(f(xi + 0.5 * h_efectivo, yi + 0.5 * h_efectivo * k2), dtype=float)
            k4 = np.asarray(f(xi + h_efectivo, yi + h_efectivo * k3), dtype=float)

            pendiente_promedio = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            y[i + 1] = yi + h_efectivo * pendiente_promedio

        if retornar_detalles:
            detalles_k.append({
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
        return x, y, detalles_k
    return x, y


# Alias para máxima compatibilidad
metodo_rk4 = rk4


# ==============================================================================
# FUNCIONES AUXILIARES PARA MOSTRAR Y GRAFICAR RESULTADOS
# ==============================================================================

def imprimir_tabla(
    x: np.ndarray,
    y: np.ndarray,
    sol_exacta: Optional[Callable] = None,
    nombres_variables: Optional[List[str]] = None
) -> None:
    """
    Imprime en consola una tabla profesional con los resultados paso a paso.
    """
    n = len(x)
    es_escalar = (y.ndim == 1)

    print("\n" + "=" * 75)
    print("TABLA DE RESULTADOS - MÉTODO RUNGE-KUTTA 4 (RK4)")
    print("=" * 75)

    if es_escalar:
        if sol_exacta is not None:
            print(f"{'Paso':<5} | {'x':<10} | {'y (RK4)':<16} | {'y (Exacta)':<16} | {'Error Absoluto':<15}")
            print("-" * 75)
            for i in range(n):
                y_ex = sol_exacta(x[i])
                err = abs(y[i] - y_ex)
                print(f"{i:<5} | {x[i]:<10.4f} | {y[i]:<16.8f} | {y_ex:<16.8f} | {err:<15.2E}")
        else:
            print(f"{'Paso':<5} | {'x':<10} | {'y (RK4)':<16}")
            print("-" * 38)
            for i in range(n):
                print(f"{i:<5} | {x[i]:<10.4f} | {y[i]:<16.8f}")
    else:
        dim = y.shape[1]
        if nombres_variables is None or len(nombres_variables) != dim:
            nombres_variables = [f"y{j+1}" for j in range(dim)]
        
        encabezado_vars = " | ".join([f"{var:<16}" for var in nombres_variables])
        ancho_total = 20 + 19 * dim
        print(f"{'Paso':<5} | {'x':<10} | {encabezado_vars}")
        print("-" * ancho_total)
        for i in range(n):
            fila_vars = " | ".join([f"{y[i, j]:<16.8f}" for j in range(dim)])
            print(f"{i:<5} | {x[i]:<10.4f} | {fila_vars}")

    print("=" * 75)


def imprimir_detalles_k(detalles_k: List[Dict[str, Any]]) -> None:
    """
    Imprime los valores intermedios de k1, k2, k3, k4 para cada paso (ideal para parciales).
    """
    if not detalles_k:
        return

    primero = detalles_k[0]['k1']
    es_escalar = np.isscalar(primero) or np.ndim(primero) == 0

    print("\n" + "-" * 75)
    print("PENDIENTES INTERMEDIAS RK4 (k1, k2, k3, k4) EN CADA PASO")
    print("-" * 75)

    if es_escalar:
        print(f"{'Paso':<5} | {'k1':<12} | {'k2':<12} | {'k3':<12} | {'k4':<12} | {'y_siguiente':<14}")
        print("-" * 75)
        for d in detalles_k:
            print(
                f"{d['paso']:<5} | "
                f"{d['k1']:<12.6f} | "
                f"{d['k2']:<12.6f} | "
                f"{d['k3']:<12.6f} | "
                f"{d['k4']:<12.6f} | "
                f"{d['y_next']:<14.8f}"
            )
    else:
        for d in detalles_k:
            print(f"Paso {d['paso']} (x = {d['xi']:.4f} -> {d['x_next']:.4f}):")
            print(f"   k1 = {np.array2string(d['k1'], precision=6, separator=', ')}")
            print(f"   k2 = {np.array2string(d['k2'], precision=6, separator=', ')}")
            print(f"   k3 = {np.array2string(d['k3'], precision=6, separator=', ')}")
            print(f"   k4 = {np.array2string(d['k4'], precision=6, separator=', ')}")
            print(f"   y_{{i+1}} = {np.array2string(d['y_next'], precision=8, separator=', ')}\n")

    print("-" * 75)


def graficar_solucion(
    x: np.ndarray,
    y: np.ndarray,
    sol_exacta: Optional[Callable] = None,
    titulo: str = "Solución con RK4",
    nombres_variables: Optional[List[str]] = None,
    x_label: str = "x",
    y_label: str = "y(x)"
) -> None:
    """
    Grafica la solución aproximada de RK4 y la compara con la exacta si existe.
    """
    plt.figure(figsize=(9, 5))
    es_escalar = (y.ndim == 1)

    if es_escalar:
        if sol_exacta is not None:
            x_denso = np.linspace(x[0], x[-1], 300)
            plt.plot(x_denso, sol_exacta(x_denso), 'b-', label='Solución Exacta', linewidth=2)
        plt.plot(x, y, 'ro--', label='Aproximación RK4', markersize=5)
        plt.plot(x[-1], y[-1], 'g*', markersize=12, label=f'Punto de cálculo x={x[-1]}')
    else:
        dim = y.shape[1]
        if nombres_variables is None or len(nombres_variables) != dim:
            nombres_variables = [f"y{j+1}(x)" for j in range(dim)]
        for j in range(dim):
            plt.plot(x, y[:, j], 'o--', label=f'RK4: {nombres_variables[j]}')

    plt.axvline(x[-1], color='gray', linestyle=':', alpha=0.7)
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(y_label, fontsize=11)
    plt.title(titulo, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==============================================================================
# BLOQUE DE EJECUCIÓN: MODIFICA ESTA SECCIÓN PARA RESOLVER CUALQUIER PROBLEMA
# ==============================================================================

if __name__ == "__main__":
    # ==========================================================================
    # EJEMPLO 1: EDO DE PRIMER ORDEN ESCALAR
    # Resuelve: dy/dx = -2*x*y, con y(0) = 1, hasta x_final = 1.0 con h = 0.2
    # ==========================================================================
    print("\n" + "=" * 75)
    print("EJEMPLO 1: EDO ESCALAR (dy/dx = -2xy)")
    print("=" * 75)

    # 1. Definir la función dy/dx = f(x, y)
    def mi_edo(x: float, y: float) -> float:
        return y * (4 * y - x**2)

    # 2. Definir parámetros
    x0 = 0.0           # Punto inicial
    y0 = 1.0           # Condición inicial y(x0) = y0
    x_final = 3.0      # PUNTO DE CÁLCULO
    h = 0.1            # Tamaño de paso h (o alternativamente n_pasos = 5)

    # Solución analítica para comparar (opcional, poner None si no se tiene)
    sol_analitica = lambda x: np.exp(-x**2)

    # 3. Resolver con RK4
    x_calc, y_calc, detalles = rk4(
        mi_edo, x0, y0, x_final, h=h, retornar_detalles=True
    )

    # 4. Imprimir resultados y detalles de k1..k4
    imprimir_tabla(x_calc, y_calc, sol_exacta=sol_analitica)
    imprimir_detalles_k(detalles)

    print(f"\nResultado final en el punto de cálculo x = {x_final}:")
    print(f"  y_RK4({x_final})   = {y_calc[-1]:.8f}")
    if sol_analitica is not None:
        y_exacta_final = sol_analitica(x_final)
        print(f"  y_Exacta({x_final})= {y_exacta_final:.8f}")
        print(f"  Error absoluto    = {abs(y_calc[-1] - y_exacta_final):.2E}")

    # 5. Graficar
    graficar_solucion(
        x_calc, y_calc, sol_exacta=sol_analitica,
        titulo=f"RK4: $dy/dx = -2xy$ | Paso $h={h}$"
    )

    # ==========================================================================
    # EJEMPLO 2 (DEMOSTRACIÓN): SISTEMA DE EDOS / EDO DE 2DO ORDEN
    # Resuelve: y'' + y = 0  ->  y1' = y2, y2' = -y1
    # con y(0) = 0, y'(0) = 1 (cuya solución exacta es y(x) = sin(x))
    # ==========================================================================
    print("\n" + "=" * 75)
    print("EJEMPLO 2: SISTEMA DE EDOS (y'' + y = 0 -> Oscilador Armónico)")
    print("=" * 75)

    def sistema_oscilador(x: float, Y: np.ndarray) -> np.ndarray:
        # Y[0] = y, Y[1] = y'
        dy1_dx = Y[1]
        dy2_dx = -Y[0]
        return np.array([dy1_dx, dy2_dx])

    Y0 = [0.0, 1.0]           # [y(0), y'(0)]
    x_obj = float(np.pi)      # Punto de cálculo: x = pi
    n = 20                    # 20 pasos

    x_s, Y_s = rk4(sistema_oscilador, 0.0, Y0, x_obj, n_pasos=n)

    imprimir_tabla(x_s, Y_s, nombres_variables=["Posición y(x)", "Velocidad y'(x)"])

    print(f"\nEn el punto de cálculo x = pi:")
    print(f"  Posición y(pi) calculada:  {Y_s[-1, 0]:.8f} (Esperado: 0.0)")
    print(f"  Velocidad y'(pi) calculada:{Y_s[-1, 1]:.8f} (Esperado: -1.0)")
