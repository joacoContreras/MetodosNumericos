import math


def rk4(g, a: float, b: float, y0: float, h: float) -> tuple[list, list]:
    """
    Resuelve una EDO dy/dt = g(t, y) usando el método clásico de Runge-Kutta de 4to orden (RK4).
    Implementación del Problema 1 del Final de Diciembre 2025.

    Args:
        g: Función g(t, y) que define la EDO dy/dt = g(t, y).
        a: Límite inferior del intervalo (t0).
        b: Límite superior del intervalo (punto de cálculo tf).
        y0: Condición inicial y(a) = y0.
        h: Tamaño de paso.

    Returns:
        tiempos: Lista de valores de t desde a hasta b.
        valores: Lista de valores calculados y(t).
    """
    n = round((b - a) / h)
    if abs(a + n * h - b) > 1e-10:
        raise ValueError("El paso debe dividir la longitud del intervalo.")

    tiempos = [a + i * h for i in range(n + 1)]
    valores = [y0]

    for i in range(n):
        t, y = tiempos[i], valores[-1]
        k1 = g(t, y)
        k2 = g(t + h / 2.0, y + (h / 2.0) * k1)
        k3 = g(t + h / 2.0, y + (h / 2.0) * k2)
        k4 = g(t + h, y + h * k3)
        valores.append(y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    return tiempos, valores


def metodo_rk4(f, x0: float, y0: float, x_final: float, h: float = 0.1) -> tuple[list, list]:
    """
    Alias con firma (f, x0, y0, x_final, h) para compatibilidad.
    """
    return rk4(f, x0, x_final, y0, h)


# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------------------
    # 1. Parámetros del problema (Final Diciembre 2025 - Problema 1)
    #    Calentamiento / enfriamiento de una barra
    # -------------------------------------------------------------
    L = 4.0               # Longitud de la barra (m)
    T0 = 100.0            # Temperatura de contorno T0 (°C)
    T1 = 20.0             # Temperatura base / ambiente T1 (°C)
    TI = 10.0             # Tiempo inicial ti (s)
    TF = 20.0             # Tiempo final tf (s) (punto de cálculo)
    T_INICIAL = 60.0      # Condición inicial: T(10) = 60.0 °C

    # Cálculo analítico exacto de la constante C y la difusividad D
    C = (L**2 / math.pi**2) * math.log((T0 - T1) / (T_INICIAL - T1))
    D_EXACTO = C / TI     # D analítico ≈ 0.1123687884

    # EDO: T'(t) = t * [ (L^2 / (D * pi^2)) * ln((T0 - T1)/(T - T1)) - t ]
    def edo(t: float, temperatura: float) -> float:
        if temperatura <= T1:
            raise ValueError("El logaritmo requiere T > T1. Revisar el paso.")
        return t * ((L**2 / (D_EXACTO * math.pi**2)) * math.log((T0 - T1) / (temperatura - T1)) - t)

    print("=" * 75)
    print("FINAL DICIEMBRE 2025 - PROBLEMA 1: RESOLUCIÓN CON RK4")
    print("Interpretación EDO: T' = t * [ L²/(D*pi²) * ln((T0-T1)/(T-T1)) - t ]")
    print(f"Condiciones: L = {L}, T0 = {T0}, T1 = {T1}, T({TI}) = {T_INICIAL}°C, Intervalo: [{TI}, {TF}] s")
    print(f"D analítico = {D_EXACTO:.10f} (interno: {D_EXACTO:.17g})")
    print("=" * 75)

    # -------------------------------------------------------------
    # 2. Resolución con RK4 para h = 0.1 y h = 0.01
    # -------------------------------------------------------------
    soluciones = {}
    for h in [0.1, 0.01]:
        tiempos, valores = rk4(edo, TI, TF, T_INICIAL, h)
        j15 = round((15 - TI) / h)
        soluciones[h] = (tiempos, valores)
        print(f"\nResultados con h = {h}:")
        print(f"  Pasos: {len(valores) - 1}")
        print(f"  T(15) = {valores[j15]:.10f} °C")
        print(f"  T(20) = {valores[-1]:.10f} °C")

    # -------------------------------------------------------------
    # 3. Presentar tabla de resultados (muestra representativa h = 0.1)
    # -------------------------------------------------------------
    t_01, v_01 = soluciones[0.1]
    print("\n-------- TABLA DE SOLUCIÓN RK4 (h = 0.1) --------")
    print(f"{'Paso':<5} | {'Tiempo t (s)':<14} | {'Temperatura T(t) (°C)':<22} | {'Pendiente T\'(t)':<16}")
    print("-" * 65)
    for i in range(0, len(t_01), 10):  # Muestra cada 10 pasos (cada 1.0 s)
        t_val = t_01[i]
        T_val = v_01[i]
        deriv = edo(t_val, T_val)
        print(f"{i:<5} | {t_val:<14.2f} | {T_val:<22.8f} | {deriv:<16.8f}")
    # Asegurar mostrar el último punto si no coincidió
    if (len(t_01) - 1) % 10 != 0:
        print(f"{len(t_01)-1:<5} | {t_01[-1]:<14.2f} | {v_01[-1]:<22.8f} | {edo(t_01[-1], v_01[-1]):<16.8f}")
    print("-" * 65)

    # -------------------------------------------------------------
    # 4. Inciso e: Diferenciación centrada de orden h² en t = 15s (h = 0.01)
    # -------------------------------------------------------------
    t_001, v_001 = soluciones[0.01]
    j = round((15 - TI) / 0.01)
    centrada = (v_001[j + 1] - v_001[j - 1]) / (2 * 0.01)
    directa = edo(t_001[j], v_001[j])
    diff_abs = abs(centrada - directa)
    diff_rel = abs((centrada - directa) / directa)

    print("\nInciso e) Diferencia centrada de orden h² (t = 15 s, h = 0.01):")
    for indice in [j - 1, j, j + 1]:
        print(f"  T({t_001[indice]:.2f}) = {v_001[indice]:.14f}")
    print(f"  Derivada centrada        = {centrada:.12f}")
    print(f"  EDO evaluada en T_RK4(15)= {directa:.12f}")
    print(f"  Diferencia absoluta      = {diff_abs:.12e}")
    print(f"  Diferencia relativa      = {diff_rel:.12e}")
    print("  (La evaluación directa usa una T aproximada; no es una solución analítica).")

    # -------------------------------------------------------------
    # 5. Graficar la evolución de la temperatura
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(t_01, v_01, color='#1864ab', label='RK4 (h = 0.1)', linewidth=2)
    plt.plot(t_001, v_001, color='#e03131', linestyle='--', label='RK4 (h = 0.01)', linewidth=1.5, alpha=0.8)
    plt.axvline(15.0, color='gray', linestyle=':', label='t = 15 s')
    plt.axvline(20.0, color='gray', linestyle='--', label='t = 20 s')
    plt.plot(15.0, v_01[round((15 - TI) / 0.1)], 'bo', markersize=6)
    plt.plot(20.0, v_01[-1], 'bs', markersize=6)
    plt.xlabel('Tiempo t (s)', fontsize=12)
    plt.ylabel('Temperatura T(t) (°C)', fontsize=12)
    plt.title('Temperatura de la barra — RK4 (Final Diciembre 2025)\n$T\'(t) = t \\cdot [\\frac{L^2}{D\\cdot\\pi^2} \\ln(\\frac{T_0-T_1}{T-T_1}) - t]$', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
