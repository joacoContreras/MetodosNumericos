import math
import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import brentq

# Definición de la función
def f(x):
    return 1.1236878843 / x - 10

# Método de Regla Falsa
def regla_falsa(f, a, b, tol):
    if f(a) * f(b) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de regla falsa.")
        return None, []

    iteracion = 0
    errores = []
    c = a
    error = float('inf')

    while error > tol:
        c_prev = c
        fa = f(a)
        fb = f(b)
        c = (a * fb - b * fa) / (fb - fa)
        iteracion += 1
        
        if iteracion == 1:
            print("1era Iteracion: c = " , c)
            
        if iteracion == 10:
            print("10ima Iteracion: c = " , c)
            
        if iteracion > 1:
            error = abs((c - c_prev) / c)
            errores.append(error)

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    return c, errores

def calcular_errores(raiz_aprox, raiz_exacta, f=None):
    error_absoluto = abs(raiz_exacta - raiz_aprox)
    error_relativo = abs(error_absoluto / raiz_exacta) if raiz_exacta != 0 else float('inf')
    error_porcentual = error_relativo * 100
    print(f"Error absoluto exacto: {error_absoluto:e}")
    print(f"Error relativo exacto: {error_relativo:e}")
    print(f"Error porcentual exacto: {error_porcentual:.8f}%")
    if f is not None:
        residuo = abs(f(raiz_aprox))
        print(f"Residuo |f(raíz aprox)|: {residuo:e}")

# Valores iniciales
a = 0.05
b = 1.0
tol = 1e-5

# Ejecutar métodos
raiz_regla_falsa, errores_regla_falsa = regla_falsa(f, a, b, tol)

# Imprimir resultados

print(f"\nMétodo de Regla Falsa:")
print(f"Raíz encontrada: {raiz_regla_falsa}")
print(f"Iteraciones: {len(errores_regla_falsa)}")

# Calcular y mostrar errores exactos
if raiz_regla_falsa is not None:
    try:
        # Raíz de referencia calculada dinámicamente con alta precisión a partir de f en [a, b]
        raiz_exacta = brentq(f, a, b)
        print(f"Raíz exacta de referencia (scipy.brentq): {raiz_exacta:.10f}")
        calcular_errores(raiz_regla_falsa, raiz_exacta, f)
    except Exception as e:
        raiz_exacta = None
        print(f"No se pudo calcular la raíz de referencia con brentq: {e}")
        if f is not None:
            print(f"Residuo |f(raíz aprox)|: {abs(f(raiz_regla_falsa)):e}")

    # Graficar resultados
    plt.figure(figsize=(12, 6))

    # Subplot 1: Gráfico de la función f(x) y la raíz encontrada
    plt.subplot(1, 2, 1)
    x_vals = np.linspace(a, b, 400)
    try:
        y_vals = f(x_vals)
        if not isinstance(y_vals, np.ndarray) or y_vals.shape != x_vals.shape:
            y_vals = np.array([f(xi) for xi in x_vals])
    except Exception:
        y_vals = np.array([f(xi) for xi in x_vals])

    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.plot(raiz_regla_falsa, f(raiz_regla_falsa), 'ro', label=f'Raíz aprox ≈ {raiz_regla_falsa:.6f}')
    if raiz_exacta is not None:
        plt.plot(raiz_exacta, f(raiz_exacta), 'k*', markersize=10, label=f'Raíz ref ≈ {raiz_exacta:.6f}')
    plt.plot([a, b], [f(a), f(b)], 'gs', label=f'Intervalo [{a}, {b}]')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Función f(x) y Raíz Encontrada')
    plt.grid(True)
    plt.legend()

    # Subplot 2: Regla falsa - Error por iteración
    plt.subplot(1, 2, 2)
    iteraciones = range(2, len(errores_regla_falsa) + 2)
    plt.plot(iteraciones, errores_regla_falsa, marker='o', linestyle='-', color='red', label='Error relativo aprox.')
    plt.yscale('log')
    plt.xlabel('Iteración')
    plt.ylabel('Error Relativo')
    plt.title('Error por Iteración - Regla Falsa')
    plt.grid(True, which='both', linestyle=':')
    plt.legend()

    plt.tight_layout()
    plt.show()
