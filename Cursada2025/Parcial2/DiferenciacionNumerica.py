import numpy as np

def calcular_derivadas_hibrido(x_puntos: np.ndarray, y_puntos: np.ndarray) -> np.ndarray:
    """
    Calcula las derivadas de un conjunto de puntos (x, y) equiespaciados
    usando la estrategia híbrida (la mejor para los exámenes):
    - Adelante para el primer punto.
    - Centradas para los puntos interiores.
    - Atrás para el último punto.

    Args:
        x_puntos: Array de numpy con las coordenadas x (equiespaciadas).
        y_puntos: Array de numpy con las coordenadas y.

    Returns:
        Un array de numpy con el valor de la derivada en cada punto.
    """
    
    n = len(x_puntos)
    if n < 3:
        raise ValueError("Se necesitan al menos 3 puntos para el método híbrido.")
    
    # Asumimos equiespaciado
    h = x_puntos[1] - x_puntos[0]
    
    # Crear un array vacío para las derivadas
    derivadas = np.zeros(n)
    
    # 1. Primer punto (Adelante)
    derivadas[0] = (y_puntos[1] - y_puntos[0]) / h
    
    # 2. Puntos interiores (Centradas)
    for i in range(1, n - 1):
        derivadas[i] = (y_puntos[i + 1] - y_puntos[i - 1]) / (2 * h)
        
    # 3. Último punto (Atrás)
    derivadas[n - 1] = (y_puntos[n - 1] - y_puntos[n - 2]) / h
    
    return derivadas

# --- Ejemplo de uso ---
# Vamos a probar con una función conocida: f(x) = x^3
# La derivada exacta es f'(x) = 3x^2

n_puntos_tabla = 6
x_tabla = np.linspace(0, 5, n_puntos_tabla)  # Puntos: 0, 1, 2, 3, 4, 5
y_tabla = x_tabla**3                         # Puntos: 0, 1, 8, 27, 64, 125

# Calcular las derivadas con nuestro método
derivadas_calculadas = calcular_derivadas_hibrido(x_tabla, y_tabla)

# Calcular las derivadas exactas para comparar
derivadas_exactas = 3 * x_tabla**2

# --- Imprimir la tabla comparativa ---
print("--- Ejemplo de Diferenciación Numérica (con f(x)=x^3) ---")
print(f"{'x':<10} {'f(x)':<10} {'f\'(x) Numérica':<18} {'f\'(x) Exacta':<15}")
print("-" * 55)
for i in range(n_puntos_tabla):
    print(f"{x_tabla[i]:<10.1f} {y_tabla[i]:<10.1f} {derivadas_calculadas[i]:<18.2f} {derivadas_exactas[i]:<15.2f}")

# Nota: Verás que el error es 0 en los puntos centrales (2.0 y 3.0).
# ¡Esto es porque la derivada centrada es exacta para parábolas y 
# tu función x^3 se comporta "localmente" como una parábola en esos puntos!
# El error es mayor en los extremos (0.0 y 5.0) porque los métodos
# de adelante/atrás son O(h) y no O(h^2).