import math

def f(x: float) -> float:
    return x**2 + 1

def integral_exacta(a: float, b: float) -> float:
    """
    Args:
        a: Límite inferior de integración.
        b: Límite superior de integración.
    
    Returns:
        El valor exacto de la integral.
    """
    integral_b = (b**3 / 3.0) + b
    integral_a = (a**3 / 3.0) + a
    return integral_b - integral_a

def regla_trapecio_compuesta(a: float, b: float, n: int) -> float:
    """
    Calcula la integral de una función usando la Regla del Trapecio Compuesta.    
    Args:
        a: Límite inferior de integración.
        b: Límite superior de integración.
        n: Número de subintervalos.
        
    Returns:
        El valor aproximado de la integral.
    """
    # Calcular h
    h = (b - a) / n
    
    # Iniciar la suma con los extremos f(a) y f(b)
    suma = f(a) + f(b)

    # Sumar los puntos interiores
    for i in range(1, n):
        xi = a + i * h
        suma += 2 * f(xi)
    
    # Multiplicar por h/2 para obtener el resultado final
    return suma * (h / 2.0)

def calcular_error_absoluto(valor_aproximado: float, valor_exacto: float) -> float:
    """
    Calcula el error absoluto exacto.
    Se basa en la fórmula e = |I_exacta - I_aproximada|.
    
    Args:
        valor_aproximado: El resultado del método numérico.
        valor_exacto: El resultado analítico.
        
    Returns:
        El error absoluto.
    """
    return abs(valor_aproximado - valor_exacto)

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # Ingresar los límites de integración
    a = float(input("Ingrese el limite inferior de integracion (a): "))
    b = float(input("Ingrese el limite superior de integracion (b): "))

    # Ingresar el número de subintervalos
    n = int(input("Ingrese el numero de subintervalos (n): "))

    # Calcular la integral aproximada
    resultado_aproximado = regla_trapecio_compuesta(a, b, n)
    resultado_exacto = integral_exacta(a, b)
    error = calcular_error_absoluto(resultado_aproximado, resultado_exacto)

    # Mostrar el resultado
    print("\nResultados para f(x) = x^2 + 1:")
    print("-----------------------------------")
    print(f"Integral aproximada: {resultado_aproximado:.6f}")
    print(f"Integral exacta:     {resultado_exacto:.6f}")
    print(f"Error absoluto exacto: {error:.6f}")