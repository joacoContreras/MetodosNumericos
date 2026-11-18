import math

def regla_simpson_compuesta(f, a: float, b: float, n: int) -> float:
    """
    Calcula la integral de una función usando la Regla de Simpson 1/3 Compuesta.
    
    Args:
        f: La función a integrar (debe aceptar un float y devolver un float).
        a: Límite inferior de integración.
        b: Límite superior de integración.
        n: Número de subintervalos (debe ser par).
        
    Returns:
        El valor aproximado de la integral.
        
    Raises:
        ValueError: Si el número de subintervalos n no es par.
    """
    # Validar que la cantidad de tramos sea par
    if n % 2 != 0:
        raise ValueError('El número de subintervalos (n) debe ser par.')

    h = (b - a) / n
    
    # Inicializar la suma con los valores de los extremos 
    suma = f(a) + f(b)

    # Sumar los valores de los puntos intermedios con coeficiente 4 (impares)
    for i in range(1, n, 2):
        x = a + i * h
        suma += 4 * f(x)

    # Sumar los valores de los puntos intermedios con coeficiente 2 (pares)
    for i in range(2, n, 2):
        x = a + i * h
        suma += 2 * f(x)

    # Multiplicar por h/3 para el resultado final
    integral = (h / 3) * suma
    return integral

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # Definición de la función a integrar
    fx = lambda x: (math.e**(2 * math.sin(x))) * (1 + math.log(x))

    # Parámetros de la integración
    a = 1
    b = 2
    n = 10
    
    try:
        resultado = regla_simpson_compuesta(fx, a, b, n)
        # SALIDA
        print(f"Usando n = {n} subintervalos:")
        print(f"El valor de la integral es: {resultado}")
    except ValueError as e:
        print(f"Error: {e}")