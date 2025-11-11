import numpy as np

def regla_trapecio_compuesta(x_puntos: np.ndarray, y_puntos: np.ndarray) -> float:
    """
    Calcula la integral de un conjunto de puntos (x, y) usando la 
    Regla del Trapecio Compuesta.
    Asume que los puntos en x están equiespaciados.
    
    Args:
        x_puntos: Array de numpy con las coordenadas x.
        y_puntos: Array de numpy con las coordenadas y.
        
    Returns:
        El valor aproximado de la integral.
    """
    # Validar que tengamos la misma cantidad de puntos
    if len(x_puntos) != len(y_puntos):
        raise ValueError("Los arrays de x e y deben tener la misma longitud.")
    if len(x_puntos) < 2:
        raise ValueError("Se necesitan al menos 2 puntos.")
        
    # Calcular h (asumiendo equiespaciado)
    h = x_puntos[1] - x_puntos[0]
    
    # Aplicar la fórmula: I = h/2 * [y0 + yn + 2 * (y1 + ... + yn-1)]
    suma_interna = np.sum(y_puntos[1:-1]) # Suma todos menos el primero y el último
    
    integral = (h / 2.0) * (y_puntos[0] + y_puntos[-1] + 2 * suma_interna)
    
    return integral

def regla_simpson_compuesta(x_puntos: np.ndarray, y_puntos: np.ndarray) -> float:
    """
    Calcula la integral de un conjunto de puntos (x, y) usando la 
    Regla de Simpson 1/3 Compuesta.
    Asume que los puntos en x están equiespaciados.
    
    Args:
        x_puntos: Array de numpy con las coordenadas x.
        y_puntos: Array de numpy con las coordenadas y.
        
    Returns:
        El valor aproximado de la integral.
    """
    n_puntos = len(x_puntos)
    
    # Validaciones
    if n_puntos != len(y_puntos):
        raise ValueError("Los arrays de x e y deben tener la misma longitud.")
    # n = n_puntos - 1 (número de intervalos)
    if (n_puntos - 1) % 2 != 0:
        raise ValueError(f"Simpson 1/3 requiere un número PAR de intervalos, "
                         f"pero se recibieron {n_puntos-1} (con {n_puntos} puntos).")
    if n_puntos < 3:
        raise ValueError("Se necesitan al menos 3 puntos.")

    # Calcular h (asumiendo equiespaciado)
    h = x_puntos[1] - x_puntos[0]
    
    # Aplicar la fórmula: I = h/3 * [y0 + yn + 4*(y_impares) + 2*(y_pares)]
    
    # Suma de términos impares (y1, y3, y5, ...)
    suma_impares = np.sum(y_puntos[1:-1:2]) # Inicia en 1, hasta el final, de 2 en 2
    
    # Suma de términos pares (y2, y4, y6, ...)
    suma_pares = np.sum(y_puntos[2:-1:2]) # Inicia en 2, hasta el final, de 2 en 2
    
    integral = (h / 3.0) * (y_puntos[0] + y_puntos[-1] + 4 * suma_impares + 2 * suma_pares)
    
    return integral

# --- Ejemplo de uso con los datos del Spline de la lección anterior ---
# Supongamos que generamos 11 PUNTOS (es decir, 10 INTERVALOS)
x_tabla = np.linspace(1.0, 2.0, 11) 
# Estos 'y_tabla' serían los valores f(x) o f'(x) de nuestra tabla
# Solo como ejemplo, usemos f(x) = x^2
y_tabla = x_tabla**2 

# x_tabla = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
# y_tabla = [1.0, 1.21, 1.44, 1.69, 1.96, 2.25, 2.56, 2.89, 3.24, 3.61, 4.0]

print("--- Ejemplo de Integración Numérica (con f(x)=x^2) ---")
# 1. Con Trapecio
integral_trapecio = regla_trapecio_compuesta(x_tabla, y_tabla)
print(f"Integral con Trapecio: {integral_trapecio:.6f}")

# 2. Con Simpson 1/3
# Tenemos 11 puntos, o sea 10 intervalos (n=10), es PAR, por lo que podemos usar Simpson.
integral_simpson = regla_simpson_compuesta(x_tabla, y_tabla)
print(f"Integral con Simpson 1/3: {integral_simpson:.6f}")

# El valor exacto de la integral de x^2 de 1 a 2 es (2^3/3 - 1^3/3) = 7/3 = 2.333333
print(f"Valor Exacto: {7/3:.6f}")
print("Nota: Simpson es mucho más preciso.")