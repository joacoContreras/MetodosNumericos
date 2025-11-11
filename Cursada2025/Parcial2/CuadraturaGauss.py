import numpy as np
from scipy.integrate import quad

def cuadratura_gauss(f, a: float, b: float, n_puntos: int) -> float:
    """
    Calcula la integral de una función f de [a, b] usando la 
    Cuadratura de Gauss-Legendre con n_puntos.
    
    Args:
        f: La función a integrar (debe ser 'callable', como f(x) o un spline).
        a: Límite inferior de integración.
        b: Límite superior de integración.
        n_puntos: El número de puntos de Gauss a usar (2, 3, 4, 5, o 6).
        
    Returns:
        El valor aproximado de la integral.
    """
    
    # Tabla de Puntos (t) y Pesos (c) de Gauss-Legendre
    datos_gauss = {
        2: (np.array([-0.5773502691896257, 0.5773502691896257]),
            np.array([1.0, 1.0])),
        
        3: (np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
            np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])),
            
        4: (np.array([-0.8611363115940526, -0.3399810435848563, 
                       0.3399810435848563, 0.8611363115940526]),
            np.array([0.3478548451374538, 0.6521451548625461, 
                       0.6521451548625461, 0.3478548451374538])),
                       
        5: (np.array([-0.9061798459386640, -0.5384693101056831, 0.0, 
                       0.5384693101056831, 0.9061798459386640]),
            np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 
                       0.4786286704993665, 0.2369268850561891])),
                       
        6: (np.array([-0.9324695142031521, -0.6612093864662645, -0.2386191860831969,
                       0.2386191860831969, 0.6612093864662645, 0.9324695142031521]),
            np.array([0.1713244923791704, 0.3607615730481386, 0.4679139345726910,
                       0.4679139345726910, 0.3607615730481386, 0.1713244923791704]))
    }
    
    if n_puntos not in datos_gauss:
        raise ValueError(f"No hay datos de Gauss para {n_puntos} puntos. Use de 2 a 6.")
        
    puntos_t, pesos_c = datos_gauss[n_puntos]
    
    suma = 0.0
    
    # Constantes para el cambio de variable
    const_mult = (b - a) / 2.0
    const_suma = (b + a) / 2.0
    
    # Aplicar la fórmula: Sum(c_i * f(x_i))
    for i in range(n_puntos):
        # 1. Convertir el punto t_i del intervalo [-1, 1] al punto x_i en [a, b]
        x_i = const_mult * puntos_t[i] + const_suma
        
        # 2. Evaluar la función en ese punto x_i
        f_de_x_i = f(x_i)
        
        # 3. Añadir a la suma ponderada
        suma += pesos_c[i] * f_de_x_i
        
    # Multiplicar la suma por la constante (b-a)/2
    integral = const_mult * suma
    
    return integral

# --- Función a integrar --- 
# Acordarse que se integra g(x) = f(x) / x
def mi_funcion(x):
    return (np.log(x**2 + 1) - np.sin(x)) / x

a_lim = 1.0
b_lim = 2.0

# Calcular integrales con Gauss
integral_g2 = cuadratura_gauss(mi_funcion, a_lim, b_lim, 2)
integral_g6 = cuadratura_gauss(mi_funcion, a_lim, b_lim, 6)

# Valor de referencia con scipy.integrate.quad
integral_ref, _ = quad(mi_funcion, a_lim, b_lim)

print(f"Integral de ln(x²+1) - sin(x) en [1, 2]:")
print(f"Con 2 puntos Gauss: {integral_g2:.10f}")
print(f"Con 6 puntos Gauss: {integral_g6:.10f}")
print(f"Referencia (quad):  {integral_ref:.10f}")
print(f"Error Gauss n=2: {abs(integral_ref-integral_g2):.2e}")
print(f"Error Gauss n=6: {abs(integral_ref-integral_g6):.2e}")