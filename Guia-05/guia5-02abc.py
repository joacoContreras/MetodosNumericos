"""
Conceptos Clave:
Interpolación vs. Extrapolación:

Interpolación: Es la estimación de valores dentro del rango 
de los puntos de datos utilizados para construir 
el polinomio. Generalmente, la interpolación es precisa 
dentro del rango.

Extrapolación: Es la estimación de valores fuera del 
rango de los puntos de datos utilizados. La extrapolación, 
especialmente con polinomios de Lagrange, puede ser poco 
precisa y puede llevar a grandes errores. Los polinomios de 
Lagrange pueden oscilar y divergir significativamente fuera 
del rango de interpolación, un fenómeno conocido como 
"fenómeno de Runge" en el caso de polinomios de orden alto.

Objetivo del Análisis:
Verificar si el polinomio de Lagrange proporciona una buena 
aproximación fuera del rango de los datos.
Graficar la función original y el polinomio interpolador en
un rango extendido, es decir, evaluando el polinomio en 
puntos fuera de los límites 𝑥0 y xN 
utilizados en la interpolación.


"""
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Definir la función f(x)
def f(x):
    return 4*x**3 - 3*x**2 + 2

# Función para calcular el polinomio de Lagrange
def lagrange_interpolation(xi, fi):
    n = len(xi)
    x = sym.Symbol('x')
    polinomio = 0
    for k in range(n):
        numerador = 1
        denominador = 1
        for j in range(n):
            if j != k:
                numerador *= (x - xi[j])
                denominador *= (xi[k] - xi[j])
        Cn_k = numerador / denominador
        polinomio += Cn_k * fi[k]
    
    return polinomio.expand()

# Rango extendido de x para análisis de extrapolación
x_vals_ext = np.linspace(-2, 2, 400)  # Aumenta el rango a [-2, 2]

# Grados a utilizar
grados = [1, 2, 3]

# Graficar f(x) original en el rango extendido
plt.figure(figsize=(10, 8))
plt.plot(x_vals_ext, f(x_vals_ext), label="f(x) = 4x^3 - 3x^2 + 2", color='black', linewidth=2)

# Realizar la interpolación y graficar
for n in grados:
    xi = np.linspace(-1, 1, n + 1)  # Puntos de interpolación en el rango [-1, 1]
    fi = f(xi)
    
    # Obtener el polinomio de Lagrange
    polinomio = lagrange_interpolation(xi, fi)
    Pn = sym.lambdify(sym.Symbol('x'), polinomio)
    
    # Graficar el polinomio de Lagrange en el rango extendido
    plt.plot(x_vals_ext, Pn(x_vals_ext), label=f'Pn(x), n={n}')
    
    # Calcular y graficar el error en el rango extendido
    error_ext = np.abs(f(x_vals_ext) - Pn(x_vals_ext))
    plt.figure(figsize=(10, 8))
    plt.plot(x_vals_ext, error_ext, label=f'Error e(x) = |f(x) - Pn(x)|, n={n}')
    plt.xlabel('x')
    plt.ylabel('Error e(x)')
    plt.title(f'Error para n={n} en rango extendido')
    plt.grid(True)
    plt.legend()
    plt.show()

# Mostrar la gráfica de la función original con los polinomios en el rango extendido
plt.xlabel('x')
plt.ylabel('f(x) y Pn(x)')
plt.title('Aproximación de Lagrange y Extrapolación')
plt.grid(True)
plt.legend()
plt.show()

