import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

#El enunciado sugiere que agregar más puntos (aumentar n) 
# no siempre mejora la aproximación en todo el dominio. 
# Esto se debe a un fenómeno conocido como el 
# "fenómeno de Runge", que ocurre cuando los polinomios 
# de mayor grado pueden oscilar excesivamente y proporcionar 
# una peor aproximación.

def f(x):
    return np.exp(-x**2)

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

# Rango de x
x_vals = np.linspace(-3, 3, 400)

# Grados a utilizar
grados = [2, 4, 6, 8]

# Graficar f(x) original
plt.figure(figsize=(10, 8))
plt.plot(x_vals, f(x_vals), label="f(x) = exp(-x^2)", color='black', linewidth=2)

# Realizar la interpolación y graficar
for n in grados:
    xi = np.linspace(-2, 2, n + 1)
    fi = f(xi)
    
    # Obtener el polinomio de Lagrange
    polinomio = lagrange_interpolation(xi, fi)
    Pn = sym.lambdify(sym.Symbol('x'), polinomio)
    
    # Graficar el polinomio de Lagrange
    plt.plot(x_vals, Pn(x_vals), label=f'Pn(x), n={n}')
    
    # Calcular y graficar el error
    error = np.abs(f(x_vals) - Pn(x_vals))
    plt.figure(figsize=(10, 8))
    plt.plot(x_vals, error, label=f'Error e(x) = |f(x) - Pn(x)|, n={n}')
    plt.xlabel('x')
    plt.ylabel('Error e(x)')
    plt.title(f'Error para n={n}')
    plt.grid(True)
    plt.legend()
    plt.show()

# Mostrar la gráfica de la función original con los polinomios
plt.xlabel('x')
plt.ylabel('f(x) y Pn(x)')
plt.title('Aproximación de Lagrange')
plt.grid(True)
plt.legend()
plt.show()
