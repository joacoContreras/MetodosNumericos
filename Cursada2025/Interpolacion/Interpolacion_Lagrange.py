# Interpolación de Lagrange mejorada
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Definir la función f(x)
def f(x):
    return x + (2/x)  # Ejemplo de función, puedes cambiarla

# Puntos xi
xi = np.array([1, 2, 2.5])

# Calcular los valores de f(x) en xi
fi = f(xi)

# PROCEDIMIENTO
# Polinomio de Lagrange
n = len(xi)
x = sym.Symbol('x')
polinomio = 0
divisorL = np.zeros(n, dtype=float)
for i in range(0, n, 1):

    # Término de Lagrange
    numerador = 1
    denominador = 1
    for j in range(0, n, 1):
        if j != i:
            numerador = numerador * (x - xi[j])
            denominador = denominador * (xi[i] - xi[j])
    terminoLi = numerador / denominador

    polinomio = polinomio + terminoLi * fi[i]
    divisorL[i] = denominador

# Simplifica el polinomio
polisimple = polinomio.expand()

# Para evaluación numérica
px = sym.lambdify(x, polisimple)

# Puntos para la gráfica
muestras = 101
a = np.min(xi)
b = np.max(xi)
pxi = np.linspace(a, b, muestras)
pfi = px(pxi)

# SALIDA
print('    valores de fi: ', fi)
print('divisores en L(i): ', divisorL)
print()
print('Polinomio de Lagrange, expresiones')
print(polinomio)
print()
print('Polinomio de Lagrange simplificado: ')
print(polisimple)

# Gráfica
plt.plot(xi, fi, 'o', label='Puntos')
plt.plot(pxi, pfi, label='Polinomio')
plt.legend()
plt.xlabel('xi')
plt.ylabel('fi')
plt.title('Interpolación Lagrange')
plt.show()
