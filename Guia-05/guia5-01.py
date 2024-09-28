import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Definir la función f(x)
def f(x):
    return x + 2/x  # Ejemplo de función, puedes cambiarla

# Puntos xi
xi = np.array([1, 2, 2.5])

# Calcular los valores de f(x) en xi
fi = f(xi)

# PROCEDIMIENTO
n = len(xi)
x = sym.Symbol('x')
polinomio = 0
Cn_k_list = []

# Calcular los coeficientes de Lagrange Cn,k(x) y verificar las propiedades
for k in range(n):
    numerador = 1
    denominador = 1
    for j in range(n):
        if j != k:
            numerador *= (x - xi[j])
            denominador *= (xi[k] - xi[j])
    
    Cn_k = numerador / denominador
    Cn_k_list.append(Cn_k)

    # Verificación de las propiedades
    Cn_k_xk = Cn_k.subs(x, xi[k])
    Cn_k_xj = [Cn_k.subs(x, xi[j]) for j in range(n) if j != k]

    print(f"Cn,k(xk) for k={k}: {Cn_k_xk} (Should be 1)")
    print(f"Cn,k(xj) for j != k: {Cn_k_xj} (Should be 0)")

    polinomio += Cn_k * fi[k]

# Simplificar el polinomio
polisimple = polinomio.expand()

# Suma de Cn,k(x) para k=0 hasta N
suma_Cn_k = sum(Cn_k_list).simplify()
print("\nSumatoria de Cn,k(x) para k=0 hasta N:", suma_Cn_k, "(Should be 1)")

# Para evaluación numérica
px = sym.lambdify(x, polisimple)

# Puntos para la gráfica
muestras = 101
a = np.min(xi)
b = np.max(xi)
pxi = np.linspace(a, b, muestras)
pfi = px(pxi)

# Gráfico de los coeficientes de Lagrange Cn,k(x)
plt.figure(figsize=(12, 8))
for k in range(n):
    Cn_k_func = sym.lambdify(x, Cn_k_list[k])
    plt.plot(pxi, Cn_k_func(pxi), label=f'C{n},{k}(x)')
plt.legend()
plt.xlabel('x')
plt.ylabel('Cn,k(x)')
plt.title('Coeficientes de Lagrange Cn,k(x)')
plt.grid(True)
plt.show()

# Gráfico del polinomio de Lagrange PN(x) y el error e(x)
# Calculando el error e(x) = |f(x) - PN(x)|
error_func = np.abs(f(pxi) - pfi)

plt.figure(figsize=(12, 8))
plt.plot(pxi, f(pxi), label='f(x)')
plt.plot(pxi, pfi, label='PN(x)', linestyle='--')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x) y PN(x)')
plt.title('Polinomio de Lagrange y f(x)')
plt.grid(True)
plt.show()

# Gráfico del error e(x)
plt.figure(figsize=(12, 8))
plt.plot(pxi, error_func, label='e(x) = |f(x) - PN(x)|')
plt.xlabel('x')
plt.ylabel('Error e(x)')
plt.title('Error e(x) = |f(x) - PN(x)|')
plt.grid(True)
plt.show()

# Verificar que e(xk) = 0 para todos k
print("\nVerificación que e(xk) = 0:")
for k in range(n):
    error_en_xk = np.abs(f(xi[k]) - px(xi[k]))
    print(f"e(x{k}) = {error_en_xk} (Should be 0)")
