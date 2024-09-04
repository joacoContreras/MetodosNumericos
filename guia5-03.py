import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Datos proporcionados
T = np.array([-260.15, -200, -100, 0, 100, 300,500])
c_rho = np.array([0.1, 0.45, 0.699, 0.87, 0.941, 1.04,8.78])

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

# Construir el polinomio de Lagrange
polinomio = lagrange_interpolation(T, c_rho)
Pn = sym.lambdify(sym.Symbol('x'), polinomio)

# Graficar los datos y el polinomio
T_vals = np.linspace(-300, 600, 400)
plt.figure(figsize=(10, 8))
plt.plot(T, c_rho, 'o', label='Datos experimentales')
plt.plot(T_vals, Pn(T_vals), label='Polinomio de Lagrange')
plt.xlabel('Temperatura [°C]')
plt.ylabel('Calor específico [kJ/kg·K]')
plt.title('Interpolación de Lagrange del Calor Específico del Aluminio')
plt.legend()
plt.grid(True)
plt.show()

# Extrapolación a T = 500°C
T_extrap = 500
c_rho_extrap = Pn(T_extrap)
print(f"Valor extrapolado del calor específico a T = {T_extrap}°C: {c_rho_extrap} kJ/kg·K")
