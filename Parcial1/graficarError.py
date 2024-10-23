import numpy as np
import matplotlib.pyplot as plt
# Definición de la función f(x)
def f(x):
    return np.log(x**2 + 1) - np.sin(x)
# Definición del polinomio P(x)
def P(x):
    return -0.05697 * x**4 + 0.3081 * x**3 - 0.2229 * x**2 + 0.2149 * x - 0.3911
# Rango de x
x_values = np.linspace(1.0, 2.0, 100)
f_values = f(x_values)
P_values = P(x_values)
# Calcular el error
error_values = np.abs(P_values - f_values)
# Graficar
plt.figure(figsize=(10, 6))
plt.plot(x_values, error_values, label='Error $e(x) = |P(x) - f(x)|$', color='blue')
plt.scatter([1.0, 1.2, 1.5, 1.75, 2.0], [0, 0, 0, 0, 0], color='red', zorder=5, label='Nodos')
plt.title('Función de Error')
plt.xlabel('x')
plt.ylabel('Error')
plt.axhline(0, color='black',linewidth=0.5, ls='--')
plt.axvline(1.0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(2.0, color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.grid()
plt.show()