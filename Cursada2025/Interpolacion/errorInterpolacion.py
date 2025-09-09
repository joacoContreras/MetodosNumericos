import numpy as np
import matplotlib.pyplot as plt

# Definir la función exacta
def f(x):
    return np.log(x**2 + 1) - np.sin(x)

# Datos de la tabla
x_datos = np.array([1.0, 1.2, 1.5, 1.75, 2.0])
y_datos = np.array([-0.148, -0.040, 0.181, 0.419, 0.700])

# Polinomio interpolador (grado 4 con polyfit)
coef = np.polyfit(x_datos, y_datos, 4)
P = np.poly1d(coef)

# Definimos un rango de x para ver el error
x_vals = np.linspace(1.0, 2.0, 200)
f_vals = f(x_vals)
P_vals = P(x_vals)

# Error absoluto
error = np.abs(P_vals - f_vals)

# Graficar
plt.plot(x_vals, error, label="Error |P(x) - f(x)|", color="red")
plt.scatter(x_datos, np.zeros_like(x_datos), color="blue", zorder=5, label="Nodos (error=0)")
plt.xlabel("x")
plt.ylabel("Error absoluto")
plt.title("Función error del polinomio interpolador")
plt.legend()
plt.grid(True)
plt.show()
