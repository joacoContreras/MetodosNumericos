import numpy as np
import matplotlib.pyplot as plt

# Sample data points (x, y)
# Replace these with your data
x = np.array([0.5, 0.8, 1.3, 2.0])
y = np.array([-0.716, -0.103, 3.419, 52.598])

# Se calcula la media de los valores de x y y usando np.mean(). 
# Estas medias se utilizan para encontrar la pendiente 
# y la intersección de la línea de mejor ajuste
mean_x = np.mean(x)
mean_y = np.mean(y)

# Cálculo de la pendiente (a) y la intersección (b):
numerator = np.sum((x - mean_x) * (y - mean_y))
denominator = np.sum((x - mean_x) ** 2)

# Calcular pendiente (a)
a = numerator / denominator

# Calcular intercepcion(b)
b = mean_y - a * mean_x

print(f"a = {a}, b = {b}")

# Generar linea de mejor ajuste
best_fit_line = a * x + b

# Graficar data points y the best-fit line
plt.scatter(x, y, label='Data points')
plt.plot(x, best_fit_line, label='Best fit line', color='red')
plt.legend()
plt.show()
