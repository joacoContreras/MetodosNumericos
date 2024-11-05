import numpy as np
import matplotlib.pyplot as plt

# Definir la función de dos variables f(x, y)
def funcion(x, y):
    return (3*(x**2)-1)/y  # Puedes cambiar esta función

# Crear una malla de puntos para graficar la función
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = funcion(X, Y)

# Crear el gráfico en 2D
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, cmap='viridis', levels=50)
plt.colorbar(contour)

# Definir puntos personalizados y graficarlos
puntos = [
    (0.20000, 0.80000),
    (0.40000, 0.58000),
    (1.80000, 2.92632),
    (2, 3.52229)
]

# Graficar los puntos en el mismo gráfico
for px, py in puntos:
    plt.plot(px, py, 'ro')  # 'ro' indica puntos rojos

# Configuración del gráfico
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gráfico de función de dos variables y puntos personalizados')

# Mostrar el gráfico
plt.show()
