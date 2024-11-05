import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la función de dos variables f(x, y)
def funcion(x, y):
    return (3*(x**2)-1)/y # Puedes cambiar esta función

# Crear una malla de puntos para graficar la función
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = funcion(X, Y)

# Crear la figura y el gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie de la función
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Definir puntos personalizados y graficarlos en el mismo gráfico
puntos = [
    (0.20000, 0.80000, funcion(0.20000, 0.80000)),
    (0.40000, 0.58000, funcion(0.40000, 0.58000)),
    (1.80000, 2.92632, funcion(1.80000, 2.92632)),
    (2, 3.52229, funcion(2, 3.52229))
    # Puedes agregar más puntos aquí
]

# Separar coordenadas para el gráfico de puntos
px, py, pz = zip(*puntos)
ax.scatter(px, py, pz, color='r', s=50, label='Puntos personalizados')

# Configuración adicional del gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gráfico de función y puntos personalizados')
ax.legend()

# Mostrar el gráfico
plt.show()
