import numpy as np
import math

# Definir la función f(x, y)
def f(x, y):
    return -2 * y + x  # Puedes reemplazar esto por la función específica del problema

# Parámetros de entrada
x0 = 0  # Valor inicial de x
y0 = 1  # Valor inicial de y
xfinal = 2  # Valor final de x
n = 10  # Número de divisiones
h = (xfinal - x0) / n  # Tamaño del paso

# Inicializar los vectores para almacenar los resultados
x = np.zeros(n + 1)
y1 = np.zeros(n + 1)
y2 = np.zeros(n + 1)
y3 = np.zeros(n + 1)
Q = np.zeros(n + 1)

# Condiciones iniciales
x[0] = x0
y1[0] = y2[0] = y3[0] = y0

# I. Calcular y1 (método de Euler)
for i in range(n):
    y1[i + 1] = y1[i] + h * f(x[i], y1[i])
    x[i + 1] = x[i] + h

# II. Calcular y2 (Euler modificado)
for i in range(n):
    yp = y2[i] + h / 2 * f(x[i], y2[i])
    xp = x[i] + h / 2
    y2[i + 1] = y2[i] + h * f(xp, yp)

# III. Calcular y3 (Euler mejorado con fracciones de h/4)
for i in range(n):
    yp = y3[i] + h / 4 * f(x[i], y3[i])
    xp = x[i] + h / 4

    yp = yp + h / 4 * f(xp, yp)
    xp = xp + h / 4

    yp = yp + h / 4 * f(xp, yp)
    xp = xp + h / 4

    y3[i + 1] = yp + h / 4 * f(xp, yp)

# IV. Calcular Q
for i in range(n + 1):
    if abs(y2[i] - y3[i]) > 1e-10:  # Evitar división por cero
        Q[i] = (1 / math.log(2)) * math.log(abs((y1[i] - y2[i]) / (y2[i] - y3[i])))
    else:
        Q[i] = float('inf')  # Definir un valor alto si no se puede calcular Q

# V. Guardar los resultados en un archivo
with open("resultados.txt", "w") as archivo:
    archivo.write("x\ty1\ty2\ty3\tQ\n")
    for i in range(n + 1):
        archivo.write(f"{x[i]:.5f}\t{y1[i]:.5f}\t{y2[i]:.5f}\t{y3[i]:.5f}\t{Q[i]:.5f}\n")

print("Cálculos completados. Los resultados se guardaron en 'resultados.txt'.")
