import numpy as np

# Definir las funciones f1 y f2
def f1(x, y1, y2):
    return y2  # Función de ejemplo, ajusta según el problema específico

def f2(x, y1, y2):
    return -y1  # Función de ejemplo, ajusta según el problema específico

# Parámetros iniciales
x0 = 0  # Valor inicial de x
y10 = 1  # Valor inicial de y1
y20 = 0  # Valor inicial de y2
xf = 2  # Valor final de x
n = 10  # Número de pasos

# Calcular h (tamaño del paso)
h = (xf - x0) / n

# Inicializar vectores
x = np.zeros(n + 1)
y1 = np.zeros(n + 1)
y2 = np.zeros(n + 1)

# Asignar valores iniciales
x[0] = x0
y1[0] = y10
y2[0] = y20

# Método de Euler
for i in range(n):
    x[i + 1] = x[i] + h
    y1[i + 1] = y1[i] + h * f1(x[i], y1[i], y2[i])
    y2[i + 1] = y2[i] + h * f2(x[i], y1[i], y2[i])

# Guardar resultados de Euler en archivo
with open("resultados_euler.txt", "w") as archivo:
    archivo.write("x\ty1\ty2\n")
    for i in range(n + 1):
        archivo.write(f"{x[i]:.5f}\t{y1[i]:.5f}\t{y2[i]:.5f}\n")

# Reinicializar para el método de RK4
x[0] = x0
y1[0] = y10
y2[0] = y20

# Método de RK4
for i in range(n):
    k11 = f1(x[i], y1[i], y2[i])
    k12 = f2(x[i], y1[i], y2[i])

    k21 = f1(x[i] + h/2, y1[i] + h/2 * k11, y2[i] + h/2 * k12)
    k22 = f2(x[i] + h/2, y1[i] + h/2 * k11, y2[i] + h/2 * k12)

    k31 = f1(x[i] + h/2, y1[i] + h/2 * k21, y2[i] + h/2 * k22)
    k32 = f2(x[i] + h/2, y1[i] + h/2 * k21, y2[i] + h/2 * k22)

    k41 = f1(x[i] + h, y1[i] + h * k31, y2[i] + h * k32)
    k42 = f2(x[i] + h, y1[i] + h * k31, y2[i] + h * k32)

    # Actualizar y1 y y2
    y1[i + 1] = y1[i] + h / 6 * (k11 + 2 * k21 + 2 * k31 + k41)
    y2[i + 1] = y2[i] + h / 6 * (k12 + 2 * k22 + 2 * k32 + k42)

    # Actualizar x
    x[i + 1] = x[i] + h

# Guardar resultados de RK4 en archivo
with open("resultados_rk4.txt", "w") as archivo:
    archivo.write("x\ty1\ty2\n")
    for i in range(n + 1):
        archivo.write(f"{x[i]:.5f}\t{y1[i]:.5f}\t{y2[i]:.5f}\n")

print("Resultados guardados en 'resultados_euler.txt' y 'resultados_rk4.txt'.")
