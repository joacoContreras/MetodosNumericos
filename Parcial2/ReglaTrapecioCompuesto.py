import math

# INGRESO
fx = lambda x: 3*math.pow(x,2) + 1  # Función a integrar

# Intervalo de integración
a = 0
b = 1
n = 1000  # SubIntervalos

# PROCEDIMIENTO
h = (b - a) / n  # Paso
suma = fx(a) + fx(b)  # Suma inicial con los extremos

# Sumar los valores de los puntos intermedios multiplicados por 2
for i in range(1, n):
    suma += 2 * fx(a + i * h)

# Multiplicar por h/2
integral = (h / 2) * suma

# SALIDA
print('Integral: ', integral)
