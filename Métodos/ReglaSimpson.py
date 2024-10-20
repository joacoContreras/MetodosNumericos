import math

# INGRESO:
fx = lambda x: x**2 + 1  # Función a integrar

# Intervalo de integración
a = 0
b = 3
n = 2   # El número de subintervalos debe ser par

# Validar que la cantidad de tramos sea par
if n % 2 == 1:
    raise ValueError('El número de subintervalos debe ser par.')

# PROCEDIMIENTO
# Regla de Simpson 1/3
h = (b - a) / n
suma = fx(a) + fx(b)

# Sumar los valores de los puntos intermedios con el coeficiente adecuado
for i in range(1, n):
    x = a + i * h
    if i % 2 == 0:
        suma += 2 * fx(x)  # Coeficiente 2 en los puntos pares
    else:
        suma += 4 * fx(x)  # Coeficiente 4 en los puntos impares

# Multiplicar por h/3
integral = (h / 3) * suma

# SALIDA
print('Integral: ', integral)
