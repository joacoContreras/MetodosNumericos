import math

# INGRESO
fx = lambda x: 3*math.pow(x,2) + 1  # Función a integrar

# Intervalo de integración
a = 0
b = 1
n = 1000  # SubIntervalos

# PROCEDIMIENTO
h = (b - a) / n  # Paso
suma = 0
for i in range(0, n-1):
    xi = (i*h)
    x_i1 = (i+1) * h
    suma += fx(xi) + fx(x_i1)

# Multiplicar por h/2
integral = (h / 2) * suma

# SALIDA
print('Integral: ', integral)
