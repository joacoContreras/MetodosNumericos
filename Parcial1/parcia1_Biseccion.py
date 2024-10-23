import math
import matplotlib.pyplot as plt

# Definición de la función que se quiere encontrar la raíz
def f(x):
    return math.log(x**2 + 1) - math.sin(x)

# Método de Bisección
def biseccion(f, a, b, tol):
    if f(a) * f(b) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de bisección.")
        return None, [], 0, 0, 0

    c = a
    iteracion = 0
    errores = []

    while True:
        c_prev = c
        c = (a + b) / 2
        iteracion += 1
        
        if c_prev != 0:
            error = abs((c - c_prev) / c) * 100  # Error porcentual
        else:
            error = abs(c - c_prev) * 100

        errores.append(error)
        
        if error < tol:  # Verificar si el error porcentual está por debajo del 0.1%
            break
        
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    return c, errores, error, iteracion

# Defino los valores iniciales
a = 1.0  # Límite inferior
b = 2.0 # Límite superior
tol = 0.1  # Tolerancia del 0.1%

# Ejecutar el método de bisección y recopilar datos
raiz_biseccion, errores_biseccion, error_final, iteraciones_biseccion = biseccion(f, a, b, tol)

# Imprimir resultados
print(f"\nMétodo de Bisección:")
print(f"Intervalo empleado: [{a}, {b}]")
print(f"Raíz obtenida: {raiz_biseccion}")
print(f"Error porcentual final: {error_final}%")
print(f"Número total de iteraciones: {iteraciones_biseccion}")
