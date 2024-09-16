import math
import matplotlib.pyplot as plt

# Definición de la función que se quiere encontrar la raíz
def f(x):
    return 0.1167*x**3 - 1.35*x**2 + 4.23*x - 3

# Método de Bisección
def biseccion(f, xl, xu, tol):
    if f(xl) * f(xu) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de bisección.")
        return None, [], 0, 0, 0

    xr = xl
    iteracion = 0
    errores = []

    while True:
        xr_prev = xr
        xr = (xl + xu) / 2
        iteracion += 1
        
        if xr_prev != 0:
            error = abs((xr - xr_prev) / xr) * 100  # Error porcentual
        else:
            error = abs(xr - xr_prev) * 100

        errores.append(error)
        
        if error < tol:  # Verificar si el error porcentual está por debajo del 0.1%
            break
        
        if f(xr) * f(xl) < 0:
            xu = xr
        else:
            xl = xr

    return xr, errores, error, iteracion

# Defino los valores iniciales
xl = 2  # Límite inferior
xu = 5  # Límite superior
tol = 0.1  # Tolerancia del 0.1%

# Ejecutar el método de bisección y recopilar datos
raiz_biseccion, errores_biseccion, error_final, iteraciones_biseccion = biseccion(f, xl, xu, tol)

# Imprimir resultados
print(f"\nMétodo de Bisección:")
print(f"Intervalo empleado: [{xl}, {xu}]")
print(f"Raíz obtenida: {raiz_biseccion}")
print(f"Error porcentual final: {error_final}%")
print(f"Número total de iteraciones: {iteraciones_biseccion}")
