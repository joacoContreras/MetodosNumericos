import math
import matplotlib.pyplot as plt

# xl: limite inferior del intervalo
# xu: Limite superior del intervalo
# tol: Tolerancia o error maximo permitido
# xr: Punto medio

# Definición de la función
def f(x):
    return math.log(x**2+1) - math.sin(x)

# Método de Regla Falsa
def regla_falsa(f, xl, xu, tol):
    if f(xl) * f(xu) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de regla falsa.")
        return None, []

    xr = xl
    iteracion = 0
    errores = []

    while True:
        xr_prev = xr
        xr = (xl*f(xu)-xu*f(xl))/(f(xu)-f(xl))
        iteracion += 1
        
        if xr_prev != 0:
            error = abs((xr - xr_prev) / xr)
        else:
            error = abs(xr - xr_prev)

        errores.append(error)
        
        if error < tol:
            break
        
        if f(xr) * f(xl) < 0:
            xu = xr
        else:
            xl = xr
            
        if iteracion == 10:
            print(f"La raíz en la iteracion 10 es: {xr}")
            print(f"error estimado en iteracion 10: {error}")
        

    return xr, errores

# Defino los valores iniciales
xl = 1.0
xu = 2.0
tol = 1e-8

# Ejecutar métodos y recopilar datos
raiz_regla_falsa, errores_regla_falsa = regla_falsa(f, xl, xu, tol)

# Número de iteraciones
iteraciones_regla_falsa = len(errores_regla_falsa)
print(f"\nMétodo de Regla Falsa:")
print(f"La raíz más pequeña encontrada es: {raiz_regla_falsa}")
print(f"Número total de iteraciones: {iteraciones_regla_falsa}")