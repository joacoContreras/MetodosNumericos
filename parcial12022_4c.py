import math
import matplotlib.pyplot as plt

# Definición de la función
def f(x):
    return 2.123*x**2 - 0.207*x - 0.9529

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

    return xr, errores

# Defino los valores iniciales
xl = 0.6
xu = 1.3
tol = 1e-8

# Ejecutar métodos y recopilar datos
raiz_regla_falsa, errores_regla_falsa = regla_falsa(f, xl, xu, tol)

# Número de iteraciones
iteraciones_regla_falsa = len(errores_regla_falsa)

# Obtener las primeras 2 y las últimas 3 iteraciones
primeros_errores = errores_regla_falsa[:2]
ultimos_errores = errores_regla_falsa[-3:]

print(f"\nMétodo de Regla Falsa:")
print(f"La raíz más pequeña encontrada es: {raiz_regla_falsa}")
print(f"Número total de iteraciones: {iteraciones_regla_falsa}")

# Mostrar los errores
print("\nErrores de las primeras 2 iteraciones:")
for i, error in enumerate(primeros_errores):
    print(f"Iteración {i+1}: Error = {error}")

print("\nErrores de las últimas 3 iteraciones:")
for i, error in enumerate(ultimos_errores):
    print(f"Iteración {iteraciones_regla_falsa-3+i+1}: Error = {error}")

# Error absoluto y porcentual exacto
raiz_exacta = 0.7272263664  # Deberías cambiar esto por la raíz exacta conocida o calculada
error_absoluto_exacto = abs(raiz_regla_falsa - raiz_exacta)
error_porcentual_exacto = (error_absoluto_exacto / abs(raiz_exacta)) * 100

print(f"\nError absoluto exacto: {error_absoluto_exacto}")
print(f"Error porcentual exacto: {error_porcentual_exacto}%")
