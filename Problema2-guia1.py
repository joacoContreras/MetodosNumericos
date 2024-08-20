import math
import matplotlib.pyplot as plt

# xl: limite inferior del intervalo
# xu: Limite superior del intervalo
# tol: Tolerancia o error maximo permitido
# xr: Punto medio

# Definición de la función
def f(x):
    return x**10 -1

# Método de Bisección
def biseccion(f, xl, xu, tol):
    if f(xl) * f(xu) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de bisección.")
        return None, []

    xr = xl
    iteracion = 0
    errores = []

    while True:
        xr_prev = xr
        xr = (xl + xu) / 2
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
xl = 0.9
xu = 1.2
tol = 1e-5

# Ejecutar métodos y recopilar datos
raiz_biseccion, errores_biseccion = biseccion(f, xl, xu, tol)
raiz_regla_falsa, errores_regla_falsa = regla_falsa(f, xl, xu, tol)

# Número de iteraciones
iteraciones_biseccion = len(errores_biseccion)
iteraciones_regla_falsa = len(errores_regla_falsa)

print(f"\nMétodo de Bisección:")
print(f"La raíz más pequeña encontrada es: {raiz_biseccion}")
print(f"Número total de iteraciones: {iteraciones_biseccion}")

print(f"\nMétodo de Regla Falsa:")
print(f"La raíz más pequeña encontrada es: {raiz_regla_falsa}")
print(f"Número total de iteraciones: {iteraciones_regla_falsa}")

# Graficar resultados
plt.figure(figsize=(12, 6))

# Gráfico de Bisección
plt.subplot(1, 2, 1)
plt.plot(errores_biseccion, marker='o', linestyle='-', color='b')
plt.yscale('log')
plt.xlabel('Número de Iteraciones')
plt.ylabel('Error')
plt.title('Método de Bisección')
plt.grid(True)

# Gráfico de Regla Falsa
plt.subplot(1, 2, 2)
plt.plot(errores_regla_falsa, marker='o', linestyle='-', color='r')
plt.yscale('log')
plt.xlabel('Número de Iteraciones')
plt.ylabel('Error')
plt.title('Método de Regla Falsa')
plt.grid(True)

plt.tight_layout()
plt.show()
