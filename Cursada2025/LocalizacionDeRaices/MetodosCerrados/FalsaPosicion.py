import math
import matplotlib.pyplot as plt

# Definición de la función
def f(x):
    return x**10 - 1

# Método de Bisección
def biseccion(f, a, b, tol):
    if f(a) * f(b) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de bisección.")
        return None, []

    iteracion = 0
    errores = []
    c = a
    error = float('inf')

    while error > tol:
        c_prev = c
        c = (a + b) / 2
        iteracion += 1

        if iteracion > 1:
            error = abs((c - c_prev) / c)
            errores.append(error)

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    return c, errores

# Método de Regla Falsa (False Position)
def regla_falsa(f, a, b, tol):
    if f(a) * f(b) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de regla falsa.")
        return None, []

    iteracion = 0
    errores = []
    c = a
    error = float('inf')

    while error > tol:
        c_prev = c
        fa = f(a)
        fb = f(b)
        c = (a * fb - b * fa) / (fb - fa)
        iteracion += 1

        if iteracion > 1:
            error = abs((c - c_prev) / c)
            errores.append(error)

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    return c, errores

# Valores iniciales
a = 0.9
b = 1.2
tol = 1e-5

# Ejecutar métodos
raiz_biseccion, errores_biseccion = biseccion(f, a, b, tol)
raiz_regla_falsa, errores_regla_falsa = regla_falsa(f, a, b, tol)

# Imprimir resultados
print(f"\nMétodo de Bisección:")
print(f"Raíz encontrada: {raiz_biseccion}")
print(f"Iteraciones: {len(errores_biseccion)}")

print(f"\nMétodo de Regla Falsa:")
print(f"Raíz encontrada: {raiz_regla_falsa}")
print(f"Iteraciones: {len(errores_regla_falsa)}")

# Graficar errores
plt.figure(figsize=(12, 6))

# Bisección
plt.subplot(1, 2, 1)
plt.plot(errores_biseccion, marker='o', linestyle='-', color='blue')
plt.yscale('log')
plt.xlabel('Iteración')
plt.ylabel('Error Relativo')
plt.title('Error por Iteración - Bisección')
plt.grid(True)

# Regla falsa
plt.subplot(1, 2, 2)
plt.plot(errores_regla_falsa, marker='o', linestyle='-', color='red')
plt.yscale('log')
plt.xlabel('Iteración')
plt.ylabel('Error Relativo')
plt.title('Error por Iteración - Regla Falsa')
plt.grid(True)

plt.tight_layout()
plt.show()
