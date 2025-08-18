import math
import matplotlib.pyplot as plt
import numpy as np 
# Definición de la función
def f(x):
    return np.log((x**2) + 1) - np.sin(x)

# Método de Regla Falsa
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
        
        if iteracion == 1:
            print("1era Iteracion: c = " , c)
            
        if iteracion == 10:
            print("10ima Iteracion: c = " , c)
            
        if iteracion > 1:
            error = abs((c - c_prev) / c)
            errores.append(error)

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

    return c, errores

# Valores iniciales
a = 1.0
b = 2.0
tol = 1e-8

# Ejecutar métodos
raiz_regla_falsa, errores_regla_falsa = regla_falsa(f, a, b, tol)

# Imprimir resultados

print(f"\nMétodo de Regla Falsa:")
print(f"Raíz encontrada: {raiz_regla_falsa}")
print(f"Iteraciones: {len(errores_regla_falsa)}")

# Graficar errores
plt.figure(figsize=(12, 6))

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
