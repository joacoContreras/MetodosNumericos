# Algoritmo Posicion Falsa para raices
# busca en intervalo [a,b]
# tolera = error

import numpy as np
import math

# Definir la función
fx = lambda x: math.log(x**2 + 1) - math.sin(x)

# INGRESO
a = 1.0
b = 2.0
tolera = 0.0001  # Tolerancia para el error

# PROCEDIMIENTO
tabla = []  # Guardar las iteraciones
errores = []  # Para almacenar los errores
tramo = abs(b-a)
fa = fx(a)
fb = fx(b)
c_prev = a  # Valor previo de c para calcular el error

while not(tramo <= tolera):
    # Calcular la aproximación c
    c = b - fb * (a - b) / (fa - fb)
    fc = fx(c)
    
    # Calcular el error absoluto entre c actual y c anterior
    error = abs(c - c_prev)
    errores.append(error)
    
    # Guardar los valores en la tabla
    unafila = [a, c, b, fa, fc, fb, tramo, error]
    cambio = np.sign(fa) * np.sign(fc)
    
    if cambio > 0:
        tramo = abs(c - a)
        a = c
        fa = fc
    else:
        tramo = abs(b - c)
        b = c
        fb = fc
    
    # Actualizar c_prev
    c_prev = c
    unafila[6] = tramo  # Actualizar el tramo
    tabla.append(unafila)

# Convertir la tabla en un array de NumPy
tabla = np.array(tabla)
ntabla = len(tabla)

# SALIDA
np.set_printoptions(precision=6)
print('Método de la Posición Falsa')
print('i', ['a', 'c', 'b'], ['f(a)', 'f(c)', 'f(b)'])
print('   Tramo', 'Error')

for i in range(0, ntabla, 1):
    print(f"{i} {tabla[i, 0:3]} {tabla[i, 3:6]}  Tramo: {tabla[i, 6]:.6f}  Error: {tabla[i, 7]:.6f}")

# Mostrar la última raíz y error final
print(f"\nRaíz aproximada: {c:.6f}  Error final: {error:.6f}")
