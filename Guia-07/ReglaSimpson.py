# Integración: Regla Simpson 1/3
import numpy as np
import matplotlib.pyplot as plt
import math
# INGRESO:
fx = lambda x: (1+(math.cos(x))**2)**0.5

# intervalo de integración
a = 0
b = math.pi/4
tramos = 10

# Validar cantidad de tramos pares
esimpar = tramos%2
while (esimpar == 1):
    print('tramos: ',tramos)
    tramos = int(input('tramos debe ser par: '))
    esimpar = tramos%2

# PROCEDIMIENTO
# Regla de Simpson 1/3
h = (b-a)/tramos
xi = a
area = 0
for i in range(0,tramos,2):
    deltaA = (h/3)*(fx(xi)+4*fx(xi+h)+fx(xi+2*h))
    area = area + deltaA
    xi = xi + 2*h

# SALIDA
print('tramos:', tramos)
print('Integral: ', area)

# GRAFICA
# fx suave aumentando muestras
muestrasfxSuave = 4*tramos + 1
xk = np.linspace(a,b,muestrasfxSuave)
fk = fx(xk)
#plt.fill_between(xj,0,fj, color='g')
plt.plot(xk,fk, label='f(x)')

# fx muestras por tramo
muestras = tramos + 1
xi = np.linspace(a,b,muestras)
fi = fx(xi)
fi0 = np.zeros(muestras) # linea base
plt.plot(xi,fi,'o', label='f(xi)')
for i in range(0,muestras,1):
    plt.vlines(xi,fi0,fi,linestyle='dotted')
plt.axhline(0)
plt.xlabel('x')
plt.ylabel('f')
plt.legend()
plt.show()