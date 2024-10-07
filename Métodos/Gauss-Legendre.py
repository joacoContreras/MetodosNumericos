# Integración: Cuadratura de Gauss de dos puntos
# modelo con varios tramos entre [a,b]
import numpy as np
import matplotlib.pyplot as plt
import math
# cuadratura de Gauss de dos puntos
def integraCuadGauss2p(fx,a,b, vertabla=False):
    ''' funcion fx, intervalo[a,b]
        ver tabla=True para ver resultados parciales 
    '''
    x0 = -1/np.sqrt(3)
    x1 = -x0
    xa = (b+a)/2 + (b-a)/2*(x0)
    xb = (b+a)/2 + (b-a)/2*(x1)
    area = ((b-a)/2)*(fx(xa) + fx(xb))
    if vertabla==True:
        print('[xa,xb,f(xa),f(xb)]')
        print([xa,xb,fx(xa),fx(xb)])
    return(area)

# INGRESO
fx = lambda x: math.e**(x**2)

# intervalo de integración
a = 0.2
b = 1.2
tramos = 4

# PROCEDIMIENTO
muestras = tramos+1
xi = np.linspace(a,b,muestras)
area = 0
for i in range(0,muestras-1,1):
    deltaA = integraCuadGauss2p(fx,xi[i],xi[i+1],vertabla=True)
    area = area + deltaA
# SALIDA
print('Integral: ', area)