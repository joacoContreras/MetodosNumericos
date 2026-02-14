# mínimos cuadrados, regresión con polinomio grado 1
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# INGRESO
xi = [1.1 , 2.1, 3.8, 5, 6.2, 8.5, 9.4, 11.7, 15.1, 15.6, 22.2]
yi = [0.0875, 0.1763, 0.2679, 0.3639, 0.4663, 0.5773, 0.7008, 0.839, 1, 1.1917, 1.4281]

# PROCEDIMIENTO
xi = np.array(xi, dtype=float)
yi = np.array(yi, dtype=float)
n  = len(xi)

# sumatorias y medias
xm  = np.mean(xi)
ym  = np.mean(yi)
sx  = np.sum(xi)
sy  = np.sum(yi)
sxy = np.sum(xi * yi)
sx2 = np.sum(xi**2)
sy2 = np.sum(yi**2)

# coeficientes a0 y a1 (para f(x) = a0 + a1*x)
a1 = (n * sxy - sx * sy) / (n * sx2 - sx**2)
a0 = ym - a1 * xm

# polinomio grado 1
x = sym.Symbol('x')
f = a0 + a1 * x

fx = sym.lambdify(x, f)
fi = fx(xi)

# coeficiente de correlación
numerador = n * sxy - sx * sy
raiz1 = np.sqrt(n * sx2 - sx**2)
raiz2 = np.sqrt(n * sy2 - sy**2)
r = numerador / (raiz1 * raiz2)

# coeficiente de determinacion
r2 = r**2
r2_porcentaje = np.around(r2 * 100, 2)

def calcular_error_pendiente(xi, yi, a0, a1, n):
    """
    Calcula el error estándar de la pendiente en la regresión lineal.
    
    Parámetros:
    xi: valores x de los datos
    yi: valores y de los datos
    a0: intercepto de la regresión
    a1: pendiente de la regresión
    n: número de puntos de datos
    
    Retorna:
    error_pendiente: error estándar de la pendiente
    """
    # Calcular los residuos
    y_pred = a0 + a1 * xi
    residuos = yi - y_pred
    
    # Calcular la suma de cuadrados de los residuos
    suma_residuos_cuadrados = np.sum(residuos**2)
    
    # Calcular la suma de cuadrados de las desviaciones de x
    x_mean = np.mean(xi)
    suma_x_desviaciones_cuadrados = np.sum((xi - x_mean)**2)
    
    # Calcular el error estándar de la pendiente
    error_pendiente = np.sqrt(suma_residuos_cuadrados / ((n-2) * suma_x_desviaciones_cuadrados))
    
    return error_pendiente

# Calcular el error de la pendiente
error_a1 = calcular_error_pendiente(xi, yi, a0, a1, n)

# SALIDA
print(' f(x) =', f)
print('Coeficiente de correlación   r  =', r)
print('Coeficiente de determinación r² =', r2)
print(f'{r2_porcentaje}% de los datos están descritos en el modelo lineal')
print(f'Error estándar de la pendiente: {error_a1:.6f}')

# gráfica
plt.plot(xi, yi, 'o', label='(xi, yi)')
plt.plot(xi, fi, color='orange', label=f'f(x) = {f}')

# líneas de error
for i in range(0, n, 1):
    y0 = np.min([yi[i], fi[i]])
    y1 = np.max([yi[i], fi[i]])
    plt.vlines(xi[i], y0, y1, color='red', linestyle='dotted')

plt.legend()
plt.xlabel('xi')
plt.title('Regresión por Mínimos Cuadrados')
plt.show()

def aproximar_raiz_fx(fx, a, b):
    """
    Aproxima la raíz de la función f(x) en el intervalo [a, b] usando el método de Brent.
    """
    try:
        raiz = brentq(fx, a, b)
        print(f"Raíz aproximada de f(x) en [{a}, {b}]: {raiz:.8f}")
        return raiz
    except Exception as e:
        print(f"No se pudo encontrar la raíz en [{a}, {b}]: {e}")
        return None

# Ejemplo de uso:
aproximar_raiz_fx(fx, 0, 3)
