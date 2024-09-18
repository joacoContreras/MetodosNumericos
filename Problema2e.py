# mínimos cuadrados, regresión con polinomio grado 1
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# INGRESO
xi = [1,1.2,1.5,1.75,2.0]
yi = [-0.148, -0.040, 0.181, 0.419, 0.700]

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

# SALIDA
print(' f(x) =', f)
print('Coeficiente de correlación   r  =', r)
print('Coeficiente de determinación r² =', r2)
print(f'{r2_porcentaje}% de los datos están descritos en el modelo lineal')

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
