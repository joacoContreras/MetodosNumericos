import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = [0.2, 0.6, 1.3, 1.4, 1.8, 2.0]
y = [-0.94, -0.26, 2.35, 2.94, 5.45, 7.20]

# Convertimos las listas a arrays de NumPy
x = np.array(x)
y = np.array(y)

# Grado del polinomio (puedes cambiar este valor para ajustar polinomios de diferentes grados)
grado = 2

# Ajuste polinomial con NumPy
coeficientes = np.polyfit(x, y, grado)

# Creamos el polinomio a partir de los coeficientes
polinomio = np.poly1d(coeficientes)

# Mostramos la ecuación del polinomio
print(f'El polinomio de grado {grado} ajustado es:')
print(polinomio)

# Generamos los valores ajustados con el polinomio
y_ajustado = polinomio(x)

# Calculamos el error cuadrático medio
error_cuadratico_medio = np.mean((y - y_ajustado)**2)
print(f'Error cuadrático medio: {error_cuadratico_medio:.4f}')

# Cálculo del coeficiente de correlación (r)
n = len(x)
sxy = np.sum(x * y_ajustado)
sx = np.sum(x)
sy = np.sum(y_ajustado)
sx2 = np.sum(x**2)
sy2 = np.sum(y_ajustado**2)

numerador = n * sxy - sx * sy
denominador = np.sqrt((n * sx2 - sx**2) * (n * sy2 - sy**2))

if denominador != 0:
    r = numerador / denominador
    print(f'Coeficiente de correlación (r): {r:.4f}')
else:
    print('El denominador es cero, no se puede calcular el coeficiente de correlación.')

# Graficamos los puntos originales y la curva ajustada
plt.scatter(x, y, color='blue', label='Datos originales')
plt.plot(x, y_ajustado, color='red', label=f'Polinomio de grado {grado}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Regresión Polinomial de grado {grado}')
plt.show()
