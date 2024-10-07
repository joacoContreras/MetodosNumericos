#Ajuste polinomial: Utiliza np.polyfit para calcular los coeficientes del polinomio que 
# mejor se ajusta a los datos.

# Generación del polinomio: np.poly1d convierte los coeficientes en un objeto polinomial 
# que se puede evaluar.

# Error cuadrático medio: Calcula la diferencia entre los valores reales (y) y 
# los valores ajustados (y_ajustado), promediando el cuadrado de estas diferencias.

# Gráfico: Muestra los puntos originales y la curva ajustada.

import numpy as np
import matplotlib.pyplot as plt

# Datos de ejemplo
x = [1.0, 1.2, 1.5, 1.75, 2.0]
y = [-0.148, -0.040, 0.181, 0.419, 0.700]

# Convertimos las listas a arrays de NumPy
x = np.array(x)
y = np.array(y)

# Grado del polinomio (puedes cambiar este valor para ajustar polinomios de diferentes grados)
grado = 4

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

# Graficamos los puntos originales y la curva ajustada
plt.scatter(x, y, color='blue', label='Datos originales')
plt.plot(x, y_ajustado, color='red', label=f'Polinomio de grado {grado}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Regresión Polinomial de grado {grado}')
plt.show()
