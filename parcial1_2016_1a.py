import math
import numpy as np
import matplotlib.pyplot as plt

# Definición de la función f(x) cuya raíz queremos encontrar
def f(x):
    return 2 * x - math.sin(3*x) + math.log(x)  # Corregido math.exp(x)

# Derivada de la función f(x) utilizando diferencias finitas
def df(xi, h):
    return (3 * f(xi) - 4 * f(xi - h) + f(xi - 2*h)) / (2 * h)  # Diferencias finitas centradas

# Método de Newton-Raphson
def newton_raphson(f, df, xViejo, tol, h):
    iteracion = 0
    error = float('inf')  # Inicializar error como infinito para que el bucle inicie

    while error > tol and iteracion < 1000:
        iteracion += 1
        derivada = df(xViejo, h)
        
        if abs(derivada) < 1e-10:
            print("Derivada demasiado pequeña. Riesgo de división por cero.")
            return None, None
        
        xNuevo = xViejo - (f(xViejo)) / derivada
        error = abs(xNuevo - xViejo)
        xViejo = xNuevo

    print("f(xNuevo) =", f(xNuevo))

    if abs(f(xNuevo)) < tol:
        print("Raíz aproximada =", xNuevo)
        print("Error =", error)
        print("Número de iteraciones =", iteracion)
        
    return xNuevo, error

# Valores iniciales
xViejo = 0.5  # Valor inicial
tol = 1e-6  # Tolerancia
h = 0.01  # Paso para diferencias finitas

# Llamada al método de Newton-Raphson
x_raiz, error_final = newton_raphson(f, df, xViejo, tol, h)

# Rango de valores para x
x = np.linspace(0.1, 10, 400)  # Genera 400 puntos entre -10 y 10

# Evaluar la función en el rango de x
y = [f(val) for val in x]

# Graficar la función
plt.plot(x, y, label="f(x) = 2x - sin(3x) + log(x)", color='blue')

# Dibujar el eje X (y=0)
plt.axhline(0, color='black',linewidth=0.5)

# Añadir etiquetas y título
plt.title("Gráfica de la función f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")

# Añadir una cuadrícula
plt.grid(True)

# Mostrar la leyenda
plt.legend()

# Mostrar la gráfica
plt.show()
