import matplotlib.pyplot as plt
import numpy as np

# Método del punto fijo

plt.style.use('classic')

def g(x):
    return x**5 - 3*x**3 - 2*x**2 + 2

def punto_fijo():
    xViejo = 0
    tol = 1e-12
    iteracion = 0
    
    while True:
        iteracion += 1
        xNuevo = g(xViejo)
        
        if abs((g(xViejo + 0.01) - g(xViejo)) / 0.01) > 1:
            print("No se cumple el criterio de convergencia")
            exit(1)
        
        error = abs(xNuevo - xViejo)
        if error < tol:
            break
        xViejo = xNuevo
        
    return xNuevo, error, iteracion

x, error, iteracion = punto_fijo()
print("X = " + str(x))
print("Error = " + str(error))
print("Iteracion = " + str(iteracion))

# Graficar la función g(x) y la recta y = x

# Crear un rango de valores de x para graficar
x_vals = np.linspace(-2, 2, 400)  # Ajusta el intervalo según sea necesario

# Evaluar la función g(x) y la recta y = x
g_vals = g(x_vals)
y_vals = x_vals

# Crear la gráfica
plt.figure(figsize=(8, 6))
plt.plot(x_vals, g_vals, label='g(x)')
plt.plot(x_vals, y_vals, label='y = x', linestyle='--', color='orange')
plt.scatter(x, x, color='red', label=f'Punto Fijo: x={x:.5f}')

# Añadir etiquetas y título
plt.xlabel('x')
plt.ylabel('y')
plt.title('Método del Punto Fijo')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()
