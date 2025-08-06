import numpy as np
import matplotlib.pyplot as plt
import math
def graficar_funcion(f, a, b, n=1000, titulo="Gráfico de la función", xlabel="x", ylabel="f(x)"):
    """
    Grafica una función matemática en el intervalo [a, b].

    Parámetros:
    - f: función a graficar (debe aceptar arreglos de numpy)
    - a: inicio del dominio
    - b: fin del dominio
    - n: cantidad de puntos evaluados (resolución)
    - titulo: título del gráfico
    - xlabel: etiqueta del eje x
    - ylabel: etiqueta del eje y
    """
    x = np.linspace(a, b, n)
    y = f(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="f(x)", color="blue")
    plt.title(titulo)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.show()

# 🧪 Ejemplo de uso:
if __name__ == "__main__":
    # Definimos una función (puede ser lambda o def)
    funcion = lambda x: math.e**(-x) - x

    # Llamamos a la función de graficar
    graficar_funcion(funcion, a=-1, b=2, titulo="f(x) = sin(x) * exp(-0.1x)")
