import numpy as np 
import matplotlib.pyplot as plt

def graficar_funcion(f, a, b, n=1000, titulo="Gráfico de la función", xlabel="x", ylabel="f(x)"):
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

if __name__ == "__main__":
    # Usamos funciones de numpy en lugar de math
    funcion = lambda x: np.log((x ** 2) + 1) - np.sin(x)

    graficar_funcion(funcion, a=1, b=2, titulo="f(x) = log(x² + 1) - sin(x)")
