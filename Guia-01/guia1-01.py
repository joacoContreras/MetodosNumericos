import matplotlib.pyplot as plt
import numpy as np

def f(x):
    # Define aquí tu función. Por ejemplo:
    return -2 + 7*x - 5*x*x + 6*x*x*x

def biseccion(f, xl, xu, tol):
    # Verifica que hay un cambio de signo en el intervalo
    if f(xl) * f(xu) > 0:
        print("La función no cambia de signo en el intervalo dado.")
        return None, 0  # Devuelve 0 iteraciones si no hay cambio de signo

    # Inicializa variables
    xr = xl
    iteracion = 0

    while True:
        xr_prev = xr
        # Calcular el punto medio del intervalo actual
        xr = (xl + xu) / 2
        iteracion += 1
        
        # Calcula el error relativo
        error = abs((xr - xr_prev) / xr)
        
        # Verifica la condición de convergencia
        if error < tol:
            break
        
        # Actualiza los límites del intervalo
        if f(xr) * f(xl) < 0:
            xu = xr
        else:
            xl = xr

    return xr, iteracion  # Devuelve la raíz y el número de iteraciones

def graficar_funcion(f, xl, xu):
    x = np.linspace(xl, xu, 400)  # Genera 400 puntos entre xl y xu
    y = f(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.5)  # Línea horizontal en y=0
    plt.axvline(0, color='black', linewidth=0.5)  # Línea vertical en x=0
    plt.title('Gráfico de la función f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    while True:
        # Solicitar intervalos y tolerancia al usuario
        try:
            xl = float(input("Ingrese el límite inferior del intervalo (xl): "))
            xu = float(input("Ingrese el límite superior del intervalo (xu): "))
            tol = float(input("Ingrese la tolerancia (tol): "))
        except ValueError:
            print("Por favor, ingrese valores numéricos válidos.")
            continue

        # Llamada a la función de bisección
        raiz, total_iteraciones = biseccion(f, xl, xu, tol)

        # Imprime el resultado final
        if raiz is not None:
            print(f"\nLa raíz más pequeña encontrada es: {raiz}")
            print(f"Número total de iteraciones: {total_iteraciones}\n")
            graficar_funcion(f, xl, xu)
        
        # Preguntar al usuario si quiere realizar otra búsqueda o salir
        salir = input("¿Desea realizar otra búsqueda? (s/n): ").strip().lower()
        if salir == 'n':
            print("Saliendo del programa...")
            break

# Ejecutar el programa
main()
