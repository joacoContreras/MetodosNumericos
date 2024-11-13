import math

def f(x, y):
    """Define la ecuación diferencial"""
    return (x * math.exp(x**2)) / y

def print_resultados(x, y, archivo):
    """Imprime los resultados en el archivo"""
    archivo.write(f"x = {x:.5f}, y = {y:.5f}\n")

def main():
    # Definir los valores iniciales
    x0 = 0
    y0 = 1
    xfinal = 1
    n = 10

    # Calcular h (incremento)
    h = 0.1  # Se eligió este valor para obtener una buena precisión

    # Inicializar valores de x e y
    x_actual = x0
    y_actual = y0

    # Archivo para guardar los resultados
    try:
        with open("resultados-Problema32.txt", "w") as archivo:
            archivo.write("Solucion:\n")
            archivo.write(f"x0 = {x_actual}, y0 = {y_actual}\n")

            # Bucle para iterar sobre los pasos de la solución
            while x_actual < xfinal:
                # Método de Euler para el primer paso
                if x_actual == x0:
                    y_actual = y_actual + h * f(x_actual, y_actual)
                    x_actual = x_actual + h
                    print_resultados(x_actual, y_actual, archivo)
                else:
                    # Método de Runge-Kutta de segundo orden
                    x_anterior = x_actual - h / 2
                    y_anterior = y_actual - h / 2 * f(x_actual, y_actual)
                    y_actual = y_actual + h * ((2 * f(x_actual, y_actual)) - f(x_anterior, y_anterior))
                    x_actual = x_actual + h
                    print_resultados(x_actual, y_actual, archivo)

        print("Resultados guardados en 'resultados-Problema32.txt'.")

    except IOError:
        print("Error al abrir el archivo.")

if __name__ == "__main__":
    main()
