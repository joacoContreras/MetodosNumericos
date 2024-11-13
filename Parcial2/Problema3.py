import math

# Definir la función f(x, y)
def f(x, y):
    return (x * math.exp(x**2)) / y

def main():
    # Definir los valores iniciales
    x0 = 0  # Valor inicial de x
    y0 = 1  # Valor inicial de y (puedes cambiar este valor según el problema)
    xfinal = 1  # Valor final de x
    n = 10 # Número de divisiones

    # Calcular h (incremento)
    h = 0.1

    # Inicializar valores de xi y yi
    xi = x0
    yi = y0

    # Archivo para guardar los resultados
    
    try:
        with open("resultados-Problema3.txt", "w") as archivo:
            archivo.write("Solucion:\n")
            archivo.write(f"x0 = {xi}, y0 = {yi}\n")
            for i in range(1,n+1):
                if i == 1:
                    yi1 = yi + h * f(xi, yi)  # Método de Euler
                    xi1 = xi + h
                    # Actualizar xi y yi para la siguiente iteración
                    xi, yi = xi1, yi1
                    archivo.write(f"x{i} = {xi1:.5f}, y{i} = {yi1:.5f}\n")
                else:
                    xim1 = xi - h/2
                    yim1 = yi - h/2 * f(xi, yi)
                    yi1 = yi + h * ((2*f(xi, yi)) - f(xim1, yim1))
                    xi1 = xi + h
                    # Actualizar xi y yi para la siguiente iteración
                    xi, yi = xi1, yi1
                    archivo.write(f"x{i} = {xi1:.5f}, y{i} = {yi1:.5f}\n")

        print("Resultados guardados en 'resultados-Problema3.txt'.")

    except IOError:
        print("Error al abrir el archivo.")

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
