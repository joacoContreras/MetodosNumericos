import math

# Definir la función f(x, y)
def f(x, y):
    return -2*x*y

def main():
    # Definir los valores iniciales
    x0 = 0  # Valor inicial de x
    y0 = 1  # Valor inicial de y (puedes cambiar este valor según el problema)
    xfinal = 1  # Valor final de x
    n = 6  # Número de divisiones

    # Calcular h (incremento)
    h = (xfinal - x0) / (n - 1)

    # Inicializar valores de xi y yi
    xi = x0
    yi = y0

    # Archivo para guardar los resultados
    try:
        with open("resultados-Heun.txt", "w") as archivo:
            archivo.write("Solucion:\n")
            archivo.write(f"x0 = {xi}, y0 = {yi}\n")
            for i in range(0,n-1):  # Iteramos de 1 a n-1
                yt = yi + h * f(xi, yi)
                xi1 = xi + h
                yi1 = yi + h * ((f(xi,yi)+f(xi1, yt))/2) 
                archivo.write(f"x{i} = {xi1:.5f}, y{i} = {yi1:.5f}, yt{i} = {yt:.5f}\n")

                # Actualizar xi y yi para la siguiente iteración
                xi, yi = xi1, yi1
        
        print("Resultados guardados en 'resultados-Heun.txt'.")

    except IOError:
        print("Error al abrir el archivo.")

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
