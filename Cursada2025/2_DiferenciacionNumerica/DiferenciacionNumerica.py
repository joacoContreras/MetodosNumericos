import math

# Definir la función f(x)
def f(x):
    return math.log(x) + 1/x  # Ejemplo de función, puedes cambiarla

def main():
    # Definir los valores de x0, xfinal y n directamente en el código
    x0 = 1.5  # Valor inicial
    xfinal = 2.5  # Valor final
    n = 3  # Número de divisiones

    # Calcular h (incremento)
    h = (xfinal - x0) / (n - 1)

    # Archivo para guardar los resultados
    try:
        with open("resultados.txt", "w") as archivo:
            # Esquema hacia adelante
            archivo.write("Esquema hacia adelante:\n")
            for i in range(n - 1):
                xi = x0 + i * h
                derivada_adelante = (f(xi + h) - f(xi)) / h
                archivo.write(f"f'({xi}) = {derivada_adelante}\n")

            # Esquema hacia atrás
            archivo.write("\nEsquema hacia atras:\n")
            for i in range(1, n):
                xi = x0 + i * h
                derivada_atras = (f(xi) - f(xi - h)) / h
                archivo.write(f"f'({xi}) = {derivada_atras}\n")

            # Esquema centrado
            archivo.write("\nEsquema centrado:\n")
            for i in range(1, n - 1):
                xi = x0 + i * h
                derivada_centrada = (f(xi + h) - f(xi - h)) / (2 * h)
                archivo.write(f"f'({xi}) = {derivada_centrada}\n")
        
        print("Resultados guardados en 'resultados.txt'.")

    except IOError:
        print("Error al abrir el archivo.")

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
