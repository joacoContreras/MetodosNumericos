import math

# Definir la función f(x)
def f(x):
    return math.pow(math.e, math.sqrt(1 + x)) * math.log(1 + 2 * math.pow(x, 2))

def main():
    # Definir los valores de x0, xfinal y n directamente en el código
    x0 = 0  # Valor inicial
    xfinal = 1  # Valor final
    n = 7  # Número de divisiones

    # Calcular h (incremento)
    h = (xfinal - x0) / (n - 1)

    # Archivo para guardar los resultados
    try:
        with open("resultados.txt", "w") as archivo:
            # Esquema hacia adelante
            archivo.write("Esquema hacia adelante:\n")
            for i in range(0, n - 1):
                xi = x0 + i * h
                derivada_adelante = (f(xi + h) - f(xi)) / h
                archivo.write(f"f'({xi}) = {derivada_adelante}\n")

            # Esquema hacia atrás (corregido)
            archivo.write("\nEsquema hacia atrás:\n")
            for i in range(1, n):
                xi = x0 + i * h
                derivada_atras = (f(xi) - f(xi - h)) / h  # Corrección
                archivo.write(f"f'({xi}) = {derivada_atras}\n")

            # Esquema centrado (corregido)
            archivo.write("\nEsquema centrado:\n")
            for i in range(1, n - 1):
                xi = x0 + i * h
                derivada_centrada = (f(xi + h) - f(xi - h)) / (2 * h)  # Corrección
                archivo.write(f"f'({xi}) = {derivada_centrada}\n")
        
        print("Resultados guardados en 'resultados.txt'.")

    except IOError:
        print("Error al abrir el archivo.")

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
