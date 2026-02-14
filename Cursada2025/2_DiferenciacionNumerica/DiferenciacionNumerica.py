import math

# Definir la función f(x)
def f(x):
    return math.exp(math.sqrt(1+x)) * math.log(1 + 2 * x**2)
def main():
    # Definir los valores de x0, xfinal y n directamente en el código
    x0 = 0  # Valor inicial
    xfinal = 1  # Valor final
    n = 4  # Número de divisiones

    # Calcular h (incremento)
    h = (xfinal - x0) / (n - 1)
    # h = 0.1  # Tamaño de paso fijo
    
    # Archivo para guardar los resultados
    try:
        with open("resultados.txt", "w") as archivo:
            # Esquema hacia adelante
            archivo.write("Esquema hacia adelante:\n")
            for i in range(int((xfinal - x0) / h)):
                xi = x0 + i * h
                derivada_adelante = (f(xi + h) - f(xi)) / h
                archivo.write(f"f'({xi}) = {derivada_adelante}\n")

            # Esquema hacia atrás
            archivo.write("\nEsquema hacia atras:\n")
            for i in range(1, int((xfinal - x0) / h) + 1):
                xi = x0 + i * h
                derivada_atras = (f(xi) - f(xi - h)) / h
                archivo.write(f"f'({xi}) = {derivada_atras}\n")

            # Esquema centrado
            archivo.write("\nEsquema centrado:\n")
            for i in range(1, int((xfinal - x0) / h)):
                xi = x0 + i * h
                derivada_centrada = (f(xi + h) - f(xi - h)) / (2 * h)
                archivo.write(f"f'({xi}) = {derivada_centrada}\n")
        
        print("Resultados guardados en 'resultados.txt'.")

    except IOError:
        print("Error al abrir el archivo.")

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
