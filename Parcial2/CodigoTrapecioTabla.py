def trapezoidal(y, h, n):
    """Metodo del trapecio para calcular la integral utilizando valores de una tabla."""
    suma = y[0] + y[n]  # Suma inicial con los extremos

    # Sumar el doble de los términos internos
    for i in range(1, n):
        suma += 2 * y[i]

    # Multiplicar por h/2 según la fórmula del trapecio
    return (suma * h) / 2

def main():
    # Límite inferior y superior de integración
    a = float(input("Ingrese el limite inferior a: "))
    b = float(input("Ingrese el limite superior b: "))
    
    # Número de subintervalos
    n = 7  # Puedes cambiar este valor si lo deseas
    h = (b - a) / (n - 1)  # Tamaño del subintervalo
    
    # Crear la lista para almacenar los valores de la función
    y = []

    print("Ingrese los valores de la funcion en los puntos:")
    for i in range(n):
        x_i = a + i * h
        valor_y = float(input(f"f({x_i}) = "))
        y.append(valor_y)

    # Cálculo de la integral aproximada
    resultado = trapezoidal(y, h, n - 1)

    # Muestra el resultado
    print(f"El valor aproximado de la integral es: {resultado}")

if __name__ == "__main__":
    main()
