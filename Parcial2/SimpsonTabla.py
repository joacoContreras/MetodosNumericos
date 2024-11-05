def simpson_1_3(x, y, n):
    if n < 2 or n % 2 == 0:
        print("El número de intervalos debe ser impar y mayor que 1.")
        return -1  # Indica un error

    integral = y[0] + y[n]  # f(x0) + f(xn)

    # Sumar los términos de f(x) en las posiciones impares
    for i in range(1, n, 2):
        integral += 4 * y[i]  # 4 * f(xi) donde i es impar

    # Sumar los términos de f(x) en las posiciones pares
    for i in range(2, n - 1, 2):
        integral += 2 * y[i]  # 2 * f(xi) donde i es par

    integral *= (x[1] - x[0]) / 3  # Multiplica por el ancho del intervalo
    return integral

def main():
    # Valores de la tabla (x, f(x))
    x = [7, 9, 11, 13, 15, 17]  # Valores de x
    y = [18, 20, 26, 30, 35, 37]  # Valores de f(x)
    n = len(x) - 1  # Número de intervalos (n debe ser impar)

    resultado = simpson_1_3(x, y, n)
    if resultado != -1:
        print(f"El valor de la integral usando Simpson 1/3 es: {resultado:.6f}")

if __name__ == "__main__":
    main()

