import math

# Definición de la función f(x) cuya raíz queremos encontrar
def f(x):
    return 3 * x + math.sin(x) - math.exp(x)  # Corregido math.exp(x)

# Derivada de la función f(x) utilizando diferencias finitas
def df(xi, h):
    return (f(xi + h) - f(xi - h)) / (2 * h)  # Diferencias finitas centradas

# Método de Newton-Raphson
def newton_raphson(f, df, xViejo, tol, h):
    iteracion = 0
    error = float('inf')  # Inicializar error como infinito para que el bucle inicie

    while error > tol and iteracion < 1000:
        iteracion += 1
        derivada = df(xViejo, h)
        
        if abs(derivada) < 1e-10:
            print("Derivada demasiado pequeña. Riesgo de división por cero.")
            return None, None
        
        xNuevo = xViejo - (f(xViejo)) / derivada
        error = abs(xNuevo - xViejo)
        xViejo = xNuevo

    print("f(xNuevo) =", f(xNuevo))

    if abs(f(xNuevo)) < tol:
        print("Raíz aproximada =", xNuevo)
        print("Error =", error)
        print("Número de iteraciones =", iteracion)
        
    return xNuevo, error

# Valores iniciales
xViejo = 0.5  # Valor inicial
tol = 1e-8  # Tolerancia
h = 0.01  # Paso para diferencias finitas

# Llamada al método de Newton-Raphson
x_raiz, error_final = newton_raphson(f, df, xViejo, tol, h)
