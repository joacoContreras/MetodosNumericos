import math

# Definición de la función f(x) cuya raíz queremos encontrar
def f(x):
    return math.log(x**2+1) - math.sin(x)

# Derivada de la función f(x)
def df(x):
    return (2*x)/(x**2+1) - math.cos(x)

# Método de Newton-Raphson
def newton_raphson(f, df, xViejo, tol):
    iteracion = 0
    error = float('inf')  # Inicializar error como infinito para que el bucle inicie
    
    while error > tol and iteracion < 1000:
        iteracion += 1
        xNuevo = xViejo - (f(xViejo)) / (df(xViejo))
        
        if abs(df(xNuevo)) < 1e-10:
            print("Derivada demasiado pequeña. Riesgo de división por cero.")
            return None, None
        
        error = abs(xNuevo - xViejo)
        xViejo = xNuevo

    print("f(xNuevo) =", f(xNuevo))
    
    if abs(f(xNuevo)) < tol:
        print("Raíz aproximada =", xNuevo)
        print("Error =", error)
        print("Número de iteraciones =", iteracion)
        
    return xNuevo, error

# Valores iniciales
xViejo = 0.1  # Valor inicial (debería estar más cerca de la raíz esperada)
tol = 1e-5  # Tolerancia

# Llamada al método de Newton-Raphson
x_raiz, error_final = newton_raphson(f, df, xViejo, tol)
