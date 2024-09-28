import math

# Definición de la función g(x) para el método de punto fijo
def g(x):
    return math.log(10 / x)

# Definición de la función f(x) cuya raíz queremos encontrar
def f(x):
    return x**3 - x - 1

# Método de Punto Fijo
def punto_fijo(g, x0, tol, max_iter):
    iteracion = 0
    error = float('inf')
    convergio = False
    
    while error > tol and iteracion < max_iter:
        x1 = g(x0)  # Siguiente aproximación
        error = abs(x1 - x0)  # Error absoluto
        
        iteracion += 1
        print(f"Iteración {iteracion}: x = {x1}, error = {error}")
        
        if error < tol:
            convergio = True
            break
        
        x0 = x1  # Actualizar el valor de x0 para la siguiente iteración
    
    return x1, iteracion, convergio

# Valores iniciales
x0 = 2.0  # Valor inicial
tol = 1e-5  # Tolerancia
max_iter = 100  # Máximo número de iteraciones

# Llamada al método de punto fijo
raiz, iteraciones, convergio = punto_fijo(g, x0, tol, max_iter)

# Resultados
if convergio:
    print(f"\nEl método convergió a la raíz: {raiz} en {iteraciones} iteraciones.")
else:
    print(f"\nEl método no convergió después de {iteraciones} iteraciones.")
