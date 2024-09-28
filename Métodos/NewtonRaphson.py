import math

# Definición de la función f(x) cuya raíz queremos encontrar
def f(x):
    return x**3 - x - 1

# Derivada de la función f(x)
def df(x):
    return 3*x**2 - 1

# Método de Newton-Raphson
def newton_raphson(f, df, x0, tol, max_iter):
    iteracion = 0
    error = float('inf')
    convergio = False
    
    while error > tol and iteracion < max_iter:
        f_x0 = f(x0)
        df_x0 = df(x0)
        
        # Verificar que la derivada no sea muy pequeña
        if abs(df_x0) < 1e-10:
            print("La derivada es muy pequeña; el método puede no converger.")
            break
        
        x1 = x0 - f_x0 / df_x0  # Siguiente aproximación
        error = abs(x1 - x0)  # Error absoluto
        
        iteracion += 1
        print(f"Iteración {iteracion}: x = {x1}, error = {error}")
        
        if error < tol:
            convergio = True
            break
        
        x0 = x1  # Actualizar el valor de x0 para la siguiente iteración
    
    # Evaluar f(xi+1) para asegurar que está cerca de cero
    f_x1 = f(x1)
    if abs(f_x1) < tol:
        print(f"f(xi+1) = {f_x1}, está cerca de cero.")
    else:
        print(f"f(xi+1) = {f_x1}, puede no estar suficientemente cerca de cero.")
    
    return x1, iteracion, convergio

# Valores iniciales
x0 = 2.0  # Valor inicial
tol = 1e-5  # Tolerancia
max_iter = 100  # Máximo número de iteraciones

# Llamada al método de Newton-Raphson
raiz, iteraciones, convergio = newton_raphson(f, df, x0, tol, max_iter)

# Resultados
if convergio:
    print(f"\nEl método convergió a la raíz: {raiz} en {iteraciones} iteraciones.")
else:
    print(f"\nEl método no convergió después de {iteraciones} iteraciones.")
