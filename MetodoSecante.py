import math

# Definición de la función f(x) cuya raíz queremos encontrar
def f(x):
    return x**3 - x - 1

# Método de la Secante
def secante(f, x0, x1, tol, max_iter):
    iteracion = 0
    error = float('inf')
    convergio = False
    
    while error > tol and iteracion < max_iter:
        f_x0 = f(x0)
        f_x1 = f(x1)
        
        # Verificar que el denominador no sea muy pequeño para evitar divisiones por cero
        if abs(f_x1 - f_x0) < 1e-10:
            print("La diferencia en f(x) es muy pequeña; el método puede no converger.")
            break
        
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)  # Siguiente aproximación
        error = abs(x2 - x1)  # Error absoluto
        
        iteracion += 1
        print(f"Iteración {iteracion}: x = {x2}, error = {error}")
        
        if error < tol:
            convergio = True
            break
        
        # Actualizar valores para la siguiente iteración
        x0, x1 = x1, x2
    
    # Evaluar f(xi+1) para asegurar que está cerca de cero
    f_x2 = f(x2)
    if abs(f_x2) < tol:
        print(f"f(xi+1) = {f_x2}, está cerca de cero.")
    else:
        print(f"f(xi+1) = {f_x2}, puede no estar suficientemente cerca de cero.")
    
    return x2, iteracion, convergio

# Valores iniciales
x0 = 2.0  # Primer valor inicial
x1 = 1.5  # Segundo valor inicial
tol = 1e-5  # Tolerancia
max_iter = 100  # Máximo número de iteraciones

# Llamada al método de la secante
raiz, iteraciones, convergio = secante(f, x0, x1, tol, max_iter)

# Resultados
if convergio:
    print(f"\nEl método convergió a la raíz: {raiz} en {iteraciones} iteraciones.")
else:
    print(f"\nEl método no convergió después de {iteraciones} iteraciones.")
