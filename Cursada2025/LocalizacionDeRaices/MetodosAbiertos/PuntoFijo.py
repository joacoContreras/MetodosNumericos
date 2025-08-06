import math

# Función g(x)
def g(x):
    return math.e**(-x)

# Función f(x)
def f(x):
    return math.e**(-x) - x

# Método de punto fijo
def punto_fijo(g, x0, tol=1e-5, max_iter=100):
    iteracion = 0
    error = float('inf')
    
    while error > tol and iteracion < max_iter:
        x1 = g(x0)
        error = abs(x1 - x0)
        iteracion += 1
        print(f"Iteración {iteracion}: x = {x1:.8f}, error = {error:.2e}")
        x0 = x1
    
    convergio = error < tol
    if convergio:
        return x1, iteracion, True
    else:
        return None, iteracion, False

# Parámetros
x0 = 1.0
tol = 1e-5
max_iter = 1000

# Ejecutar
raiz, iteraciones, convergio = punto_fijo(g, x0, tol, max_iter)

# Resultados
if convergio:
    print(f"\n✅ Converge a raíz: {raiz} en {iteraciones} iteraciones.")
else:
    print(f"\n❌ No converge después de {iteraciones} iteraciones.")
