import math

def newton_raphson(f, df, x0, tol=1e-5, max_iter=100):
    iteracion = 0
    error = float('inf')

    while error > tol and iteracion < max_iter:
        f_x0 = f(x0)
        df_x0 = df(x0)
        
        if df_x0 == 0:
            print("Derivada cero. No se puede continuar.")
            return None, iteracion, False
        
        x1 = x0 - f_x0 / df_x0
        error = abs(x1 - x0)
        iteracion += 1
        print(f"Iteración {iteracion}: x = {x1:.8f}, error = {error:.2e}")
        x0 = x1
    
    convergio = error < tol
    if convergio:
        return x1, iteracion, True
    else:
        return None, iteracion, False

# Función y derivada
def f(x):
    return math.e**(-x) - x

def df(x):
    return -math.e**(-x) - 1

# Llamada al método
raiz, iteraciones, convergio = newton_raphson(f, df, x0=1.0)

# Resultado
if convergio:
    print(f"\n✅ Converge a raíz: {raiz} en {iteraciones} iteraciones.")
else:
    print(f"\n❌ No converge después de {iteraciones} iteraciones.")
