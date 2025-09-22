import math
import numpy as np
def newton_raphson(f, df, x0, h, tol=1e-6, max_iter=100):
    iteracion = 0
    error = float('inf')

    while error > tol and iteracion < max_iter:
        f_x0 = f(x0)
        df_x0_h = df(x0, h)
        
        if df_x0_h == 0:
            print("Derivada cero. No se puede continuar.")
            return None, iteracion, False
        
        x1 = x0 - f_x0 / df_x0_h
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
    return 2 * x + np.log(x) - np.sin(3 * x)

def df(x, h):
    return ((3 * f(x) - 4 * f(x - h)  + f(x - 2 * h)) / (2 * h))

# Llamada al método
raiz, iteraciones, convergio = newton_raphson(f, df, x0=0.5, h = 0.01)

# Resultado
if convergio:
    print(f"\n✅ Converge a raíz: {raiz} en {iteraciones} iteraciones.")
else:
    print(f"\n❌ No converge después de {iteraciones} iteraciones.")
