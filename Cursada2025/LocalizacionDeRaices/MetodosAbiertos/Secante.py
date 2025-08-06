def secante(f, x0, x1, tol=1e-5, max_iter=100):
    iteracion = 0
    error = float('inf')

    while error > tol and iteracion < max_iter:
        f_x0 = f(x0)
        f_x1 = f(x1)

        if f_x1 - f_x0 == 0:
            print("División por cero. No se puede continuar.")
            return None, iteracion, False
        
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        error = abs(x2 - x1)
        iteracion += 1

        print(f"Iteración {iteracion}: x = {x2:.8f}, error = {error:.2e}")
        
        # Actualizamos para la próxima iteración
        x0, x1 = x1, x2

    convergio = error < tol
    if convergio:
        return x2, iteracion, True
    else:
        return None, iteracion, False

import math

def f(x):
    return math.e**(-x) - x

# Llamamos al método con dos valores iniciales
raiz, iteraciones, convergio = secante(f, x0=0.5, x1=1.0)

if convergio:
    print(f"\n✅ Converge a raíz: {raiz} en {iteraciones} iteraciones.")
else:
    print(f"\n❌ No converge después de {iteraciones} iteraciones.")
