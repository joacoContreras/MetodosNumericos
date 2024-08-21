import math

# Definición de la función f(x) cuya raíz queremos encontrar
def f(x):
    return x**5 - 3*x**3 - 2*x**2 + 2

# Método de la Secante
def metodo_secante(f, x0, x1, tol):
    iteracion = 0
    error = float('inf')  # Inicializar error como infinito para que el bucle inicie
    
    while error > tol and iteracion < 1000:
        iteracion += 1
        
        # Calcular la siguiente aproximación usando la fórmula de la secante
        if f(x1) - f(x0) == 0:
            print("División por cero detectada en la fórmula de la secante.")
            return None, None
        
        xNuevo = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        
        # Calcular el error relativo
        error = abs(xNuevo - x1)
        
        # Actualizar valores para la próxima iteración
        x0 = x1
        x1 = xNuevo

    print("f(xNuevo) =", f(xNuevo))
    
    if abs(f(xNuevo)) < tol:
        print("Raíz aproximada =", xNuevo)
        print("Error =", error)
        print("Número de iteraciones =", iteracion)
        
    return xNuevo, error

# Valores iniciales
x0 = 0.0  # Primer valor inicial (debería estar cerca de la raíz esperada)
x1 = 1.0  # Segundo valor inicial (diferente de x0 y cerca de la raíz)
tol = 1e-5  # Tolerancia

# Llamada al método de la Secante
x_raiz, error_final = metodo_secante(f, x0, x1, tol)
