import numpy as np 
def f(x):
    # Define aquí tu función:
    return np.log(x**2 + 1) - np.sin(x)

def biseccion(f, a, b, tol):
    # Verifica que hay un cambio de signo en el intervalo
    if f(a) * f(b) > 0:
        print("La función no cambia de signo en el intervalo dado.")
        return None, 0, None

    iteracion = 0
    c = a
    error = float('inf')  # Inicializa con un valor muy grande

    while error > tol:
        c_prev = c
        c = (a + b) / 2
        iteracion += 1

        # Evita división por cero en error
        if c != 0:
            error = abs((c - c_prev))
        else:
            error = abs(c - c_prev)

        # Imprime si estás en la iteración 10 (opcional)
        if iteracion == 10:
            print(f"[DEBUG] Iteración 10: raíz ≈ {c}, error estimado ≈ {error:e}")

        # Verifica si la raíz exacta fue encontrada (opcional)
        if f(c) == 0:
            break

        # Actualiza los límites del intervalo
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return c, iteracion, error

def calcular_errores(raiz_aprox, raiz_exacta):
    error_absoluto = abs(raiz_exacta - raiz_aprox)
    error_porcentual = abs(error_absoluto / raiz_exacta) * 100
    print(f"Error absoluto exacto: {error_absoluto:e}")
    print(f"Error porcentual exacto: {error_porcentual:.8f}%")

# Valores iniciales
a = 1
b = 2
tol = 1e-4

# Llamada a la función de bisección
raiz, total_iteraciones, error = biseccion(f, a, b, tol)

# Imprime el resultado final
if raiz is not None:
    print(f"\nLa raíz más pequeña encontrada es: {raiz}")
    print(f"Número total de iteraciones: {total_iteraciones}")
    print(f"Error estimado: {error:e}")
    # Calcula y muestra los errores exactos
    raiz_exacta = 26.961 / 35.444
    calcular_errores(raiz, raiz_exacta)
