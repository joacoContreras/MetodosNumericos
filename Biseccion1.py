
# xl: limite inferior del intervalo
# xu: Limite superior del intervalo
# tol: Tolerancia o error maximo permitido
# xr: Punto medio

def f(x):
    # Define aquí tu función. Por ejemplo:
    return -2+7*x-5*x*x+6*x*x*x

def biseccion(f, xl, xu, tol):
    # Verifica que hay un cambio de signo en el intervalo
    if f(xl) * f(xu) > 0:
        print("La función no cambia de signo en el intervalo dado.")
        return None, 0  # Devuelve 0 iteraciones si no hay cambio de signo

    # Inicializa variables
    xr = xl
    iteracion = 0

    while True:
        xr_prev = xr
        # Calcular el punto medio del intervalo actual
        xr = (xl + xu) / 2
        iteracion += 1
        
        # Calcula el error relativo
        error = abs((xr - xr_prev) / xr)
        
        # Imprime la iteración actual y los valores
        # print(f"Iteración {iteracion}: xl = {xl}, xu = {xu}, xr = {xr}, error = {error}")
        
        # Verifica la condición de convergencia
        if error < tol:
            break
        
        # Actualiza los límites del intervalo
        if f(xr) * f(xl) < 0:
            xu = xr
        else:
            xl = xr

    return xr, iteracion  # Devuelve la raíz y el número de iteraciones

# Valores iniciales
xl = 0.33
xu = 0.334
tol = 1e-4

# Llamada a la función de bisección
raiz, total_iteraciones = biseccion(f, xl, xu, tol)

# Imprime el resultado final
if raiz is not None:
    print(f"La raíz más pequeña encontrada es: {raiz}")
    print(f"Número total de iteraciones: {total_iteraciones}")
