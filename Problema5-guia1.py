import math

# Definir la función de eficiencia
def efficiency(x):
    gamma = 5.0 / 3.0
    ln_x = math.log(x)
    return (1.0 / x) + (3.0 * (1.0 - 1.0 / x)) / (2.0 * ln_x) - 0.3

# Método de bisección para encontrar la raíz
def bisection(a, b, tol):
    iterations = 0

    while (b - a) / 2.0 > tol:
        c = (a + b) / 2.0
        if efficiency(c) == 0.0:
            return c, iterations
        elif efficiency(a) * efficiency(c) < 0:
            b = c
        else:
            a = c
        iterations += 1

    return (a + b) / 2.0, iterations

# Parámetros iniciales
a = 1.1  # Límite inferior del intervalo para bisección
b = 40.0 # Límite superior del intervalo para bisección
tol = 1e-6  # Tolerancia para la bisección

# Encontrar la raíz usando el método de bisección
root, iterations = bisection(a, b, tol)

print(f"La relación T2/T1 para una eficiencia del 30% es aproximadamente: {root:.6f}")
print(f"Número de iteraciones: {iterations}")



# Constantes del problema
v = 5/3  # gamma
n_deseada = 0.3  # Eficiencia deseada del 30%

# Definición de la función que se quiere encontrar la raíz
def f(T_ratio):
    T1_T2 = T_ratio
    numerator = math.log(T1_T2) - (1 - 1 / T1_T2)
    denominator = math.log(T1_T2) + (1 - 1 / T1_T2) / (v - 1)
    return numerator / denominator - n_deseada

# Método de Bisección
def biseccion(f, xl, xu, tol):
    if f(xl) * f(xu) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de bisección.")
        print(f"f({xl}) = {f(xl)}, f({xu}) = {f(xu)}")
        return None, []

    xr = xl
    iteracion = 0
    errores = []

    while True:
        xr_prev = xr
        xr = (xl + xu) / 2
        iteracion += 1
        
        if xr_prev != 0:
            error = abs((xr - xr_prev) / xr)
        else:
            error = abs(xr - xr_prev)

        errores.append(error)
        
        if error < tol:
            break
        
        if f(xr) * f(xl) < 0:
            xu = xr
        else:
            xl = xr

    return xr, errores

# Método de Regla Falsa
def regla_falsa(f, xl, xu, tol):
    if f(xl) * f(xu) > 0:
        print("La función no cambia de signo en el intervalo dado para el método de regla falsa.")
        print(f"f({xl}) = {f(xl)}, f({xu}) = {f(xu)}")
        return None, []

    xr = xl
    iteracion = 0
    errores = []

    while True:
        xr_prev = xr
        f_xl = f(xl)
        f_xu = f(xu)
        xr = (xl * f_xu - xu * f_xl) / (f_xu - f_xl)
        iteracion += 1
        
        if xr_prev != 0:
            error = abs((xr - xr_prev) / xr)
        else:
            error = abs(xr - xr_prev)

        errores.append(error)
        
        if error < tol:
            break
        
        if f(xr) * f(xl) < 0:
            xu = xr
        else:
            xl = xr

    return xr, errores

# Defino los valores iniciales para los métodos
xl = 0.1  # Límite inferior de T2/T1 (ajustado)
xu = 3.0  # Límite superior de T2/T1 (ajustado)
tol = 1e-5  # Tolerancia

# Ejecutar métodos y recopilar datos
raiz_biseccion, errores_biseccion = biseccion(f, xl, xu, tol)
raiz_regla_falsa, errores_regla_falsa = regla_falsa(f, xl, xu, tol)

# Número de iteraciones
iteraciones_biseccion = len(errores_biseccion)
iteraciones_regla_falsa = len(errores_regla_falsa)

# Imprimir resultados
print(f"\nMétodo de Bisección:")
if raiz_biseccion is not None:
    print(f"La raíz (T2/T1) encontrada es: {raiz_biseccion}")
    print(f"Número total de iteraciones: {iteraciones_biseccion}")

print(f"\nMétodo de Regla Falsa:")
if raiz_regla_falsa is not None:
    print(f"La raíz (T2/T1) encontrada es: {raiz_regla_falsa}")
    print(f"Número total de iteraciones: {iteraciones_regla_falsa}")
