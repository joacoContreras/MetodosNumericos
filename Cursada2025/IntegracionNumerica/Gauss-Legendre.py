import numpy as np

# Función con manejo de x = -1
def fx(x):
    if np.isclose(x, -1.0):  # manejo del punto singular
        return 1.0  # límite sin(0)/0 = 1
    return np.sin(x+1) / (x+1)

# Intervalo
a = -1
b = 1

puntos = 2  # Debe ser entre 2 y 6

# Gauss-Legendre
if 2 <= puntos <= 6:
    match puntos:
        case 2:
            c = [1, 1]
            xg = [-0.577350269, 0.577350269]
        case 3:
            c = [0.5555556, 0.8888889, 0.5555556]
            xg = [-0.774596669, 0.0, 0.774596669]
        case 4:
            c = [0.3478548, 0.6521452, 0.6521452, 0.3478548]
            xg = [-0.861136312, -0.339981044, 0.339981044, 0.861136312]
        case 5:
            c = [0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269]
            xg = [-0.906179846, -0.538469310, 0.0, 0.538469310, 0.906179846]
        case 6:
            c = [0.1713245, 0.3607616, 0.4679139, 0.4679139, 0.3607616, 0.1713245]
            xg = [-0.932469514, -0.661209386, -0.238619186, 0.238619186, 0.661209386, 0.932469514]
    
    # Transformación a [a, b] y cálculo
    integral = 0
    for ci, xi in zip(c, xg):
        xp = (b-a)/2 * xi + (b+a)/2
        integral += ci * fx(xp)
    integral *= (b-a)/2

    print(f"Integral aproximada con {puntos} puntos: {integral}")
else:
    print("El número de puntos debe estar entre 2 y 6")
