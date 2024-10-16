import numpy as np
import matplotlib.pyplot as plt
import math

# INGRESO
fx = lambda x: math.sin(2*x)*math.e**(-x)

# Intervalo de integración
a = 0
b = math.pi

puntos = 3 # Debe ser entre 2 y 6

# PROCEDIMIENTO
if puntos > 1 and puntos < 7:
    match puntos:
        case 2:
            c0, c1 = 1, 1
            x0, x1 = -0.577350269, 0.577350269
            integral = ((b-a)/2) * (c0 * fx(((b-a)/2)*x0 + (b+a)/2) + 
                                    c1 * fx(((b-a)/2)* x1 + (b+a)/2))
            print('a = ',a,' b = ', b, ' puntos = ',puntos)
            print('Integral: ', integral)
        case 3:
            c0, c1, c2 = 0.5555556, 0.8888889, 0.5555556
            x0, x1, x2 = -0.774596669, 0.0, 0.774596669
            integral = ((b-a)/2) * (c0 * fx((b-a)*x0/2 + (b+a)/2) + 
                                     c1 * fx((b-a)*x1/2 + (b+a)/2) + 
                                     c2 * fx((b-a)*x2/2 + (b+a)/2))
            print('Integral: ', integral)
        case 4:
            c0, c1, c2, c3 = 0.3478548, 0.6521452, 0.6521452, 0.3478548
            x0, x1, x2, x3 = -0.861136312, -0.339981044, 0.339981044, 0.861136312
            integral = ((b-a)/2) * (c0 * fx((b-a)*x0/2 + (b+a)/2) + 
                                     c1 * fx((b-a)*x1/2 + (b+a)/2) + 
                                     c2 * fx((b-a)*x2/2 + (b+a)/2) + 
                                     c3 * fx((b-a)*x3/2 + (b+a)/2))
            print('Integral: ', integral)
        case 5:
            c0, c1, c2, c3, c4 = 0.2369269, 0.4786287, 0.5688889, 0.4786287, 0.2369269
            x0, x1, x2, x3, x4 = -0.906179846, -0.538469310, 0.0, 0.538469310, 0.906179846
            integral = ((b-a)/2) * (c0 * fx((b-a)*x0/2 + (b+a)/2) + 
                                     c1 * fx((b-a)*x1/2 + (b+a)/2) + 
                                     c2 * fx((b-a)*x2/2 + (b+a)/2) + 
                                     c3 * fx((b-a)*x3/2 + (b+a)/2) + 
                                     c4 * fx((b-a)*x4/2 + (b+a)/2))
            print('Integral: ', integral)
        case 6:
            c0, c1, c2, c3, c4, c5 = 0.1713245, 0.3607616, 0.4679139, 0.4679139, 0.3607616, 0.1713245
            x0, x1, x2, x3, x4, x5 = -0.932469514, -0.661209386, -0.238619186, 0.238619186, 0.661209386, 0.932469514
            integral = ((b-a)/2) * (c0 * fx((b-a)*x0/2 + (b+a)/2) + 
                                     c1 * fx((b-a)*x1/2 + (b+a)/2) + 
                                     c2 * fx((b-a)*x2/2 + (b+a)/2) + 
                                     c3 * fx((b-a)*x3/2 + (b+a)/2) + 
                                     c4 * fx((b-a)*x4/2 + (b+a)/2) + 
                                     c5 * fx((b-a)*x5/2 + (b+a)/2))
            print('Integral: ', integral)
