import matplotlib.pyplot as plt

# Metodo del punto fijo

plt.style.use('classic')

def g(x):
    return (4 - x) ** (1 / 2)

def punto_fijo():
    xViejo = 0
    tol = 1e-6
    iteracion = 0
    
    while True:
        iteracion += 1
        xNuevo = g(xViejo)
        
        if abs((g(xViejo + 0.01) - g(xViejo)) / 0.01) > 1:
            print("No se cumple el criterio de convergencia")
            exit(1)
        
        error = abs(xNuevo - xViejo)
        if error < tol:
            break
        xViejo = xNuevo
        
    return xNuevo, error, iteracion

x, error, iteracion = punto_fijo()
print("X = " + str(x))
print("Error = " + str(error))
print("Iteracion = " + str(iteracion))
