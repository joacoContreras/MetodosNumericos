# Método de Gauss
# Solución a Sistemas de Ecuaciones
# de la forma A.X=B
import numpy as np

def eliminacion_gaussiana_con_determinante(A, B):
    # PROCEDIMIENTO
    casicero = 1e-15  # Considerar como 0
    A = np.array(A, dtype=float)
    AB = np.concatenate((A, B), axis=1)
    AB0 = np.copy(AB)
    
    tamano = np.shape(AB)
    n = tamano[0]
    m = tamano[1]
    
    # Inicializar determinante
    determinante = 1.0

    # Pivoteo parcial por filas
    for i in range(0, n-1, 1):
        columna = abs(AB[i:, i])
        dondemax = np.argmax(columna)
        
        if dondemax != 0:
            # Intercambiar filas
            temporal = np.copy(AB[i, :])
            AB[i, :] = AB[dondemax+i, :]
            AB[dondemax+i, :] = temporal
            determinante *= -1  # Cambiar signo del determinante cuando se intercambian filas

    AB1 = np.copy(AB)
    
    # Eliminación hacia adelante
    for i in range(0, n-1, 1):
        pivote = AB[i, i]
        determinante *= pivote  # Multiplicar el determinante por el pivote
        
        if abs(pivote) < casicero:
            determinante = 0
            break

        adelante = i + 1
        for k in range(adelante, n, 1):
            factor = AB[k, i] / pivote
            AB[k, :] -= AB[i, :] * factor

    # Añadir el último pivote al determinante
    determinante *= AB[-1, -2]

    # Sustitución hacia atrás
    X = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        suma = 0
        for j in range(i+1, n, 1):
            suma += AB[i, j] * X[j]
        b = AB[i, -1]
        X[i] = (b - suma) / AB[i, i]

    X = np.transpose([X])
    
    # SALIDA
    print('Matriz aumentada inicial:')
    print(AB0)
    print('\nMatriz tras pivoteo parcial:')
    print(AB1)
    print('\nMatriz tras eliminación hacia adelante:')
    print(AB)
    print('\nDeterminante de la matriz A:', determinante)
    print('\nSolución del sistema (X):')
    print(X)
    
    return X, determinante

# Ejemplo de uso:
A = np.array([[1,2,1,4],[0,2,4,3],[4,2,2,1],[-3,1,3,2]])
B = np.array([[13],[28],[20],[6]])

solucion, determinante = eliminacion_gaussiana_con_determinante(A, B)
