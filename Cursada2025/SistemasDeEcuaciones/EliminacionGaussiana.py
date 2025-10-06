import numpy as np

def eliminacion_gaussiana(A, b):
    A = A.copy().astype(float)  # Copia para no modificar el original
    b = b.copy().astype(float)
    n = len(b)

    # Triangulación
    for i in range(n):
        # Pivoteo parcial
        max_index = np.argmax(np.abs(A[i:n, i])) + i
        if abs(A[max_index, i]) < 1e-10:
            raise ValueError("El sistema no tiene solución única.")
        
        if max_index != i:
            A[[i, max_index]] = A[[max_index, i]]
            b[[i, max_index]] = b[[max_index, i]]

        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        suma = np.sum(A[i, i+1:] * x[i+1:])
        x[i] = (b[i] - suma) / A[i, i]
    
    return x

A = np.array([[7, 14.2], [14.2, 32.86]], dtype=float)
b = np.array([9.076, 23.1896], dtype=float)

solucion = eliminacion_gaussiana(A, b)
print("La solución es:", solucion)
