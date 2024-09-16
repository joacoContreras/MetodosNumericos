import numpy as np

def eliminacion_gaussiana(A, b, tol=1e-10):
    n = len(b)
    pivoteo_usado = False
    
    # Triangulación
    for i in range(n):
        # Si el valor en la diagonal es muy pequeño, usa pivoteo parcial
        if abs(A[i, i]) < tol:
            # Pivoteo parcial
            max_index = np.argmax(np.abs(A[i:n, i])) + i
            if abs(A[max_index, i]) < tol:
                raise ValueError("El sistema no tiene solución única.")
            
            if max_index != i:
                # Intercambiar filas
                A[[i, max_index]] = A[[max_index, i]]
                b[[i, max_index]] = b[[max_index, i]]
                pivoteo_usado = True  # Marcamos que se usó pivoteo parcial

        # Eliminación Gaussiana
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # Imprimir matriz triangular
    print("\nMatriz triangular obtenida:")
    print(A)
    
    if pivoteo_usado:
        print("\nSe empleó pivoteo parcial durante el proceso.")
    else:
        print("\nNo fue necesario emplear pivoteo parcial. Eliminación simple fue suficiente.")

    # Sustitución hacia atrás
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        suma = np.sum(A[i, i+1:] * x[i+1:])
        x[i] = (b[i] - suma) / A[i, i]
    
    return x

# Ejemplo de uso:
A = np.array([[2, -1, 3, -4], 
              [1, 0, -2, 2], 
              [3, -4, -1, 1], 
              [-1, -1, 3, 1]], dtype=float)
b = np.array([5, 3, -4, 10], dtype=float)

solucion = eliminacion_gaussiana(A, b)
print("\nLa solución es:", solucion)
