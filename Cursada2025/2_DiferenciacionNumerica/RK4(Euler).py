
import numpy as np
import math

def metodo_euler(f, x0: float, y0: float, x_final: float, n_pasos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve una EDO de primer orden usando el método de Euler.

    Args:
        f: La función f(x, y) de la EDO.
        x0: Valor inicial de x.
        y0: Valor inicial de y.
        x_final: Valor final de x.
        n_pasos: El número de pasos a realizar.

    Returns:
        Una tupla con dos arrays de numpy: los valores de x y los valores de y.
    """
    h = (x_final - x0) / n_pasos
    
    x = np.zeros(n_pasos + 1)
    y = np.zeros(n_pasos + 1)
    
    x[0] = x0
    y[0] = y0
    
    for i in range(n_pasos):
        y[i+1] = y[i] + h * f(x[i], y[i])
        x[i+1] = x[i] + h
        
    return x, y

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # 1. Definir la función y los parámetros del problema
    def f(x, y):
        return -2 * y + x

    x0 = 0.0
    y0 = 1.0
    x_final = 2.0
    N = 10  # Número de pasos para la solución más gruesa (y1)

    # 2. Calcular las tres soluciones usando el MISMO método (Euler)
    # y1 con N pasos (h)
    x1, y1 = metodo_euler(f, x0, y0, x_final, N)
    
    # y2 con 2N pasos (h/2)
    x2, y2 = metodo_euler(f, x0, y0, x_final, 2 * N)

    # y3 con 4N pasos (h/4)
    x3, y3 = metodo_euler(f, x0, y0, x_final, 4 * N)
    
    # 3. Calcular el Factor de Convergencia Q en los puntos comunes
    Q = np.zeros(N + 1)
    for i in range(N + 1):
        # y1[i] se compara con y2[2*i] y y3[4*i]
        numerador = y1[i] - y2[2*i]
        denominador = y2[2*i] - y3[4*i]
        
        if abs(denominador) > 1e-12:  # Evitar división por cero
            Q[i] = (1 / math.log(2)) * math.log(abs(numerador / denominador))
        else:
            Q[i] = float('inf')  # Indicar que no se pudo calcular

    # 4. Guardar los resultados en un archivo
    try:
        with open("resultados_Q.txt", "w") as archivo:
            archivo.write("Factor de Convergencia Q para el Metodo de Euler\n")
            archivo.write("--------------------------------------------------\n")
            archivo.write("x\t\ty1(h)\t\ty2(h/2)\t\ty3(h/4)\t\tQ\n")
            archivo.write("--------------------------------------------------\n")
            for i in range(N + 1):
                archivo.write(f"{x1[i]:.5f}\t\t{y1[i]:.5f}\t\t{y2[2*i]:.5f}\t\t{y3[4*i]:.5f}\t\t{Q[i]:.5f}\n")
        
        print("Cálculos completados. Los resultados se guardaron en 'resultados_Q.txt'.")
    except IOError as e:
        print(f"Error al escribir el archivo: {e}")