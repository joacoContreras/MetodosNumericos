import math

def f(x: float, y: float) -> float:
    """
    Define la función f(x, y) de la EDO dy/dx = f(x, y).
    """
    return -2 * x * y

def metodo_euler(f, x0: float, y0: float, xfinal: float, n_intervalos: int) -> tuple[list, list]:
    """
    Resuelve una EDO de primer orden usando el método de Euler.

    Args:
        f: La función f(x, y).
        x0: Valor inicial de x.
        y0: Valor inicial de y.
        xfinal: Valor final de x.
        n_intervalos: El número de subintervalos o pasos.

    Returns:
        Una tupla con dos listas: los valores de x y los valores de y.
    """
    h = (xfinal - x0) / n_intervalos
    
    # Listas para almacenar los resultados
    x_valores = [x0]
    y_valores = [y0]

    # Inicializar con los valores de partida
    xi = x0
    yi = y0

    # Bucle para calcular cada paso
    for _ in range(n_intervalos):
        yi = yi + h * f(xi, yi)  # Fórmula de Euler
        xi = xi + h
        
        x_valores.append(xi)
        y_valores.append(yi)
        
    return x_valores, y_valores

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # Condiciones iniciales y parámetros
    x0 = 0.0
    y0 = 1.0
    xfinal = 1.0
    n = 5  # Número de subintervalos (h = 0.2, como en tus notas)

    # Calcular la solución
    x_res, y_res = metodo_euler(f, x0, y0, xfinal, n)

    # Archivo para guardar los resultados
    try:
        with open("resultados-Euler.txt", "w") as archivo:
            archivo.write("Resultados del Metodo de Euler:\n")
            for i in range(len(x_res)):
                archivo.write(f"x_{i} = {x_res[i]:.5f}, y_{i} = {y_res[i]:.5f}\n")
        
        print(f"Resultados guardados en 'resultados-Euler.txt' con n = {n} intervalos.")

    except IOError:
        print("Error al abrir el archivo.")
