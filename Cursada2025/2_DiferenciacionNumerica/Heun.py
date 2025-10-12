import math

def f(x: float, y: float) -> float:
    """
    Define la función f(x, y) de la EDO dy/dx = f(x, y).
    """
    return -2 * x * y

def metodo_heun(f, x0: float, y0: float, xfinal: float, n_intervalos: int) -> tuple[list, list]:
    """
    Resuelve una EDO de primer orden usando el método de Heun (predictor-corrector).

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
    
    x_valores = [x0]
    y_valores = [y0]

    xi = x0
    yi = y0

    for _ in range(n_intervalos):
        # Predictor (Paso de Euler) [cite: 1170]
        y_predictor = yi + h * f(xi, yi)
        
        # Actualizar xi para el corrector
        xi = xi + h
        
        # Corrector (Paso de Heun) [cite: 1171]
        pendiente_promedio = (f(x_valores[-1], yi) + f(xi, y_predictor)) / 2
        yi = yi + h * pendiente_promedio
        
        x_valores.append(xi)
        y_valores.append(yi)
        
    return x_valores, y_valores

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # Condiciones iniciales y parámetros
    x0 = 0.0
    y0 = 1.0
    xfinal = 1.0
    n = 5  # Número de subintervalos (para h=0.2)

    # Calcular la solución
    x_res, y_res = metodo_heun(f, x0, y0, xfinal, n)

    # Archivo para guardar los resultados
    try:
        with open("resultados-Heun.txt", "w") as archivo:
            archivo.write("Resultados del Metodo de Heun:\n")
            archivo.write("--------------------------------\n")
            for i, (x_val, y_val) in enumerate(zip(x_res, y_res)):
                archivo.write(f"x_{i} = {x_val:.5f}, y_{i} = {y_val:.5f}\n")
        
        print(f"Resultados guardados en 'resultados-Heun.txt' con n = {n} intervalos.")

    except IOError:
        print("Error al abrir el archivo.")