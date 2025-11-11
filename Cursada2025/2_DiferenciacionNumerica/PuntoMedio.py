import math

def f(x: float, y: float) -> float:
    """
    Define la función f(x, y) de la EDO dy/dx = f(x, y).
    """
    return -2 * x * y

def metodo_punto_medio(f, x0: float, y0: float, xfinal: float, n_intervalos: int) -> tuple[list, list]:
    """
    Resuelve una EDO de primer orden usando el Método del Punto Medio.

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
        # 1. Estimar el valor en el punto medio del intervalo
        x_mid = xi + h / 2
        y_mid = yi + (h / 2) * f(xi, yi)
        
        # 2. Calcular la pendiente en el punto medio
        pendiente_mid = f(x_mid, y_mid)
        
        # 3. Usar la pendiente del punto medio para dar el paso completo
        yi = yi + h * pendiente_mid
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
    n = 5  # Número de subintervalos (para h=0.2)

    # Calcular la solución
    x_res, y_res = metodo_punto_medio(f, x0, y0, xfinal, n)

    # Archivo para guardar los resultados
    try:
        with open("resultados-PM.txt", "w") as archivo:
            archivo.write("Resultados del Metodo del Punto Medio:\n")
            archivo.write("-------------------------------------\n")
            for i, (x_val, y_val) in enumerate(zip(x_res, y_res)):
                archivo.write(f"x_{i} = {x_val:.5f}, y_{i} = {y_val:.5f}\n")
        
        print(f"Resultados guardados en 'resultados-PM.txt' con n = {n} intervalos.")

    except IOError:
        print("Error al abrir el archivo.")