import math


def calcular_n(x0: float, xfinal: float, h: float) -> int:
    """Calcula n para una malla uniforme con paso de magnitud h."""
    if not all(math.isfinite(valor) for valor in (x0, xfinal, h)):
        raise ValueError('Los extremos y h deben ser números finitos.')
    if h <= 0:
        raise ValueError('h debe ser mayor que cero.')

    cociente = abs(xfinal - x0) / h
    if not math.isfinite(cociente):
        raise ValueError('El intervalo o la cantidad de pasos es demasiado grande.')
    n = round(cociente)
    if n < 1 or not math.isclose(cociente, n, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            'h debe dividir el intervalo en un número entero positivo de pasos. '
            f'Se obtuvo n = {cociente:.12g}.'
        )
    return n


def f(x: float, y: float) -> float:
    """
    Define la función f(x, y) de la EDO dy/dx = f(x, y).
    """
    return y * (4 * y - x**2)

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


def crear_tabla(x_valores: list, y_valores: list) -> str:
    """Organiza los resultados; usa notación científica para valores grandes."""
    encabezado = f'{"Paso (i)":>8} | {"x_i":>14} | {"y_i (Euler)":>20}'
    separador = '-' * len(encabezado)
    filas = [encabezado, separador]
    for i, (xi, yi) in enumerate(zip(x_valores, y_valores)):
        filas.append(f'{i:8d} | {xi:14.5f} | {yi:20.10g}')
    return '\n'.join(filas)


# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # Condiciones iniciales y parámetros
    x0 = 0.0
    y0 = 1.0
    xfinal = 3.0
    h = 0.1  # Magnitud del paso; n se calcula automáticamente.

    try:
        n = calcular_n(x0, xfinal, h)
    except ValueError as error:
        raise SystemExit(f'Error: {error}')

    # Calcular la solución
    x_res, y_res = metodo_euler(f, x0, y0, xfinal, n)

    resultados = (
        'Resultados del método de Euler\n'
        f'x0 = {x0:g}, y0 = {y0:g}, xfinal = {xfinal:g}, h = {h:g}\n'
        f'n = |xfinal - x0| / h = {n} intervalos ({n + 1} puntos)\n\n'
        f'{crear_tabla(x_res, y_res)}\n'
    )
    print(resultados)

    # Archivo para guardar los resultados
    try:
        with open("resultados-Euler.txt", "w", encoding="utf-8") as archivo:
            archivo.write(resultados)
        
        print(f"Resultados guardados en 'resultados-Euler.txt' con n = {n} intervalos.")

    except IOError:
        print("Error al abrir el archivo.")
