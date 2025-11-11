def metodo_rk4(f, x0: float, y0: float, x_final: float, n_pasos: int) -> tuple[list, list]:
    """
    Resuelve una EDO de primer orden usando el método de Runge-Kutta de 4to orden.
    Sigue la implementación clásica descrita en tus apuntes.

    Args:
        f: La función f(x, y) de la EDO dy/dx = f(x, y).
        x0: Valor inicial de x.
        y0: Valor inicial de y.
        x_final: Valor final de x.
        n_pasos: El número de pasos o iteraciones.

    Returns:
        Una tupla con dos listas: los valores de x y los valores de y.
    """
    h = (x_final - x0) / n_pasos
    
    x_valores = [x0]
    y_valores = [y0]

    xi = x0
    yi = y0

    for _ in range(n_pasos):
        # Fórmulas de RK4 clásico
        k1 = f(xi, yi)
        k2 = f(xi + h / 2, yi + (k1 * h) / 2)
        k3 = f(xi + h / 2, yi + (k2 * h) / 2)
        k4 = f(xi + h, yi + k3 * h)
        
        # Calcular el siguiente valor de y
        yi = yi + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        xi = xi + h
        
        x_valores.append(xi)
        y_valores.append(yi)
        
    return x_valores, y_valores

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    # 1. Definir la función a resolver
    def f(x, y):
        return x + y

    # 2. Ingresar las condiciones iniciales y los parámetros
    print('Ingrese las condiciones iniciales:')
    x0_input = float(input('x0 = '))
    y0_input = float(input('y0 = '))

    print('Ingrese el punto de cálculo:')
    xn_input = float(input('xn = '))

    print('Ingrese el número de pasos:')
    pasos_input = int(input('Número de pasos = '))

    # 3. Llamar al método RK4 para obtener los resultados
    x_resultados, y_resultados = metodo_rk4(f, x0_input, y0_input, xn_input, pasos_input)

    # 4. Presentar la solución en una tabla
    print('\n-------- SOLUCIÓN --------')
    print('--------------------------')
    print('Paso\tx_i\t\ty_i')
    print('--------------------------')
    for i in range(len(x_resultados)):
        print(f"{i}\t{x_resultados[i]:.4f}\t\t{y_resultados[i]:.4f}")
    print('--------------------------')
    
    print(f'\nResultado final: En x = {x_resultados[-1]:.4f}, y = {y_resultados[-1]:.4f}')