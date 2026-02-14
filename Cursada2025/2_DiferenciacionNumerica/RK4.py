def metodo_rk4(f, x0: float, y0: float, x_final: float, h: float = 0.1) -> tuple[list, list]:
    """
    Resuelve una EDO de primer orden usando el método de Runge-Kutta de 4to orden (alternativo).
    Usa la fórmula: yi+1 = yi + h/8*(k1 + 3*k2 + 3*k3 + k4)

    Args:
        f: La función f(x, y) de la EDO dy/dx = f(x, y).
        x0: Valor inicial de x.
        y0: Valor inicial de y.
        x_final: Valor final de x.
        h: Tamaño de paso (default: 0.1).

    Returns:
        Una tupla con dos listas: los valores de x y los valores de y.
    """
    x_valores = [x0]
    y_valores = [y0]

    xi = x0
    yi = y0

    while xi < x_final:
        # Fórmulas de RK4 alternativo
        k1 = f(xi, yi)
        k2 = f(xi + h / 3, yi + (k1 * h) / 3)
        k3 = f(xi + 2 * h / 3, yi + (2 * k2 * h) / 3)
        k4 = f(xi + h, yi + k1 * h - k2 * h + k3 * h)
        
        # Calcular el siguiente valor de y usando la fórmula alternativa
        yi = yi + (h / 8) * (k1 + 3 * k2 + 3 * k3 + k4)
        xi = xi + h
        
        x_valores.append(xi)
        y_valores.append(yi)
        
    return x_valores, y_valores

# --- Bloque principal de ejecución ---
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 1. Definir la función a resolver: dy/dx = y - x^2 + 1
    def f(x, y):
        return y - x**2 + 1
    
    # Solución exacta para dy/dx = y - x^2 + 1 con y(0) = 0.5
    # La solución exacta es: y(x) = (x + 1)^2 - 0.5*e^x
    def solucion_exacta(x):
        return (x + 1)**2 - 0.5 * np.exp(x)

    # 2. Ingresar las condiciones iniciales y los parámetros
    print('Ingrese las condiciones iniciales:')
    x0_input = float(input('x0 = '))
    y0_input = float(input('y0 = '))

    print('Ingrese el punto de cálculo:')
    xn_input = float(input('xn = '))

    h = 0.1  # Tamaño de paso fijo
    print(f'Tamaño de paso: h = {h}')

    # 3. Llamar al método RK4 para obtener los resultados
    x_resultados, y_resultados = metodo_rk4(f, x0_input, y0_input, xn_input, h)

    # 4. Presentar la solución en una tabla
    print('\n-------- SOLUCIÓN --------')
    print('--------------------------------------------------------------')
    print('Paso\tx_i\t\ty_aprox\t\ty_exacta\t\tError')
    print('--------------------------------------------------------------')
    for i in range(len(x_resultados)):
        y_exacta = solucion_exacta(x_resultados[i])
        error = abs(y_exacta - y_resultados[i])
        print(f"{i}\t{x_resultados[i]:.4f}\t\t{y_resultados[i]:.8f}\t{y_exacta:.8f}\t{error:.8f}")
    print('--------------------------------------------------------------')
    
    print(f'\nResultado final: En x = {x_resultados[-1]:.4f}')
    print(f'y_aproximado = {y_resultados[-1]:.8f}')
    print(f'y_exacto     = {solucion_exacta(x_resultados[-1]):.8f}')
    print(f'Error absoluto = {abs(solucion_exacta(x_resultados[-1]) - y_resultados[-1]):.8f}')
    
    # 5. Graficar la solución exacta vs aproximada
    x_exacto = np.linspace(x0_input, xn_input, 100)
    y_exacto = solucion_exacta(x_exacto)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_exacto, y_exacto, 'b-', label='Solución Exacta', linewidth=2)
    plt.plot(x_resultados, y_resultados, 'ro-', label='RK4 Alternativo (h=0.1)', markersize=6)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Método RK4 Alternativo: Solución Exacta vs Aproximada\ndy/dx = y - x² + 1', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()