"""
=============================================================================
Método de Interpolación de Newton mediante Diferencias Divididas
=============================================================================
Este módulo permite:
1. Construir la tabla de diferencias divididas a partir de una tabla con 
   dos columnas (x, y).
2. Obtener el polinomio interpolador de Newton tanto en su forma estándar
   (con factores de Newton) como en su forma polinómica simplificada (expandida).
3. Evaluar cualquier número dado en el polinomio obtenido.
4. Graficar la función interpolante, los puntos muestreados y el punto evaluado.
=============================================================================
"""

import sys
import argparse
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def cargar_datos(tabla_o_archivo):
    """
    Carga y valida los datos de una tabla de 2 columnas (x, y).
    Puede recibir:
    - Un arreglo o lista de pares [[x0, y0], [x1, y1], ...]
    - Una tupla de arreglos (x, y)
    - La ruta a un archivo de texto o CSV
    """
    if isinstance(tabla_o_archivo, str):
        # Intentar cargar desde archivo CSV o texto separado por comas / espacios
        try:
            datos = np.loadtxt(tabla_o_archivo, delimiter=',')
        except Exception:
            datos = np.loadtxt(tabla_o_archivo)
        x = datos[:, 0]
        y = datos[:, 1]
    elif isinstance(tabla_o_archivo, tuple) and len(tabla_o_archivo) == 2:
        x = np.asarray(tabla_o_archivo[0], dtype=float)
        y = np.asarray(tabla_o_archivo[1], dtype=float)
    else:
        datos = np.asarray(tabla_o_archivo, dtype=float)
        if datos.ndim != 2 or datos.shape[1] != 2:
            raise ValueError("La tabla debe contener exactamente dos columnas: [x, y].")
        x = datos[:, 0]
        y = datos[:, 1]

    if len(x) < 2:
        raise ValueError("Se requieren al menos 2 puntos para realizar la interpolación.")

    # Verificar puntos x duplicados
    if len(np.unique(x)) != len(x):
        raise ValueError("Los valores de 'x' deben ser todos distintos entre sí.")

    # Ordenar los puntos respecto a x
    indices_orden = np.argsort(x)
    return x[indices_orden], y[indices_orden]


def tabla_diferencias_divididas(x, y):
    """
    Calcula la matriz completa de diferencias divididas de Newton.
    
    Retorna:
    - tabla: Matriz de dimensión (n, n) donde la columna j representa
             las diferencias divididas de orden j.
    """
    n = len(x)
    tabla = np.zeros((n, n), dtype=float)
    tabla[:, 0] = y  # Diferencias de orden 0: f[x_i] = y_i

    for j in range(1, n):
        for i in range(n - j):
            tabla[i, j] = (tabla[i + 1, j - 1] - tabla[i, j - 1]) / (x[i + j] - x[i])

    return tabla


def imprimir_tabla_diferencias(x, tabla):
    """
    Muestra en consola la tabla de diferencias divididas con formato alineado.
    """
    n = len(x)
    encabezados = ["i", "x_i", "f[x_i]"] + [f"{k}° Dif" for k in range(1, n)]
    
    # Ancho de columna
    ancho_col = 13
    encabezado_str = f"{'i':>3} | {'x_i':>{ancho_col}} | {'f[x_i]':>{ancho_col}} | " + \
                     " | ".join(f"{encabezados[k + 2]:>{ancho_col}}" for k in range(1, n))
    separador = "-" * len(encabezado_str)

    print("\n" + "=" * len(encabezado_str))
    print("TABLA DE DIFERENCIAS DIVIDIDAS DE NEWTON")
    print("=" * len(encabezado_str))
    print(encabezado_str)
    print(separador)

    for i in range(n):
        fila_str = f"{i:3d} | {x[i]:{ancho_col}.5f} | {tabla[i, 0]:{ancho_col}.5f} | "
        cols_dif = []
        for j in range(1, n):
            if i <= n - 1 - j:
                cols_dif.append(f"{tabla[i, j]:{ancho_col}.5f}")
            else:
                cols_dif.append(f"{'':>{ancho_col}}")
        fila_str += " | ".join(cols_dif)
        print(fila_str)

    print(separador + "\n")


def formatear_forma_newton(x, coeficientes):
    """
    Genera la representación en texto tradicional del polinomio en la base de Newton:
    P(x) = a_0 + a_1*(x - x_0) + a_2*(x - x_0)*(x - x_1) + ...
    """
    terminos = []
    for i, c in enumerate(coeficientes):
        if abs(c) < 1e-12 and i > 0:
            continue
        signo = ("+ " if c >= 0 else "- ") if i > 0 else ("" if c >= 0 else "-")
        c_abs = abs(c)
        if i == 0:
            terminos.append(f"{c:.6g}")
        else:
            factores = "".join([
                f"(x - {x[k]:.4g})" if x[k] >= 0 else f"(x + {abs(x[k]):.4g})"
                for k in range(i)
            ])
            terminos.append(f"{signo}{c_abs:.6g}*{factores}")
    return " ".join(terminos)


def obtener_polinomio_simbolico(x, coeficientes):
    """
    Construye las expresiones simbólicas del polinomio de Newton usando Sympy.
    
    Retorna:
    - texto_newton: Cadena con la fórmula en base de Newton tradicional.
    - poly_expandido: Polinomio simplificado en potencias canónicas de x.
    - poly_func: Función evaluable numérica (lambdify).
    """
    var_x = sp.Symbol('x')
    poly_newton = sp.Float(coeficientes[0])
    termino_producto = 1

    for i in range(1, len(coeficientes)):
        termino_producto *= (var_x - float(x[i - 1]))
        poly_newton += sp.Float(coeficientes[i]) * termino_producto

    poly_expandido = sp.expand(poly_newton)
    texto_newton = formatear_forma_newton(x, coeficientes)
    poly_func = sp.lambdify(var_x, poly_expandido, modules=['numpy'])

    return texto_newton, poly_expandido, poly_func


def evaluar_newton(x, coeficientes, x_eval):
    """
    Evalúa el polinomio de Newton en uno o varios valores usando el algoritmo
    anidado de Horner para polinomios en base de Newton (óptimo en precisión y tiempo).
    """
    n = len(coeficientes)
    x_val = np.asarray(x_eval, dtype=float)
    resultado = np.full_like(x_val, coeficientes[-1], dtype=float)

    for k in range(n - 2, -1, -1):
        resultado = resultado * (x_val - x[k]) + coeficientes[k]

    if resultado.ndim == 0:
        return float(resultado)
    return resultado


def graficar_interpolacion(x, y, poly_func, x_eval=None, y_eval=None, ruta_guardado="interpolacion_newton.png"):
    """
    Genera un gráfico con los puntos originales, la curva del polinomio de Newton
    y el punto evaluado.
    """
    # Rango para graficar
    x_min, x_max = np.min(x), np.max(x)
    if x_eval is not None:
        x_min = min(x_min, np.min(x_eval))
        x_max = max(x_max, np.max(x_eval))
    
    margen = (x_max - x_min) * 0.08
    if margen == 0:
        margen = 1.0
    x_graf = np.linspace(x_min - margen, x_max + margen, 500)
    y_graf = poly_func(x_graf)

    plt.figure(figsize=(9, 5.5))
    plt.plot(x_graf, y_graf, label="Polinomio de Newton $P(x)$", color="#1f77b4", linewidth=2)
    plt.plot(x, y, 'o', label="Puntos de la tabla", color="#2ca02c", markersize=8, markeredgecolor='black')

    if x_eval is not None and y_eval is not None:
        if np.isscalar(x_eval):
            plt.plot(x_eval, y_eval, '*', label=f"Punto evaluado ($x={x_eval:.4f}$, $y={y_eval:.4f}$)",
                     color="#d62728", markersize=14, markeredgecolor='black')
            plt.annotate(f"P({x_eval:.3f}) = {y_eval:.4f}",
                         (x_eval, y_eval),
                         textcoords="offset points",
                         xytext=(10, 10),
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.6),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        else:
            plt.plot(x_eval, y_eval, '*', label="Puntos evaluados", color="#d62728",
                     markersize=12, markeredgecolor='black')

    plt.title("Interpolación Polinómica de Newton (Diferencias Divididas)", fontsize=13, fontweight='bold')
    plt.xlabel("x", fontsize=11)
    plt.ylabel("P(x)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()

    # Guardar en archivo y mostrar si es posible
    try:
        plt.savefig(ruta_guardado, dpi=150)
        print(f"[+] Gráfico guardado exitosamente en: '{ruta_guardado}'")
    except Exception as e:
        print(f"[!] No se pudo guardar la imagen del gráfico: {e}")

    try:
        plt.show()
    except Exception:
        pass


def interpolar_newton(tabla_o_archivo, x_eval=None, graficar=True, ruta_grafico="interpolacion_newton.png"):
    """
    Función principal de resolución:
    - Procesa los datos de la tabla.
    - Calcula diferencias divididas y coeficientes de Newton.
    - Obtiene expresiones simbólicas.
    - Valúa el polinomio en x_eval si se especifica.
    - Opcionalmente genera la gráfica.
    """
    x, y = cargar_datos(tabla_o_archivo)
    n = len(x)

    # 1. Calcular tabla de diferencias divididas
    tabla_dif = tabla_diferencias_divididas(x, y)
    coeficientes = tabla_dif[0, :]  # Coeficientes a_0, a_1, ..., a_{n-1}

    # Mostrar la tabla
    imprimir_tabla_diferencias(x, tabla_dif)

    # 2. Construir expresiones del polinomio
    poly_newton, poly_expandido, poly_func = obtener_polinomio_simbolico(x, coeficientes)

    print("=" * 70)
    print("POLINOMIO INTERPOLADOR DE NEWTON")
    print("=" * 70)
    print("Coeficientes a_k = f[x_0, ..., x_k]:")
    for k, c in enumerate(coeficientes):
        print(f"  a_{k} = {c:+.8f}")

    print("\n1) Forma de factores de Newton:")
    print(f"  P(x) = {poly_newton}")

    print("\n2) Forma simplificada (expandida):")
    print(f"  P(x) = {poly_expandido}")
    print("=" * 70)

    # 3. Valuación en x_eval
    y_eval = None
    if x_eval is not None:
        y_eval = evaluar_newton(x, coeficientes, x_eval)
        print(f"\nEVALUACIÓN EN x = {x_eval}:")
        if np.isscalar(x_eval):
            if x_eval < x[0] or x_eval > x[-1]:
                print("  [ADVERTENCIA] El valor de x está fuera del rango muestreado [x_min, x_max].")
                print("                Se trata de una extrapolación.")
            print(f"  P({x_eval}) = {y_eval:.8f}")
        else:
            for xi, yi in zip(np.asarray(x_eval), np.asarray(y_eval)):
                print(f"  P({xi}) = {yi:.8f}")
        print("=" * 70 + "\n")

    # 4. Gráfica
    if graficar:
        graficar_interpolacion(x, y, poly_func, x_eval=x_eval, y_eval=y_eval, ruta_guardado=ruta_grafico)

    return {
        "x": x,
        "y": y,
        "tabla_diferencias": tabla_dif,
        "coeficientes": coeficientes,
        "polinomio_newton": poly_newton,
        "polinomio_expandido": poly_expandido,
        "evaluacion": {"x_eval": x_eval, "y_eval": y_eval}
    }


# =============================================================================
# BLOQUE DE EJECUCIÓN DIRECTA
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolación de Newton mediante Diferencias Divididas para tablas de 2 columnas."
    )
    parser.add_argument(
        "--archivo", type=str, default=None,
        help="Ruta a un archivo .txt o .csv con 2 columnas (x, y)."
    )
    parser.add_argument(
        "--eval", type=float, default=2.5,
        help="Número a valuar en el polinomio interpolador obtenido (por defecto: 2.5)."
    )
    parser.add_argument(
        "--no-grafica", action="store_true",
        help="Desactiva la generación y guardado del gráfico."
    )

    args = parser.parse_args()

    # Si no se pasa archivo por argumento, usamos una tabla de prueba de 2 columnas
    if args.archivo:
        fuente_datos = args.archivo
    else:
        # Tabla de ejemplo con 2 columnas: [x, y]
        # (Modifica libremente estos valores según tus necesidades)
        fuente_datos = [
            [0, 0],
            [1.0, 2.1],
            [2.0, 3.8],
            [3.0, 5.1],
            [4.0, 5.9],
            [5.0, 6.3]
        ]
        print(">> Usando tabla de datos de ejemplo por defecto:")
        print("   x    |    y   ")
        print("   -------------")
        for fila in fuente_datos:
            print(f"  {fila[0]:4.1f}  |  {fila[1]:4.1f}")

    # Ejecutar la interpolación y valuar el número indicado
    numero_a_valuar = args.eval
    interpolar_newton(
        tabla_o_archivo=fuente_datos,
        x_eval=numero_a_valuar,
        graficar=not args.no_grafica,
        ruta_grafico="interpolacion_newton.png"
    )
