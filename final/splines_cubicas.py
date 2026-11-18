"""
=============================================================================
Método de Interpolación mediante Splines Cúbicas (Trazadores Cúbicos)
=============================================================================
Este módulo permite:
1. Calcular los trazadores o splines cúbicos a partir de una tabla de dos 
   columnas (x, y), resolviendo el sistema tridiagonal de momentos/curvaturas.
2. Soporta condiciones de frontera:
   - 'natural' (por defecto): S''(x_0) = 0 y S''(x_n) = 0.
   - 'sujeta' / 'clamped': S'(x_0) = f'(x_0) y S'(x_n) = f'(x_n).
3. Mostrar las ecuaciones polinómicas de cada subintervalo [x_i, x_{i+1}]
   tanto en forma centrada como en forma expandida canónica:
       S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
4. Evaluar cualquier número dado en el spline correspondiente, identificando
   automáticamente a qué tramo pertenece.
5. Graficar los datos originales, cada tramo cúbico suave y el punto evaluado.
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

    if len(x) < 3:
        raise ValueError("Se requieren al menos 3 puntos para construir splines cúbicos.")

    # Verificar valores repetidos en x
    if len(np.unique(x)) != len(x):
        raise ValueError("Los valores de 'x' deben ser todos distintos entre sí.")

    # Ordenar los puntos según x en orden ascendente
    indices_orden = np.argsort(x)
    return x[indices_orden], y[indices_orden]


def calcular_splines_cubicas(x, y, tipo_frontera='natural', derivadas_frontera=(0.0, 0.0)):
    """
    Construye el sistema tridiagonal y calcula los coeficientes (a_i, b_i, c_i, d_i)
    para cada subintervalo [x_i, x_{i+1}], i = 0, ..., n-1.
    
    Donde:
        S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3
    """
    N = len(x)
    n = N - 1  # Número de tramos/subintervalos

    h = np.diff(x)  # h_i = x_{i+1} - x_i
    a = y.copy()    # a_i = y_i

    # Matriz del sistema tridiagonal A * c = r (de tamaño (n+1) x (n+1))
    A = np.zeros((n + 1, n + 1), dtype=float)
    r = np.zeros(n + 1, dtype=float)

    # Filas interiores (continuidad de la primera y segunda derivada)
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        r[i] = 3.0 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1])

    # Condiciones de frontera
    tipo = tipo_frontera.strip().lower()
    if tipo == 'natural':
        # S''(x_0) = 0  =>  2*c_0 = 0  =>  c_0 = 0
        # S''(x_n) = 0  =>  2*c_n = 0  =>  c_n = 0
        A[0, 0] = 1.0
        r[0] = 0.0
        A[n, n] = 1.0
        r[n] = 0.0
    elif tipo in ['sujeta', 'clamped']:
        fp0, fpn = derivadas_frontera
        # S'(x_0) = fp0  =>  2*h_0*c_0 + h_0*c_1 = 3*((a_1 - a_0)/h_0 - fp0)
        A[0, 0] = 2.0 * h[0]
        A[0, 1] = h[0]
        r[0] = 3.0 * ((a[1] - a[0]) / h[0] - fp0)

        # S'(x_n) = fpn  =>  h_{n-1}*c_{n-1} + 2*h_{n-1}*c_n = 3*(fpn - (a_n - a_{n-1})/h_{n-1})
        A[n, n - 1] = h[n - 1]
        A[n, n] = 2.0 * h[n - 1]
        r[n] = 3.0 * (fpn - (a[n] - a[n - 1]) / h[n - 1])
    else:
        raise ValueError(f"Tipo de frontera no reconocido: '{tipo_frontera}'. Usa 'natural' o 'sujeta'.")

    # Resolver c_i (curvaturas divididas por 2)
    c = np.linalg.solve(A, r)

    # Determinar b_i y d_i para cada tramo i = 0, ..., n-1
    b = np.zeros(n, dtype=float)
    d = np.zeros(n, dtype=float)
    for i in range(n):
        b[i] = (a[i + 1] - a[i]) / h[i] - (h[i] / 3.0) * (2.0 * c[i] + c[i + 1])
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])

    # Estructurar la información de cada tramo
    tramos = []
    var_x = sp.Symbol('x')
    for i in range(n):
        x_i = float(x[i])
        x_ip1 = float(x[i + 1])
        ai, bi, ci, di = float(a[i]), float(b[i]), float(c[i]), float(d[i])

        # Expresión centrada en x_i
        expr_centrada = ai + bi * (var_x - x_i) + ci * (var_x - x_i)**2 + di * (var_x - x_i)**3
        # Expresión expandida canónica
        expr_expandida = sp.expand(expr_centrada)

        # Coeficientes expandidos: A*x^3 + B*x^2 + C*x + D
        A_exp = di
        B_exp = ci - 3.0 * di * x_i
        C_exp = bi - 2.0 * ci * x_i + 3.0 * di * (x_i**2)
        D_exp = ai - bi * x_i + ci * (x_i**2) - di * (x_i**3)

        tramos.append({
            "indice": i,
            "x_inicio": x_i,
            "x_fin": x_ip1,
            "a": ai,
            "b": bi,
            "c": ci,
            "d": di,
            "exp_centrada": expr_centrada,
            "exp_expandida": expr_expandida,
            "coef_expandidos": (A_exp, B_exp, C_exp, D_exp)
        })

    return {
        "x": x,
        "y": y,
        "h": h,
        "a": a[:n],
        "b": b,
        "c": c,
        "d": d,
        "tramos": tramos,
        "tipo_frontera": tipo_frontera
    }


def mostrar_polinomios(tramos, tipo_frontera="natural"):
    """
    Imprime en consola las ecuaciones de cada subintervalo con formato claro.
    """
    print("\n" + "=" * 80)
    print(f"POLINOMIOS DE SPLINES CÚBICAS (Condición de Frontera: {tipo_frontera.upper()})")
    print("Forma centrada:  S_i(x) = a_i + b_i*(x - x_i) + c_i*(x - x_i)^2 + d_i*(x - x_i)^3")
    print("=" * 80)

    for tramo in tramos:
        i = tramo["indice"]
        xi = tramo["x_inicio"]
        xf = tramo["x_fin"]
        ai, bi, ci, di = tramo["a"], tramo["b"], tramo["c"], tramo["d"]
        A, B, C, D = tramo["coef_expandidos"]

        print(f"\n--- Tramo {i + 1}: x en [{xi:.4f}, {xf:.4f}] ---")
        print(f"  Coeficientes:")
        print(f"    a_{i} = {ai:+.8f} (constante)")
        print(f"    b_{i} = {bi:+.8f} (lineal)")
        print(f"    c_{i} = {ci:+.8f} (cuadrático)")
        print(f"    d_{i} = {di:+.8f} (cúbico)")
        print(f"  Forma centrada en x_{i} = {xi}:")
        print(f"    S_{i + 1}(x) = ({ai:.6f}) + ({bi:.6f})*(x - {xi:.4f}) + ({ci:.6f})*(x - {xi:.4f})^2 + ({di:.6f})*(x - {xi:.4f})^3")
        print(f"  Forma expandida estándar:")
        print(f"    S_{i + 1}(x) = {tramo['exp_expandida']}")

    print("=" * 80 + "\n")


def evaluar_spline(splines_info, x_eval):
    """
    Evalúa el spline en un número x_eval (o lista de números).
    Identifica automáticamente a qué subintervalo pertenece y aplica
    el polinomio correspondiente.
    
    Retorna:
    - Para un único número: (y_eval, indice_tramo, tramo_dict)
    - Para un arreglo: (y_eval_array, indices_tramos)
    """
    x = splines_info["x"]
    tramos = splines_info["tramos"]
    n_tramos = len(tramos)

    def evaluar_un_punto(val):
        # Determinar el subintervalo
        if val <= x[0]:
            idx = 0
            alerta = "extrapolacion_izq" if val < x[0] else "exacto"
        elif val >= x[-1]:
            idx = n_tramos - 1
            alerta = "extrapolacion_der" if val > x[-1] else "exacto"
        else:
            # Buscar el intervalo correspondiente
            idx = 0
            for k in range(n_tramos):
                if x[k] <= val <= x[k + 1]:
                    idx = k
                    break
            alerta = "interpolacion"

        tramo = tramos[idx]
        dx = val - tramo["x_inicio"]
        y_val = tramo["a"] + tramo["b"] * dx + tramo["c"] * (dx**2) + tramo["d"] * (dx**3)
        return float(y_val), idx, tramo, alerta

    if np.isscalar(x_eval):
        return evaluar_un_punto(float(x_eval))
    else:
        valores_y = []
        indices = []
        alertas = []
        for val in x_eval:
            vy, idx, _, al = evaluar_un_punto(float(val))
            valores_y.append(vy)
            indices.append(idx)
            alertas.append(al)
        return np.array(valores_y), indices, None, alertas


def graficar_splines(splines_info, x_eval=None, y_eval=None, ruta_guardado="splines_cubicas.png"):
    """
    Grafica los puntos muestreados, cada tramo del spline cúbico de forma suave,
    y resalta el punto evaluado.
    """
    x = splines_info["x"]
    y = splines_info["y"]
    tramos = splines_info["tramos"]

    plt.figure(figsize=(9, 5.5))

    # Graficar cada tramo cúbico individualmente
    colores_tramos = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, tramo in enumerate(tramos):
        x_local = np.linspace(tramo["x_inicio"], tramo["x_fin"], 100)
        dx = x_local - tramo["x_inicio"]
        y_local = tramo["a"] + tramo["b"] * dx + tramo["c"] * (dx**2) + tramo["d"] * (dx**3)
        
        color = colores_tramos[idx % len(colores_tramos)]
        plt.plot(x_local, y_local, label=f"Tramo {idx + 1} [{tramo['x_inicio']:.1f}, {tramo['x_fin']:.1f}]",
                 color=color, linewidth=2)

    # Graficar puntos originales de la tabla
    plt.plot(x, y, 'ko', markersize=8, label="Puntos de la tabla", zorder=5)

    # Graficar punto evaluado
    if x_eval is not None and y_eval is not None:
        if np.isscalar(x_eval):
            plt.plot(x_eval, y_eval, '*', color='red', markersize=14,
                     markeredgecolor='black', label=f"Evaluado ($x={x_eval:.4f}$, $y={y_eval:.4f}$)", zorder=6)
            plt.annotate(f"S({x_eval:.3f}) = {y_eval:.4f}",
                         (x_eval, y_eval),
                         textcoords="offset points",
                         xytext=(15, 12),
                         bbox=dict(boxstyle="round,pad=0.3", fc="#ffff99", alpha=0.8),
                         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        else:
            plt.plot(x_eval, y_eval, '*', color='red', markersize=12,
                     markeredgecolor='black', label="Puntos evaluados", zorder=6)

    plt.title(f"Interpolación por Splines Cúbicas ({splines_info['tipo_frontera'].capitalize()})",
              fontsize=13, fontweight='bold')
    plt.xlabel("x", fontsize=11)
    plt.ylabel("S(x)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    try:
        plt.savefig(ruta_guardado, dpi=150)
        print(f"[+] Gráfico guardado exitosamente en: '{ruta_guardado}'")
    except Exception as e:
        print(f"[!] No se pudo guardar la imagen del gráfico: {e}")

    try:
        plt.show()
    except Exception:
        pass


def interpolar_splines(tabla_o_archivo, x_eval=None, tipo_frontera='natural',
                       derivadas_frontera=(0.0, 0.0), graficar=True,
                       ruta_grafico="splines_cubicas.png"):
    """
    Función principal de resolución:
    - Carga los datos de la tabla.
    - Calcula los coeficientes de las splines cúbicas.
    - Muestra las ecuaciones de cada tramo.
    - Evalúa en x_eval si se especifica.
    - Grafica los resultados.
    """
    x, y = cargar_datos(tabla_o_archivo)

    # 1. Resolver el spline cúbico
    splines_info = calcular_splines_cubicas(
        x, y,
        tipo_frontera=tipo_frontera,
        derivadas_frontera=derivadas_frontera
    )

    # 2. Imprimir polinomios de cada subintervalo
    mostrar_polinomios(splines_info["tramos"], tipo_frontera=tipo_frontera)

    # 3. Valuación en x_eval
    y_eval = None
    if x_eval is not None:
        print("=" * 80)
        print(f"EVALUACIÓN EN x = {x_eval}")
        print("=" * 80)
        if np.isscalar(x_eval):
            y_eval, idx, tramo, alerta = evaluar_spline(splines_info, x_eval)
            if "extrapolacion" in alerta:
                print(f"  [ADVERTENCIA] El valor x = {x_eval} está fuera del rango [{x[0]}, {x[-1]}].")
                print("                Se utiliza la extensión del polinomio del extremo más cercano (extrapolación).")
            print(f"  Punto evaluado: x = {x_eval}")
            print(f"  Subintervalo correspondiente: Tramo {idx + 1} [{tramo['x_inicio']:.4f}, {tramo['x_fin']:.4f}]")
            print(f"  Ecuación utilizada:")
            print(f"    S_{idx + 1}(x) = {tramo['a']:.6f} + ({tramo['b']:.6f})*(x - {tramo['x_inicio']:.4f}) + "
                  f"({tramo['c']:.6f})*(x - {tramo['x_inicio']:.4f})^2 + ({tramo['d']:.6f})*(x - {tramo['x_inicio']:.4f})^3")
            print(f"\n  >>> Resultado evaluado: S({x_eval}) = {y_eval:.8f}")
        else:
            y_eval, indices, _, alertas = evaluar_spline(splines_info, x_eval)
            for xi, yi, idx, al in zip(x_eval, y_eval, indices, alertas):
                print(f"  S({xi}) = {yi:.8f} (Tramo {idx + 1}: [{splines_info['tramos'][idx]['x_inicio']:.2f}, {splines_info['tramos'][idx]['x_fin']:.2f}])")
        print("=" * 80 + "\n")

    # 4. Gráfica
    if graficar:
        graficar_splines(splines_info, x_eval=x_eval, y_eval=y_eval, ruta_guardado=ruta_grafico)

    return {
        "splines_info": splines_info,
        "evaluacion": {"x_eval": x_eval, "y_eval": y_eval}
    }


# =============================================================================
# BLOQUE DE EJECUCIÓN DIRECTA
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolación mediante Splines Cúbicas para tablas de 2 columnas."
    )
    parser.add_argument(
        "--archivo", type=str, default=None,
        help="Ruta a un archivo .txt o .csv con 2 columnas (x, y)."
    )
    parser.add_argument(
        "--eval", type=float, default=2.5,
        help="Número a valuar en el spline obtenido (por defecto: 2.5)."
    )
    parser.add_argument(
        "--frontera", type=str, default="natural", choices=["natural", "sujeta"],
        help="Tipo de condición de frontera: 'natural' o 'sujeta' (por defecto: 'natural')."
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
                        [1.0, 5.381],
                        [1.3, 8.672],
                        [1.4, 9.592],
                        [1.45, 9.988],
                        [1.6, 10.853],
                        [1.72, 11.146],
                        [1.8, 11.136],
                        [1.93, 10.780],
                        [2.0, 10.435]
        ]
        print(">> Usando tabla de datos de ejemplo por defecto:")
        print("   x    |    y   ")
        print("   -------------")
        for fila in fuente_datos:
            print(f"  {fila[0]:4.1f}  |  {fila[1]:4.1f}")

    # Ejecutar la interpolación por splines cúbicas y valuar el número indicado
    numero_a_valuar = args.eval
    interpolar_splines(
        tabla_o_archivo=fuente_datos,
        x_eval=numero_a_valuar,
        tipo_frontera=args.frontera,
        graficar=not args.no_grafica,
        ruta_grafico="splines_cubicas.png"
    )
