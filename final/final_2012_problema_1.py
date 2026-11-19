"""Final del 19/12/2012 — Problema 1: Gauss-Seidel con relajación (SOR).

Ejecutar: python final/final_2012_problema_1.py
Requiere NumPy. Para otro ejercicio, editar A y b en DATOS DEL PROBLEMA.
Cada fila de A es una ecuación; b contiene sus términos independientes.
"""

import argparse
from dataclasses import dataclass

import numpy as np


def construir_sistema_calor(n=100, h=0.01, fuente=1.0,
                            temperatura_izquierda=0.0):
    """Construye la matriz del enunciado sin escribir sus 100 filas a mano.

    Primera fila: T_1 = temperatura_izquierda.
    Filas interiores: T_(i-1) - 2*T_i + T_(i+1) = -fuente*h².
    Última fila: T_(n-1) - T_n = 0.
    """
    if (isinstance(n, (bool, np.bool_))
            or not isinstance(n, (int, np.integer)) or n < 2):
        raise ValueError("n debe ser un entero mayor o igual a 2.")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h debe ser positivo y finito.")
    if not np.isfinite(fuente) or not np.isfinite(temperatura_izquierda):
        raise ValueError("La fuente y la temperatura deben ser finitas.")

    matriz = np.zeros((n, n))
    terminos = np.zeros(n)
    matriz[0, 0] = 1.0
    terminos[0] = temperatura_izquierda
    for i in range(1, n - 1):
        matriz[i, i - 1:i + 2] = [1.0, -2.0, 1.0]
        terminos[i] = -fuente * h**2
    matriz[-1, -2:] = [1.0, -1.0]
    return matriz, terminos


# ======================== DATOS DEL PROBLEMA ========================
# CASO ACTUAL: matriz de 100 x 100 del final de 2012, con h = 0.01.
N = 100
H = 0.01
A, b = construir_sistema_calor(n=N, h=H)

# PARA OTRO EJERCICIO: reemplazar la línea "A, b = ..." por tu matriz
# y tu vector. Por ejemplo, para resolver:
#     4*x1 + x2 = 6
#      x1 + 3*x2 = 7
# escribir (quitando los #):
# A = np.array([
#     [4, 1],
#     [1, 3],
# ], dtype=float)
# b = np.array([6, 7], dtype=float)
# El tamaño del sistema se toma de A; no hace falta cambiar N ni H.
# También podés modificar elementos sueltos: A[0, 0] = 4; b[0] = 6.
# Los índices de Python empiezan en 0: A[0, 1] es fila 1, columna 2.

X0 = None                       # None: comenzar con un vector de ceros.
TOL = 1e-7
MAX_ITER = 100_000
CRITERIO = "cota"                # "cota" o "incremento".
OMEGAS = [i / 10 for i in range(10, 20)]  # 1.0 de referencia; 1.1 a 1.9 SOR.
MOSTRAR_MATRIZ = False           # Los sistemas de hasta 10 filas se muestran siempre.
TABLA_COMPLETA = False
# ===================================================================


@dataclass
class ResultadoSOR:
    x: np.ndarray
    iteraciones: int
    incremento: float
    residuo: float
    cota_error: float


def validar_sistema(a, b):
    """Comprueba las dimensiones y los valores necesarios para aplicar SOR."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("A debe ser una matriz cuadrada no vacía.")
    if b.shape != (len(a),):
        raise ValueError("b debe contener un término independiente por fila de A.")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("A y b deben contener valores finitos.")
    if np.any(np.diag(a) == 0):
        raise ValueError("La diagonal de A tiene ceros: reordenar las ecuaciones.")
    return a, b


def sor(a, b, omega=1.0, tol=1e-7, max_iterations=100_000, *,
        x0=None, criterio="cota"):
    """Resuelve A*x=b usando los coeficientes de la matriz recibida.

    omega=1: Gauss-Seidel; 1<omega<2: sobrerrelajación.
    La convergencia depende de A y de omega; no se garantiza para toda matriz.

    Fórmula por fila:
        gs = (b_i - sum(j != i, a_ij*x_j)) / a_ii
        x_i_nuevo = x_i_anterior + omega*(gs - x_i_anterior)
    Al recorrer las filas en orden, x_j ya es nuevo cuando j<i y sigue
    siendo el anterior cuando j>i. Se relajan todas las filas.

    Criterios absolutos de parada, en norma infinito:
      cota: ||A^-1||inf * ||A*x-b||inf <= tol (predeterminado).
      incremento: ||x_nuevo-x_anterior||inf <= tol.
    La cota limita el error real, salvo redondeo. El incremento solo mide
    cuánto cambió la aproximación. La inversa se usa para calcular la cota;
    las aproximaciones se obtienen mediante las iteraciones SOR.
    """
    a, b = validar_sistema(a, b)
    n = len(b)
    if not np.isfinite(omega) or not 0 < omega < 2:
        raise ValueError("omega debe cumplir 0 < omega < 2.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol debe ser positiva y finita.")
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 1):
        raise ValueError("max_iterations debe ser un entero positivo.")
    if criterio not in ("cota", "incremento"):
        raise ValueError("criterio debe ser 'cota' o 'incremento'.")
    inicial = np.zeros(n) if x0 is None else np.array(x0, dtype=float, copy=True)
    if inicial.shape != (n,) or not np.all(np.isfinite(inicial)):
        raise ValueError("x0 debe contener un valor inicial finito por incógnita.")
    norma_inversa = float(np.linalg.norm(np.linalg.inv(a), np.inf))

    # Se omiten los coeficientes cero para acelerar la matriz grande del final.
    # Esto también funciona si ingresás una matriz llena o de otro tamaño.
    filas = [[(int(j), float(a[i, j])) for j in np.flatnonzero(a[i]) if j != i]
             for i in range(n)]
    diagonal = np.diag(a).tolist()
    terminos = b.tolist()
    x = inicial.tolist()
    filas_no_cero, columnas_no_cero = np.nonzero(a)
    coeficientes = a[filas_no_cero, columnas_no_cero]

    for iteration in range(1, max_iterations + 1):
        incremento = 0.0
        for i in range(n):
            suma = sum(coef * x[j] for j, coef in filas[i])
            gs = (terminos[i] - suma) / diagonal[i]
            cambio = omega * (gs - x[i])
            x[i] += cambio
            incremento = max(incremento, abs(cambio))

        vector = np.array(x)
        if not np.all(np.isfinite(vector)):
            raise FloatingPointError("SOR produjo valores no finitos.")
        # Equivale a A @ vector - b, sumando solo coeficientes no nulos.
        ax = np.bincount(filas_no_cero,
                         weights=coeficientes * vector[columnas_no_cero], minlength=n)
        residuo = float(np.max(np.abs(ax - b)))
        cota = norma_inversa * residuo
        if not np.isfinite(cota):
            raise FloatingPointError("La cota del error produjo un valor no finito.")
        medida = cota if criterio == "cota" else incremento
        if medida <= tol:
            return ResultadoSOR(vector, iteration, incremento, residuo, cota)

    raise RuntimeError(f"SOR no convergió con omega={omega:g} en "
                       f"{max_iterations} iteraciones: {criterio}={medida:.3e}.")


def problema_1(a, b, tol=TOL, max_iterations=MAX_ITER, criterio=CRITERIO,
               tabla_completa=TABLA_COMPLETA, x0=None):
    a, b = validar_sistema(a, b)
    n = len(b)
    referencia = np.linalg.solve(a, b)
    print("\nGAUSS-SEIDEL CON SOBRERRELAJACIÓN (SOR) | A*x = b")
    print(f"Sistema de {n} ecuaciones; tolerancia absoluta = {tol:.1e}")
    if MOSTRAR_MATRIZ or n <= 10:
        print("Matriz A:\n", a)
        print("Vector b:", b)
    print("Vector inicial:", "ceros" if x0 is None else x0)
    print(f"Criterio de parada: {criterio}")
    print("Referencia de comparación: resolución directa de la misma matriz A.")
    encabezado = (f"{'omega':>5} | {'Iteraciones':>11} | {'Incremento':>12}"
                  f" | {'Residuo':>12} | {'Cota error':>12} | {'Error vs ref.':>13}")
    print("\n" + encabezado)
    print("-" * len(encabezado))
    resultados = []
    for omega in OMEGAS:
        try:
            run = sor(a, b, omega, tol, max_iterations, x0=x0, criterio=criterio)
        except RuntimeError:
            print(f"{omega:5.1f} | sin convergencia en {max_iterations} iteraciones")
            continue
        resultados.append((omega, run))
        error = float(np.max(np.abs(run.x - referencia)))
        print(f"{omega:5.1f} | {run.iteraciones:11d} | {run.incremento:12.3e}"
              f" | {run.residuo:12.3e} | {run.cota_error:12.3e} | {error:13.3e}")

    candidatos = [(w, run) for w, run in resultados if w > 1]
    if not candidatos:
        raise RuntimeError("Ningún omega de sobrerrelajación alcanzó la tolerancia.")
    omega, mejor = min(candidatos, key=lambda item: (item[1].iteraciones, item[0]))
    print(f"\nMejor omega de sobrerrelajación convergente: {omega:.1f} "
          f"({mejor.iteraciones} barridos; criterio: {criterio}).")
    if len(candidatos) < sum(w > 1 for w in OMEGAS):
        print("Búsqueda incompleta: aumentar MAX_ITER para comparar todos los candidatos.")
    print("En caso de empate se elige el menor omega.")
    error = float(np.max(np.abs(mejor.x - referencia)))
    print(f"Error máximo frente a la referencia directa: {error:.6e}")
    print(f"Error frente a referencia <= tolerancia: {'SI' if error <= tol else 'NO'}")
    if criterio == "incremento" and error > tol:
        print("El incremento cumple la tolerancia; el error frente a referencia aún la supera.")
        print("Usar CRITERIO = 'cota' para exigir también la cota del error.")

    indices = (range(n) if tabla_completa or n <= 10 else
               sorted({0, 1, (n - 1)//4, (n - 1)//2, 3*(n - 1)//4, n - 2, n - 1}))
    print("\nSOLUCIÓN (todas las incógnitas si TABLA_COMPLETA = True)")
    print(f"{'Variable':>8} | {'SOR':>14} | {'Referencia':>14} | {'Error absoluto':>14}")
    print("-" * 61)
    for i in indices:
        print(f"{'x' + str(i + 1):>8} | {mejor.x[i]:14.10f}"
              f" | {referencia[i]:14.10f} | {abs(mejor.x[i] - referencia[i]):14.3e}")
    return mejor


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--max-iter", type=int, default=MAX_ITER)
    parser.add_argument("--criterio", choices=("cota", "incremento"), default=CRITERIO)
    parser.add_argument("--tabla-completa", action="store_true", default=TABLA_COMPLETA)
    # Solo si se indican estas opciones se reconstruye la matriz de la barra.
    # Sin ellas, se usan exactamente A y b de DATOS DEL PROBLEMA.
    parser.add_argument("--n", type=int, default=None, help="Reconstruir la barra con n nodos.")
    parser.add_argument("--h", type=float, default=None, help="Reconstruir la barra con paso h.")
    args = parser.parse_args()
    try:
        matriz, terminos = A, b
        if args.n is not None or args.h is not None:
            matriz, terminos = construir_sistema_calor(
                n=N if args.n is None else args.n, h=H if args.h is None else args.h)
        return problema_1(matriz, terminos, tol=args.tol, max_iterations=args.max_iter,
                          criterio=args.criterio, tabla_completa=args.tabla_completa, x0=X0)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
