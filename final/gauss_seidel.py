"""Gauss-Seidel y SOR para sistemas lineales A*x = b.

Ejecutar: python final/gauss_seidel.py
Requiere NumPy y gauss.py, ubicado en esta misma carpeta.

Para otro ejercicio, cambiar únicamente la sección DATOS DEL PROBLEMA:
    A, b: matriz de coeficientes y términos independientes.
    X0: vector inicial; None significa comenzar con ceros.
    OMEGA: 1 para Gauss-Seidel; entre 1 y 2 para sobrerrelajación (SOR).
    MAX_ITER: cantidad de iteraciones, o máximo si se indica TOL.
    TOL: None para realizar exactamente MAX_ITER iteraciones; número positivo
         para parar cuando cambio y residuo, en norma infinito, sean <= TOL.
    INDICES_MOSTRAR: números de incógnitas desde 1; None muestra todas.

A debe ser cuadrada y tener diagonal no nula. La convergencia depende de A
y de omega; 0 < omega < 2 no la garantiza para cualquier matriz.
El cambio entre iteraciones y el residuo no son el error respecto de Gauss.
"""

import numpy as np

from gauss import gauss


def construir_sistema_problema4(n=15):
    """Arma x1 = xn = 1 y x_(i-1) - 2*x_i + x_(i+1) = 1 en el interior."""
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError('n debe ser un entero mayor o igual a 2.')
    A = (np.diag(np.full(n, -2.0))
         + np.diag(np.ones(n - 1), 1)
         + np.diag(np.ones(n - 1), -1))
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    return A, np.ones(n)


# ======================== DATOS DEL PROBLEMA ========================
A, b = construir_sistema_problema4(n=15)

# Para ingresar cualquier otro sistema, reemplazar la línea anterior por:
# A = np.array([[4, 1], [2, 3]], dtype=float)
# b = np.array([1, 2], dtype=float)
# y ajustar INDICES_MOSTRAR a [1, 2] o None.

X0 = None                       # None equivale a np.zeros(len(b)).
OMEGA = 1.75                     # Usar 1.0 para Gauss-Seidel sin relajación.
MAX_ITER = 20
TOL = None                      # Exactamente 20 iteraciones en este problema.
INDICES_MOSTRAR = [1, 5, 10, 15]  # Numeración matemática, desde 1.
MOSTRAR_MATRIZ = True
MOSTRAR_ITERACIONES = True
# ===================================================================


def gauss_seidel(A, b, x0=None, *, omega=1.0, max_iter=20, tol=None):
    """Devuelve (x, historial), sin modificar los datos de entrada.

    historial incluye la iteración 0 y, para cada barrido completo, una copia
    de x, el cambio máximo y el residuo máximo. Con tol=None no se detiene
    anticipadamente. Con tol positiva, exige cambio <= tol Y residuo <= tol.

    Actualización SOR de cada componente:
        gs = (b_i - sum(j<i, a_ij*x_j_nuevo)
                  - sum(j>i, a_ij*x_j_anterior)) / a_ii
        x_i_nuevo = (1-omega)*x_i_anterior + omega*gs
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError('A debe ser una matriz cuadrada no vacía.')
    n = len(A)
    if b.shape != (n,):
        raise ValueError('b debe contener un término independiente por fila de A.')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError('A y b deben contener números finitos.')
    if np.any(np.diag(A) == 0):
        raise ValueError('Hay ceros en la diagonal: reordenar las ecuaciones.')
    if not np.isfinite(omega) or not 0 < omega < 2:
        raise ValueError('omega debe cumplir 0 < omega < 2.')
    if (isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 1):
        raise ValueError('max_iter debe ser un entero positivo.')
    if tol is not None and (not np.isfinite(tol) or tol <= 0):
        raise ValueError('tol debe ser None o un número positivo y finito.')

    x = np.zeros(n) if x0 is None else np.array(x0, dtype=float, copy=True)
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        raise ValueError('x0 debe contener un valor inicial finito por incógnita.')

    with np.errstate(over='raise', invalid='raise', divide='raise'):
        residuo = float(np.max(np.abs(A @ x - b)))
        historial = [{'iteracion': 0, 'x': x.copy(),
                      'cambio': None, 'residuo': residuo}]
        for k in range(1, max_iter + 1):
            anterior = x.copy()
            for i in range(n):
                # Las componentes anteriores a i ya están actualizadas.
                suma_inferior = A[i, :i] @ x[:i]
                suma_superior = A[i, i + 1:] @ anterior[i + 1:]
                valor_gs = (b[i] - suma_inferior - suma_superior) / A[i, i]
                x[i] = (1 - omega) * anterior[i] + omega * valor_gs

            if not np.all(np.isfinite(x)):
                raise FloatingPointError('La iteración produjo NaN o infinito.')
            cambio = float(np.max(np.abs(x - anterior)))
            residuo = float(np.max(np.abs(A @ x - b)))
            historial.append({'iteracion': k, 'x': x.copy(),
                              'cambio': cambio, 'residuo': residuo})
            if tol is not None and cambio <= tol and residuo <= tol:
                break

    return x, historial


def error_relativo_enunciado(x_g, x_d):
    """sum|x_G - x_D| / sum|x_G|, sobre TODAS las componentes.

    G corresponde al resultado iterativo y D al método directo.
    Devuelve NaN si el denominador es cero (error relativo no definido).
    """
    x_g, x_d = np.asarray(x_g, dtype=float), np.asarray(x_d, dtype=float)
    if x_g.ndim != 1 or x_g.size == 0 or x_g.shape != x_d.shape:
        raise ValueError('Se necesitan dos vectores no vacíos de igual tamaño.')
    if not np.all(np.isfinite(x_g)) or not np.all(np.isfinite(x_d)):
        raise ValueError('Los vectores deben contener números finitos.')
    denominador = float(np.sum(np.abs(x_g)))
    if denominador == 0:
        return float('nan')
    return float(np.sum(np.abs(x_g - x_d)) / denominador)


def indices_tabla(n, indices):
    """Convierte números de incógnitas (desde 1) a índices Python (desde 0)."""
    indices = list(range(1, n + 1)) if indices is None else list(indices)
    if not indices or any(
        isinstance(i, (bool, np.bool_))
        or not isinstance(i, (int, np.integer)) or not 1 <= i <= n
        for i in indices
    ):
        raise ValueError(f'INDICES_MOSTRAR debe ser None o una lista entre 1 y {n}.')
    return [i - 1 for i in indices]


def imprimir_iteraciones(historial, indices):
    encabezado = (f'{"Iter.":>5} | '
                  + ' | '.join(f"{'x' + str(i + 1):>15}" for i in indices)
                  + f' | {"Cambio max":>13} | {"Residuo max":>13}')
    print('\nITERACIONES (cambio y residuo en norma infinito)')
    print(encabezado)
    print('-' * len(encabezado))
    for fila in historial:
        valores = ' | '.join(f'{fila["x"][i]:15.9f}' for i in indices)
        cambio = '-' if fila['cambio'] is None else f'{fila["cambio"]:.6e}'
        print(f'{fila["iteracion"]:5d} | {valores}'
              f' | {cambio:>13} | {fila["residuo"]:13.6e}')


def main():
    np.set_printoptions(precision=9, suppress=True, linewidth=140)
    indices = indices_tabla(len(b), INDICES_MOSTRAR)
    print('GAUSS-SEIDEL CON RELAJACIÓN (SOR)')
    if MOSTRAR_MATRIZ:
        print('Matriz A:')
        print(A)
        print('Vector b:', b)
    print('Vector inicial:', np.zeros(len(b)) if X0 is None else X0)
    print(f'omega = {OMEGA:g}; máximo de iteraciones = {MAX_ITER}; tolerancia = {TOL}')
    print('Se actualizan todas las componentes, incluidas x1 y xn, con la misma fórmula SOR.')

    x_g, historial = gauss_seidel(A, b, X0, omega=OMEGA, max_iter=MAX_ITER, tol=TOL)
    # Eliminación gaussiana con pivoteo parcial, implementada en gauss.py.
    x_d = gauss(A, b, mostrar=False)
    if MOSTRAR_ITERACIONES:
        imprimir_iteraciones(historial, indices)

    ultima = historial[-1]
    print(f'\nIteraciones realizadas: {ultima["iteracion"]}')
    if TOL is not None:
        convergio = ultima['cambio'] <= TOL and ultima['residuo'] <= TOL
        print('Tolerancia alcanzada.' if convergio else 'No se alcanzó la tolerancia indicada.')
    print('Vector aproximado completo:')
    print(x_g)
    print('Vector obtenido por eliminación gaussiana:')
    print(x_d)

    print('\nCOMPARACIÓN DE LAS COMPONENTES SOLICITADAS')
    encabezado = (f'{"Variable":>8} | {"Gauss-Seidel/SOR":>18}'
                  f' | {"Gauss directo":>18} | {"Error absoluto":>18}')
    print(encabezado)
    print('-' * len(encabezado))
    for i in indices:
        print(f'{"x" + str(i + 1):>8} | {x_g[i]:18.10f} | {x_d[i]:18.10f}'
              f' | {abs(x_g[i] - x_d[i]):18.10e}')

    error = error_relativo_enunciado(x_g, x_d)
    print('\nError global del enunciado: sum|x_G - x_D| / sum|x_G|')
    print(f'Calculado con las {len(b)} componentes, no solo las mostradas.')
    if np.isnan(error):
        print('Error relativo no definido: sum|x_G| = 0.')
    else:
        print(f'err = {error:.12f} = {100 * error:.8f}%')
    print(f'Residuo máximo iterativo |A*x_G - b| = {ultima["residuo"]:.6e}')
    print(f'Residuo máximo directo |A*x_D - b| = {np.max(np.abs(A @ x_d - b)):.6e}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FloatingPointError) as error:
        raise SystemExit(f'Error: {error}')
