"""Final diciembre 2025, problema 1. Ejecutar: python problema1.py

Los métodos usan solo la biblioteca estándar. Para los gráficos se requiere
matplotlib. Se interpreta literalmente T' = t*f(D), como figura en la foto.
Los archivos se guardan junto a este script. No requiere conexión a internet.
"""
from math import log, pi, isfinite
from pathlib import Path
import csv

L, T0, T1 = 4.0, 100.0, 20.0
TI, TF, T_INICIAL = 10.0, 20.0, 60.0
TOL = 1e-5
SALIDA = Path(__file__).resolve().parent
C = L**2 / pi**2 * log((T0 - T1) / (T_INICIAL - T1))
D_EXACTO = C / TI


def f(d):
    return C / d - TI


def df(d):
    return -C / d**2


def falsa_posicion(f, a, b, tol=1e-5, max_iter=10000):
    """Filas: iteración, raíz, cambio absoluto, residuo.

    Para este problema la tolerancia se aplica a AMBOS últimos valores.
    El primer cambio es infinito: todavía no existen dos aproximaciones.
    """
    fa, fb = f(a), f(b)
    if fa == 0:
        return a, [(0, a, 0.0, 0.0)]
    if fb == 0:
        return b, [(0, b, 0.0, 0.0)]
    if fa * fb > 0:
        raise ValueError('El intervalo debe presentar cambio de signo.')
    anterior = None
    tabla = []
    for k in range(1, max_iter + 1):
        x = (a * fb - b * fa) / (fb - fa)
        fx = f(x)
        error = float('inf') if anterior is None else abs(x - anterior)
        tabla.append((k, x, error, abs(fx)))
        if fx == 0 or (error <= tol and abs(fx) <= tol):
            return x, tabla
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
        anterior = x
    raise RuntimeError('Falsa posición no convergió.')


def newton(f, df, x0, tol=1e-5, max_iter=100):
    tabla = []
    x = x0
    for k in range(1, max_iter + 1):
        derivada = df(x)
        if derivada == 0:
            raise ValueError('Derivada nula en Newton.')
        nuevo = x - f(x) / derivada
        if not isfinite(nuevo) or nuevo <= 0:
            raise ValueError('Newton abandonó el dominio D > 0.')
        error, residuo = abs(nuevo - x), abs(f(nuevo))
        tabla.append((k, nuevo, error, residuo))
        if error <= tol and residuo <= tol:
            return nuevo, tabla
        x = nuevo
    raise RuntimeError('Newton no convergió.')


def edo(t, temperatura):
    if temperatura <= T1:
        raise ValueError('El logaritmo requiere T > T1. Revisar el paso.')
    return t * (L**2 / (D_EXACTO * pi**2)
                * log((T0 - T1) / (temperatura - T1)) - t)


def rk4(g, a, b, y0, h):
    n = round((b - a) / h)
    if abs(a + n * h - b) > 1e-10:
        raise ValueError('El paso debe dividir la longitud del intervalo.')
    tiempos, valores = [a + i * h for i in range(n + 1)], [y0]
    for i in range(n):
        t, y = tiempos[i], valores[-1]
        k1 = g(t, y)
        k2 = g(t + h / 2, y + h * k1 / 2)
        k3 = g(t + h / 2, y + h * k2 / 2)
        k4 = g(t + h, y + h * k3)
        valores.append(y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    return tiempos, valores


def guardar_csv(nombre, encabezado, filas):
    with (SALIDA / nombre).open('w', newline='', encoding='utf-8') as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado)
        escritor.writerows(filas)


def main():
    lineas = []

    def mostrar(texto=''):
        print(texto)
        lineas.append(texto)

    mostrar('FINAL DICIEMBRE 2025 - PROBLEMA 1')
    mostrar("Interpretación: T' = t * [L²/(D*pi²)*ln((T0-T1)/(T-T1)) - t]")
    mostrar(f'a) D analítico = {D_EXACTO:.10f} (interno: {D_EXACTO:.17g})')
    mostrar(f'f(0.05) = {f(0.05):.12e}; f(1.0) = {f(1):.12e}')
    mostrar('Parada: cambio absoluto <= 1e-5 Y residuo <= 1e-5.')
    for nombre, resolver in [
        ('b) Falsa posición', lambda: falsa_posicion(f, .05, 1., TOL)),
        ('c) Newton-Raphson', lambda: newton(f, df, .05, TOL)),
    ]:
        raiz, tabla = resolver()
        mostrar('\n' + nombre)
        mostrar('iter          D                 cambio absoluto       |f(D)|')
        for k, x, cambio, residuo in tabla:
            mostrar(f'{k:3d}   {x:.10f}       {cambio:.12e}   {residuo:.12e}')
        mostrar(f'Raíz: {raiz:.10f}; iteraciones: {len(tabla)}')
        mostrar(f'Error absoluto real: {abs(raiz - D_EXACTO):.12e}')
        archivo = 'falsa_posicion.csv' if nombre.startswith('b') else 'newton.csv'
        guardar_csv(archivo, ['iteracion', 'D', 'cambio_absoluto', 'residuo'], tabla)

    soluciones = []
    mostrar('\nd) RK4: se usa D analítico sin redondearlo.')
    for h in [.1, .01]:
        tiempos, valores = rk4(edo, TI, TF, T_INICIAL, h)
        j = round((15 - TI) / h)
        mostrar(f'h = {h}: pasos = {len(valores)-1}; '
                f'T(15) = {valores[j]:.10f}; T(20) = {valores[-1]:.10f}')
        etiqueta = str(h).replace('.', '_')
        guardar_csv(f'rk4_h_{etiqueta}.csv', ['t', 'T'], zip(tiempos, valores))
        soluciones.append((h, tiempos, valores))
        if h == .01:
            centrada = (valores[j + 1] - valores[j - 1]) / (2 * h)
            directa = edo(tiempos[j], valores[j])
            mostrar('\ne) Diferencia centrada de orden h²:')
            for indice in [j-1, j, j+1]:
                mostrar(f'T({tiempos[indice]:.2f}) = {valores[indice]:.14f}')
            mostrar(f'Derivada centrada = {centrada:.12f}')
            mostrar(f'EDO evaluada en T_RK4(15) = {directa:.12f}')
            mostrar(f'Diferencia absoluta = {abs(centrada - directa):.12e}')
            mostrar(f'Diferencia relativa = {abs((centrada - directa)/directa):.12e}')
            mostrar('La evaluación directa usa una T aproximada; no es una solución analítica.')

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        mostrar('\nPara generar los gráficos, instalar matplotlib y volver a ejecutar.')
    else:
        for h, tiempos, valores in soluciones:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(tiempos, valores, color='#1864ab', label=f'RK4, h={h}')
            ax.set(xlabel='Tiempo t', ylabel='Temperatura T(t)',
                   title=f'Temperatura de la barra — RK4, h={h}')
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(SALIDA / f'rk4_h_{str(h).replace(".", "_")}.png', dpi=160)
            plt.close(fig)
        mostrar('\nSe guardaron los dos gráficos en archivos separados.')
    (SALIDA / 'resultados.txt').write_text('\n'.join(lineas) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
