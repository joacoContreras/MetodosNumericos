"""Localiza raíces desde una tabla y registra el método de bisección.

Ejemplos de ejecución desde la raíz del proyecto:
    python final/biseccion_tabla.py --decimales 5
    python final/biseccion_tabla.py --tolerancia '10^-5'
    python final/biseccion_tabla.py --intervalo 1 --raiz-exacta 0.8146223736513345

Sin una fórmula se busca la raíz del INTERPOLANTE LINEAL de la tabla.
Si se conoce la función, se puede definir FUNCION más abajo. Bisección
requiere que esa función sea continua en el intervalo elegido.
El error real solo se calcula si se proporciona una raíz de referencia.
Este archivo utiliza únicamente la biblioteca estándar de Python.
"""

import argparse
from bisect import bisect_right
from fractions import Fraction
import logging
import math
from pathlib import Path


# ======================= DATOS CONFIGURABLES =======================
# Cada fila contiene [x, f(x)]. Se ordenarán por x automáticamente.
FUENTE_DATOS = [
    [0.5, -0.716],
    [0.8, -0.103],
    [1.3, 3.419],
    [2.0, 52.598],
]
DECIMALES = 5              # Tolerancia absoluta: 10**(-DECIMALES).
RAIZ_EXACTA = None         # Con fórmula: completar si se conoce la raíz exacta.
# Sin fórmula, la raíz analítica del tramo lineal se calcula como referencia.
FUNCION = None             # None: interpolación lineal de FUENTE_DATOS.
# Ejemplo con fórmula: FUNCION = lambda x: x**2 - 2
# En ese caso se evalúa FUNCION en los x de FUENTE_DATOS para buscar intervalos.
# Para a*exp(x**2)+b: FUNCION = lambda x: a * math.exp(x**2) + b
# ===================================================================


def validar_tabla(tabla):
    """Valida y ordena pares finitos; no permite abscisas repetidas."""
    datos = []
    for fila in tabla:
        if len(fila) != 2:
            raise ValueError('Cada fila debe contener exactamente [x, f(x)].')
        x, y = map(float, fila)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Todos los datos deben ser números finitos.')
        datos.append((x, y))
    if len(datos) < 2:
        raise ValueError('Se necesitan al menos dos puntos en la tabla.')
    datos.sort()
    if any(datos[i][0] == datos[i - 1][0] for i in range(1, len(datos))):
        raise ValueError('Los valores de x deben ser distintos.')
    return datos


def evaluar(funcion, x):
    """Evita continuar si una evaluación falla o devuelve NaN o infinito."""
    try:
        valor = float(funcion(x))
    except (ArithmeticError, ValueError, TypeError) as error:
        raise ValueError(f'No se pudo evaluar la función en x={x}: {error}') from error
    if not math.isfinite(valor):
        raise ValueError(f'La función no devuelve un valor finito en x={x}.')
    return valor


def signos_opuestos(y1, y2):
    """Compara signos sin multiplicar valores que podrían desbordarse."""
    return (y1 < 0 < y2) or (y2 < 0 < y1)


def crear_interpolador_lineal(tabla):
    """Construye una función continua por tramos sin extrapolar los datos."""
    datos = validar_tabla(tabla)
    nodos = [fila[0] for fila in datos]

    def interpolar(x):
        if not nodos[0] <= x <= nodos[-1]:
            raise ValueError('El punto está fuera del dominio de la tabla.')
        i = min(bisect_right(nodos, x) - 1, len(datos) - 2)
        x0, y0 = datos[i]
        x1, y1 = datos[i + 1]
        if x == x0:
            return y0
        if x == x1:
            return y1
        proporcion = (x - x0) / (x1 - x0)
        return (1 - proporcion) * y0 + proporcion * y1

    return interpolar


def buscar_intervalos(tabla, logger=None):
    """Devuelve cambios de signo y ceros tabulados como intervalos [x, x].

    La continuidad garantiza al menos una raíz en cada cambio de signo.
    Una tabla sin cambios de signo no permite descartar otras raíces,
    por ejemplo raíces de multiplicidad par entre dos nodos.
    """
    logger = logger or logging.getLogger(__name__)
    datos = validar_tabla(tabla)
    intervalos = []
    for i, (x, y) in enumerate(datos):
        if y == 0:
            intervalos.append((x, x))
            logger.info('Cero tabulado encontrado: x=%.12g.', x)
        if i + 1 < len(datos):
            siguiente_x, siguiente_y = datos[i + 1]
            cambia = signos_opuestos(y, siguiente_y)
            logger.info(
                'Examinar [%.12g, %.12g]: f(a)=%+.8e, f(b)=%+.8e -> %s.',
                x, siguiente_x, y, siguiente_y,
                'cambio de signo' if cambia else 'sin cambio de signo',
            )
            if cambia:
                intervalos.append((x, siguiente_x))
    return intervalos


def biseccion(funcion, a, b, tolerancia=1e-5, raiz_exacta=None,
              max_iteraciones=1000, logger=None):
    """Devuelve raíz, errores, intervalos, convergencia e historial.

    Cota absoluta: max(c-a, b-c), aproximadamente (b-a)/2. Es el criterio
    de parada y acota la distancia a alguna raíz del intervalo si la
    función es continua y las evaluaciones numéricas conservan los signos.
    Error aproximado: |c_actual-c_anterior|; porcentual: ese error/|c|*100.
    Error exacto: |raiz_exacta-c|; porcentual: ese error/|raiz_exacta|*100.
    Los porcentajes con denominador cero se devuelven como None.
    """
    logger = logger or logging.getLogger(__name__)
    a, b, tolerancia = float(a), float(b), float(tolerancia)
    if not all(math.isfinite(v) for v in (a, b, tolerancia)):
        raise ValueError('Los extremos y la tolerancia deben ser finitos.')
    if a > b or tolerancia <= 0 or not math.isfinite(b - a):
        raise ValueError('Se requiere a <= b y una tolerancia positiva.')
    if not isinstance(max_iteraciones, int) or max_iteraciones < 1:
        raise ValueError('max_iteraciones debe ser un entero positivo.')
    if raiz_exacta is not None:
        raiz_exacta = float(raiz_exacta)
        if not math.isfinite(raiz_exacta) or not a <= raiz_exacta <= b:
            raise ValueError('La raíz de referencia debe ser finita y estar en [a, b].')

    inicial = (a, b)
    fa, fb = evaluar(funcion, a), evaluar(funcion, b)
    historial = []
    error_aproximado = porcentaje_aproximado = None
    iteraciones = 0
    convergio = False

    logger.info('\nBisección en [%.12g, %.12g], tolerancia absoluta %.8e.', a, b, tolerancia)
    logger.info('Error aproximado: |c_k-c_(k-1)|; porcentual: error aproximado/|c_k|*100.')
    logger.info('%4s %16s %16s %16s %14s %14s %14s %14s',
                'Iter', 'a', 'b', 'c', 'f(c)', 'Cota absoluta', 'Error aprox.', 'Error aprox.%')

    if fa == 0 or fb == 0:
        c, fc = (a, fa) if fa == 0 else (b, fb)
        a = b = c
        cota = 0.0
        convergio = True
        motivo = 'Se encontró un cero numérico en un extremo; no se requirieron iteraciones.'
    elif not signos_opuestos(fa, fb):
        raise ValueError('El intervalo no presenta un cambio de signo ni un cero en los extremos.')
    else:
        anterior = None
        motivo = 'Se alcanzó el máximo de iteraciones sin satisfacer la tolerancia.'
        for iteraciones in range(1, max_iteraciones + 1):
            c = a + (b - a) / 2
            fc = evaluar(funcion, c)
            cota = max(c - a, b - c)
            if anterior is not None:
                error_aproximado = abs(c - anterior)
                porcentaje_aproximado = error_aproximado / abs(c) * 100 if c != 0 else None

            historial.append({
                'iteracion': iteraciones, 'a': a, 'b': b, 'c': c, 'f_c': fc,
                'cota_error_absoluto': cota,
                'error_absoluto_aproximado': error_aproximado,
                'error_porcentual_aproximado': porcentaje_aproximado,
            })
            logger.info('%4d %16.10f %16.10f %16.10f %+14.6e %14.6e %14s %14s',
                        iteraciones, a, b, c, fc, cota,
                        formatear_error(error_aproximado), formatear_error(porcentaje_aproximado))

            if fc == 0:
                a = b = c
                cota = 0.0
                convergio = True
                motivo = 'Se encontró f(c)=0 en la evaluación numérica.'
                break
            if cota <= tolerancia:
                convergio = True
                motivo = 'La cota del error absoluto es menor o igual que la tolerancia.'
                break
            if c == a or c == b:
                motivo = 'La precisión de float impide seguir dividiendo el intervalo.'
                break
            # Se conserva el intervalo usado para la última aproximación.
            if iteraciones == max_iteraciones:
                break
            if signos_opuestos(fa, fc):
                b, fb = c, fc
                logger.info('     f(a) y f(c) tienen signos opuestos: se conserva [a, c].')
            else:
                a, fa = c, fc
                logger.info('     f(c) y f(b) tienen signos opuestos: se conserva [c, b].')
            anterior = c

    error_exacto = abs(raiz_exacta - c) if raiz_exacta is not None else None
    porcentaje_exacto = (
        error_exacto / abs(raiz_exacta) * 100
        if raiz_exacta is not None and raiz_exacta != 0 else None
    )
    logger.info('\nMotivo de parada: %s', motivo)
    return {
        'raiz': c, 'f_raiz': fc, 'iteraciones': iteraciones,
        'intervalo_inicial': inicial, 'intervalo_final': (a, b),
        'convergio': convergio, 'motivo': motivo,
        'cota_error_absoluto': cota,
        'error_absoluto_aproximado': error_aproximado,
        'error_porcentual_aproximado': porcentaje_aproximado,
        'error_absoluto_exacto': error_exacto,
        'error_porcentual_exacto': porcentaje_exacto,
        'historial': historial,
    }


def formatear_error(valor):
    """N/D indica primera iteración, falta de referencia o división por cero."""
    return 'N/D' if valor is None else f'{valor:.8e}'


def leer_tolerancia(texto):
    """Acepta 1e-5, 0.00001, 10^-5 y 10**-5 sin utilizar eval."""
    texto = texto.strip().replace(' ', '').replace('**', '^')
    try:
        valor = 10.0 ** int(texto[3:]) if texto.startswith('10^') else float(texto)
        if not math.isfinite(valor) or valor <= 0:
            raise ValueError
        return valor
    except (ValueError, OverflowError) as error:
        raise argparse.ArgumentTypeError('Indicá una tolerancia positiva, por ejemplo 1e-5 o 10^-5.') from error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument('--decimales', type=int, default=DECIMALES,
                           help='Usa tolerancia absoluta 10**(-d); por defecto d=5.')
    precision.add_argument('--tolerancia', type=leer_tolerancia,
                           help='Tolerancia absoluta, por ejemplo 1e-5 o 10^-5.')
    parser.add_argument('--intervalo', type=int, default=1,
                        help='Número del intervalo encontrado que se desea resolver (desde 1).')
    parser.add_argument('--raiz-exacta', type=float, default=RAIZ_EXACTA,
                        help='Raíz de referencia para calcular los errores exactos.')
    parser.add_argument('--max-iteraciones', type=int, default=1000)
    parser.add_argument('--log', type=Path, default=Path(__file__).with_suffix('.log'),
                        help='Ruta del registro; se reemplaza en cada ejecución.')
    args = parser.parse_args()
    if args.tolerancia is None and not 0 <= args.decimales <= 300:
        parser.error('--decimales debe estar entre 0 y 300.')
    tolerancia = args.tolerancia if args.tolerancia is not None else 10.0 ** (-args.decimales)

    logger = logging.getLogger('biseccion_tabla')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    manejadores = [logging.StreamHandler()]
    try:
        manejadores.append(logging.FileHandler(args.log, mode='w', encoding='utf-8'))
        for manejador in manejadores:
            manejador.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(manejador)
        logger.info('LOCALIZACIÓN DE RAÍCES DESDE UNA TABLA')
        datos = validar_tabla(FUENTE_DATOS)
        if FUNCION is None:
            funcion = crear_interpolador_lineal(datos)
            logger.info('Modelo: interpolación lineal por tramos de los datos.')
            logger.info('La tolerancia se aplica a la raíz del interpolante, no al error del modelo.')
        else:
            funcion = FUNCION
            datos = [(x, evaluar(funcion, x)) for x, _ in datos]
            logger.info('Modelo: FUNCION. Se recalculan sus valores en los nodos de la tabla.')
        logger.info('Tabla utilizada:')
        for x, y in datos:
            logger.info('  x=%16.10f | f(x)=%+.10e', x, y)

        intervalos = buscar_intervalos(datos, logger)
        if not intervalos:
            logger.info('No se detectaron cambios de signo ni ceros tabulados.')
            logger.info('Esto no demuestra que la función carezca de raíces entre los nodos.')
            return 0
        logger.info('\nIntervalos/candidatos encontrados:')
        for indice, (a, b) in enumerate(intervalos, start=1):
            logger.info('  %d: [%.12g, %.12g]', indice, a, b)
        if not 1 <= args.intervalo <= len(intervalos):
            raise ValueError(f'--intervalo debe estar entre 1 y {len(intervalos)}.')
        a, b = intervalos[args.intervalo - 1]
        logger.info('Se resuelve el candidato %d; usá --intervalo para elegir otro.', args.intervalo)
        raiz_referencia = args.raiz_exacta
        if raiz_referencia is None and FUNCION is None:
            # La raíz del tramo lineal es a - f(a)*(b-a)/(f(b)-f(a)).
            # Fraction evita redondeos intermedios al operar con los decimales
            # tabulados; al pasar a float se obtiene la referencia numérica.
            valores = dict(datos)
            if a == b:
                raiz_referencia = a
            else:
                ar, br = Fraction(str(a)), Fraction(str(b))
                far, fbr = Fraction(str(valores[a])), Fraction(str(valores[b]))
                raiz_referencia = float(ar - far * (br - ar) / (fbr - far))
            logger.info('Raíz analítica del interpolante (referencia): %.16g', raiz_referencia)
            logger.info('El error exacto informado corresponde al interpolante, no a la función original desconocida.')
        resultado = biseccion(funcion, a, b, tolerancia, raiz_referencia,
                              args.max_iteraciones, logger)

        logger.info('\nRESULTADO')
        logger.info('Convergencia: %s', 'sí' if resultado['convergio'] else 'no')
        logger.info('Raíz aproximada: %.16g', resultado['raiz'])
        logger.info('f(raíz aproximada): %+.10e', resultado['f_raiz'])
        logger.info('Intervalo final: [%.16g, %.16g]', *resultado['intervalo_final'])
        logger.info('Iteraciones empleadas: %d', resultado['iteraciones'])
        logger.info('Tolerancia absoluta solicitada: %.8e', tolerancia)
        for etiqueta, clave in [
            ('Cota del error absoluto', 'cota_error_absoluto'),
            ('Error absoluto aproximado entre iteraciones', 'error_absoluto_aproximado'),
            ('Error porcentual aproximado (%)', 'error_porcentual_aproximado'),
            ('Error absoluto exacto respecto de la referencia', 'error_absoluto_exacto'),
            ('Error porcentual exacto respecto de la referencia (%)', 'error_porcentual_exacto'),
        ]:
            logger.info('%s: %s', etiqueta, formatear_error(resultado[clave]))
        if raiz_referencia is None:
            logger.info('Error exacto N/D: proporcioná --raiz-exacta si conocés la raíz del modelo.')
        logger.info('Porcentajes N/D: no hay aproximación anterior o el denominador es cero.')
        logger.info('Registro guardado en: %s', args.log.resolve())
        return 0 if resultado['convergio'] else 1
    except (ValueError, OSError) as error:
        logger.error('ERROR: %s', error)
        return 1
    finally:
        for manejador in manejadores:
            logger.removeHandler(manejador)
            manejador.close()


if __name__ == '__main__':
    raise SystemExit(main())
