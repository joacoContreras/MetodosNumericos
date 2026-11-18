"""Interpola una tabla con Lagrange y localiza una raíz con error controlado.

Ejecución desde la raíz del proyecto:
    python final/lagrange_tabla.py --decimales 5
    python final/lagrange_tabla.py --tolerancia '10^-5'
    python final/lagrange_tabla.py --intervalo 1 --raiz-exacta VALOR

Lagrange construye P(x) a partir de TODOS los puntos de la tabla. Luego se
detectan cambios de signo entre nodos y se aplica bisección a P(x).
La tolerancia controla la localización de una raíz del polinomio, no el
error de interpolación respecto de una función original desconocida.

Requiere SymPy y el archivo biseccion_tabla.py de esta misma carpeta.
"""

import argparse
import logging
import math
from pathlib import Path

import sympy as sym

# Permite ejecutar el archivo directamente o importarlo como final.lagrange_tabla.
if __package__:
    from .biseccion_tabla import (
        biseccion, buscar_intervalos, formatear_error, leer_tolerancia, validar_tabla,
    )
else:
    from biseccion_tabla import (
        biseccion, buscar_intervalos, formatear_error, leer_tolerancia, validar_tabla,
    )


# ======================= DATOS CONFIGURABLES =======================
FUENTE_DATOS = [
    [0.5, -0.716],
    [0.8, -0.103],
    [1.3, 3.419],
    [2.0, 52.598],
]
DECIMALES = 5       # Tolerancia absoluta: 10**(-DECIMALES).
RAIZ_EXACTA = None  # Opcional: raíz conocida del polinomio para medir el error real.
# ===================================================================


def calcular_polinomio_lagrange(tabla, logger=None):
    """Devuelve la expresión de SymPy P(x) que interpola los pares [x, y].

    P(x) = suma(y_i * L_i(x)),
    L_i(x) = producto((x-x_j)/(x_i-x_j)) para todos los j distintos de i.

    Los decimales se convierten en racionales antes de operar. Esto evita
    que el redondeo de los coeficientes altere la interpolación de los nodos.
    """
    logger = logger or logging.getLogger(__name__)
    datos = validar_tabla(tabla)
    nodos = [sym.Rational(str(x)) for x, _ in datos]
    valores = [sym.Rational(str(y)) for _, y in datos]
    x = sym.Symbol('x')
    polinomio = sym.S.Zero

    logger.info('\nCONSTRUCCIÓN DEL POLINOMIO DE LAGRANGE')
    logger.info('P(x) = suma(y_i * L_i(x)); se utilizan los %d puntos.', len(datos))
    for i, nodo in enumerate(nodos):
        base = sym.S.One
        for j, otro_nodo in enumerate(nodos):
            if i != j:
                base *= (x - otro_nodo) / (nodo - otro_nodo)
        polinomio += valores[i] * base
        logger.info('  i=%d: x_i=%s, y_i=%s, L_i(x)=%s', i, nodo, valores[i], base)

    polinomio = sym.expand(polinomio)
    logger.info('P(x) exacto para los decimales tabulados = %s', polinomio)
    logger.info('P(x) con coeficientes decimales = %s', sym.N(polinomio, 12))
    if polinomio == 0:
        logger.info('P(x) es idénticamente cero: cualquier punto del dominio es raíz.')
    return polinomio


def crear_funcion_lagrange(polinomio):
    """Convierte P(x) en una función escalar para el algoritmo de raíces.

    Evalúa mediante Horner y aritmética racional; solo el resultado final
    se convierte en float. Así se evita expandir y redondear coeficientes
    numéricos en cada iteración y se conservan los ceros tabulados.
    """
    x = sym.Symbol('x')
    coeficientes = sym.Poly(polinomio, x).all_coeffs()

    def evaluar_polinomio(valor):
        valor = float(valor)
        if not math.isfinite(valor):
            raise ValueError('El punto de evaluación debe ser finito.')
        punto = sym.Rational(str(valor))
        resultado = sym.S.Zero
        for coeficiente in coeficientes:
            resultado = resultado * punto + coeficiente
        return float(resultado)

    return evaluar_polinomio


def localizar_raiz_lagrange(tabla, tolerancia=1e-5, numero_intervalo=1,
                            raiz_exacta=None, max_iteraciones=1000, logger=None):
    """Construye Lagrange, encuentra un intervalo y localiza una raíz.

    Devuelve el diccionario de bisección con el polinomio y los intervalos
    detectados añadidos. Si no hay cambios de signo ni ceros en los nodos,
    devuelve None; eso no descarta raíces entre nodos del mismo signo.
    numero_intervalo permite elegir entre los candidatos, contando desde 1.
    """
    logger = logger or logging.getLogger(__name__)
    datos = validar_tabla(tabla)
    logger.info('Tabla utilizada:')
    for x, y in datos:
        logger.info('  x=%16.10f | y=%+.10e', x, y)
    polinomio = calcular_polinomio_lagrange(datos, logger)
    funcion = crear_funcion_lagrange(polinomio)

    logger.info('\nBÚSQUEDA DE INTERVALOS')
    logger.info('Como P(x_i)=y_i, los cambios de signo de la tabla también son de P(x).')
    intervalos = buscar_intervalos(datos, logger)
    if not intervalos:
        logger.info('No se detectaron cambios de signo ni ceros tabulados.')
        logger.info('Esto no descarta otras raíces del polinomio entre nodos del mismo signo.')
        return None
    for indice, (a, b) in enumerate(intervalos, start=1):
        logger.info('  Candidato %d: [%.12g, %.12g]', indice, a, b)
    if not 1 <= numero_intervalo <= len(intervalos):
        raise ValueError(f'El número de intervalo debe estar entre 1 y {len(intervalos)}.')

    a, b = intervalos[numero_intervalo - 1]
    logger.info('Se elige el candidato %d: [%.12g, %.12g].', numero_intervalo, a, b)
    logger.info('Lagrange es la interpolación; bisección reduce el intervalo para controlar el error.')
    logger.info('En el registro de bisección, f(c) representa P(c).')
    resultado = biseccion(funcion, a, b, tolerancia, raiz_exacta, max_iteraciones, logger)
    resultado['polinomio_lagrange'] = polinomio
    resultado['intervalos_detectados'] = intervalos
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument('--decimales', type=int, default=DECIMALES,
                           help='Tolerancia absoluta 10**(-d); por defecto d=5.')
    precision.add_argument('--tolerancia', type=leer_tolerancia,
                           help='Admite 1e-5, 0.00001 y 10^-5.')
    parser.add_argument('--intervalo', type=int, default=1,
                        help='Candidato a resolver, contando desde 1.')
    parser.add_argument('--raiz-exacta', type=float, default=RAIZ_EXACTA,
                        help='Raíz conocida del polinomio para calcular los errores exactos.')
    parser.add_argument('--max-iteraciones', type=int, default=1000)
    parser.add_argument('--log', type=Path, default=Path(__file__).with_suffix('.log'),
                        help='Archivo de registro; se reemplaza en cada ejecución.')
    args = parser.parse_args()
    if args.tolerancia is None and not 0 <= args.decimales <= 300:
        parser.error('--decimales debe estar entre 0 y 300.')
    tolerancia = args.tolerancia if args.tolerancia is not None else 10.0 ** (-args.decimales)

    logger = logging.getLogger('lagrange_tabla')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    manejadores = [logging.StreamHandler()]
    try:
        manejadores.append(logging.FileHandler(args.log, mode='w', encoding='utf-8'))
        for manejador in manejadores:
            manejador.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(manejador)
        logger.info('LOCALIZACIÓN DE RAÍCES: LAGRANGE + BISECCIÓN')
        logger.info('La tolerancia controla el error de la raíz de P(x).')
        logger.info('No incluye el error de interpolación respecto de la función original.')
        resultado = localizar_raiz_lagrange(
            FUENTE_DATOS, tolerancia, args.intervalo, args.raiz_exacta,
            args.max_iteraciones, logger,
        )
        if resultado is None:
            logger.info('Registro guardado en: %s', args.log.resolve())
            return 0

        logger.info('\nRESULTADO')
        logger.info('Convergencia: %s', 'sí' if resultado['convergio'] else 'no')
        logger.info('Raíz aproximada de P(x): %.16g', resultado['raiz'])
        logger.info('P(raíz aproximada): %+.10e', resultado['f_raiz'])
        logger.info('Intervalo inicial: [%.16g, %.16g]', *resultado['intervalo_inicial'])
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
        if args.raiz_exacta is None:
            logger.info('Error exacto N/D: se requiere una raíz de referencia (--raiz-exacta).')
        logger.info('Error porcentual aproximado = |c_k-c_(k-1)|/|c_k| * 100.')
        logger.info('N/D indica falta de referencia, primera iteración o denominador cero.')
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
