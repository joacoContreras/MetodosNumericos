"""Final del 19/12/2012 — Problema 2: Gauss, regresión y bisección.

Ejecutar: python final/final_2012_problema_2.py
Requiere NumPy.

CÓMO FUNCIONA
1. Para cada velocidad v se calcula D(v) = integral de integrando(x, v).
2. Con los pares (v, D(v)) se ajusta un polinomio por mínimos cuadrados.
3. Para cada límite se resuelve P(v) - límite = 0 mediante bisección.
   Las raíces corresponden al polinomio ajustado, no a la integral original.

CÓMO REUTILIZARLO
- Cambiar intervalos, velocidades, grado, límites y tolerancia en DATOS.
- Si cambia la fórmula de la integral, editar integrando(x, v) allí mismo.
- Para usar una tabla ya dada, llamar ajustar_polinomio(v, datos, grado):
  no es necesario calcular una integral para reutilizar esa función.
- La bisección requiere valores de signos opuestos en los extremos
  (o una raíz exactamente en un extremo). Revisar el intervalo si falla.
"""

import math

import numpy as np


# ======================== DATOS DEL PROBLEMA ========================
# Integral: D(v) = integral entre A_INTEGRAL y B_INTEGRAL de f(x, v) dx.
A_INTEGRAL = 0.0
B_INTEGRAL = 5.0
PUNTOS_GAUSS = 3                 # Número de nodos de Gauss-Legendre.

# Fórmula del enunciado: f(x,v) = 10 * exp(0.05*v*x**0.02).
FACTOR = 10.0
COEF_EXPONENCIAL = 0.05
POTENCIA_X = 0.02


def integrando(x, v):
    """EDITAR AQUÍ si otro ejercicio tiene una función diferente.

    x es la variable de integración; v es el parámetro que queda fijo
    durante cada integral. Por ejemplo, para integrar v*x**2 + 3:
        return v * x**2 + 3
    """
    return FACTOR * math.exp(COEF_EXPONENCIAL * v * x**POTENCIA_X)


# np.linspace(inicio, fin, cantidad) incluye ambos extremos.
# Los datos actuales son v = 0, 2.5, 5, ..., 50.
# También se puede escribir VELOCIDADES = np.array([0, 10, 20, 30, 40, 50]).
VELOCIDADES = np.linspace(0.0, 50.0, 21)
GRADO_POLINOMIO = 4              # Grado 4 significa 5 coeficientes.
LIMITES_CARGA = (500.0, 550.0)    # Se busca una velocidad por cada límite.

# Intervalo en el que se buscan las raíces de P(v) - límite.
# Conviene mantenerlo dentro del rango de las velocidades usadas en el ajuste.
INTERVALO_BISECCION = (0.0, 50.0)
TOL_BISECCION = 1e-10            # Tolerancia en el ancho del intervalo de v.
MAX_ITER_BISECCION = 1000
# ===================================================================


def cuadratura_gauss(f, a, b, puntos=3):
    """Aproxima la integral de cualquier f(x) en [a,b].

    Los nodos xi y pesos wi de Gauss-Legendre están definidos en [-1,1].
    Cambio de variable:
        x = centro + semiancho*xi
        dx = semiancho*dxi
    Por eso la aproximación es semiancho * sum(wi * f(xi_transformado)).
    Los nodos y pesos dependen de puntos, no de la función a integrar.
    """
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        raise ValueError("La integral requiere extremos finitos con a < b.")
    if (isinstance(puntos, (bool, np.bool_))
            or not isinstance(puntos, (int, np.integer)) or puntos < 1):
        raise ValueError("PUNTOS_GAUSS debe ser un entero positivo.")
    nodos, pesos = np.polynomial.legendre.leggauss(puntos)
    centro, semiancho = (a + b)/2, (b - a)/2
    valores = np.array([f(centro + semiancho*xi) for xi in nodos], dtype=float)
    if not np.all(np.isfinite(valores)):
        raise ValueError("El integrando produjo valores no finitos.")
    return float(semiancho * (pesos @ valores))


def carga_gauss(v):
    """Calcula una carga usando exactamente la fórmula y los datos de arriba.

    La función lambda deja v fija; cuadratura_gauss solo varía x.
    """
    return cuadratura_gauss(lambda x: integrando(x, v),
                            A_INTEGRAL, B_INTEGRAL, PUNTOS_GAUSS)


def ajustar_polinomio(v, cargas, grado):
    """Devuelve (coeficientes, centro, escala) de un ajuste por mínimos cuadrados.

    Se usa z=(v-centro)/escala para llevar el rango de v a [-1,1] y evitar
    potencias muy grandes. La matriz tiene columnas [1, z, z², ..., z**grado].
    np.linalg.lstsq busca los coeficientes que minimizan sum(D_i-P(v_i))².
    Es una regresión: el polinomio no tiene que pasar por todos los datos.

    IMPORTANTE: los coeficientes corresponden a potencias de z, no de v.
    Para evaluar el ajuste, usar evaluar_polinomio con el centro y la escala.
    """
    v, cargas = np.asarray(v, dtype=float), np.asarray(cargas, dtype=float)
    if v.ndim != 1 or v.size < 2 or cargas.shape != v.shape:
        raise ValueError("Se necesitan dos vectores del mismo tamaño, con al menos 2 datos.")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(cargas)):
        raise ValueError("Las velocidades y cargas deben ser finitas.")
    if (isinstance(grado, (bool, np.bool_))
            or not isinstance(grado, (int, np.integer)) or grado < 0):
        raise ValueError("El grado debe ser un entero mayor o igual a cero.")
    if np.unique(v).size < max(2, grado + 1):
        raise ValueError("Se necesitan al menos grado+1 velocidades distintas y un rango no nulo.")
    centro = float((v.min() + v.max())/2)
    escala = float((v.max() - v.min())/2)
    z = (v - centro)/escala
    matriz = np.column_stack([z**potencia for potencia in range(grado + 1)])
    coeficientes = np.linalg.lstsq(matriz, cargas, rcond=None)[0]
    return coeficientes, centro, escala


def evaluar_polinomio(v, coeficientes, centro, escala):
    """Evalúa c0 + c1*z + c2*z² + ...; sirve para un número o un vector."""
    z = (np.asarray(v, dtype=float) - centro)/escala
    return np.polynomial.polynomial.polyval(z, coeficientes)


def biseccion(f, a, b, tol=1e-10, max_iter=1000):
    """Resuelve f(v)=0 reduciendo a la mitad un intervalo con cambio de signo.

    Se conserva el subintervalo que sigue encerrando un cambio de signo.
    Para cuando b-a <= tol y devuelve su punto medio: el error en v es
    como máximo tol/2, salvo redondeo. Esta tol no mide el error del ajuste.
    """
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        raise ValueError("La bisección requiere extremos finitos con a < b.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("La tolerancia de bisección debe ser positiva y finita.")
    if (isinstance(max_iter, (bool, np.bool_))
            or not isinstance(max_iter, (int, np.integer)) or max_iter < 1):
        raise ValueError("El máximo de iteraciones debe ser un entero positivo.")
    fa, fb = float(f(a)), float(f(b))
    if not np.isfinite(fa) or not np.isfinite(fb):
        raise ValueError("La función produjo valores no finitos en los extremos.")
    if fa == 0:
        return a
    if fb == 0:
        return b
    if (fa > 0) == (fb > 0):
        raise ValueError(f"Sin cambio de signo en [{a:g}, {b:g}]: "
                         "revisar INTERVALO_BISECCION y el límite buscado.")
    for _ in range(max_iter):
        medio = (a + b)/2
        if b - a <= tol:
            return medio
        if medio == a or medio == b:
            raise RuntimeError("La tolerancia es menor que la precisión numérica disponible.")
        fm = float(f(medio))
        if not np.isfinite(fm):
            raise ValueError("La función produjo un valor no finito durante la bisección.")
        if fm == 0:
            return medio
        if (fa > 0) != (fm > 0):
            b = medio
        else:
            a, fa = medio, fm
    raise RuntimeError("La bisección alcanzó el máximo de iteraciones.")


def pendiente_minima(coeficientes, centro, escala, a, b):
    """Mínimo de dP/dv en [a,b], para comprobar si P es creciente.

    dP/dv = (dP/dz)/escala. Sus mínimos pueden estar en los extremos
    o en puntos interiores donde la segunda derivada se anula.
    """
    derivada = np.polynomial.polynomial.polyder(coeficientes)
    segunda = np.polynomial.polynomial.polyder(derivada)
    za, zb = (a - centro)/escala, (b - centro)/escala
    candidatos = [za, zb]
    for raiz in np.polynomial.polynomial.polyroots(segunda):
        if abs(raiz.imag) < 1e-10 and za < raiz.real < zb:
            candidatos.append(raiz.real)
    return float(np.min(np.polynomial.polynomial.polyval(candidatos, derivada))/escala)


def problema_2():
    print("\nPROBLEMA 2 | Integral, ajuste polinomial y búsqueda de velocidades")
    velocidades = np.asarray(VELOCIDADES, dtype=float)
    cargas = np.array([carga_gauss(v) for v in velocidades])
    coeficientes, centro, escala = ajustar_polinomio(velocidades, cargas, GRADO_POLINOMIO)

    def polinomio(v):
        return evaluar_polinomio(v, coeficientes, centro, escala)

    print(f"Integral en [{A_INTEGRAL:g}, {B_INTEGRAL:g}]; Gauss con {PUNTOS_GAUSS} puntos.")
    print(f"Ajuste de grado {GRADO_POLINOMIO} con {len(velocidades)} datos.")
    print(f"\n{'v (m/s)':>10} | {'D por Gauss':>14} | {'P(v)':>14} | {'D - P(v)':>13}")
    print("-" * 62)
    for v, carga in zip(velocidades, cargas):
        print(f"{v:10.4f} | {carga:14.6f} | {polinomio(v):14.6f} | {carga-polinomio(v):13.3e}")

    print(f"\nVariable del ajuste: z = (v - {centro:g}) / {escala:g}")
    print("P(v) = c0 + c1*z + c2*z² + ...")
    for i, coeficiente in enumerate(coeficientes):
        print(f"c{i} = {coeficiente:.12g}")
    print("\nRaíces del polinomio ajustado:")
    a, b = INTERVALO_BISECCION
    raices = []
    for limite in LIMITES_CARGA:
        raiz = biseccion(lambda v: polinomio(v) - limite, a, b,
                          tol=TOL_BISECCION, max_iter=MAX_ITER_BISECCION)
        raices.append(raiz)
        print(f"Límite {limite:g}: v = {raiz:.9f} m/s; P(v) = {polinomio(raiz):.9f}")
    minima = pendiente_minima(coeficientes, centro, escala, a, b)
    print(f"\nMínimo de dP/dv en [{a:g}, {b:g}]: {minima:.6f}")
    if minima > 0:
        print("El ajuste es creciente en ese intervalo: cada límite tiene a lo sumo una raíz.")
    return coeficientes, centro, escala, np.array(raices)


if __name__ == "__main__":
    try:
        problema_2()
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        raise SystemExit(f"Error: {error}")
