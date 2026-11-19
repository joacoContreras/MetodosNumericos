"""Final del 19/12/2012 — Problema 3: trayectoria mediante Runge-Kutta 4.

Ejecutar: python final/final_2012_problema_3.py
Requiere NumPy.

CÓMO FUNCIONA
1. El estado contiene [posición horizontal, altura, velocidad x, velocidad y].
2. parachutist_rhs(t, estado) calcula las cuatro derivadas de ese estado.
3. rk4_step combina cuatro evaluaciones para avanzar un paso de tiempo.
4. Si la altura cruza el suelo, la bisección localiza el impacto dentro
   de ese último paso. Se comparan varios pasos para revisar la convergencia.

CÓMO REUTILIZARLO
- Para otra caída: editar constantes físicas, ESTADO_INICIAL y PASOS en DATOS.
- Para otro modelo de resistencia o densidad: editar las funciones de DATOS.
- Para otra ecuación diferencial: reutilizar rk4_step(f, t, estado, dt).
  Por ejemplo, para y'=-2*y con y(0)=1:
      siguiente = rk4_step(lambda t, y: -2*y, 0.0, np.array([1.0]), 0.1)
  f debe devolver una derivada por cada componente del estado.
- ground_impact es específico de una trayectoria: interpreta la componente 1
  del vector como altura y busca el primer cruce descendente observado.
"""

import math

import numpy as np


# ======================== DATOS DEL PROBLEMA ========================
# Unidades: metros, segundos y kilogramos. Altura y velocidad vertical
# son positivas hacia arriba; por eso la gravedad aparece con signo menos.
MASA = 80.0                     # kg
AREA = 30.0                     # m²
GRAVEDAD = 9.8                   # m/s²
COEF_ARRASTRE_X = 0.005
COEF_ARRASTRE_Y = 1.0

# Modelo atmosférico del enunciado:
# rho(h) = RHO_REFERENCIA * ((TEMPERATURA_REFERENCIA - GRADIENTE*h)
#                          / TEMPERATURA_REFERENCIA)**EXPONENTE_DENSIDAD
RHO_REFERENCIA = 1.225           # kg/m³
TEMPERATURA_REFERENCIA = 300.0   # K
GRADIENTE_TEMPERATURA = 0.0065   # K/m
EXPONENTE_DENSIDAD = -0.15

# Orden OBLIGATORIO: [x inicial, altura inicial, vx inicial, vy inicial].
# Ejemplo de otra caída: ESTADO_INICIAL = np.array([0, 100, 5, 0], dtype=float)
ESTADO_INICIAL = np.array([0.0, 2300.0, 2.0, 0.0])
TIEMPO_INICIAL = 0.0             # s
TIEMPO_MAXIMO = 1000.0           # Instante final permitido, en s.
ALTURA_SUELO = 0.0               # m
PASOS = (0.2, 0.1, 0.05)        # Se resuelve de nuevo con cada dt, en s.
TOL_IMPACTO = 1e-10              # Ancho del intervalo de TIEMPO de la bisección.
# TOL_IMPACTO no es una tolerancia en altura ni controla el error global de RK4.
# Para estudiar ese error, comparar los resultados al reducir PASOS.
# Para simular sin resistencia del aire: poner ambos COEF_ARRASTRE en 0.


def densidad_aire(altura):
    """EDITAR AQUÍ para cambiar la dependencia de la densidad con la altura.

    Para densidad constante, reemplazar el cuerpo por:
        return RHO_REFERENCIA
    La potencia actual necesita una base estrictamente positiva.
    """
    base = (TEMPERATURA_REFERENCIA - GRADIENTE_TEMPERATURA*altura) / TEMPERATURA_REFERENCIA
    if base <= 0:
        raise ValueError("La altura está fuera del dominio del modelo de densidad.")
    return RHO_REFERENCIA * base**EXPONENTE_DENSIDAD


def parachutist_rhs(t, estado):
    """EDITAR AQUÍ si cambian las ecuaciones de movimiento.

    Recibe el tiempo y el estado; devuelve [dx/dt, dh/dt, dvx/dt, dvy/dt].
    Este modelo no depende explícitamente de t, pero se recibe para poder
    agregar después fuerzas que varíen con el tiempo.

    dx/dt = vx; dh/dt = vy
    dvx/dt = -rho*AREA/(2*MASA) * Cx * |vx|*vx
    dvy/dt = -GRAVEDAD - rho*AREA/(2*MASA) * Cy * |vy|*vy

    |v|*v conserva el signo de v: con el signo menos, el arrastre se opone
    al movimiento tanto al subir como al bajar.
    """
    x, altura, vx, vy = estado
    factor = densidad_aire(altura) * AREA / (2*MASA)
    ax = -factor * COEF_ARRASTRE_X * abs(vx) * vx
    ay = -GRAVEDAD - factor * COEF_ARRASTRE_Y * abs(vy) * vy
    return np.array([vx, vy, ax, ay])
# ===================================================================


def rk4_step(f, t, estado, dt):
    """Avanza un paso de un sistema y'=f(t,y), sin modificar el estado recibido.

    k1: pendiente al principio.
    k2: pendiente a mitad de paso, prediciendo con k1.
    k3: otra pendiente a mitad de paso, ahora prediciendo con k2.
    k4: pendiente al final, prediciendo con k3.
    El promedio ponderado (k1+2*k2+2*k3+k4)/6 produce el avance de RK4.
    Aquí las k son DERIVADAS; el factor dt se aplica al construir los estados
    intermedios y al realizar el avance final.
    """
    estado = np.asarray(estado, dtype=float)
    if estado.ndim != 1 or estado.size == 0 or not np.all(np.isfinite(estado)):
        raise ValueError("El estado debe ser un vector no vacío de valores finitos.")
    if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0:
        raise ValueError("El tiempo debe ser finito y dt positivo y finito.")

    def derivada(instante, vector):
        valor = np.asarray(f(instante, vector), dtype=float)
        if valor.shape != estado.shape or not np.all(np.isfinite(valor)):
            raise ValueError("f debe devolver una derivada finita por componente del estado.")
        return valor

    k1 = derivada(t, estado)
    k2 = derivada(t + dt/2, estado + dt*k1/2)
    k3 = derivada(t + dt/2, estado + dt*k2/2)
    k4 = derivada(t + dt, estado + dt*k3)
    siguiente = estado + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    if not np.all(np.isfinite(siguiente)):
        raise FloatingPointError("RK4 produjo un estado no finito; revisar el modelo y dt.")
    return siguiente


def biseccion(f, a, b, tol=1e-10):
    """Busca f(tau)=0 en un intervalo de duraciones [a,b].

    Se conserva la mitad que encierra el cambio de signo hasta que el ancho
    es <= tol. El error de localización en el tiempo es entonces <= tol/2,
    salvo redondeo y el error de la trayectoria aproximada mediante RK4.
    """
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("TOL_IMPACTO debe ser positiva y finita.")
    if not np.isfinite(a) or not np.isfinite(b) or a >= b:
        raise ValueError("La bisección necesita extremos finitos con a < b.")
    fa, fb = float(f(a)), float(f(b))
    if not np.isfinite(fa) or not np.isfinite(fb):
        raise ValueError("La función de impacto produjo valores no finitos.")
    if fa == 0:
        return a
    if fb == 0:
        return b
    if (fa > 0) == (fb > 0):
        raise ValueError("El intervalo debe encerrar un cambio de signo.")
    for _ in range(1000):
        medio = (a + b)/2
        if b - a <= tol:
            return medio
        if medio == a or medio == b:
            raise RuntimeError("La tolerancia es menor que la precisión numérica disponible.")
        fm = float(f(medio))
        if not np.isfinite(fm):
            raise ValueError("La función de impacto produjo un valor no finito.")
        if fm == 0:
            return medio
        if (fa > 0) != (fm > 0):
            b = medio
        else:
            a, fa = medio, fm
    raise RuntimeError("La bisección del impacto alcanzó el máximo de iteraciones.")


def ground_impact(dt):
    """Devuelve (tiempo de impacto, estado de impacto) con los datos de arriba.

    Cada llamada parte otra vez de ESTADO_INICIAL. No modifica ese vector.
    Se integra mientras la altura sea mayor que ALTURA_SUELO. Cuando un paso
    la deja debajo, se busca una duración tau entre 0 y dt que haga altura
    igual al suelo, avanzando desde el estado ANTERIOR al cruce.
    El último paso se recorta si llegaría más allá de TIEMPO_MAXIMO.
    """
    estado = np.array(ESTADO_INICIAL, dtype=float, copy=True)
    if estado.shape != (4,) or not np.all(np.isfinite(estado)):
        raise ValueError("ESTADO_INICIAL debe contener [x, altura, vx, vy] finitos.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Cada paso de PASOS debe ser positivo y finito.")
    if (not np.isfinite(TIEMPO_INICIAL) or not np.isfinite(TIEMPO_MAXIMO)
            or TIEMPO_MAXIMO <= TIEMPO_INICIAL):
        raise ValueError("TIEMPO_MAXIMO debe ser finito y mayor que TIEMPO_INICIAL.")
    if not np.isfinite(ALTURA_SUELO) or not np.isfinite(TOL_IMPACTO) or TOL_IMPACTO <= 0:
        raise ValueError("ALTURA_SUELO debe ser finita y TOL_IMPACTO positiva y finita.")
    parametros = [MASA, AREA, GRAVEDAD, COEF_ARRASTRE_X, COEF_ARRASTRE_Y,
                  RHO_REFERENCIA, TEMPERATURA_REFERENCIA,
                  GRADIENTE_TEMPERATURA, EXPONENTE_DENSIDAD]
    if not np.all(np.isfinite(parametros)) or MASA <= 0 or AREA < 0 or TEMPERATURA_REFERENCIA <= 0:
        raise ValueError("Revisar constantes físicas: deben ser finitas, con masa y temperatura > 0 y área >= 0.")
    if estado[1] < ALTURA_SUELO:
        raise ValueError("La altura inicial no puede estar debajo del suelo.")
    if estado[1] == ALTURA_SUELO:
        return TIEMPO_INICIAL, estado

    cantidad = math.ceil((TIEMPO_MAXIMO - TIEMPO_INICIAL)/dt)
    for paso in range(cantidad):
        tiempo = TIEMPO_INICIAL + paso*dt
        duracion = min(dt, TIEMPO_MAXIMO - tiempo)
        if duracion <= 0:
            break
        siguiente = rk4_step(parachutist_rhs, tiempo, estado, duracion)
        if siguiente[1] <= ALTURA_SUELO:
            def altura_sobre_suelo(parcial):
                # En tau=0 se usa el estado actual, sin realizar un paso.
                if parcial == 0:
                    return estado[1] - ALTURA_SUELO
                return rk4_step(parachutist_rhs, tiempo, estado, parcial)[1] - ALTURA_SUELO

            tau = biseccion(altura_sobre_suelo, 0.0, duracion, tol=TOL_IMPACTO)
            impacto = estado.copy() if tau == 0 else rk4_step(parachutist_rhs, tiempo, estado, tau)
            return tiempo + tau, impacto
        estado = siguiente
    raise RuntimeError(f"No se observó un cruce con el suelo antes de t={TIEMPO_MAXIMO:g} s.")


def problema_3():
    print("\nPROBLEMA 3 | Trayectoria mediante Runge-Kutta de cuarto orden")
    print(f"Estado inicial [x, altura, vx, vy]: {ESTADO_INICIAL}")
    print(f"Masa = {MASA:g} kg; área = {AREA:g} m²; gravedad = {GRAVEDAD:g} m/s².")
    print(f"Suelo = {ALTURA_SUELO:g} m; intervalo de tiempo = [{TIEMPO_INICIAL:g}, {TIEMPO_MAXIMO:g}] s.")
    print(f"Tolerancia de localización del impacto en tiempo: {TOL_IMPACTO:.1e} s.")
    print(f"\n{'dt (s)':>8} | {'t impacto (s)':>15} | {'x impacto (m)':>15}"
          f" | {'altura - suelo':>14} | {'vx (m/s)':>12} | {'vy (m/s)':>12}")
    print("-" * 94)
    resultados = []
    for dt in PASOS:
        tiempo, impacto = ground_impact(dt)
        resultados.append((dt, tiempo, impacto))
        print(f"{dt:8.4f} | {tiempo:15.9f} | {impacto[0]:15.9f}"
              f" | {impacto[1]-ALTURA_SUELO:14.3e} | {impacto[2]:12.6f} | {impacto[3]:12.6f}")
    print("\nComparar los resultados al disminuir dt para revisar la convergencia.")
    return resultados


if __name__ == "__main__":
    try:
        problema_3()
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        raise SystemExit(f"Error: {error}")
