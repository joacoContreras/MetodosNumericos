import math

# Parámetros del problema
a = 0.0  # Límite inferior del intervalo
b = 1.0  # Límite superior del intervalo
delta = 1e-4  # Tolerancia δ

# ==============================
# ANOTACIONES TEÓRICAS
# ==============================
# Objetivo:
# Encontrar el número mínimo de bisecciones n que garantiza que el punto medio
# c_n es una aproximación a una raíz (r) de la función con un error menor que
# un valor prefijado δ.
#
# Desigualdad de la Bisección:
# En una bisección, se cumple que:
# |r - c_n| ≤ |b - a| / 2^n
#
# Aquí:
# - r es la raíz real.
# - c_n es el punto medio en la n-ésima iteración.
# - b y a son los extremos del intervalo.
# - n es el número de iteraciones o bisecciones.
#
# Queremos que el error |r - c_n| sea menor o igual a una tolerancia δ.
# Por lo tanto, partimos de la siguiente desigualdad:
#
# |r - c_n| ≤ |b - a| / 2^n ≤ δ
#
# Resolviendo para n:
# 1. |b - a| / 2^n ≤ δ
# 2. |b - a| ≤ δ * 2^n
# 3. |b - a| / δ ≤ 2^n
# 4. n ≥ log2(|b - a| / δ)
#
# Entonces, el número mínimo de bisecciones n necesarias es:
# n = ceil(log2(|b - a| / δ))

# ==============================
# CÁLCULO DEL NÚMERO MÍNIMO DE BISECCIONES
# ==============================
# Calculando el número mínimo de bisecciones n que garantiza que el punto medio
# está dentro de la tolerancia δ respecto a la raíz r.

n = math.ceil(math.log2(abs(b - a) / delta))

# ==============================
# RESULTADO
# ==============================
print(f"El número mínimo de bisecciones necesarias para alcanzar una tolerancia δ = {delta} es: {n}")

# ==============================
# EXPLICACIÓN DEL RESULTADO
# ==============================
# Este resultado indica que se necesitan al menos {n} iteraciones de bisección
# para asegurar que el punto medio c_n esté dentro de la tolerancia δ
# respecto a la raíz r de la función.
