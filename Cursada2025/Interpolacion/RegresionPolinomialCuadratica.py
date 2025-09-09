import numpy as np

# =======================
# DATOS
# =======================
x = np.array([0.2, 0.6, 1.3, 1.4, 1.8, 2.0])
y = np.array([-0.94, -0.26, 2.35, 2.94, 5.45, 7.20])

# =======================
# (a) REGRESIÓN POLINOMIAL CUADRÁTICA
# =======================
# Ajuste con polyfit de grado 2
coef = np.polyfit(x, y, 2)   # devuelve [a, b, c]
a, b, c = coef
print("(a) Coeficientes regresión cuadrática:")
print(f"a = {a}, b = {b}, c = {c}")

# Polinomio
P = np.poly1d(coef)

# Coeficiente de correlación R
y_pred = P(x)
SS_res = np.sum((y - y_pred)**2)
SS_tot = np.sum((y - np.mean(y))**2)
R2 = 1 - SS_res/SS_tot
R = np.sqrt(R2)
print(f"Coeficiente de correlación R = {R}")