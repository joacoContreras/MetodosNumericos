import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Datos proporcionados en la guía
Re = [0.1, 1, 10, 100, 1000, 10000]  # Número de Reynolds
CD = [24.5, 5.5, 1.0, 0.4, 0.3, 0.2]  # Coeficiente de arrastre (CD)

# Convertir a escala logarítmica
log_Re = np.log10(Re)
log_CD = np.log10(CD)

# Crear la interpolación spline cúbica natural en la escala logarítmica
spline = CubicSpline(log_Re, log_CD, bc_type='natural')

# Valores de Re a interpolar
Re_interpolados = [5, 50, 500, 5000]
log_Re_interpolados = np.log10(Re_interpolados)

# Calcular los valores interpolados de log_CD
log_CD_interpolados = spline(log_Re_interpolados)

# Volver a la escala original (exponencial)
CD_interpolados = 10**log_CD_interpolados

# Mostrar los resultados de la interpolación
for i in range(len(Re_interpolados)):
    print(f"Para Re = {Re_interpolados[i]}, CD interpolado = {CD_interpolados[i]:.4f}")

# Graficar
plt.plot(Re, CD, 'o', label="Datos originales")
plt.plot(Re_interpolados, CD_interpolados, 's', label="Interpolaciones")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Reynolds (Re)')
plt.ylabel('Coeficiente de arrastre (CD)')
plt.title('Interpolación spline cúbica en escala logarítmica')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
