# Volver a ejecutar el código sin dependencias de descarga

import numpy as np
from scipy import stats

# Datos de la Mecha B (sin el outlier 6.15)
mecha_b = np.array([5.96, 5.99, 6.02, 6.01, 6.02, 5.95, 5.99, 5.99, 6.01, 5.97, 
                    5.98, 5.96, 5.98, 5.95, 5.97, 5.97, 5.95, 5.98, 5.98])

# Datos de la Mecha A (Promedio y Desviación)
mean_a = 6.01
std_a = 0.03
var_a = std_a ** 2

# Calcular la varianza de la Mecha B
var_b = np.var(mecha_b, ddof=1)

# Prueba F: comparación de varianzas
f_statistic = var_a / var_b

# Grados de libertad
df_a = 16 - 1  # Mecha A tiene 16 muestras
df_b = len(mecha_b) - 1

# Obtener el valor p para la prueba F
p_value = 2 * min(stats.f.cdf(f_statistic, df_a, df_b), 1 - stats.f.cdf(f_statistic, df_a, df_b))

f_statistic, p_value
