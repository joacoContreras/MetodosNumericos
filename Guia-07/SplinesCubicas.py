import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13])
y = np.array([2.51, 3.30, 4.04, 4.70, 5.22, 5.54, 5.78, 5.40, 5.57, 5.70, 5.8, 5.90])
n = len(x) - 1
h = np.zeros(n)
for i in range(0, n):
    h[i] = x[i+1] - x[i]

u = np.zeros(n-1)
l = np.zeros(n-1)
for i in range(0, n-1):
    u[i] = h[i] / (h[i] + h[i+1])
    l[i] = 1 - u[i]

d = np.zeros(n-1)
for i in range(1, n-1):
    d[i] = 3 * (l[i-1] * (y[i+2] - y[i+1]) / h[i+1] + u[i-1] * (y[i+1] - y[i]) / h[i])

d[0] = 3 * (l[0] * (y[1] - y[0]) / h[0] + u[0] * (y[2] - y[1]) / h[1]) - l[0] * 0.8
d[n-2] = 3 * (l[n-2] * (y[n-1] - y[n-2]) / h[n-2] + u[n-2] * (y[n] - y[n-1]) / h[n-1]) - l[n-2] * 0.2

A = np.zeros([n-1, n-1])
for i in range(1, n-2):
    A[i, i-1] = l[i-1]
    A[i, i] = 2
    A[i, i+1] = u[i]

A[0, 0] = A[n-2, n-2] = 2
A[0, 1] = u[0]
A[n-2, n-3] = l[n-2]

M0 = np.linalg.solve(A, d)
M = np.zeros(n+1)
for i in range(1, n):
    M[i] = M0[i-1]
M[0] = 0.8
M[n] = 0.2

# Dibujo
for i in range(0, n):
    x0 = np.linspace(x[i], x[i+1], 100)
    y0 = y[i] * ((x0 - x[i+1])**2) * (h[i] + 2 * (x0 - x[i])) / (h[i]**3) + \
         y[i+1] * ((x0 - x[i])**2) * (h[i] + 2 * (x[i+1] - x0)) / (h[i]**3) + \
         M[i] * ((x0 - x[i+1])**2) * (x0 - x[i]) / (h[i]**2) + \
         M[i+1] * ((x0 - x[i])**2) * (x0 - x[i+1]) / (h[i]**2)
    plt.plot(x0, y0, color='red')

plt.plot(x, y, marker='+', mec='r', mfc='w', label="Primitivo")
plt.title("Interpolación de splines cúbicos de la curva ")
plt.legend()
plt.show()
