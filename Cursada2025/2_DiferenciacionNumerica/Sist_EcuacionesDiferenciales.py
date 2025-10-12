import numpy as np
from typing import List, Callable

# --- MÉTODOS NUMÉRICOS GENÉRICOS ---

def resolver_sistema_euler(
    funciones: List[Callable], 
    x0: float, 
    y0_vec: List[float], 
    x_final: float, 
    n_pasos: int
) -> tuple[np.ndarray, List[np.ndarray]]:
    """
    Resuelve un sistema de EDOs de primer orden usando el método de Euler.
    """
    num_ecuaciones = len(funciones)
    h = (x_final - x0) / n_pasos

    x = np.linspace(x0, x_final, n_pasos + 1)
    y_resultados = [np.zeros(n_pasos + 1) for _ in range(num_ecuaciones)]

    # Asignar valores iniciales
    for i in range(num_ecuaciones):
        y_resultados[i][0] = y0_vec[i]

    # Bucle del método de Euler
    for i in range(n_pasos):
        y_actual = [y[i] for y in y_resultados]
        for j in range(num_ecuaciones):
            y_resultados[j][i + 1] = y_resultados[j][i] + h * funciones[j](x[i], *y_actual)
            
    return x, y_resultados

def resolver_sistema_rk4(
    funciones: List[Callable], 
    x0: float, 
    y0_vec: List[float], 
    x_final: float, 
    n_pasos: int
) -> tuple[np.ndarray, List[np.ndarray]]:
    """
    Resuelve un sistema de EDOs de primer orden usando el método de Runge-Kutta de 4to orden.
    """
    num_ecuaciones = len(funciones)
    h = (x_final - x0) / n_pasos

    x = np.linspace(x0, x_final, n_pasos + 1)
    y_resultados = [np.zeros(n_pasos + 1) for _ in range(num_ecuaciones)]

    for i in range(num_ecuaciones):
        y_resultados[i][0] = y0_vec[i]

    for i in range(n_pasos):
        y_actual = [y[i] for y in y_resultados]
        k1 = np.array([f(x[i], *y_actual) for f in funciones])
        
        y_k2 = [y_actual[j] + 0.5 * h * k1[j] for j in range(num_ecuaciones)]
        k2 = np.array([f(x[i] + 0.5 * h, *y_k2) for f in funciones])
        
        y_k3 = [y_actual[j] + 0.5 * h * k2[j] for j in range(num_ecuaciones)]
        k3 = np.array([f(x[i] + 0.5 * h, *y_k3) for f in funciones])
        
        y_k4 = [y_actual[j] + h * k3[j] for j in range(num_ecuaciones)]
        k4 = np.array([f(x[i] + h, *y_k4) for f in funciones])
        
        for j in range(num_ecuaciones):
            y_resultados[j][i + 1] = y_resultados[j][i] + (h / 6.0) * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j])

    return x, y_resultados

# --- FUNCIÓN AUXILIAR PARA GUARDAR RESULTADOS ---

def guardar_resultados(nombre_archivo: str, encabezados: List[str], x: np.ndarray, y_lista: List[np.ndarray]):
    """
    Guarda los resultados de la simulación en un archivo de texto.
    """
    with open(nombre_archivo, "w") as archivo:
        archivo.write("\t".join(encabezados) + "\n")
        for i in range(len(x)):
            linea = f"{x[i]:.5f}"
            for y_res in y_lista:
                linea += f"\t{y_res[i]:.5f}"
            archivo.write(linea + "\n")
    print(f"Resultados guardados en '{nombre_archivo}'.")

# --- BLOQUE PRINCIPAL DE EJECUCIÓN ---

if __name__ == "__main__":
    # 1. Definir el sistema de ecuaciones a resolver
    def f1(x, y1, y2):
        return y2  # dy1/dx = y2

    def f2(x, y1, y2):
        return -y1 # dy2/dx = -y1

    # 2. Establecer parámetros y condiciones iniciales
    funciones_sistema = [f1, f2]
    condiciones_iniciales = [1.0, 0.0]  # y1(0)=1, y2(0)=0
    x_inicial, x_final = 0.0, 2.0
    pasos = 10

    # 3. Resolver con el método de Euler y guardar
    x_euler, y_euler = resolver_sistema_euler(funciones_sistema, x_inicial, condiciones_iniciales, x_final, pasos)
    guardar_resultados("resultados_euler_optimizado.txt", ["x", "y1", "y2"], x_euler, y_euler)

    # 4. Resolver con el método de RK4 y guardar
    x_rk4, y_rk4 = resolver_sistema_rk4(funciones_sistema, x_inicial, condiciones_iniciales, x_final, pasos)
    guardar_resultados("resultados_rk4_optimizado.txt", ["x", "y1", "y2"], x_rk4, y_rk4)