import math

def falsa_posicion_tabla(f, a, b, tol=1e-6, max_iter=100):
    """
    Método de Falsa Posición (Regula Falsi) con tabla de iteraciones.
    Devuelve la raíz aproximada y la tabla de iteraciones.
    """
    if f(a) * f(b) >= 0:
        raise ValueError("El intervalo no es válido: f(a) y f(b) deben tener signos opuestos")

    tabla = []
    c_prev = None

    for i in range(1, max_iter+1):
        # Fórmula de la falsa posición
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        fc = f(c)

        # Error estimado: |c - c_prev|, si no es la primera iteración
        if c_prev is None:
            error = None
        else:
            error = abs(c - c_prev)

        # Guardar fila de la tabla
        tabla.append({
            "Iter": i,
            "a": a,
            "b": b,
            "c": c,
            "f(c)": fc,
            "Error": error
        })

        # Comprobar tolerancia
        if abs(fc) < tol:
            break

        # Actualizar extremos según regla de signos
        if f(a) * fc < 0:
            b = c
        else:
            a = c

        c_prev = c

    return c, tabla

# Función a resolver
f = lambda x: math.log(x**2 + 1) - math.sin(x)

# Ejecutar método
raiz, tabla = falsa_posicion_tabla(f, 1, 2)

# Mostrar tabla
print(f"{'Iter':>4} | {'a':>10} | {'b':>10} | {'c':>10} | {'f(c)':>12} | {'Error':>12}")
print("-" * 65)
for fila in tabla:
    error_str = f"{fila['Error']:.6f}" if fila['Error'] is not None else "-"
    print(f"{fila['Iter']:>4} | {fila['a']:>10.6f} | {fila['b']:>10.6f} | {fila['c']:>10.6f} | {fila['f(c)']:>12.6f} | {error_str:>12}")

print(f"\nRaíz aproximada: {raiz:.6f} encontrada en {len(tabla)} iteraciones")