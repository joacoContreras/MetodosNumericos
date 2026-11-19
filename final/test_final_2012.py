"""Verificaciones: python -m unittest discover -s final -p 'test_final_2012.py'."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

import final_2012_problema_1 as final


class ProblemaUnoTests(unittest.TestCase):
    def test_sistema_del_enunciado(self):
        a, b = final.construir_sistema_calor(n=4)
        np.testing.assert_array_equal(a, [[1, 0, 0, 0], [1, -2, 1, 0],
                                          [0, 1, -2, 1], [0, 0, 1, -1]])
        np.testing.assert_array_equal(b, [0, -0.0001, -0.0001, 0])

    def test_solucion_discreta_contra_solver_directo(self):
        for n in (2, 3, 12, 100):
            with self.subTest(n=n):
                a, b = final.construir_sistema_calor(n, 0.03, 2.5, -1.0)
                j = np.arange(n)
                exacta = -1.0 + 2.5 * 0.03**2 * j * (2*n - 3 - j) / 2
                np.testing.assert_allclose(exacta, np.linalg.solve(a, b), atol=1e-12)
                np.testing.assert_allclose(a @ exacta, b, atol=1e-14)
        a, b = final.construir_sistema_calor()
        self.assertAlmostEqual(np.linalg.solve(a, b)[-1], 0.4851)

    def test_barrido_sor_con_vector_inicial_no_nulo(self):
        a, b = final.construir_sistema_calor(5, 0.1, 2.0, 3.0)
        inicial = np.array([1.0, -2.0, 0.5, 4.0, -1.0])
        esperado = inicial.copy()
        omega = 1.4
        for i in range(5):
            gs = (b[i] - a[i, :i] @ esperado[:i]
                  - a[i, i+1:] @ inicial[i+1:]) / a[i, i]
            esperado[i] = (1 - omega)*inicial[i] + omega*gs
        resultado = final.sor(a, b, omega, x0=inicial,
                              max_iterations=1, tol=1e6, criterio="incremento")
        np.testing.assert_allclose(resultado.x, esperado, atol=1e-14)
        np.testing.assert_array_equal(inicial, [1, -2, 0.5, 4, -1])
        self.assertNotEqual(resultado.x[-1], resultado.x[-2])

    def test_cota_y_residuo_contra_matriz_densa(self):
        for n in (2, 3, 15):
            with self.subTest(n=n):
                a, b = final.construir_sistema_calor(n)
                resultado = final.sor(a, b, 1.5, tol=1e-9)
                residuo = np.linalg.norm(a @ resultado.x - b, np.inf)
                error = np.linalg.norm(resultado.x - np.linalg.solve(a, b), np.inf)
                self.assertAlmostEqual(resultado.residuo, residuo, delta=1e-15)
                self.assertAlmostEqual(resultado.cota_error,
                                       (1 + n*(n-1)/2) * residuo, delta=1e-12)
                self.assertLessEqual(resultado.cota_error, 1e-9)
                self.assertLessEqual(error, resultado.cota_error + 1e-14)

    def test_matriz_general_con_coeficientes_fuera_de_las_tres_diagonales(self):
        a = np.array([[6, 1, 0, 1], [1, 7, 1, 0],
                      [0, 1, 8, 1], [1, 0, 1, 9]], dtype=float)
        esperado = np.array([1, -2, 3, -4], dtype=float)
        b = a @ esperado
        copia_a, copia_b = a.copy(), b.copy()
        resultado = final.sor(a, b, 1.2, tol=1e-10)
        np.testing.assert_allclose(resultado.x, esperado, atol=1e-10)
        np.testing.assert_array_equal(a, copia_a)
        np.testing.assert_array_equal(b, copia_b)
        self.assertLessEqual(np.max(np.abs(resultado.x - esperado)),
                             resultado.cota_error + 1e-14)

    def test_main_usa_matriz_y_vector_de_la_seccion_de_datos(self):
        # N y H quedan con los valores del final: el tamaño debe salir de A.
        a = np.array([[4, 1], [1, 3]], dtype=float)
        b = np.array([6, 7], dtype=float)
        with patch.object(final, 'A', a), patch.object(final, 'b', b), \
                patch('sys.argv', ['final_2012_problema_1.py']), \
                redirect_stdout(StringIO()) as salida:
            resultado = final.main()
        np.testing.assert_allclose(resultado.x, [1, 2], atol=1e-7)
        self.assertIn('Sistema de 2 ecuaciones', salida.getvalue())

    def test_incremento_no_equivale_a_error_real(self):
        a, b = final.construir_sistema_calor(n=12)
        resultado = final.sor(a, b, 1.0, tol=1e-7, criterio="incremento")
        self.assertLessEqual(resultado.incremento, 1e-7)
        self.assertGreater(np.max(np.abs(resultado.x - np.linalg.solve(a, b))), 1e-7)

    def test_datos_invalidos_y_falta_de_convergencia(self):
        for kwargs in ({"n": 1}, {"n": True}, {"h": 0}, {"fuente": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                final.construir_sistema_calor(**kwargs)
        a, b = final.construir_sistema_calor(4)
        for kwargs in ({"tol": -1}, {"max_iterations": 1.5},
                       {"criterio": "otro"}, {"x0": [0]},
                       {"omega": 0}, {"omega": 2}, {"omega": np.nan}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                final.sor(a, b, **kwargs)
        for matriz, terminos in (([[1, 2]], [1]), ([[0, 1], [1, 2]], [1, 2]),
                                 (np.eye(2), [1]), ([[np.nan]], [1]),
                                 ([[1, 1], [1, 1]], [1, 1])):
            with self.subTest(matriz=matriz), self.assertRaises(ValueError):
                final.sor(matriz, terminos)
        with self.assertRaises(RuntimeError):
            final.sor(a, b, 1.5, max_iterations=1)


if __name__ == "__main__":
    unittest.main()
