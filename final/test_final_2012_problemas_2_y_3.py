"""Pruebas: python -m unittest discover -s final -p 'test_final_2012*.py'."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

import numpy as np

import final_2012_problema_2 as p2
import final_2012_problema_3 as p3


class ProblemaDosTests(unittest.TestCase):
    def test_cambio_de_variable_de_gauss(self):
        # Dos puntos integran exactamente un polinomio de grado tres.
        valor = p2.cuadratura_gauss(lambda x: x**3, -1, 2, puntos=2)
        self.assertAlmostEqual(valor, 15/4, places=13)

    def test_ajuste_y_derivada_con_otro_rango(self):
        v = np.array([10, 12, 18, 30], dtype=float)
        datos = 3 + 2*v - 0.5*v**2
        coef, centro, escala = p2.ajustar_polinomio(v, datos, 2)
        np.testing.assert_allclose(p2.evaluar_polinomio(v, coef, centro, escala),
                                   datos, atol=1e-12)
        self.assertAlmostEqual(p2.pendiente_minima(coef, centro, escala, 10, 30),
                               -28, places=11)
        self.assertEqual(p2.pendiente_minima([3], 0, 1, 0, 1), 0)
        self.assertEqual(p2.pendiente_minima([3, 4], 0, 2, 0, 1), 2)

    def test_seccion_de_datos_modifica_toda_la_resolucion(self):
        datos = dict(A_INTEGRAL=0.0, B_INTEGRAL=1.0, PUNTOS_GAUSS=2,
                     integrando=lambda x, v: v**2 + x,
                     VELOCIDADES=np.linspace(0, 4, 5), GRADO_POLINOMIO=2,
                     LIMITES_CARGA=(4, 9), INTERVALO_BISECCION=(0, 4))
        with patch.multiple(p2, **datos), redirect_stdout(StringIO()):
            coef, centro, escala, raices = p2.problema_2()
        np.testing.assert_allclose(raices, np.sqrt([3.5, 8.5]), atol=1e-9)
        self.assertEqual(len(coef), 3)
        self.assertEqual((centro, escala), (2, 2))

    def test_raices_del_final_original(self):
        with redirect_stdout(StringIO()):
            *_, raices = p2.problema_2()
        np.testing.assert_allclose(raices, [45.412392301, 47.297734881],
                                   rtol=0, atol=1e-8)

    def test_biseccion_y_errores_de_entrada(self):
        for modulo in (p2, p3):
            with self.subTest(modulo=modulo.__name__):
                self.assertAlmostEqual(modulo.biseccion(lambda x: x*x - 2, 0, 2),
                                       np.sqrt(2), places=9)
                self.assertEqual(modulo.biseccion(lambda x: x, 0, 1), 0)
                with self.assertRaises(ValueError):
                    modulo.biseccion(lambda x: x*x + 1, 0, 2)
                with self.assertRaises(ValueError):
                    modulo.biseccion(lambda x: x, 0, 2, tol=0)
        with self.assertRaises(ValueError):
            p2.ajustar_polinomio([1, 2], [3, 4], 2)
        with self.assertRaises(ValueError):
            p2.cuadratura_gauss(lambda x: x, 0, 1, puntos=0)


class ProblemaTresTests(unittest.TestCase):
    def test_rk4_usa_tiempo_y_no_modifica_el_estado(self):
        inicial = np.array([3.0])
        siguiente = p3.rk4_step(lambda t, y: np.array([t]), 2, inicial, 0.5)
        np.testing.assert_allclose(siguiente, [4.125], atol=1e-14)
        np.testing.assert_array_equal(inicial, [3.0])
        esperado = 1 - 0.2 + 0.2**2/2 - 0.2**3/6 + 0.2**4/24
        np.testing.assert_allclose(
            p3.rk4_step(lambda t, y: -2*y, 0, np.array([1.0]), 0.1),
            [esperado], atol=1e-14)

    def test_constantes_fisicas_y_signo_del_arrastre(self):
        with patch.multiple(p3, MASA=100.0, AREA=10.0, GRAVEDAD=12.0,
                            COEF_ARRASTRE_X=0.5, COEF_ARRASTRE_Y=0.25), \
                patch.object(p3, 'densidad_aire', return_value=2.0):
            derivadas = p3.parachutist_rhs(0, [0, 10, 10, -4])
        np.testing.assert_allclose(derivadas, [10, -4, -5, -11.6], atol=1e-14)

    def test_impacto_con_otro_suelo_y_tiempo_inicial(self):
        inicial = np.array([1.0, 50.0, 3.0, 0.0])
        with patch.multiple(p3, AREA=0.0, GRAVEDAD=10.0,
                            ESTADO_INICIAL=inicial, TIEMPO_INICIAL=2.0,
                            TIEMPO_MAXIMO=10.0, ALTURA_SUELO=5.0):
            tiempo, impacto = p3.ground_impact(0.7)
        # Caída libre: 50 - 10*tau²/2 = 5 -> tau=3; x=1+3*tau.
        self.assertAlmostEqual(tiempo, 5.0, places=9)
        np.testing.assert_allclose(impacto, [10, 5, 3, -30], rtol=0, atol=2e-9)
        np.testing.assert_array_equal(inicial, [1, 50, 3, 0])

    def test_no_integra_mas_alla_del_tiempo_maximo(self):
        tiempos = []

        def derivadas(t, estado):
            tiempos.append(t)
            return np.zeros(4)

        with patch.multiple(p3, TIEMPO_INICIAL=2.0, TIEMPO_MAXIMO=2.25,
                            parachutist_rhs=derivadas):
            with self.assertRaises(RuntimeError):
                p3.ground_impact(0.4)
        self.assertAlmostEqual(max(tiempos), 2.25)

    def test_impacto_del_final_original(self):
        tiempo, impacto = p3.ground_impact(0.05)
        self.assertAlmostEqual(tiempo, 353.245736044, places=8)
        self.assertAlmostEqual(impacto[0], 516.611678760, places=8)
        self.assertLess(abs(impacto[1]), 1e-8)

    def test_estado_en_el_suelo_y_datos_invalidos(self):
        with patch.object(p3, 'ESTADO_INICIAL', np.array([2, 0, 3, 0])):
            tiempo, impacto = p3.ground_impact(0.1)
        self.assertEqual(tiempo, p3.TIEMPO_INICIAL)
        np.testing.assert_array_equal(impacto, [2, 0, 3, 0])
        for dt in (0, -1, np.nan):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                p3.ground_impact(dt)
        with self.assertRaises(ValueError):
            p3.rk4_step(lambda t, y: [1, 2], 0, np.array([1.0]), 0.1)


if __name__ == '__main__':
    unittest.main()
