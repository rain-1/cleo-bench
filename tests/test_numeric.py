from __future__ import annotations

import unittest

import mpmath as mp

from cleo_bench.numeric import abs_rel_close, evaluate_expression_numeric, evaluate_integral_numeric


class NumericTests(unittest.TestCase):
    def test_evaluate_expression(self) -> None:
        out = evaluate_expression_numeric(r"\frac{1}{2}")
        self.assertIsNone(out.error)
        self.assertIsNotNone(out.value)
        assert out.value is not None
        self.assertAlmostEqual(float(out.value), 0.5, places=8)

    def test_evaluate_integral(self) -> None:
        out = evaluate_integral_numeric(r"\int_0^1 x\,dx")
        self.assertIsNone(out.error)
        self.assertIsNotNone(out.value)
        assert out.value is not None
        self.assertAlmostEqual(float(out.value), 0.5, places=7)

    def test_abs_rel_threshold(self) -> None:
        a = mp.mpf("1.0000004")
        b = mp.mpf("1.0")
        delta, passed = abs_rel_close(a, b, tolerance=1e-6)
        self.assertIsNotNone(delta)
        self.assertTrue(passed)

        a2 = mp.mpf("1.01")
        delta2, passed2 = abs_rel_close(a2, b, tolerance=1e-6)
        self.assertIsNotNone(delta2)
        self.assertFalse(passed2)

    def test_expression_with_pi_constant_symbol(self) -> None:
        out = evaluate_expression_numeric(
            r"\frac{\Gamma\left(\frac14\right)^2}{4\,\sqrt{2\,\pi}}\left(\ln2-\pi\right)"
        )
        self.assertIsNone(out.error)
        self.assertIsNotNone(out.value)

    def test_integral_with_large_prefix(self) -> None:
        out = evaluate_integral_numeric(
            r"{\large\int}_0^{\pi/2}\arctan^2\left(\frac{\sin x}{\sqrt3+\cos x}\right)dx"
        )
        self.assertIsNone(out.error)
        self.assertIsNotNone(out.value)

    def test_integral_with_hypergeometric_notation(self) -> None:
        out = evaluate_integral_numeric(
            r"\int_0^1{_2F_1}\left(-\frac{1}{4},\frac{5}{4};\,1;\,\frac{x}{2}\right)^2dx"
        )
        self.assertIsNone(out.error)
        self.assertIsNotNone(out.value)

    def test_challenging_known_closed_form_integral(self) -> None:
        # Non-trivial but standard benchmark integral:
        # \int_0^\infty ln(1+x^2)/(1+x^2) dx = \pi ln 2
        integral_out = evaluate_integral_numeric(
            r"\int_0^\infty \frac{\ln(1+x^2)}{1+x^2}\,dx"
        )
        expr_out = evaluate_expression_numeric(r"\pi\ln 2")

        self.assertIsNone(integral_out.error)
        self.assertIsNone(expr_out.error)
        self.assertIsNotNone(integral_out.value)
        self.assertIsNotNone(expr_out.value)

        delta, passed = abs_rel_close(integral_out.value, expr_out.value, tolerance=1e-8)
        self.assertIsNotNone(delta)
        self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()
