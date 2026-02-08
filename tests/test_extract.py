from __future__ import annotations

import unittest

from cleo_bench.extract import (
    extract_integral_latex,
    extract_reference_latex,
    strip_numeric_hints,
)


class ExtractTests(unittest.TestCase):
    def test_extract_integral_from_title(self) -> None:
        title = r"Integral $\int_0^1 x\,dx$"
        body = "No extra math"
        self.assertEqual(extract_integral_latex(title, body), r"\int_0^1 x\,dx")

    def test_extract_integral_from_body(self) -> None:
        title = "Some post"
        body = r"Need this: $$\int_0^{\infty} e^{-x}\,dx$$"
        self.assertEqual(
            extract_integral_latex(title, body), r"\int_0^{\infty} e^{-x}\,dx"
        )

    def test_extract_integral_prefers_body_over_title(self) -> None:
        title = r"Integral ${\large\int}_0^1 x\,dx$"
        body = r"Need: $$\int_0^1 x\,dx$$"
        self.assertEqual(extract_integral_latex(title, body), r"\int_0^1 x\,dx")

    def test_extract_reference_assignment(self) -> None:
        answer = r"Here $$I=\frac{\pi}{2}$$ and more."
        self.assertEqual(extract_reference_latex(answer), r"\frac{\pi}{2}")

    def test_extract_reference_from_align_prefers_non_im_form(self) -> None:
        answer = (
            r"$$\begin{align}"
            r"\int_0^1 f(x)dx &= \pi-\operatorname{Ti}_2\left(\frac12\right)\\"
            r"&= \pi-\Im\,\chi_2\left(\frac{i}{2}\right),"
            r"\end{align}$$"
        )
        out = extract_reference_latex(answer)
        self.assertIsNotNone(out)
        assert out is not None
        self.assertIn(r"\operatorname{Ti}_2", out)
        self.assertNotIn(r"\Im", out)

    def test_strip_numeric_hints(self) -> None:
        text = (
            "I need to evaluate this integral.\n"
            "Its numeric value is 0.1563713913757117012308376.\n"
            "Find a closed form."
        )
        out = strip_numeric_hints(text)
        self.assertIn("I need to evaluate", out)
        self.assertIn("Find a closed form", out)
        self.assertNotIn("numeric value", out)
        self.assertNotIn("0.156371391", out)


if __name__ == "__main__":
    unittest.main()
