"""Numeric parsing and evaluation helpers for LaTeX integrals/expressions."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Callable

import mpmath as mp
import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.latex import parse_latex

from .constants import DEFAULT_DPS, DEFAULT_TOLERANCE


@dataclass
class EvalOutcome:
    value: mp.mpf | None
    value_str: str | None
    error: str | None


KNOWN_SYMBOL_CONSTANTS: dict[str, sp.Expr] = {
    "pi": sp.pi,
    "Pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "i": sp.I,
    "I": sp.I,
    "gamma": sp.EulerGamma,
    "G": sp.Catalan,
}

FORMAT_COMMAND_RE = re.compile(
    r"\\(?:large|Large|LARGE|huge|Huge|displaystyle|textstyle|scriptstyle|scriptscriptstyle|big|Big|bigg|Bigg)\b"
)
ENV_RE = re.compile(r"\\(?:begin|end)\{[A-Za-z*]+\}")
TAG_RE = re.compile(r"\\tag(?:\{[^{}]*\}|[0-9]+)")
HSPACE_RE = re.compile(r"\\hspace\*?\{[^{}]*\}")


def _fmt(value: mp.mpf, dps: int) -> str:
    return mp.nstr(value, n=dps)


def _replace_hypergeometric_notation(s: str) -> str:
    # {_2F_1}(a,b;c;z) -> \H_{21}(a,b,c,z), later mapped to sympy.hyper.
    s = re.sub(r"\{?\s*_(\d+)F_(\d+)\s*\}?", r"\\H_{\1\2}", s)
    s = s.replace(";", ",")
    return s


def _strip_outer_assignment(s: str, must_contain: str | None = None) -> str:
    if "=" not in s:
        return s
    parts = [part.strip() for part in s.split("=") if part.strip()]
    if len(parts) <= 1:
        return s
    if must_contain:
        for part in parts[1:]:
            if must_contain in part:
                return part
        for part in parts:
            if must_contain in part:
                return part
    return parts[-1]


def _normalize_latex(latex: str) -> str:
    s = (latex or "").strip()
    s = s.strip("$")

    s = s.replace("\n", " ")
    s = s.replace("&", " ")
    s = s.replace("\\left", "")
    s = s.replace("\\right", "")
    s = s.replace("\\middle|", "|")
    s = s.replace("\\!", "")
    s = s.replace("\\,", " ")
    s = s.replace("\\;", " ")
    s = s.replace("\\:", " ")
    s = s.replace("\\quad", " ")
    s = s.replace("\\qquad", " ")

    s = s.replace("\\\\+", " + ")
    s = s.replace("\\\\-", " - ")
    s = s.replace("\\\\", " ")

    s = FORMAT_COMMAND_RE.sub(" ", s)
    s = HSPACE_RE.sub(" ", s)
    s = ENV_RE.sub(" ", s)
    s = TAG_RE.sub(" ", s)
    s = re.sub(r"\\vphantom\{[^{}]*\}", "", s)
    s = re.sub(r"\\text\{([A-Za-z]+)\}", r"\\\1", s)

    s = s.replace("\\tfrac", "\\frac")
    s = s.replace("\\dfrac", "\\frac")

    # Keep operator names as plain function names.
    s = re.sub(r"\\operatorname\{([A-Za-z]+)_([0-9]+)\}\s*([A-Za-z])", r"\\\1_{\2}(\3)", s)
    s = re.sub(
        r"\\operatorname\{([A-Za-z]+)\}_\{?([0-9]+)\}?\s*([A-Za-z])",
        r"\\\1_{\2}(\3)",
        s,
    )
    s = re.sub(r"\\operatorname\{([A-Za-z]+)\}\s*([A-Za-z])", r"\\\1(\2)", s)
    s = re.sub(r"\\operatorname\{([A-Za-z]+)_([0-9]+)\}", r"\\\1_{\2}", s)
    s = re.sub(r"\\operatorname\{([A-Za-z]+)\}_\{?([0-9]+)\}?", r"\\\1_{\2}", s)
    s = re.sub(r"\\operatorname\{([A-Za-z]+)\}", r"\\\1", s)

    # Common missing-brace shortcuts.
    s = re.sub(r"\\frac\\([A-Za-z]+)\{", r"\\frac{\\\1}{", s)
    s = re.sub(r"\\frac\\([A-Za-z]+)([0-9])", r"\\frac{\\\1}{\2}", s)
    s = re.sub(r"\\frac([0-9])([0-9])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac([0-9])([A-Za-z])", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\sqrt\[([0-9]+)\]([0-9A-Za-z])", r"\\sqrt[\1]{\2}", s)
    s = re.sub(r"\\sqrt([0-9A-Za-z])", r"\\sqrt{\1}", s)
    s = re.sub(r"\\([A-Za-z]+)\^([0-9]+)", r"\\\1^{\2}", s)

    # Log/ln with implicit arguments.
    s = re.sub(r"\\ln\^\{?([0-9]+)\}?\s*([0-9A-Za-z]+)", r"\\ln^{\1}(\2)", s)
    s = re.sub(r"\\(ln|log)\s*([0-9]+(?:\.[0-9]+)?)", r"\\\1(\2)", s)

    # Wrap a few special functions when they are written with implicit argument.
    for fn in ("Ci", "Si", "Ei", "erf", "arccot", "F"):
        s = re.sub(rf"(\\{fn}(?:_\{{?[0-9]+\}}?)?)\s+([A-Za-z])", r"\1(\2)", s)
    s = re.sub(r"\\Ci([A-Za-z])", r"\\Ci(\1)", s)
    s = re.sub(r"\\Si([A-Za-z])", r"\\Si(\1)", s)
    s = re.sub(r"\\Ei([A-Za-z])", r"\\Ei(\1)", s)

    # Normalize differential token.
    s = re.sub(r"\\mathrm\s*\{\s*d\s*\}\s*([A-Za-z])", r"d\1", s)
    s = re.sub(r"\\mathrm\s*d\s*([A-Za-z])", r"d\1", s)
    s = re.sub(r"\\rm\s*d\s*([A-Za-z])", r"d\1", s)
    s = re.sub(r"\\operatorname\s*d\s*\\!?\s*([A-Za-z])", r"d\1", s)

    # Remove braces around standalone command (e.g. {\int}).
    s = re.sub(r"\{\s*\\([A-Za-z]+)\s*\}", r"\\\1", s)
    s = re.sub(r"(\\infty)([A-Za-z])", r"\1 \2", s)

    s = _replace_hypergeometric_notation(s)
    s = re.sub(r"\s+", " ", s).strip()
    return _balance_braces(s)


def _balance_braces(s: str) -> str:
    out: list[str] = []
    depth = 0
    for ch in s:
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            if depth > 0:
                depth -= 1
                out.append(ch)
        else:
            out.append(ch)
    if depth > 0:
        out.extend("}" * depth)
    return "".join(out)


def _parse_function_name(name: str) -> tuple[str, int | None]:
    m = re.match(r"^([A-Za-z]+)(?:_\{?([0-9]+)\}?)?$", name)
    if not m:
        return name, None
    base = m.group(1)
    idx = int(m.group(2)) if m.group(2) else None
    return base, idx


def _ti(order: int, z: sp.Expr) -> sp.Expr:
    return (sp.polylog(order, sp.I * z) - sp.polylog(order, -sp.I * z)) / (2 * sp.I)


def _legendre_chi(order: int, z: sp.Expr) -> sp.Expr:
    return sp.Rational(1, 2) ** order * z * sp.lerchphi(z**2, order, sp.Rational(1, 2))


def _hyper_from_h_idx(idx: int, args: tuple[sp.Expr, ...]) -> sp.Expr | None:
    idx_str = str(idx)
    if len(idx_str) < 2:
        return None
    p = int(idx_str[:-1]) if len(idx_str) > 1 else int(idx_str[0])
    q = int(idx_str[-1])
    if p <= 0 or q <= 0:
        return None
    if len(args) != p + q + 1:
        return None
    a_params = list(args[:p])
    b_params = list(args[p : p + q])
    z = args[-1]
    return sp.hyper(a_params, b_params, z)


def _apply_known_symbol_constants(expr: sp.Expr) -> sp.Expr:
    replacements: dict[sp.Symbol, sp.Expr] = {}
    for sym in expr.free_symbols:
        mapped = KNOWN_SYMBOL_CONSTANTS.get(sym.name)
        if mapped is not None:
            replacements[sym] = mapped
    if replacements:
        expr = expr.xreplace(replacements)
    return expr


def _transform_undefined_functions(expr: sp.Expr) -> sp.Expr:
    replacements: dict[sp.Expr, sp.Expr] = {}
    for fn in expr.atoms(AppliedUndef):
        name = fn.func.__name__
        args = fn.args
        base, idx = _parse_function_name(name)

        if base == "Li" and idx is not None and len(args) == 1:
            replacements[fn] = sp.polylog(idx, args[0])
            continue
        if base == "Gamma" and len(args) == 1:
            replacements[fn] = sp.gamma(args[0])
            continue
        if base == "zeta" and len(args) in (1, 2):
            replacements[fn] = sp.zeta(*args)
            continue
        if base == "Ti" and idx is not None and len(args) == 1:
            replacements[fn] = _ti(idx, args[0])
            continue
        if base == "Phi" and len(args) == 3:
            replacements[fn] = sp.lerchphi(args[0], args[1], args[2])
            continue
        if base == "chi" and idx is not None and len(args) == 1:
            replacements[fn] = _legendre_chi(idx, args[0])
            continue
        if base == "Ei" and len(args) == 1:
            replacements[fn] = sp.Ei(args[0])
            continue
        if base == "Si" and len(args) == 1:
            replacements[fn] = sp.Si(args[0])
            continue
        if base == "Ci" and len(args) == 1:
            replacements[fn] = sp.Ci(args[0])
            continue
        if base == "arccot" and len(args) == 1:
            replacements[fn] = sp.acot(args[0])
            continue
        if base == "K" and idx is None and len(args) == 1:
            replacements[fn] = sp.elliptic_k(args[0])
            continue
        if base == "K" and idx is not None and len(args) == 1:
            replacements[fn] = sp.besselk(idx, args[0])
            continue
        if base == "E" and idx is None and len(args) == 1:
            replacements[fn] = sp.elliptic_e(args[0])
            continue
        if base == "Pi" and len(args) == 2:
            replacements[fn] = sp.elliptic_pi(args[0], args[1])
            continue
        if base == "F" and len(args) == 1:
            replacements[fn] = sp.dawson(args[0])
            continue
        if base == "H" and idx is not None:
            hyper = _hyper_from_h_idx(idx, args)
            if hyper is not None:
                replacements[fn] = hyper
                continue

    if replacements:
        expr = expr.xreplace(replacements)
    expr = _apply_known_symbol_constants(expr)
    return expr


def _is_effectively_real(sympy_value: sp.Expr, dps: int) -> tuple[bool, mp.mpf | None]:
    v = sp.N(sympy_value, dps)
    if hasattr(v, "free_symbols") and v.free_symbols:
        names = ",".join(sorted(sym.name for sym in v.free_symbols))
        raise ValueError(f"unresolved_symbols:{names}")

    rv = sp.N(sp.re(v), dps)
    iv = sp.N(sp.im(v), dps)
    if hasattr(rv, "free_symbols") and rv.free_symbols:
        names = ",".join(sorted(sym.name for sym in rv.free_symbols))
        raise ValueError(f"unresolved_real_symbols:{names}")
    if hasattr(iv, "free_symbols") and iv.free_symbols:
        names = ",".join(sorted(sym.name for sym in iv.free_symbols))
        raise ValueError(f"unresolved_imag_symbols:{names}")

    rv_m = mp.mpf(str(rv))
    iv_m = mp.mpf(str(iv))
    if abs(iv_m) <= mp.mpf(f"1e-{max(20, dps // 2)}"):
        return True, rv_m
    return False, None


def parse_latex_expression(latex: str) -> sp.Expr:
    normalized = _normalize_latex(latex)
    normalized = _strip_outer_assignment(normalized)
    expr = parse_latex(normalized)
    expr = _transform_undefined_functions(expr)
    return expr


def evaluate_expression_numeric(latex: str | None, dps: int = DEFAULT_DPS) -> EvalOutcome:
    if not latex:
        return EvalOutcome(None, None, "empty_expression")
    mp.mp.dps = dps
    try:
        expr = parse_latex_expression(latex)
        is_real, value = _is_effectively_real(expr, dps)
        if not is_real or value is None:
            return EvalOutcome(None, None, "expression_evaluated_to_complex")
        return EvalOutcome(value, _fmt(value, dps), None)
    except Exception as ex:  # noqa: BLE001
        return EvalOutcome(None, None, f"expression_parse_eval_error: {type(ex).__name__}: {ex}")


def _to_bound_numeric(expr: sp.Expr, dps: int) -> mp.mpf:
    expr = _apply_known_symbol_constants(expr)
    if expr in (sp.oo,):
        return mp.inf
    if expr in (-sp.oo,):
        return -mp.inf
    is_real, value = _is_effectively_real(expr, dps)
    if not is_real or value is None:
        raise ValueError("integral_bound_not_real")
    return value


def _integral_components_from_parsed(expr: sp.Expr) -> tuple[sp.Expr, sp.Symbol, sp.Expr, sp.Expr]:
    if isinstance(expr, sp.Equality):
        if isinstance(expr.rhs, sp.Integral):
            expr = expr.rhs
        elif isinstance(expr.lhs, sp.Integral):
            expr = expr.lhs

    if not isinstance(expr, sp.Integral):
        raise ValueError("not_an_integral_expression")
    if not expr.limits:
        raise ValueError("integral_has_no_limits")
    limit = expr.limits[0]
    if len(limit) != 3:
        raise ValueError("integral_limit_not_definite")
    var, lower, upper = limit
    if not isinstance(var, sp.Symbol):
        raise ValueError("integral_variable_not_symbol")
    return expr.function, var, lower, upper


def _fallback_parse_integral_text(latex: str) -> tuple[str, str, str, str]:
    s = _normalize_latex(latex).replace(" ", "")
    if not s.startswith("\\int"):
        raise ValueError("integral_missing_int_prefix")
    i = len("\\int")

    def read_token(idx: int) -> tuple[str, int]:
        if idx >= len(s):
            raise ValueError("unexpected_end_of_integral")
        if s[idx] == "{":
            depth = 1
            j = idx + 1
            while j < len(s) and depth > 0:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            if depth != 0:
                raise ValueError("unbalanced_braces_in_integral")
            return s[idx + 1 : j - 1], j
        if s[idx] == "\\":
            if s.startswith("\\infty", idx):
                return "\\infty", idx + len("\\infty")
            if s.startswith("\\pi", idx):
                return "\\pi", idx + len("\\pi")
            j = idx + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            return s[idx:j], j
        return s[idx], idx + 1

    lower = upper = None
    if i < len(s) and s[i] == "_":
        lower, i = read_token(i + 1)
    if i < len(s) and s[i] == "^":
        upper, i = read_token(i + 1)
    if lower is None or upper is None:
        raise ValueError("integral_missing_bounds")

    rest = s[i:]
    rest = rest.replace("\\mathrm{d}", "d")
    rest = rest.replace("\\mathrm", "")
    m = re.search(r"d([A-Za-z])$", rest)
    if not m:
        raise ValueError("integral_missing_differential")
    var = m.group(1)
    integrand = rest[: m.start()]
    if not integrand:
        raise ValueError("integral_missing_integrand")
    return integrand, var, lower, upper


def _integrate_numeric(
    integrand: sp.Expr,
    var: sp.Symbol,
    lower: sp.Expr,
    upper: sp.Expr,
    dps: int,
) -> mp.mpf:
    mp.mp.dps = dps
    integrand = _transform_undefined_functions(integrand)
    lower = _transform_undefined_functions(lower)
    upper = _transform_undefined_functions(upper)

    extra_symbols = (integrand.free_symbols | lower.free_symbols | upper.free_symbols) - {var}
    if extra_symbols:
        names = ",".join(sorted(sym.name for sym in extra_symbols))
        raise ValueError(f"integral_unresolved_symbols:{names}")

    func = sp.lambdify(var, integrand, modules=["mpmath"])
    lower_num = _to_bound_numeric(lower, dps)
    upper_num = _to_bound_numeric(upper, dps)

    if lower_num == -mp.inf and upper_num == mp.inf:
        return mp.quad(func, [-mp.inf, 0, mp.inf])
    if lower_num == -mp.inf:
        return mp.quad(func, [-mp.inf, upper_num])
    if upper_num == mp.inf:
        return mp.quad(func, [lower_num, mp.inf])
    return mp.quad(func, [lower_num, upper_num])


def evaluate_integral_numeric(latex: str | None, dps: int = DEFAULT_DPS) -> EvalOutcome:
    if not latex:
        return EvalOutcome(None, None, "empty_integral")
    mp.mp.dps = dps
    normalized = _normalize_latex(latex)
    normalized = _strip_outer_assignment(normalized, must_contain="\\int")

    if "\\int" in normalized and not normalized.startswith("\\int"):
        normalized = normalized[normalized.find("\\int") :]

    # First attempt: parse LaTeX directly to a SymPy Integral.
    try:
        expr = parse_latex(normalized)
        expr = _transform_undefined_functions(expr)
        integrand, var, lower, upper = _integral_components_from_parsed(expr)
        value = _integrate_numeric(integrand, var, lower, upper, dps)
        return EvalOutcome(value, _fmt(value, dps), None)
    except Exception:
        pass

    # Fallback parser for common \int_{}^{} ... dx structure.
    try:
        integrand_txt, var_txt, lower_txt, upper_txt = _fallback_parse_integral_text(normalized)
        integrand = _transform_undefined_functions(parse_latex(integrand_txt))
        var = sp.Symbol(var_txt)
        lower = _transform_undefined_functions(parse_latex(lower_txt))
        upper = _transform_undefined_functions(parse_latex(upper_txt))
        value = _integrate_numeric(integrand, var, lower, upper, dps)
        return EvalOutcome(value, _fmt(value, dps), None)
    except Exception as ex:  # noqa: BLE001
        return EvalOutcome(None, None, f"integral_parse_eval_error: {type(ex).__name__}: {ex}")


def abs_rel_close(
    left: mp.mpf | None,
    right: mp.mpf | None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[str | None, bool | None]:
    if left is None or right is None:
        return None, None
    abs_err = abs(left - right)
    rel_err = abs_err / max(mp.mpf("1.0"), abs(right))
    passed = (abs_err <= tolerance) or (rel_err <= tolerance)
    return mp.nstr(abs_err, n=25), bool(passed)


def maybe_run_async(fn: Callable[[], object]) -> object:
    """Run a callable even when an event loop already exists."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return fn()
    return loop.run_until_complete(asyncio.to_thread(fn))
