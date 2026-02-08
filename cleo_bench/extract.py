"""Extraction and prompt normalization utilities."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

MATH_BLOCK_PATTERNS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
    re.compile(r"\$(.+?)\$", re.DOTALL),
]

INTEGRAL_TAIL_RE = re.compile(r"(\\int[\s\S]{0,1200}?(?<![A-Za-z])d\s*[A-Za-z])")
ASSIGNMENT_LHS_RE = re.compile(r"^[A-Za-z](?:_[A-Za-z0-9]+)?$")
FORMAT_CMD_RE = re.compile(
    r"\\(?:large|Large|LARGE|huge|Huge|displaystyle|textstyle|scriptstyle|scriptscriptstyle)\b"
)
ENV_RE = re.compile(r"\\(?:begin|end)\{[A-Za-z*]+\}")
TAG_RE = re.compile(r"\\tag(?:\{[^{}]*\}|[0-9]+)")
HSPACE_RE = re.compile(r"\\hspace\*?\{[^{}]*\}")
INTEGRAL_BLOB_RE = re.compile(
    r"(\\int(?:\\limits)?[\s\S]{0,1400}?(?:"
    r"(?<![A-Za-z])d\s*[A-Za-z]|"
    r"\\mathrm\s*\{\s*d\s*\}\s*[A-Za-z]|"
    r"\\rm\s*d\s*[A-Za-z]|"
    r"\\operatorname\s*d\s*\\!?\s*[A-Za-z]"
    r"))"
)

HINT_LINE_PATTERNS = [
    re.compile(r"numeric\s+value\s+is", re.IGNORECASE),
    re.compile(r"\\approx"),
    re.compile(r"≈"),
    re.compile(r"\bapproximately\b", re.IGNORECASE),
]

LONG_DECIMAL_RE = re.compile(r"\d+\.\d{10,}")


@dataclass
class ExtractionResult:
    integral_latex: str | None
    cleo_reference_latex: str | None
    accepted_reference_latex: str | None
    prompt_full_question_sanitized: str
    prompt_integral_only: str | None
    is_integral_candidate: bool


def html_to_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text("\n")
    text = html.unescape(text)
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line.strip())
    return cleaned.strip()


def extract_math_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    seen: set[str] = set()
    for pattern in MATH_BLOCK_PATTERNS:
        for m in pattern.finditer(text):
            body = _clean_latex(m.group(1))
            if body and body not in seen:
                seen.add(body)
                blocks.append(body)
    return blocks


def _clean_latex(value: str | None) -> str | None:
    if value is None:
        return None
    out = value.strip()
    out = out.strip("$")
    out = out.strip()
    out = out.rstrip(". ")
    return out or None


def _normalize_block_for_extraction(block: str) -> str:
    text = block.replace("\n", " ")
    text = text.replace("&", " ")
    text = text.replace("\\\\+", " + ")
    text = text.replace("\\\\-", " - ")
    text = text.replace("\\\\", " ; ")
    text = FORMAT_CMD_RE.sub(" ", text)
    text = HSPACE_RE.sub(" ", text)
    text = ENV_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_integral_subexpr(text: str) -> str | None:
    normalized = _normalize_block_for_extraction(text)
    m = INTEGRAL_BLOB_RE.search(normalized)
    if m:
        return _clean_latex(m.group(1))
    m2 = INTEGRAL_TAIL_RE.search(normalized)
    if m2:
        return _clean_latex(m2.group(1))
    return None


def extract_integral_latex(title: str, body_text: str) -> str | None:
    # Body usually contains the full expression; titles are often shortened.
    for text in (body_text, title):
        blocks = extract_math_blocks(text)
        for block in blocks:
            if "\\int" in block:
                piece = _extract_integral_subexpr(block)
                if piece:
                    return piece
        piece = _extract_integral_subexpr(text)
        if piece:
            return piece
    return None


def _extract_assignment_rhs(block: str) -> str | None:
    if "=" not in block:
        return None
    lhs, rhs = block.split("=", 1)
    lhs_norm = re.sub(r"[{}\\\s]", "", lhs)
    rhs_norm = _clean_latex(rhs)
    if not rhs_norm:
        return None
    if ASSIGNMENT_LHS_RE.match(lhs_norm):
        return rhs_norm
    return None


def _candidate_score(candidate: str) -> int:
    score = 0
    c = candidate.strip()
    if not c:
        return -999
    if "\\int" in c:
        score -= 60
    if "\\begin" in c or "\\end" in c:
        score -= 40
    if "=" in c:
        score -= 15
    if "\\pm" in c:
        score -= 10
    if "\\Im" in c or "\\Re" in c:
        score -= 6
    if "\\Li" in c or "\\Gamma" in c or "\\zeta" in c:
        score += 4
    if re.search(r"(?<!\\)\b[xtsuab]\b", c):
        score -= 4
    if len(c) > 400:
        score -= 5
    return score


def _maybe_stitch_adjacent_arg(current: str, next_block: str | None) -> str:
    if not next_block:
        return current
    if "=" in next_block or "\\int" in next_block:
        return current
    if not re.search(r"\\[A-Za-z_]+$", current):
        return current
    next_clean = _clean_latex(next_block)
    if not next_clean:
        return current
    return f"{current}\\left({next_clean}\\right)"


def extract_reference_latex(answer_text: str) -> str | None:
    blocks = extract_math_blocks(answer_text)
    if not blocks:
        return None

    candidates: list[str] = []

    for idx, block in enumerate(blocks):
        normalized = _normalize_block_for_extraction(block)
        if not normalized:
            continue

        direct_rhs = _extract_assignment_rhs(normalized)
        if direct_rhs:
            stitched = _maybe_stitch_adjacent_arg(
                direct_rhs,
                _normalize_block_for_extraction(blocks[idx + 1]) if idx + 1 < len(blocks) else None,
            )
            candidates.append(stitched)

        # Parse multi-equality constructs (e.g. align environments).
        for segment in normalized.split(";"):
            seg = _clean_latex(segment)
            if not seg or "=" not in seg:
                continue
            rhs_parts = [_clean_latex(part) for part in seg.split("=")[1:]]
            for rhs in rhs_parts:
                if rhs:
                    candidates.append(rhs)

        # If there is no equation, keep it as a fallback candidate.
        if "=" not in normalized and "\\approx" not in normalized:
            c = _clean_latex(normalized)
            if c:
                candidates.append(c)

    filtered = [c for c in candidates if c and "\\approx" not in c]
    if not filtered:
        return None

    filtered.sort(key=lambda c: _candidate_score(c), reverse=True)
    return _clean_latex(filtered[0])


def strip_numeric_hints(text: str) -> str:
    if not text:
        return text
    out_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(p.search(line) for p in HINT_LINE_PATTERNS):
            continue
        if LONG_DECIMAL_RE.search(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def build_full_prompt(title: str, question_body_text: str) -> str:
    sanitized = strip_numeric_hints(question_body_text)
    return (
        f"Title: {title}\n\n"
        f"Question:\n{sanitized}\n\n"
        "Return your final answer as JSON with key \"final_expression_latex\"."
    )


def build_integral_prompt(integral_latex: str | None) -> str | None:
    if not integral_latex:
        return None
    return (
        "Evaluate the following integral and return only the final closed form.\n\n"
        f"$$ {integral_latex} $$\n\n"
        "Return your final answer as JSON with key \"final_expression_latex\"."
    )


def extract_item_fields(
    title_raw: str,
    question_body_html: str,
    cleo_body_html: str,
    accepted_body_html: str | None,
) -> ExtractionResult:
    question_text = html_to_text(question_body_html)
    cleo_text = html_to_text(cleo_body_html)
    accepted_text = html_to_text(accepted_body_html) if accepted_body_html else ""

    integral_latex = extract_integral_latex(title_raw, question_text)
    cleo_reference = extract_reference_latex(cleo_text)
    accepted_reference = extract_reference_latex(accepted_text) if accepted_text else None

    full_prompt = build_full_prompt(title_raw, question_text)
    integral_prompt = build_integral_prompt(integral_latex)

    return ExtractionResult(
        integral_latex=integral_latex,
        cleo_reference_latex=cleo_reference,
        accepted_reference_latex=accepted_reference,
        prompt_full_question_sanitized=full_prompt,
        prompt_integral_only=integral_prompt,
        is_integral_candidate=integral_latex is not None,
    )
