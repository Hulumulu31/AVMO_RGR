from __future__ import annotations

from fractions import Fraction
from pathlib import Path


Matrix = list[list[Fraction]]
Sense = str  # "min" | "max"


def parse_fraction(token: str) -> Fraction:
    """
    Поддержка:
    - целые: "5", "-3"
    - дроби: "7/9", "-10/4"
    """
    token = token.strip()
    if "/" in token:
        a, b = token.split("/", 1)
        return Fraction(int(a.strip()), int(b.strip()))
    return Fraction(int(token), 1)


def read_matrix(path: Path) -> Matrix:
    text = path.read_text(encoding="utf-8")
    rows: list[list[Fraction]] = []
    for raw_line in text.splitlines():
        # Поддержка комментариев: всё после '#' игнорируется
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        # Разрешаем запятые и табы как разделители (часто встречается в конспектах/экспорте)
        line = line.replace(",", " ").replace("\t", " ")
        parts = [p for p in line.split() if p]
        rows.append([parse_fraction(p) for p in parts])

    if not rows:
        raise ValueError("Пустой входной файл.")
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("Матрица должна быть прямоугольной (одинаковое число столбцов).")
    if len(rows) < 2 or width < 2:
        raise ValueError("Ожидается матрица размера (m+1)x(n+1), минимум 2x2.")
    return rows


def read_problem(path: Path) -> tuple[Sense, Matrix]:
    """
    Формат:
    - опционально: первая значимая строка (после удаления комментариев) может быть 'min' или 'max'
    - далее: матрица (m+1)x(n+1)
    """
    text = path.read_text(encoding="utf-8")
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        line = line.replace(",", " ").replace("\t", " ")
        cleaned_lines.append(line)

    if not cleaned_lines:
        raise ValueError("Пустой входной файл.")

    first = cleaned_lines[0].strip().lower()
    if first in ("min", "max"):
        sense: Sense = first
        matrix_lines = cleaned_lines[1:]
    else:
        sense = "max"
        matrix_lines = cleaned_lines

    if not matrix_lines:
        raise ValueError("После строки 'min/max' должна идти матрица.")

    rows: list[list[Fraction]] = []
    for line in matrix_lines:
        parts = [p for p in line.split() if p]
        rows.append([parse_fraction(p) for p in parts])

    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError("Матрица должна быть прямоугольной (одинаковое число столбцов).")
    if len(rows) < 2 or width < 2:
        raise ValueError("Ожидается матрица размера (m+1)x(n+1), минимум 2x2.")

    return sense, rows

