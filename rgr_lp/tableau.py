from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


Tableau = list[list[Fraction]]  # (m+1) x (n+1): last row is objective, last col is RHS


@dataclass(frozen=True)
class Pivot:
    row: int
    col: int


def fmt_frac(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def format_tableau(tableau: Tableau, m: int, n: int, basis: list[int] | None = None) -> str:
    """
    Красивый вывод симплекс-таблицы/таблицы Жордана–Гаусса.
    Столбцы: x1..xn и CO (свободные члены).
    Последняя строка: Z.
    """
    headers = [f"x{j+1}" for j in range(n)] + ["CO"]
    rows = []
    for i in range(m):
        bp = f"x{basis[i]+1}" if basis is not None and basis[i] >= 0 else "-"
        rows.append([bp] + [fmt_frac(tableau[i][j]) for j in range(n)] + [fmt_frac(tableau[i][n])])
    rows.append(["Z"] + [fmt_frac(tableau[m][j]) for j in range(n)] + [fmt_frac(tableau[m][n])])

    headers2 = ["б.п."] + headers
    widths = [len(h) for h in headers2]
    for r in rows:
        for k, cell in enumerate(r):
            widths[k] = max(widths[k], len(cell))

    def render_row(r: list[str]) -> str:
        return "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(r))

    out = [render_row(headers2), render_row(["-" * w for w in widths])]
    out.extend(render_row(r) for r in rows)
    return "\n".join(out)


def pivot(tableau: Tableau, p: Pivot) -> None:
    """
    Выполняет жорданово исключение по разрешающему элементу tableau[p.row][p.col].
    Приводит столбец p.col к единичному (1 в pivot-row, 0 в остальных).
    """
    pr, pc = p.row, p.col
    pe = tableau[pr][pc]
    if pe == 0:
        raise ZeroDivisionError("Разрешающий элемент равен 0.")

    rows = len(tableau)
    cols = len(tableau[0])

    # Normalize pivot row
    if pe != 1:
        inv = Fraction(pe.denominator, pe.numerator)
        tableau[pr] = [v * inv for v in tableau[pr]]

    # Eliminate in other rows
    for r in range(rows):
        if r == pr:
            continue
        factor = tableau[r][pc]
        if factor == 0:
            continue
        tableau[r] = [tableau[r][c] - factor * tableau[pr][c] for c in range(cols)]


def extract_plan(tableau: Tableau, basis: list[int], n: int, m: int) -> list[Fraction]:
    """
    Возвращает x длины n для текущей таблицы, предполагая:
    - в строках 0..m-1 базисные переменные basis[i]
    - соответствующие базисные столбцы образуют единичную матрицу
    """
    rhs_col = n
    x = [Fraction(0) for _ in range(n)]
    for i in range(m):
        j = basis[i]
        if j < 0:
            continue
        x[j] = tableau[i][rhs_col]
    return x


def objective_value(tableau: Tableau, n: int, m: int) -> Fraction:
    # В нашей конвенции в правом нижнем углу хранится свободный член Z-строки.
    return tableau[m][n]

