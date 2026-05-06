from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from .tableau import Pivot, Tableau, format_tableau, pivot


@dataclass(frozen=True)
class SimplexStep:
    pivot: Pivot
    entering: int
    leaving_row: int


def is_optimal(tableau: Tableau, m: int, n: int) -> bool:
    """
    Конвенция как в лекциях: в Z-строке хранятся -c'_j.
    Для max оптимум: все коэффициенты Z-строки при переменных должны быть >= 0.
    """
    z_row = tableau[m]
    return all(z_row[j] >= 0 for j in range(n))


def choose_entering_column(tableau: Tableau, m: int, n: int) -> int | None:
    """
    Выбираем разрешающий столбец: самый отрицательный коэффициент в Z-строке.
    """
    z = tableau[m]
    best_j = None
    best_val = Fraction(0)
    for j in range(n):
        if z[j] < best_val:
            best_val = z[j]
            best_j = j
    return best_j


def choose_leaving_row(tableau: Tableau, m: int, n: int, entering_col: int) -> int | None:
    """
    Классическое симплексное отношение: b_i / a_i,entering при a_i,entering > 0.
    Берём минимум.
    """
    rhs = n
    best_i = None
    best_ratio = None
    for i in range(m):
        a = tableau[i][entering_col]
        if a <= 0:
            continue
        ratio = tableau[i][rhs] / a
        if best_ratio is None or ratio < best_ratio:
            best_ratio = ratio
            best_i = i
    return best_i


def iterate_simplex(
    tableau: Tableau,
    basis: list[int],
    m: int,
    n: int,
    *,
    max_iters: int = 10_000,
    steps: bool = False,
    print_fn: Callable[[str], None] | None = None,
) -> str:
    """
    Модифицирует tableau и basis на месте.
    Возвращает статус: OPTIMAL / UNBOUNDED.
    """
    if print_fn is None:
        print_fn = print

    if steps:
        print_fn("\nСимплекс-метод: итерационный процесс.")
        print_fn("Стартовая симплекс-таблица:")
        print_fn(format_tableau(tableau, m=m, n=n, basis=basis))

    for _ in range(max_iters):
        if is_optimal(tableau, m=m, n=n):
            if steps:
                print_fn("\nКритерий оптимальности выполнен (в Z-строке нет отрицательных элементов).")
            return "OPTIMAL"

        entering = choose_entering_column(tableau, m=m, n=n)
        if entering is None:
            return "OPTIMAL"

        leaving_row = choose_leaving_row(tableau, m=m, n=n, entering_col=entering)
        if leaving_row is None:
            if steps:
                print_fn(
                    "\nПризнак неограниченности: выбран столбец с отрицательной оценкой, "
                    "но в нём нет положительных элементов в строках ограничений."
                )
            return "UNBOUNDED"

        if steps:
            print_fn("\n--- Итерация ---")
            print_fn(f"Разрешающий столбец: x{entering+1} (самая отрицательная оценка в Z-строке).")
            print_fn("Симплексные отношения (b_i / a_i) для a_i>0:")
            rhs = n
            for i in range(m):
                a = tableau[i][entering]
                if a > 0:
                    ratio = tableau[i][rhs] / a
                    print_fn(f"  строка {i+1}: {tableau[i][rhs]} / {a} = {ratio}")
            print_fn(f"Разрешающая строка: {leaving_row+1}")
            print_fn(f"Разрешающий элемент: a[{leaving_row+1},{entering+1}] = {tableau[leaving_row][entering]}")

        pivot(tableau, Pivot(leaving_row, entering))
        basis[leaving_row] = entering

        if steps:
            print_fn("Таблица после преобразования:")
            print_fn(format_tableau(tableau, m=m, n=n, basis=basis))

    raise RuntimeError("Превышен лимит итераций симплекс-метода (возможен цикл).")


def zero_reduced_cost_nonbasic(tableau: Tableau, basis: list[int], m: int, n: int) -> list[int]:
    """
    Возвращает индексы НЕбазисных переменных с нулевой оценкой в оптимальной таблице
    (признак альтернативного оптимума из лекции 4: -c'_j = 0).
    """
    z = tableau[m]
    basis_set = set(basis)
    return [j for j in range(n) if j not in basis_set and z[j] == 0]


def can_pivot_in_column(tableau: Tableau, m: int, n: int, col: int) -> bool:
    return any(tableau[i][col] > 0 for i in range(m))

