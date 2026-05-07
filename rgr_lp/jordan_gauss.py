from __future__ import annotations

from fractions import Fraction
from typing import Callable

from .tableau import Pivot, Tableau, format_tableau, pivot


def _ensure_nonnegative_rhs(tableau: Tableau, m: int, n: int) -> None:
    rhs = n
    for i in range(m):
        if tableau[i][rhs] < 0:
            tableau[i] = [-v for v in tableau[i]]


def find_initial_basis_by_jordan(
    tableau: Tableau,
    m: int,
    n: int,
    *,
    steps: bool = False,
    print_fn: Callable[[str], None] | None = None,
) -> list[int] | None:
    """
    Находит опорное решение (базис) жордановыми исключениями, не ориентируясь на Z-строку,
    но сохраняя неотрицательность столбца свободных членов (идея из лекции 5).

    Возвращает basis длины m: basis[i] = индекс базисной переменной в i-й строке.
    Модифицирует tableau на месте (приводит выбранные столбцы к единичным).

    Если базис размера m построить не удаётся — возвращает None.
    """
    if print_fn is None:
        print_fn = print

    _ensure_nonnegative_rhs(tableau, m=m, n=n)

    basis = [-1] * m
    used_rows: set[int] = set()

    if steps:
        print_fn("Жордан–Гаусс: построение опорного плана (игнорируем Z-строку).")
        print_fn("Начальная таблица:")
        print_fn(format_tableau(tableau, m=m, n=n, basis=basis))

    # Идём по столбцам 0..n-1 и повторяем проходы, пока не получится базис размера m.
    # После pivot коэффициенты меняются: столбец, в котором раньше не было положительного
    # элемента в свободной строке, на следующем проходе может подойти (как в классной работе).
    max_passes = max(n * m, 1) * 10
    pass_num = 0
    while len(used_rows) < m and pass_num < max_passes:
        pass_num += 1
        progress_this_pass = False

        if steps and pass_num > 1:
            print_fn(f"\n--- Проход по столбцам №{pass_num} (ещё свободные строки: {m - len(used_rows)}) ---")

        for col in range(n):
            if len(used_rows) == m:
                break
            # уже базисный столбец?
            if col in basis:
                continue

            if steps:
                print_fn(f"\nВыбираем столбец x{col+1} для попытки ввода в базис.")

            # минимум отношения b_i / a_i,col среди a_i,col > 0 в строках без базиса
            best_row = None
            best_ratio = None
            for row in range(m):
                if row in used_rows:
                    continue
                a = tableau[row][col]
                if a <= 0:
                    continue
                b = tableau[row][n]
                ratio = b / a
                if best_ratio is None or ratio < best_ratio:
                    best_ratio = ratio
                    best_row = row

            if best_row is None:
                if steps:
                    print_fn("  Нет подходящей строки (в столбце нет положительных элементов).")
                continue

            if steps:
                print_fn(
                    f"  Разрешающая строка: {best_row+1} (мин. отношение b/a = {best_ratio})."
                )
                print_fn(f"  Разрешающий элемент: a[{best_row+1},{col+1}] = {tableau[best_row][col]}")

            pivot(tableau, Pivot(best_row, col))
            basis[best_row] = col
            used_rows.add(best_row)
            progress_this_pass = True

            if steps:
                print_fn("  После преобразования:")
                print_fn(format_tableau(tableau, m=m, n=n, basis=basis))

        if not progress_this_pass:
            break

    if any(j < 0 for j in basis):
        if steps:
            print_fn("\nНе удалось сформировать базис размера m.")
        return None
    _ensure_nonnegative_rhs(tableau, m=m, n=n)
    if steps:
        print_fn("\nОпорный план построен. Итоговая таблица:")
        print_fn(format_tableau(tableau, m=m, n=n, basis=basis))
    return basis


def check_inconsistent_constraints(tableau: Tableau, m: int, n: int) -> bool:
    """
    Грубая проверка несовместности: строка ограничений вида 0 ... 0 | b, где b != 0.
    (для already-reduced таблиц; помогает корректно возвращать INFEASIBLE)
    """
    rhs = n
    for i in range(m):
        if all(tableau[i][j] == 0 for j in range(n)) and tableau[i][rhs] != 0:
            return True
    return False

