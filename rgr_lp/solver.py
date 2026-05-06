from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable

from .io import Matrix
from .jordan_gauss import check_inconsistent_constraints, find_initial_basis_by_jordan
from .simplex import (
    can_pivot_in_column,
    iterate_simplex,
    zero_reduced_cost_nonbasic,
)
from .tableau import Tableau, extract_plan, fmt_frac, objective_value, pivot, Pivot


@dataclass(frozen=True)
class GeneralSolution:
    vertices: list[list[Fraction]]  # список оптимальных опорных планов


@dataclass(frozen=True)
class ParametricOptimalSet:
    """
    Параметрическое описание множества оптимальных решений:
    - выбираем небазисные переменные с нулевой оценкой как параметры t_k >= 0
    - остальные небазисные (с положительной оценкой) фиксируем нулём
    - базисные выражаем через t_k и добавляем условия x_B >= 0
    """

    parameter_vars: list[int]  # индексы x_j, которые стали параметрами
    fixed_zero_vars: list[int]  # индексы x_j, зафиксированные в 0
    basis_vars: list[int]  # индексы базисных переменных по строкам
    equations: list[str]  # строки вида "x_i = ... "
    constraints: list[str]  # строки вида "t_k >= 0", "x_i >= 0 => ..."


@dataclass(frozen=True)
class SolveResult:
    status: str  # OPTIMAL / INFEASIBLE / UNBOUNDED
    x: list[Fraction]
    z: Fraction
    c: list[Fraction]
    message: str | None = None
    general_solution: GeneralSolution | None = None
    parametric_optimal_set: ParametricOptimalSet | None = None


def solve_lp_canonical(
    mat: Matrix,
    *,
    steps: bool = False,
    print_fn: Callable[[str], None] | None = None,
) -> SolveResult:
    """
    mat: (m+1)x(n+1)
      первые m строк: A|b
      последняя строка: c|c0 (обычно 0)
    """
    # TODO: реализовать через Жордан–Гаусс + симплекс
    m_plus_1 = len(mat)
    n_plus_1 = len(mat[0])
    m = m_plus_1 - 1
    n = n_plus_1 - 1

    c = mat[m][:n]

    # Формируем таблицу: m строк ограничений + Z-строка.
    # Z-строка хранит -c (как в лекции 4): оптимум, когда все элементы >= 0.
    tableau: Tableau = []
    for i in range(m):
        tableau.append(list(mat[i][:]))
    tableau.append([-ci for ci in c] + [Fraction(0)])

    if print_fn is None:
        print_fn = print

    basis = find_initial_basis_by_jordan(tableau, m=m, n=n, steps=steps, print_fn=print_fn)
    if basis is None or check_inconsistent_constraints(tableau, m=m, n=n):
        return SolveResult(
            status="INFEASIBLE",
            x=[Fraction(0) for _ in range(n)],
            z=Fraction(0),
            c=c,
            message="Не удалось построить опорное решение методом Жордана–Гаусса (возможна несовместность ограничений).",
        )

    # Проверка допустимости текущего опорного плана (x>=0)
    x0 = extract_plan(tableau, basis=basis, n=n, m=m)
    if any(v < 0 for v in x0):
        return SolveResult(
            status="INFEASIBLE",
            x=[Fraction(0) for _ in range(n)],
            z=Fraction(0),
            c=c,
            message="Опорный план, найденный Жорданом–Гауссом, не удовлетворяет x>=0.",
        )

    status = iterate_simplex(tableau, basis=basis, m=m, n=n, steps=steps, print_fn=print_fn)
    if status == "UNBOUNDED":
        return SolveResult(
            status="UNBOUNDED",
            x=[Fraction(0) for _ in range(n)],
            z=Fraction(0),
            c=c,
            message="Целевая функция не ограничена сверху (признак неограниченности).",
        )

    x_opt = extract_plan(tableau, basis=basis, n=n, m=m)
    z_opt = objective_value(tableau, n=n, m=m)

    # Альтернативный оптимум: ищем нулевые оценки у небазисных.
    zeros = zero_reduced_cost_nonbasic(tableau, basis=basis, m=m, n=n)
    vertices: list[list[Fraction]] = [x_opt]

    # Пытаемся построить дополнительные оптимальные опорные планы одним шагом симплекс-преобразования
    # по столбцу с нулевой оценкой (замечание из лекции 4).
    for col in zeros:
        if not can_pivot_in_column(tableau, m=m, n=n, col=col):
            continue

        # копируем таблицу и базис
        t2: Tableau = [row[:] for row in tableau]
        b2 = basis[:]

        # leaving row: минимальное отношение b_i / a_i,col
        rhs = n
        best_row = None
        best_ratio = None
        for i in range(m):
            a = t2[i][col]
            if a <= 0:
                continue
            ratio = t2[i][rhs] / a
            if best_ratio is None or ratio < best_ratio:
                best_ratio = ratio
                best_row = i
        if best_row is None:
            continue

        pivot(t2, Pivot(best_row, col))
        b2[best_row] = col
        x2 = extract_plan(t2, basis=b2, n=n, m=m)

        if x2 not in vertices:
            vertices.append(x2)

        # ограничимся несколькими вершинами, чтобы вывод не разрастался бесконечно
        if len(vertices) >= 6:
            break

    general = GeneralSolution(vertices=vertices) if len(vertices) > 1 else None

    # Строим "общий вид" для оценки 5: параметризация оптимального множества
    parametric: ParametricOptimalSet | None = None
    if len(zeros) > 0:
        basis_set = set(basis)
        z_row = tableau[m]

        # Небазисные с >0 оценкой в оптимальной таблице можно (и нужно) фиксировать в 0,
        # чтобы оставаться оптимальными при max.
        fixed_zero = [j for j in range(n) if j not in basis_set and z_row[j] > 0 and j not in zeros]

        # Параметры t_k соответствуют x_{zeros[k]}
        param_vars = zeros[:]

        eqs: list[str] = []
        cons: list[str] = []

        # Параметры
        for k, j in enumerate(param_vars, start=1):
            eqs.append(f"x{j+1} = t{k}")
            cons.append(f"t{k} >= 0")

        # Зафиксированные нулём
        for j in fixed_zero:
            eqs.append(f"x{j+1} = 0")

        # Базисные через параметры:
        # строка i: x_basis + sum_{j not basis} a_ij * x_j = CO
        # => x_basis = CO - sum a_ij * x_j
        for i in range(m):
            bvar = basis[i]
            rhs = tableau[i][n]
            parts: list[str] = [fmt_frac(rhs)]
            for k, j in enumerate(param_vars, start=1):
                a = tableau[i][j]
                if a == 0:
                    continue
                # x_b = rhs - a*t
                # печатаем аккуратно знаком
                if a > 0:
                    parts.append(f"- ({fmt_frac(a)})*t{k}")
                else:
                    parts.append(f"+ ({fmt_frac(-a)})*t{k}")
            expr = " ".join(parts) if parts else "0"
            eqs.append(f"x{bvar+1} = {expr}")
            cons.append(f"x{bvar+1} >= 0  <=>  {expr} >= 0")

        parametric = ParametricOptimalSet(
            parameter_vars=param_vars,
            fixed_zero_vars=fixed_zero,
            basis_vars=basis[:],
            equations=eqs,
            constraints=cons,
        )

    return SolveResult(
        status="OPTIMAL",
        x=x_opt,
        z=z_opt,
        c=c,
        general_solution=general,
        parametric_optimal_set=parametric,
    )

