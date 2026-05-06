from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

from rgr_lp.io import read_matrix
from rgr_lp.solver import solve_lp_canonical


def _fmt_frac(x: Fraction) -> str:
    if x.denominator == 1:
        return str(x.numerator)
    return f"{x.numerator}/{x.denominator}"


def main() -> None:
    # Чтобы русские сообщения корректно печатались в Windows-терминале.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="РГР: решение ЗЛП в канонической форме (Жордан–Гаусс + симплекс)."
    )
    parser.add_argument("input", type=Path, help="Путь к файлу с матрицей (m+1)x(n+1).")
    parser.add_argument(
        "--no-steps",
        action="store_true",
        help="Не печатать промежуточные таблицы и шаги (по умолчанию печатаются).",
    )
    args = parser.parse_args()

    mat = read_matrix(args.input)
    result = solve_lp_canonical(mat, steps=not args.no_steps)

    print(f"Статус: {result.status}")
    if result.message:
        print(result.message)

    if result.status != "OPTIMAL":
        return

    print("\nОптимальный план:")
    for i, xi in enumerate(result.x, start=1):
        print(f"x{i} = {_fmt_frac(xi)}")
    print(f"Z* = {_fmt_frac(result.z)}")

    if result.parametric_optimal_set is not None:
        s = result.parametric_optimal_set
        print("\nБесконечно много оптимальных решений.")
        print("Общий вид (параметрическое описание множества оптимальных решений):")
        for line in s.equations:
            print(f"  {line}")
        print("Условия на параметры и неотрицательность:")
        for line in s.constraints:
            print(f"  {line}")

        # Дополнительно: λ-представление (как на 4), чтобы было видно альтернативный оптимум
        if result.general_solution is not None and len(result.general_solution.vertices) >= 2:
            x_a = result.general_solution.vertices[0]
            x_b = result.general_solution.vertices[1]
            print("\nТакже (альтернативный оптимум): одномерное семейство между двумя оптимальными опорными планами:")
            print(f"  X^1 = ({', '.join(_fmt_frac(v) for v in x_a)})")
            print(f"  X^2 = ({', '.join(_fmt_frac(v) for v in x_b)})")
            print("  X(lambda) = (1 - lambda)*X^1 + lambda*X^2, where 0 <= lambda <= 1.")

            # Несколько конкретных точек
            for lam in (Fraction(0), Fraction(1, 2), Fraction(1)):
                x_lam = [(Fraction(1) - lam) * a + lam * b for a, b in zip(x_a, x_b)]
                print(
                    f"  lambda={_fmt_frac(lam)} -> X=({', '.join(_fmt_frac(v) for v in x_lam)})"
                )
    elif result.general_solution is not None:
        # запасной режим (на случай, если параметризация не построена)
        print("\nБесконечно много оптимальных решений.")
        print("Несколько найденных оптимальных опорных планов:")
        for k, xk in enumerate(result.general_solution.vertices, start=1):
            z_k = sum(ci * vi for ci, vi in zip(result.c, xk))
            print(f"  X^{k} (Z={_fmt_frac(z_k)}): ({', '.join(_fmt_frac(v) for v in xk)})")


if __name__ == "__main__":
    main()

