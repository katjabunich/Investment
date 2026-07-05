#!/usr/bin/env python3
# ============================================================================
# cashflow.py — советник по размеру ежемесячного инвестиционного взноса
# ============================================================================
#
# Что делает скрипт:
#   1. Читает config/cashflow.yaml (доходы, расходы, подушка безопасности,
#      крупные плановые траты) — либо берёт данные из флагов --income/--expenses.
#   2. Считает текущую картину: свободный денежный поток, норму сбережений,
#      заполненность подушки безопасности, ежемесячный резерв под крупные
#      плановые траты.
#   3. Даёт рекомендацию по ежемесячному взносу в инвестиции по прозрачным
#      правилам (описаны ниже и продублированы в выводе программы).
#
# ----------------------------------------------------------------------------
# ПРАВИЛА РАСЧЁТА РЕКОМЕНДОВАННОГО ВЗНОСА (в порядке применения):
# ----------------------------------------------------------------------------
#
#   Шаг 0. Свободный денежный поток (СДП) = доход − расходы.
#          Если СДП <= 0 — инвестировать не из чего, сначала нужно сократить
#          расходы или увеличить доход.
#
#   Шаг 1. Буфер на непредвиденное. Часть СДП сразу откладывается "в сторону"
#          и не участвует в дальнейшем распределении:
#            - income_stability = stable   -> 10% от СДП
#            - income_stability = variable -> 20% от СДП
#          (нерегулярный доход сложнее прогнозировать, поэтому подушка
#          прочности при каждом расчёте берётся больше).
#          Остаток после буфера называем "доступная сумма".
#
#   Шаг 2. Резерв под крупные плановые траты. Для каждой траты из
#          planned_large_expenses считаем amount / months_until (сколько
#          нужно откладывать в месяц, чтобы накопить к сроку). Если
#          months_until <= 0 — деньги нужны уже сейчас, вся сумма считается
#          немедленным резервом текущего месяца (и выводится предупреждение).
#          Сумма всех таких резервов вычитается из доступной суммы.
#
#   Шаг 3. Пополнение подушки безопасности. Целевой размер подушки =
#          суммарные месячные расходы × emergency_fund_months_target.
#          Если текущая подушка меньше целевой (gap > 0), из того, что
#          осталось после шагов 1-2, направляем на подушку долю:
#            - income_stability = stable   -> 50% остатка (но не больше gap)
#            - income_stability = variable -> 70% остатка (подушка приоритетнее)
#
#   Шаг 4. Всё, что осталось после шагов 1-3, — рекомендованный ежемесячный
#          взнос в инвестиции.
#
#   Если доступной суммы не хватает даже на шаги 1-2 (резервы под крупные
#   траты "съедают" весь буферизованный СДП), рекомендованный взнос считается
#   0, а дефицит указывается отдельно — крупные траты стоит пересмотреть или
#   растянуть по срокам.
# ----------------------------------------------------------------------------
#
# Использование:
#   python3 tools/cashflow.py                      # взять всё из config/cashflow.yaml
#   python3 tools/cashflow.py --income 6000 --expenses 3500   # быстрая прикидка
#   python3 tools/cashflow.py --config config/my_cashflow.yaml
#
# ============================================================================

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "cashflow.yaml"

BUFFER_PCT = {"stable": 0.10, "variable": 0.20}
FUND_PRIORITY_PCT = {"stable": 0.50, "variable": 0.70}


# ----------------------------------------------------------------------------
# Загрузка и подготовка данных
# ----------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {path}\n"
            f"Создайте его на основе примера или укажите другой путь через --config."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def total_expenses(monthly_expenses: Any) -> float:
    """monthly_expenses — либо словарь категорий, либо число, либо словарь с ключом total."""
    if isinstance(monthly_expenses, (int, float)):
        return float(monthly_expenses)
    if isinstance(monthly_expenses, dict):
        if "total" in monthly_expenses:
            return float(monthly_expenses["total"])
        return float(sum(float(v) for v in monthly_expenses.values()))
    raise ValueError(
        "monthly_expenses должен быть числом или словарём категорий "
        "(см. пример в config/cashflow.yaml)."
    )


def income_stability_key(value: str) -> str:
    value = (value or "stable").strip().lower()
    if value not in BUFFER_PCT:
        raise ValueError(
            f"income_stability должен быть 'stable' или 'variable', получено: {value!r}"
        )
    return value


def fmt(value: float) -> str:
    return f"{value:,.0f}".replace(",", " ")


# ----------------------------------------------------------------------------
# Расчёты
# ----------------------------------------------------------------------------

def compute_large_expense_reserves(planned_large_expenses: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Возвращает (сумма ежемесячных резервов, список предупреждений)."""
    monthly_reserve = 0.0
    warnings: list[str] = []
    for item in planned_large_expenses or []:
        name = item.get("name", "без названия")
        amount = float(item.get("amount", 0))
        months_until = item.get("months_until", 0)
        try:
            months_until = float(months_until)
        except (TypeError, ValueError):
            months_until = 0

        if months_until <= 0:
            monthly_reserve += amount
            warnings.append(
                f"«{name}»: срок наступил или не указан (months_until={months_until}) — "
                f"вся сумма {fmt(amount)} учтена как немедленный резерв этого месяца."
            )
        else:
            monthly_reserve += amount / months_until

    return monthly_reserve, warnings


def build_recommendation(
    income: float,
    expenses: float,
    stability: str,
    fund_current: float,
    fund_months_target: float,
    planned_large_expenses: list[dict[str, Any]],
    current_investment: float,
) -> dict[str, Any]:
    steps: list[str] = []

    free_cash_flow = income - expenses
    savings_rate = (free_cash_flow / income * 100) if income else 0.0

    fund_target = expenses * fund_months_target
    fund_gap = max(fund_target - fund_current, 0.0)
    fund_filled_pct = (fund_current / fund_target * 100) if fund_target > 0 else 100.0

    buffer_pct = BUFFER_PCT[stability]
    fund_priority_pct = FUND_PRIORITY_PCT[stability]

    large_reserve, large_warnings = compute_large_expense_reserves(planned_large_expenses)

    if free_cash_flow <= 0:
        steps.append(
            f"Шаг 0: свободный денежный поток = {fmt(income)} − {fmt(expenses)} = "
            f"{fmt(free_cash_flow)} <= 0. Инвестировать не из чего — сначала нужно "
            f"сократить расходы или увеличить доход."
        )
        return {
            "free_cash_flow": free_cash_flow,
            "savings_rate": savings_rate,
            "fund_target": fund_target,
            "fund_gap": fund_gap,
            "fund_filled_pct": fund_filled_pct,
            "buffer_pct": buffer_pct,
            "buffer_amount": 0.0,
            "large_reserve": large_reserve,
            "large_warnings": large_warnings,
            "fund_allocation": 0.0,
            "recommended_investment": 0.0,
            "deficit": -free_cash_flow + large_reserve,
            "steps": steps,
        }

    steps.append(
        f"Шаг 0: свободный денежный поток = доход {fmt(income)} − расходы {fmt(expenses)} "
        f"= {fmt(free_cash_flow)} (норма сбережений {savings_rate:.1f}%)."
    )

    buffer_amount = free_cash_flow * buffer_pct
    available = free_cash_flow - buffer_amount
    steps.append(
        f"Шаг 1: буфер на непредвиденное ({stability}, {buffer_pct*100:.0f}% от СДП) = "
        f"{fmt(buffer_amount)}. Доступно дальше: {fmt(available)}."
    )

    available_after_large = available - large_reserve
    if planned_large_expenses:
        steps.append(
            f"Шаг 2: резерв под крупные плановые траты = {fmt(large_reserve)}/мес "
            f"({len(planned_large_expenses)} шт.). Остаток: {fmt(available_after_large)}."
        )
    else:
        steps.append("Шаг 2: крупных плановых трат не указано, резерв = 0.")

    deficit = 0.0
    if available_after_large < 0:
        deficit = -available_after_large
        available_after_large = 0.0
        steps.append(
            f"Внимание: резерв под крупные траты превышает доступную сумму на "
            f"{fmt(deficit)} — на инвестиции и подушку в этом месяце ничего не остаётся."
        )

    fund_allocation = 0.0
    if fund_gap > 0 and available_after_large > 0:
        fund_allocation = min(available_after_large * fund_priority_pct, fund_gap)
        steps.append(
            f"Шаг 3: подушка не заполнена (не хватает {fmt(fund_gap)}, заполнено "
            f"{fund_filled_pct:.0f}%). Направляем в подушку {fund_priority_pct*100:.0f}% "
            f"остатка, но не больше недостающего: {fmt(fund_allocation)}."
        )
    elif fund_gap > 0:
        steps.append(
            f"Шаг 3: подушка не заполнена (не хватает {fmt(fund_gap)}), но доступной "
            f"суммы после шагов 1-2 не осталось — пополнение подушки в этом месяце "
            f"невозможно."
        )
    else:
        steps.append("Шаг 3: подушка безопасности уже заполнена по целевой сумме, пропускаем.")

    recommended_investment = max(available_after_large - fund_allocation, 0.0)
    steps.append(
        f"Шаг 4: рекомендованный взнос в инвестиции = {fmt(available_after_large)} − "
        f"{fmt(fund_allocation)} (в подушку) = {fmt(recommended_investment)}."
    )

    return {
        "free_cash_flow": free_cash_flow,
        "savings_rate": savings_rate,
        "fund_target": fund_target,
        "fund_gap": fund_gap,
        "fund_filled_pct": fund_filled_pct,
        "buffer_pct": buffer_pct,
        "buffer_amount": buffer_amount,
        "large_reserve": large_reserve,
        "large_warnings": large_warnings,
        "fund_allocation": fund_allocation,
        "recommended_investment": recommended_investment,
        "deficit": deficit,
        "steps": steps,
    }


# ----------------------------------------------------------------------------
# Вывод
# ----------------------------------------------------------------------------

def print_report(
    income: float,
    expenses: float,
    stability: str,
    fund_current: float,
    fund_months_target: float,
    current_investment: float,
    result: dict[str, Any],
) -> None:
    print("=" * 78)
    print("СОВЕТНИК ПО РАЗМЕРУ ИНВЕСТИЦИОННОГО ВЗНОСА")
    print("=" * 78)

    print("\n-- Текущая картина --")
    print(f"Доход (нетто, в месяц):        {fmt(income)}")
    print(f"Расходы (в месяц):             {fmt(expenses)}")
    print(f"Свободный денежный поток:      {fmt(result['free_cash_flow'])}")
    print(f"Норма сбережений:              {result['savings_rate']:.1f}%")
    print(f"Стабильность дохода:           {stability}")

    print("\n-- Подушка безопасности --")
    print(f"Целевой размер (расходы x {fund_months_target:g} мес.): {fmt(result['fund_target'])}")
    print(f"Текущий размер:                {fmt(fund_current)}")
    print(f"Заполненность цели:            {result['fund_filled_pct']:.0f}%")
    if result["fund_gap"] > 0:
        print(f"Не хватает до цели:            {fmt(result['fund_gap'])}")
    else:
        print("Подушка заполнена по цели.")

    if result["large_reserve"] > 0:
        print("\n-- Крупные плановые траты --")
        print(f"Суммарный резерв в месяц:      {fmt(result['large_reserve'])}")
        for w in result["large_warnings"]:
            print(f"  ! {w}")

    print("\n-- Расчёт рекомендации по шагам --")
    for step in result["steps"]:
        print(f"  {step}")

    print("\n-- Итог --")
    recommended = result["recommended_investment"]
    print(f"Рекомендованный ежемесячный взнос в инвестиции: {fmt(recommended)}")
    print(f"Буфер на непредвиденное (отдельно, не инвестируется): {fmt(result['buffer_amount'])}")

    if result.get("deficit", 0) > 0:
        print(
            f"\nВнимание: дефицит {fmt(result['deficit'])} — плановых трат и/или буфера "
            f"больше, чем позволяет денежный поток. Стоит пересмотреть сроки/суммы "
            f"крупных трат или сократить расходы."
        )

    diff = recommended - current_investment
    print(f"\nТекущий взнос:                 {fmt(current_investment)}")
    if abs(diff) < 1:
        print("Рекомендация: текущий взнос уже соответствует расчёту, менять не нужно.")
    elif diff > 0:
        print(f"Рекомендация: можно увеличить взнос на {fmt(diff)} (до {fmt(recommended)}).")
    else:
        print(f"Рекомендация: стоит уменьшить взнос на {fmt(-diff)} (до {fmt(recommended)}), "
              f"чтобы не жертвовать подушкой безопасности/резервами.")

    print(
        f"\nПодставьте это число ({fmt(recommended)}) в config/portfolio.yaml -> "
        f"monthly_contribution и запустите tools/forecast.py"
    )
    print("=" * 78)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Советник по размеру ежемесячного инвестиционного взноса "
        "на основе доходов и расходов."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Путь к YAML-конфигу (по умолчанию config/cashflow.yaml).",
    )
    parser.add_argument(
        "--income",
        type=float,
        default=None,
        help="Быстрая прикидка: чистый ежемесячный доход (переопределяет конфиг).",
    )
    parser.add_argument(
        "--expenses",
        type=float,
        default=None,
        help="Быстрая прикидка: суммарные ежемесячные расходы (переопределяет конфиг).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1
    except yaml.YAMLError as e:
        print(f"Ошибка чтения YAML: {e}", file=sys.stderr)
        return 1

    try:
        income = args.income if args.income is not None else float(config["monthly_income_net"])
        expenses = (
            args.expenses if args.expenses is not None else total_expenses(config["monthly_expenses"])
        )
        stability = income_stability_key(config.get("income_stability", "stable"))
        fund_current = float(config.get("emergency_fund_current", 0))
        fund_months_target = float(config.get("emergency_fund_months_target", 6))
        current_investment = float(config.get("current_monthly_investment", 0))
        planned_large_expenses = config.get("planned_large_expenses", []) or []
    except (KeyError, ValueError, TypeError) as e:
        print(f"Ошибка в конфигурации: {e}", file=sys.stderr)
        return 1

    result = build_recommendation(
        income=income,
        expenses=expenses,
        stability=stability,
        fund_current=fund_current,
        fund_months_target=fund_months_target,
        planned_large_expenses=planned_large_expenses,
        current_investment=current_investment,
    )

    print_report(
        income=income,
        expenses=expenses,
        stability=stability,
        fund_current=fund_current,
        fund_months_target=fund_months_target,
        current_investment=current_investment,
        result=result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
