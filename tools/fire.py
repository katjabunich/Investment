#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Калькулятор Coast/Barista FIRE: «когда можно перестать полноценно работать».

Инструмент отвечает на вопрос: через сколько лет можно прекратить полноценно
работать (перестать вносить взносы и начать снимать деньги на жизнь), чтобы с
достаточной вероятностью капитала хватило до глубокой старости.

Метод — тот же Монте-Карло движок, что и в tools/forecast.py: генерация матрицы
месячных доходностей (блочный бутстрэп или годовой резерв) и налог Box 3
переиспользуются напрямую (импорт, не копипаст).

Модель (три источника пассивного дохода поверх ликвидного портфеля):
  * Перебирается «год перехода» T = 1..20. До года T включительно вносятся
    ежемесячные взносы (с ежегодной индексацией). После года T взносы
    прекращаются и начинаются изъятия на жизнь.
  * Ликвидное ведро (IBKR, Box 3): взносы до перехода, изъятия после, налог
    Box 3 считается каждый год (в т.ч. в фазе изъятий).
  * Чистое ежемесячное изъятие после T = расходы − активные side_income
    (по возрасту) − (с 68) AOW − (с 68) нетто-аннуитет lijfrente. Не ниже нуля;
    излишек в ликвидное НЕ докладываем (консервативно). Всё в СЕГОДНЯШНИХ ценах,
    изъятие индексируется инфляцией (реальная величина постоянна).
  * Lijfrente-ведро (если включено): взносы до T, рост по тем же путям
    доходностей, БЕЗ Box 3; возврат налога от вычета докладывается в ликвидное
    ведро (до T). На 68 капитал конвертируется в аннуитет 68..payout_end_age
    (реальная ставка 1,5%), нетто = выплата × (1 − payout_tax_rate).
  * Успех пути = ликвидное ведро оставалось > 0 вплоть до life_expectancy.

Прогоняются два сценария распределения одного и того же взноса «из кармана»:
  * A — весь взнос в ликвидное ведро, lijfrente выключен;
  * B — часть взноса в lijfrente (+возврат налога в ликвидное), остальное как A.

Пример:
    python3 tools/fire.py --json reports/fire_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Переиспользуем движок forecast.py: конфиг, источник доходностей, генерацию
# матрицы доходностей, налог Box 3 и форматирование сумм.
from forecast import (  # noqa: E402
    SIM_START_YEAR,
    _box3_year_tax,
    _fmt,
    _generate_return_matrix,
    load_config,
    resolve_box3,
    resolve_return_source,
)

N_PATHS_DEFAULT = 10_000
SWITCH_YEARS_MAX = 20                    # перебираем переход через 1..20 лет
MILESTONE_SWITCH_YEARS = [5, 10, 15]     # для сводки P(успех) через 5/10/15 лет
SEED_DEFAULT = 42
THRESHOLDS = [0.85, 0.75]                # пороги надёжности для «самого раннего года»
LIJFRENTE_REAL_RATE = 0.015              # реальная ставка аннуитета lijfrente (68..end)


# ---------------------------------------------------------------------------
# Предрасчёт пассивных доходов свободной фазы (в СЕГОДНЯШНИХ ценах, нетто)
# ---------------------------------------------------------------------------
def _side_income_by_month(config: dict, n_months: int) -> np.ndarray:
    """Вектор суммы активных side_income по каждому месяцу (сегодняшние цены).

    Каждый поток длится, пока возраст < until_age. Возраст в месяце m =
    current_age + m//12 (сравнение по границе месяцев: m < (until_age-age)*12).
    """
    fire = config["fire"]
    current_age = int(fire["current_age"])
    out = np.zeros(n_months, dtype=float)
    for stream in fire.get("side_income", []):
        monthly = float(stream["monthly"])
        cutoff_m = (int(stream["until_age"]) - current_age) * 12
        hi = max(0, min(cutoff_m, n_months))
        out[:hi] += monthly
    return out


def _aow_by_month(config: dict, n_months: int) -> np.ndarray:
    """Вектор нетто-AOW по каждому месяцу (сегодняшние цены).

    Если aow.enabled=false — нули (возврат к «старой» модели без AOW).
    Начинается с месяца (start_age − current_age)*12.
    """
    fire = config["fire"]
    aow = fire.get("aow", {})
    out = np.zeros(n_months, dtype=float)
    if not aow.get("enabled", False):
        return out
    current_age = int(fire["current_age"])
    start_m = (int(aow["start_age"]) - current_age) * 12
    monthly = float(aow["monthly_amount"]) * float(aow.get("fraction", 1.0))
    lo = max(0, min(start_m, n_months))
    out[lo:] += monthly
    return out


# ---------------------------------------------------------------------------
# Симуляция одной комбинации (сценарий A/B × расходы × год перехода)
# ---------------------------------------------------------------------------
def simulate_fire_path_success(
    config: dict,
    ret: np.ndarray,
    box3: dict | None,
    expenses_month: float,
    liquid_contrib: float,
    lijfrente_enabled: bool,
    lijfrente_contrib: float,
    switch_year_T: int,
) -> float:
    """Вероятность успеха для одного «года перехода» T.

    ret — заранее сгенерированная матрица месячных доходностей (n_paths, n_months),
    одна и та же для всех T и сценариев (сопоставимость на общих путях).

    По месяцам m (0-based):
      * m < T*12 — накопление: в начале месяца в ликвидное ведро вносится взнос
        (индексация annual_contribution_growth); если lijfrente включён — ещё и
        возврат налога от его вычета. В lijfrente-ведро идёт свой взнос. Затем
        обоим ведрам начисляется месячная доходность пути.
      * m >= T*12 — изъятия: чистое изъятие = расходы − side_income − AOW −
        нетто-аннуитет lijfrente (всё в сегодняшних ценах), не ниже нуля,
        домноженное на инфляционный множитель. Взносы прекращаются.
    В конце каждого 12-месячного года из ЛИКВИДНОГО ведра вычитается налог Box 3
    (lijfrente от Box 3 освобождён). Путь провален, как только ликвидное <= 0.
    """
    n_paths, n_months = ret.shape
    fire = config["fire"]
    current_age = int(fire["current_age"])

    start_value = float(config["current_value"])
    infl = float(config.get("inflation", 0.0))
    monthly_infl = (1.0 + infl) ** (1.0 / 12.0) - 1.0
    g = float(config.get("annual_contribution_growth", 0.0))

    switch_m = switch_year_T * 12

    # Пассивные доходы свободной фазы (сегодняшние цены, нетто) — векторы по месяцам.
    side_month = _side_income_by_month(config, n_months)
    aow_month = _aow_by_month(config, n_months)

    # Параметры lijfrente-аннуитета (пенсионный возраст берём из aow.start_age = 68).
    lijf = fire.get("lijfrente", {})
    pension_age = int(fire.get("aow", {}).get("start_age", 68))
    payout_start_m = (pension_age - current_age) * 12
    payout_end_m = (int(lijf.get("payout_end_age", 95)) - current_age) * 12
    payout_tax = float(lijf.get("payout_tax_rate", 0.0))
    refund_rate = float(lijf.get("tax_refund_rate", 0.0))
    n_pay = max(1, payout_end_m - payout_start_m)
    r_m = (1.0 + LIJFRENTE_REAL_RATE) ** (1.0 / 12.0) - 1.0
    annuity_factor = (r_m / (1.0 - (1.0 + r_m) ** (-n_pay))) if r_m > 0 else 1.0 / n_pay
    # Множитель перевода номинала на момент 68 в сегодняшние (реальные) деньги.
    deflate_payout = 1.0 / (1.0 + monthly_infl) ** payout_start_m

    value = np.full(n_paths, start_value, dtype=float)
    failed = np.zeros(n_paths, dtype=bool)
    v_year_start = value.copy()          # ликвидное на начало текущего года

    lij_value = np.zeros(n_paths, dtype=float)   # lijfrente-ведро
    lijf_net = 0.0                               # нетто-аннуитет (скаляр 0 или вектор)
    annuitized = False

    for m in range(n_months):
        # --- Ликвидное ведро: денежный поток месяца ---
        if m < switch_m:
            cash = liquid_contrib * (1.0 + g) ** (m // 12)
            if lijfrente_enabled:
                # Возврат налога от вычета lijfrente докладывается в ликвидное.
                cash += lijfrente_contrib * (1.0 + g) ** (m // 12) * refund_rate
        else:
            in_payout = lijfrente_enabled and payout_start_m <= m < payout_end_m
            lijf_pay = lijf_net if in_payout else 0.0
            net_real = np.maximum(
                0.0, expenses_month - side_month[m] - aow_month[m] - lijf_pay
            )
            cash = -net_real * (1.0 + monthly_infl) ** m

        value = (value + cash) * (1.0 + ret[:, m])

        # --- Налог Box 3 (только ликвидное ведро) в конце года ---
        if box3 is not None and (m + 1) % 12 == 0:
            sim_year = (m + 1) // 12
            cal_year = SIM_START_YEAR + sim_year
            r_year = np.prod(1.0 + ret[:, m - 11 : m + 1], axis=1) - 1.0
            tax = _box3_year_tax(
                np.maximum(v_year_start, 0.0), r_year, cal_year, sim_year, box3
            )
            value = value - tax
            v_year_start = value.copy()

        failed |= value <= 0.0

        # --- Lijfrente-ведро: рост и конвертация в аннуитет на 68 ---
        if lijfrente_enabled and not annuitized:
            lij_cash = (
                lijfrente_contrib * (1.0 + g) ** (m // 12) if m < switch_m else 0.0
            )
            lij_value = (lij_value + lij_cash) * (1.0 + ret[:, m])
            if (m + 1) == payout_start_m:
                # Капитал на начало 68-го года -> реальный аннуитет 68..end.
                k_real = lij_value * deflate_payout
                lijf_net = k_real * annuity_factor * (1.0 - payout_tax)
                annuitized = True

    return float((~failed).mean())


def run_scenario(
    config: dict,
    ret: np.ndarray,
    box3: dict | None,
    scenario: dict,
    expenses_month: float,
    thresholds: list[float],
) -> dict:
    """Прогнать один сценарий (A/B) при одном уровне расходов по всем T = 1..20.

    scenario — dict с ключами key, label, liquid_contrib, lijfrente_enabled,
    lijfrente_contrib. Возвращает by_switch_year (список по T) и earliest —
    словарь {порог: самый ранний T с P(успех) >= порога, либо None}.
    """
    fire = config["fire"]
    current_age = int(fire["current_age"])

    by_switch_year = []
    earliest: dict[str, dict | None] = {f"{t:.2f}": None for t in thresholds}
    for T in range(1, SWITCH_YEARS_MAX + 1):
        p = simulate_fire_path_success(
            config, ret, box3, expenses_month,
            scenario["liquid_contrib"], scenario["lijfrente_enabled"],
            scenario["lijfrente_contrib"], T,
        )
        rec = {
            "T": T,
            "age": current_age + T,
            "calendar_year": SIM_START_YEAR + T,
            "p_success": round(p, 4),
        }
        by_switch_year.append(rec)
        for t in thresholds:
            key = f"{t:.2f}"
            if earliest[key] is None and p >= t:
                earliest[key] = dict(rec)

    return {
        "scenario_key": scenario["key"],
        "scenario_label": scenario["label"],
        "expenses": expenses_month,
        "liquid_contribution": scenario["liquid_contrib"],
        "lijfrente_enabled": scenario["lijfrente_enabled"],
        "lijfrente_contribution": scenario["lijfrente_contrib"],
        "by_switch_year": by_switch_year,
        "earliest_ok": earliest,
    }


# ---------------------------------------------------------------------------
# Вывод
# ---------------------------------------------------------------------------
def _p_at(by_switch_year: list[dict], T: int) -> dict | None:
    """Найти запись по году перехода T."""
    for rec in by_switch_year:
        if rec["T"] == T:
            return rec
    return None


def _earliest_txt(earliest: dict, t: float, currency: str) -> str:
    """Текстовое описание самого раннего года перехода для порога t."""
    eo = earliest.get(f"{t:.2f}")
    if eo is None:
        return f"за {SWITCH_YEARS_MAX} лет не достигается"
    return (f"через {eo['T']} лет — {eo['calendar_year']}, возраст {eo['age']} "
            f"(P = {eo['p_success']*100:.1f}%)")


def print_report(
    config: dict, results: list[dict], scenario_defs: list[dict],
    thresholds: list[float], n_paths: int, seed: int, box3: dict | None,
) -> None:
    """Напечатать сравнительный отчёт A vs B по всем уровням расходов (на русском)."""
    fire = config["fire"]
    currency = config.get("base_currency", "EUR")
    current_age = int(fire["current_age"])
    life_exp = int(fire["life_expectancy"])
    horizon = life_exp - current_age
    infl = float(config.get("inflation", 0.0))
    aow = fire.get("aow", {})

    print("=" * 78)
    print("FIRE-КАЛЬКУЛЯТОР: когда можно перестать полноценно работать (A vs B)")
    print("=" * 78)
    print(
        f"Профиль: возраст {current_age}, портфель {_fmt(config['current_value'])} "
        f"{currency}, взнос «из кармана» {_fmt(config['monthly_contribution'])} "
        f"{currency}/мес (+{config.get('annual_contribution_growth', 0)*100:.0f}%/год)"
    )
    # Пассивные доходы свободной фазы.
    streams = ", ".join(
        f"{s['name']} {_fmt(s['monthly'])} до {s['until_age']}"
        for s in fire.get("side_income", [])
    )
    print(f"Side income (нетто, сегодня): {streams}")
    if aow.get("enabled", False):
        aow_net = float(aow["monthly_amount"]) * float(aow.get("fraction", 1.0))
        print(
            f"AOW: с {aow['start_age']} лет, {_fmt(aow_net)} {currency}/мес нетто "
            f"(={_fmt(aow['monthly_amount'])} × {aow.get('fraction', 1.0)})"
        )
    else:
        print("AOW: выключен (старая модель)")
    print(f"Инфляция: {infl*100:.1f}%/год")
    if box3 is not None:
        mode = ("реформа с {}".format(box3["reform_start_year"])
                if box3.get("model_reform") else "текущий вменённый режим")
        print(f"Налог Box 3: ВКЛЮЧЁН на ликвидном ведре ({mode})")
    else:
        print("Налог Box 3: выключен")
    print(
        f"Дожитие до {life_exp} лет ({horizon} лет от сегодня) | "
        f"путей: {n_paths:,}".replace(",", " ")
        + f" | seed: {seed}"
    )
    print("Сценарии распределения одного и того же взноса «из кармана»:")
    for sc in scenario_defs:
        lij = (f"+ lijfrente {_fmt(sc['lijfrente_contrib'])} (+возврат налога в ликвидное)"
               if sc["lijfrente_enabled"] else "lijfrente выключен")
        print(f"  {sc['key']}: ликвидно {_fmt(sc['liquid_contrib'])} {currency}/мес, {lij}")

    # Сгруппировать по расходам, показать A и B рядом.
    expenses_list = sorted({r["expenses"] for r in results})
    for expenses in expenses_list:
        print("\n" + "-" * 78)
        print(f"РАСХОДЫ {_fmt(expenses)} {currency}/мес (сегодня)")
        print("-" * 78)
        for sc in scenario_defs:
            res = next(r for r in results
                       if r["expenses"] == expenses and r["scenario_key"] == sc["key"])
            print(f"  [{sc['key']}] {sc['label']}")
            for T in MILESTONE_SWITCH_YEARS:
                rec = _p_at(res["by_switch_year"], T)
                if rec is None:
                    continue
                print(
                    f"      Переход через {T:>2} лет ({rec['calendar_year']}, "
                    f"возраст {rec['age']}): P(успех) = {rec['p_success']*100:5.1f}%"
                )
            for t in thresholds:
                print(f"      Ранний год P>={t*100:.0f}%: {_earliest_txt(res['earliest_ok'], t, currency)}")

    # Итоговая сводка простыми словами.
    print("\n" + "=" * 78)
    print("СВОДКА: во сколько можно «выйти» (порог 85%), A vs B")
    print("=" * 78)
    for expenses in expenses_list:
        parts = []
        for sc in scenario_defs:
            res = next(r for r in results
                       if r["expenses"] == expenses and r["scenario_key"] == sc["key"])
            eo = res["earliest_ok"]["0.85"]
            parts.append(
                f"{sc['key']}: {eo['calendar_year']} (возраст {eo['age']})"
                if eo is not None else f"{sc['key']}: недостижимо за {SWITCH_YEARS_MAX} лет"
            )
        print(f"  Расходы {_fmt(expenses)} {currency}: " + " | ".join(parts))


def export_json(
    config: dict, results: list[dict], scenario_defs: list[dict],
    thresholds: list[float], n_paths: int, seed: int, used_monthly: bool,
    box3: dict | None, out_path: Path,
) -> None:
    """Сохранить результаты в машиночитаемый JSON (каждый сценарий с scenario_label)."""
    fire = config["fire"]
    result = {
        "params": {
            "base_currency": config.get("base_currency", "EUR"),
            "current_age": int(fire["current_age"]),
            "life_expectancy": int(fire["life_expectancy"]),
            "horizon_years": int(fire["life_expectancy"]) - int(fire["current_age"]),
            "current_value": float(config["current_value"]),
            "monthly_contribution": float(config["monthly_contribution"]),
            "annual_contribution_growth": float(
                config.get("annual_contribution_growth", 0.0)),
            "inflation": float(config.get("inflation", 0.0)),
            "side_income": fire.get("side_income", []),
            "aow": fire.get("aow", {}),
            "lijfrente": fire.get("lijfrente", {}),
            "thresholds": thresholds,
            "n_paths": int(n_paths),
            "seed": seed,
            "base_year": SIM_START_YEAR,
            "return_source": "monthly_block_bootstrap" if used_monthly
            else "annual_bootstrap_fallback",
            "box3_enabled": box3 is not None,
            "scenario_defs": scenario_defs,
        },
        "scenarios": results,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nJSON-результаты сохранены: {out_path}")


# ---------------------------------------------------------------------------
# Оркестрация
# ---------------------------------------------------------------------------
def build_scenario_defs(config: dict) -> list[dict]:
    """Построить определения сценариев A и B из конфига.

    A — весь взнос в ликвидное ведро, lijfrente выключен.
    B — часть взноса (lijfrente.monthly_contribution) в lijfrente, остальное в
        ликвидное; возврат налога докладывается в ликвидное (внутри симуляции).
    """
    total = float(config["monthly_contribution"])
    lij_contrib = float(config["fire"]["lijfrente"]["monthly_contribution"])
    return [
        {
            "key": "A",
            "label": "весь взнос в ликвидное, lijfrente выключен",
            "liquid_contrib": total,
            "lijfrente_enabled": False,
            "lijfrente_contrib": 0.0,
        },
        {
            "key": "B",
            "label": f"{_fmt(total - lij_contrib)} ликвидно + {_fmt(lij_contrib)} lijfrente",
            "liquid_contrib": total - lij_contrib,
            "lijfrente_enabled": True,
            "lijfrente_contrib": lij_contrib,
        },
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="FIRE-калькулятор: когда можно перестать полноценно работать.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--paths", type=int, default=N_PATHS_DEFAULT,
                        help="Число симуляционных путей Монте-Карло.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT,
                        help="Seed ГСЧ для воспроизводимости.")
    parser.add_argument("--json", type=str, default=None, dest="json_path",
                        help="Сохранить результаты в JSON по этому пути.")
    parser.add_argument("--no-box3", action="store_true", dest="no_box3",
                        help="Отключить налог Box 3 (расчёт до налога).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    t0 = time.perf_counter()
    args = parse_args(argv)
    config = load_config()
    if "fire" not in config:
        print("Ошибка: в config/portfolio.yaml отсутствует секция 'fire'.")
        return 1
    fire = config["fire"]

    n_paths = args.paths
    seed = args.seed
    life_exp = int(fire["life_expectancy"])
    current_age = int(fire["current_age"])
    years = life_exp - current_age

    expenses_list = [float(x) for x in fire["expenses_scenarios"]]
    scenario_defs = build_scenario_defs(config)

    box3 = resolve_box3(config, args.no_box3)
    monthly_src, annual_src, used_monthly = resolve_return_source(config)
    if not used_monthly:
        print("ВНИМАНИЕ: скачанных месячных серий нет — использую ГОДОВОЙ резерв "
              "(data/fallback_annual_returns.csv, приближение).")

    # Матрица доходностей генерируется ОДИН раз и переиспользуется для всех T,
    # сценариев и уровней расходов — так пути сопоставимы, а расчёт быстр.
    rng = np.random.default_rng(seed)
    ret = _generate_return_matrix(years, n_paths, rng, monthly_src, annual_src)

    results = []
    for expenses in expenses_list:
        for sc in scenario_defs:
            results.append(
                run_scenario(config, ret, box3, sc, expenses, THRESHOLDS)
            )

    print_report(config, results, scenario_defs, THRESHOLDS, n_paths, seed, box3)

    if args.json_path:
        export_json(config, results, scenario_defs, THRESHOLDS, n_paths, seed,
                    used_monthly, box3, Path(args.json_path))

    elapsed = time.perf_counter() - t0
    print(f"\nВремя выполнения: {elapsed:.1f} с")
    print("Помните: прошлые доходности не гарантируют будущих. Это не финсовет.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
