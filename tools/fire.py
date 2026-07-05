#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Калькулятор Coast/Barista FIRE: «когда можно перестать полноценно работать».

Инструмент отвечает на вопрос: через сколько лет можно прекратить полноценно
работать (перестать вносить взносы и начать снимать деньги на жизнь), чтобы с
достаточной вероятностью портфеля хватило до глубокой старости.

Метод — тот же Монте-Карло движок, что и в tools/forecast.py: генерация матрицы
месячных доходностей (блочный бутстрэп или годовой резерв) и налог Box 3
переиспользуются напрямую (импорт, не копипаст).

Модель:
  * Перебирается «год перехода» T = 1..20. До года T включительно человек вносит
    ежемесячный взнос (с ежегодной индексацией). После года T взносы прекращаются
    и начинаются ежемесячные изъятия на жизнь.
  * Чистое изъятие = (расходы − подработка), задаётся в СЕГОДНЯШНИХ ценах и
    индексируется инфляцией (реальная величина постоянна). Подработка длится до
    возраста part_time_until_age; после — изъятие равно полным расходам.
  * Налог Box 3 продолжает считаться каждый год и в фазе изъятий (по стоимости
    портфеля на начало года).
  * Путь считается успешным, если портфель оставался > 0 вплоть до life_expectancy.

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
# матрицы доходностей, налог Box 3, вектор взносов и форматирование сумм.
from forecast import (  # noqa: E402
    REPORTS_DIR,
    SIM_START_YEAR,
    _box3_year_tax,
    _contribution_vector,
    _fmt,
    _generate_return_matrix,
    load_config,
    resolve_box3,
    resolve_return_source,
)

N_PATHS_DEFAULT = 10_000
SWITCH_YEARS_MAX = 20          # перебираем переход через 1..20 лет
MILESTONE_SWITCH_YEARS = [5, 10, 15]  # для сводки P(успех) через 5/10/15 лет
SEED_DEFAULT = 42


# ---------------------------------------------------------------------------
# Симуляция одной комбинации (сценарий расходов × подработка × год перехода)
# ---------------------------------------------------------------------------
def simulate_fire_path_success(
    config: dict,
    ret: np.ndarray,
    box3: dict | None,
    expenses_month: float,
    part_time_month: float,
    switch_year_T: int,
) -> float:
    """Вероятность успеха для одного «года перехода» T.

    ret — заранее сгенерированная матрица месячных доходностей (n_paths, n_months),
    одна и та же для всех T и сценариев (сопоставимость на общих путях).

    Логика по месяцам m (0-based):
      * m < T*12  — фаза накопления: в начале месяца вносится взнос (индексация
        annual_contribution_growth), затем начисляется месячная доходность;
      * m >= T*12 — фаза изъятий: в начале месяца изымается (расходы − подработка)
        в сегодняшних ценах, домноженные на инфляционный множитель (реальная
        величина изъятия постоянна), затем начисляется доходность.
    В конце каждого 12-месячного года вычитается налог Box 3 (по стоимости на
    начало года). Путь помечается провальным, как только капитал станет <= 0.
    Возвращает долю успешных путей (портфель > 0 до конца горизонта).
    """
    n_paths, n_months = ret.shape
    fire = config["fire"]
    current_age = int(fire["current_age"])
    part_until = int(fire["part_time_until_age"])

    start_value = float(config["current_value"])
    infl = float(config.get("inflation", 0.0))
    monthly_infl = (1.0 + infl) ** (1.0 / 12.0) - 1.0
    g = float(config.get("annual_contribution_growth", 0.0))
    base_contrib = float(config["monthly_contribution"])

    switch_m = switch_year_T * 12
    # Подработка длится, пока возраст < part_until (в месяцах от старта).
    cutoff_m = (part_until - current_age) * 12

    value = np.full(n_paths, start_value, dtype=float)
    failed = np.zeros(n_paths, dtype=bool)
    v_year_start = value.copy()  # стоимость на начало текущего года

    for m in range(n_months):
        if m < switch_m:
            # Взнос (положительный денежный поток).
            cash = base_contrib * (1.0 + g) ** (m // 12)
        else:
            # Изъятие (отрицательный поток), индексированное инфляцией.
            pt = part_time_month if m < cutoff_m else 0.0
            net_real = expenses_month - pt
            cash = -net_real * (1.0 + monthly_infl) ** m

        value = (value + cash) * (1.0 + ret[:, m])

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

    return float((~failed).mean())


def run_scenario(
    config: dict,
    ret: np.ndarray,
    box3: dict | None,
    expenses_month: float,
    part_time_month: float,
    threshold: float,
) -> dict:
    """Прогнать один сценарий (расходы × подработка) по всем годам перехода 1..20.

    Возвращает словарь: expenses, part_time, by_switch_year (список по T),
    earliest_ok (самый ранний T с P(успех) >= threshold, либо None).
    """
    fire = config["fire"]
    current_age = int(fire["current_age"])

    by_switch_year = []
    earliest_ok = None
    for T in range(1, SWITCH_YEARS_MAX + 1):
        p = simulate_fire_path_success(
            config, ret, box3, expenses_month, part_time_month, T
        )
        rec = {
            "T": T,
            "age": current_age + T,
            "calendar_year": SIM_START_YEAR + T,
            "p_success": round(p, 4),
        }
        by_switch_year.append(rec)
        if earliest_ok is None and p >= threshold:
            earliest_ok = dict(rec)

    return {
        "expenses": expenses_month,
        "part_time": part_time_month,
        "by_switch_year": by_switch_year,
        "earliest_ok": earliest_ok,
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


def print_report(
    config: dict, scenarios: list[dict], threshold: float,
    n_paths: int, seed: int, box3: dict | None,
) -> None:
    """Напечатать компактный отчёт по всем сценариям на русском."""
    fire = config["fire"]
    currency = config.get("base_currency", "EUR")
    current_age = int(fire["current_age"])
    life_exp = int(fire["life_expectancy"])
    horizon = life_exp - current_age
    infl = float(config.get("inflation", 0.0))
    pt_until = int(fire["part_time_until_age"])

    print("=" * 72)
    print("FIRE-КАЛЬКУЛЯТОР: когда можно перестать полноценно работать")
    print("=" * 72)
    print(
        f"Профиль: возраст {current_age}, портфель {_fmt(config['current_value'])} "
        f"{currency}, взнос {_fmt(config['monthly_contribution'])} {currency}/мес "
        f"(+{config.get('annual_contribution_growth', 0)*100:.0f}%/год)"
    )
    print(f"Инфляция: {infl*100:.1f}%/год | подработка длится до {pt_until} лет")
    if box3 is not None:
        mode = ("реформа с {}".format(box3["reform_start_year"])
                if box3.get("model_reform") else "текущий вменённый режим")
        print(f"Налог Box 3: ВКЛЮЧЁН ({mode})")
    else:
        print("Налог Box 3: выключен")
    print(
        f"Дожитие до {life_exp} лет ({horizon} лет от сегодня) | "
        f"путей: {n_paths:,}".replace(",", " ")
        + f" | seed: {seed} | порог успеха: {threshold*100:.0f}%"
    )
    print(
        "Расходы/подработка заданы в СЕГОДНЯШНИХ ценах (изъятия индексируются "
        "инфляцией)."
    )

    for sc in scenarios:
        pt = sc["part_time"]
        pt_label = (f"подработка {_fmt(pt)} {currency}/мес"
                    if pt > 0 else "БЕЗ подработки (полный FIRE)")
        print("\n" + "-" * 72)
        print(f"Расходы {_fmt(sc['expenses'])} {currency}/мес | {pt_label}")
        print("-" * 72)
        for T in MILESTONE_SWITCH_YEARS:
            rec = _p_at(sc["by_switch_year"], T)
            if rec is None:
                continue
            print(
                f"  Переход через {T:>2} лет (в {rec['calendar_year']}, "
                f"возраст {rec['age']}): P(успех) = {rec['p_success']*100:5.1f}%"
            )
        eo = sc["earliest_ok"]
        if eo is not None:
            print(
                f"  ==> Самый ранний год перехода с P >= {threshold*100:.0f}%: "
                f"через {eo['T']} лет — в {eo['calendar_year']}, возраст "
                f"{eo['age']} (P = {eo['p_success']*100:.1f}%)"
            )
        else:
            last = sc["by_switch_year"][-1]
            print(
                f"  ==> За {SWITCH_YEARS_MAX} лет порог {threshold*100:.0f}% не "
                f"достигается (максимум P = {last['p_success']*100:.1f}% при "
                f"переходе через {last['T']} лет)"
            )

    # Понятная сводка по всем комбинациям.
    print("\n" + "=" * 72)
    print("СВОДКА (простыми словами)")
    print("=" * 72)
    for sc in scenarios:
        pt = sc["part_time"]
        eo = sc["earliest_ok"]
        pt_txt = (f"подработке {_fmt(pt)} {currency}" if pt > 0
                  else "без подработки")
        if eo is not None:
            print(
                f"  При расходах {_fmt(sc['expenses'])} {currency} и {pt_txt}: "
                f"можно в {eo['calendar_year']} (возраст {eo['age']}), "
                f"вероятность успеха {eo['p_success']*100:.0f}%."
            )
        else:
            print(
                f"  При расходах {_fmt(sc['expenses'])} {currency} и {pt_txt}: "
                f"за {SWITCH_YEARS_MAX} лет надёжно (>= {threshold*100:.0f}%) "
                f"выйти не получается."
            )


def export_json(
    config: dict, scenarios: list[dict], threshold: float,
    n_paths: int, seed: int, used_monthly: bool, box3: dict | None,
    out_path: Path,
) -> None:
    """Сохранить результаты в машиночитаемый JSON."""
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
            "part_time_income": float(fire["part_time_income"]),
            "part_time_until_age": int(fire["part_time_until_age"]),
            "success_threshold": float(threshold),
            "n_paths": int(n_paths),
            "seed": seed,
            "base_year": SIM_START_YEAR,
            "return_source": "monthly_block_bootstrap" if used_monthly
            else "annual_bootstrap_fallback",
            "box3_enabled": box3 is not None,
        },
        "scenarios": scenarios,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nJSON-результаты сохранены: {out_path}")


# ---------------------------------------------------------------------------
# Оркестрация
# ---------------------------------------------------------------------------
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
    threshold = float(fire["success_threshold"])
    life_exp = int(fire["life_expectancy"])
    current_age = int(fire["current_age"])
    years = life_exp - current_age

    part_time = float(fire["part_time_income"])
    expenses_list = [float(x) for x in fire["expenses_scenarios"]]
    # 6 комбинаций: каждый уровень расходов × {с подработкой, без подработки}.
    combos = [(e, part_time) for e in expenses_list] + \
             [(e, 0.0) for e in expenses_list]

    box3 = resolve_box3(config, args.no_box3)
    monthly_src, annual_src, used_monthly = resolve_return_source(config)
    if not used_monthly:
        print("ВНИМАНИЕ: скачанных месячных серий нет — использую ГОДОВОЙ резерв "
              "(data/fallback_annual_returns.csv, приближение).")

    # Матрица доходностей генерируется ОДИН раз и переиспользуется для всех T и
    # сценариев — так пути сопоставимы, а расчёт быстр.
    rng = np.random.default_rng(seed)
    ret = _generate_return_matrix(years, n_paths, rng, monthly_src, annual_src)

    scenarios = []
    for expenses_month, pt_month in combos:
        scenarios.append(
            run_scenario(config, ret, box3, expenses_month, pt_month, threshold)
        )

    print_report(config, scenarios, threshold, n_paths, seed, box3)

    if args.json_path:
        export_json(config, scenarios, threshold, n_paths, seed,
                    used_monthly, box3, Path(args.json_path))

    elapsed = time.perf_counter() - t0
    print(f"\nВремя выполнения: {elapsed:.1f} с")
    print("Помните: прошлые доходности не гарантируют будущих. Это не финсовет.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
