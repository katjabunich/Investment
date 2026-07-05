#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Долгосрочный прогноз капитала для пассивного инвестора (Монте-Карло).

Инструмент моделирует рост портфеля методом бутстрэпа исторических доходностей:

  * если в data/ есть скачанные месячные серии (см. tools/fetch_data.py) —
    используется БЛОЧНЫЙ бутстрэп месячных доходностей (блоки 12-24 мес.),
    что сохраняет автокорреляцию и кластеризацию волатильности;
  * иначе — резервный годовой бутстрэп по data/fallback_annual_returns.csv
    (с предупреждением о меньшей точности).

Учитываются: стартовый капитал, ежемесячные взносы с ежегодной индексацией,
инфляция (для перевода в реальные деньги). На выходе — таблица перцентилей
капитала по вехам, вероятность достичь цели, сравнение с «наивным» прогнозом
под фиксированную ставку и график-«веер» в reports/forecast_fan.png.

Примеры:
    python3 tools/forecast.py --years 20 --seed 42
    python3 tools/forecast.py --goal 750000 --contribution 1500
    python3 tools/forecast.py --compare 500,1000,2000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без графического окна — только сохранение в файл
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "portfolio.yaml"
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FALLBACK_CSV = DATA_DIR / "fallback_annual_returns.csv"

# Соответствие прокси-серии -> колонка в fallback_annual_returns.csv.
FALLBACK_COLUMNS = {
    "msci_world": "msci_world_total_return",
    "sp500": "sp500_total_return",
    "nasdaq100": "nasdaq100_total_return",
    "world_ex_us": "world_ex_us_total_return",
    "em": "em_total_return",
    "small_value": "small_value_total_return",
}

N_PATHS_DEFAULT = 10_000
BLOCK_MIN, BLOCK_MAX = 12, 24  # длина блоков месячного бутстрэпа (в месяцах)
PERCENTILES = [5, 25, 50, 75, 95]
MILESTONES = [5, 10, 15, 20]  # вехи в годах для итоговой таблицы
SIM_START_YEAR = 2026  # календарный год начала симуляции (старт — середина 2026)


# ---------------------------------------------------------------------------
# Загрузка конфигурации и исторических доходностей
# ---------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict:
    """Прочитать YAML-конфиг портфеля."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_box3(config: dict, disabled: bool) -> dict | None:
    """Вернуть параметры Box 3 (dict) или None, если налог отключён.

    disabled=True (флаг --no-box3) или box3.enabled=false в конфиге -> None.
    """
    box3 = config.get("box3")
    if disabled or not box3 or not box3.get("enabled", False):
        return None
    return box3


def _read_monthly_series(series: str) -> pd.Series | None:
    """Прочитать месячные цены data/<series>.csv и вернуть месячные доходности.

    Возвращает Series доходностей (доли) или None, если файла нет/он пустой.
    """
    path = DATA_DIR / f"{series}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "close" not in df.columns or len(df) < 24:
        return None
    df = df.dropna(subset=["close"]).sort_values("date")
    returns = df["close"].astype(float).pct_change().dropna()
    if returns.empty:
        return None
    return returns.reset_index(drop=True)


def build_portfolio_monthly_returns(config: dict) -> pd.Series | None:
    """Собрать взвешенные месячные доходности портфеля из скачанных серий.

    Все прокси-серии holdings должны присутствовать в data/, иначе возвращается
    None (тогда используется годовой резерв). Серии выравниваются по длине
    (берётся общий хвост — самые свежие N наблюдений).
    """
    holdings = config.get("holdings", [])
    per_proxy: dict[str, pd.Series] = {}
    for h in holdings:
        proxy = h["proxy"]
        if proxy not in per_proxy:
            s = _read_monthly_series(proxy)
            if s is None:
                return None  # хотя бы одной серии нет — уходим в резерв
            per_proxy[proxy] = s

    min_len = min(len(s) for s in per_proxy.values())
    if min_len < 24:
        return None

    weighted = np.zeros(min_len)
    total_weight = sum(h["weight"] for h in holdings)
    for h in holdings:
        w = h["weight"] / total_weight
        tail = per_proxy[h["proxy"]].to_numpy()[-min_len:]
        weighted += w * tail
    return pd.Series(weighted)


def build_portfolio_annual_returns(config: dict) -> np.ndarray:
    """Собрать взвешенные ГОДОВЫЕ доходности портфеля из резервного датасета.

    Разные прокси начинаются в разные годы (у некоторых серий есть пропуски —
    NaN — в ранние годы). Чтобы веса портфеля были согласованы по каждому году,
    используем ТОЛЬКО те годы, где есть данные по ВСЕМ нужным прокси одновременно
    (общее пересечение окон). Об этом печатается предупреждение.
    """
    if not FALLBACK_CSV.exists():
        raise FileNotFoundError(
            f"Нет ни скачанных серий, ни резерва {FALLBACK_CSV}."
        )
    df = pd.read_csv(FALLBACK_CSV, comment="#")
    holdings = config.get("holdings", [])
    total_weight = sum(h["weight"] for h in holdings)

    # Определяем колонки, участвующие в портфеле, и их фактическое покрытие.
    used_cols: list[tuple[str, float]] = []
    for h in holdings:
        proxy = h["proxy"]
        col = FALLBACK_COLUMNS.get(proxy)
        if col is None or col not in df.columns:
            # Неизвестный прокси в резерве — используем мировой рынок как замену.
            col = "msci_world_total_return"
        w = h["weight"] / total_weight
        used_cols.append((col, w))

    unique_cols = sorted({c for c, _ in used_cols})
    # Маска годов, где по ВСЕМ нужным колонкам есть данные (нет NaN).
    complete = df[unique_cols].notna().all(axis=1)
    sub = df.loc[complete].reset_index(drop=True)
    if sub.empty:
        raise ValueError(
            "В резервном датасете нет ни одного года с данными по всем "
            f"прокси портфеля: {unique_cols}"
        )

    # Диагностика: какой год стал общим стартом и из-за какой(их) серии(й).
    full_start = int(df["year"].iloc[0])
    used_start = int(sub["year"].iloc[0])
    used_end = int(sub["year"].iloc[-1])
    n_years = len(sub)
    if used_start > full_start:
        # Ищем прокси, которые «обрезали» историю (их первый непустой год позже).
        latecomers = []
        for col in unique_cols:
            first_valid = df.loc[df[col].notna(), "year"]
            if not first_valid.empty and int(first_valid.iloc[0]) == used_start:
                latecomers.append(col)
        why = ", ".join(latecomers) if latecomers else "нехватка ранних данных"
        print(
            f"ВНИМАНИЕ: годовая история портфеля используется с {used_start} "
            f"по {used_end} ({n_years} лет).\n"
            f"  Более ранние годы ({full_start}-{used_start - 1}) отброшены, "
            f"т.к. по части прокси нет данных (позже всех начинается: {why})."
        )
    else:
        print(
            f"Годовая история портфеля: {used_start}-{used_end} ({n_years} лет)."
        )

    weighted = np.zeros(len(sub))
    for col, w in used_cols:
        weighted += w * sub[col].astype(float).to_numpy()
    return weighted


# ---------------------------------------------------------------------------
# Симуляция Монте-Карло
# ---------------------------------------------------------------------------
def _block_bootstrap_paths(
    monthly_returns: np.ndarray,
    n_months: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Сгенерировать матрицу месячных доходностей блочным бутстрэпом.

    Возвращает массив формы (n_paths, n_months). Блоки случайной длины
    12-24 мес. склеиваются встык, что сохраняет автокорреляцию рынка.
    """
    src = monthly_returns
    src_len = len(src)
    out = np.empty((n_paths, n_months), dtype=float)

    for p in range(n_paths):
        filled = 0
        row = out[p]
        while filled < n_months:
            block_len = int(rng.integers(BLOCK_MIN, BLOCK_MAX + 1))
            start = int(rng.integers(0, src_len))
            # Циклический срез, чтобы не упираться в конец истории.
            idx = (start + np.arange(block_len)) % src_len
            take = min(block_len, n_months - filled)
            row[filled : filled + take] = src[idx[:take]]
            filled += take
    return out


def _annual_bootstrap_to_monthly(
    annual_returns: np.ndarray,
    n_months: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Резервный путь: бутстрэп ГОДОВЫХ доходностей, разложенных на месяцы.

    Каждый год выбирается случайно (с возвращением) и переводится в
    эквивалентную месячную ставку (1+r)**(1/12)-1, применяемую 12 месяцев.
    """
    n_years = int(np.ceil(n_months / 12))
    out = np.empty((n_paths, n_months), dtype=float)
    for p in range(n_paths):
        picks = rng.integers(0, len(annual_returns), size=n_years)
        monthly = (1.0 + annual_returns[picks]) ** (1.0 / 12.0) - 1.0
        expanded = np.repeat(monthly, 12)[:n_months]
        out[p] = expanded
    return out


def _generate_return_matrix(
    years: int,
    n_paths: int,
    rng: np.random.Generator,
    monthly_returns: np.ndarray | None,
    annual_returns: np.ndarray | None,
) -> np.ndarray:
    """Сгенерировать матрицу месячных доходностей (n_paths, n_months)."""
    n_months = years * 12
    if monthly_returns is not None:
        return _block_bootstrap_paths(monthly_returns, n_months, n_paths, rng)
    return _annual_bootstrap_to_monthly(annual_returns, n_months, n_paths, rng)


def _contribution_vector(
    n_months: int, monthly_contribution: float, contrib_growth: float
) -> np.ndarray:
    """Вектор взносов по месяцам с ежегодной индексацией."""
    contrib = np.empty(n_months)
    for m in range(n_months):
        contrib[m] = monthly_contribution * (1.0 + contrib_growth) ** (m // 12)
    return contrib


def _box3_year_tax(
    v_start: np.ndarray,
    r_year: np.ndarray,
    cal_year: int,
    sim_year: int,
    box3: dict,
) -> np.ndarray:
    """Налог Box 3 за один год для всех путей (вектор >= 0).

    v_start — стоимость портфеля на начало года (после налога прошлого года);
    r_year  — фактическая доходность инвестиций за этот год этого пути
              (сложный процент 12 месяцев, БЕЗ учёта взносов);
    cal_year — календарный год (SIM_START_YEAR + sim_year);
    sim_year — номер года симуляции (1..N), нужен для индексации льготы.
    """
    zeros = np.zeros_like(v_start)
    # До первого налогового года налога нет (2026 уплачен вне модели).
    if cal_year < int(box3["first_tax_year"]):
        return zeros

    use_reform = bool(box3["model_reform"]) and cal_year >= int(
        box3["reform_start_year"]
    )

    if use_reform:
        # Реформа: налог с фактического прироста стоимости за год.
        # Прирост = v_start * r_year (взносы приростом не считаются).
        # Убытки -> 0, без переноса на будущие годы (консервативно).
        gain = v_start * r_year
        taxable = np.maximum(0.0, gain - float(box3["reform_tax_free_result"]))
        return float(box3["reform_tax_rate"]) * taxable

    # Текущий режим: вменённая доходность от капитала выше льготы.
    allowance = float(box3["tax_free_allowance"]) * (
        1.0 + float(box3["allowance_growth"])
    ) ** sim_year
    base_above = np.maximum(0.0, v_start - allowance)  # облагаемая база
    deemed_income = float(box3["deemed_return"]) * base_above

    if box3["tegenbewijs"]:
        # Фактический доход по облагаемой доле базы:
        # v_start * r_year * (base_above / v_start) = r_year * base_above.
        actual_income = r_year * base_above
        income = np.minimum(deemed_income, actual_income)
    else:
        income = deemed_income

    income = np.maximum(0.0, income)
    return float(box3["tax_rate"]) * income


def _build_capital(
    ret: np.ndarray,
    start_value: float,
    contrib: np.ndarray,
    box3: dict | None,
) -> np.ndarray:
    """Построить матрицу капитала (n_paths, n_months+1) из готовой матрицы доходностей.

    Если box3 задан — в конце каждого 12-месячного года считается и вычитается
    налог Box 3 (это уменьшает базу для дальнейшего роста). Векторизовано по путям.
    """
    n_paths, n_months = ret.shape
    capital = np.empty((n_paths, n_months + 1))
    capital[:, 0] = start_value
    value = np.full(n_paths, start_value)
    v_year_start = value.copy()  # стоимость на начало текущего года
    for m in range(n_months):
        # Взнос вносится в начале месяца, затем начисляется месячная доходность.
        value = (value + contrib[m]) * (1.0 + ret[:, m])
        capital[:, m + 1] = value
        if box3 is not None and (m + 1) % 12 == 0:
            sim_year = (m + 1) // 12  # номер завершившегося года (1..N)
            cal_year = SIM_START_YEAR + sim_year
            # Фактическая доходность года = сложный процент 12 месяцев пути.
            r_year = np.prod(1.0 + ret[:, m - 11 : m + 1], axis=1) - 1.0
            tax = _box3_year_tax(v_year_start, r_year, cal_year, sim_year, box3)
            value = value - tax
            capital[:, m + 1] = value
            v_year_start = value.copy()
    return capital


def simulate(
    config: dict,
    years: int,
    n_paths: int,
    rng: np.random.Generator,
    monthly_contribution: float,
    monthly_returns: np.ndarray | None,
    annual_returns: np.ndarray | None,
    box3: dict | None = None,
) -> np.ndarray:
    """Прогнать симуляцию капитала помесячно.

    Возвращает матрицу номинального капитала формы (n_paths, n_months+1),
    где столбец 0 — стартовый капитал, далее — на конец каждого месяца.
    Если box3 задан — капитал показан ПОСЛЕ вычета налога Box 3.
    """
    n_months = years * 12
    start_value = float(config["current_value"])
    contrib_growth = float(config.get("annual_contribution_growth", 0.0))
    ret = _generate_return_matrix(
        years, n_paths, rng, monthly_returns, annual_returns
    )
    contrib = _contribution_vector(n_months, monthly_contribution, contrib_growth)
    return _build_capital(ret, start_value, contrib, box3)


def simulate_with_shadow(
    config: dict,
    years: int,
    n_paths: int,
    rng: np.random.Generator,
    monthly_contribution: float,
    monthly_returns: np.ndarray | None,
    annual_returns: np.ndarray | None,
    box3: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Симуляция с налогом и «теневая» без налога НА ТЕХ ЖЕ путях.

    Возвращает (capital_после_налога, capital_без_налога). Если box3 is None —
    обе матрицы совпадают (налог не применяется).
    """
    n_months = years * 12
    start_value = float(config["current_value"])
    contrib_growth = float(config.get("annual_contribution_growth", 0.0))
    ret = _generate_return_matrix(
        years, n_paths, rng, monthly_returns, annual_returns
    )
    contrib = _contribution_vector(n_months, monthly_contribution, contrib_growth)
    capital_tax = _build_capital(ret, start_value, contrib, box3)
    capital_pretax = _build_capital(ret, start_value, contrib, None)
    return capital_tax, capital_pretax


# ---------------------------------------------------------------------------
# Аналитика и вывод
# ---------------------------------------------------------------------------
def deflator(config: dict, years: int) -> np.ndarray:
    """Множители перевода номинала в реальные деньги по каждому месяцу (0..N)."""
    infl = float(config.get("inflation", 0.0))
    n_months = years * 12
    monthly_infl = (1.0 + infl) ** (1.0 / 12.0) - 1.0
    months = np.arange(n_months + 1)
    return 1.0 / (1.0 + monthly_infl) ** months


def naive_projection(
    config: dict, years: int, monthly_contribution: float, annual_rate: float
) -> float:
    """Детерминированный «наивный» прогноз под фиксированную годовую ставку."""
    n_months = years * 12
    monthly_rate = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
    contrib_growth = float(config.get("annual_contribution_growth", 0.0))
    value = float(config["current_value"])
    for m in range(n_months):
        c = monthly_contribution * (1.0 + contrib_growth) ** (m // 12)
        value = (value + c) * (1.0 + monthly_rate)
    return value


def _fmt(x: float) -> str:
    """Формат денежной суммы: с разделителями тысяч и без копеек."""
    return f"{x:,.0f}".replace(",", " ")


def total_contributed(
    config: dict, years: int, monthly_contribution: float
) -> float:
    """Сколько всего внесено взносов за горизонт (без учёта стартового капитала)."""
    n_months = years * 12
    g = float(config.get("annual_contribution_growth", 0.0))
    return sum(monthly_contribution * (1.0 + g) ** (m // 12) for m in range(n_months))


def print_percentile_table(
    capital: np.ndarray, deflate: np.ndarray, currency: str, years: int
) -> None:
    """Напечатать таблицу перцентилей капитала по вехам (номинал и реал)."""
    print(f"\nПерцентили капитала по вехам ({currency}):")
    header = "  Год | " + " | ".join(f"P{p:<2}" for p in PERCENTILES)
    for label, mult in (("НОМИНАЛЬНЫЕ (в деньгах будущего)", None),
                        ("РЕАЛЬНЫЕ (в сегодняшней покупательной способности)", deflate)):
        print(f"\n  {label}:")
        print("  " + "-" * 68)
        print("  {:>4} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
            "лет", *[f"P{p}" for p in PERCENTILES]))
        print("  " + "-" * 68)
        for y in MILESTONES:
            if y > years:
                continue
            col = y * 12
            vals = capital[:, col]
            if mult is not None:
                vals = vals * mult[col]
            pct = np.percentile(vals, PERCENTILES)
            print("  {:>4} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
                y, *[_fmt(v) for v in pct]))
        # Финальный горизонт, если он не совпал с вехами.
        if years not in MILESTONES:
            vals = capital[:, years * 12]
            if mult is not None:
                vals = vals * mult[years * 12]
            pct = np.percentile(vals, PERCENTILES)
            print("  {:>4} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}".format(
                years, *[_fmt(v) for v in pct]))


def print_goal_probability(
    capital: np.ndarray, deflate: np.ndarray, goal: float, years: int, currency: str
) -> None:
    """Вероятность достичь целевой суммы к концу горизонта (номинал и реал)."""
    final_nom = capital[:, years * 12]
    final_real = final_nom * deflate[years * 12]
    p_nom = float((final_nom >= goal).mean()) * 100
    p_real = float((final_real >= goal).mean()) * 100
    print(f"\nВероятность достичь цели {_fmt(goal)} {currency} за {years} лет:")
    print(f"  в номинальных деньгах: {p_nom:5.1f}%")
    print(f"  в реальных деньгах:    {p_real:5.1f}%")


def print_naive_comparison(
    config: dict, capital: np.ndarray, years: int,
    monthly_contribution: float, currency: str
) -> None:
    """Сравнить медиану симуляции с наивным прогнозом под 5% и 7%."""
    median_final = float(np.percentile(capital[:, years * 12], 50))
    print(f"\nСравнение с «наивным» прогнозом под фиксированную ставку "
          f"(номинал, {years} лет):")
    print(f"  Медиана симуляции (P50):        {_fmt(median_final):>14} {currency}")
    for rate in (0.05, 0.07):
        naive = naive_projection(config, years, monthly_contribution, rate)
        print(f"  Наивный прогноз под {rate*100:.0f}% годовых: "
              f"{_fmt(naive):>14} {currency}")


def print_box3_comparison(
    capital: np.ndarray, capital_pretax: np.ndarray, deflate: np.ndarray,
    years: int, currency: str, box3: dict,
) -> None:
    """Сравнить медиану без налога и после налога Box 3 на конце горизонта."""
    col = years * 12
    med_tax_nom = float(np.percentile(capital[:, col], 50))
    med_pre_nom = float(np.percentile(capital_pretax[:, col], 50))
    med_tax_real = float(np.percentile(capital[:, col] * deflate[col], 50))
    med_pre_real = float(np.percentile(capital_pretax[:, col] * deflate[col], 50))
    cost_nom = med_pre_nom - med_tax_nom
    cost_real = med_pre_real - med_tax_real
    drop = (1.0 - med_tax_real / med_pre_real) * 100 if med_pre_real else 0.0
    mode = ("реформа с {} (налог на прирост)".format(box3["reform_start_year"])
            if box3.get("model_reform")
            else "текущий вменённый режим весь горизонт")
    print(f"\nНалог Box 3 (Нидерланды), режим: {mode}")
    print(f"  Медиана P50 через {years} лет БЕЗ налога:    "
          f"{_fmt(med_pre_nom):>14} {currency} (номинал) | "
          f"{_fmt(med_pre_real):>14} {currency} (реал)")
    print(f"  Медиана P50 через {years} лет ПОСЛЕ налога:  "
          f"{_fmt(med_tax_nom):>14} {currency} (номинал) | "
          f"{_fmt(med_tax_real):>14} {currency} (реал)")
    print(f"  Стоимость Box 3 за {years} лет (разница медиан): "
          f"{_fmt(cost_nom):>14} {currency} (номинал) | "
          f"{_fmt(cost_real):>14} {currency} (реал)")
    print(f"  Медиана после налога ниже безналоговой на {drop:.1f}%.")


def make_fan_chart(
    capital: np.ndarray, years: int, currency: str, out_path: Path,
    monthly_contribution: float
) -> None:
    """Нарисовать график-«веер» перцентилей номинального капитала по годам."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    n_months = years * 12
    x = np.arange(n_months + 1) / 12.0

    pcts = {p: np.percentile(capital, p, axis=0) for p in PERCENTILES}

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, pcts[5], pcts[95], alpha=0.20, color="#1f77b4",
                    label="P5–P95 (широкий диапазон)")
    ax.fill_between(x, pcts[25], pcts[75], alpha=0.35, color="#1f77b4",
                    label="P25–P75 (вероятный диапазон)")
    ax.plot(x, pcts[50], color="#08306b", linewidth=2.0, label="P50 (медиана)")

    ax.set_title(
        f"Прогноз капитала на {years} лет (взнос {_fmt(monthly_contribution)} "
        f"{currency}/мес)\nМонте-Карло, блочный бутстрэп исторических доходностей",
        fontsize=12,
    )
    ax.set_xlabel("Годы")
    ax.set_ylabel(f"Капитал, {currency} (номинал)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: _fmt(v))
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nГрафик-веер сохранён: {out_path}")


def _percentile_dict(values: np.ndarray) -> dict:
    """Вернуть словарь {p5,p25,p50,p75,p95} перцентилей для массива значений."""
    pct = np.percentile(values, PERCENTILES)
    return {f"p{p}": float(round(v, 2)) for p, v in zip(PERCENTILES, pct)}


def export_json(
    config: dict, capital: np.ndarray, deflate: np.ndarray, years: int,
    n_paths: int, seed: int | None, monthly_contribution: float,
    goal: float | None, used_monthly: bool, out_path: Path,
    box3: dict | None = None, capital_pretax: np.ndarray | None = None,
) -> None:
    """Сохранить результаты симуляции в машиночитаемый JSON.

    Структура: params, milestones (перцентили по каждому году 1..N в номинале и
    реале), goal (вероятности достижения), naive (наивные прогнозы под 5% и 7%),
    contributed_total и fan (перцентили реального капитала по каждому году 0..N).

    Если задан box3 — основные milestones/fan показаны ПОСЛЕ налога; добавляются
    секция "box3" (параметры и стоимость налога) и "fan_pretax" (медиана без
    налога для линии сравнения на графике).
    """
    currency = config.get("base_currency", "EUR")

    # Вехи — по каждому году горизонта (1..N): номинал и реал.
    milestones = []
    for y in range(1, years + 1):
        col = y * 12
        nominal = capital[:, col]
        real = nominal * deflate[col]
        milestones.append({
            "year": y,
            "nominal": _percentile_dict(nominal),
            "real": _percentile_dict(real),
        })

    # Вероятность достижения цели к концу горизонта.
    goal_block = None
    if goal is not None:
        final_nom = capital[:, years * 12]
        final_real = final_nom * deflate[years * 12]
        goal_block = {
            "amount": float(goal),
            "prob_nominal": float(round((final_nom >= goal).mean() * 100, 2)),
            "prob_real": float(round((final_real >= goal).mean() * 100, 2)),
        }

    # Веер — перцентили РЕАЛЬНОГО капитала по каждому году 0..N.
    fan_years = list(range(years + 1))
    fan: dict = {"years": fan_years}
    for p in PERCENTILES:
        fan[f"real_p{p}"] = []
    for y in fan_years:
        col = y * 12
        real = capital[:, col] * deflate[col]
        pct = np.percentile(real, PERCENTILES)
        for p, v in zip(PERCENTILES, pct):
            fan[f"real_p{p}"].append(float(round(v, 2)))

    result = {
        "params": {
            "base_currency": currency,
            "current_value": float(config["current_value"]),
            "monthly_contribution": float(monthly_contribution),
            "annual_contribution_growth": float(
                config.get("annual_contribution_growth", 0.0)),
            "inflation": float(config.get("inflation", 0.0)),
            "years": int(years),
            "n_paths": int(n_paths),
            "seed": seed,
            "return_source": "monthly_block_bootstrap" if used_monthly
            else "annual_bootstrap_fallback",
        },
        "milestones": milestones,
        "goal": goal_block,
        "naive": {
            "rate5": float(round(
                naive_projection(config, years, monthly_contribution, 0.05), 2)),
            "rate7": float(round(
                naive_projection(config, years, monthly_contribution, 0.07), 2)),
        },
        "contributed_total": float(round(
            total_contributed(config, years, monthly_contribution), 2)),
        "fan": fan,
    }

    # Секция Box 3: параметры налога, стоимость и «теневая» медиана без налога.
    if box3 is not None and capital_pretax is not None:
        col = years * 12
        med_tax_nom = float(np.percentile(capital[:, col], 50))
        med_pre_nom = float(np.percentile(capital_pretax[:, col], 50))
        med_tax_real = float(np.percentile(capital[:, col] * deflate[col], 50))
        med_pre_real = float(np.percentile(capital_pretax[:, col] * deflate[col], 50))

        # Теневой веер БЕЗ налога: только медиана реального капитала (0..N).
        fan_pretax_p50 = []
        for y in fan_years:
            c = y * 12
            fan_pretax_p50.append(float(round(
                float(np.percentile(capital_pretax[:, c] * deflate[c], 50)), 2)))

        result["box3"] = {
            "enabled": True,
            "first_tax_year": int(box3["first_tax_year"]),
            "tax_free_allowance": float(box3["tax_free_allowance"]),
            "allowance_growth": float(box3["allowance_growth"]),
            "deemed_return": float(box3["deemed_return"]),
            "tax_rate": float(box3["tax_rate"]),
            "tegenbewijs": bool(box3["tegenbewijs"]),
            "reform_start_year": int(box3["reform_start_year"]),
            "reform_tax_rate": float(box3["reform_tax_rate"]),
            "reform_tax_free_result": float(box3["reform_tax_free_result"]),
            "model_reform": bool(box3["model_reform"]),
            "median_no_tax_nominal": float(round(med_pre_nom, 2)),
            "median_no_tax_real": float(round(med_pre_real, 2)),
            "median_tax_cost_nominal_20y": float(round(med_pre_nom - med_tax_nom, 2)),
            "median_tax_cost_real_20y": float(round(med_pre_real - med_tax_real, 2)),
        }
        result["fan_pretax"] = {"real_p50": fan_pretax_p50}
    else:
        result["box3"] = {"enabled": False}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(f"\nJSON-результаты сохранены: {out_path}")


# ---------------------------------------------------------------------------
# Оркестрация
# ---------------------------------------------------------------------------
def resolve_return_source(config: dict) -> tuple:
    """Определить источник доходностей: (месячные | None, годовые | None, флаг)."""
    monthly = build_portfolio_monthly_returns(config)
    if monthly is not None and len(monthly) >= 24:
        return monthly.to_numpy(), None, True
    annual = build_portfolio_annual_returns(config)
    return None, annual, False


def run_single_scenario(
    config: dict, years: int, n_paths: int, seed: int | None,
    monthly_contribution: float, goal: float | None,
    monthly_src: np.ndarray | None, annual_src: np.ndarray | None,
    used_monthly: bool, box3: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Прогнать один сценарий и напечатать полный отчёт.

    Возвращает (capital_после_налога, capital_без_налога) на одних путях.
    """
    currency = config.get("base_currency", "EUR")
    rng = np.random.default_rng(seed)
    capital, capital_pretax = simulate_with_shadow(
        config, years, n_paths, rng, monthly_contribution,
        monthly_src, annual_src, box3,
    )
    deflate = deflator(config, years)

    contributed = total_contributed(config, years, monthly_contribution)
    print(f"\nВзнос: {_fmt(monthly_contribution)} {currency}/мес "
          f"(индексация {config.get('annual_contribution_growth', 0)*100:.1f}%/год)")
    print(f"Всего внесёте за {years} лет: {_fmt(contributed)} {currency} "
          f"(+ стартовые {_fmt(config['current_value'])} {currency})")
    if box3 is not None:
        print("Перцентили ниже показаны ПОСЛЕ вычета налога Box 3.")

    print_percentile_table(capital, deflate, currency, years)
    if goal is not None:
        print_goal_probability(capital, deflate, goal, years, currency)
    print_naive_comparison(config, capital, years, monthly_contribution, currency)
    if box3 is not None:
        print_box3_comparison(
            capital, capital_pretax, deflate, years, currency, box3)
    return capital, capital_pretax


def run_compare(
    config: dict, years: int, n_paths: int, seed: int | None,
    contributions: list[float], goal: float | None,
    monthly_src: np.ndarray | None, annual_src: np.ndarray | None,
    box3: dict | None,
) -> None:
    """Сравнить несколько сценариев ежемесячного взноса по итоговым перцентилям."""
    currency = config.get("base_currency", "EUR")
    deflate = deflator(config, years)
    final_col = years * 12
    print(f"\n{'='*70}\nСРАВНЕНИЕ СЦЕНАРИЕВ ВЗНОСОВ (итог через {years} лет)\n{'='*70}")
    if box3 is not None:
        print("  (значения ПОСЛЕ вычета налога Box 3)")
    print("  {:>10} | {:>12} | {:>12} | {:>12}".format(
        "взнос/мес", "P25 реал", "P50 реал", "P75 реал"))
    print("  " + "-" * 54)
    for c in contributions:
        rng = np.random.default_rng(seed)  # один seed -> сопоставимые сценарии
        capital = simulate(config, years, n_paths, rng, c, monthly_src,
                           annual_src, box3)
        final_real = capital[:, final_col] * deflate[final_col]
        p25, p50, p75 = np.percentile(final_real, [25, 50, 75])
        prob = ""
        if goal is not None:
            prob = f"  P(цель)={float((final_real >= goal).mean())*100:.0f}%"
        print("  {:>10} | {:>12} | {:>12} | {:>12}{}".format(
            _fmt(c), _fmt(p25), _fmt(p50), _fmt(p75), prob))
    print("  (значения — в реальных деньгах, сегодняшняя покупательная способность)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Долгосрочный Монте-Карло прогноз капитала пассивного инвестора.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--years", type=int, default=None,
                        help="Горизонт в годах (по умолчанию из конфига).")
    parser.add_argument("--paths", type=int, default=N_PATHS_DEFAULT,
                        help="Число симуляционных путей Монте-Карло.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed ГСЧ для воспроизводимости.")
    parser.add_argument("--contribution", type=float, default=None,
                        help="Переопределить ежемесячный взнос (сценарий «а если»).")
    parser.add_argument("--goal", type=float, default=None,
                        help="Целевая сумма для оценки вероятности достижения.")
    parser.add_argument("--compare", type=str, default=None,
                        help="Сравнить взносы, напр. --compare 500,1000,2000")
    parser.add_argument("--json", type=str, default=None, dest="json_path",
                        help="Сохранить результаты симуляции в JSON по этому пути.")
    parser.add_argument("--no-box3", action="store_true", dest="no_box3",
                        help="Отключить нидерландский налог Box 3 (прогноз до налога).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    args = parse_args(argv)
    config = load_config()
    currency = config.get("base_currency", "EUR")
    years = args.years or int(config.get("horizon_years", 20))
    n_paths = args.paths
    base_contribution = float(
        args.contribution
        if args.contribution is not None
        else config["monthly_contribution"]
    )

    print("=" * 70)
    print("ДОЛГОСРОЧНЫЙ ПРОГНОЗ КАПИТАЛА (Монте-Карло)")
    print("=" * 70)

    monthly_src, annual_src, used_monthly = resolve_return_source(config)
    if used_monthly:
        print(f"Источник доходностей: скачанные МЕСЯЧНЫЕ серии из data/ "
              f"(блочный бутстрэп, блоки {BLOCK_MIN}-{BLOCK_MAX} мес.).")
    else:
        print("ВНИМАНИЕ: скачанных месячных серий не найдено.")
        print("  Использую резерв data/fallback_annual_returns.csv (ГОДОВОЙ бутстрэп).")
        print("  Это приближение. Для точности запустите: python3 tools/fetch_data.py")
    print(f"Путей симуляции: {n_paths:,}".replace(",", " ") +
          f" | горизонт: {years} лет | seed: {args.seed}")

    box3 = resolve_box3(config, args.no_box3)
    if box3 is not None:
        print("Налог Box 3 (Нидерланды): ВКЛЮЧЁН (прогноз показан после налога).")
    else:
        print("Налог Box 3: выключен (прогноз до налога).")

    if args.compare:
        try:
            contributions = [float(x) for x in args.compare.split(",") if x.strip()]
        except ValueError:
            print("Ошибка: --compare ожидает числа через запятую, напр. 500,1000,2000")
            return 1
        run_compare(config, years, n_paths, args.seed, contributions,
                    args.goal, monthly_src, annual_src, box3)
        return 0

    capital, capital_pretax = run_single_scenario(
        config, years, n_paths, args.seed, base_contribution, args.goal,
        monthly_src, annual_src, used_monthly, box3,
    )
    make_fan_chart(
        capital, years, currency, REPORTS_DIR / "forecast_fan.png",
        base_contribution,
    )
    if args.json_path:
        export_json(
            config, capital, deflator(config, years), years, n_paths,
            args.seed, base_contribution, args.goal, used_monthly,
            Path(args.json_path), box3, capital_pretax,
        )
    print("\nГотово. Помните: прошлые доходности не гарантируют будущих.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
