#!/usr/bin/env python3
"""Советник по просадкам для пассивного инвестора.

Инструмент читает историю котировок отслеживаемых серий (файлы вида
data/<series>.csv с колонками date,close — тот же формат, что создаёт
tools/fetch_data.py), считает текущую просадку от исторического максимума
и подсказывает, нужно ли докупить актив сверх обычного плана.

Логика траншей: чем глубже просадка, тем больше рекомендуемая докупка
(настраивается в config/monitor.yaml). Чтобы не советовать одну и ту же
докупку повторно, инструмент ведёт журнал уже сделанных докупок
(data/dip_purchases.csv) и сверяет его с текущим «эпизодом просадки»
(периодом от последнего исторического максимума до восстановления).

Примеры использования:
    python tools/monitor.py
    python tools/monitor.py --series msci_world
    python tools/monitor.py --series msci_world --log-purchase 600 --level -20

Если файлов с данными нет, сначала запустите tools/fetch_data.py локально.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "monitor.yaml"
)


# --------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------

@dataclass
class Tranche:
    """Один порог докупки: просадка (%) -> количество взносов."""

    drawdown: float  # отрицательное число, например -10
    installments: float


@dataclass
class SeriesConfig:
    """Настройки одной отслеживаемой серии котировок."""

    name: str
    file: str
    monthly_contribution: float
    cash_reserve: float = 0.0


@dataclass
class MonitorConfig:
    """Полная конфигурация советника по просадкам."""

    series: list = field(default_factory=list)  # list[SeriesConfig]
    currency: str = "EUR"
    tranches: list = field(default_factory=list)  # list[Tranche]
    purchase_log: str = "data/dip_purchases.csv"
    episode_min_depth: float = -10.0


def load_config(path: str = DEFAULT_CONFIG_PATH) -> MonitorConfig:
    """Загружает и валидирует config/monitor.yaml."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {path}\n"
            "Создайте config/monitor.yaml (см. пример в репозитории)."
        )

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    series_list = []
    for item in raw.get("series", []):
        series_list.append(
            SeriesConfig(
                name=item["name"],
                file=item["file"],
                monthly_contribution=float(item.get("monthly_contribution", 0)),
                cash_reserve=float(item.get("cash_reserve", 0)),
            )
        )

    tranches = [
        Tranche(drawdown=float(t["drawdown"]), installments=float(t["installments"]))
        for t in raw.get("tranches", [])
    ]
    # На всякий случай сортируем от менее глубокой просадки к более глубокой
    # (drawdown ближе к 0 -> более отрицательный).
    tranches.sort(key=lambda t: t.drawdown, reverse=True)

    return MonitorConfig(
        series=series_list,
        currency=raw.get("currency", "EUR"),
        tranches=tranches,
        purchase_log=raw.get("purchase_log", "data/dip_purchases.csv"),
        episode_min_depth=float(raw.get("episode_min_depth", -10.0)),
    )


# --------------------------------------------------------------------------
# Данные и расчёт просадки
# --------------------------------------------------------------------------

def read_series(path: str) -> pd.DataFrame:
    """Читает CSV с колонками date,close и возвращает DataFrame с просадкой.

    Возвращаемый DataFrame отсортирован по дате и содержит колонки:
      close       — цена закрытия
      peak        — исторический максимум на данный момент (running max)
      drawdown_pct — просадка от исторического максимума, в процентах (<=0)
    """

    df = pd.read_csv(path)
    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(
            f"Файл {path} должен содержать колонки 'date' и 'close', "
            f"а содержит: {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    df["close"] = df["close"].astype(float)

    df["peak"] = df["close"].cummax()
    df["drawdown_pct"] = (df["close"] / df["peak"] - 1.0) * 100.0

    return df


def find_current_episode_start(df: pd.DataFrame) -> Optional[pd.Timestamp]:
    """Определяет дату начала текущего эпизода просадки.

    Эпизод — это период после последнего исторического максимума (drawdown
    вернулась к 0). Если сейчас на историческом максимуме (просадки нет),
    возвращает None.
    """

    last_drawdown = df["drawdown_pct"].iloc[-1]
    if last_drawdown >= -1e-9:
        return None

    # Точки, где серия была на своём историческом максимуме (drawdown_pct ~ 0).
    at_peak_mask = df["drawdown_pct"] >= -1e-9
    peak_dates = df.loc[at_peak_mask, "date"]
    if peak_dates.empty:
        # Просадка длится с самого начала доступных данных.
        return df["date"].iloc[0]
    return peak_dates.iloc[-1]


def active_tranche(current_drawdown: float, tranches: list) -> Optional[Tranche]:
    """Возвращает самый глубокий транш, порог которого пробит просадкой."""

    crossed = [t for t in tranches if current_drawdown <= t.drawdown]
    if not crossed:
        return None
    # tranches уже отсортированы от менее глубокого к более глубокому
    # (drawdown ближе к 0 -> самое отрицательное в конце), берём самый глубокий.
    return min(crossed, key=lambda t: t.drawdown)


def crossed_tranches(current_drawdown: float, tranches: list) -> list:
    """Возвращает все транши, пороги которых пробиты текущей просадкой,
    отсортированные от менее глубокого к более глубокому."""

    crossed = [t for t in tranches if current_drawdown <= t.drawdown]
    crossed.sort(key=lambda t: t.drawdown, reverse=True)
    return crossed


# --------------------------------------------------------------------------
# Журнал докупок
# --------------------------------------------------------------------------

JOURNAL_FIELDS = ["date", "series", "drawdown_level", "amount"]


def read_journal(path: str) -> pd.DataFrame:
    """Читает журнал докупок. Если файла нет — возвращает пустой DataFrame."""

    if not os.path.exists(path):
        return pd.DataFrame(columns=JOURNAL_FIELDS)

    df = pd.read_csv(path)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def log_purchase(path: str, series_name: str, drawdown_level: float, amount: float,
                  when: Optional[datetime] = None) -> None:
    """Дописывает запись о сделанной докупке в журнал (создаёт файл при необходимости)."""

    when = when or datetime.now()
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(JOURNAL_FIELDS)
        writer.writerow([when.strftime("%Y-%m-%d"), series_name, drawdown_level, amount])


def levels_already_purchased(journal: pd.DataFrame, series_name: str,
                              episode_start: Optional[pd.Timestamp]) -> set:
    """Множество уровней траншей, уже докупленных в текущем эпизоде просадки."""

    if journal.empty or episode_start is None:
        return set()

    mask = journal["series"] == series_name
    mask &= journal["date"] >= episode_start
    return set(journal.loc[mask, "drawdown_level"].astype(float).tolist())


# --------------------------------------------------------------------------
# Историческая статистика по просадкам
# --------------------------------------------------------------------------

@dataclass
class DrawdownEpisode:
    peak_date: pd.Timestamp
    trough_date: pd.Timestamp
    trough_depth: float  # отрицательный процент
    recovery_date: Optional[pd.Timestamp]  # None, если ещё не восстановилась
    duration_days: int  # peak -> recovery (или peak -> последняя дата, если не восстановилась)
    recovery_days: Optional[int]  # trough -> recovery


def find_historical_episodes(df: pd.DataFrame, min_depth: float = -10.0) -> list:
    """Находит все исторические эпизоды просадки глубже min_depth (например, -10%).

    Эпизод начинается на историческом максимуме (drawdown_pct == 0) и
    заканчивается, когда серия снова обновляет максимум. Возвращает список
    DrawdownEpisode, отсортированный от самой глубокой просадки к менее глубокой.
    """

    episodes = []
    n = len(df)
    i = 0
    dates = df["date"].tolist()
    drawdowns = df["drawdown_pct"].tolist()

    while i < n:
        # Ищем начало нового эпизода: точку на пике (drawdown == 0).
        if drawdowns[i] > -1e-9:
            peak_idx = i
            j = i + 1
            trough_idx = i
            # Идём вперёд, пока не найдём новое восстановление до пика (drawdown == 0)
            while j < n and drawdowns[j] <= -1e-9:
                if drawdowns[j] < drawdowns[trough_idx]:
                    trough_idx = j
                j += 1

            recovered = j < n  # j указывает на точку восстановления (drawdown == 0) либо конец данных
            if trough_idx != peak_idx and drawdowns[trough_idx] <= min_depth + 1e-9:
                recovery_date = dates[j] if recovered else None
                duration_end_idx = j if recovered else (n - 1)
                episodes.append(
                    DrawdownEpisode(
                        peak_date=dates[peak_idx],
                        trough_date=dates[trough_idx],
                        trough_depth=drawdowns[trough_idx],
                        recovery_date=recovery_date,
                        duration_days=(dates[duration_end_idx] - dates[peak_idx]).days,
                        recovery_days=(recovery_date - dates[trough_idx]).days if recovery_date is not None else None,
                    )
                )
            i = j
        else:
            i += 1

    episodes.sort(key=lambda e: e.trough_depth)
    return episodes


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------

def format_signal(series_cfg: SeriesConfig, current_drawdown: float,
                   tranche: Optional[Tranche], already_purchased: bool,
                   currency: str) -> str:
    lines = []
    lines.append(f"Текущая просадка: {current_drawdown:.1f}%")

    if tranche is None:
        lines.append("Статус: нет сигнала (просадка меньше порога первого транша).")
        return "\n".join(lines)

    amount = tranche.installments * series_cfg.monthly_contribution
    if already_purchased:
        lines.append(
            f"Статус: сигнал уже отработан — на уровне {tranche.drawdown:.0f}% "
            f"докупка уже зафиксирована в журнале в этом эпизоде просадки."
        )
    else:
        lines.append(
            f"Статус: СИГНАЛ — докупить {amount:.2f} {currency} "
            f"(транш {tranche.installments:g} взнос(а/ов)) на уровне {tranche.drawdown:.0f}%."
        )
    return "\n".join(lines)


def format_episode(idx: int, ep: DrawdownEpisode) -> str:
    recovery = (
        ep.recovery_date.strftime("%Y-%m-%d") if ep.recovery_date is not None else "ещё не восстановилась"
    )
    recovery_days = f"{ep.recovery_days} дн." if ep.recovery_days is not None else "—"
    return (
        f"  {idx}. {ep.peak_date.strftime('%Y-%m-%d')} -> дно {ep.trough_date.strftime('%Y-%m-%d')} "
        f"({ep.trough_depth:.1f}%) -> восстановление: {recovery}\n"
        f"     длительность просадки: {ep.duration_days} дн., "
        f"время восстановления от дна: {recovery_days}"
    )


def report_for_series(series_cfg: SeriesConfig, cfg: MonitorConfig) -> None:
    print("=" * 70)
    print(f"Серия: {series_cfg.name}  (файл: {series_cfg.file})")
    print("=" * 70)

    if not os.path.exists(series_cfg.file):
        print(
            f"Нет данных: файл '{series_cfg.file}' не найден.\n"
            f"Сначала запустите локально: python tools/fetch_data.py "
            f"(он должен создать этот файл с колонками date,close)."
        )
        print()
        return

    df = read_series(series_cfg.file)
    if df.empty:
        print(f"Файл '{series_cfg.file}' пуст — нечего анализировать.")
        print()
        return

    current_drawdown = df["drawdown_pct"].iloc[-1]
    last_date = df["date"].iloc[-1]
    episode_start = find_current_episode_start(df)

    journal = read_journal(cfg.purchase_log)
    purchased_levels = levels_already_purchased(journal, series_cfg.name, episode_start)

    tranche = active_tranche(current_drawdown, cfg.tranches)
    already = tranche is not None and tranche.drawdown in purchased_levels

    print(f"Дата последних данных: {last_date.strftime('%Y-%m-%d')}")
    print(f"Резерв кэша под докупки: {series_cfg.cash_reserve:.2f} {cfg.currency}")
    print()
    print(format_signal(series_cfg, current_drawdown, tranche, already, cfg.currency))

    # Если пройдено несколько траншей, но не все уровни зафиксированы в журнале —
    # покажем это отдельно, чтобы не потерять из виду более ранние (менее глубокие) уровни.
    crossed = crossed_tranches(current_drawdown, cfg.tranches)
    missed = [t for t in crossed if t.drawdown not in purchased_levels and t is not tranche]
    if missed:
        print()
        print("Также ещё не зафиксированы докупки на уровнях (в рамках этого эпизода):")
        for t in missed:
            amount = t.installments * series_cfg.monthly_contribution
            print(f"  - {t.drawdown:.0f}%: {amount:.2f} {cfg.currency}")

    # Историческая статистика по просадкам.
    episodes = find_historical_episodes(df, cfg.episode_min_depth)
    print()
    print(f"История просадок глубже {cfg.episode_min_depth:.0f}% (топ-5 худших по данным):")
    if not episodes:
        print("  Просадок глубже порога в истории не найдено.")
    else:
        for i, ep in enumerate(episodes[:5], start=1):
            print(format_episode(i, ep))
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Советник по просадкам для пассивного инвестора."
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH, help="путь к config/monitor.yaml"
    )
    parser.add_argument(
        "--series", help="анализировать только эту серию (по умолчанию — все из конфига)"
    )
    parser.add_argument(
        "--log-purchase", type=float, metavar="AMOUNT",
        help="записать в журнал сделанную докупку на сумму AMOUNT (требует --series и --level)"
    )
    parser.add_argument(
        "--level", type=float, metavar="LEVEL",
        help="уровень просадки (в %%), на котором сделана докупка, например -20"
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, yaml.YAMLError, KeyError) as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1

    if not cfg.series:
        print("В конфигурации не указано ни одной серии для отслеживания.", file=sys.stderr)
        return 1

    series_by_name = {s.name: s for s in cfg.series}

    if args.log_purchase is not None:
        if not args.series or args.level is None:
            print(
                "Для --log-purchase нужно также указать --series и --level.",
                file=sys.stderr,
            )
            return 1
        if args.series not in series_by_name:
            print(f"Неизвестная серия '{args.series}'. Доступные: {list(series_by_name)}", file=sys.stderr)
            return 1
        log_purchase(cfg.purchase_log, args.series, args.level, args.log_purchase)
        print(
            f"Записано в журнал: {args.series}, уровень {args.level:.0f}%, "
            f"сумма {args.log_purchase:.2f} {cfg.currency} ({cfg.purchase_log})"
        )
        return 0

    targets = [series_by_name[args.series]] if args.series else cfg.series
    if args.series and args.series not in series_by_name:
        print(f"Неизвестная серия '{args.series}'. Доступные: {list(series_by_name)}", file=sys.stderr)
        return 1

    for series_cfg in targets:
        report_for_series(series_cfg, cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
