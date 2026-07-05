#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Загрузка месячных исторических цен для прокси-серий портфеля.

Запускается ПОЛЬЗОВАТЕЛЕМ ЛОКАЛЬНО (в изолированной среде агента сеть к
финансовым сайтам заблокирована). Скрипт:

  1. читает config/portfolio.yaml (секцию data_sources и holdings);
  2. для каждой нужной прокси-серии скачивает месячные котировки со Stooq
     (https://stooq.com/q/d/l/?s=<symbol>&i=m), при неудаче пробует yfinance
     (если установлен);
  3. сохраняет результат в data/<series>.csv с колонками date,close.

Примеры запуска:
    python3 tools/fetch_data.py                 # все серии из holdings
    python3 tools/fetch_data.py --series msci_world
    python3 tools/fetch_data.py --force         # перекачать, игнорируя кэш
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import yaml

# Корень репозитория (на уровень выше каталога tools/).
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "portfolio.yaml"
DATA_DIR = ROOT / "data"

# Вежливый User-Agent, чтобы не выглядеть анонимным ботом.
USER_AGENT = (
    "Mozilla/5.0 (compatible; PassiveInvestorForecast/1.0; "
    "+local research tool)"
)
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=m"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Прочитать YAML-конфиг портфеля и вернуть его как словарь."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def series_needed(config: dict) -> list[str]:
    """Вернуть список прокси-серий, реально используемых в holdings."""
    proxies = {h["proxy"] for h in config.get("holdings", [])}
    return sorted(proxies)


def resolve_symbols(config: dict, series: str) -> dict:
    """Найти символы (stooq/yf) для серии в секции data_sources конфига."""
    sources = config.get("data_sources", {})
    if series not in sources:
        raise KeyError(
            f"Для серии '{series}' нет записи в data_sources конфига. "
            f"Добавьте её в config/portfolio.yaml."
        )
    return sources[series]


def _http_get(url: str, retries: int = 3, pause: float = 2.0) -> str:
    """Выполнить GET с повторными попытками. Вернуть тело ответа как текст."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (HTTPError, URLError, TimeoutError) as err:
            last_err = err
            print(f"  попытка {attempt}/{retries} не удалась: {err}")
            if attempt < retries:
                time.sleep(pause * attempt)
    raise RuntimeError(f"Не удалось скачать {url}: {last_err}")


def fetch_from_stooq(symbol: str) -> pd.DataFrame:
    """Скачать месячные котировки со Stooq. Вернуть DataFrame[date, close]."""
    url = STOOQ_URL.format(symbol=symbol)
    print(f"  Stooq: {url}")
    text = _http_get(url)
    # Stooq на ошибку/лимит отдаёт короткий текст вместо CSV.
    if "Date" not in text.splitlines()[0]:
        raise ValueError(f"Stooq вернул не-CSV ответ: {text[:80]!r}")
    raw = pd.read_csv(io.StringIO(text))
    if "Close" not in raw.columns or "Date" not in raw.columns:
        raise ValueError(f"Неожиданные колонки Stooq: {list(raw.columns)}")
    out = raw[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    out = out.dropna(subset=["close"])
    return out


def fetch_from_yfinance(ticker: str) -> pd.DataFrame:
    """Резервная загрузка через yfinance (если пакет установлен)."""
    try:
        import yfinance as yf  # noqa: WPS433 (ленивый импорт, пакет опционален)
    except ImportError as err:
        raise RuntimeError("yfinance не установлен (pip install yfinance)") from err

    print(f"  yfinance: тикер {ticker}, интервал 1mo")
    df = yf.download(
        ticker, interval="1mo", auto_adjust=True, progress=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance не вернул данные по {ticker}")
    # У новых версий yfinance колонки могут быть MultiIndex.
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = close.reset_index()
    out.columns = ["date", "close"]
    out = out.dropna(subset=["close"])
    return out


def fetch_series(series: str, symbols: dict) -> pd.DataFrame:
    """Скачать одну серию: сначала Stooq, при неудаче — yfinance."""
    errors = []
    stooq_symbol = symbols.get("stooq")
    if stooq_symbol:
        try:
            return fetch_from_stooq(stooq_symbol)
        except Exception as err:  # noqa: BLE001 — логируем и идём к фолбэку
            errors.append(f"Stooq: {err}")
            print(f"  Stooq не сработал: {err}")

    yf_ticker = symbols.get("yf")
    if yf_ticker:
        try:
            return fetch_from_yfinance(yf_ticker)
        except Exception as err:  # noqa: BLE001
            errors.append(f"yfinance: {err}")
            print(f"  yfinance не сработал: {err}")

    raise RuntimeError(
        f"Не удалось получить серию '{series}'. Причины: " + "; ".join(errors)
    )


def save_series(series: str, df: pd.DataFrame) -> Path:
    """Нормализовать и сохранить серию в data/<series>.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").drop_duplicates(subset="date")
    out_path = DATA_DIR / f"{series}.csv"
    df[["date", "close"]].to_csv(out_path, index=False)
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Загрузка месячных котировок прокси-серий портфеля."
    )
    parser.add_argument(
        "--series",
        nargs="*",
        default=None,
        help="Какие серии качать (по умолчанию — все из holdings конфига).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перекачать даже если файл data/<series>.csv уже существует.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    args = parse_args(argv)
    config = load_config()
    targets = args.series if args.series else series_needed(config)

    if not targets:
        print("В конфиге нет прокси-серий для загрузки.")
        return 1

    print(f"К загрузке серии: {', '.join(targets)}")
    failures = []
    for series in targets:
        out_path = DATA_DIR / f"{series}.csv"
        if out_path.exists() and not args.force:
            print(f"[{series}] уже есть ({out_path}); пропуск (--force для перекачки).")
            continue
        print(f"[{series}] загрузка...")
        try:
            symbols = resolve_symbols(config, series)
            df = fetch_series(series, symbols)
            saved = save_series(series, df)
            print(f"[{series}] сохранено: {saved} ({len(df)} строк)")
        except Exception as err:  # noqa: BLE001
            failures.append(series)
            print(f"[{series}] ОШИБКА: {err}")

    if failures:
        print(
            f"\nНе удалось скачать: {', '.join(failures)}. "
            f"forecast.py при отсутствии данных использует "
            f"data/fallback_annual_returns.csv."
        )
        return 2
    print("\nГотово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
