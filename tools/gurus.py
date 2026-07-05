#!/usr/bin/env python3
"""Трекер 13F-отчётов известных инвесторов через SEC EDGAR.

Форма 13F-HR — обязательное квартальное раскрытие долгосрочных позиций
институциональных инвесторов в SEC. Инструмент скачивает список последних
13F-HR фильингов инвестора из data.sec.gov, разбирает XML-таблицу позиций
(information table) и сохраняет снапшот в data/13f/<cik>_<период>.csv.
Команда diff сравнивает два последних снапшота и показывает, что изменилось
в портфеле.

Использование:
    python tools/gurus.py list
    python tools/gurus.py fetch --investor "Berkshire Hathaway (Warren Buffett)"
    python tools/gurus.py fetch --all
    python tools/gurus.py diff --investor "Berkshire Hathaway (Warren Buffett)"

Ограничения:
  - Используется только последняя "страница" фильингов из submissions JSON
    (поле filings.recent) — для активных фондов с многолетней историей
    более старые 13F могут не попасть в неё без отдельной пагинации.
  - С 2023 года SEC изменила единицы измерения value в information table
    с тысяч долларов на целые доллары для новых фильингов — при сравнении
    очень старых и новых отчётов это может исказить видимые изменения сумм
    (веса позиций в процентах от этого не страдают).
  - Для инвесторов без проставленного CIK в config/gurus.yaml (cik: null)
    команды fetch/diff пропускают запись с предупреждением.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "gurus.yaml"
)

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVE_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/index.json"
SEC_ARCHIVE_FILE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{filename}"


class GuruDataError(Exception):
    """Ошибка получения/разбора данных 13F (сеть, формат, отсутствие данных)."""


# --------------------------------------------------------------------------
# Конфигурация
# --------------------------------------------------------------------------

@dataclass
class Investor:
    name: str
    cik: Optional[str]  # 10-значная строка с ведущими нулями, либо None


@dataclass
class GurusConfig:
    investors: list  # list[Investor]
    user_agent: str
    request_delay_seconds: float
    output_dir: str
    significant_change_pct: float


def _normalize_cik(raw) -> Optional[str]:
    if raw is None:
        return None
    digits = str(raw).strip()
    if not digits:
        return None
    return digits.zfill(10)


def load_config(path: str = DEFAULT_CONFIG_PATH) -> GurusConfig:
    """Загружает config/gurus.yaml."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    investors = []
    for item in raw.get("investors", []):
        investors.append(Investor(name=item["name"], cik=_normalize_cik(item.get("cik"))))

    return GurusConfig(
        investors=investors,
        user_agent=raw.get("user_agent", "private investor katja.bunich@gmail.com"),
        request_delay_seconds=float(raw.get("request_delay_seconds", 0.5)),
        output_dir=raw.get("output_dir", "data/13f"),
        significant_change_pct=float(raw.get("significant_change_pct", 20)),
    )


# --------------------------------------------------------------------------
# Сетевой слой (SEC EDGAR), с throttling и обязательным User-Agent
# --------------------------------------------------------------------------

class SecClient:
    """Тонкий клиент к SEC EDGAR с соблюдением требований по User-Agent
    и минимальной паузе между запросами."""

    def __init__(self, user_agent: str, delay_seconds: float = 0.5):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "user_agent должен содержать осмысленный контакт (например, email) — "
                "так требует SEC EDGAR fair access policy."
            )
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self.delay_seconds = delay_seconds
        self._last_request_ts = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        remaining = self.delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def get(self, url: str) -> requests.Response:
        self._throttle()
        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
        except requests.exceptions.RequestException as exc:
            raise GuruDataError(f"Сетевая ошибка при запросе {url}: {exc}") from exc
        finally:
            self._last_request_ts = time.monotonic()

        if resp.status_code == 404:
            raise GuruDataError(f"Не найдено (404): {url}")
        if resp.status_code == 403:
            raise GuruDataError(
                f"Доступ запрещён (403) для {url}. Проверьте User-Agent в config/gurus.yaml "
                f"(SEC требует реальный контакт) и не превышайте лимит запросов."
            )
        if not resp.ok:
            raise GuruDataError(f"Ошибка HTTP {resp.status_code} при запросе {url}")
        return resp

    def get_json(self, url: str) -> dict:
        return self.get(url).json()

    def get_text(self, url: str) -> str:
        return self.get(url).text


# --------------------------------------------------------------------------
# Поиск и скачивание 13F-HR фильингов
# --------------------------------------------------------------------------

def fetch_submissions(client: SecClient, cik10: str) -> dict:
    """Скачивает submissions JSON для компании с данным CIK (10 цифр)."""

    url = SEC_SUBMISSIONS_URL.format(cik10=cik10)
    return client.get_json(url)


def list_13f_filings(submissions: dict) -> list:
    """Извлекает список фильингов формы 13F-HR (без поправок /A) из
    submissions JSON, отсортированный по отчётному периоду (reportDate)
    от новых к старым, с дедупликацией по периоду."""

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession_numbers = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    primary_docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form != "13F-HR":
            continue
        filings.append(
            {
                "form": form,
                "accessionNumber": accession_numbers[i],
                "filingDate": filing_dates[i],
                "reportDate": report_dates[i],
                "primaryDocument": primary_docs[i] if i < len(primary_docs) else None,
            }
        )

    # Сортировка по отчётному периоду, новые сначала; при дублях периода
    # (например, поправка была подана как отдельный 13F-HR) оставляем
    # запись с более поздней датой подачи.
    filings.sort(key=lambda f: (f["reportDate"], f["filingDate"]), reverse=True)
    seen_periods = set()
    deduped = []
    for f in filings:
        if f["reportDate"] in seen_periods:
            continue
        seen_periods.add(f["reportDate"])
        deduped.append(f)
    return deduped


def _accession_nodash(accession_number: str) -> str:
    return accession_number.replace("-", "")


def find_information_table_filename(client: SecClient, cik_int: int, accession_nodash: str) -> str:
    """Находит имя файла information table (XML со списком позиций) в
    каталоге фильинга, используя машиночитаемый index.json каталога."""

    url = SEC_ARCHIVE_INDEX_URL.format(cik_int=cik_int, accession_nodash=accession_nodash)
    index = client.get_json(url)
    items = index.get("directory", {}).get("item", [])

    xml_items = [it for it in items if it.get("name", "").lower().endswith(".xml")]
    if not xml_items:
        raise GuruDataError(f"В каталоге фильинга {accession_nodash} не найдено XML-файлов.")

    # Основной способ: файл с "infotable" в имени (стандартное соглашение SEC).
    for it in xml_items:
        if "infotable" in it["name"].lower():
            return it["name"]

    # Запасной способ: information table обычно заметно крупнее, чем
    # обложка (primary_doc.xml) — берём самый большой XML, не являющийся
    # титульной страницей.
    candidates = [it for it in xml_items if it["name"].lower() != "primary_doc.xml"]
    if not candidates:
        candidates = xml_items
    candidates.sort(key=lambda it: int(it.get("size", 0) or 0), reverse=True)
    return candidates[0]["name"]


def _local_tag(tag: str) -> str:
    """Убирает XML namespace из имени тега: '{ns}tag' -> 'tag'."""
    return tag.rsplit("}", 1)[-1]


def _find_text(elem: ET.Element, tag_name: str) -> Optional[str]:
    """Ищет первый текст потомка elem (на любой глубине) с локальным именем
    tag_name, игнорируя namespace."""

    for child in elem.iter():
        if _local_tag(child.tag) == tag_name:
            return (child.text or "").strip() or None
    return None


def parse_information_table(xml_text: str) -> pd.DataFrame:
    """Разбирает XML information table формы 13F-HR в DataFrame.

    Работает с любым namespace-префиксом (ns1:informationTable,
    без префикса, и т.п.) — поиск идёт по локальному имени тега.

    Возвращает DataFrame с колонками: nameOfIssuer, cusip, value, shares.
    value — как указано в фильинге (см. предупреждение о единицах измерения
    в докстринге модуля). shares — количество акций/долей (sshPrnamt).
    """

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise GuruDataError(f"Не удалось разобрать XML information table: {exc}") from exc

    rows = []
    for elem in root.iter():
        if _local_tag(elem.tag) != "infoTable":
            continue
        name_of_issuer = _find_text(elem, "nameOfIssuer")
        cusip = _find_text(elem, "cusip")
        value_raw = _find_text(elem, "value")
        shares_raw = _find_text(elem, "sshPrnamt")

        if name_of_issuer is None or cusip is None:
            continue

        try:
            value = float(value_raw) if value_raw is not None else 0.0
        except ValueError:
            value = 0.0
        try:
            shares = float(shares_raw) if shares_raw is not None else 0.0
        except ValueError:
            shares = 0.0

        rows.append({"nameOfIssuer": name_of_issuer, "cusip": cusip, "value": value, "shares": shares})

    if not rows:
        raise GuruDataError("Information table не содержит ни одной позиции (infoTable).")

    return pd.DataFrame(rows)


def aggregate_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Группирует позиции по (cusip, nameOfIssuer) — на случай, если один
    и тот же эмитент разбит на несколько строк (разные классы акций,
    put/call и т.п.) — и добавляет вес позиции в портфеле (weight_pct)."""

    grouped = (
        df.groupby(["cusip", "nameOfIssuer"], as_index=False)[["value", "shares"]]
        .sum()
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    total_value = grouped["value"].sum()
    grouped["weight_pct"] = (grouped["value"] / total_value * 100.0) if total_value > 0 else 0.0
    return grouped


# --------------------------------------------------------------------------
# Снапшоты на диске
# --------------------------------------------------------------------------

def snapshot_path(output_dir: str, cik10: str, report_date: str) -> str:
    return os.path.join(output_dir, f"{cik10}_{report_date}.csv")


def save_snapshot(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)


def list_saved_snapshots(output_dir: str, cik10: str) -> list:
    """Возвращает отсортированный (от старых к новым) список (report_date, path)
    уже сохранённых снапшотов для данного CIK."""

    if not os.path.isdir(output_dir):
        return []
    prefix = f"{cik10}_"
    result = []
    for fname in os.listdir(output_dir):
        if fname.startswith(prefix) and fname.endswith(".csv"):
            report_date = fname[len(prefix):-len(".csv")]
            result.append((report_date, os.path.join(output_dir, fname)))
    result.sort(key=lambda t: t[0])
    return result


# --------------------------------------------------------------------------
# fetch: скачать и сохранить снапшоты
# --------------------------------------------------------------------------

def fetch_investor(client: SecClient, investor: Investor, output_dir: str, periods: int = 2) -> list:
    """Скачивает последние `periods` квартальных 13F-HR инвестора и
    сохраняет снапшоты (пропускает уже скачанные периоды). Возвращает
    список путей к сохранённым/уже существующим файлам снапшотов."""

    if investor.cik is None:
        print(f"[{investor.name}] пропущено: CIK не указан в config/gurus.yaml (см. TODO).")
        return []

    print(f"[{investor.name}] CIK {investor.cik}: запрашиваю список фильингов...")
    try:
        submissions = fetch_submissions(client, investor.cik)
    except GuruDataError as exc:
        print(f"[{investor.name}] ошибка: {exc}")
        return []

    filings = list_13f_filings(submissions)
    if not filings:
        print(f"[{investor.name}] 13F-HR фильингов не найдено.")
        return []

    cik_int = int(investor.cik)
    saved_paths = []
    for filing in filings[:periods]:
        report_date = filing["reportDate"]
        out_path = snapshot_path(output_dir, investor.cik, report_date)
        if os.path.exists(out_path):
            print(f"[{investor.name}] период {report_date}: снапшот уже есть ({out_path}), пропускаю скачивание.")
            saved_paths.append(out_path)
            continue

        accession_nodash = _accession_nodash(filing["accessionNumber"])
        try:
            xml_filename = find_information_table_filename(client, cik_int, accession_nodash)
            xml_url = SEC_ARCHIVE_FILE_URL.format(
                cik_int=cik_int, accession_nodash=accession_nodash, filename=xml_filename
            )
            xml_text = client.get_text(xml_url)
            positions = parse_information_table(xml_text)
            aggregated = aggregate_positions(positions)
        except GuruDataError as exc:
            print(f"[{investor.name}] период {report_date}: ошибка обработки — {exc}")
            continue

        save_snapshot(aggregated, out_path)
        print(f"[{investor.name}] период {report_date}: сохранено {len(aggregated)} позиций -> {out_path}")
        saved_paths.append(out_path)

    return saved_paths


# --------------------------------------------------------------------------
# diff: сравнение двух последних снапшотов
# --------------------------------------------------------------------------

def diff_snapshots(prev_df: pd.DataFrame, curr_df: pd.DataFrame, significant_change_pct: float) -> dict:
    """Сравнивает два снапшота позиций (по cusip).

    Возвращает словарь с ключами:
      new     — позиции, появившиеся в curr, которых не было в prev
      closed  — позиции, которые были в prev, но исчезли в curr
      changed — общие позиции, у которых вес в портфеле изменился более
                чем на significant_change_pct процентов относительно
                предыдущего веса
    """

    prev_idx = prev_df.set_index("cusip")
    curr_idx = curr_df.set_index("cusip")

    new_cusips = curr_idx.index.difference(prev_idx.index)
    closed_cusips = prev_idx.index.difference(curr_idx.index)
    common_cusips = curr_idx.index.intersection(prev_idx.index)

    new_positions = curr_idx.loc[new_cusips].sort_values("value", ascending=False).reset_index()
    closed_positions = prev_idx.loc[closed_cusips].sort_values("value", ascending=False).reset_index()

    changed_rows = []
    for cusip in common_cusips:
        old_weight = prev_idx.loc[cusip, "weight_pct"]
        new_weight = curr_idx.loc[cusip, "weight_pct"]
        # На случай дублей индекса (не должно происходить после агрегации, но на всякий случай).
        if hasattr(old_weight, "__len__"):
            old_weight = float(old_weight.iloc[0])
        if hasattr(new_weight, "__len__"):
            new_weight = float(new_weight.iloc[0])

        if old_weight == 0:
            continue
        pct_change = (new_weight - old_weight) / old_weight * 100.0
        if abs(pct_change) >= significant_change_pct:
            name = curr_idx.loc[cusip, "nameOfIssuer"]
            if hasattr(name, "__len__") and not isinstance(name, str):
                name = name.iloc[0]
            changed_rows.append(
                {
                    "cusip": cusip,
                    "nameOfIssuer": name,
                    "old_weight_pct": old_weight,
                    "new_weight_pct": new_weight,
                    "change_pct": pct_change,
                }
            )

    changed_df = pd.DataFrame(changed_rows)
    if not changed_df.empty:
        changed_df = changed_df.sort_values("change_pct", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    return {"new": new_positions, "closed": closed_positions, "changed": changed_df}


def print_diff_report(investor_name: str, prev_period: str, curr_period: str,
                       curr_df: pd.DataFrame, diff: dict, significant_change_pct: float) -> None:
    print("=" * 70)
    print(f"{investor_name}: сравнение {prev_period} -> {curr_period}")
    print("=" * 70)

    print(f"\nТоп-10 позиций на {curr_period} (по весу в портфеле):")
    top10 = curr_df.sort_values("value", ascending=False).head(10)
    for i, row in enumerate(top10.itertuples(index=False), start=1):
        print(f"  {i:>2}. {row.nameOfIssuer:<40} вес {row.weight_pct:5.1f}%   value={row.value:,.0f}")

    new_positions = diff["new"]
    print(f"\nНовые позиции ({len(new_positions)}):")
    if new_positions.empty:
        print("  нет")
    else:
        for row in new_positions.itertuples(index=False):
            print(f"  + {row.nameOfIssuer:<40} вес {row.weight_pct:5.1f}%")

    closed_positions = diff["closed"]
    print(f"\nЗакрытые позиции ({len(closed_positions)}):")
    if closed_positions.empty:
        print("  нет")
    else:
        for row in closed_positions.itertuples(index=False):
            print(f"  - {row.nameOfIssuer:<40} (был вес {row.weight_pct:5.1f}%)")

    changed = diff["changed"]
    print(f"\nЗначимые изменения долей (>{significant_change_pct:.0f}%, {len(changed)}):")
    if changed.empty:
        print("  нет")
    else:
        for row in changed.itertuples(index=False):
            direction = "+" if row.change_pct > 0 else ""
            print(
                f"  {row.nameOfIssuer:<40} {row.old_weight_pct:5.1f}% -> {row.new_weight_pct:5.1f}% "
                f"({direction}{row.change_pct:.0f}%)"
            )
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_list(cfg: GurusConfig) -> int:
    print("Настроенные инвесторы:")
    for inv in cfg.investors:
        if inv.cik is None:
            print(f"  - {inv.name}: CIK не указан (см. TODO в config/gurus.yaml)")
            continue
        snapshots = list_saved_snapshots(cfg.output_dir, inv.cik)
        periods = ", ".join(p for p, _ in snapshots) if snapshots else "нет сохранённых снапшотов"
        print(f"  - {inv.name} (CIK {inv.cik}): {periods}")
    return 0


def cmd_fetch(cfg: GurusConfig, investor_name: Optional[str], periods: int) -> int:
    client = SecClient(cfg.user_agent, cfg.request_delay_seconds)
    targets = cfg.investors
    if investor_name:
        targets = [inv for inv in cfg.investors if inv.name == investor_name]
        if not targets:
            print(f"Инвестор '{investor_name}' не найден в config/gurus.yaml.", file=sys.stderr)
            return 1

    for inv in targets:
        fetch_investor(client, inv, cfg.output_dir, periods=periods)
    return 0


def cmd_diff(cfg: GurusConfig, investor_name: str) -> int:
    investor = next((inv for inv in cfg.investors if inv.name == investor_name), None)
    if investor is None:
        print(f"Инвестор '{investor_name}' не найден в config/gurus.yaml.", file=sys.stderr)
        return 1
    if investor.cik is None:
        print(f"У инвестора '{investor_name}' не указан CIK (см. TODO в config/gurus.yaml).", file=sys.stderr)
        return 1

    snapshots = list_saved_snapshots(cfg.output_dir, investor.cik)
    if len(snapshots) < 2:
        print(
            f"Недостаточно сохранённых снапшотов для '{investor_name}' "
            f"(найдено {len(snapshots)}, нужно минимум 2). "
            f"Сначала запустите: python tools/gurus.py fetch --investor \"{investor_name}\"",
            file=sys.stderr,
        )
        return 1

    prev_period, prev_path = snapshots[-2]
    curr_period, curr_path = snapshots[-1]
    prev_df = pd.read_csv(prev_path)
    curr_df = pd.read_csv(curr_path)

    diff = diff_snapshots(prev_df, curr_df, cfg.significant_change_pct)
    print_diff_report(investor.name, prev_period, curr_period, curr_df, diff, cfg.significant_change_pct)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Трекер 13F-отчётов известных инвесторов (SEC EDGAR).")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="путь к config/gurus.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="показать настроенных инвесторов и сохранённые снапшоты")

    p_fetch = sub.add_parser("fetch", help="скачать последние 13F-HR с SEC EDGAR")
    p_fetch.add_argument("--investor", help="имя инвестора из конфига (по умолчанию — все)")
    p_fetch.add_argument("--periods", type=int, default=2, help="сколько последних кварталов скачать (по умолчанию 2)")

    p_diff = sub.add_parser("diff", help="сравнить два последних сохранённых снапшота инвестора")
    p_diff.add_argument("--investor", required=True, help="имя инвестора из конфига")

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, yaml.YAMLError, KeyError) as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1

    if args.command == "list":
        return cmd_list(cfg)
    if args.command == "fetch":
        return cmd_fetch(cfg, args.investor, args.periods)
    if args.command == "diff":
        return cmd_diff(cfg, args.investor)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
