#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Одна кнопка: обновить данные, построить отчёт и открыть его в браузере.

Для нетехнического пользователя. Запуск:

    python3 start.py

Скрипт по шагам:
  1. пытается обновить исторические котировки (tools/fetch_data.py).
     Если интернета к финансовым сайтам нет — мягко предупреждает и
     продолжает на встроенных данных;
  2. строит наглядный отчёт (tools/report.py);
  3. открывает reports/report.html в браузере.

Никакие технические ошибки наружу не выбрасываются — только понятные
объяснения на русском.
"""

from __future__ import annotations

import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FETCH = ROOT / "tools" / "fetch_data.py"
REPORT = ROOT / "tools" / "report.py"
OUT_HTML = ROOT / "reports" / "report.html"

LINE = "-" * 60


def _run(script: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=timeout,
    )


def step_fetch() -> None:
    """Шаг 1: обновить данные. Ошибки сети — не критичны."""
    print("Шаг 1 из 3. Обновляю исторические данные...")
    try:
        proc = _run(FETCH, timeout=120)
        if proc.returncode == 0:
            print("  Данные обновлены.")
        else:
            print("  Не удалось обновить данные (вероятно, нет доступа в интернет).")
            print("  Ничего страшного: отчёт будет построен на встроенных данных.")
    except subprocess.TimeoutExpired:
        print("  Обновление данных заняло слишком долго — пропускаю этот шаг.")
        print("  Отчёт будет построен на встроенных данных.")
    except Exception:
        print("  Не удалось обновить данные — продолжаю на встроенных данных.")


def step_report() -> bool:
    """Шаг 2: построить отчёт. Возвращает True при успехе."""
    print("\nШаг 2 из 3. Строю отчёт...")
    try:
        proc = _run(REPORT, timeout=300)
        if proc.returncode == 0 and OUT_HTML.exists():
            print("  Отчёт готов.")
            return True
        print("  Не получилось построить отчёт.")
        detail = (proc.stderr or proc.stdout or "").strip()
        if detail:
            print("  Подробности (последние строки):")
            for line in detail.splitlines()[-5:]:
                print("    " + line)
        return False
    except subprocess.TimeoutExpired:
        print("  Построение отчёта заняло слишком долго. Попробуйте ещё раз.")
        return False
    except Exception as exc:
        print(f"  Не получилось построить отчёт: {exc}")
        return False


def step_open() -> None:
    """Шаг 3: открыть отчёт в браузере."""
    print("\nШаг 3 из 3. Открываю отчёт в браузере...")
    url = OUT_HTML.resolve().as_uri()
    try:
        opened = webbrowser.open(url)
    except Exception:
        opened = False
    if opened:
        print("  Готово. Отчёт должен открыться в вашем браузере.")
    else:
        print("  Не удалось открыть браузер автоматически.")
    print(f"\n  Файл отчёта: {OUT_HTML}")
    print("  Откройте его двойным щелчком, если он не открылся сам.")


def main() -> int:
    print(LINE)
    print("ИНВЕСТИЦИОННЫЙ ОТЧЁТ — сборка в один клик")
    print(LINE)
    step_fetch()
    ok = step_report()
    if not ok:
        print("\n" + LINE)
        print("Отчёт построить не удалось. Проверьте, что установлены")
        print("библиотеки из requirements.txt (pip install -r requirements.txt).")
        print(LINE)
        return 1
    step_open()
    print("\n" + LINE)
    print("Всё готово.")
    print(LINE)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(130)
