#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прогон 2025 года через ту же модель, что считает 2026 — и сверка
с реальной декларацией.

Смысл: декларация 2025 известна точно, значит она даёт независимую
точку опоры. Если модель воспроизводит её, расчёту на 2026 можно верить.

Box 3 не считается — владелица платит его сама.

    python3 tools/check2025.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tax_zzp import YEARS  # noqa: E402

P = YEARS[2025]

# --- ИСХОДНЫЕ ДАННЫЕ 2025 ---------------------------------------------------
# Из отчётности (jaarrekening):
OPBRENGSTEN = 91_156
ACTIVA = 2_978          # overige materiële vaste activa, списаны сразу
KOSTEN = 7_068          # andere kosten
WINST = 81_110          # subtotaal, сходится: 91 156 − 2 978 − 7 068

KIA_RATE = 0.28         # 2025: 28 % при инвестициях 2 901 — 70 602
KIA = round(ACTIVA * KIA_RATE)

# Из декларации — то, с чем сверяемся:
DECL_BASE = 66_068      # belastbare winst
DECL_BOX1 = 19_800      # 20 677 минус Box 3 877
DECL_ZVW = 3_475


def scale(base: float) -> float:
    tax, low = 0.0, 0.0
    for high, rate in P["BRACKETS"]:
        if base > low:
            tax += (min(base, high) - low) * rate
        low = high
    return tax


def ahk(base: float) -> float:
    if base <= P["AHK_START"]:
        return P["AHK_MAX"]
    if base >= P["AHK_ZERO"]:
        return 0.0
    return max(0.0, P["AHK_MAX"] - (base - P["AHK_START"]) * P["AHK_RATE"])


def ak(inkomen: float) -> float:
    if inkomen <= P["AK_TOP"]:
        return P["AK_MAX"]
    if inkomen >= P["AK_ZERO"]:
        return 0.0
    return max(0.0, P["AK_MAX"] - (inkomen - P["AK_TOP"]) * P["AK_RATE"])


def f(x: float) -> str:
    return f"{x:>11,.0f}".replace(",", " ")


def main() -> None:
    print("=" * 70)
    print("ПРОГОН 2025 ЧЕРЕЗ МОДЕЛЬ 2026 — СВЕРКА С ДЕКЛАРАЦИЕЙ")
    print("=" * 70)
    print(f"  Totaal opbrengsten                  {f(OPBRENGSTEN)}")
    print(f"  − overige materiële vaste activa    {f(-ACTIVA)}")
    print(f"  − andere kosten                     {f(-KOSTEN)}")
    print(f"  = WINST                             {f(WINST)}")
    print()

    zelf = P["ZELFSTANDIGENAFTREK"]
    start = P["STARTERSAFTREK"]
    print(f"  − zelfstandigenaftrek               {f(-zelf)}")
    print(f"  − startersaftrek                    {f(-start)}")
    print(f"  − KIA {KIA_RATE:.0%} от {ACTIVA:,}".replace(",", " ")
          + " " * 17 + f"{f(-KIA)}")
    after_aftrek = WINST - zelf - start - KIA
    print(f"  =                                   {f(after_aftrek)}")

    mkb = after_aftrek * P["MKB_VRIJSTELLING"]
    base = after_aftrek - mkb
    print(f"  − MKB-winstvrijstelling {P['MKB_VRIJSTELLING']:.1%}       {f(-mkb)}")
    print(f"  = НАЛОГООБЛАГАЕМАЯ БАЗА             {f(base)}")
    print(f"    в декларации                      {f(DECL_BASE)}"
          f"   расхождение {base - DECL_BASE:+.0f}")
    print()

    # tariefsaanpassing: под ограничение попадают ondernemersaftrek и MKB.
    # KIA — investeringsaftrek, в список не входит.
    posten = zelf + start + mkb
    correctie = P["TARIEFSAANPASSING"] * min(
        posten, max(0.0, base + posten - P["TARIEF_THRESHOLD"]))

    it = scale(base)
    k_ahk, k_ak = ahk(base), ak(base)
    box1 = max(0.0, it + correctie - k_ahk - k_ak)
    zvw = min(base, P["ZVW_MAX_BASE"]) * P["ZVW_RATE"]

    print(f"  Налог по шкале                      {f(it)}")
    print(f"  + tariefsaanpassing {P['TARIEFSAANPASSING']:.2%}"
          f"           {f(correctie)}")
    print(f"  − algemene heffingskorting          {f(-k_ahk)}")
    print(f"  − arbeidskorting                    {f(-k_ak)}")
    print("  " + "-" * 66)
    print(f"  BOX 1                               {f(box1)}")
    print(f"    в декларации                      {f(DECL_BOX1)}"
          f"   расхождение {box1 - DECL_BOX1:+.0f}")
    print()
    print(f"  ZVW {P['ZVW_RATE']:.2%}                            {f(zvw)}")
    print(f"    в декларации                      {f(DECL_ZVW)}"
          f"   расхождение {zvw - DECL_ZVW:+.0f}")
    print()
    print(f"  ВСЕГО (без Box 3)                   {f(box1 + zvw)}")
    print(f"    в декларации                      {f(DECL_BOX1 + DECL_ZVW)}"
          f"   расхождение {box1 + zvw - DECL_BOX1 - DECL_ZVW:+.0f}")
    print("=" * 70)

    err = abs(box1 + zvw - DECL_BOX1 - DECL_ZVW)
    share = err / (DECL_BOX1 + DECL_ZVW)
    print(f"  Ошибка модели: {err:.0f} EUR на {DECL_BOX1 + DECL_ZVW:,} "
          f"= {share:.2%}".replace(",", " "))
    print("  Модель воспроизводит декларацию — расчёту на 2026 можно верить."
          if share < 0.005 else "  Расхождение великовато, разбираться.")
    print("=" * 70)
    print()
    print("  Параметры 2025 сверены с belastingdienst.nl:")
    print(f"    шкала {P['BRACKETS'][0][1]:.2%} до {P['BRACKETS'][0][0]:,}"
          f" · {P['BRACKETS'][1][1]:.2%} до {P['BRACKETS'][1][0]:,}"
          f" · {P['BRACKETS'][2][1]:.2%} выше".replace(",", " "))
    print(f"    AHK {P['AHK_MAX']:,} от {P['AHK_START']:,} по "
          f"{P['AHK_RATE']:.3%}".replace(",", " "))
    print(f"    arbeidskorting {P['AK_MAX']:,} от {P['AK_TOP']:,} по "
          f"{P['AK_RATE']:.3%}".replace(",", " "))
    print(f"    Zvw {P['ZVW_RATE']:.2%}, потолок {P['ZVW_MAX_BASE']:,}"
          .replace(",", " "))


if __name__ == "__main__":
    main()
