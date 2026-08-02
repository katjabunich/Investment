#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прикидка налога ZZP по шкале Box 1 — сколько отложить на июньский платёж.

Считает: прибыль -> вычеты предпринимателя -> налогооблагаемая база ->
подоходный налог по шкале -> минус налоговые скидки -> плюс взнос ZVW.

Отдельно показывает, что даёт взнос в lijfrente: у него двойной эффект —
уменьшает базу Box 1 и выводит капитал из Box 3.

ВНИМАНИЕ: параметры 2026 года приблизительные. Это ориентир для планирования
резерва, а не декларация. Точную цифру даёт форма Belastingdienst.

    python3 tools/tax_zzp.py --revenue 101306 --expenses 11435
    python3 tools/tax_zzp.py --revenue 101306 --expenses 11435 --lijfrente 6000
"""

from __future__ import annotations

import argparse

# --- параметры 2026 (ориентировочные) --------------------------------------
BRACKETS = [(38_883, 0.3570), (79_137, 0.3756), (float("inf"), 0.4950)]
ZELFSTANDIGENAFTREK = 1_200      # план поэтапного снижения
MKB_VRIJSTELLING = 0.127
ZVW_RATE = 0.0526
ZVW_MAX_BASE = 77_000

AHK_MAX = 3_068                  # algemene heffingskorting
AHK_START = 28_406               # выше этого убывает
AHK_RATE = 0.06337
AHK_ZERO = 76_817

AK_MAX = 5_599                   # arbeidskorting
AK_TOP = 39_957                  # где достигает максимума
AK_RATE = 0.0651                 # скорость убывания выше
AK_ZERO = 124_934

# Box 3 — для оценки побочного эффекта lijfrente
BOX3_EFFECTIVE = 0.0216          # 6% вменённых x 36%


def income_tax(base: float) -> float:
    """Подоходный налог по шкале Box 1."""
    tax, low = 0.0, 0.0
    for high, rate in BRACKETS:
        if base > low:
            tax += (min(base, high) - low) * rate
        low = high
    return tax


def algemene_heffingskorting(base: float) -> float:
    if base <= AHK_START:
        return AHK_MAX
    if base >= AHK_ZERO:
        return 0.0
    return max(0.0, AHK_MAX - (base - AHK_START) * AHK_RATE)


def arbeidskorting(arbeidsinkomen: float) -> float:
    if arbeidsinkomen <= AK_TOP:
        return AK_MAX
    if arbeidsinkomen >= AK_ZERO:
        return 0.0
    return max(0.0, AK_MAX - (arbeidsinkomen - AK_TOP) * AK_RATE)


def compute(revenue: float, expenses: float, lijfrente: float = 0.0) -> dict:
    profit = revenue - expenses
    after_zelf = max(0.0, profit - ZELFSTANDIGENAFTREK)
    after_mkb = after_zelf * (1 - MKB_VRIJSTELLING)
    base = max(0.0, after_mkb - lijfrente)

    it = income_tax(base)
    ahk = algemene_heffingskorting(base)
    ak = arbeidskorting(after_mkb)
    it_net = max(0.0, it - ahk - ak)
    zvw = min(profit, ZVW_MAX_BASE) * ZVW_RATE

    return {
        "revenue": revenue, "expenses": expenses, "profit": profit,
        "after_zelf": after_zelf, "after_mkb": after_mkb, "base": base,
        "income_tax": it, "ahk": ahk, "ak": ak, "income_tax_net": it_net,
        "zvw": zvw, "total": it_net + zvw, "net": profit - it_net - zvw,
    }


def marginal_rate(base: float) -> float:
    for high, rate in BRACKETS:
        if base <= high:
            return rate
    return BRACKETS[-1][1]


def _f(x: float) -> str:
    return f"{x:>12,.0f}".replace(",", " ")


def report(r: dict, lijfrente: float = 0.0) -> None:
    print("=" * 66)
    print("НАЛОГ ZZP — ПРИКИДКА ПО ШКАЛЕ 2026")
    print("=" * 66)
    print(f"  Выручка                              {_f(r['revenue'])}")
    print(f"  − деловые расходы                    {_f(-r['expenses'])}")
    print(f"  = ПРИБЫЛЬ                            {_f(r['profit'])}")
    print()
    print(f"  − zelfstandigenaftrek                {_f(-ZELFSTANDIGENAFTREK)}")
    print(f"  − MKB-winstvrijstelling 12,7 %       {_f(-(r['after_zelf'] - r['after_mkb']))}")
    if lijfrente:
        print(f"  − взнос в lijfrente                  {_f(-lijfrente)}")
    print(f"  = НАЛОГООБЛАГАЕМАЯ БАЗА              {_f(r['base'])}")
    print()
    print(f"  Подоходный налог по шкале            {_f(r['income_tax'])}")
    print(f"  − algemene heffingskorting           {_f(-r['ahk'])}")
    print(f"  − arbeidskorting                     {_f(-r['ak'])}")
    print(f"  = подоходный к уплате                {_f(r['income_tax_net'])}")
    print(f"  + взнос ZVW 5,26 %                   {_f(r['zvw'])}")
    print("  " + "-" * 62)
    print(f"  ВСЕГО НАЛОГОВ                        {_f(r['total'])}")
    print(f"  Эффективная ставка от выручки        {r['total'] / r['revenue']:>11.1%}")
    print()
    print(f"  Остаётся после налогов               {_f(r['net'])}")
    print(f"  В месяц                              {_f(r['net'] / 12)}")
    print()
    mr = marginal_rate(r["base"])
    print(f"  Предельная ставка на следующий евро  {mr:>11.2%}")
    to_top = BRACKETS[1][0] - r["base"]
    if to_top > 0:
        print(f"  До верхней ставки 49,5 % осталось    {_f(to_top)}")
    print("=" * 66)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--revenue", type=float, required=True, help="выручка за год")
    p.add_argument("--expenses", type=float, default=0.0, help="деловые расходы за год")
    p.add_argument("--lijfrente", type=float, default=0.0, help="взнос в lijfrente за год")
    a = p.parse_args()

    base = compute(a.revenue, a.expenses)
    report(base)

    if a.lijfrente:
        with_l = compute(a.revenue, a.expenses, a.lijfrente)
        print()
        report(with_l, a.lijfrente)
        saved = base["total"] - with_l["total"]
        box3 = a.lijfrente * BOX3_EFFECTIVE
        print()
        print("=" * 66)
        print(f"ЧТО ДАЁТ LIJFRENTE {a.lijfrente:,.0f} EUR/ГОД".replace(",", " "))
        print("=" * 66)
        print(f"  Экономия налога Box 1                {_f(saved)}")
        print(f"  Экономия Box 3 (2,16 % с суммы)      {_f(box3)}")
        print(f"  ИТОГО в первый год                   {_f(saved + box3)}")
        print(f"  На каждый вложенный евро             {(saved + box3) / a.lijfrente:>11.1%}")
        print("=" * 66)

    print("\nПараметры 2026 приблизительные. Это ориентир для планирования")
    print("резерва, а не декларация — точную цифру даёт форма Belastingdienst.")


if __name__ == "__main__":
    main()
