#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Прикидка налога ZZP по шкале Box 1 — сколько отложить на июньский платёж.

Считает: прибыль -> вычеты предпринимателя -> налогооблагаемая база ->
подоходный налог по шкале -> минус налоговые скидки -> плюс взнос ZVW.

Отдельно показывает, что даёт взнос в lijfrente: у него двойной эффект —
уменьшает базу Box 1 и выводит капитал из Box 3.

Параметры 2026 сверены с belastingdienst.nl (август 2026): шкала, обе
скидки, Zvw. Это по-прежнему ориентир для планирования резерва, а не
декларация — округления и частные обстоятельства даёт только форма.

    python3 tools/tax_zzp.py --revenue 101306 --expenses 11435
    python3 tools/tax_zzp.py --revenue 101306 --expenses 11435 --lijfrente 6000
"""

from __future__ import annotations

import argparse

# --- ЧТО УЧТЕНО И ЧТО НЕТ --------------------------------------------------
#
# ПРИМЕНЯЕТСЯ:
#   zelfstandigenaftrek 1 200  — ТРЕБУЕТ urencriterium (1 225 часов в год
#       на предприятие). При доходе 105 236 от одного клиента критерий
#       заведомо выполняется по факту, но Belastingdienst требует
#       urenregistratie — журнал часов. Если журнала нет и вычет снимут,
#       налог вырастет на 587.
#   MKB-winstvrijstelling 12,7 % — БЕЗ urencriterium, даётся всегда.
#   algemene heffingskorting — убывает от 29 736, при базе 80 841 равна нулю.
#   arbeidskorting — убывает от 45 592, здесь 3 390 из максимальных 5 685.
#   Zvw 4,85 % — база 80 841 чуть выше потолка 79 409, поэтому взнос упёрся
#       в потолок, НО запас всего 1 432: при меньшей прибыли Zvw снова
#       начинает работать на пределе и добавляет 4,85 п.п. к предельной ставке.
#
# НЕ ПРИМЕНЯЕТСЯ И ЭТО ВЕРНО:
#   startersaftrek 2 123 — только 3 раза за первые 5 лет предпринимательства.
#       Владелица больше не начинающая, поэтому вычета нет. Если бы он ещё
#       был доступен, налог был бы меньше на 1 020 — проверить, не остался
#       ли неиспользованный год, стоит один раз.
#   meewerkaftrek — партнёр в предприятии не работает.
#   IACK (комбинированная скидка) — детей до 12 лет нет.
#   FOR / oudedagsreserve — отменена с 2023.
#   middeling — отменена, последний период 2022-2024.
#   KIA — только если за календарный год куплено оборудования больше 2 901;
#       считается отдельно, см. docs/what-matters.md.
#
# НЕ СЧИТАЕТСЯ ЗДЕСЬ ВООБЩЕ, НО ПРИХОДИТ ТЕМ ЖЕ СЧЁТОМ:
#   НАЛОГ BOX 3 на капитал. Это отдельная строка того же годового
#   начисления, и клиент её НЕ компенсирует — он компенсирует налог
#   с гонорара. Оценка за 2026: ~2 500. См. docs/tax-reserve.md.
#
# --- параметры по годам (СВЕРЕНЫ с belastingdienst.nl) ----------------------
# 2026 — август 2026. 2025 — сверено там же, нужно для проверки модели
# на реальной декларации: она даёт независимую точку опоры.
YEARS = {
    2026: dict(
        BRACKETS=[(38_883, 0.3575), (78_426, 0.3756), (float("inf"), 0.4950)],
        ZELFSTANDIGENAFTREK=1_200,
        STARTERSAFTREK=0,              # исчерпан
        MKB_VRIJSTELLING=0.127,
        ZVW_RATE=0.0485, ZVW_MAX_BASE=79_409,
        AHK_MAX=3_115, AHK_START=29_736, AHK_RATE=0.06398, AHK_ZERO=78_426,
        AK_MAX=5_685, AK_TOP=45_592, AK_RATE=0.06510, AK_ZERO=132_920,
        TARIEFSAANPASSING=0.1194, TARIEF_THRESHOLD=78_426,
    ),
    2025: dict(
        BRACKETS=[(38_441, 0.3582), (76_817, 0.3748), (float("inf"), 0.4950)],
        ZELFSTANDIGENAFTREK=2_470,
        STARTERSAFTREK=2_123,          # в 2025 ещё применялся
        MKB_VRIJSTELLING=0.127,
        ZVW_RATE=0.0526, ZVW_MAX_BASE=75_864,
        AHK_MAX=3_068, AHK_START=28_406, AHK_RATE=0.06337, AHK_ZERO=76_817,
        AK_MAX=5_599, AK_TOP=43_071, AK_RATE=0.06510, AK_ZERO=129_078,
        TARIEFSAANPASSING=0.1202, TARIEF_THRESHOLD=76_817,
    ),
}

DEFAULT_YEAR = 2026
_P = YEARS[DEFAULT_YEAR]
BRACKETS = _P["BRACKETS"]
ZELFSTANDIGENAFTREK = _P["ZELFSTANDIGENAFTREK"]
STARTERSAFTREK = _P["STARTERSAFTREK"]
MKB_VRIJSTELLING = _P["MKB_VRIJSTELLING"]
ZVW_RATE = _P["ZVW_RATE"]
ZVW_MAX_BASE = _P["ZVW_MAX_BASE"]
AHK_MAX = _P["AHK_MAX"]; AHK_START = _P["AHK_START"]
AHK_RATE = _P["AHK_RATE"]; AHK_ZERO = _P["AHK_ZERO"]
AK_MAX = _P["AK_MAX"]; AK_TOP = _P["AK_TOP"]
AK_RATE = _P["AK_RATE"]; AK_ZERO = _P["AK_ZERO"]
TARIEFSAANPASSING = _P["TARIEFSAANPASSING"]
TARIEF_THRESHOLD = _P["TARIEF_THRESHOLD"]

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


def compute(revenue: float, expenses: float, lijfrente: float = 0.0,
            kia: float = 0.0) -> dict:
    profit = revenue - expenses - kia
    after_zelf = max(0.0, profit - ZELFSTANDIGENAFTREK - STARTERSAFTREK)
    after_mkb = after_zelf * (1 - MKB_VRIJSTELLING)
    base = max(0.0, after_mkb - lijfrente)

    # вычеты, чью ставку ограничивает tariefsaanpassing: ondernemersaftrek
    # и MKB-winstvrijstelling. KIA — investeringsaftrek, в список НЕ входит.
    posten = (profit - after_mkb) if profit > after_mkb else 0.0
    correctie = TARIEFSAANPASSING * min(
        posten, max(0.0, base + posten - TARIEF_THRESHOLD))

    it = income_tax(base) + correctie
    ahk = algemene_heffingskorting(base)
    # ВАЖНО: arbeidsinkomen для arbeidskorting — прибыль ДО ondernemersaftrek
    # и ДО MKB-winstvrijstelling (после KIA). Подтверждено декларацией 2025:
    # 81 110 − 837 = 80 273 даёт 3 177 при 3 178 в декларации.
    ak = arbeidskorting(profit)
    it_net = max(0.0, it - ahk - ak)
    zvw = min(after_mkb, ZVW_MAX_BASE) * ZVW_RATE

    return {
        "revenue": revenue, "expenses": expenses, "kia": kia, "profit": profit,
        "after_zelf": after_zelf, "after_mkb": after_mkb, "base": base,
        "posten": posten, "correctie": correctie,
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
    print(f"  + взнос Zvw 4,85 %                   {_f(r['zvw'])}")
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
