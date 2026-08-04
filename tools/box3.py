#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Налог Box 3 за год — строка того же счёта, что и налог с гонорара.

Зачем отдельный инструмент. tools/tax_zzp.py считает Box 1 и ZVW — то, что
компенсирует клиент. Box 3 приходит ТЕМ ЖЕ годовым начислением, но клиент
его не компенсирует: он платит за налог с гонорара, а не за налог на
капитал. В резерве под июнь эта строка должна стоять отдельно.

Срез — 1 ЯНВАРЯ. Считается по составу капитала на эту дату, а не по
сегодняшнему. Купленное в феврале в базу года не попадает.

    python3 tools/box3.py --investments 160000 --savings 30000
    python3 tools/box3.py --investments 160000 --savings 30000 --actual-return 4200

Правило tegenbewijs: если фактическая доходность за год ниже вменённой,
платить можно с фактической. Флаг --actual-return показывает, что это даёт.
"""

from __future__ import annotations

import argparse

# --- параметры 2026 (ориентировочные) --------------------------------------
ALLOWANCE = 59_357               # heffingsvrij vermogen на человека
RATE = 0.36                      # ставка налога на вменённый доход
DEEMED_INVESTMENTS = 0.0600      # вменённая доходность инвестиций
DEEMED_SAVINGS = 0.0128          # вменённая доходность вкладов
DEEMED_DEBTS = 0.0270            # вменённая ставка по долгам (уменьшает базу)


def compute(investments: float, savings: float, debts: float = 0.0,
            actual_return: float | None = None) -> dict:
    """Налог Box 3 за год. Все суммы — на срез 1 января."""
    grondslag = investments + savings - debts
    if grondslag <= ALLOWANCE:
        return {"grondslag": grondslag, "share": 0.0, "deemed": 0.0,
                "taxable": 0.0, "tax": 0.0, "used_actual": False}

    # Льгота уменьшает базу, а доля облагаемого капитала применяется
    # к суммарному вменённому доходу — так считает Belastingdienst.
    share = (grondslag - ALLOWANCE) / grondslag
    deemed = (investments * DEEMED_INVESTMENTS
              + savings * DEEMED_SAVINGS
              - debts * DEEMED_DEBTS)

    # Tegenbewijs: платим с меньшего из вменённого и фактического.
    used_actual = actual_return is not None and actual_return < deemed
    basis = actual_return if used_actual else deemed

    taxable = max(0.0, basis) * share
    return {"grondslag": grondslag, "share": share, "deemed": deemed,
            "taxable": taxable, "tax": taxable * RATE,
            "used_actual": used_actual}


def _f(x: float) -> str:
    return f"{x:>12,.0f}".replace(",", " ")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--investments", type=float, required=True,
                   help="ETF, акции, крипта, золото — на 1 января")
    p.add_argument("--savings", type=float, default=0.0,
                   help="счета и вклады — на 1 января")
    p.add_argument("--debts", type=float, default=0.0, help="долги — на 1 января")
    p.add_argument("--actual-return", type=float, default=None,
                   help="фактическая доходность за год в евро (tegenbewijs)")
    a = p.parse_args()

    r = compute(a.investments, a.savings, a.debts, a.actual_return)

    print("=" * 66)
    print("НАЛОГ BOX 3 — СРЕЗ 1 ЯНВАРЯ")
    print("=" * 66)
    print(f"  Инвестиции                           {_f(a.investments)}")
    print(f"  Вклады и счета                       {_f(a.savings)}")
    if a.debts:
        print(f"  Долги                                {_f(-a.debts)}")
    print(f"  = база до льготы                     {_f(r['grondslag'])}")
    print(f"  − heffingsvrij vermogen              {_f(-ALLOWANCE)}")
    print(f"  доля облагаемого капитала            {r['share']:>11.1%}")
    print()
    print(f"  Вменённый доход                      {_f(r['deemed'])}")
    if a.actual_return is not None:
        print(f"  Фактическая доходность               {_f(a.actual_return)}")
        print(f"  Применено правило tegenbewijs        "
              f"{'ДА' if r['used_actual'] else 'нет':>12}")
    print(f"  Облагаемый доход после доли          {_f(r['taxable'])}")
    print("  " + "-" * 62)
    print(f"  НАЛОГ BOX 3 ({RATE:.0%})                    {_f(r['tax'])}")
    if r["grondslag"]:
        print(f"  От капитала                          {r['tax'] / r['grondslag']:>11.2%}")
    print("=" * 66)
    print()
    print("Клиент этот налог НЕ компенсирует — он платит за налог с гонорара.")
    print("В резерве под июнь строка стоит отдельно от Box 1 и ZVW.")
    print("Параметры 2026 приблизительные; точную цифру даёт форма Belastingdienst.")


if __name__ == "__main__":
    main()
