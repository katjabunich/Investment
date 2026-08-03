# Построчный пересчёт ABN AMRO (личный счёт), январь-июль 2026.
# Правила категоризации уточнены владелицей 03.08.2026:
#   Vip District -> одежда (не сертификаты)
#   iHerb -> медицина (витамины)
#   Pulcinos -> кафе (ресторан в Тилбурге, не поездка)
#   аптеки за границей -> медицина, не поездки
#   входящие Tikkie в дни бронирований -> компенсации поездок
# Знак: расход положительный, поступление отрицательное.

TRAVEL = [           # брони: отели и билеты
    ("26.01 Booking",            98.00),
    ("30.03 SunExpress via IATA",196.14),
    ("30.03 TUI Airlines Belgium",232.99),
    ("25.04 Booking (PayPal)",   570.44),
    ("14.05 SunExpress inflight",  2.99),
    ("09.06 Booking (PayPal)",   431.51),
    ("09.06 Ryanair (PayPal)",   173.84),
    ("09.06 Ryanair (PayPal)",   124.74),
    ("13.06 Booking (PayPal)",   522.00),
    ("16.06 Booking (PayPal)",    68.63),
    # 12.06 Booking 468,80 возвращён 13.06 -> в расход не входит
]

# Входящие, совпадающие по датам с бронированиями
TRAVEL_COMP = [
    ("24.01 Tikkie",             190.00),
    ("02.03 Tikkie",             200.00),
    ("03.03 Tikkie",             200.00),
    ("24.04 Sofiia Medvedieva",  125.00),
    ("25.04 Tikkie",             125.00),
    ("06.06 Tikkie",             125.00),
    ("11.06 Tikkie",             150.00),
    ("11.06 Tikkie",               7.50),
    ("15.06 Tikkie",             480.00),   # день PayPal -468,80
    ("16.06 Tikkie",             125.00),   # день PayPal -522,00
    ("16.06 Tikkie",              35.00),
]

HAIR = [("21.01", 300), ("24.03", 300), ("22.05", 240), ("23.07", 300)]

COSMETICS_2026 = [
    ("09.04 THG Beauty Limited", 51.76),
    ("04.04 THG Beauty (PayPal)",48.30),
    ("25.07 Paula's Choice",    145.40),
]
COSMETICS_2025 = [   # из построчного разбора сен-дек 2025 + PayPal
    ("21.12 THG Beauty",        183.54),
    ("25.12 NICHE BEAUTY LAB",   61.11),
    ("25.12 Korean-Skincare",    19.47),
    ("ETOS, прочее сен-дек",    110.71),
]

CLOTHES_2026_OUT = [269.80, 985.67, 664.28]          # Zalando
CLOTHES_2026_BACK = [189.90, 43.00, 159.00, 43.00,   # возвраты Zalando
                     274.84, 236.29, 100.00, 93.99, 53.00]
VIP_DISTRICT = [200.00, 192.00, 96.00, 36.00]        # -> одежда

MED_EXTRA_2026 = [   # сверх регулярных 133/мес, посчитанных по 2025
    ("09.04 iHerb (витамины)",  109.83),
    ("аптеки через PayPal",     250.57),
    ("22.06 PHARMA CHAUVET",     11.85),
    ("19.06 PHARMACIE TOUR M",   22.50),
    ("01.04 Apotheek Hoefstraat",17.99),
    ("Infomedics x2",            28.15),
]


def s(rows):
    return sum(v for _, v in rows) if rows and isinstance(rows[0], tuple) else sum(rows)


print("=" * 68)
print("ПЕРЕСЧЁТ ПО УТОЧНЁННЫМ ПРАВИЛАМ")
print("=" * 68)

t_gross, t_comp = s(TRAVEL), s(TRAVEL_COMP)
t26 = t_gross - t_comp
t25 = 1941.79 - (900.00 + 269.00)          # сен-дек 2025 из cat.py
print("\nПОЕЗДКИ (брони: отели и билеты)")
print(f"  2026 янв-июл: валовые {t_gross:8.2f}  возвраты {t_comp:8.2f}  нетто {t26:8.2f}")
print(f"  2025 сен-дек: валовые {1941.79:8.2f}  возвраты {1169.00:8.2f}  нетто {t25:8.2f}")
print(f"  ЗА 11 МЕСЯЦЕВ нетто {t26 + t25:.2f}  =  {(t26 + t25) / 11:.0f} EUR/мес")
print(f"  (доля возврата: {t_comp / t_gross:.0%} в 2026)")

hair = s([(d, v) for d, v in HAIR]) + 450.00   # +26.11.2025 Aram 350 и Demetra 100
cosm = s(COSMETICS_2026) + s(COSMETICS_2025)
print("\nКРАСОТА И УХОД")
print(f"  парикмахер: 5 визитов за 11 мес       {hair:8.2f}  = {hair / 11:3.0f}/мес")
print(f"  косметика (без iHerb - он в медицину) {cosm:8.2f}  = {cosm / 11:3.0f}/мес")
print(f"  ИТОГО {(hair + cosm) / 11:.0f} EUR/мес")

cl26 = sum(CLOTHES_2026_OUT) - sum(CLOTHES_2026_BACK) + sum(VIP_DISTRICT)
cl_all = cl26 + 809.27                          # сен-дек 2025 нетто
print("\nОДЕЖДА (Vip District включён)")
print(f"  2026: Zalando {sum(CLOTHES_2026_OUT):.2f} - возвраты {sum(CLOTHES_2026_BACK):.2f}"
      f" + Vip District {sum(VIP_DISTRICT):.2f} = {cl26:.2f}")
print(f"  2025 сен-дек нетто {809.27:.2f}")
print(f"  ИТОГО {cl_all / 11:.0f} EUR/мес")

med = 133.0 + s(MED_EXTRA_2026) / 11
print("\nМЕДИЦИНА (регулярная, с витаминами и аптеками в поездках)")
print(f"  база по 2025                      133/мес")
print(f"  + iHerb, аптеки, Infomedics       {s(MED_EXTRA_2026) / 11:3.0f}/мес")
print(f"  ИТОГО {med:.0f} EUR/мес")
print("=" * 68)
