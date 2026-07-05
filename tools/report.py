#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генератор наглядного HTML-отчёта для пассивного инвестора.

Скрипт собирает единую самодостаточную страницу reports/report.html:

  1. запускает симуляцию прогноза (tools/forecast.py --json) во временный файл
     и читает её результат (перцентили капитала по годам, вероятность цели);
  2. читает config/portfolio.yaml (стоимость, взнос, состав) и
     config/monitor.yaml (пороги траншей докупки на просадке);
  3. рисует всё это ИНЛАЙН-SVG (без matplotlib-картинок, без CDN, без внешних
     ссылок) — файл можно открыть на любом компьютере офлайн.

Запуск:
    python3 tools/report.py
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FORECAST = ROOT / "tools" / "forecast.py"
PORTFOLIO_YAML = ROOT / "config" / "portfolio.yaml"
MONITOR_YAML = ROOT / "config" / "monitor.yaml"
REPORTS_DIR = ROOT / "reports"
OUT_HTML = REPORTS_DIR / "report.html"

# Параметры прогона симуляции для отчёта.
YEARS = 20
GOAL = 1_000_000          # целевая сумма, для которой считаем вероятность
SEED = 42                 # фиксируем — чтобы отчёт был воспроизводимым

# Фиксированные цвета позиций портфеля (по тикеру).
POSITION_COLORS = {
    "SXR8": "#2a78d6",
    "EXUS": "#1baf7a",
    "EIMI": "#eda100",
    "EQQB": "#008300",
    "AVWS": "#4a3aa7",
}
FAN_BLUE = "#2a78d6"      # цвет веера прогноза

NBSP_THIN = " "      # узкий неразрывный пробел — разделитель разрядов
MILESTONE_YEARS = [5, 10, 15, 20]


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def run_forecast() -> dict:
    """Запустить forecast.py --json во временный файл и вернуть результат."""
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="r", encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [
                sys.executable, str(FORECAST),
                "--years", str(YEARS),
                "--goal", str(GOAL),
                "--seed", str(SEED),
                "--json", str(tmp_path),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "forecast.py завершился с ошибкой:\n" + (proc.stderr or proc.stdout)
            )
        with open(tmp_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Форматирование
# ---------------------------------------------------------------------------
def fmt_money(x: float, currency: str = "EUR") -> str:
    """1 234 567 € — с узкими пробелами-разрядами."""
    sign = "€" if currency == "EUR" else currency
    body = f"{round(x):,}".replace(",", NBSP_THIN)
    return f"{body} {sign}"


def fmt_int(x: float) -> str:
    return f"{round(x):,}".replace(",", NBSP_THIN)


def esc(s: str) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------------------
# SVG: график-веер реального капитала
# ---------------------------------------------------------------------------
def _nice_step(vmax: float) -> float:
    """Подобрать «круглый» шаг сетки, чтобы получилось ~4-6 линий."""
    raw = vmax / 5.0
    mag = 10 ** (len(str(int(raw))) - 1)
    for m in (1, 2, 2.5, 5, 10):
        if m * mag >= raw:
            return m * mag
    return 10 * mag


def build_fan_svg(fan: dict, currency: str) -> str:
    """Инлайн-SVG веера: полосы P5-P95 и P25-P75, жирная медиана, подписи концов."""
    years = fan["years"]
    p5, p25, p50 = fan["real_p5"], fan["real_p25"], fan["real_p50"]
    p75, p95 = fan["real_p75"], fan["real_p95"]

    W, H = 900, 430
    ml, mr, mt, mb = 66, 150, 22, 46
    pw, ph = W - ml - mr, H - mt - mb
    ymax = _nice_step(max(p95)) * (int(max(p95) / _nice_step(max(p95))) + 1)
    xmax = years[-1]

    def sx(year: float) -> float:
        return ml + (year / xmax) * pw

    def sy(val: float) -> float:
        return mt + ph - (val / ymax) * ph

    def band(lower: list[float], upper: list[float]) -> str:
        pts = [f"{sx(y):.1f},{sy(v):.1f}" for y, v in zip(years, upper)]
        pts += [f"{sx(y):.1f},{sy(v):.1f}" for y, v in zip(reversed(years), reversed(lower))]
        return " ".join(pts)

    def line(vals: list[float]) -> str:
        return " ".join(f"{sx(y):.1f},{sy(v):.1f}" for y, v in zip(years, vals))

    parts: list[str] = []
    parts.append(
        f'<svg viewBox="0 0 {W} {H}" width="100%" '
        f'role="img" aria-label="Прогноз реального капитала по годам" '
        f'preserveAspectRatio="xMidYMid meet" font-family="inherit">'
    )

    # Горизонтальная сетка + подписи по оси Y (в млн).
    step = _nice_step(ymax)
    v = 0.0
    while v <= ymax + 1:
        y = sy(v)
        parts.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" '
            f'class="grid" />'
        )
        label = "0" if v == 0 else f"{v / 1_000_000:g} млн"
        parts.append(
            f'<text x="{ml - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'class="axis num">{label}</text>'
        )
        v += step

    # Подписи по оси X (годы).
    for yr in (0, 5, 10, 15, 20):
        x = sx(yr)
        parts.append(
            f'<text x="{x:.1f}" y="{mt + ph + 24:.1f}" text-anchor="middle" '
            f'class="axis num">{yr}</text>'
        )
    parts.append(
        f'<text x="{ml + pw / 2:.1f}" y="{H - 6:.1f}" text-anchor="middle" '
        f'class="axis">лет</text>'
    )

    # Полосы диапазонов.
    parts.append(
        f'<polygon points="{band(p5, p95)}" fill="{FAN_BLUE}" '
        f'fill-opacity="0.16" stroke="none" />'
    )
    parts.append(
        f'<polygon points="{band(p25, p75)}" fill="{FAN_BLUE}" '
        f'fill-opacity="0.30" stroke="none" />'
    )
    # Медиана.
    parts.append(
        f'<polyline points="{line(p50)}" fill="none" stroke="{FAN_BLUE}" '
        f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round" />'
    )

    # Подписи концов (год 20).
    ends = [
        ("P95", p95[-1], FAN_BLUE),
        ("P75", p75[-1], FAN_BLUE),
        ("P50", p50[-1], FAN_BLUE),
        ("P25", p25[-1], FAN_BLUE),
        ("P5", p5[-1], FAN_BLUE),
    ]
    lx = ml + pw + 8
    for name, val, _ in ends:
        y = sy(val)
        parts.append(
            f'<circle cx="{ml + pw:.1f}" cy="{y:.1f}" r="2.5" fill="{FAN_BLUE}" />'
        )
        parts.append(
            f'<text x="{lx:.1f}" y="{y - 2:.1f}" class="end-lbl">{name}</text>'
        )
        parts.append(
            f'<text x="{lx:.1f}" y="{y + 12:.1f}" class="end-val num">'
            f'{fmt_money(val, currency)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML-блоки
# ---------------------------------------------------------------------------
def build_hero(data: dict, portfolio: dict, currency: str) -> str:
    """Четыре крупных числа в шапке."""
    current = portfolio["current_value"]
    contrib = portfolio["monthly_contribution"]
    real20 = data["milestones"][YEARS - 1]["real"]["p50"]
    goal = data.get("goal")

    cards = [
        ("Портфель сейчас", fmt_money(current, currency), ""),
        ("Взнос в месяц", fmt_money(contrib, currency), ""),
        (f"Медиана через {YEARS} лет",
         fmt_money(real20, currency), "в сегодняшних деньгах"),
    ]
    if goal:
        cards.append((
            "Вероятность цели",
            f"{round(goal['prob_real'])} %",
            f"накопить {fmt_money(goal['amount'], currency)} (реально)",
        ))

    items = []
    for title, big, sub in cards:
        sub_html = f'<div class="hero-sub">{esc(sub)}</div>' if sub else ""
        items.append(
            f'<div class="hero-card"><div class="hero-title">{esc(title)}</div>'
            f'<div class="hero-num num">{big}</div>{sub_html}</div>'
        )
    return f'<section class="hero">{"".join(items)}</section>'


def build_milestones_table(data: dict, currency: str) -> str:
    """Таблица вех 5/10/15/20 лет — реальные перцентили."""
    rows = []
    ms_by_year = {m["year"]: m for m in data["milestones"]}
    for yr in MILESTONE_YEARS:
        if yr not in ms_by_year:
            continue
        r = ms_by_year[yr]["real"]
        cells = "".join(
            f'<td class="num">{fmt_money(r[p], currency)}</td>'
            for p in ("p5", "p25", "p50", "p75", "p95")
        )
        rows.append(f'<tr><th scope="row">{yr} лет</th>{cells}</tr>')

    return f'''<section class="card">
  <h2>Капитал по вехам <span class="muted">— реальные деньги (сегодняшняя покупательная способность)</span></h2>
  <div class="table-wrap">
  <table>
    <thead><tr><th scope="col">Срок</th>
      <th scope="col">P5 (пессимист.)</th><th scope="col">P25</th>
      <th scope="col">P50 (медиана)</th><th scope="col">P75</th>
      <th scope="col">P95 (оптимист.)</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
</section>'''


def build_composition(portfolio: dict) -> str:
    """Состав портфеля горизонтальными полосами."""
    holdings = portfolio.get("holdings", [])
    total_w = sum(h["weight"] for h in holdings)
    bars = []
    for h in holdings:
        ticker = h["ticker"]
        pct = h["weight"] / total_w * 100
        color = POSITION_COLORS.get(ticker, "#888888")
        name = esc(h.get("name", ""))
        bars.append(f'''<div class="pos">
      <div class="pos-head">
        <span class="pos-ticker">{esc(ticker)}</span>
        <span class="pos-pct num">{pct:.0f} %</span>
      </div>
      <div class="pos-name">{name}</div>
      <div class="bar-track"><div class="bar-fill" style="width:{pct:.2f}%;background:{color}"></div></div>
    </div>''')
    return f'''<section class="card">
  <h2>Состав портфеля</h2>
  <div class="composition">{"".join(bars)}</div>
</section>'''


def build_dip_plan(monitor: dict, currency: str) -> str:
    """Карточка «план докупок на просадке»."""
    base = monitor.get("currency", currency)
    # базовый взнос для расчёта суммы транша
    series = monitor.get("series", [])
    monthly = series[0]["monthly_contribution"] if series else 2700
    tranches = monitor.get("tranches", [])
    rows = []
    for t in tranches:
        dd = t["drawdown"]
        amount = t["installments"] * monthly
        rows.append(f'''<div class="dip-row">
      <span class="dip-dd num">{dd} %</span>
      <span class="dip-arrow">докупить</span>
      <span class="dip-amt num">+ {fmt_money(amount, base)}</span>
    </div>''')
    return f'''<section class="card">
  <h2>План докупок на просадке <span class="muted">— из резерва, сверх обычного взноса</span></h2>
  <div class="dip">{"".join(rows)}</div>
</section>'''


def build_html(data: dict, portfolio: dict, monitor: dict) -> str:
    currency = data["params"].get("base_currency", "EUR")
    hero = build_hero(data, portfolio, currency)
    fan = build_fan_svg(data["fan"], currency)
    table = build_milestones_table(data, currency)
    comp = build_composition(portfolio)
    dip = build_dip_plan(monitor, currency)
    gen_date = datetime.now().strftime("%d.%m.%Y")
    src = data["params"].get("return_source")
    src_note = ("исторические месячные данные" if src == "monthly_block_bootstrap"
                else "встроенный резервный датасет годовых доходностей")

    css = CSS
    return f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Инвестиционный отчёт</title>
<style>{css}</style>
</head>
<body>
<main>
  <header class="top">
    <h1>Прогноз капитала</h1>
    <div class="top-sub">Монте-Карло, {fmt_int(data["params"]["n_paths"])} сценариев · источник: {src_note}</div>
  </header>

  {hero}

  <section class="card">
    <h2>Веер прогноза <span class="muted">— реальный капитал по годам (полосы P5–P95 и P25–P75, жирная линия — медиана)</span></h2>
    <div class="chart">{fan}</div>
  </section>

  {table}

  <div class="two-col">
    {comp}
    {dip}
  </div>

  <footer>
    <div class="gen">Отчёт сгенерирован {gen_date}</div>
    <div class="disclaimer">Прошлые доходности не гарантируют будущих; это не индивидуальная инвестиционная рекомендация.</div>
  </footer>
</main>
</body>
</html>'''


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CSS = """
:root{
  --bg:#f9f9f7; --card:#fcfcfb; --text:#0b0b0b; --secondary:#52514e;
  --grid:#e1e0d9;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0d0d0d; --card:#1a1a19; --text:#ffffff; --secondary:#c3c2b7;
    --grid:#2c2c2a;
  }
}
*{box-sizing:border-box;}
html,body{margin:0;padding:0;}
body{
  background:var(--bg); color:var(--text);
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.45; -webkit-font-smoothing:antialiased;
}
.num{font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1;}
main{max-width:1000px; margin:0 auto; padding:32px 24px 48px;}
.top{margin-bottom:24px;}
h1{font-size:26px; font-weight:650; margin:0 0 4px;}
.top-sub{color:var(--secondary); font-size:14px;}
h2{font-size:16px; font-weight:600; margin:0 0 16px;}
h2 .muted{font-weight:400; color:var(--secondary); font-size:13px;}
.muted{color:var(--secondary);}

.hero{
  display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:14px; margin-bottom:20px;
}
.hero-card{
  background:var(--card); border:1px solid var(--grid); border-radius:12px;
  padding:18px 20px;
}
.hero-title{font-size:13px; color:var(--secondary); margin-bottom:8px;}
.hero-num{font-size:30px; font-weight:650; letter-spacing:-0.5px;}
.hero-sub{font-size:12px; color:var(--secondary); margin-top:6px;}

.card{
  background:var(--card); border:1px solid var(--grid); border-radius:12px;
  padding:20px 22px; margin-bottom:20px;
}
.chart{width:100%;}

svg .grid{stroke:var(--grid); stroke-width:1;}
svg .axis{fill:var(--secondary); font-size:12px;}
svg .end-lbl{fill:var(--secondary); font-size:11px; font-weight:600;}
svg .end-val{fill:var(--text); font-size:12px;}

.table-wrap{overflow-x:auto;}
table{width:100%; border-collapse:collapse; font-size:14px;}
th,td{text-align:right; padding:10px 8px; white-space:nowrap;}
thead th{
  color:var(--secondary); font-weight:500; font-size:12px;
  border-bottom:1px solid var(--grid);
}
th[scope=col]:first-child, th[scope=row]{text-align:left;}
th[scope=row]{font-weight:600;}
tbody tr+tr td, tbody tr+tr th{border-top:1px solid var(--grid);}

.two-col{display:grid; grid-template-columns:1fr 1fr; gap:20px;}
.two-col .card{margin-bottom:0;}
@media (max-width:720px){ .two-col{grid-template-columns:1fr;} }

.composition{display:flex; flex-direction:column; gap:16px;}
.pos-head{display:flex; justify-content:space-between; align-items:baseline;}
.pos-ticker{font-weight:650; font-size:15px;}
.pos-pct{font-weight:600; font-size:15px;}
.pos-name{font-size:12px; color:var(--secondary); margin:2px 0 6px;}
.bar-track{background:var(--grid); border-radius:6px; height:10px; overflow:hidden;}
.bar-fill{height:100%; border-radius:6px;}

.dip{display:flex; flex-direction:column; gap:12px;}
.dip-row{
  display:grid; grid-template-columns:auto 1fr auto; align-items:center;
  gap:12px; padding:10px 12px; border:1px solid var(--grid); border-radius:8px;
}
.dip-dd{font-weight:650; font-size:16px;}
.dip-arrow{color:var(--secondary); font-size:13px;}
.dip-amt{font-weight:600; font-size:15px;}

footer{margin-top:28px; color:var(--secondary); font-size:12px;}
.gen{margin-bottom:4px;}
.disclaimer{font-style:italic;}
"""


# ---------------------------------------------------------------------------
def main() -> int:
    data = run_forecast()
    portfolio = load_yaml(PORTFOLIO_YAML)
    monitor = load_yaml(MONITOR_YAML)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_text = build_html(data, portfolio, monitor)
    with open(OUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(html_text)
    print(f"Отчёт сохранён: {OUT_HTML}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
