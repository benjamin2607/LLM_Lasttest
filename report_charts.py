"""
Self-contained SVG charts and HTML rendering for the generated reports.

No external libraries or network resources are used, so reports open offline and
can be archived as they are. Charts follow a few fixed rules: one y-axis per chart,
at most two series per chart (blue, orange), thin lines with point markers, recessive
grid, a legend for two series, and light/dark mode via ``prefers-color-scheme``.
"""
import html
import math
import re
import time

SVG_STYLE = """
  .bg { fill: var(--surface); }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis-text { fill: var(--text-3); font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .title { fill: var(--text); font: 600 13px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .legend { fill: var(--text-2); font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .value { fill: var(--text-2); font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .limit { stroke: var(--critical); stroke-width: 1; stroke-dasharray: 4 4; }
  .limit-text { fill: var(--text-2); font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .s1 { stroke: var(--s1); fill: none; } .s1-dot { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
  .s2 { stroke: var(--s2); fill: none; } .s2-dot { fill: var(--s2); stroke: var(--surface); stroke-width: 2; }
  .s1-key { fill: var(--s1); } .s2-key { fill: var(--s2); }
  .bar { fill: var(--s1); }
"""
SVG_VARS = """
  svg { --surface: #fcfcfb; --grid: #e6e5e0; --text: #0b0b0b; --text-2: #52514e; --text-3: #7a7974;
        --s1: #2a78d6; --s2: #eb6834; --critical: #d03b3b; }
  @media (prefers-color-scheme: dark) {
    svg { --surface: #1a1a19; --grid: #2c2c29; --text: #f0efec; --text-2: #c3c2b7; --text-3: #8f8e86;
          --s1: #3987e5; --s2: #d95926; }
  }
"""


def nice_max(value):
    if not value or value <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(value))
    for step in (1, 2, 2.5, 5, 10):
        if value <= step * power:
            return step * power
    return 10 * power


def fmt_num(value):
    if value is None:
        return "–"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def line_chart(title, labels, series, unit="", x_title="", y_max=None, limit=None, width=640, height=300,
               standalone=True):
    """
    Line chart over ordinal x positions (e.g. load-test stages).

    ``series`` is a list of (name, values) with at most two entries; values may contain None.
    ``limit`` is an optional (value, label) drawn as a dashed horizontal line.
    """
    left, right, top, bottom = 52, 16, 44 if len(series) > 1 else 34, 44
    pw, ph = width - left - right, height - top - bottom
    values = [v for _, vals in series for v in vals if v is not None]
    if limit and values and max(values) < limit[0] * 0.5:
        limit = None  # Far below the limit: scale to the data instead of flattening it under the limit line.
    peak = max(values + ([limit[0] * 1.1] if limit else [])) if values else 1
    ymax = y_max or nice_max(peak)
    n = len(labels)

    def x(i):
        return left + (pw / 2 if n == 1 else pw * i / (n - 1))

    def y(v):
        return top + ph - min(v, ymax) / ymax * ph

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
           f'role="img" aria-label="{html.escape(title)}">',
           f"<style>{SVG_VARS if standalone else ''}{SVG_STYLE}</style>",
           f'<rect class="bg" width="{width}" height="{height}" rx="10"/>',
           f'<text class="title" x="{left}" y="20">{html.escape(title)}</text>']
    if len(series) > 1:
        lx = left
        for k, (name, _) in enumerate(series):
            out.append(f'<rect class="s{k + 1}-key" x="{lx}" y="29" width="14" height="3" rx="1.5"/>')
            out.append(f'<text class="legend" x="{lx + 19}" y="34">{html.escape(name)}</text>')
            lx += 30 + 7 * len(name)
    for g in range(5):
        v = ymax * g / 4
        gy = y(v)
        out.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{gy:.1f}" y2="{gy:.1f}"/>')
        out.append(f'<text class="axis-text" x="{left - 6}" y="{gy + 4:.1f}" text-anchor="end">'
                   f'{fmt_num(v)}{html.escape(unit) if g == 4 else ""}</text>')
    for i, label in enumerate(labels):
        out.append(f'<text class="axis-text" x="{x(i):.1f}" y="{height - 24}" text-anchor="middle">'
                   f'{html.escape(str(label))}</text>')
    if x_title:
        out.append(f'<text class="axis-text" x="{left + pw / 2:.1f}" y="{height - 8}" text-anchor="middle">'
                   f'{html.escape(x_title)}</text>')
    if limit and limit[0] <= ymax:
        ly = y(limit[0])
        out.append(f'<line class="limit" x1="{left}" x2="{width - right}" y1="{ly:.1f}" y2="{ly:.1f}"/>')
        out.append(f'<text class="limit-text" x="{width - right}" y="{ly - 5:.1f}" text-anchor="end">'
                   f'{html.escape(limit[1])}</text>')
    for k, (name, vals) in enumerate(series):
        cls = f"s{k + 1}"
        segment = []
        for i, v in enumerate(vals):
            if v is None:
                if len(segment) > 1:
                    out.append(f'<polyline class="{cls}" stroke-width="2" stroke-linejoin="round" '
                               f'points="{" ".join(segment)}"/>')
                segment = []
                continue
            segment.append(f"{x(i):.1f},{y(v):.1f}")
        if len(segment) > 1:
            out.append(f'<polyline class="{cls}" stroke-width="2" stroke-linejoin="round" points="{" ".join(segment)}"/>')
        for i, v in enumerate(vals):
            if v is not None:
                out.append(f'<circle class="{cls}-dot" cx="{x(i):.1f}" cy="{y(v):.1f}" r="4">'
                           f'<title>{html.escape(name)} at {labels[i]}: {fmt_num(v)}{html.escape(unit)}</title></circle>')
        # Direct label on the last value keeps the chart readable without hovering.
        last = next(((i, v) for i, v in reversed(list(enumerate(vals))) if v is not None), None)
        if last:
            out.append(f'<text class="value" x="{x(last[0]) - 6:.1f}" y="{y(last[1]) - 9:.1f}" text-anchor="end">'
                       f'{fmt_num(last[1])}{html.escape(unit)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _frame(title, width, height, standalone):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" '
            f'role="img" aria-label="{html.escape(title)}">',
            f"<style>{SVG_VARS if standalone else ''}{SVG_STYLE}</style>",
            f'<rect class="bg" width="{width}" height="{height}" rx="10"/>',
            f'<text class="title" x="52" y="20">{html.escape(title)}</text>']


def _y_axis(out, ymax, unit, left, right, top, ph, width):
    for g in range(5):
        v = ymax * g / 4
        gy = top + ph - v / ymax * ph
        out.append(f'<line class="grid" x1="{left}" x2="{width - right}" y1="{gy:.1f}" y2="{gy:.1f}"/>')
        out.append(f'<text class="axis-text" x="{left - 6}" y="{gy + 4:.1f}" text-anchor="end">'
                   f'{fmt_num(v)}{html.escape(unit) if g == 4 else ""}</text>')


def time_chart(title, times, series, unit="", y_max=None, width=640, height=280, standalone=False):
    """
    Line chart over time (epoch seconds) with clock-time ticks, e.g. per-minute values of a
    live session. ``series`` holds at most two (name, values) pairs; None values leave gaps.
    """
    from datetime import datetime  # Local import keeps the module's top-level dependencies minimal.
    left, right, top, bottom = 52, 16, 44 if len(series) > 1 else 34, 30
    pw, ph = width - left - right, height - top - bottom
    values = [v for _, vals in series for v in vals if v is not None]
    ymax = y_max or nice_max(max(values) if values else 1)
    t0, t1 = (min(times), max(times)) if times else (0, 1)
    if t1 - t0 < 60:
        t1 = t0 + 60

    def x(t):
        return left + (t - t0) / (t1 - t0) * pw

    def y(v):
        return top + ph - min(v, ymax) / ymax * ph

    out = _frame(title, width, height, standalone)
    if len(series) > 1:
        lx = left
        for k, (name, _) in enumerate(series):
            out.append(f'<rect class="s{k + 1}-key" x="{lx}" y="29" width="14" height="3" rx="1.5"/>')
            out.append(f'<text class="legend" x="{lx + 19}" y="34">{html.escape(name)}</text>')
            lx += 30 + 7 * len(name)
    _y_axis(out, ymax, unit, left, right, top, ph, width)
    multi_day = datetime.fromtimestamp(t0).date() != datetime.fromtimestamp(t1).date()
    # Ticks on round clock times, spaced so that about six labels fit.
    step = next((m * 60 for m in (1, 2, 5, 10, 15, 30, 60, 120, 240, 360, 720, 1440)
                 if (t1 - t0) / (m * 60) <= 6), 1440 * 60)
    local_shift = -time.timezone if not time.localtime(t0).tm_isdst else -time.altzone
    tick = math.ceil((t0 + local_shift) / step) * step - local_shift
    while tick <= t1:
        label = datetime.fromtimestamp(tick).strftime("%d.%m. %H:%M" if multi_day else "%H:%M")
        out.append(f'<text class="axis-text" x="{x(tick):.1f}" y="{height - 10}" text-anchor="middle">{label}</text>')
        tick += step
    dense = len(times) > 40
    for k, (name, vals) in enumerate(series):
        cls, segment = f"s{k + 1}", []
        points = list(zip(times, vals))
        for t, v in points + [(None, None)]:
            if v is None:
                if len(segment) > 1:
                    out.append(f'<polyline class="{cls}" stroke-width="2" stroke-linejoin="round" '
                               f'points="{" ".join(segment)}"/>')
                elif len(segment) == 1 and dense:
                    # A single value between gaps has no line to draw: show it as a point.
                    cx, cy = segment[0].split(",")
                    out.append(f'<circle class="{cls}-dot" cx="{cx}" cy="{cy}" r="3"/>')
                segment = []
                continue
            segment.append(f"{x(t):.1f},{y(v):.1f}")
        for t, v in points:
            if v is not None and not dense:
                out.append(f'<circle class="{cls}-dot" cx="{x(t):.1f}" cy="{y(v):.1f}" r="3.5">'
                           f'<title>{html.escape(name)} {datetime.fromtimestamp(t):%H:%M}: '
                           f'{fmt_num(v)}{html.escape(unit)}</title></circle>')
    out.append("</svg>")
    return "\n".join(out)


def histogram_chart(title, values, unit="", bins=12, width=640, height=280, standalone=False):
    """Histogram of a distribution, e.g. response times; bars with a 2 px gap and rounded tops."""
    left, right, top, bottom = 52, 16, 34, 30
    pw, ph = width - left - right, height - top - bottom
    lo, hi = 0.0, max(values) if values else 1.0
    step = nice_max((hi - lo) / bins) if hi > lo else 1.0
    n = max(1, int(math.ceil((hi - lo) / step)))
    counts = [0] * n
    for v in values:
        counts[min(n - 1, int((v - lo) // step))] += 1
    ymax = nice_max(max(counts) if counts else 1)
    out = _frame(title, width, height, standalone)
    _y_axis(out, ymax, "", left, right, top, ph, width)
    bw = pw / n
    for i, c in enumerate(counts):
        if c:
            h = c / ymax * ph
            x0, y0, w = left + i * bw + 1, top + ph - h, max(1.0, bw - 2)
            r = min(4, w / 2, h)
            out.append(f'<path class="bar" d="M{x0:.1f},{top + ph:.1f} V{y0 + r:.1f} Q{x0:.1f},{y0:.1f} {x0 + r:.1f},{y0:.1f} '
                       f'H{x0 + w - r:.1f} Q{x0 + w:.1f},{y0:.1f} {x0 + w:.1f},{y0 + r:.1f} V{top + ph:.1f} Z">'
                       f'<title>{fmt_num(lo + i * step)}–{fmt_num(lo + (i + 1) * step)}{html.escape(unit)}: '
                       f'{c} requests</title></path>')
    for i in range(0, n + 1, max(1, n // 6)):
        out.append(f'<text class="axis-text" x="{left + i * bw:.1f}" y="{height - 10}" text-anchor="middle">'
                   f'{fmt_num(lo + i * step)}{html.escape(unit) if i == 0 else ""}</text>')
    out.append("</svg>")
    return "\n".join(out)


# --- Key figure tiles shared by the load test and live session reports ---
TILE_CSS = """<style>
.meta { color: var(--text-2); margin-top: -6px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin: 16px 0; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
.tile .label { color: var(--text); font-size: 13px; font-weight: 700; letter-spacing: -.005em; }
.tile .big { font-size: 24px; font-weight: 650; margin-top: 4px; }
.tile .sub { color: var(--text-3); font-size: 12px; }
.tile table.stats { width: 100%; margin-top: 6px; font-size: 14px; }
.tile table.stats th, .tile table.stats td { border: 0; padding: 2px 0; }
.tile table.stats th { text-align: left; color: var(--text-2); font-weight: 500; }
.tile table.stats td { text-align: right; font-weight: 650; }
.banner { border: 1px solid var(--critical); border-left: 4px solid var(--critical); background: var(--surface);
  border-radius: 8px; padding: 10px 12px; margin: 12px 0; font-weight: 600; }
</style>"""


def count_tile(label, value, note=""):
    """Tile with one large value and a note below."""
    return (f"<div class='tile'><div class='label'>{html.escape(label)}</div><div class='big'>{html.escape(str(value))}</div>"
            f"<div class='sub'>{html.escape(note)}</div></div>")


def stat_tile(label, stats, unit="", note=None):
    """Tile listing average, median, p95 and maximum below each other (missing values show as –)."""
    rows = "".join(
        f"<tr><th>{name}</th><td>{html.escape(fmt_num(stats.get(key)) + unit) if stats.get(key) is not None else '–'}</td></tr>"
        for name, key in (("Ø", "mean"), ("Median", "median"), ("p95", "p95"), ("Max", "max")))
    note = note if note is not None else (f"n = {stats['n']}" if stats.get("n") is not None else "")
    return (f"<div class='tile'><div class='label'>{html.escape(label)}</div><table class='stats'>{rows}</table>"
            f"<div class='sub'>{html.escape(note)}</div></div>")


def tiles(items):
    return "<div class='tiles'>" + "".join(items) + "</div>"


# --- Minimal Markdown rendering for the reports this project generates ---
def _inline(text):
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', text)
    return text


def markdown_to_html(markdown, charts_html=None, insert_before=None):
    """
    Render the subset of Markdown used in the generated reports (headings, lists, tables,
    quotes, code blocks). ``charts_html`` is inserted before the first heading that starts
    with ``insert_before``, or appended at the end.
    """
    out, in_list, in_code, table = [], False, False, []
    inserted = charts_html is None

    def flush_table():
        nonlocal table
        if not table:
            return
        rows = [r for r in table if not re.match(r"^\|[\s:|-]+\|$", r)]
        cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
        out.append("<div class='table'><table><thead><tr>" + "".join(f"<th>{_inline(c)}</th>" for c in cells[0])
                   + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in row) + "</tr>"
                                                       for row in cells[1:]) + "</tbody></table></div>")
        table = []

    for line in markdown.splitlines():
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line))
            continue
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if in_list and not line.startswith("- "):
            out.append("</ul>")
            in_list = False
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            if not inserted and insert_before and level == 2 and text.startswith(insert_before):
                out.append(charts_html)
                inserted = True
            out.append(f"<h{level}>{_inline(text)}</h{level}>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_inline(line[2:])}</li>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{_inline(line[2:])}</blockquote>")
        elif line.strip():
            out.append(f"<p>{_inline(line)}</p>")
    flush_table()
    if in_list:
        out.append("</ul>")
    if not inserted:
        out.append(charts_html)
    return "\n".join(out)


def html_page(title, body):
    """Wrap rendered content in a complete, self-contained HTML page with light/dark styling."""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --bg: #f4f4f2; --surface: #fcfcfb; --border: #e2e1dc; --grid: #e6e5e0; --text: #0b0b0b;
  --text-2: #52514e; --text-3: #7a7974; --s1: #2a78d6; --s2: #eb6834; --critical: #d03b3b; }}
@media (prefers-color-scheme: dark) {{ :root {{ color-scheme: dark; --bg: #121211; --surface: #1a1a19; --border: #2e2e2b;
  --grid: #2c2c29; --text: #f0efec; --text-2: #c3c2b7; --text-3: #8f8e86; --s1: #3987e5; --s2: #d95926; }} }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 22px; margin: 0 0 12px; }} h2 {{ font-size: 17px; margin: 28px 0 10px; }} h3 {{ font-size: 15px; }}
code {{ background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 0 4px; font-size: 12px; }}
pre {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; overflow: auto; font-size: 12px; }}
blockquote {{ margin: 8px 0; padding: 8px 12px; border-left: 4px solid var(--critical); background: var(--surface); }}
.table {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 13px; }}
th, td {{ border-bottom: 1px solid var(--grid); padding: 6px 8px; text-align: right; }} th {{ color: var(--text-2); }}
th:last-child, td:last-child {{ text-align: left; }}
.charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }}
.charts figure {{ margin: 0; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--surface); }}
@media (max-width: 520px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style></head><body><main>
{body}
</main></body></html>
"""
