#!/usr/bin/env python3
"""charts.py — render the one benchmark figure the README shows, with no dependencies at all.

Numbers come from docs/results.json, never from this file, so the picture cannot drift away from the
measurement. Hand-written SVG rather than a plotting library: this repo's whole claim is that it needs
nothing but numpy and sqlite, and a chart is not a good reason to break that.

Each bar carries its own count and percentage, so the images read correctly in grey, in colour-blind
palettes, and in a screen reader's alt text.

    python scripts/charts.py            # writes docs/img/held-out.svg
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = json.load(open(os.path.join(HERE, "docs", "results.json")))
OUT = os.path.join(HERE, "docs", "img")
os.makedirs(OUT, exist_ok=True)

INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
BAR = "#4a6fa5"
BAR_REF = "#9aa5b1"
POS = "#2f7d4f"
NEG = "#a4442f"

FONT = "font-family='-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif'"


def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def stage_svg(rows, title, subtitle):
    """Horizontal bars, 0-100%, with a grey bar for the ceiling so it is never read as a result."""
    w, label_w, pad = 760, 268, 18
    row_h, top = 46, 76
    plot_w = w - label_w - pad - 96
    h = top + row_h * len(rows) + 46
    out = ["<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' viewBox='0 0 %d %d' "
           "role='img' aria-label='%s'>" % (w, h, w, h, esc(title)),
           "<rect width='%d' height='%d' fill='white'/>" % (w, h),
           "<text x='%d' y='30' %s font-size='17' font-weight='600' fill='%s'>%s</text>"
           % (pad, FONT, INK, esc(title)),
           "<text x='%d' y='52' %s font-size='12' fill='%s'>%s</text>" % (pad, FONT, MUTED, esc(subtitle))]

    for pct in (0, 25, 50, 75, 100):
        x = label_w + plot_w * pct / 100.0
        out.append("<line x1='%.1f' y1='%d' x2='%.1f' y2='%d' stroke='%s' stroke-width='1'/>"
                   % (x, top - 6, x, top + row_h * len(rows), GRID))
        out.append("<text x='%.1f' y='%d' %s font-size='11' fill='%s' text-anchor='middle'>%d%%</text>"
                   % (x, top + row_h * len(rows) + 18, FONT, MUTED, pct))

    for i, r in enumerate(rows):
        y = top + i * row_h
        pct = 100.0 * r["correct_first"] / r["of"]
        colour = BAR_REF if i == len(rows) - 1 else BAR
        bw = max(plot_w * pct / 100.0, 2)
        out.append("<text x='%d' y='%d' %s font-size='13' fill='%s'>%s</text>"
                   % (pad, y + 18, FONT, INK, esc(r["stage"])))
        if r.get("note"):
            out.append("<text x='%d' y='%d' %s font-size='10.5' fill='%s'>%s</text>"
                       % (pad, y + 33, FONT, MUTED, esc(r["note"])))
        out.append("<rect x='%d' y='%d' width='%.1f' height='22' rx='3' fill='%s'/>"
                   % (label_w, y + 6, bw, colour))
        out.append("<text x='%.1f' y='%d' %s font-size='12.5' font-weight='500' fill='%s'>%d of %d (%d%%)</text>"
                   % (label_w + bw + 8, y + 21, FONT, INK, r["correct_first"], r["of"], round(pct)))

    out.append("</svg>")
    return "\n".join(out)


def shipped(rows):
    """The README shows what the kit does and the ceiling it is measured against — nothing that needs an
    extra install, so the chart cannot read as a menu of upgrades."""
    return [r for r in rows if not r.get("note", "").startswith("optional")]


held = shipped(DATA["held_out"])
open(os.path.join(OUT, "held-out.svg"), "w").write(stage_svg(
    held, "Questions kept back from tuning",
    "These %d questions were never used while building, so this is the honest picture" % held[0]["of"]))

for f in ("held-out.svg",):
    print("%-14s %6d bytes" % (f, os.path.getsize(os.path.join(OUT, f))))
