#!/usr/bin/env python3
"""charts.py — render the benchmark into SVGs for the README, with no dependencies at all.

Numbers come from docs/results.json, never from this file, so the picture cannot drift away from the
measurement. Hand-written SVG rather than a plotting library: this repo's whole claim is that it needs
nothing but numpy and sqlite, and a chart is not a good reason to break that.

Each bar carries its own count and percentage, so the images read correctly in grey, in colour-blind
palettes, and in a screen reader's alt text.

    python scripts/charts.py            # writes docs/img/*.svg
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


def rejected_svg(rows):
    """Signed bars around a zero line: what each idea did to the number of questions answered first."""
    w, label_w, pad = 760, 300, 18
    row_h, top = 40, 88
    vals = [r["effect"] for r in rows]
    lo, hi = min(vals + [0]), max(vals + [0])
    span = max(abs(lo), abs(hi), 1)
    plot_w = w - label_w - pad - 30
    mid = label_w + plot_w / 2.0
    # Reserve room for the value label on the outward side of the longest bar, otherwise "-7 questions"
    # runs back over the row label and gets visually clipped.
    reserve = 96
    scale = max(plot_w / 2.0 - reserve, 10) / span
    h = top + row_h * len(rows) + 44

    out = ["<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d' viewBox='0 0 %d %d' "
           "role='img' aria-label='Ideas that sounded good and were measured out'>" % (w, h, w, h),
           "<rect width='%d' height='%d' fill='white'/>" % (w, h),
           "<text x='%d' y='30' %s font-size='17' font-weight='600' fill='%s'>"
           "Ideas that sounded good and were measured out</text>" % (pad, FONT, INK),
           "<text x='%d' y='52' %s font-size='12' fill='%s'>"
           "Each one was built and run against the same questions, then dropped when it did not help</text>"
           % (pad, FONT, MUTED),
           "<text x='%d' y='72' %s font-size='11' fill='%s'>bars to the left mean fewer questions answered "
           "first; to the right, more</text>" % (pad, FONT, MUTED)]

    out.append("<line x1='%.1f' y1='%d' x2='%.1f' y2='%d' stroke='%s' stroke-width='1.5'/>"
               % (mid, top - 4, mid, top + row_h * len(rows), INK))

    for i, r in enumerate(rows):
        y = top + i * row_h
        v = r["effect"]
        colour = POS if v > 0 else (NEG if v < 0 else MUTED)
        out.append("<text x='%d' y='%d' %s font-size='12.5' fill='%s'>%s</text>"
                   % (pad, y + 20, FONT, INK, esc(r["change"])))
        bw = abs(v) * scale
        if v >= 0:
            out.append("<rect x='%.1f' y='%d' width='%.1f' height='18' rx='3' fill='%s'/>"
                       % (mid, y + 6, max(bw, 2), colour))
            tx, anchor = mid + bw + 8, "start"
        else:
            out.append("<rect x='%.1f' y='%d' width='%.1f' height='18' rx='3' fill='%s'/>"
                       % (mid - bw, y + 6, max(bw, 2), colour))
            tx, anchor = mid - bw - 8, "end"
        label = ("%+d question%s" % (v, "" if abs(v) == 1 else "s")) if v else "no change"
        # never let the label start left of the label column
        if anchor == "end" and tx - 88 < label_w:
            tx, anchor = label_w + 6, "start"
        out.append("<text x='%.1f' y='%d' %s font-size='12' fill='%s' text-anchor='%s'>%s</text>"
                   % (tx, y + 20, FONT, INK, anchor, label))

    out.append("</svg>")
    return "\n".join(out)


held, dev = DATA["held_out"], DATA["dev"]
open(os.path.join(OUT, "held-out.svg"), "w").write(stage_svg(
    held, "Questions kept back from tuning",
    "These %d questions were never used while building, so this is the honest picture" % held[0]["of"]))
open(os.path.join(OUT, "dev.svg"), "w").write(stage_svg(
    dev, "The other %d questions, used while building" % dev[0]["of"],
    "Same three stages on the same corpus — the two pictures agree, which is the point"))
open(os.path.join(OUT, "rejected.svg"), "w").write(rejected_svg(DATA["rejected"]))

for f in sorted(os.listdir(OUT)):
    print("%-14s %6d bytes" % (f, os.path.getsize(os.path.join(OUT, f))))
