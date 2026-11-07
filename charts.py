"""House charts: midnight background, yellow and teal, coral only for the bad news. 16:9 PNGs at 1920x1080.

Every chart also writes <name>_anchors.json: pixel points (origin top-left) the screencast tool can zoom to.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

BG, PANEL, FG, MUTED, GRID = "#080d18", "#122036", "#f4f6fa", "#a0acc0", "#2a3a55"
YELLOW, TEAL, CORAL = "#ffcc33", "#4fd2c2", "#ff8a65"

HERE = Path(__file__).resolve().parent
BODY, TITLE = "DejaVu Sans", "DejaVu Sans"
for f in sorted((HERE / "fonts").glob("Inter-*.ttf")):
    font_manager.fontManager.addfont(str(f))
    BODY = TITLE = "Inter"
for p in [HERE / "fonts/Anton-Regular.ttf", Path.home() / ".local/share/fonts/Anton-Regular.ttf",
          Path("/usr/share/fonts/truetype/anton/Anton-Regular.ttf")]:
    if p.exists():
        font_manager.fontManager.addfont(str(p))
        TITLE = "Anton"
        break
plt.rcParams.update({"font.family": BODY, "axes.unicode_minus": False})


def figure():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120, facecolor=BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=17)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def titles(fig, title, sub=None, foot=None):
    fig.text(0.035, 0.945, title, color=FG, fontsize=36, family=TITLE, va="top")
    if sub:
        fig.text(0.035, 0.858, sub, color=MUTED, fontsize=19, va="top")
    if foot:
        fig.text(0.035, 0.025, foot, color=MUTED, fontsize=13, va="bottom")


def save(fig, ax, path, points, texts=None):
    """points: {key: (x, y) in data coords of ax}; texts: {key: matplotlib Text} (anchor = its centre)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    an = {"size": [w, h]}
    for k, t in (texts or {}).items():
        bb = t.get_window_extent()
        an[k] = [round((bb.x0 + bb.x1) / 2), round(h - (bb.y0 + bb.y1) / 2)]
    for k, (x, y) in points.items():
        X, Y = ax.transData.transform((mdates.date2num(x) if hasattr(x, "year") else x, y))
        an[k] = [round(float(X)), round(float(h - Y))]
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    Path(path).with_name(Path(path).stem + "_anchors.json").write_text(json.dumps(an))


def pct(x, _=None):
    return f"{x:.0%}"


def dollars(x, _=None):
    return f"${x:,.0f}"


# ---------------------------------------------------------------- part A
def umd_century(growth, mkt_growth, pub, crashes, path, foot):
    fig, ax = figure()
    fig.subplots_adjust(left=0.08, right=0.91, top=0.80, bottom=0.10)
    ax.axvspan(pub, growth.index[-1], color=PANEL, alpha=0.9, lw=0)
    ax.plot(mkt_growth.index, mkt_growth, color=TEAL, lw=2.2, alpha=0.9, label="Stock market minus T-bills")
    ax.plot(growth.index, growth, color=YELLOW, lw=3.0, label="Momentum factor (UMD)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}" if v >= 1 else f"${v:.2f}"))
    ax.axvline(pub, color=FG, lw=2, ls="--")
    top = max(growth.max(), mkt_growth.max())
    ax.text(pub, top * 1.25, "  Jegadeesh & Titman publish, 1993", color=FG, fontsize=19, va="bottom")
    ax.set_ylim(min(growth.min(), mkt_growth.min()) * 0.6, top * 4)
    ax.set_xlim(growth.index[0], growth.index[-1])
    end = growth.index[-1]
    hi_up = growth.iloc[-1] >= mkt_growth.iloc[-1]
    ax.annotate(f"${growth.iloc[-1]:,.0f}", (end, growth.iloc[-1]), xytext=(10, 10 if hi_up else -10),
                textcoords="offset points", color=YELLOW, fontsize=20, ha="left", va="center", weight="bold")
    ax.annotate(f"${mkt_growth.iloc[-1]:,.0f}", (end, mkt_growth.iloc[-1]), xytext=(10, -10 if hi_up else 10),
                textcoords="offset points", color=TEAL, fontsize=20, ha="left", va="center", weight="bold")
    pts = {"pub": (pub, growth.asof(pub)), "end": (end, growth.iloc[-1])}
    for key, (d, label) in crashes.items():
        ax.annotate(label, (d, growth.asof(d)), xytext=(0, -60), textcoords="offset points", color=CORAL,
                    fontsize=17, ha="center", arrowprops=dict(arrowstyle="-", color=CORAL, lw=1.5))
        pts[key] = (d, growth.asof(d))
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=18)
    titles(fig, "$1 in momentum, 1927 to 2026", "Ken French UMD: buy past winners, sell past losers, monthly. Log scale, before costs.", foot)
    save(fig, ax, path, pts)


def sharpe_bars(rows, path, foot):
    """rows: list of (label, umd_sharpe, umd_t, mkt_sharpe, mkt_t)."""
    import numpy as np
    fig, ax = figure()
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.19)
    x = np.arange(len(rows))
    w = 0.36
    pts = {}
    for i, (lab, us, ut, ms, mt) in enumerate(rows):
        b1 = ax.bar(i - w / 2, us, w, color=YELLOW, label="Momentum (UMD)" if i == 0 else None)
        ax.bar(i + w / 2, ms, w, color=TEAL, label="Stock market" if i == 0 else None)
        ax.text(i - w / 2, us + 0.02, f"{us:.2f}", ha="center", va="bottom", color=FG, fontsize=22, weight="bold")
        ax.text(i + w / 2, ms + 0.02, f"{ms:.2f}", ha="center", va="bottom", color=FG, fontsize=22, weight="bold")
        tcol = CORAL if ut < 2 else MUTED
        ax.text(i - w / 2, -0.035, f"t {ut:.1f}", ha="center", va="top", color=tcol, fontsize=16, weight="bold" if ut < 2 else None)
        ax.text(i + w / 2, -0.035, f"t {mt:.1f}", ha="center", va="top", color=MUTED, fontsize=16)
        pts[f"umd{i}"] = (i - w / 2, us)
        pts[f"mkt{i}"] = (i + w / 2, ms)
        del b1
    ax.set_xticks(x, [r[0] for r in rows], color=FG, fontsize=17)
    ax.tick_params(axis="x", length=0, pad=30)
    ax.axhline(0, color=GRID, lw=1.5)
    ax.set_ylim(-0.12, max(max(r[1], r[3]) for r in rows) * 1.28)
    ax.set_ylabel("Sharpe ratio (annualised)", color=MUTED, fontsize=18)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, labelcolor=FG, fontsize=18, ncol=2)
    titles(fig, "Momentum's Sharpe ratio, before and after the paper",
           "t below each bar: under 2 (coral) means the period alone can't tell the edge from luck", foot)
    save(fig, ax, path, pts)


def underwater(dd, crashes, path, title, sub, foot, color=CORAL, other=None, other_label=None, own_label=None, split=None):
    fig, ax = figure()
    fig.subplots_adjust(left=0.08, right=0.96, top=0.80, bottom=0.10)
    if other is not None:
        ax.plot(other.index, other, color=TEAL, lw=1.8, label=other_label)
    ax.fill_between(dd.index, dd, 0, color=color, alpha=0.55, lw=0, label=own_label)
    ax.plot(dd.index, dd, color=color, lw=1.4)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.set_ylim(min(dd.min(), other.min() if other is not None else 0) * 1.18, 0.03)
    ax.set_xlim(dd.index[0], dd.index[-1])
    pts = {}
    span = (dd.index[-1] - dd.index[0]).days
    for key, (d, label) in crashes.items():
        right = (d - dd.index[0]).days > 0.7 * span
        ax.annotate(label, (d, dd.loc[d]), xytext=(-18 if right else 18, -6), textcoords="offset points", color=FG,
                    fontsize=18, va="top", ha="right" if right else "left")
        pts[key] = (d, dd.loc[d])
    if split is not None:
        ax.axvline(split, color=FG, lw=1.8, ls="--")
        ax.text(split, 0.015, "  test", color=FG, fontsize=17, va="bottom")
        ax.text(split, 0.015, "design  ", color=MUTED, fontsize=17, va="bottom", ha="right")
    if own_label:
        ax.legend(loc="lower left", frameon=False, labelcolor=FG, fontsize=17)
    titles(fig, title, sub, foot)
    save(fig, ax, path, pts)


# ---------------------------------------------------------------- part B
def equity(curves, split, path, title, sub, foot):
    """curves: list of (series, label, color, lw). split: date of the design/test line, or None."""
    fig, ax = figure()
    fig.subplots_adjust(left=0.09, right=0.88, top=0.80, bottom=0.10)
    pts = {}
    for i, (s, lab, col, lw) in enumerate(curves):
        ax.plot(s.index, s, color=col, lw=lw, label=lab)
        pts[f"end{i}"] = (s.index[-1], s.iloc[-1])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    money = lambda v, _=None: f"${v / 1e6:,.2f}M" if v >= 1e6 else f"${v / 1000:,.0f}k"  # noqa: E731
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_xlim(curves[0][0].index[0], curves[0][0].index[-1])
    if split is not None:
        ax.axvline(split, color=FG, lw=1.8, ls="--")
        hi = ax.get_ylim()[1]
        ax.text(split, hi, "  test: never seen while choosing", color=FG, fontsize=17, va="top")
        ax.text(split, hi, "design: settings chosen here  ", color=MUTED, fontsize=17, va="top", ha="right")
        pts["split"] = (split, curves[0][0].asof(split))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, labelcolor=FG, fontsize=17)
    fig.canvas.draw()
    last_y = -1e9
    for i in sorted(range(len(curves)), key=lambda i: curves[i][0].iloc[-1]):   # end labels, nudged apart
        s, lab, col, lw = curves[i]
        y = ax.transData.transform((mdates.date2num(s.index[-1]), s.iloc[-1]))[1]
        dy = max(0.0, last_y + 30 - y)
        last_y = y + dy
        ax.annotate(money(s.iloc[-1]), (s.index[-1], s.iloc[-1]), xytext=(10, dy * 72 / 120),
                    textcoords="offset points", color=col, fontsize=18, va="center", weight="bold")
    titles(fig, title, sub, foot)
    save(fig, ax, path, pts)


def holdings(held, split, path, title, sub, foot):
    """held: DataFrame month x industry of 0/1."""
    import numpy as np
    fig, ax = figure()
    fig.subplots_adjust(left=0.15, right=0.97, top=0.80, bottom=0.10)
    ax.grid(False)
    cols = list(held.columns)
    for j, c in enumerate(cols):
        on = held[c].values > 0
        xs = mdates.date2num(held.index)
        width = np.diff(np.append(xs, xs[-1] + 31))
        test = held.index >= split
        ax.bar(xs[on & ~test], 0.72, width=width[on & ~test], bottom=j - 0.36, align="edge", color=TEAL, alpha=0.75, lw=0)
        ax.bar(xs[on & test], 0.72, width=width[on & test], bottom=j - 0.36, align="edge", color=YELLOW, lw=0)
    ax.set_yticks(range(len(cols)), cols, color=FG, fontsize=17)
    ax.set_ylim(-0.7, len(cols) - 0.3)
    ax.invert_yaxis()
    ax.xaxis_date()
    ax.set_xlim(held.index[0], held.index[-1])
    ax.axvline(split, color=FG, lw=1.8, ls="--")
    titles(fig, title, sub, foot)
    save(fig, ax, path, {"split": (split, 0)})


def trade_table(rows, totals, path, title, sub, foot):
    """rows: list of dicts date, side, name, price, value, fee."""
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cols = [(0.06, "Date", "left"), (0.22, "Order", "left"), (0.33, "Industry", "left"),
            (0.62, "Order value", "right"), (0.78, "Fee", "right"), (0.94, "Fee %", "right")]
    y0, dy = 0.745, 0.043
    for x, lab, ha in cols:
        ax.text(x, y0, lab, color=MUTED, fontsize=17, ha=ha, va="center", weight="bold")
    ax.plot([0.05, 0.95], [y0 - dy / 2, y0 - dy / 2], color=GRID, lw=1.5)
    for i, r in enumerate(rows):
        y = y0 - (i + 1) * dy
        col = TEAL if r["side"] == "BUY" else YELLOW
        vals = [r["date"], r["side"], r["name"], f"${r['value']:,.0f}", f"${r['fee']:,.2f}", f"{r['fee'] / r['value']:.2%}"]
        for (x, _, ha), v, c in zip(cols, vals, [FG, col, FG, FG, FG, MUTED]):
            ax.text(x, y, v, color=c, fontsize=17, ha=ha, va="center", family="DejaVu Sans Mono" if ha == "right" else None)
    yb = y0 - (len(rows) + 1) * dy - 0.02
    ax.plot([0.05, 0.95], [yb + dy / 2, yb + dy / 2], color=GRID, lw=1.5)
    t = ax.text(0.06, yb - 0.01, totals, color=YELLOW, fontsize=20, ha="left", va="center", weight="bold")
    fig.text(0.035, 0.945, title, color=FG, fontsize=36, family=TITLE, va="top")
    fig.text(0.035, 0.858, sub, color=MUTED, fontsize=19, va="top")
    fig.text(0.035, 0.025, foot, color=MUTED, fontsize=13, va="bottom")
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    bb = t.get_window_extent()
    an = {"size": [w, h], "totals": [round((bb.x0 + bb.x1) / 2), round(h - (bb.y0 + bb.y1) / 2)],
          "fee_col": [round(0.78 * w - 40), round(h - (y0 - dy * (len(rows) / 2 + 0.5)) * h)]}
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    Path(path).with_name(Path(path).stem + "_anchors.json").write_text(json.dumps(an))


def trade_chart(price, spans, entries, exits, fees_txt, path, title, sub, foot):
    """price: daily index; spans: list of (start, end) held; entries/exits: lists of dates."""
    fig, ax = figure()
    fig.subplots_adjust(left=0.08, right=0.96, top=0.80, bottom=0.10)
    for a, b in spans:
        ax.axvspan(a, b, color=TEAL, alpha=0.13, lw=0)
    ax.plot(price.index, price, color=FG, lw=1.6)
    ax.plot(entries, price.reindex(entries), "^", color=TEAL, ms=17, mec=BG, mew=1.5, label="buy (enters the top list)")
    ax.plot(exits, price.reindex(exits), "v", color=YELLOW, ms=17, mec=BG, mew=1.5, label="sell (drops out)")
    ax.set_xlim(price.index[0], price.index[-1])
    ax.legend(loc="upper left", frameon=False, labelcolor=FG, fontsize=17)
    box = ax.text(0.99, 0.04, fees_txt, transform=ax.transAxes, color=YELLOW, fontsize=19, ha="right", va="bottom", weight="bold",
            bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=GRID))
    pts = {}
    if entries:
        pts["entry"] = (entries[0], price.loc[entries[0]])
        pts["entry_last"] = (entries[-1], price.loc[entries[-1]])
        mid = entries[len(entries) // 2]
        pts["entry_mid"] = (mid, price.loc[mid])
    if exits:
        pts["exit"] = (exits[0], price.loc[exits[0]])
    titles(fig, title, sub, foot)
    save(fig, ax, path, pts, texts={"fees": box})
