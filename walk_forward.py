"""momentum, walk-forward: re-pick the settings every year from the past only, trade the next year. 1928 to 2026.

The main run (momentum.py) froze its settings on 1999-2015 and tested them once on 2016-2026.
One split can be lucky or unlucky. A walk-forward repeats the test every year:
  1. On 1 January, look back over the last TRAIN years only.
  2. Of the same 15 settings (6, 9 or 12 month lookback x hold the top 1 to 5 industries), keep the best Sharpe.
  3. Trade that setting for the next 12 months, with the same fees. Then repeat.
Every year of the result is out of sample: it was traded with settings chosen before it started.

Same engine (bt), data (10 Ken French industry portfolios, daily) and cost preset as momentum.py.
"""
import json
import sys
from pathlib import Path

import bt
import numpy as np
import pandas as pd

import charts
import data
from momentum import (CAPITAL, COSTS, LOOKBACKS, OUT, PUBLISHED, TEST, TOP_NS, benchmark, stats_daily,
                      strategy)

TRAIN = 10                      # years of history used to pick the settings each January
ALSO = [5, 15, "all"]           # other training windows, reported for robustness (not chosen from)
BLOCK = 20                      # bt run length in years (see settings_returns)
FIRST_TRAIN_YEAR = 1928         # every setting has a full 12-month lookback by then (data starts July 1926)


def settings_returns(d):
    """Daily returns, after costs, of all 15 settings plus the two benchmarks, 1926-2026.

    bt is run in 20-year blocks, each with a 2-year warm-up whose returns are dropped (one 100-year run trips
    bt's order-sizing loop once the account has compounded for decades). The warm-up means every block starts
    already invested, so the joined series is continuous; only the $1 commission is measured on a reset balance."""
    names = [c for c in d.columns if c not in ("Market", "RF")]
    d = d.loc[:TEST[1]]
    parts = []
    for start in range(1926, int(TEST[1][:4]) + 1, BLOCK):
        a, b = f"{max(1926, start - 2)}-01-01", f"{start + BLOCK - 1}-12-31"
        dd = d.loc[a:b]
        px = (1 + dd[names]).cumprod() * 100
        mkt_px = ((1 + dd["Market"]).cumprod() * 100).to_frame("Market")
        tests = [bt.Backtest(strategy(f"L{lb}_N{n}", n, lb), px, initial_capital=CAPITAL,
                             commissions=COSTS.fee, integer_positions=False)
                 for lb in LOOKBACKS for n in TOP_NS]
        tests.append(bt.Backtest(benchmark("Market"), mkt_px, initial_capital=CAPITAL, commissions=COSTS.fee,
                                 integer_positions=False))
        tests.append(bt.Backtest(benchmark("EqualWeight", monthly=True), px, initial_capital=CAPITAL,
                                 commissions=COSTS.fee, integer_positions=False))
        r = bt.run(*tests).prices.pct_change().iloc[1:]
        parts.append(r.loc[f"{start}-01-01":])
        print(f"      block {start}-{min(start + BLOCK - 1, int(TEST[1][:4]))} done", flush=True)
    return pd.concat(parts), d["RF"]


def sharpe(r, rf):
    ex = r - rf.reindex(r.index).fillna(0)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0


def walk(rets, rf, grid, train):
    """Chain one year at a time. Returns (daily out-of-sample returns, list of yearly picks)."""
    years = sorted(set(rets.index.year))
    first = FIRST_TRAIN_YEAR + (10 if train == "all" else train)
    picks, parts = [], []
    for y in [y for y in years if y >= first]:
        a = FIRST_TRAIN_YEAR if train == "all" else y - train
        past = rets.loc[f"{a}-01-01":f"{y - 1}-12-31"]
        scores = {k: sharpe(past[k], rf) for k in grid}
        k = max(scores, key=scores.get)
        nxt = rets.loc[f"{y}-01-01":f"{y}-12-31"]
        picks.append({"year": y, "pick": k, "train_sharpe": scores[k], "year_sharpe": sharpe(nxt[k], rf),
                      "year_return": float((1 + nxt[k]).prod() - 1),
                      "market_return": float((1 + nxt["Market"]).prod() - 1)})
        parts.append(nxt[k])
    return pd.concat(parts), picks


def compare(wf, rets, rf, a, b):
    w = wf.loc[a:b]
    m = rets["Market"].loc[w.index]
    e = rets["EqualWeight"].loc[w.index]
    return {"walk_forward": stats_daily(w, rf), "market": stats_daily(m, rf), "equal_weight": stats_daily(e, rf)}


def chart_equity(wf, mkt, ew, path, foot):
    fig, ax = charts.figure()
    fig.subplots_adjust(left=0.09, right=0.86, top=0.80, bottom=0.10)
    ax.axvspan(PUBLISHED, wf.index[-1], color=charts.PANEL, alpha=0.9, lw=0)
    curves = [((1 + ew).cumprod() * 1, "All 10 industries, equal weight", charts.MUTED, 1.8),
              ((1 + mkt).cumprod() * 1, "US stock market, buy and hold", charts.TEAL, 2.4),
              ((1 + wf).cumprod() * 1, "Momentum, re-picked every January, after costs", charts.YELLOW, 3.0)]
    pts = {}
    for i, (s, lab, col, lw) in enumerate(curves):
        ax.plot(s.index, s, color=col, lw=lw, label=lab)
        pts[f"end{i}"] = (s.index[-1], s.iloc[-1])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(charts.FuncFormatter(lambda v, _=None: f"${v:,.0f}"))
    ax.set_xlim(wf.index[0], wf.index[-1])
    hi = ax.get_ylim()[1]
    ax.axvline(PUBLISHED, color=charts.FG, lw=1.8, ls="--")
    ax.text(PUBLISHED, hi, "  the paper, 1993", color=charts.FG, fontsize=17, va="top")
    s = curves[2][0]
    pts["pub"] = (PUBLISHED, s.asof(PUBLISHED))
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, labelcolor=charts.FG, fontsize=17)
    last = -1e9
    fig.canvas.draw()
    for i in sorted(range(3), key=lambda i: curves[i][0].iloc[-1]):
        s, lab, col, lw = curves[i]
        y = ax.transData.transform((charts.mdates.date2num(s.index[-1]), s.iloc[-1]))[1]
        dy = max(0.0, last + 30 - y)
        last = y + dy
        v = s.iloc[-1]
        ax.annotate(f"${v:,.0f}", (s.index[-1], v), xytext=(10, dy * 72 / 120), textcoords="offset points",
                    color=col, fontsize=18, va="center", weight="bold")
    charts.titles(fig, f"$1 in momentum, re-picked every year, {wf.index[0]:%Y} to {wf.index[-1]:%Y}",
                  f"Each January: best of 15 settings over the past {TRAIN} years, then trade the next year. "
                  f"Costs: {COSTS.short()}.", foot)
    charts.save(fig, ax, path, pts)


def chart_bars(rows, path, foot):
    """rows: list of (label, walk-forward Sharpe, market Sharpe)."""
    fig, ax = charts.figure()
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.14)
    x = np.arange(len(rows))
    w = 0.36
    wf = [r[1] for r in rows]
    mk = [r[2] for r in rows]
    b1 = ax.bar(x - w / 2, wf, w, color=charts.YELLOW, label="Momentum, walk-forward, after costs")
    b2 = ax.bar(x + w / 2, mk, w, color=charts.TEAL, label="US stock market, buy and hold")
    pts = {}
    for i, (r1, r2) in enumerate(zip(b1, b2)):
        for j, (r, v) in enumerate(((r1, wf[i]), (r2, mk[i]))):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", color=charts.FG, fontsize=22,
                    ha="center", va="bottom", weight="bold")
            pts[f"{'wf' if j == 0 else 'mkt'}{i}"] = (r.get_x() + r.get_width() / 2, v / 2)
    ax.set_xticks(x, [r[0] for r in rows], color=charts.FG, fontsize=19)
    ax.set_ylabel("Sharpe ratio", color=charts.MUTED, fontsize=18)
    ax.set_ylim(0, max(wf + mk) * 1.3)
    ax.legend(loc="upper right", frameon=False, labelcolor=charts.FG, fontsize=17)
    charts.titles(fig, "Walk-forward: before and after the paper",
                  "Sharpe ratio. Every year traded with settings picked only from the years before it.", foot)
    charts.save(fig, ax, path, pts)


def chart_picks(picks, path, foot):
    fig, ax = charts.figure()
    fig.subplots_adjust(left=0.10, right=0.97, top=0.78, bottom=0.12)
    yrs = [p["year"] for p in picks]
    lb = [int(p["pick"].split("_")[0][1:]) for p in picks]
    n = [int(p["pick"].split("_")[1][1:]) for p in picks]
    ax.step(yrs, lb, where="post", color=charts.YELLOW, lw=3, label="lookback, months")
    ax.step(yrs, n, where="post", color=charts.TEAL, lw=3, label="industries held")
    ax.axvline(PUBLISHED.year, color=charts.FG, lw=1.8, ls="--")
    ax.set_ylim(0, 13.5)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 9, 12])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, labelcolor=charts.FG, fontsize=17, ncol=2)
    changes = sum(1 for a, b in zip(picks, picks[1:]) if a["pick"] != b["pick"])
    charts.titles(fig, "The settings it picked, year by year",
                  f"Best of 15 settings over the past {TRAIN} years. The pick changed {changes} times in {len(picks)} years.",
                  foot)
    charts.save(fig, ax, path, {"pub": (PUBLISHED.year, 12)})
    return changes


def main():
    OUT.mkdir(exist_ok=True)
    vintage = data.fetch()
    foot = f"Data: {vintage}. Code: github.com/WallStVerdict/momentum-after-the-paper (MIT). Education, not advice."
    print(f"walk-forward | {vintage}")
    print(f"[1/2] backtest all 15 settings, 1926-{TEST[1][:4]}, costs {COSTS.short()} ...", flush=True)
    rets, rf = settings_returns(data.daily_industries())
    grid = [f"L{lb}_N{n}" for lb in LOOKBACKS for n in TOP_NS]

    print(f"[2/2] walk forward: each January pick the best of 15 on the past {TRAIN} years, trade the next")
    runs = {}
    for train in [TRAIN] + ALSO:
        wf, picks = walk(rets, rf, grid, train)
        end = wf.index[-1]
        runs[str(train)] = {
            "picks": picks,
            "full": compare(wf, rets, rf, wf.index[0], end),
            "before_paper": compare(wf, rets, rf, wf.index[0], "1993-12-31"),
            "after_paper": compare(wf, rets, rf, "1994-01-01", end),
            "last_10y": compare(wf, rets, rf, "2016-01-01", end),
            "years_beating_market": sum(p["year_return"] > p["market_return"] for p in picks),
            "years": len(picks),
        }
        if train == TRAIN:
            main_wf, main_picks = wf, picks

    r = runs[str(TRAIN)]
    changes = chart_picks(main_picks, OUT / "wf_picks.png", foot)
    r["pick_changes"] = changes
    chart_equity(main_wf, rets["Market"].loc[main_wf.index], rets["EqualWeight"].loc[main_wf.index],
                 OUT / "wf_equity.png", foot)
    lab = {"full": f"All years\n{main_wf.index[0]:%Y}-{main_wf.index[-1]:%Y}",
           "before_paper": f"Before the paper\n{main_wf.index[0]:%Y}-1993",
           "after_paper": f"After the paper\n1994-{main_wf.index[-1]:%Y}", "last_10y": f"Last 10 years\n2016-{main_wf.index[-1]:%Y}"}
    chart_bars([(lab[k], r[k]["walk_forward"]["sharpe"], r[k]["market"]["sharpe"])
                for k in ("before_paper", "after_paper", "last_10y")], OUT / "wf_sharpe_bars.png", foot)

    L = [f"momentum walk-forward | {vintage}",
         f"Each January: best of {len(grid)} settings by Sharpe over the past N years, traded the next 12 months.",
         f"Costs: {COSTS.describe()}. Sharpe on returns above T-bills.", "",
         f"{'train':<7}{'period':<26}{'WF Sharpe':>10}{'mkt':>7}{'EW':>7}{'WF CAGR':>9}{'mkt CAGR':>10}{'WF maxDD':>10}"]
    for k, run in runs.items():
        for per in ("full", "before_paper", "after_paper", "last_10y"):
            c = run[per]
            w, m, e = c["walk_forward"], c["market"], c["equal_weight"]
            L.append(f"{k:<7}{per + ' ' + w['start'][:4] + '-' + w['end'][:4]:<26}{w['sharpe']:>10.2f}{m['sharpe']:>7.2f}"
                     f"{e['sharpe']:>7.2f}{w['cagr'] * 100:>8.1f}%{m['cagr'] * 100:>9.1f}%{w['max_drawdown'] * 100:>9.0f}%")
        L.append(f"{'':<7}years beating the market: {run['years_beating_market']} of {run['years']}")
    L += ["", f"Main run (train {TRAIN}y): the pick changed {changes} times in {len(main_picks)} years.",
          "Picks (year: setting, Sharpe on the training years -> Sharpe that year):"]
    for p in main_picks:
        L.append(f"  {p['year']}: {p['pick']:<7} {p['train_sharpe']:5.2f} -> {p['year_sharpe']:5.2f}")
    L.append("Education, not advice.")
    (OUT / "walk_forward.txt").write_text("\n".join(L) + "\n")
    flat = {"train_years": TRAIN, "wf_start": r["full"]["walk_forward"]["start"][:4],
            "wf_full_sharpe": r["full"]["walk_forward"]["sharpe"], "mkt_full_sharpe": r["full"]["market"]["sharpe"],
            "wf_pre_sharpe": r["before_paper"]["walk_forward"]["sharpe"], "mkt_pre_sharpe": r["before_paper"]["market"]["sharpe"],
            "wf_post_sharpe": r["after_paper"]["walk_forward"]["sharpe"], "mkt_post_sharpe_daily": r["after_paper"]["market"]["sharpe"],
            "wf_last10_sharpe": r["last_10y"]["walk_forward"]["sharpe"], "mkt_last10_sharpe": r["last_10y"]["market"]["sharpe"],
            "wf_pre_cagr_pct": r["before_paper"]["walk_forward"]["cagr"] * 100, "mkt_pre_cagr_pct": r["before_paper"]["market"]["cagr"] * 100,
            "wf_post_cagr_pct": r["after_paper"]["walk_forward"]["cagr"] * 100, "mkt_post_cagr_pct": r["after_paper"]["market"]["cagr"] * 100,
            "wf_years_beating": r["years_beating_market"], "wf_years": r["years"], "wf_pick_changes": changes}
    (OUT / "walk_forward.json").write_text(json.dumps({**flat, "runs": runs}, indent=1, default=float) + "\n")
    p = OUT / "summary.json"                 # add the headline numbers to momentum.py's summary, so the video reads one file
    base = json.loads(p.read_text()) if p.exists() else {}
    base.update(flat)
    p.write_text(json.dumps(base, indent=1, default=float) + "\n")
    for per in ("before_paper", "after_paper", "last_10y"):
        c = r[per]
        print(f"      {per:<13} walk-forward Sharpe {c['walk_forward']['sharpe']:.2f} vs market {c['market']['sharpe']:.2f}")
    print(f"      years beating the market: {r['years_beating_market']} of {r['years']} | pick changed {changes} times")
    print("results/  walk_forward.txt  walk_forward.json  wf_equity.png  wf_sharpe_bars.png  wf_picks.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
