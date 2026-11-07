"""momentum: 100 years of edge. Then the paper came out.

A. The factor as published: Ken French's momentum factor (UMD), monthly, 1927 to 2026.
   Sharpe and t-stat before and after Jegadeesh & Titman (1993), the crashes, and a deflated Sharpe check.
B. Do it yourself: each month hold the top N of 10 industry portfolios by 12-1 month return (bt engine),
   with fees and slippage. Settings chosen on 1999-2015 only, then tested on 2016-2026.
C. A check against rules fixed before the run: HOLDS UP / PARTLY / DOESN'T. The README gives the overall verdict, walk-forward included.

Open data only (Ken French data library), cached in data/ and checksummed in data.sha256.
"""
import json
import sys
from pathlib import Path
from statistics import NormalDist

import bt
import numpy as np
import pandas as pd

import charts
import data
from costs import PRESETS

OUT = Path("results")
COSTS = PRESETS["etf_retail"]       # one preset for every order, printed on screen
CAPITAL = 100_000
DESIGN = ("1999-01-01", "2015-12-31")   # settings are chosen here ...
TEST = ("2016-01-01", "2026-09-30")     # ... and only then tested here
LOOKBACKS = [6, 9, 12]                  # months of past return, skipping the latest month
TOP_NS = [1, 2, 3, 4, 5]                # how many industries to hold
PUBLISHED = pd.Timestamp("1993-03-31")  # Journal of Finance, vol. 48, no. 1
Z = NormalDist()
EULER = 0.5772156649015329


# ---------------------------------------------------------------- the strategy
def momentum_score(month_end_prices, lookback=12):
    """12-1 momentum: the return over the past `lookback` months, skipping the most recent month."""
    return month_end_prices.shift(1) / month_end_prices.shift(lookback) - 1


class BuyWinners(bt.Algo):
    """On the first trading day of each month, select the top_n industries by 12-1 momentum."""

    def __init__(self, top_n, lookback):
        super().__init__()
        self.top_n, self.lookback = top_n, lookback

    def __call__(self, target):
        month_ends = target.universe.resample("ME").last().iloc[:-1]   # finished months only: no look-ahead
        if len(month_ends) < self.lookback:
            return False                                                # not enough history yet
        score = momentum_score(month_ends, self.lookback).iloc[-1].dropna()
        winners = score.nlargest(self.top_n).index                     # rank them, keep the top N winners
        target.temp["selected"] = list(winners)
        return len(winners) > 0


def strategy(name, top_n, lookback):
    return bt.Strategy(name, [bt.algos.RunMonthly(), BuyWinners(top_n, lookback),
                              bt.algos.WeighEqually(), bt.algos.Rebalance()])


def benchmark(name, monthly=False):
    when = bt.algos.RunMonthly() if monthly else bt.algos.RunOnce()
    return bt.Strategy(name, [when, bt.algos.SelectAll(), bt.algos.WeighEqually(), bt.algos.Rebalance()])


# ---------------------------------------------------------------- statistics
def stats_monthly(r):
    """r: monthly excess (or long-short) returns."""
    n = len(r)
    growth = (1 + r).cumprod()
    return {"start": r.index[0].strftime("%Y-%m"), "end": r.index[-1].strftime("%Y-%m"), "months": n,
            "mean_ann": float(r.mean() * 12), "cagr": float(growth.iloc[-1] ** (12 / n) - 1),
            "vol_ann": float(r.std(ddof=1) * np.sqrt(12)),
            "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(12)),
            "t": float(r.mean() / r.std(ddof=1) * np.sqrt(n)),
            "worst_month": float(r.min()), "worst_month_date": r.idxmin().strftime("%b %Y"),
            "max_drawdown": float((growth / growth.cummax() - 1).min())}


def stats_daily(r, rf):
    """r: daily strategy returns; rf: daily T-bill. Sharpe on excess returns."""
    ex = r - rf.reindex(r.index).fillna(0)
    years = len(r) / 252
    growth = (1 + r).cumprod()
    sr = float(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
    return {"start": r.index[0].strftime("%Y-%m-%d"), "end": r.index[-1].strftime("%Y-%m-%d"),
            "cagr": float(growth.iloc[-1] ** (1 / years) - 1), "vol_ann": float(r.std(ddof=1) * np.sqrt(252)),
            "sharpe": sr, "t": sr * np.sqrt(years), "max_drawdown": float((growth / growth.cummax() - 1).min())}


def moments(r):
    z = (r - r.mean()) / r.std(ddof=0)
    return float((z ** 3).mean()), float((z ** 4).mean())


def prob_sharpe(sr, sr0, t, skew, kurt):
    """Probabilistic Sharpe Ratio (per-period Sharpe sr vs benchmark sr0), Bailey & Lopez de Prado."""
    z = (sr - sr0) * np.sqrt(t - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return Z.cdf(float(z))


def expected_max_sharpe(sr_std, n):
    """E[max Sharpe] of n independent tries with no skill (Bailey & Lopez de Prado 2014)."""
    if n <= 1:
        return 0.0
    return sr_std * ((1 - EULER) * Z.inv_cdf(1 - 1 / n) + EULER * Z.inv_cdf(1 - 1 / (n * np.e)))


def deflated(r, n_tries, sr_std=None):
    """Deflated Sharpe of per-period returns r, if it was the best of n_tries. sr_std defaults to 1/sqrt(T-1)."""
    sr = r.mean() / r.std(ddof=1)
    skew, kurt = moments(r)
    sd = sr_std if sr_std is not None else 1 / np.sqrt(len(r) - 1)
    return prob_sharpe(sr, expected_max_sharpe(sd, n_tries), len(r), skew, kurt)


def breakeven_tries(r):
    """Largest number of tries this result could be the best of and still pass DSR >= 0.95."""
    lo, hi = 1, 10 ** 12
    if deflated(r, 1) < 0.95:
        return 0
    while hi - lo > 1:
        mid = int(np.sqrt(lo * hi)) if hi > 100 * lo else (lo + hi) // 2
        lo, hi = (mid, hi) if deflated(r, mid) >= 0.95 else (lo, mid)
    return lo


def window(s, a, b):
    return s.loc[pd.Timestamp(a):pd.Timestamp(b)]


# ---------------------------------------------------------------- part A
def part_a(m):
    umd, mkt = m["UMD"], m["Mkt-RF"]
    end = umd.index[-1]
    last10 = end - pd.DateOffset(months=119) + pd.offsets.MonthEnd(0)
    periods = [("Full sample", umd.index[0], end),
               ("J&T's own sample", "1965-01-31", "1989-12-31"),
               ("Before publication", umd.index[0], "1993-12-31"),
               ("After publication", "1994-01-31", end),
               ("Last 10 years", last10, end)]
    rows = []
    for label, a, b in periods:
        u, k = window(umd, a, b), window(mkt, a, b)
        rows.append({"period": label, "umd": stats_monthly(u), "mkt": stats_monthly(k)})
    full, post = umd, window(umd, "1994-01-31", end)
    pre = window(umd, umd.index[0], "1993-12-31")
    skew, kurt = moments(full)
    dsr = {"full_n1": deflated(full, 1), "full_n16": deflated(full, 16), "post_n1": deflated(post, 1),
           "breakeven_n_full": breakeven_tries(full), "skew_full": skew, "kurt_full": kurt}
    decay = 1 - post.mean() / pre.mean()
    growth = (1 + umd).cumprod()
    dd = growth / growth.cummax() - 1
    return rows, dsr, float(decay), growth, (1 + mkt).cumprod(), dd


# ---------------------------------------------------------------- part B
def run_backtests(px, mkt_px, costs):
    tests = [bt.Backtest(strategy(f"L{lb}_N{n}", n, lb), px, initial_capital=CAPITAL,
                         commissions=costs.fee, integer_positions=False)
             for lb in LOOKBACKS for n in TOP_NS]
    tests.append(bt.Backtest(benchmark("Market"), mkt_px, initial_capital=CAPITAL, commissions=costs.fee,
                             integer_positions=False))
    tests.append(bt.Backtest(benchmark("EqualWeight", monthly=True), px, initial_capital=CAPITAL,
                             commissions=costs.fee, integer_positions=False))
    return bt.run(*tests)


def orders(res, name, costs):
    tx = res.get_transactions(name).reset_index()
    tx.columns = ["date", "security", "price", "quantity"]
    tx["value"] = (tx["price"] * tx["quantity"]).abs()
    tx["fee"] = [costs.fee(q, p) for q, p in zip(tx["quantity"], tx["price"])]
    tx["side"] = np.where(tx["quantity"] > 0, "BUY", "SELL")
    return tx


def part_b(d):
    names = [c for c in d.columns if c not in ("Market", "RF")]
    d = d.loc["1998-01-01":TEST[1]]
    px = (1 + d[names]).cumprod() * 100          # total-return indexes (dividends included)
    mkt_px = ((1 + d["Market"]).cumprod() * 100).to_frame("Market")
    rf = d["RF"]

    res = run_backtests(px, mkt_px, COSTS)
    rets = res.prices.pct_change().iloc[1:]
    grid = []
    for lb in LOOKBACKS:
        for n in TOP_NS:
            k = f"L{lb}_N{n}"
            grid.append({"name": k, "lookback": lb, "top_n": n,
                         "design": stats_daily(window(rets[k], *DESIGN), rf),
                         "test": stats_daily(window(rets[k], *TEST), rf)})
    best = max(grid, key=lambda g: g["design"]["sharpe"])   # chosen on the design window only
    k = best["name"]
    design_sr = np.array([g["design"]["sharpe"] for g in grid]) / np.sqrt(252)
    dsr_design = deflated(window(rets[k], *DESIGN) - rf.reindex(window(rets[k], *DESIGN).index),
                          len(grid), sr_std=float(design_sr.std(ddof=1)))

    gross = bt.run(bt.Backtest(strategy(k, best["top_n"], best["lookback"]), px, initial_capital=CAPITAL,
                               integer_positions=False))
    harsh = bt.run(bt.Backtest(strategy(k, best["top_n"], best["lookback"]), px, initial_capital=CAPITAL,
                               commissions=PRESETS["harsh"].fee, integer_positions=False))
    g_r = gross.prices[k].pct_change().iloc[1:]
    h_r = harsh.prices[k].pct_change().iloc[1:]

    tx = orders(res, k, COSTS)
    tx_test = tx[(tx.date >= TEST[0]) & (tx.date <= TEST[1])]
    eq = res.prices[k] * CAPITAL / 100
    out = {
        "chosen": {"name": k, "lookback": best["lookback"], "top_n": best["top_n"], "tries": len(grid),
                   "dsr_design": dsr_design},
        "grid": grid,
        "strategy": {w: stats_daily(window(rets[k], *span), rf) for w, span in (("design", DESIGN), ("test", TEST))},
        "market": {w: stats_daily(window(rets["Market"], *span), rf) for w, span in (("design", DESIGN), ("test", TEST))},
        "equal_weight": {w: stats_daily(window(rets["EqualWeight"], *span), rf) for w, span in (("design", DESIGN), ("test", TEST))},
        "gross_test": stats_daily(window(g_r, *TEST), rf),
        "harsh_test": stats_daily(window(h_r, *TEST), rf),
        "settings_beating_market_test": sum(g["test"]["sharpe"] > stats_daily(window(rets["Market"], *TEST), rf)["sharpe"]
                                            for g in grid),
        "test_orders": int(len(tx_test)), "test_fees": float(tx_test.fee.sum()),
        "test_fees_pct_avg_equity": float(tx_test.fee.sum() / window(eq, *TEST).mean()),
        "test_turnover_per_year": float(tx_test.value.sum() / window(eq, *TEST).mean() / (len(window(eq, *TEST)) / 252)),
        "all_orders": int(len(tx)), "all_fees": float(tx.fee.sum()),
    }
    return out, res, px, tx, names


# ---------------------------------------------------------------- charts for part B
def b_charts(b, res, px, tx, names, foot):
    k = b["chosen"]["name"]
    start = pd.Timestamp(DESIGN[0])
    split = pd.Timestamp(TEST[0])

    def from_start(s):
        s = s.loc[start:]
        return s / s.iloc[0] * CAPITAL

    strat, mkt, ew = from_start(res.prices[k]), from_start(res.prices["Market"]), from_start(res.prices["EqualWeight"])
    label = f"Top {b['chosen']['top_n']} industries by {b['chosen']['lookback']}-1 month return"
    charts.equity([(ew, "All 10 industries, equal weight", charts.MUTED, 1.8), (mkt, "US stock market, buy and hold", charts.TEAL, 2.4),
                   (strat, label + ", after costs", charts.YELLOW, 3.0)],
                  split, OUT / "diy_equity.png", "Do it yourself: $100k, 1999 to 2026",
                  f"{label}, monthly. Costs: {COSTS.short()}.", foot)

    def from_test(s):
        s = s.loc[split:]
        return s / s.iloc[0] * CAPITAL

    charts.equity([(from_test(res.prices["EqualWeight"]), "All 10 industries, equal weight", charts.MUTED, 1.8),
                   (from_test(res.prices["Market"]), "US stock market, buy and hold", charts.TEAL, 2.4),
                   (from_test(res.prices[k]), label + ", after costs", charts.YELLOW, 3.0)],
                  None, OUT / "diy_equity_test.png", "The test: $100k from January 2016",
                  f"Settings frozen on 1999-2015 data. Costs: {COSTS.short()}.", foot)

    sd, md = strat / strat.cummax() - 1, mkt / mkt.cummax() - 1
    worst = sd.idxmin()
    charts.underwater(sd, {"worst": (worst, f"{sd.min():.0%} ({worst:%b %Y})")}, OUT / "diy_drawdown.png",
                      "Drawdown: strategy vs the market", "How far below its last high each one was, after costs",
                      foot, color=charts.CORAL, other=md, other_label="US stock market", own_label="Momentum rotation", split=split)

    w = res.get_security_weights(k).loc[start:].resample("ME").last()
    held = (w[names] > 0.001).astype(int).rename(columns=INDUSTRY_NAMES)
    charts.holdings(held, split, OUT / "diy_holdings.png", "What it held, month by month",
                    f"{label}. Each bar is a month held. Yellow = the test years.", foot)

    t = tx[(tx.date >= TEST[0])].sort_values("date")
    rows = [{"date": r.date.strftime("%Y-%m-%d"), "side": r.side, "name": INDUSTRY_NAMES.get(r.security, r.security),
             "value": r.value, "fee": r.fee} for r in t.tail(13).itertuples()]
    charts.trade_table(rows, f"2016-2026: {b['test_orders']} orders, ${b['test_fees']:,.0f} in fees and slippage"
                             f" ({b['test_fees_pct_avg_equity']:.1%} of average equity)",
                       OUT / "diy_trades.png", "The trade list, with fees", f"Last {len(rows)} orders. {COSTS.describe()}.", foot)

    # one holding, entries and exits marked: the industry it moved in and out of most in the test window
    on_all = res.get_security_weights(k)[names] > 0.001
    starts = on_all & ~on_all.shift(1, fill_value=False)      # a buy into an industry not held the day before
    stops = ~on_all & on_all.shift(1, fill_value=False)       # a sale of the whole position
    on, starts, stops = on_all.loc[split:], starts.loc[split:], stops.loc[split:]
    pick = (starts.sum() + stops.sum()).idxmax()
    s_on = on[pick]
    entries = list(s_on.index[starts[pick]])
    exits = list(s_on.index[stops[pick]])
    spans, cur = [], None
    for dt, v in s_on.items():
        if v and cur is None:
            cur = dt
        if not v and cur is not None:
            spans.append((cur, dt))
            cur = None
    if cur is not None:
        spans.append((cur, s_on.index[-1]))
    price = px[pick].loc[split:]
    price = price / price.iloc[0] * 100
    tp = t[t.security == pick]
    nm = INDUSTRY_NAMES.get(pick, pick)
    charts.trade_chart(price, spans, entries, exits,
                       f"{len(tp)} orders in {nm}: ${tp.fee.sum():,.0f} in fees", OUT / "diy_trade_chart.png",
                       f"One holding: {nm}, 2016 to 2026",
                       "Total-return index, 100 = Jan 2016. Shaded = held. Orders fill at the close.", foot)
    return {"trade_chart_industry": nm, "trade_chart_entries": len(entries), "trade_chart_exits": len(exits),
            "trade_chart_orders": int(len(tp)), "trade_chart_fees": float(tp.fee.sum())}


INDUSTRY_NAMES = {"NoDur": "Consumer staples", "Durbl": "Consumer durables", "Manuf": "Manufacturing",
                  "Enrgy": "Energy", "HiTec": "Tech", "Telcm": "Telecom", "Shops": "Retail",
                  "Hlth": "Healthcare", "Utils": "Utilities", "Other": "Other"}


# ---------------------------------------------------------------- verdict
def verdict(a_rows, b):
    post = next(r for r in a_rows if r["period"] == "After publication")["umd"]
    factor_ok = post["t"] >= 2.0                                                   # the factor, after the paper
    diy_ok = (b["strategy"]["test"]["sharpe"] > b["market"]["test"]["sharpe"]
              and b["strategy"]["test"]["cagr"] > b["market"]["test"]["cagr"])     # DIY beats the market, out of sample
    v = "HOLDS UP" if factor_ok and diy_ok else "DOESN'T" if not factor_ok and not diy_ok else "PARTLY"
    return v, factor_ok, diy_ok


def fmt_pct(x):
    return f"{x * 100:+.1f}%" if x < 0 else f"{x * 100:.1f}%"


def main():
    OUT.mkdir(exist_ok=True)
    vintage = data.fetch()
    print(f"momentum | {vintage}")
    foot = f"Data: {vintage}. Code: github.com/WallStVerdict/momentum-after-the-paper (MIT). Education, not advice."

    # A
    m = data.monthly_factors()
    print(f"[1/3] the factor as published: UMD, {m.index[0]:%Y-%m} to {m.index[-1]:%Y-%m}")
    a_rows, dsr, decay, growth, mkt_growth, dd = part_a(m)
    for r in a_rows:
        u = r["umd"]
        print(f"      {r['period']:<20} Sharpe {u['sharpe']:.2f}  t {u['t']:.1f}")
    post_mkt = next(r for r in a_rows if r["period"] == "After publication")["mkt"]
    print(f"      {'market after pub.':<20} Sharpe {post_mkt['sharpe']:.2f}  t {post_mkt['t']:.1f}")
    full = a_rows[0]["umd"]
    print(f"      worst month {full['worst_month'] * 100:.1f}% ({full['worst_month_date']})")

    crash09 = dd.loc["2009"].idxmin()
    crash32 = dd.loc["1932"].idxmin()
    charts.umd_century(growth, mkt_growth, PUBLISHED,
                       {"crash1932": (crash32, "1932"), "crash2009": (crash09, "2009")},
                       OUT / "umd_century.png", foot)
    charts.sharpe_bars([(r["period"].replace("J&T's own sample", "J&T sample\n1965-1989")
                         .replace("Before publication", "Before\n1927-1993").replace("After publication", "After\n1994-2026")
                         .replace("Full sample", "Full\n1927-2026").replace("Last 10 years", "Last 10 years\n2016-2026"),
                         r["umd"]["sharpe"], r["umd"]["t"], r["mkt"]["sharpe"], r["mkt"]["t"]) for r in a_rows],
                       OUT / "sharpe_bars.png", foot)
    worst32, worst09 = m["UMD"].loc["1932"], m["UMD"].loc["2009"]
    charts.underwater(dd, {"crash1932": (crash32, f"1932: worst month {worst32.min():.0%}"),
                           "crash2009": (crash09, f"2009: worst month {worst09.min():.0%}")},
                      OUT / "umd_drawdown.png", "Momentum crashes",
                      "Drawdown of the momentum factor (UMD) from its last high, before costs", foot)

    # B
    print(f"[2/3] do it yourself: top N of 10 industries by past return, bt engine")
    print(f"      costs: {COSTS.short()}, every buy and every sell")
    d = data.daily_industries()
    b, res, px, tx, names = part_b(d)
    b.update(b_charts(b, res, px, tx, names, foot))
    c, s, mk = b["chosen"], b["strategy"], b["market"]
    print(f"      design 1999-2015: {c['tries']} settings tried, best {c['lookback']}-1, top {c['top_n']}"
          f"  Sharpe {s['design']['sharpe']:.2f}")
    print(f"      test 2016-2026: Sharpe {s['test']['sharpe']:.2f} vs market {mk['test']['sharpe']:.2f}"
          f" | a year {fmt_pct(s['test']['cagr'])} vs {fmt_pct(mk['test']['cagr'])}")
    print(f"      fees 2016-2026: ${b['test_fees']:,.0f} on {b['test_orders']} orders"
          f" | settings that beat the market: {b['settings_beating_market_test']}/{c['tries']}")

    # C
    v, factor_ok, diy_ok = verdict(a_rows, b)
    post = a_rows[3]["umd"]
    print(f"[3/3] PRE-SET RULE, factor + frozen test: {v}")
    print(f"      after the paper: Sharpe {post['sharpe']:.2f}, t {post['t']:.1f}; mean return {decay:.0%} lower")

    write_summary(a_rows, dsr, decay, b, v, factor_ok, diy_ok, vintage)
    print("results/  summary.txt  summary.json  9 charts")
    return 0


def write_summary(a_rows, dsr, decay, b, v, factor_ok, diy_ok, vintage):
    L = [f"momentum | {vintage}", "",
         "A. THE FACTOR AS PUBLISHED (Ken French UMD, monthly, before costs)",
         f"{'period':<20}{'months':>7}{'ann.ret':>9}{'Sharpe':>8}{'t':>6}{'worst month':>21}{'max DD':>9}"]
    for r in a_rows:
        u = r["umd"]
        L.append(f"{r['period']:<20}{u['months']:>7}{u['mean_ann'] * 100:>8.1f}%{u['sharpe']:>8.2f}{u['t']:>6.1f}"
                 f"{u['worst_month'] * 100:>9.1f}% {u['worst_month_date']:>10}{u['max_drawdown'] * 100:>8.0f}%")
    L.append("market (Mkt-RF), same periods")
    for r in a_rows:
        k = r["mkt"]
        L.append(f"{r['period']:<20}{k['months']:>7}{k['mean_ann'] * 100:>8.1f}%{k['sharpe']:>8.2f}{k['t']:>6.1f}")
    L += [f"Mean UMD return after 1993 vs before: {decay:.0%} lower (McLean & Pontiff 2016: 58% average for 97 predictors)",
          f"Probabilistic Sharpe, full sample, N=1:    {dsr['full_n1']:.4f}  (needs 0.95; skew {dsr['skew_full']:.1f}, kurtosis {dsr['kurt_full']:.1f})",
          f"Deflated Sharpe, full sample, N=16:        {dsr['full_n16']:.4f}  (J&T tried 16 formation/holding combos)",
          f"Tries it could be the best of and pass:    {dsr['breakeven_n_full']:,}",
          f"Probabilistic Sharpe, after 1993, N=1:     {dsr['post_n1']:.4f}", "",
          "B. DO IT YOURSELF (bt, 10 Ken French industry portfolios, daily, monthly rotation)",
          f"Costs: {COSTS.describe()}. Start ${CAPITAL:,}.",
          f"Chosen on design window {DESIGN[0][:4]}-{DESIGN[1][:4]} from {b['chosen']['tries']} settings: "
          f"{b['chosen']['lookback']}-1 month return, top {b['chosen']['top_n']}",
          f"Deflated Sharpe of that pick, N={b['chosen']['tries']}: {b['chosen']['dsr_design']:.3f} (needs 0.95)",
          f"{'':<26}{'CAGR':>8}{'vol':>8}{'Sharpe':>8}{'t':>6}{'max DD':>9}"]
    for w in ("design", "test"):
        L.append(f"{w} {b['strategy'][w]['start'][:4]}-{b['strategy'][w]['end'][:4]}")
        for lab, s in (("  strategy, after costs", b["strategy"][w]), ("  market, buy and hold", b["market"][w]),
                       ("  10 industries, equal wt", b["equal_weight"][w])):
            L.append(f"{lab:<26}{s['cagr'] * 100:>7.1f}%{s['vol_ann'] * 100:>7.1f}%{s['sharpe']:>8.2f}{s['t']:>6.1f}{s['max_drawdown'] * 100:>8.0f}%")
    g, h = b["gross_test"], b["harsh_test"]
    L += [f"  strategy before costs     {g['cagr'] * 100:>6.1f}%{g['vol_ann'] * 100:>7.1f}%{g['sharpe']:>8.2f}",
          f"  strategy, harsh costs     {h['cagr'] * 100:>6.1f}%{h['vol_ann'] * 100:>7.1f}%{h['sharpe']:>8.2f}   ({PRESETS['harsh'].describe()})",
          f"Test orders {b['test_orders']}, fees ${b['test_fees']:,.0f} ({b['test_fees_pct_avg_equity']:.1%} of average equity), "
          f"turnover {b['test_turnover_per_year']:.1f}x a year",
          f"All {len(b['grid'])} settings (Sharpe design -> test); beat the market's test Sharpe: "
          f"{b['settings_beating_market_test']} of {len(b['grid'])}"]
    for gr in b["grid"]:
        L.append(f"  {gr['lookback']:>2}-1, top {gr['top_n']}: {gr['design']['sharpe']:.2f} -> {gr['test']['sharpe']:.2f}")
    L += ["", f"C. PRE-SET RULE, factor + frozen test: {v}",
          f"  factor after publication t >= 2: {'yes' if factor_ok else 'no'}",
          f"  DIY beats the market 2016-2026 on Sharpe and return, after costs: {'yes' if diy_ok else 'no'}",
          "Education, not advice."]
    (OUT / "summary.txt").write_text("\n".join(L) + "\n")

    post = a_rows[3]
    flat = {"rule_verdict": v, "factor_post_ok": factor_ok, "diy_ok": diy_ok, "decay_pct": round(decay * 100, 1),
            "umd_full_sharpe": a_rows[0]["umd"]["sharpe"], "umd_full_t": a_rows[0]["umd"]["t"],
            "umd_pre_sharpe": a_rows[2]["umd"]["sharpe"], "umd_pre_t": a_rows[2]["umd"]["t"],
            "umd_post_sharpe": post["umd"]["sharpe"], "umd_post_t": post["umd"]["t"],
            "mkt_post_sharpe": post["mkt"]["sharpe"], "mkt_post_t": post["mkt"]["t"],
            "umd_last10_sharpe": a_rows[4]["umd"]["sharpe"], "umd_last10_t": a_rows[4]["umd"]["t"],
            "umd_worst_month_pct": a_rows[0]["umd"]["worst_month"] * 100,
            "umd_worst_month_date": a_rows[0]["umd"]["worst_month_date"],
            "psr_full": dsr["full_n1"], "dsr_full_n16": dsr["full_n16"], "psr_post": dsr["post_n1"],
            "breakeven_tries": dsr["breakeven_n_full"],
            "diy_lookback": b["chosen"]["lookback"], "diy_top_n": b["chosen"]["top_n"], "diy_tries": b["chosen"]["tries"],
            "diy_design_sharpe": b["strategy"]["design"]["sharpe"], "diy_test_sharpe": b["strategy"]["test"]["sharpe"],
            "mkt_test_sharpe": b["market"]["test"]["sharpe"], "diy_test_cagr_pct": b["strategy"]["test"]["cagr"] * 100,
            "mkt_test_cagr_pct": b["market"]["test"]["cagr"] * 100, "diy_test_fees": b["test_fees"],
            "diy_test_orders": b["test_orders"], "settings_beating_market": b["settings_beating_market_test"], "diy_gross_test_sharpe": b["gross_test"]["sharpe"],
            "trade_chart_industry": b["trade_chart_industry"], "trade_chart_fees": b["trade_chart_fees"],
            "trade_chart_orders": b["trade_chart_orders"]}
    detail = {"part_a": a_rows, "deflated": dsr, "part_b": b}
    (OUT / "summary.json").write_text(json.dumps({**flat, "detail": detail}, indent=1, default=float) + "\n")


if __name__ == "__main__":
    sys.exit(main())
