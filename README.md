# Momentum after the paper

<!-- YOUTUBE THUMBNAIL GOES HERE. After upload replace the next line with:
     [![Watch the episode](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/VIDEO_ID) -->
**Episode video: link coming after upload.**

**Claim tested:** "Momentum had 100 years of edge. Then the paper came out."

Momentum means buying what went up over the past 3 to 12 months and selling what went down.
Jegadeesh & Titman published it in 1993 (*Returns to Buying Winners and Selling Losers*, Journal of Finance 48(1), 65-91).
Carhart (1997, *On Persistence in Mutual Fund Performance*, Journal of Finance 52(1), 57-82) added a momentum factor to the Fama-French three-factor model.
Ken French publishes a momentum factor ("Mom", often called UMD) every month back to 1927.

**Why this one:** it is the most famous anomaly in finance, the claim is public and dated (1993),
and the data to test it is free for every month back to 1927. That gives us a clean before and after.

We ask three things.
Did the factor keep working after the paper?
Can you capture it yourself, after fees, on years you never looked at?
And if you re-pick the settings every year from the past only (a walk-forward), does it still hold?

## Reproduce

```
make reproduce        # builds .venv on first use, downloads the data, then runs (about 3.5 min)
make docker           # or in Docker (python:3.12-slim), nothing else needed. DOCKER=podman works too
```

The Ken French files are not ours to redistribute, so the first run downloads them into `data/` from the library.
`data.sha256` is the fingerprint of the vintage behind the numbers below (CRSP 202609).
The library is rebuilt every month. If a newer vintage is live you get a WARNING, and the numbers may move a little.
`make data` fetches the newest copy and updates the fingerprint.

## How it works

**A. The factor as published.** Monthly UMD and the market (Mkt-RF) from the Ken French data library.
Sharpe ratio, t-stat and drawdowns for the full sample, Jegadeesh & Titman's own sample (1965-1989),
before publication (1927-1993), after publication (1994-2026) and the last 10 years. Before costs.
Then the Probabilistic and Deflated Sharpe Ratio from episode 1 (Bailey & Lopez de Prado).

**B. Do it yourself.** The `bt` engine. Universe: Ken French's 10 industry portfolios, daily, dividends included.
On the first trading day of each month, rank them by past return, skipping the latest month (the "12-1" rule).
Hold the top N, equal weight. Orders fill at that day's close. The ranking uses finished months only, so no look-ahead.
Costs on every order (`costs.py`, preset `etf_retail`): $1.00 + 2 bps spread + 3 bps slippage.
Settings tried: lookback 6, 9 or 12 months x top 1 to 5 = 15 settings.
The best one on **1999-2015** is frozen, then tested on **2016 to Sep 2026**.
Benchmarks: the US stock market bought and held (Ken French market return, our stand-in for SPY), and all 10 industries equal weight.
Engine check: bt's buy-and-hold market over 2016-2026 matches compounding the raw data by hand (14.95% a year, Sharpe 0.73).

**C. The pre-set rule.** Fixed before the run, on the factor and the frozen test only:
HOLDS UP if the factor's t-stat after publication is at least 2 *and* the do-it-yourself version beats the market
on both Sharpe and annual return in the test years, after costs. DOESN'T if neither. PARTLY otherwise.

**D. Walk-forward** (`walk_forward.py`). One frozen split can be lucky. So every January, from 1938 to 2026:
look back over the past 10 years only, keep the best of the same 15 settings by Sharpe, trade it the next 12 months
with the same fees. Every traded year is out of sample. Also run with 5-year, 15-year and all-history training
windows, reported, not chosen from. bt runs in 20-year blocks with a 2-year warm-up (one 100-year run trips bt's
order-sizing loop); the joined series is continuous.

Files: `momentum.py` (parts A, B, C), `walk_forward.py` (part D), `data.py` (download, checksum, parse), `costs.py` (fee presets),
`charts.py` (house charts), `results/` (summary + charts).

## Results (Ken French CRSP vintage 202609)

**A. The momentum factor, before costs**

| Period | Months | Mean return a year | Sharpe | t-stat | Market Sharpe |
|---|---:|---:|---:|---:|---:|
| Full sample, 1927-2026 | 1,197 | 7.4% | 0.46 | 4.6 | 0.45 |
| J&T's own sample, 1965-1989 | 300 | 9.9% | 0.80 | 4.0 | 0.27 |
| Before publication, 1927-1993 | 804 | 8.8% | 0.55 | 4.5 | 0.40 |
| **After publication, 1994-2026** | 393 | 4.5% | **0.27** | **1.5** | 0.60 |
| Last 10 years, Oct 2016-Sep 2026 | 120 | 3.1% | 0.22 | 0.7 | 0.82 |

- Worst month: **-52.6% (Aug 1932)**. Worst after publication: -34.4% (Apr 2009). Deepest drawdown: -78%.
- The average return after 1993 is **49% lower** than before. McLean & Pontiff (2016, Journal of Finance 71(1)) found
  returns 58% lower after publication, on average across 97 published predictors.
- Probabilistic Sharpe, full sample, one try: 0.9999 (needs 0.95). This counts because N = 1:
  UMD is one fixed, published rule, not the best of a thousand backtests.
  Deflated for the 16 formation and holding combinations Jegadeesh & Titman tested: 0.987, still passes.
  It would still pass as the best of up to 96 tries (assuming independent tries with no skill).
- After publication alone: 0.927. It does not pass.

**B. Do it yourself: top 2 industries by 6-1 month return** (chosen on 1999-2015 from 15 settings)

| | CAGR | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Design 1999-2015, strategy after costs | 11.3% | 0.54 | -46% |
| Design 1999-2015, market buy and hold | 5.4% | 0.27 | -55% |
| **Test 2016-2026, strategy after costs** | **14.7%** | **0.62** | -38% |
| **Test 2016-2026, market buy and hold** | **15.0%** | **0.73** | -34% |
| Test 2016-2026, 10 industries equal weight | 13.1% | 0.67 | -37% |
| Test 2016-2026, strategy before costs | 15.3% | 0.65 | |
| Test 2016-2026, strategy with harsh costs ($5 + 20 bps) | 13.2% | 0.56 | |

- Test years: 353 orders, $87,384 in fees and slippage (4.8% of average equity), turnover 9.0x a year.
- **None of the 15 settings beat the market's test Sharpe** (best: 0.71 vs 0.73).
- The pick that won the design window landed in the bottom half of the 15 in the test (0.62; textbook 12-1 top 3 got 0.70). Its Deflated Sharpe on the design window was 0.974 (N = 15).

**D. Walk-forward, 1938-2026** (re-picked every January on the past 10 years, after costs; Sharpe above T-bills)

| Period | Walk-forward | Market | 10 industries, equal weight |
|---|---:|---:|---:|
| Before the paper, 1938-1993 | **0.75** | 0.59 | 0.67 |
| After the paper, 1994-2026 | **0.41** | 0.52 | 0.55 |
| Last 10 years, 2016-2026 | **0.38** | 0.73 | 0.67 |

- Beat the market in 55 of 89 years. The pick changed 51 times in 89 years.
- Other training windows (5 years, 15 years, all history): before the paper the walk-forward beat the market every time
  (0.69-0.76 vs 0.58-0.65). After it: 0.41 to 0.56 vs 0.52. Last 10 years: 0.36 to 0.58 vs 0.73, behind every time.
- Full table: `results/walk_forward.txt`. Charts: `wf_equity.png`, `wf_sharpe_bars.png`, `wf_picks.png`.

## Prior work (this is not our discovery)

- Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*, Journal of Finance 48(1), 65-91. The claim.
- Carhart (1997), *On Persistence in Mutual Fund Performance*, Journal of Finance 52(1), 57-82. The momentum factor.
- Moskowitz & Grinblatt (1999), *Do Industries Explain Momentum?*, Journal of Finance 54(4), 1249-1290. Industry momentum, what our DIY version trades.
- Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*, Journal of Finance 68(3), 929-985. Momentum in other markets and asset classes.
- McLean & Pontiff (2016), *Does Academic Research Destroy Stock Return Predictability?*, Journal of Finance 71(1), 5-32. 97 predictors: returns 26% lower out of sample, 58% lower after publication.
- Daniel & Moskowitz (2016), *Momentum Crashes*, Journal of Financial Economics 122(2), 221-247. The 1932 and 2009 crashes.
- Bailey & Lopez de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management 40(5), 94-107. Episode 1.

## Verdict: PARTLY

The rule we fixed in advance says DOESN'T, and `momentum.py` prints that: the factor's t-stat after the paper is under 2, and the frozen test lost to the market.
Our overall call is PARTLY, because of the walk-forward we added after that rule was written.
After 1993 its Sharpe ranged from 0.41 to 0.56 across four look-backs, against the market's 0.52, so on Sharpe it roughly tied at best.
That judgement is ours, and you can read it against the numbers above.

Momentum was real. Over 99 years its t-stat is 4.6, and it survives the deflation test.
After Jegadeesh & Titman published it, the average return fell by about half and the t-stat dropped to 1.5.
Simply holding the market had a higher Sharpe ratio over the same years.
Our own version, traded with fees on years it never saw, made 14.7% a year against the market's 15.0%, with more risk.

The honest twist: the edge did not go to zero.
After 1993 the factor still averaged 4.5% a year before costs.
But that is no longer enough to tell it apart from luck, and after costs it did not beat buying the market.

What would change it: a long run of strong momentum years, or a version with much lower costs and crash protection.
The return gap in the test is small (0.3 points a year), so a different end date could flip that part.
The Sharpe gap (0.62 vs 0.73) is wider.

## Data and limits

- Ken French's industry portfolios are value-weighted baskets of US stocks, not ETFs.
  We planned Stooq ETF prices. On 1 Nov 2026 Stooq's download link answered with a JavaScript bot check, so we used open,
  dividend-inclusive Ken French data instead and priced every order as if traded through a liquid sector ETF.
- The market benchmark is Ken French's US market return (CRSP), a stand-in for SPY. No fund fee is deducted from it.
- UMD is a long-short paper portfolio. Its numbers are before trading costs, shorting costs and taxes.
- Sharpe ratios use returns over the 1-month T-bill. t-stat = mean / standard error.
- Part A uses monthly returns and parts B and D daily returns, so the market's Sharpe for the same years differs a little:
  1994-2026 is 0.60 from monthly returns and 0.52 from daily returns.

## Licences

- Our code (`*.py`, `Makefile`, `Dockerfile`): MIT, see `LICENSE`. Third-party notes: `NOTICE`.
- Data: Kenneth R. French Data Library, Copyright Eugene F. Fama and Kenneth R. French, built from CRSP.
  Not included here; `make reproduce` downloads it.
- Engine: `bt` and `ffn` (MIT). Also pandas, numpy, scipy, scikit-learn (BSD), matplotlib (PSF-style). No AGPL code is used.
- Fonts: Inter (SIL Open Font License 1.1, `fonts/OFL-Inter.txt`). Anton is used for chart titles only if installed locally.

Education, not advice.
