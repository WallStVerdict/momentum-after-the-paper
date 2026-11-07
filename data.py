"""Ken French data library: download once, cache under data/, check against data.sha256.

The library is rebuilt every month from a new CRSP vintage, so a fresh download can differ slightly
from the copy we checksummed. The files are not ours to redistribute, so the first run downloads them.
data.sha256 holds the vintage behind the published numbers: a newer one prints a WARNING and the numbers may move a little.
`python data.py --refresh` downloads the current vintage and updates data.sha256.
"""
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data"
SUMS = Path(__file__).resolve().parent / "data.sha256"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {   # cached csv  <-  zip on the Ken French data library
    "F-F_Momentum_Factor.csv": "F-F_Momentum_Factor_CSV.zip",                    # UMD, monthly
    "F-F_Research_Data_Factors.csv": "F-F_Research_Data_Factors_CSV.zip",        # Mkt-RF, RF, monthly
    "F-F_Research_Data_Factors_daily.csv": "F-F_Research_Data_Factors_daily_CSV.zip",
    "10_Industry_Portfolios_Daily.csv": "10_Industry_Portfolios_daily_CSV.zip",  # 10 industries, daily
}


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def download(name):
    with urllib.request.urlopen(BASE + FILES[name], timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    member = next(m for m in z.namelist() if m.lower() == name.lower())
    (DATA / name).write_bytes(z.read(member))


def fetch(refresh=False):
    """Make sure every csv is cached and matches data.sha256. Returns the vintage line."""
    DATA.mkdir(exist_ok=True)
    known = {}
    if SUMS.exists():
        for line in SUMS.read_text().splitlines():
            if line.strip():
                h, f = line.split()
                known[Path(f).name] = h
    for name in FILES:
        if refresh or not (DATA / name).exists():
            print(f"      download {FILES[name]}", flush=True)
            download(name)
        h = sha256(DATA / name)
        if name in known and known[name] != h:
            if refresh:
                print(f"      NOTE {name}: new vintage, data.sha256 updated; numbers may move a little")
            else:
                print(f"      WARNING {name} does not match data.sha256: these are not the published numbers")
                continue
        known[name] = h
    if refresh or not SUMS.exists():
        SUMS.write_text("".join(f"{known[n]}  data/{n}\n" for n in FILES))
    head = (DATA / "F-F_Momentum_Factor.csv").read_text(errors="replace")
    m = re.search(r"(\d{6}) CRSP", head)
    return f"Ken French data library, CRSP vintage {m.group(1) if m else '?'}"


def _table(path, section=0, width=6):
    """The section-th table of a Ken French csv: rows keyed YYYYMM (width 6) or YYYYMMDD (width 8)."""
    rows, header, k = [], None, -1
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(","):
            k += 1
            header = [c.strip() for c in line.split(",")[1:]]
            continue
        if k == section and header and re.match(rf"^\s*\d{{{width}}}\s*,", line):
            parts = [c.strip() for c in line.split(",")]
            rows.append([parts[0]] + [float(x) for x in parts[1:]])
        elif k == section and rows:
            break
    df = pd.DataFrame(rows, columns=["date"] + header).set_index("date")
    fmt = "%Y%m" if width == 6 else "%Y%m%d"
    df.index = pd.to_datetime(df.index, format=fmt)
    if width == 6:
        df.index = df.index + pd.offsets.MonthEnd(0)
    df = df.mask(df <= -99.99)       # -99.99 / -999 mean missing
    return df / 100.0                # percent -> decimal


def monthly_factors():
    """Monthly UMD, Mkt-RF and RF as decimals, indexed by month end."""
    mom = _table(DATA / "F-F_Momentum_Factor.csv")["Mom"].rename("UMD")
    ff = _table(DATA / "F-F_Research_Data_Factors.csv")[["Mkt-RF", "RF"]]
    return ff.join(mom, how="inner").dropna()


def daily_industries():
    """Daily value-weighted returns of the 10 industry portfolios, plus the market and RF (decimals)."""
    ind = _table(DATA / "10_Industry_Portfolios_Daily.csv", section=0, width=8)
    ff = _table(DATA / "F-F_Research_Data_Factors_daily.csv", width=8)
    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("Market")
    return ind.join(mkt, how="inner").join(ff["RF"], how="inner").dropna()


if __name__ == "__main__":
    print(fetch(refresh="--refresh" in sys.argv))
