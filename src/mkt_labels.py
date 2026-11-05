"""Market-based distress indicators from the TSETMC adjusted daily prices (data/tse_prices/<symbol>-ت.csv).

For every firm-year (symbol, fiscal year t) of data/dataset_model.csv the outcome window is the next fiscal year
(year end of t, year end of t+1]. Within it:
  ret      adjusted return over the window (last close in the window / last close up to the year end of t - 1)
  mdd      maximum drawdown of the adjusted close within the window
  max_gap  longest period without trading, in calendar days, including the window edges (trading halts)
  y_crash  ret < -0.5;  y_halt  max_gap >= 120;  y_mkt = y_crash or y_halt
Output: data/market_labels.csv
"""
import os, glob, datetime as dt
import numpy as np, pandas as pd, jdatetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PR = os.path.join(ROOT, "data", "tse_prices")


def j2g(s):
    y, m, d = (int(x) for x in s.replace("/", "-").split("-")[:3])
    for dd in (d, d - 1, d - 2):
        try: return jdatetime.date(y, m, dd).togregorian()
        except ValueError: pass
    return None


def main():
    d = pd.read_csv(os.path.join(ROOT, "data", "dataset_model.csv"))[["symbol", "fy", "year_end"]]
    d["symbol"] = d.symbol.str.strip(); d = d.dropna()
    ye = {(s, int(f)): y for s, f, y in zip(d.symbol, d.fy, d.year_end)}
    out = []
    for s, g in d.groupby("symbol"):
        fn = os.path.join(PR, f"{s}-ت.csv")
        if not os.path.exists(fn): continue
        p = pd.read_csv(fn)
        if p.empty: continue
        p["g"] = pd.to_datetime(p["date"]).dt.date; p = p.sort_values("g")
        first_trade = p.g.iloc[0]
        for fy in g.fy.astype(int):
            a = j2g(ye[(s, fy)])
            nxt = ye.get((s, fy + 1))
            if nxt is None:
                y_, m_, d_ = ye[(s, fy)].split("/"); nxt = f"{int(y_) + 1}/{m_}/{d_}"
            b = j2g(nxt)
            if a is None or b is None or b > dt.date.today() or a < first_trade: continue
            before = p[p.g <= a]; win = p[(p.g > a) & (p.g <= b)]
            if before.empty: continue
            c0 = float(before.adjClose.iloc[-1])
            dates = [a] + list(win.g) + [b]
            gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            if win.empty:
                ret, mdd = np.nan, np.nan
            else:
                c = np.r_[c0, win.adjClose.values.astype(float)]
                ret = c[-1] / c0 - 1; mdd = float(np.min(c / np.maximum.accumulate(c)) - 1)
            mg = max(gaps)
            out.append(dict(symbol=s, fy=fy, ret=ret, mdd=mdd, max_gap=mg, n_trade=len(win),
                            y_crash=int(np.isfinite(ret) and ret < -0.5), y_halt=int(mg >= 120)))
    r = pd.DataFrame(out); r["y_mkt"] = ((r.y_crash == 1) | (r.y_halt == 1)).astype(int)
    r.to_csv(os.path.join(ROOT, "data", "market_labels.csv"), index=False)
    print(len(r), "firm-years;", r[["y_crash", "y_halt", "y_mkt"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
