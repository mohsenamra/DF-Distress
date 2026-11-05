"""Daily adjusted price history of the Iranian panel firms from TSETMC (for a market-based distress check).

One request per symbol, a few seconds apart; resumable (symbols already saved are skipped).
Output: data/tse_prices/<symbol>.csv (adjusted daily prices, with Jalali dates) and data/tse_prices/_status.csv
"""
import os, sys, time, csv
import pytse_client as tse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "tse_prices")
os.makedirs(OUT, exist_ok=True)


def panel_symbols():
    with open(os.path.join(ROOT, "data", "dataset_model.csv"), encoding="utf-8-sig", newline="") as f:
        return sorted({r["symbol"].strip() for r in csv.DictReader(f) if r.get("symbol")})


def main():
    syms = panel_symbols(); status_fn = os.path.join(OUT, "_status.csv")
    done = set()
    if os.path.exists(status_fn):
        with open(status_fn, encoding="utf-8") as f: done = {r["symbol"] for r in csv.DictReader(f) if r["status"] in ("ok", "not_found")}
    new = not os.path.exists(status_fn)
    with open(status_fn, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "status", "rows", "error"])
        if new: w.writeheader()
        todo = [s for s in syms if s not in done]
        print(f"{len(todo)} symbols to download ({len(done)} done)", flush=True)
        for n, s in enumerate(todo, 1):
            rec = dict(symbol=s, status="error", rows=0, error="")
            for attempt in range(3):
                try:
                    res = tse.download(symbols=[s], write_to_csv=True, include_jdate=True, base_path=OUT, adjust=True)
                    df = res.get(s)
                    rec.update(status="ok", rows=0 if df is None else len(df), error=""); break
                except Exception as e:
                    msg = str(e)[:200]; rec["error"] = msg
                    if "Cannot find symbol" in msg: rec["status"] = "not_found"; break
                    time.sleep(10 * (attempt + 1))
            w.writerow(rec); f.flush()
            if n % 25 == 0: print(f"{n}/{len(todo)}", flush=True)
            time.sleep(2)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
