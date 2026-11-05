"""Volume indices from the year-end Codal monthly activity reports (production firms).

For each firm-year the report of the last month of the fiscal year holds, per product, the cumulative production
and sales of the year (group 'from the start of the fiscal year to <period end>') and, from report version V2 on,
the same figures for the same period of the previous year. The volume index of year t over t-1 is a Laspeyres
quantity index with previous-year unit prices:
    I_t = sum_p q_{t,p} c_{t-1,p} / sum_p q_{t-1,p} c_{t-1,p}
computed for production (main) and for sales quantities. When the previous-year group is missing (V1 reports),
the previous year's report is used, matching products by name.
Output: data/codal_monthly/volume_index.csv  (symbol, fy, prod_index, sales_index, n_products, source)
Usage: python parse_monthly.py <json files...>
"""
import json, re, sys, collections, math
import pandas as pd

DIG = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2)


def norm(s):
    s = str(s or "").translate(DIG).replace("ي", "ی").replace("ك", "ک").replace("‌", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def num(s):
    s = norm(s).replace(",", "")
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def table_rows(rec):
    tabs = [t for t in rec.get("tables", []) if "تولید" in norm(t["title"]) or "محصولات" in norm(t["title"])]
    if not tabs: return None
    rows = collections.defaultdict(dict)
    for r, c, v in tabs[0]["cells"]: rows[r][c] = v
    return rows


def groups(rows):
    """Start column -> header label, from the first header row."""
    first = min(rows)
    return {c: norm(v) for c, v in rows[first].items()}


def find_group(gr, year, month):
    tag = f"{year}/{month:02d}"
    for c, lab in sorted(gr.items()):
        if lab.startswith("از ابتدای سال") and tag in lab and "اصلاح" not in lab:
            last = c
    return locals().get("last")


def products(rows, col):
    """name -> (production, sales qty, unit price, amount) for one column group starting at `col`."""
    out = {}
    hdr = sorted(rows)[:2]
    for r in sorted(rows)[2:]:
        row = rows[r]; name = norm(row.get(1, ""))
        unit = norm(row.get(2, ""))
        if not name or not unit or name.startswith("جمع") or "برگشت" in name or "تخفیف" in name: continue
        prod, sale, price, amt = (num(row.get(col + k)) for k in range(4))
        if prod is None and sale is None: continue
        key = f"{name}|{unit}"
        p = out.get(key, [0.0, 0.0, None, 0.0])
        p[0] += prod or 0; p[1] += sale or 0; p[3] += amt or 0
        if price: p[2] = price
        out[key] = p
    return out


def laspeyres(cur, prev, k):
    num_, den = 0.0, 0.0
    for key, pv in prev.items():
        cv = cur.get(key)
        if cv is None: continue
        w = pv[2] if pv[2] and pv[2] > 0 else (pv[3] / pv[1] if pv[1] else None)
        if not w or pv[k] <= 0: continue
        num_ += cv[k] * w; den += pv[k] * w
    return (num_ / den if den > 0 else math.nan), den


def main(files):
    recs = {}
    for f in files:
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("status") == "ok": recs[r["key"]] = r
    cur_tab, prev_tab = {}, {}
    for key, r in recs.items():
        rows = table_rows(r)
        if not rows: continue
        y, m = (int(x) for x in norm(r.get("period_end") or r["title"].split()[-1]).split("/")[:2])
        gr = groups(rows)
        gc, gp = find_group(gr, y, m), find_group(gr, y - 1, m)
        if gc: cur_tab[(r["symbol"], r["fy"])] = products(rows, gc)
        if gp: prev_tab[(r["symbol"], r["fy"])] = products(rows, gp)
    out = []
    for (s, fy), cur in cur_tab.items():
        prev, src = prev_tab.get((s, fy)), "same_report"
        if not prev:
            prev, src = cur_tab.get((s, fy - 1)), "previous_report"
        if not prev: continue
        pi, pw = laspeyres(cur, prev, 0); si, sw = laspeyres(cur, prev, 1)
        out.append(dict(symbol=s, fy=fy, prod_index=pi, sales_index=si, n_products=len(cur), source=src))
    df = pd.DataFrame(out).sort_values(["symbol", "fy"])
    df.to_csv("volume_index.csv", index=False)
    return df


if __name__ == "__main__":
    d = main(sys.argv[1:])
    print(len(d), d.source.value_counts().to_dict())
    print(d[["prod_index", "sales_index"]].describe().round(3))
