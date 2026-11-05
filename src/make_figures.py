"""Figures for the manuscript (print quality, 600 dpi PNG + vector PDF)."""
import os, re, ast, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from scipy.stats import rankdata, studentized_range

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#9a9994"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm",
                     "font.family": "serif", "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "savefig.facecolor": SURF, "axes.facecolor": SURF, "lines.linewidth": 1.6})

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=600, bbox_inches="tight")
    plt.close(fig)

def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# ---------------------------------------------------------------- Fig. 1 framework
def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 3.5)); ax.set_xlim(0, 104); ax.set_ylim(0, 48); ax.axis("off")
    def box(x, y, w, h, title, body, col):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=col + "1f", ec=col, lw=1.0))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=7.6, fontfamily="cmb10", color=INK, linespacing=1.15)
        ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top", fontsize=6.0, color=INK2, linespacing=1.4)
    def arrow(x0, y0, x1, y1, txt=None, side="top"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
        if txt:
            if side == "top": ax.text((x0 + x1) / 2, max(y0, y1) + 1.0, txt, ha="center", va="bottom", fontsize=6.6, color=INK2)
            else: ax.text(max(x0, x1) + 1.0, (y0 + y1) / 2, txt, ha="left", va="center", fontsize=6.6, color=INK2)
    box(1, 27, 21, 19, "(a) Data", "Financial statements,\nyear $t$ (Codal / public)\nindustry, capacity\nlabel: distress at $t$+1", C[2])
    box(27, 27, 22, 19, "(b) Calibrated base", "LightGBM, 5-fold\ncross-fitting\nPlatt map $\\rightarrow s_0(x)$\n(Section 5.2)", C[0])
    box(54, 27, 23, 19, "(c) Decision-focused\ncorrection", "\nboosted trees $F(x)$\nscore-function gradient\nof realised M2 cost\n(Sections 5.3 to 5.6)", C[1])
    box(82, 27, 21, 19, "(d) Selection", "validation pools\nexact M2 regret\n$\\hat R\\in\\{0,25,\\ldots,150\\}$\n(Section 5.7)", C[6])
    box(27, 2, 22, 19, "Training pools", "$N=20$ suppliers per pool\nprices (Eq. 5), capacities\ncommon random numbers\nrealised outcomes $\\xi$", MUTED)
    box(54, 2, 23, 19, "Solution cache", "exact M2 solutions $\\mathcal{K}_k$\nclosed-form evaluation\n(Lemma 1, Eq. 22)\nexact solve w.p. 0.01", MUTED)
    box(82, 2, 21, 19, "(e) Decision", "$\\hat p=\\sigma(s_0+F_{\\hat R})$\nM2: select suppliers,\nallocate orders,\nreserve backup", C[5])
    arrow(22.6, 36.5, 26.4, 36.5); arrow(49.6, 36.5, 53.4, 36.5, "$s_0$"); arrow(77.6, 36.5, 81.4, 36.5, "$F_R$")
    arrow(38, 21.6, 38, 26.4); arrow(49.6, 11.5, 53.4, 11.5); arrow(65.5, 21.6, 65.5, 26.4, "$\\hat g$", side="right")
    arrow(92.5, 26.4, 92.5, 21.6, "$\\hat R$", side="right")
    save(fig, "fig1_framework")

# ---------------------------------------------------------------- Fig. 2 Theorem 1 geometry
def fig2():
    a_i, a_j, b_i, b_j, ti, tj = 1.00, 1.03, 0.80, 0.70, 0.25, 0.30
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True); fig.subplots_adjust(wspace=0.12)
    g = np.linspace(0, 0.6, 400); P, Q = np.meshgrid(g, g)
    bayes = (a_i + b_i * P) < (a_j + b_j * Q)            # True: supplier i optimal
    for k, ax in enumerate(axs):
        clean(ax); ax.grid(False)
        ax.contourf(P, Q, bayes, levels=[-0.5, 0.5, 1.5], colors=[C[1] + "33", C[0] + "33"])
        pj = (a_i - a_j + b_i * g) / b_j
        ax.plot(g, pj, color=INK, lw=1.6)
        ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.6); ax.set_xlabel("$p_i$ (distress probability of supplier $i$)")
        if k == 0: ax.set_ylabel("$p_j$ (distress probability of supplier $j$)")
        ax.text(0.03, 0.53, "Bayes: order from $i$", color=INK, fontsize=7.5, bbox=dict(fc=SURF, ec="none", pad=1.2, alpha=0.85))
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5] + ([0.6] if k == 1 else []))
        ax.text(0.585, 0.04, "Bayes: order from $j$", color=INK, fontsize=7.5, ha="right")
        ax.text(0.40, 0.418, r"$a_i+\beta_ip_i=a_j+\beta_jp_j$", fontsize=6.8, color=INK2, ha="center", va="bottom", rotation=48.8,
                transform_rotates_text=True, rotation_mode="anchor")
    ax = axs[0]; ax.set_title("(a) Bayes decision regions", fontsize=8.5, loc="left", color=INK)
    ax = axs[1]; ax.set_title("(b) Any label-based rule", fontsize=8.5, loc="left", color=INK)
    ax.axvline(ti, color=INK2, lw=1.0, ls=(0, (4, 2))); ax.axhline(tj, color=INK2, lw=1.0, ls=(0, (4, 2)))
    ax.add_patch(Polygon([[0, 0], [ti, 0], [ti, tj], [0, tj]], closed=True, fill=False, hatch="////", ec=INK2, lw=0))
    ax.text(ti / 2, tj / 2, "label cell $\\mathcal{C}$\n(both 'safe')\none decision", ha="center", va="center", fontsize=6.6,
            color=INK, bbox=dict(fc=SURF, ec="none", pad=1.5))
    ax.text(ti + 0.005, 0.585, r"$\tau_i$", fontsize=8, color=INK2); ax.text(0.585, tj + 0.008, r"$\tau_j$", fontsize=8, color=INK2, ha="right")
    for (x, y, lab) in [(0.02, tj - 0.03, "$p'$"), (ti - 0.03, 0.02, "$p''$")]:
        ax.plot(x, y, "o", ms=5, mfc=INK, mec=SURF, mew=1.2); ax.text(x + 0.014, y + 0.012, lab, fontsize=8, bbox=dict(fc=SURF, ec="none", pad=0.6))
    save(fig, "fig2_theorem1")


# ---------------------------------------------------------------- Flowchart of training and selection
def fig_flowchart():
    fig, ax = plt.subplots(figsize=(7.2, 8.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 118); ax.axis("off")
    def node(x, y, w, h, txt, kind="proc", col=C[0], fs=6.4):
        if kind == "term":
            p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={h/2}", fc=INK2, ec=INK2, lw=0.9)
            ax.add_patch(p); ax.text(x, y, txt, ha="center", va="center", fontsize=fs + 0.4, color=SURF, fontfamily="cmb10"); return
        if kind == "dec":
            ax.add_patch(Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2], [x - w / 2, y]], closed=True, fc=C[3] + "2a", ec=C[3], lw=0.9))
        else:
            ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=col + "1c", ec=col, lw=0.9))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.3)
    def arr(pts, lab=None, lpos=None):
        for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
            ax.plot([x0, x1], [y0, y1], color=INK2, lw=0.9, solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
        if lab: ax.text(*lpos, lab, fontsize=6.3, color=INK, ha="center", va="center", fontfamily="cmb10")
    A, B, R = 19, 60, 88
    # column A: preparation
    node(A, 112, 22, 5, "Start", "term")
    node(A, 101, 34, 10, "Training years: features $x$,\nlabels $y$ (distress at $t$+1)", col=C[2])
    node(A, 87, 34, 11, "Cross-fitted LightGBM (5 folds)\n+ Platt map $\\rightarrow$ base score $s_0(x)$", col=C[0])
    node(A, 71, 34, 14, "Build 60 training and 15\nvalidation pools ($N$=20, $S$=100);\nseed cache $\\mathcal{K}_k$ with 4 exact\nM2 solutions per pool", col=C[0])
    node(A, 57, 34, 7, "$F\\leftarrow 0$,  $r\\leftarrow 0$", col=C[0])
    for y0, y1 in ((109.5, 106), (96, 92.5), (81.5, 78), (64, 60.5)): arr([(A, y0), (A, y1)])
    # column B: one boosting round
    node(B, 109, 36, 10, "For each training pool $k$: draw $M$=8\nantithetic $Z_m$;  $p_m=\\sigma(s+\\varepsilon Z_m)$", col=C[1])
    arr([(A + 17, 57), (A + 20, 57), (A + 20, 109), (B - 18, 109)])
    node(B, 94, 24, 11, "$u<0.01$ ?", "dec")
    arr([(B, 104), (B, 99.5)])
    node(R, 94, 16, 13, "Solve M2\nexactly; add\n$z$ to cache $\\mathcal{K}_k$", col=C[6])
    arr([(B + 12, 94), (R - 8, 94)], "yes", (B + 15.5, 96))
    node(B, 78, 36, 11, "Pick $z\\in\\mathcal{K}_k$ with the lowest SAA\nmean-CVaR under $p_m$ (Lemma 1)", col=C[1])
    arr([(B, 88.5), (B, 83.5)], "no", (B + 3, 86.5))
    arr([(R, 87.5), (R, 78), (B + 18, 78)])
    node(B, 64, 36, 9, "Realised cost $L_m=C(z;\\xi^{\\mathrm{real}})$\n(Lemma 1, closed form)", col=C[1])
    arr([(B, 72.5), (B, 68.5)])
    node(B, 50, 36, 11, "Score-function gradient $\\hat g$;\n$G=\\kappa\\hat g+\\mu(p-y)$,  Hessian $H$", col=C[1])
    arr([(B, 59.5), (B, 55.5)])
    node(B, 36, 36, 9, "Fit 4-leaf tree $f_r$ on $(G,H)$;\n$F\\leftarrow F+\\nu f_r$,  $r\\leftarrow r+1$", col=C[1])
    arr([(B, 44.5), (B, 40.5)])
    node(B, 21, 24, 11, "$r<R_{\\max}$ ?", "dec")
    arr([(B, 31.5), (B, 26.5)])
    arr([(B + 12, 21), (98.5, 21), (98.5, 112), (B + 18, 112)], "yes", (B + 15.5, 23))
    # column A lower: selection
    node(A, 38, 34, 12, "Validation pools: exact M2\nregret of $\\sigma(s_0+F_R)$ for\n$R\\in\\{0,25,\\ldots,150\\}$", col=C[5])
    arr([(B - 12, 21), (A + 20, 21), (A + 20, 38), (A + 17, 38)], "no", (B - 15.5, 23))
    node(A, 25, 34, 7, "$\\hat R=\\mathrm{arg\\,min}_R$ validation regret", col=C[5])
    arr([(A, 32), (A, 28.5)])
    node(A, 14, 34, 7, "Output $\\hat p=\\sigma(s_0+F_{\\hat R})\\rightarrow$ M2", col=C[5])
    arr([(A, 21.5), (A, 17.5)])
    node(A, 3.5, 22, 5, "End", "term")
    arr([(A, 10.5), (A, 6)])
    ax.text(B, 117, "one boosting round", fontsize=6.6, color=INK2, ha="center", style="italic")
    save(fig, "fig3_flowchart")

# ---------------------------------------------------------------- data
D = pd.read_csv(os.path.join(RES, "decisions_all2.csv")); D["key"] = D.model + "|" + D.rule
MAIN = [("DFM2|PI-M2", "DF-Distress"), ("CAL-BASE|PI-M2", "CAL-BASE"), ("LGBM-Focal|PI-M2", "LGBM-Focal"),
        ("LGBM|PI-M2", "LGBM"), ("XGB|PI-M2", "XGB"), ("CE-L15|PI-M2", "CE-L15"), ("DFD-L15|PI-M2", "DF-Distress-M1"),
        ("LR|PI-M2", "LR"), ("SPO+-L4|PI-M2", "SPO+"), ("LGBM|LB-M2", "LGBM + label screening"), ("NAIVE|M2", "Risk-blind")]

# ---------------------------------------------------------------- Fig. 3 means with bootstrap CIs
def fig3():
    rng = np.random.default_rng(0); dsn = [("IR", "Iran"), ("US", "United States"), ("PL", "Poland"), ("TW", "Taiwan")]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 3.4), sharey=True)
    for ax, (ds, name) in zip(axs, dsn):
        clean(ax); ax.grid(axis="y", visible=False)
        for r, (k, lab) in enumerate(MAIN):
            x = D[(D.dataset == ds) & (D.key == k)].nregret.values * 100
            bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
            col = C[0] if k.startswith("DFM2") else (C[1] if "LB-M2" in k else GRAY)
            ax.plot([lo, hi], [r, r], color=col, lw=1.6, solid_capstyle="round")
            ax.plot(x.mean(), r, "o", ms=5, mfc=col, mec=SURF, mew=1.2)
        ax.set_title(name, fontsize=8.5, color=INK); ax.set_xlim(left=0)
        ax.set_xlabel("Normalised regret (%)")
    axs[0].set_yticks(range(len(MAIN))); axs[0].set_yticklabels([l for _, l in MAIN]); axs[0].invert_yaxis()
    save(fig, "fig4_regret_ci")

# ---------------------------------------------------------------- Fig. 4 CD diagram
def fig4():
    keys = ['DFM2|PI-M2', 'CAL-BASE|PI-M2', 'LGBM-Focal|PI-M2', 'LGBM|PI-M2', 'XGB|PI-M2', 'XGB-CS|PI-M2', 'LR|PI-M2',
            'LR-CS|PI-M2', 'CE-L15|PI-M2', 'DFD-L15|PI-M2', 'SPO+-L4|PI-M2', 'DBB-L4|PI-M2', 'PFYL-L4|PI-M2', 'NAIVE|M2']
    names = {'DFM2|PI-M2': 'DF-Distress', 'CAL-BASE|PI-M2': 'CAL-BASE', 'LGBM-Focal|PI-M2': 'LGBM-Focal', 'LGBM|PI-M2': 'LGBM',
             'XGB|PI-M2': 'XGB', 'XGB-CS|PI-M2': 'XGB-CS', 'LR|PI-M2': 'LR', 'LR-CS|PI-M2': 'LR-CS', 'CE-L15|PI-M2': 'CE-L15',
             'DFD-L15|PI-M2': 'DF-Distress-M1', 'SPO+-L4|PI-M2': 'SPO+', 'DBB-L4|PI-M2': 'DBB', 'PFYL-L4|PI-M2': 'PFYL', 'NAIVE|M2': 'Risk-blind'}
    P = D.pivot_table(index=["dataset", "fold"], columns="key", values="nregret", aggfunc="mean")[keys].dropna()
    R = np.array([rankdata(r) for r in P.values]).mean(0); order = np.argsort(R); k, n = len(keys), len(P)
    cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); ax.axis("off"); lo, hi = 1, 14
    X = lambda r: 10 + (r - lo) / (hi - lo) * 80
    ax.set_xlim(0, 100); ax.set_ylim(-10, 12)
    ax.plot([X(lo), X(hi)], [8, 8], color=INK2, lw=0.8)
    for t in range(lo, hi + 1):
        ax.plot([X(t), X(t)], [8, 8.6], color=INK2, lw=0.8); ax.text(X(t), 9.2, str(t), ha="center", fontsize=7, color=INK2)
    ax.plot([X(lo), X(lo + cd)], [11, 11], color=INK, lw=1.2); ax.text(X(lo + cd / 2), 11.4, f"CD = {cd:.2f}", ha="center", fontsize=7)
    half = int(np.ceil(k / 2))
    for idx, j in enumerate(order):
        r = R[j]; col = C[0] if keys[j].startswith("DFM2") else INK2
        if idx < half:
            y = 6 - idx * 2.0; ax.plot([X(r), X(r), 8], [8, y, y], color=col, lw=0.8)
            ax.text(7.5, y, f"{names[keys[j]]} ({r:.2f})", ha="right", va="center", fontsize=7, color=INK, fontfamily="cmb10" if col == C[0] else None)
        else:
            y = 6 - (k - 1 - idx) * 2.0; ax.plot([X(r), X(r), 92], [8, y, y], color=col, lw=0.8)
            ax.text(92.5, y, f"({r:.2f}) {names[keys[j]]}", ha="left", va="center", fontsize=7, color=INK)
    # cliques: maximal groups within CD
    sr = np.sort(R); groups = []
    for i in range(k):
        jx = max(j for j in range(k) if sr[j] - sr[i] <= cd)
        if jx > i and not any(g[0] <= i and g[1] >= jx for g in groups): groups.append((i, jx))
    for gi, (i, j) in enumerate(groups):
        y = 7.2 - 0.55 * gi; ax.plot([X(sr[i]) - 0.4, X(sr[j]) + 0.4], [y, y], color=INK, lw=2.2, solid_capstyle="round")
    save(fig, "fig5_cd_diagram"); return cd

# ---------------------------------------------------------------- Fig. 5 & 6 sensitivity
S = pd.concat([pd.read_csv(os.path.join(RES, f"sens_{n}.csv")) for n in ["IR", "US", "PL", "TW"]]); S["key"] = S.model + "|" + S.rule
SENS = [("DFM2|PI-M2", "DF-Distress", C[0], "o"), ("CAL-BASE|PI-M2", "CAL-BASE", C[2], "s"), ("LGBM|PI-M2", "LGBM", C[1], "^"),
        ("LGBM-Focal|PI-M2", "LGBM-Focal", C[4], "D"), ("XGB|PI-M2", "XGB", C[3], "v")]
def sens_fig(configs, xs, xlabel, name, letter_b):
    m = (S.groupby(["config", "key"]).nregret.mean() * 100).unstack()
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axs[0]; clean(ax)
    for k, lab, col, mk in SENS:
        y = [m.loc[c, k] for c in configs]; ax.plot(xs, y, color=col, marker=mk, ms=5, mec=SURF, mew=1.0, label=lab)
    ax.set_xlabel(xlabel); ax.set_ylabel("Normalised regret (%)"); ax.set_xticks(xs); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) Regret", fontsize=8.5, loc="left")
    ax = axs[1]; clean(ax)
    red = [(1 - m.loc[c, "DFM2|PI-M2"] / m.loc[c, "LGBM|PI-M2"]) * 100 for c in configs]
    ax.bar(xs, red, width=(xs[-1] - xs[0]) / (len(xs) * 3.2), color=C[0])
    for x, r in zip(xs, red): ax.text(x, r + 0.6, f"{r:.1f}%", ha="center", fontsize=7, color=INK)
    ax.set_xlabel(xlabel); ax.set_ylabel("Regret reduction vs. LGBM (%)"); ax.set_xticks(xs); ax.set_ylim(0, max(red) * 1.25)
    ax.set_title(letter_b, fontsize=8.5, loc="left"); ax.grid(axis="x", visible=False)
    save(fig, name)

# ---------------------------------------------------------------- Fig. 7 SAA
def fig7():
    A = pd.read_csv(os.path.join(RES, "saa_IR.csv")); ref = A.groupby(["fold", "pool"]).obj_oos.transform("min")
    A["gap"] = (A.obj_oos - ref) / ref * 100; g = A.groupby("S").agg(gap=("gap", "mean"), gse=("gap", "sem"), t=("secs", "mean"))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axs[0]; clean(ax); ax.errorbar(g.index, g.gap, yerr=1.96 * g.gse, color=C[0], marker="o", ms=5, mec=SURF, capsize=2, lw=1.6)
    ax.axvline(100, color=MUTED, lw=0.8, ls=(0, (3, 2))); ax.text(104, g.gap.max() * 0.92, "S = 100 (used)", fontsize=7, color=INK2)
    ax.set_xscale("log"); ax.set_xticks(g.index); ax.set_xticklabels(g.index); ax.set_xlabel("Number of scenarios S")
    ax.set_ylabel("Out-of-sample optimality gap (%)"); ax.set_title("(a) Solution quality", fontsize=8.5, loc="left")
    ax = axs[1]; clean(ax); ax.plot(g.index, g.t, color=C[1], marker="s", ms=5, mec=SURF)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(g.index); ax.set_xticklabels(g.index)
    ax.set_xlabel("Number of scenarios S"); ax.set_ylabel("M2 solve time (s)"); ax.set_title("(b) Computation", fontsize=8.5, loc="left")
    save(fig, "fig7_saa")

# ---------------------------------------------------------------- Fig. 8 validation curves (Iran)
def fig8():
    txt = open(os.path.join(RES, "log2_IR.txt"), encoding="utf-8").read()
    curves = re.findall(r"IR (\d{4}) DFM2 \d+s val% (\{.*?\}) -> rounds (\d+)", txt)
    fig, ax = plt.subplots(figsize=(4.8, 3.0)); clean(ax)
    ends = []
    for i, (yr, d, best) in enumerate(curves):
        d = ast.literal_eval(d); r = sorted(d); v = [d[x] / d[0] for x in r]
        ax.plot(r, v, color=C[0], alpha=0.35 + 0.09 * i, lw=1.3)
        ax.plot(int(best), d[int(best)] / d[0], "o", ms=4.5, mfc=C[0], mec=SURF, mew=1.0, zorder=3)
        ends.append([v[-1], yr])
    ends.sort(); gap = 0.028                                  # de-overlap end labels
    for k in range(1, len(ends)):
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
    for y, yr in ends: ax.text(153, y, yr, fontsize=6.5, va="center", color=INK2)
    ax.axhline(1, color=MUTED, lw=0.8)
    ax.plot([], [], "o", ms=4.5, mfc=C[0], mec=SURF, label=r"selected $\hat R$ (minimum validation regret)")
    ax.legend(loc="lower left", fontsize=6.5, frameon=False)
    ax.set_xlabel("Correction rounds $R$"); ax.set_ylabel("Validation regret relative to $R=0$")
    ax.set_xlim(0, 165); ax.set_xticks(range(0, 151, 25))
    save(fig, "fig6_validation_curves")

if __name__ == "__main__":
    fig1(); fig2(); fig_flowchart(); fig3(); cd = fig4()
    sens_fig(["lambda=0", "base", "lambda=0.6"], [0.0, 0.3, 0.6], r"CVaR weight $\lambda$", "fig8_sensitivity_lambda", "(b) Gain of DF-Distress")
    sens_fig(["kappa_p=0", "kappa_p=0.05", "base", "kappa_p=0.20"], [0.0, 0.05, 0.10, 0.20], r"Price-risk slope $\kappa_p$",
             "fig9_sensitivity_kappa", "(b) Gain of DF-Distress")
    fig7(); fig8(); print("CD", round(cd, 2), "figures in", FIG)
