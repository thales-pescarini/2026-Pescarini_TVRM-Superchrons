# Fig. 4: Raman spectroscopy of carbonaceous material (Section 3.2).
# Linear background on 1000-1800 cm-1, four bands (D1, G, D2 Lorentzian; D3 Gaussian),
# R1 = D1/G peak height, R2 = A(D1)/[A(D1)+A(D2)+A(G)], T from Rahl et al. (2005).
# Writes Fig4.pdf/.png and the per-spectrum and per-sample tables.
# Usage: python figures/fig4_rscm.py [spectra directory]
import sys, os, glob
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams

DATA_DIR     = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "raman_spectra")
OUT_FIG      = "Fig4"
OUT_MEAS     = "nama_rscm_per_measurement.csv"
OUT_SAMP     = "nama_rscm_per_sample.csv"
FIT_LO, FIT_HI = 1000.0, 1800.0
CAL_ERR      = 50.0
SPECTRA_GLOB = "*.txt"

D1_WIN   = (1335.0, 1362.0)
D3_WIN   = (1480.0, 1560.0)
G_WIN    = (1575.0, 1602.0)
D2_WIN   = (1605.0, 1635.0)

def T_rahl(R1, R2):    return 737.3 + 320.9*R1 - 1067.0*R2 - 80.638*R1**2

def read_spectrum(path):
    first = open(path).readline()
    delim = "," if "," in first else None
    skip = 0
    if not first.lstrip().startswith("#"):
        parts = first.replace(",", " ").split()
        try:
            float(parts[0]); float(parts[1])
        except Exception:
            skip = 1
    arr = np.genfromtxt(path, delimiter=delim, comments="#", skip_header=skip)
    x, y = arr[:, 0].astype(float), arr[:, 1].astype(float)
    o = np.argsort(x)
    return x[o], y[o]

def sample_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.rsplit("_", 1)[0]

def lorentz(A, c, w, x):  return A / (1.0 + ((x - c)/w)**2)
def gauss(A, c, w, x):    return A * np.exp(-np.log(2.0) * ((x - c)/w)**2)

def model(p, x):
    A1, c1, w1, A3, c3, w3, Ag, cg, wg, A2, c2, w2 = p[:12]
    y = (lorentz(A1, c1, w1, x) + gauss(A3, c3, w3, x)
         + lorentz(Ag, cg, wg, x) + lorentz(A2, c2, w2, x))
    return y

def baseline(xf, yf):
    y0 = np.median(yf[xf <= FIT_LO + 12]); y1 = np.median(yf[xf >= FIT_HI - 12])
    x0 = xf[xf <= FIT_LO + 12].mean();     x1 = xf[xf >= FIT_HI - 12].mean()
    return y0 + (y1 - y0) * (xf - x0) / (x1 - x0)

def fit_spectrum(x, y):
    m = (x >= FIT_LO) & (x <= FIT_HI)
    xf, yf = x[m], y[m]
    yc = yf - baseline(xf, yf)
    A1g = max(yc[(xf > 1320) & (xf < 1380)].max(), 1.0)
    Agg = max(yc[(xf > 1560) & (xf < 1620)].max(), 1.0)
    c1g = xf[(xf > 1320) & (xf < 1380)][np.argmax(yc[(xf > 1320) & (xf < 1380)])]
    cgg = xf[(xf > 1560) & (xf < 1620)][np.argmax(yc[(xf > 1560) & (xf < 1620)])]
    p0 = [A1g, c1g, 28, 0.35*A1g, 1510, 60, Agg, min(max(cgg, 1577), 1600), 20,
          0.4*Agg, 1618, 13]
    lb = [0, D1_WIN[0], 8,  0, D3_WIN[0], 25, 0, G_WIN[0], 6,  0, D2_WIN[0], 5]
    ub = [3*A1g, D1_WIN[1], 95, 0.7*A1g, D3_WIN[1], 110,
          2.5*Agg, G_WIN[1], 50, 1.5*Agg, D2_WIN[1], 30]
    p0 = np.clip(p0, lb, ub)
    res = least_squares(lambda p: model(p, xf) - yc, p0, bounds=(lb, ub), max_nfev=30000)
    p = res.x
    A1, c1, w1, A3, c3, w3, Ag, cg, wg, A2, c2, w2 = p[:12]
    aD1 = np.pi*A1*w1; aG = np.pi*Ag*wg; aD2 = np.pi*A2*w2
    R1 = A1/Ag; R2 = aD1/(aD1 + aG + aD2)
    ss = np.sum((model(p, xf) - yc)**2); tot = np.sum((yc - yc.mean())**2)
    out = dict(R1=R1, R2=R2, T_rahl=T_rahl(R1, R2),
               D1_pos=c1, D1_fwhm=2*w1, D3_pos=c3, D3_fwhm=2*w3,
               G_pos=cg, G_fwhm=2*wg, D2_pos=c2, D2_fwhm=2*w2,
               r2fit=1.0 - ss/tot)
    return out, (xf, yc, p)

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, SPECTRA_GLOB)))
    if not files:
        sys.exit(f"No spectra ({SPECTRA_GLOB}) in '{DATA_DIR}'.")
    rows = []
    for f in files:
        x, y = read_spectrum(f)
        r, _ = fit_spectrum(x, y)
        r["file"] = os.path.basename(f); r["sample"] = sample_name(f)
        rows.append(r)
    meas = pd.DataFrame(rows)
    cols = ["sample","file","R1","R2","T_rahl",
            "D1_pos","D1_fwhm","D3_pos","D3_fwhm","G_pos","G_fwhm","D2_pos","D2_fwhm","r2fit"]
    meas = meas[cols].sort_values(["sample","file"]).reset_index(drop=True)
    meas.to_csv(OUT_MEAS, index=False, float_format="%.3f")

    g = meas.groupby("sample")
    samp = pd.DataFrame({"n": g.size(),
                         "T_rahl_mean": g["T_rahl"].mean(),
                         "T_rahl_sd":   g["T_rahl"].std(ddof=1),
                         "R1_mean": g["R1"].mean(), "R2_mean": g["R2"].mean()}).reset_index()
    samp.to_csv(OUT_SAMP, index=False, float_format="%.2f")

    allT = meas["T_rahl"].values
    sd = allT.std(ddof=1)
    print(f"{samp.shape[0]} samples, {len(allT)} spectra, {samp['n'].min()}-{samp['n'].max()} spectra per sample")
    print(f"Series: mean {allT.mean():.1f} C, 2 sd {2*sd:.1f} C, range {allT.min():.1f}-{allT.max():.1f} C")
    print(f"Mean fit R^2: {meas['r2fit'].mean():.4f}, minimum {meas['r2fit'].min():.4f}")
    print("\nPer sample:")
    for _, r in samp.iterrows():
        s = "   --" if r['n'] == 1 else f"{2*r['T_rahl_sd']:5.1f}"
        print(f"  {r['sample']:<8} n={int(r['n'])}  T = {r['T_rahl_mean']:6.1f}  2 sd = {s}")

    # example spectrum: sample closest to the series mean
    s_mean = meas["T_rahl"].mean()
    s_rep = samp.iloc[(samp["T_rahl_mean"] - s_mean).abs().argmin()]["sample"]
    sub = meas[meas["sample"] == s_rep]
    rep_file = sub.iloc[(sub["T_rahl"] - sub["T_rahl"].mean()).abs().argmin()]["file"]
    ex_path = os.path.join(DATA_DIR, rep_file)
    xex, yex = read_spectrum(ex_path)
    ex_res, ex_fit = fit_spectrum(xex, yex)
    make_figure(meas, samp, allT, ex_fit, ex_res, sample_name(ex_path))
    print(f"\nSaved: {OUT_FIG}.pdf/.png, {OUT_MEAS}, {OUT_SAMP}")

def make_figure(meas, samp, allT, ex_fit, ex_res, ex_name):
    rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                     "axes.linewidth":0.8,"mathtext.default":"regular"})
    xf, yc, p = ex_fit
    A1,c1,w1,A3,c3,w3,Ag,cg,wg,A2,c2,w2 = p[:12]
    xx = np.linspace(FIT_LO, FIT_HI, 1600)
    vir = [cm.viridis(t) for t in (0.0, 0.35, 0.62, 0.88)]      # D1, D3, G, D2
    bands = [("D1",lorentz,(A1,c1,w1),vir[0]), ("D3",gauss,(A3,c3,w3),vir[1]),
             ("G", lorentz,(Ag,cg,wg),vir[2]), ("D2",lorentz,(A2,c2,w2),vir[3])]

    fig = plt.figure(figsize=(8.5, 3.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[4,1],
                          hspace=0.07, wspace=0.26, left=0.075, right=0.985,
                          top=0.93, bottom=0.14)
    axA = fig.add_subplot(gs[0,0]); axR = fig.add_subplot(gs[1,0], sharex=axA)
    axB = fig.add_subplot(gs[:,1])

    axA.plot(xf, yc, ".", ms=1.6, color="0.55", label="data (532 nm)")
    axA.plot(xx, model(p, xx), "-", color="k", lw=1.3, label="fitted spectrum")
    for nm,f,pp,col in bands:
        axA.fill_between(xx, f(*pp,xx), color=col, alpha=.18, lw=0)
        axA.plot(xx, f(*pp,xx), color=col, lw=0.9)
        cc = pp[1]
        axA.annotate(nm, (cc, f(*pp, np.array([cc]))[0]), xytext=(0,3),
                     textcoords="offset points", ha="center", fontsize=9.5,
                     color=col, fontweight="bold")
    axA.set_ylabel("Intensity (baseline-corr.) [a.u.]", fontsize=8.5)
    axA.tick_params(labelbottom=False); axA.set_xlim(FIT_LO, FIT_HI)
    axA.set_title(f"(a) Representative fit — sample {ex_name}", fontsize=9, loc="left")
    axA.text(0.975, 0.93, f"$R_1$ = {ex_res['R1']:.2f}\n$R_2$ = {ex_res['R2']:.3f}\n"
             f"$R^2$ = {ex_res['r2fit']:.3f}", transform=axA.transAxes, ha="right",
             va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.6))
    axA.legend(fontsize=7.5, loc="upper left", frameon=False, handlelength=1.3)
    axR.plot(xf, yc - model(p, xf), "-", color="0.4", lw=0.5)
    axR.axhline(0, color="k", lw=0.5, ls=":")
    axR.set_xlim(FIT_LO, FIT_HI); axR.set_xlabel("Raman shift [cm$^{-1}$]", fontsize=8.5)
    axR.set_ylabel("res.", fontsize=7.5); axR.tick_params(labelsize=7.5)

    sp = samp.sort_values("T_rahl_mean", ascending=False).reset_index(drop=True)
    mu = allT.mean(); sdser = allT.std(ddof=1); xpos = np.arange(sp.shape[0])
    axB.axhspan(mu-CAL_ERR, mu+CAL_ERR, color="#d62728", alpha=0.05, zorder=0)
    axB.axhline(mu, color="#d62728", lw=0.8, ls="--", alpha=0.6, zorder=1)
    for i, s in enumerate(sp["sample"]):
        ts = meas.loc[meas["sample"] == s, "T_rahl"].values
        axB.plot(np.full_like(ts, xpos[i]), ts, "o", ms=3.5, mfc="0.6", mec="none", zorder=3)
    yerr = 2*sp["T_rahl_sd"].fillna(0.0).values
    axB.errorbar(xpos, sp["T_rahl_mean"], yerr=yerr, fmt="o", ms=7, mfc="#d62728",
                 mec="k", ecolor="0.3", elinewidth=1.1, capsize=3, zorder=4)
    lab = [f"{s} (n={int(n)})" for s, n in zip(sp["sample"], sp["n"])]
    axB.set_xticks(xpos); axB.set_xticklabels(lab, fontsize=7.0, rotation=45, ha="right")
    axB.set_ylabel("Peak $T$ [°C]  (Rahl 2005)", fontsize=8.5)
    axB.set_xlabel("Sample", fontsize=8.5)
    axB.set_title("(b) Per-sample estimate", fontsize=9, loc="left")
    axB.set_xlim(-0.6, sp.shape[0]-0.4)
    axB.grid(axis="y", color="0.9", lw=0.6); axB.set_axisbelow(True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0],[0], marker="o", ls="none", mfc="0.6", mec="none", ms=5,
               label="individual measurement"),
        Line2D([0],[0], marker="o", ls="none", mfc="#d62728", mec="k", ms=7,
               label="sample mean ± 2σ"),
        Line2D([0],[0], color="#d62728", lw=1, ls="--",
               label=f"series mean ({mu:.0f} ± {2*sdser:.0f} °C, 2σ)"),
        Patch(fc="#d62728", alpha=0.10, label=f"±{CAL_ERR:.0f} °C calibration (95 %)"),
    ]
    axB.legend(handles=handles, loc="upper right", fontsize=7.3, frameon=True,
               framealpha=0.95, borderpad=0.4, handlelength=1.4, labelspacing=0.4,
               edgecolor="0.8")
    fig.savefig(OUT_FIG + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
