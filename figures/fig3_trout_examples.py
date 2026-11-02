# Fig. 3: TROUT examples, one specimen per drill core. (a) orthogonal projection,
# rotated so that the fitted C1 (viscous) component points to the present-field
# declination at the sites; (b) C1 and C2 unblocking spectra; (c) m/m0.
# Needs the MagIC tables of Pescarini et al. (2025) and trout.py (see README).
# Usage: python figures/fig3_trout_examples.py <data root> [run directory] [output stem]
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trout"))
import trout as tr
from pmagpy import pmag

if len(sys.argv) < 2:
    sys.exit("Usage: python figures/fig3_trout_examples.py <data root> [run directory] [output stem]")
DATA_ROOT = Path(sys.argv[1]).expanduser().resolve()
DATASET_DIRS = {
    "DT": DATA_ROOT / "_TROUT_MAGIC_STAGING" / "DT",
    "B1": DATA_ROOT / "1B" / "1B_TT_set1_",
    "B2": DATA_ROOT / "1B" / "1B_TT_set2_",
    "G":  DATA_ROOT / "1G",
    "A1": DATA_ROOT / "1A" / "1A_T_1_",
    "A2": DATA_ROOT / "1A" / "1A_T_2_",
}
RUN = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data" / "trout_run"
FIGDIR = Path(".")
STEM = sys.argv[3] if len(sys.argv) > 3 else "Fig3"
SPECTRUM_POINTS = 10000
TARGET_DEC = 341.9          # present-day field declination at the sites (IGRF)
XB_MAX = 450.0              # x limit of panel (b)
TRIO = ["Dt_57", "1B_TT_set2_121", "1G_TT_18", "1A_T_1_7"]

def dskey(sp):
    for pref, k in (("1B_TT_set1", "B1"), ("1B_TT_set2", "B2"), ("1A_T_1", "A1"),
                    ("1A_T_2", "A2"), ("1G", "G"), ("Dt", "DT")):
        if sp.startswith(pref):
            return k

# as in trout/trout_batch.py
def load_magic_dataset(name, directory):
    import os
    wd = str(Path(directory)) + os.sep
    meas, samp, site = tr.import_direction_data(wd)
    meas = meas.copy()
    meas["specimen"] = meas["specimen"].fillna("").astype(str).str.strip()
    return {"meas_data": meas, "sample_data": samp, "site_data": site}

def prepare_truncated_trout_input(zijd_data, Ts_base):
    zijd_data = np.asarray(zijd_data, dtype=float)
    Ts_base = np.asarray(Ts_base, dtype=float).reshape(-1)
    vector_norms = np.linalg.norm(zijd_data, axis=1)
    normalization = float(np.nanmax(vector_norms))
    zijd_clean = zijd_data / normalization
    flipped = np.flip(zijd_clean, axis=0)
    diff = np.diff(flipped, axis=0)
    appended = np.vstack([np.zeros(3), diff])
    residual = float(np.linalg.norm(zijd_clean[-1]))
    norm_data = np.linalg.norm(appended, axis=1) + residual
    Ms = np.flip(np.cumsum(norm_data))
    Ms_scaled = Ms / float(np.nanmax(Ms))
    m = minimize(tr.bestfit_tanh, x0=[1.0, 0.1], args=(Ts_base, Ms_scaled),
                 method="L-BFGS-B", bounds=[(1.0, 100.0), (-10.0, 10.0)],
                 options={"eps": np.sqrt(np.finfo("float32").eps)})
    expon = float(m.x[1])
    Ts_data = np.asarray(tr.scale_xs(Ts_base, expon), dtype=float)
    return {"zijd_data": zijd_clean, "Ts_base": Ts_base, "Ts_data": Ts_data,
            "expon": expon}

def calculate_trout_spectra(par_dict, Ts_base, expon):
    cs = np.asarray(par_dict["cs"], dtype=float)
    mus = np.asarray(par_dict["mus"], dtype=float)
    sds = np.asarray(par_dict["sds"], dtype=float)
    p_stars = np.asarray(par_dict["p_stars"], dtype=float)
    q_stars = np.asarray(par_dict["q_stars"], dtype=float)
    Ts_base = np.asarray(Ts_base, dtype=float)
    axis_full = np.linspace(float(np.min(Ts_base)), float(np.max(Ts_base)), SPECTRUM_POINTS)
    scaled_axis_full = np.asarray(tr.scale_xs(axis_full, expon), dtype=float)
    transformation_exponent = np.exp(-abs(expon))
    pdfs = []
    for k in range(len(cs)):
        p_value = np.exp(p_stars[k])
        q_star = float(q_stars[k])
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_q = np.divide(1.0, q_star)
        q_value = 2.0 / np.pi * np.arctan(inv_q)
        pdf = cs[k] * tr.SGG_pdf(scaled_axis_full, mus[k], sds[k], p_value, q_value)
        pdf = np.asarray(pdf, dtype=float)
        h = np.heaviside(expon, 0)
        transform = h * np.max(axis_full) - (h * 2.0 - 1.0) * axis_full
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            pdf *= transformation_exponent * transform ** (transformation_exponent - 1.0)
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        pdfs.append(pdf)
    pdfs = np.asarray(pdfs, dtype=float)
    return axis_full[:-1], pdfs[:, :-1]

def R_spec2geo(az, pl):
    # columns: specimen axes in the geographic frame (as pmag.dogeo)
    cols = []
    for d, i in ((0.0, 0.0), (90.0, 0.0), (0.0, 90.0)):
        dg, ig = pmag.dogeo(d, i, az, pl)
        cols.append(pmag.dir2cart([dg, ig, 1.0]))
    return np.array(cols, float).T

def azpl_for(sp):
    smp = pd.read_csv(DATASET_DIRS[dskey(sp)] / "samples.txt", skiprows=1, sep="\t")
    smp = smp.dropna(subset=["azimuth"])
    return float(smp["azimuth"].iloc[0]), float(smp["dip"].iloc[0])

def rotz(v, dd):
    d = np.deg2rad(dd)
    R = np.array([[np.cos(d), -np.sin(d), 0], [np.sin(d), np.cos(d), 0], [0, 0, 1]])
    return v @ R.T

def eqarea(dec, inc):
    dec = np.deg2rad(np.asarray(dec, float)); inc = np.asarray(inc, float)
    r = np.sqrt(2.0) * np.sin(np.deg2rad(45.0 - np.abs(inc) / 2.0))
    return r * np.sin(dec), r * np.cos(dec), inc >= 0

GRN, PRP = "#1b7837", "#762a83"
dom_cmap = LinearSegmentedColormap.from_list("grn_prp", [GRN, PRP])
status = pd.read_csv(RUN / "trout_status.csv").set_index("specimen")
cacheDS = {}

def series(sp, seed):
    k = dskey(sp)
    if k not in cacheDS:
        cacheDS[k] = load_magic_dataset(k, DATASET_DIRS[k])
    ds = cacheDS[k]
    np.random.seed(seed)
    zf, Tf, _, _, dt, _ = tr.prepare_specimen_data(sp, ds["meas_data"], ds["sample_data"], ds["site_data"])
    return np.asarray(zf, float), np.asarray(Tf, float).reshape(-1)

data = {}
for spec in TRIO:
    par = json.load(open(RUN / "PARAMS" / f"{spec}__params.json"))
    zf, Tf = series(spec, int(status.loc[spec, "seed"]))
    keep = int(par["keep"]); zu, Tu = zf[:keep], Tf[:keep]
    tin = prepare_truncated_trout_input(zu, Tu)
    pdict = {k: np.asarray(par[k], float) for k in ("cs", "mus", "sds", "p_stars", "q_stars", "Bs")}
    axisT, pdfs = calculate_trout_spectra(pdict, tin["Ts_base"], tin["expon"])
    pdfs = np.nan_to_num(np.asarray(pdfs, float), nan=0.0, posinf=0.0, neginf=0.0)
    az, pl = azpl_for(spec)
    R = R_spec2geo(az, pl)                      # v_geo = R @ v_spec
    z_spec = zf @ R
    zu_spec = zu @ R
    Bs_spec = pdict["Bs"] @ R
    dirs_spec = np.array(pmag.cart2dir(Bs_spec))
    delta = (TARGET_DEC - dirs_spec[0, 0] + 540.0) % 360.0 - 180.0
    data[spec] = dict(delta=delta, keep=keep, Ts_full=Tf, Ts_used=Tu,
                      Ts_data=np.asarray(tin["Ts_data"], float),
                      axis=np.asarray(axisT, float), pdfs=pdfs, z_full=z_spec,
                      zfR=rotz(z_spec, delta), zuR=rotz(zu_spec, delta), BsR=rotz(Bs_spec, delta),
                      cs=pdict["cs"], mus=pdict["mus"], sds=pdict["sds"],
                      ps=pdict["p_stars"], qs=pdict["q_stars"])

NR = len(TRIO)
fig, axes = plt.subplots(NR, 3, figsize=(13.2, 3.6 * NR))
axes = np.atleast_2d(axes)

for irow, spec in enumerate(TRIO):
    v = data[spec]
    AX = "E"   # N/Up on the y axis in all rows
    HIDX = 1 if AX == "E" else 0
    OIDX = 0 if AX == "E" else 1
    OLAB = "N" if AX == "E" else "E"
    zfR, zuR, BsR, delta = v["zfR"], v["zuR"], v["BsR"], v["delta"]
    Tf, Tu, Ts_data = v["Ts_full"], v["Ts_used"], v["Ts_data"]
    axT = v["axis"]; pdfs = v["pdfs"]
    cs, mus, sds, ps, qs = v["cs"], v["mus"], v["sds"], v["ps"], v["qs"]
    dirsR = np.array(pmag.cart2dir(BsR))
    print(f"{spec}: delta {delta:+.1f} | C1' {dirsR[0,0]:.1f}/{dirsR[0,1]:.1f} | "
          f"C2' {dirsR[1,0]:.1f}/{dirsR[1,1]:.1f} | keep {v['keep']}/{len(Tf)}")

    tg = np.linspace(Ts_data.min(), Ts_data.max(), 400)
    Fs = np.zeros((2, len(tg)))
    for i in range(2):
        p_v = np.exp(ps[i]); q_v = 2.0 / np.pi * np.arctan(1.0 / qs[i]) if qs[i] != 0 else 1.0
        Fs[i] = 1.0 - np.asarray(tr.SGG_cdf(tg, mus[i], sds[i], p_v, q_v), float)
    pred = (cs[0] * np.outer(Fs[0], BsR[0]) + cs[1] * np.outer(Fs[1], BsR[1])) + zuR[-1]
    dF = np.abs(np.gradient(Fs, axis=1))
    wC2 = (cs[1] * dF[1]) / (cs[0] * dF[0] + cs[1] * dF[1] + 1e-30)

    # (a)
    Ap = axes[irow, 0]
    H = zfR[:, HIDX]; O = zfR[:, OIDX]; Uu = -zfR[:, 2]
    Ap.plot(H, O, "-", color="k", lw=0.7, zorder=2)
    Ap.plot(H, O, "o", mfc="k", mec="k", ms=4.6, zorder=3)
    Ap.plot(H, Uu, "-", color="k", lw=0.7, zorder=2)
    Ap.plot(H, Uu, "s", mfc="white", mec="k", ms=4.6, zorder=3)
    for Y in (pred[:, OIDX], -pred[:, 2]):
        pts = np.column_stack([pred[:, HIDX], Y]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        Ap.add_collection(LineCollection(segs, cmap=dom_cmap, array=wC2[:-1], lw=3.0, alpha=0.85, zorder=4))
    Ap.axhline(0, color="k", lw=0.8); Ap.axvline(0, color="k", lw=0.8)
    Ap.set_aspect("equal", adjustable="datalim")
    norms = np.linalg.norm(zfR, axis=1); lastkept = v["keep"] - 1; lab_m = 10.0
    for k, T in enumerate(Tf):
        take = (k == 0) or (k == lastkept)
        if not take and norms[k] > 0.12 * norms[0] and (lab_m - norms[k]) > 0.10 * norms[0]:
            take = True
        if take:
            lab_m = norms[k]
            Ap.annotate(f"{int(T)}", (H[k], O[k]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    Ap.set_xlabel(f"{AX}, arbitrary units"); Ap.set_ylabel(f"{OLAB} / Up, arbitrary units")
    ttl = "(a) Zijderveld\n" if irow == 0 else ""
    Ap.set_title(f"{ttl}Specimen {spec}", fontsize=11 if irow == 0 else 10, loc="left",
                 fontweight="bold" if irow == 0 else "normal")
    if irow == 0:
        Ap.legend(handles=[Line2D([0], [0], marker="o", ls="-", mfc="k", mec="k", color="k", ms=4.6, lw=0.7, label=f"horizontal projection ({AX}-{OLAB})"),
                           Line2D([0], [0], marker="s", ls="-", mfc="white", mec="k", color="k", ms=4.6, lw=0.7, label=f"vertical projection ({AX}-Up)")],
                  loc="lower left", fontsize=7.2, frameon=True)
    x0, x1 = Ap.get_xlim(); y0, y1 = Ap.get_ylim()
    def densify(a, b, n=6):
        aa = []; bb = []
        for i in range(len(a) - 1):
            aa.append(np.linspace(a[i], a[i + 1], n)); bb.append(np.linspace(b[i], b[i + 1], n))
        return np.concatenate(aa), np.concatenate(bb)
    Hd1, Od = densify(H, O); Hd2, Ud = densify(H, Uu)
    fx = (np.concatenate([Hd1, Hd2]) - x0) / (x1 - x0); fy = (np.concatenate([Od, Ud]) - y0) / (y1 - y0)
    corners = {"upper right": (0.66, 1, 0.64, 1), "lower right": (0.66, 1, 0, 0.36),
               "upper left": (0, 0.34, 0.64, 1), "lower left": (0, 0.34, 0, 0.36)}
    occ = {k: int(np.sum((fx >= a) & (fx <= b) & (fy >= c) & (fy <= d))) for k, (a, b, c, d) in corners.items()}
    occ["lower left"] += (40 if irow == 0 else 0)
    S = inset_axes(Ap, width="30%", height="30%", loc=min(occ, key=occ.get))
    th = np.linspace(0, 2 * np.pi, 200)
    S.plot(np.cos(th), np.sin(th), color="k", lw=0.8)
    for ang in range(0, 360, 30):
        S.plot([0.96 * np.sin(np.deg2rad(ang)), np.sin(np.deg2rad(ang))],
               [0.96 * np.cos(np.deg2rad(ang)), np.cos(np.deg2rad(ang))], color="k", lw=0.6)
    S.plot([-1, 1], [0, 0], color="k", lw=0.3); S.plot([0, 0], [-1, 1], color="k", lw=0.3)
    dd = np.array(pmag.cart2dir(zfR))
    x_, y_, low = eqarea(dd[:, 0], dd[:, 1])
    S.plot(x_[low], y_[low], "o", mfc="k", mec="k", ms=2.6)
    S.plot(x_[~low], y_[~low], "o", mfc="white", mec="k", ms=2.6, mew=0.5)
    for i, col in ((0, GRN), (1, PRP)):
        xs, ys, lo = eqarea([dirsR[i, 0]], [dirsR[i, 1]])
        S.plot(xs, ys, "s", mfc=(col if lo[0] else "white"), mec=col, ms=4.5, mew=1.2)
    S.set_xlim(-1.15, 1.15); S.set_ylim(-1.15, 1.15); S.set_aspect("equal"); S.axis("off")

    # (b)
    B = axes[irow, 1]
    axTb = axT[:-8]; pb = pdfs[:, :-8]   # edge effect of the axis transform
    B.fill_between(axTb, pb[0], color=GRN, alpha=0.22, lw=0)
    B.fill_between(axTb, pb[1], color=PRP, alpha=0.22, lw=0)
    B.fill_between(axTb, np.minimum(pb[0], pb[1]), color="0.45", alpha=0.45, lw=0)
    B.plot(axTb, pb[0], color=GRN, lw=1.6)
    B.plot(axTb, pb[1], color=PRP, lw=1.6)
    B.set_xlim(0, XB_MAX); B.set_ylim(bottom=0)
    B.set_xlabel(r"$T_{UB}^{lab}$ ($^\circ$C)"); B.set_ylabel("f(T) (scaled)")
    if irow == 0:
        B.set_title("(b) Unblocking spectra", fontsize=11, fontweight="bold")
        B.legend(handles=[Line2D([0], [0], color=GRN, lw=2, label="C1"),
                          Line2D([0], [0], color=PRP, lw=2, label="C2"),
                          plt.Rectangle((0, 0), 1, 1, fc="0.45", alpha=0.45, label="overlap")],
                 loc="upper right", fontsize=8, frameon=True)

    # (c)
    C = axes[irow, 2]
    m = np.linalg.norm(v["z_full"], axis=1); m = m / m[0]
    C.plot(Tf, m, "-", color="k", lw=1.0)
    C.plot(Tf, m, "o", mfc="k", mec="k", ms=4.6)
    C.grid(alpha=0.3, lw=0.5)
    C.set_xlabel(r"$T_{UB}^{lab}$ ($^\circ$C)"); C.set_ylabel(r"$m/m_0$")
    C.set_xlim(-8, max(Tf) * 1.02); C.set_ylim(-0.02, max(1.06, 1.05 * float(m.max())))
    if irow == 0:
        C.set_title("(c) Demagnetization curve", fontsize=11, fontweight="bold")

fig.tight_layout()
FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / f"{STEM}.pdf", dpi=300, bbox_inches="tight")
fig.savefig(FIGDIR / f"{STEM}.png", dpi=200, bbox_inches="tight")
print("saved", FIGDIR / STEM)
