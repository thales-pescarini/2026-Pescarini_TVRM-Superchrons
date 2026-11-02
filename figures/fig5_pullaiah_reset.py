# Fig. 5: thermoviscous reset of the C2 remanence at peak conditions (266 C, 7 Myr).
# (a) Pullaiah diagram for magnetite and pyrrhotite; (b) C2 spectra of the 31 ensemble
# specimens partitioned by the reset thresholds and the pyrrhotite Curie point.
# Usage: python figures/fig5_pullaiah_reset.py [run directory] [output stem]
import os, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data", "trout_run")
OUT = sys.argv[2] if len(sys.argv) > 2 else "Fig5"

tau0, tlab = 1e-9, 2400.0
YR = 3.15576e7; tres = 7e6*YR; Tpk = 266.0; TC_PYR = 320.0; F_PYR = 0.85
def bM(TK): return (np.maximum(1 - TK/853.15, 1e-12)/(1 - 273.15/853.15))**(0.40025*2.0)
def bP(TK): return (np.maximum(1 - TK/593.15, 1e-12)/(1 - 273.15/593.15))**(0.52*2.6)
def lntau(T, Tub, b):
    TK, UK = T + 273.15, Tub + 273.15
    return np.log(tau0) + np.log(tlab/tau0)*(UK/TK)*b(TK)/b(UK)
CP = brentq(lambda u: lntau(Tpk, u, bP) - np.log(tres), 267, 319.9)
CM = brentq(lambda u: lntau(Tpk, u, bM) - np.log(tres), 267, 570)
assert abs(CP - 285.33) < 0.01 and abs(CM - 394.35) < 0.01, (CP, CM)

mem = open(os.path.join(D, "ensemble_membership.txt")).read().split()
G = np.linspace(0, 700, 7001); S = []; rows = []
for s in mem:
    d = pd.read_csv(os.path.join(D, "C2_SPECTRA", f"{s}__C2_spectrum.csv"))
    T, p = d.temperature_C.values, d.C2_normalized_by_area.values
    p = p/np.trapezoid(p, T)
    I = lambda a, b: np.trapezoid(np.where((T >= a) & (T < b), p, 0.0), T)
    z = [I(0, CP), I(CP, TC_PYR), I(TC_PYR, CM), I(CM, 1e4)]
    rows.append(dict(spec=s, z1=z[0], z2=z[1], z3=z[2], z4=z[3],
                     reset=z[0] + (1 - F_PYR)*z[1] + z[2], pyr_not_reset=F_PYR*z[1]))
    S.append(np.interp(G, T, p, left=0.0, right=0.0))
S = np.array(S); mean = S.mean(0)
R = pd.DataFrame(rows).set_index("spec"); R.to_csv(OUT + "_fractions_per_specimen.csv")
ms = lambda k: (100*R[k].mean(), 2*100*R[k].std(ddof=1))   # mean and 2 sd

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
    "axes.linewidth": 0.7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.6,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.minor.width": 0.5,
    "legend.frameon": False, "legend.fontsize": 7.2, "hatch.linewidth": 0.5,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})
INK, MUTED = "0.12", "0.45"
XL = (100, 450)

fig = plt.figure(figsize=(5.6, 6.9))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.9], hspace=0.06,
                      left=0.115, right=0.975, top=0.93, bottom=0.075)
axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1], sharex=axA)

# (a)
T = np.linspace(XL[0], XL[1], 3000)
FAM_C = "0.62"
LS_PO = (0, (4.5, 2.2))
# family curves every 50 C (Mag) and 40 C (Po) from each reset threshold
fam = {"mag": (bM, 580.0, "-", [round(CM) + 50*k for k in range(-5, 4) if k != 0], CM),
       "po":  (bP, TC_PYR, LS_PO, [round(CP) + 40*k for k in range(-4, 1) if k != 0] + [310], CP)}
for key, (b, Tc, ls, ubs, C) in fam.items():
    TT = T[T < Tc - 0.02]
    for Tub in ubs:
        axA.plot(TT, np.exp(lntau(TT, Tub, b)), color=FAM_C, ls=ls, lw=0.8, zorder=2)
    axA.plot(TT, np.exp(lntau(TT, C, b)), color=INK, ls=ls, lw=1.7, zorder=4,
             path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
axA.set_yscale("log"); axA.set_ylim(10, 3e17)
ticks = {"1 min": 60, "1 h": 3600, "1 yr": YR, "1 kyr": 1e3*YR, "1 Myr": 1e6*YR, "1 Gyr": 1e9*YR}
axA.set_yticks(list(ticks.values())); axA.set_yticklabels(list(ticks.keys()))
axA.yaxis.set_minor_locator(mpl.ticker.NullLocator())
PURPLE = "#6a3d9a"
axA.axhspan(5e6*YR, 10e6*YR, color=PURPLE, alpha=0.13, lw=0, zorder=1)
axA.axhline(tres, color=PURPLE, lw=1.0, ls=(0, (1.2, 1.6)), zorder=3)
axA.axhline(tlab, color=INK, lw=0.7, ls=(0, (1.2, 1.6)), zorder=3)
axA.text(XL[0] + 3, tres*3.2, "Peak residence time, 7 Myr (5–10 Myr)", color=PURPLE, fontsize=7,
         va="bottom", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
axA.text(XL[0] + 3, tlab*1.6, "Laboratory heating time, 40 min", color=INK, fontsize=7,
         va="bottom", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
STAR = "#F7A21B"
axA.plot(Tpk, tres, marker="*", ms=14, mfc=STAR, mec=INK, mew=0.8, zorder=8)
for C, name in [(CP, "Po"), (CM, "Mag")]:
    axA.plot(C, tlab, "o", ms=5.5, mfc="white", mec=INK, mew=1.0, zorder=8)
    axA.text(C - 3, 22, f"{name} reset\nthreshold\n{C:.0f} °C", ha="right", va="bottom",
             fontsize=6.8, color=INK, zorder=7, linespacing=1.1,
             bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.6))
axA.set_ylabel(r"Relaxation time  $\tau$")
axA.tick_params(labelbottom=False)
fig.legend([Line2D([], [], color=FAM_C, lw=1.2, ls="-"),
            Line2D([], [], color=FAM_C, lw=1.2, ls=LS_PO),
            Line2D([], [], color=INK, lw=1.7, ls="-"),
            Line2D([], [], color=INK, lw=1.7, ls=LS_PO),
            Line2D([], [], marker="*", ms=10, mfc=STAR, mec=INK, ls=""),
            Line2D([], [], marker="o", ms=5, mfc="white", mec=INK, ls="")],
           ["Magnetite (Mag)", "Pyrrhotite (Po)", "Mag reset threshold", "Po reset threshold",
            "Peak condition (266 °C, 7 Myr)", "Threshold at laboratory time (40 min)"],
           loc="upper left", bbox_to_anchor=(0.105, 0.986), ncol=3, handlelength=2.4,
           labelspacing=0.3, columnspacing=1.0, fontsize=6.9)
axA.text(0.012, 0.975, "a", transform=axA.transAxes, fontsize=10, fontweight="bold", va="top",
         zorder=9, bbox=dict(fc="white", ec="none", pad=1.0))

# (b)
zones = [(XL[0], CP, "#762a83", 0.28), (CP, TC_PYR, "#3b6e8f", 0.28),
         (TC_PYR, CM, "#1b7837", 0.26), (CM, XL[1] + 50, "#b59a2a", 0.34)]
for y in S:
    axB.plot(G, y, color="0.55", lw=0.5, alpha=0.36, zorder=0)
for a, b_, c, al in zones:
    axB.fill_between(G, 0, mean, where=(G >= a) & (G <= b_), color=c, alpha=al, lw=0,
                     interpolate=True, zorder=1)
axB.plot(G, mean, color=INK, lw=1.8, zorder=10)
for x in (CP, TC_PYR, CM):
    axB.axvline(x, color="0.35", lw=0.6, zorder=2)
    axA.axvline(x, color="0.82", lw=0.5, zorder=0)
ytop = S[:, (G > XL[0]) & (G < XL[1])].max()*1.42
axB.set_ylim(0, ytop); axB.set_xlim(*XL)

z = {k: ms(k) for k in ["z1", "z2", "z3", "z4", "reset", "pyr_not_reset"]}
lab = [((XL[0] + CP)/2 - 25, f"$T_{{ub}}^{{lab}}$ < {CP:.0f} °C\nMag + Po reset\n{z['z1'][0]:.0f} ± {z['z1'][1]:.0f} %"),
       ((CP + TC_PYR)/2, f"{CP:.0f}–320 °C\nPo not reset\nMag reset\n{z['z2'][0]:.0f} ± {z['z2'][1]:.0f} %\n"
                          f"(Po {z['pyr_not_reset'][0]:.0f} ± {z['pyr_not_reset'][1]:.0f} %)"),
       ((TC_PYR + CM)/2, f"320–{CM:.0f} °C\nMag reset\n{z['z3'][0]:.0f} ± {z['z3'][1]:.0f} %"),
       ((CM + XL[1])/2, f"> {CM:.0f} °C\nMag not reset\n{z['z4'][0]:.0f} ± {z['z4'][1]:.0f} %")]
for x, t in lab:
    axB.text(x, ytop*0.975, t, ha="center", va="top", fontsize=6.5, color=INK, linespacing=1.15,
             zorder=11, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
axB.legend([Line2D([], [], color="0.55", lw=0.8, alpha=0.6), Line2D([], [], color=INK, lw=1.7)],
           [f"Individual specimens (n = {len(S)})", "Ensemble mean"],
           loc="upper left", bbox_to_anchor=(0.025, 0.70), handlelength=2.2, labelspacing=0.35)
axB.text(XL[0] + 12, ytop*0.52, f"Reset at peak:\n{z['reset'][0]:.0f} ± {z['reset'][1]:.0f} % of C2 moment",
         fontsize=7.4, fontweight="bold", color=INK, va="top")
axB.text(XL[0] + 12, ytop*0.40, "Fractions: mean ± 2σ\nacross specimens (n = 31)",
         fontsize=6.8, color=MUTED, va="top")
axB.set_xlabel(r"Laboratory unblocking temperature  $T_{ub}^{lab}$ (°C)")
axB.set_ylabel(r"Normalized moment density (°C$^{-1}$)")
axB.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
axB.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(10))
axB.text(0.015, 0.97, "b", transform=axB.transAxes, fontsize=10, fontweight="bold", va="top")

for ext in ("pdf", "png", "svg"):
    fig.savefig(f"{OUT}.{ext}", dpi=400 if ext == "png" else None)
print(f"C_Po {CP:.2f}  C_Mag {CM:.2f}"); [print(k, f"{a:.1f} ± {s:.1f}") for k, (a, s) in z.items()]
