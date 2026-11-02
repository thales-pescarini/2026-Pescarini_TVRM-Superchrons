# Selection gates for the TROUT C2 spectra. A specimen enters the ensemble when it passes
#  L1: low-temperature tail (edge median <= 0.03, mean of the lowest 8% of the span <= 0.05,
#      95th percentile <= 0.10; spectrum normalized to its maximum)
#  H1: m/m0 at the end of the fitting window <= 0.05
#  H2: C2 peak at least 30 C below the top of the window
#  A1: no abrupt edge (20% -> 2% transition narrower than 8 C and 3% of that side's span,
#      on the smoothed monotonic envelope)
# Writes gates_result.csv; the accepted specimens are those in ensemble_membership.txt.
# Usage: python trout/gates.py [run directory]
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "trout_run"

M_END_MAX, PEAK_MARGIN = 0.05, 30.0
MAX_EDGE_MEDIAN, MAX_TAIL_MEAN, MAX_TAIL_P95, TAIL_FRACTION = 0.03, 0.05, 0.10, 0.08
UP_LVL, LO_LVL, MIN_W_C, MIN_W_FRAC, MIN_SIDE = 0.20, 0.02, 8.0, 0.03, 20.0


def smooth(y):
    n = len(y); w = max(11, int(round(0.015 * n)))
    if w % 2 == 0: w += 1
    w = min(w, n if n % 2 == 1 else n - 1)
    if w <= 3 or w < 5: return np.clip(y, 0, 1)
    s = savgol_filter(y, w, min(3, w - 2), mode="interp"); s = np.clip(s, 0, 1)
    m = np.nanmax(s); return s / m if m > 0 else s


def cross_inc(x, y, lvl):
    c = np.where(y >= lvl)[0]
    if len(c) == 0: return np.nan
    i = c[0]
    if i == 0: return x[0]
    y0, y1 = y[i - 1], y[i]
    return x[i] if y1 == y0 else x[i - 1] + (lvl - y0) / (y1 - y0) * (x[i] - x[i - 1])


def cross_dec(x, y, lvl):
    c = np.where(y <= lvl)[0]
    if len(c) == 0: return np.nan
    i = c[0]
    if i == 0: return x[0]
    y0, y1 = y[i - 1], y[i]
    return x[i] if y1 == y0 else x[i - 1] + (lvl - y0) / (y1 - y0) * (x[i] - x[i - 1])


def core(sp):
    for k in ("1B_TT_set1", "1B_TT_set2", "1A_T_1", "1A_T_2", "1G", "Dt"):
        if sp.startswith(k): return k


stat = pd.read_csv(RUN / "trout_status.csv")
rows = []
for _, r in stat.iterrows():
    sp = r.specimen
    d = pd.read_csv(RUN / "C2_SPECTRA" / f"{sp}__C2_spectrum.csv")
    x = d.temperature_C.values; y = d.C2_normalized_to_max.values
    span = x.max() - x.min()
    lo = x <= x.min() + TAIL_FRACTION * span
    ep = max(10, int(np.ceil(0.005 * len(y)))); ep = min(ep, lo.sum())
    L1 = (np.nanmedian(y[:ep]) <= MAX_EDGE_MEDIAN and np.nanmean(y[lo]) <= MAX_TAIL_MEAN
          and np.nanpercentile(y[lo], 95) <= MAX_TAIL_P95)
    H1 = float(r.m_end) <= M_END_MAX
    ys = smooth(y); ipk = int(np.argmax(ys)); pkT = float(x[ipk])
    H2 = (x.max() - pkT) >= PEAK_MARGIN
    ab_lo = ab_hi = False
    lo_span = pkT - x.min(); hi_span = x.max() - pkT
    if lo_span >= MIN_SIDE:
        yl = np.maximum.accumulate(ys[:ipk + 1])
        x2 = cross_inc(x[:ipk + 1], yl, LO_LVL); x20 = cross_inc(x[:ipk + 1], yl, UP_LVL)
        if np.isfinite(x2) and np.isfinite(x20) and x20 > x2:
            w = x20 - x2; ab_lo = (w < MIN_W_C) and (w / lo_span < MIN_W_FRAC)
    if hi_span >= MIN_SIDE:
        yh = np.maximum.accumulate(ys[ipk:][::-1])[::-1]
        x20 = cross_dec(x[ipk:], yh, UP_LVL); x2 = cross_dec(x[ipk:], yh, LO_LVL)
        if np.isfinite(x20) and np.isfinite(x2) and x2 > x20:
            w = x2 - x20; ab_hi = (w < MIN_W_C) and (w / hi_span < MIN_W_FRAC)
    A1 = not (ab_lo or ab_hi)
    rows.append(dict(specimen=sp, L1=L1, H1=H1, m_end=round(float(r.m_end), 3), H2=H2,
                     peak_C=round(pkT, 1), window_top_C=round(float(x.max()), 1),
                     abrupt_low=ab_lo, abrupt_high=ab_hi, A1=A1,
                     FINAL=L1 and H1 and H2 and A1))

df = pd.DataFrame(rows)
df["core"] = df.specimen.map(core)
df.to_csv(RUN / "gates_result.csv", index=False)
print(f"Accepted: {int(df.FINAL.sum())}/{len(df)}")
print(df.groupby("core").agg(accepted=("FINAL", "sum"), n=("specimen", "count")).to_string())
print("\nRejected and why:")
for _, r in df[~df.FINAL].iterrows():
    why = []
    if not r.L1: why.append("low-temperature tail")
    if not r.H1: why.append(f"m_end {r.m_end}")
    if not r.H2: why.append(f"peak < 30 C below window top ({r.peak_C}/{r.window_top_C})")
    if r.abrupt_low: why.append("abrupt low side")
    if r.abrupt_high: why.append("abrupt high side")
    print(f"  {r.specimen:16s} {'; '.join(why)}")
