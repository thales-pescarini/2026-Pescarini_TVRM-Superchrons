# Table S5: reset thresholds and partition of the C2 moment as a function of T_peak
# (t_res = 7 Myr, f_Po = 0.85), for the ensemble-mean spectrum and the 31 specimens.
import numpy as np, pandas as pd
from scipy.optimize import brentq
import common
import blocking as P
import fig7_coherence as f7

mem, specs = common.ensemble_spectra()
Tm, wm, _ = f7.load_trout_spectrum(common.DATA / 'ensemble31_mean_C2_spectrum.csv')


def thr(bf, Tp, lo, hi):
    TK = lambda T: T + 273.15
    tgt = np.log(P.RESIDENCE * P.Ma)
    f = lambda Tub: (np.log(P.TAU0) + np.log(P.T_LAB / P.TAU0) * (TK(Tub) / TK(Tp)) * (bf(TK(Tp)) / bf(TK(Tub)))) - tgt
    return brentq(f, lo, hi)


def part(T, w, CP, CM, f=0.85):
    po = T < 320
    reset = (w * ((po & (T < CP)) * f + po * (1 - f) * (T < CM) + (~po) * (T < CM))).sum()
    return reset, (w * (po & (T >= CP)) * f).sum(), (w * (T >= CM)).sum()


rows = []
for Tp in [225, 235, 246, 250, 255, 266, 276, 286, 296, 307]:
    CP = thr(P.bP, Tp, 150, 319.99); CM = thr(P.bM, Tp, 250, 560)
    r, pu, mu = part(Tm, wm, CP, CM)
    per = np.array([part(T, w, CP, CM)[0] for T, w in specs])
    rows.append(dict(T_peak_C=Tp, Tub_crit_Po_C=round(CP, 1), Tub_crit_Mag_C=round(CM, 1),
        reset_mean_spectrum_pct=round(100*r, 1), unreset_Po_pct=round(100*pu, 1), unreset_Mag_pct=round(100*mu, 1),
        reset_specimens_pct=f'{100*per.mean():.1f} ± {200*per.std(ddof=1):.1f}'))
    print(rows[-1])
g = lambda Tp: part(Tm, wm, thr(P.bP, Tp, 150, 319.99), thr(P.bM, Tp, 250, 560))[0] - 0.5
print('T_peak at which 50% is reset:', round(brentq(g, 225, 300), 1))
pd.DataFrame(rows).to_csv(common.OUT / 'TableS5_reset_Tpeak.csv', index=False)
