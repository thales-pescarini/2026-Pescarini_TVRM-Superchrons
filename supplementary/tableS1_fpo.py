# Table S1 / Text S1: sensitivity to the pyrrhotite fraction f_Po.
# Homogeneous f_Po = 0.5-1.0 and two graded cases (f_Po linear in the cumulative moment
# below 320 C, 0.70 -> 1.00 and 1.00 -> 0.70, mean 0.85). 2.0 C/Myr, 400 histories.
import numpy as np, pandas as pd
import common
import blocking as P
import fig7_coherence as f7
import cstats

RATE = 2.0
mem, specs = common.ensemble_spectra()
Tall = np.concatenate([T for T, w in specs]); Wall = np.concatenate([w / len(specs) for T, w in specs])
m = Tall < 320; o = np.argsort(Tall[m]); Ts = Tall[m][o]; cw = np.cumsum(Wall[m][o]); cw /= cw[-1]
Fcum = lambda T: np.interp(T, Ts, cw)


def ffun(kind, v):
    if kind == 'h':
        return lambda T: np.full_like(np.asarray(T, float), v)
    return lambda T: 0.85 + v * (2 * Fcum(T) - 1)


def elements(T, w, rate, f):
    A = []; W = []

    def add(Tc, wc, args):
        a, _, dd = P.blocking_solution(Tc, *args, rate)
        if np.isfinite(a) and np.isfinite(dd) and dd > 0:
            A.append(a); W.append(wc)

    for Tc, wc, fc in zip(T, w, f(T)):
        if Tc >= 320:
            add(Tc, wc, (P.bM, P.TC_M_K, 0.40025, 2.0, P.C_M))
        else:
            if Tc < P.C_P:
                add(Tc, wc * fc, (P.bP, P.TC_P_K, 0.52, 2.6, P.C_P))
            add(Tc, wc * (1 - fc), (P.bM, P.TC_M_K, 0.40025, 2.0, P.C_M))
    return np.array(A), np.array(W)


Pol = cstats.polarity([f7.ENSEMBLE_SEED0 + i for i in range(f7.N_REALIZATIONS)])
rows = []
for kind, v, name in [('h', .5, 'f_Po = 0.50'), ('h', .6, 'f_Po = 0.60'), ('h', .7, 'f_Po = 0.70'),
                      ('h', .85, 'f_Po = 0.85 (reference)'), ('h', 1., 'f_Po = 1.00'),
                      ('g', .15, 'Graded, 0.70 -> 1.00'), ('g', -.15, 'Graded, 1.00 -> 0.70')]:
    f = ffun(kind, v); A = []; W = []; meds = []; resets = []
    for T, w in specs:
        a, ww = elements(T, w, RATE, f)
        A.append(a); W.append(ww / len(specs))
        meds.append(P.wquant(a, ww, 0.5)); resets.append(ww.sum())
    meds = np.array(meds)
    st = cstats.stats(np.concatenate(A), np.concatenate(W), Pol)
    g = st['gross_fraction']
    rows.append(dict(scenario=name,
        reset_pct=f"{100*np.mean(resets):.0f} ± {200*np.std(resets, ddof=1):.0f}",
        mean_of_medians_Ma=f"{meds.mean():.1f} ± {2*meds.std(ddof=1):.1f}",
        delay_from_507Ma_Myr=round(P.PEAK_START - meds.mean(), 1),
        medians_in_MRS=int(((meds >= P.MRS_END) & (meds <= P.MRS_START)).sum()),
        gross_pre_MRS_post_pct='/'.join(f'{100*x:.0f}' for x in g),
        C_bulk='{:.2f} ({:.2f}–{:.2f})'.format(st['Cbulk'][1], *st['Cbulk'][::2]),
        MRS_share_pct='{:.0f} ({:.0f}–{:.0f})'.format(100*st['MRS_share'][1], 100*st['MRS_share'][0], 100*st['MRS_share'][2]),
        RMS_eff_pre_post_pct=f"{100*st['rms_eff'][0]:.0f}/{100*st['rms_eff'][2]:.0f}",
        resultant_reversed_all=st['reversed_all']))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(common.OUT / 'TableS1_fpo.csv', index=False)
