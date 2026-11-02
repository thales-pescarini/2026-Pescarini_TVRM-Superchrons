# Table S6: ensemble blocking age (mean of the specimen medians) for T_peak at the 2-sigma
# RSCM limits and cooling rates of 1-5 C/Myr; reset thresholds recomputed for each T_peak.
# The csv gives the dispersion as 1 sigma; Table S6 quotes 2 sigma.
import numpy as np
import pandas as pd
import common
import blocking as E

mem, specs = common.ensemble_spectra()
rows = []
for tpk in [225.0, 246.0, 266.0, 286.0, 307.0]:
    E.T_PEAK = tpk
    E.C_P = E.reset_threshold(E.bP, E.RESIDENCE, 200.0, 319.5)
    E.C_M = E.reset_threshold(E.bM, E.RESIDENCE, 300.0, 570.0)
    for rate in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
        med = []
        for T, w in specs:
            A, W, _ = E.elements(T, w, rate)
            med.append(E.wquant(A, W, 0.5))
        med = np.array(med)
        k = int(((med >= E.MRS_END) & (med <= E.MRS_START)).sum())
        rows.append([tpk, rate, E.C_P, E.C_M, med.mean(), med.std(ddof=1), k])
        print(f'{tpk:5.0f} {rate:4.1f} {E.C_P:6.1f} {E.C_M:6.1f} {med.mean():7.1f} {med.std(ddof=1):5.1f} {k:3d}')
cols = ['T_peak_C', 'rate_C_per_Myr', 'reset_thr_Po_C', 'reset_thr_Mag_C', 'mean_of_medians_Ma', 'sd_1sigma_Myr', 'k_in_MRS_of_31']
pd.DataFrame(rows, columns=cols).to_csv(common.OUT / 'TableS6_Tpeak_rate.csv', index=False, float_format='%.3f')
