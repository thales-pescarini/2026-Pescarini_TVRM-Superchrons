# Equal-weight mean C2 spectrum of the 31 ensemble specimens (input of Fig. 7a).
# Each spectrum is normalized to unit area, interpolated on a 0.25 C grid and averaged.
# Usage: python model/build_mean_spectrum.py [run directory] [output csv]
import os, sys
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'trout_run')
out = sys.argv[2] if len(sys.argv) > 2 else 'ensemble31_mean_C2_spectrum.csv'

mem = open(os.path.join(D, 'ensemble_membership.txt')).read().split()
S = []; Tmax = 0
for s in mem:
    d = pd.read_csv(os.path.join(D, 'C2_SPECTRA', f'{s}__C2_spectrum.csv'))
    T, p = d.temperature_C.values.astype(float), d.C2_spectrum_raw.values.astype(float)
    p = p / np.trapezoid(p, T)
    S.append((T, p)); Tmax = max(Tmax, T.max())
G = np.arange(0.0, np.ceil(Tmax) + 0.25, 0.25)
M = np.mean([np.interp(G, T, p, left=0.0, right=0.0) for T, p in S], axis=0)
M = M / np.trapezoid(M, G)
pd.DataFrame({'T': G, 'density': M}).to_csv(out, index=False)
print(out, len(mem), 'specimens')
