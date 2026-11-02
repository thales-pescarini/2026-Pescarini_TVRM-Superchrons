# Blocking ages and moments of all spectral elements of the 31 ensemble specimens at
# 2 C/Myr (input of Fig. 7b and Tables S1-S3). Each specimen has weight 1/31 before
# the reset selection.
# Usage: python model/build_blocking_npz.py [run directory]
import os, sys
import numpy as np
from blocking import RATE, elements, load_spec, wquant

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'data', 'trout_run')

mem = open(os.path.join(D, 'ensemble_membership.txt')).read().split()
ages, wts, idx, meds = [], [], [], []
for k, s in enumerate(mem):
    T, w = load_spec(os.path.join(D, 'C2_SPECTRA', f'{s}__C2_spectrum.csv'), dec=5)
    A, W, _ = elements(T, w, RATE)
    meds.append(wquant(A, W, 0.5))
    ages.append(A); wts.append(W / 31.0); idx.append(np.full(A.size, k))
print(f'mean of medians {np.mean(meds):.1f} Ma, sd {np.std(meds, ddof=1):.1f} Myr')
np.savez('ensemble31_instantaneous_blocking_2C_Myr.npz', age_Ma=np.concatenate(ages),
         weight_mean_original_C2=np.concatenate(wts), specimen_index=np.concatenate(idx),
         specimen_names=np.array(mem))
