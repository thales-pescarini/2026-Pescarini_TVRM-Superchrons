# Paths and inputs shared by the supplementary scripts.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RUN = DATA / 'trout_run'
OUT = ROOT / 'supplementary' / 'outputs'
OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / 'model'), str(ROOT / 'figures')]


def ensemble_spectra(dec=5):
    # the 31 ensemble C2 spectra, as in Fig. 6
    from blocking import load_spec
    mem = (RUN / 'ensemble_membership.txt').read_text().split()
    return mem, [load_spec(RUN / 'C2_SPECTRA' / f'{s}__C2_spectrum.csv', dec=dec) for s in mem]
