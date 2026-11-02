# Table S3 / Text S3: numerical convergence of the central-age statistics: time step,
# spectral sampling, number and seeds of histories, and a bootstrap over histories.
import json
import numpy as np, pandas as pd
import common
import blocking as P
import cstats


def build(dec):
    mem, specs = common.ensemble_spectra(dec=dec); A = []; W = []; meds = []
    for T, w in specs:
        a, ww, _ = P.elements(T, w, 2.0)
        A.append(a); W.append(ww / len(specs)); meds.append(P.wquant(a, ww, .5))
    return np.concatenate(A), np.concatenate(W), np.array(meds)


A5, W5, m5 = build(5)
ref = [200 + i for i in range(400)]
Pref = cstats.polarity(ref)
fmt = lambda q, d=3: f'{q[1]:.{d}f} ({q[0]:.{d}f}–{q[2]:.{d}f})'


def row(name, st):
    return dict(variation=name, C_bulk=fmt(st['Cbulk']), MRS_share=fmt(st['MRS_share']),
                RMS_eff_pre_post=f"{st['rms_eff'][0]:.3f}/{st['rms_eff'][2]:.3f}",
                resultant_reversed_all=st['reversed_all'])


rows = [row('Reference (0.01 Myr, every 5th step, seeds 200–599)', cstats.stats(A5, W5, Pref)),
        row('Time step 0.005 Myr', cstats.stats(A5, W5, cstats.polarity(ref, dt=0.005), dt=0.005))]
A1, W1, m1 = build(1)
rows += [row('Every temperature step', cstats.stats(A1, W1, Pref)),
         row('First 100 histories', cstats.stats(A5, W5, Pref[:100])),
         row('First 200 histories', cstats.stats(A5, W5, Pref[:200])),
         row('Independent seeds 600–999', cstats.stats(A5, W5, cstats.polarity([600 + i for i in range(400)])))]
for r in rows:
    print(r, flush=True)
pd.DataFrame(rows).to_csv(common.OUT / 'TableS3_convergence.csv', index=False)

rng = np.random.default_rng(1); bs = []
for b in range(500):
    st = cstats.stats(A5, W5, Pref[rng.integers(0, 400, 400)])
    bs.append([st['Cbulk'][1], st['MRS_share'][1], st['rms_eff'][0], st['rms_eff'][2]])
out = dict(bootstrap_95pct=dict(zip(['Cbulk_median', 'MRS_share_median', 'rms_pre', 'rms_post'],
                                    np.percentile(np.array(bs), [2.5, 97.5], axis=0).T.round(4).tolist())),
           n_subelements=dict(every_5th_step=int(A5.size), every_step=int(A1.size)),
           mean_of_medians_Ma=dict(every_5th_step=round(float(m5.mean()), 2), every_step=round(float(m1.mean()), 2)),
           max_abs_specimen_median_diff_Myr=round(float(np.max(np.abs(m1 - m5))), 3))
print(json.dumps(out, indent=1))
json.dump(out, open(common.OUT / 'TableS3_bootstrap.json', 'w'), indent=1)
