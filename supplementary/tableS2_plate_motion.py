# Table S2 / Text S2: plate-motion sensitivity, remanence parallel to the field.
# (i) element-level RMS coherence (telegraph limit) for the geometry with the largest
#     rotation of the field direction between 500 and 430 Ma;
# (ii) bulk efficiency (central-age convention), 400 histories, 64 Euler axes x 11 latitudes.
import json
import numpy as np, pandas as pd
import common
import fig7_coherence as f7, cstats, plate_geometry as pg
from blocking import elements

SPEEDS = [1., 2.5, 5., 10., 20.]
LAMS = np.array([0.05, 0.1, 0.5, 1., 2., 3., 5., 10., 20.])
axes = pg.fibonacci_axes(64)

# (i) element level, ensemble-mean spectrum (every 5th step)
Ts, wT, _ = f7.load_trout_spectrum(common.DATA / 'ensemble31_mean_C2_spectrum.csv')
A, W, th = elements(Ts[::5], wT[::5], 2.0)
grid = f7.tf
phi = f7.gumbel_density(grid, A, th)
phi /= phi.sum(1)[:, None]


def srms(mt):
    rho = np.exp(-2 * LAMS * f7.MODEL_TIME_STEP)[None, :, None]
    h = np.zeros((A.size, LAMS.size, 3)); q = np.zeros((A.size, LAMS.size))
    for j in range(grid.size):
        fj = (phi[:, j][:, None] * mt[j][None, :])[:, None, :]
        q += (fj * fj).sum(-1) + 2 * (fj * h).sum(-1)
        h = rho * (h + fj)
    return (W @ np.sqrt(np.maximum(q, 0))) / W.sum()


best = None
for lat in pg.PALEOLATITUDES_DEG:
    for ai, ax in enumerate(axes):
        m = pg.gad_response(np.array([500., 430.]), lat, ax, 10.)
        ang = np.degrees(np.arccos(np.clip(m[0] @ m[1], -1, 1)))
        if best is None or ang > best[0]:
            best = (ang, lat, ai)
ax = axes[best[2]]
m0 = pg.gad_response(np.array([pg.REFERENCE_AGE_MA]), best[1], ax, 0.)[0]
base = srms(np.tile(m0, (grid.size, 1)))
closed = float(W @ np.sqrt(f7.gumbel_element_ms_coherence(2 * 2.0 * th)) / W.sum())
el = {sp: srms(pg.gad_response(grid, best[1], ax, sp)) - base for sp in SPEEDS}

# (ii) bulk
d = np.load(common.DATA / 'ensemble31_instantaneous_blocking_2C_Myr.npz')
Ab, Wb = d['age_Ma'], d['weight_mean_original_C2']
Pol = cstats.polarity([200 + i for i in range(400)])
gg, _ = cstats.make_grid()
ins = (Ab >= 430) & (Ab <= 500); atom = ins & (Ab == 500); o = ins & ~atom
gb = np.zeros(gg.size + 1)
gb[:-1] = np.bincount(np.floor((Ab[o] - cstats.Y) / f7.MODEL_TIME_STEP).astype(int), weights=Wb[o], minlength=gg.size)
gb[-1] = Wb[atom].sum()
ages = np.append(gg, 500.0); G = gb.sum(); k = gb > 0
S = Pol[:, k].astype(float) * gb[k][None, :]
q0 = S.sum(1); C0 = np.abs(q0) / G
res = {sp: [] for sp in SPEEDS}
for lat in pg.PALEOLATITUDES_DEG:
    for axb in axes:
        for sp in SPEEDS:
            mt = pg.gad_response(ages[k], lat, axb, sp)
            mref = pg.gad_response(np.array([pg.REFERENCE_AGE_MA]), lat, axb, sp)[0]
            V = S @ mt
            dC = np.abs(np.linalg.norm(V, axis=1) / G - C0)
            Vf = q0[:, None] * mref[None, :]
            dots = np.einsum('ij,ij->i', V, Vf)
            ang = np.degrees(np.arccos(np.clip(dots / np.maximum(np.linalg.norm(V, axis=1) * np.linalg.norm(Vf, axis=1), 1e-30), -1, 1)))
            res[sp].append((np.median(dC), np.percentile(dC, 95), np.mean(dots > 0), np.median(ang)))
rows = []
for sp in SPEEDS:
    r = np.array(res[sp])
    rows.append(dict(v_cm_per_yr=sp, dCbarRMS_lambda_0p05=float(el[sp][0]), dCbarRMS_lambda_2=float(el[sp][4]),
                     max_median_abs_dCbulk=r[:, 0].max(), max_p95_abs_dCbulk=r[:, 1].max(),
                     reversed_resultant_pct=100 * r[:, 2].min(), angle_median_across_geometries_deg=np.median(r[:, 3]),
                     angle_largest_median_deg=r[:, 3].max()))
df = pd.DataFrame(rows)
print(df.to_string())
df.to_csv(common.OUT / 'TableS2_plate_motion.csv', index=False)
info = dict(geometries_per_speed=len(res[1.]), worst_geometry=dict(lat=float(best[1]), axis=int(best[2])),
            n_subelements=int(A.size), fixed_CbarRMS_lambda2=float(base[4]), closed_form_lambda2=closed,
            reference_Cbulk_median=float(np.median(C0)))
json.dump(info, open(common.OUT / 'TableS2_plate_motion.json', 'w'), indent=1)
print(info)
