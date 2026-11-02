# Interval statistics in the central-age convention of Section 3.5: every element takes
# the polarity of the history at its blocking age (same as Fig. 7b).
import numpy as np
import fig7_coherence as f7

Y = f7.PANEL_B_MODEL_AGE_YOUNG
DT = f7.MODEL_TIME_STEP


def make_grid(dt=DT):
    grid = Y + (np.arange(int(round((f7.MODEL_AGE_OLD - Y) / dt))) + 0.5) * dt
    query = np.sort(np.append(grid, f7.AGE_OLD))
    return grid, query


def polarity(seeds, dt=DT):
    grid, query = make_grid(dt)
    mid = np.searchsorted(query, grid)
    at500 = int(np.searchsorted(query, f7.AGE_OLD))
    P = np.empty((len(seeds), grid.size + 1), dtype=np.float32)
    for k, s in enumerate(seeds):
        p = f7.realize(s, grid=query, young=Y)
        P[k, :-1] = p[mid]
        P[k, -1] = p[at500]
    return P


def stats(A, W, P, dt=DT):
    grid, _ = make_grid(dt)
    masks = ((A > f7.MRS_START) & (A <= f7.AGE_OLD), (A >= f7.MRS_END) & (A <= f7.MRS_START),
             (A >= f7.AGE_YOUNG) & (A < f7.MRS_END))
    MG = np.zeros((3, grid.size + 1))
    for i, m in enumerate(masks):
        atom = m & (A == f7.AGE_OLD)
        MG[i, -1] = W[atom].sum()
        o = m & ~atom
        idx = np.floor((A[o] - Y) / dt).astype(int)
        MG[i, :-1] = np.bincount(idx, weights=W[o], minlength=grid.size)
    g = MG.sum(1)
    G = g.sum()
    m = P @ MG.T            # signed moment per interval
    tot = m.sum(1)
    q = lambda x: [float(v) for v in np.percentile(x, [2.5, 50, 97.5])]
    eff = m / g
    return dict(gross_fraction=(g / G).tolist(), Cbulk=q(np.abs(tot) / G), reversed_all=bool((tot < 0).all()),
                MRS_share=q(m[:, 1] / tot), rms_eff=[float(np.sqrt(np.mean(eff[:, i]**2))) for i in range(3)])
