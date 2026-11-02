# Continuous-cooling single-domain blocking model (Section 3.4), parameters of Table 1.
import numpy as np
import pandas as pd
from scipy.optimize import brentq

kB = 8.617333262e-5         # eV/K
TAU0 = 1e-9                 # s
SPY = 365.25 * 24 * 3600.0
Ma = 1e6 * SPY
T_LAB = 40 * 60.0           # s
TC_M_K = 853.15
TC_P_K = 593.15
T_PEAK = 266.0
PEAK_START = 507.0
COOLING_ONSET = 500.0
RESIDENCE = PEAK_START - COOLING_ONSET
FPYR = 0.85
MRS_START = 485.0
MRS_END = 466.1
RATE = 2.0                  # C/Myr


def bM(TK):
    return (np.maximum(1 - TK / TC_M_K, 1e-12) / (1 - 273.15 / TC_M_K)) ** (0.40025 * 2.0)


def bP(TK):
    return (np.maximum(1 - TK / TC_P_K, 1e-12) / (1 - 273.15 / TC_P_K)) ** (0.52 * 2.6)


def ref_barrier(TubC, bf):
    TubK = TubC + 273.15
    return kB * TubK * np.log(T_LAB / TAU0) / bf(TubK)


def log_tau_at(TC_, TubC, bf):
    TK = TC_ + 273.15
    TubK = TubC + 273.15
    return np.log(TAU0) + np.log(T_LAB / TAU0) * (TubK / TK) * (bf(TK) / bf(TubK))


def reset_threshold(bf, res, lo, hi):
    tgt = np.log(res * Ma)
    return brentq(lambda T: log_tau_at(T_PEAK, T, bf) - tgt, lo, hi)


C_P = reset_threshold(bP, RESIDENCE, 200.0, 319.5)
C_M = reset_threshold(bM, RESIDENCE, 300.0, 560.0)


def blocking_solution(TubC, bf, TcK, alpha, n, ceil, rate):
    # blocking age (Ma), T_B (C) and 5*theta (Myr); nan if not reset or never blocked
    if TubC >= ceil:
        return np.nan, np.nan, np.nan
    de0 = ref_barrier(TubC, bf)
    cr = rate / Ma

    def theta_at(TBC):
        TBK = TBC + 273.15
        E = de0 * bf(TBK)
        q = 1 + alpha * n * TBK / (TcK - TBK)
        if E <= 0 or q <= 0 or not np.isfinite(E) or not np.isfinite(q):
            return np.nan
        return kB * TBK**2 / (E * q * cr)

    def residual(TBC):
        TBK = TBC + 273.15
        E = de0 * bf(TBK)
        th = theta_at(TBC)
        if not np.isfinite(th) or th <= 0:
            return np.nan
        return E / (kB * TBK) - np.log(1.78 * th / TAU0)

    top = T_PEAK - 1e-6
    rt = residual(top)
    if np.isfinite(rt) and rt > 0:          # already blocked when cooling starts
        th = theta_at(T_PEAK)
        if not np.isfinite(th) or th <= 0:
            return np.nan, np.nan, np.nan
        return COOLING_ONSET, T_PEAK, 5.0 * th / Ma
    rb = residual(0.0)
    if not np.isfinite(rb) or rb < 0:       # still unblocked at 0 C
        return np.nan, np.nan, np.nan
    try:
        TBC = brentq(residual, 0.0, top)
    except Exception:
        return np.nan, np.nan, np.nan
    th = theta_at(TBC)
    if not np.isfinite(th) or th <= 0:
        return np.nan, np.nan, np.nan
    return COOLING_ONSET - (T_PEAK - TBC) / rate, TBC, 5.0 * th / Ma


def elements(Ts, wT, rate, fpyr=FPYR):
    # blocking age, weight and theta (Myr) of every reset spectral element;
    # below 320 C a fraction fpyr of the moment is carried by pyrrhotite
    A, W, TH = [], [], []

    def add(T, w, args):
        a, _, d = blocking_solution(T, *args, rate)
        if np.isfinite(a) and np.isfinite(d) and d > 0:
            A.append(a); W.append(w); TH.append(d / 5.0)

    for T, w in zip(Ts, wT):
        if T >= 320.0:
            add(T, w, (bM, TC_M_K, 0.40025, 2.0, C_M))
        else:
            if T < C_P:
                add(T, w * fpyr, (bP, TC_P_K, 0.52, 2.6, C_P))
            add(T, w * (1 - fpyr), (bM, TC_M_K, 0.40025, 2.0, C_M))
    return np.array(A), np.array(W), np.array(TH)


def load_spec(path, tcol='temperature_C', dcol='C2_spectrum_raw', dec=5):
    # spectrum as weights w_i = rho_i dT_i (trapezoidal cells), every dec-th step
    d = pd.read_csv(path)
    T = d[tcol].values.astype(float)
    r = d[dcol].values.astype(float)
    ok = np.isfinite(T) & np.isfinite(r)
    T, r = T[ok], r[ok]
    o = np.argsort(T)
    T, r = T[o], r[o]
    uT, inv = np.unique(T, return_inverse=True)
    if uT.size != T.size:
        s = np.zeros_like(uT); c = np.zeros_like(uT)
        np.add.at(s, inv, r); np.add.at(c, inv, 1.0)
        T, r = uT, s / c
    if dec > 1:
        T, r = T[::dec], r[::dec]
    dT = np.empty_like(T)
    dT[0] = 0.5 * (T[1] - T[0]); dT[-1] = 0.5 * (T[-1] - T[-2]); dT[1:-1] = 0.5 * (T[2:] - T[:-2])
    w = r * dT
    return T, w / w.sum()


def wquant(A, W, p):
    o = np.argsort(A)
    c = np.cumsum(W[o]) / W.sum()
    return float(np.interp(p, c, A[o]))


def gumbel_age_density(GA, A, W, TH):
    # moment-weighted density of acquisition ages; Gumbel kernel of scale theta
    # with its mean at the blocking age
    z = (A[None, :] - GA[:, None]) / TH[None, :] + 0.5772156649015329
    k = np.exp(-z - np.exp(-z)) / TH[None, :]
    return (W[None, :] * k).sum(1)
