# Fig. 7: element-wise reversal-coherence filter (Section 3.5).
# (a) local RMS coherence at 2 C/Myr, from the ensemble-mean C2 spectrum;
# (b) coherent-moment efficiency per interval, 31 specimens at 2 C/Myr;
# (c) moment-weighted RMS coherence versus reversal rate for several cooling rates.
# Also writes Fig7_MRS_SHARE.json and prints the time-step convergence test (Text S3).
# Usage: python figures/fig7_coherence.py

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, MultipleLocator
from scipy.special import digamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import blocking as B

DATA = ROOT / 'data'

RATES = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
REFERENCE_RATE = 2.0
N_REALIZATIONS = 400
N_DISPLAY_REALIZATIONS = 100
ENSEMBLE_SEED0 = 200
DISPLAY_SEED = 30
MIN_RENDER_SIGMA_MYR = 1.0      # display only: age resolution of the measured spectrum
GROSS_SUPPORT_THRESHOLD = 0.010

# Moyero Reversed Superchron: onset 485 Ma, end 466.1 Ma (Ahanin et al. 2026)
MRS_START = B.MRS_START
MRS_END = B.MRS_END
MIX_LO = 454.0
ORD_SIL_BOUNDARY = 443.8

AGE_OLD = 500.0
AGE_YOUNG = 430.0
PLOT_AGE_STEP = 0.02
MODEL_AGE_OLD = 510.0
MODEL_AGE_YOUNG = 340.0
MODEL_TIME_STEP = 0.01
PANEL_B_MODEL_AGE_YOUNG = 405.0

# Documented zones after Pavlov & Gallet (2005); +1 normal, -1 reversed
DOCUMENTED_ZONES = [
    (MRS_END, MRS_START, -1.0),     # MRS
    (465.2, MRS_END, +1.0),         # N1
    (462.2, 465.2, -1.0),           # R2
    (460.4, 462.2, -1.0),           # poorly documented, continued as reversed
    (457.5, 460.4, +1.0),           # N2
    (455.5, 457.5, -1.0),           # R3
    (455.2, 455.5, +1.0),           # short normal zone between R3 and R4
    (454.0, 455.2, -1.0),           # R4
]
# Cambrian reversal rates (Myr^-1), after Li et al. (2023)
CAMBRIAN_SEGMENTS = [(MRS_START, 490.0, 3.0), (490.0, 497.0, 12.0), (497.0, 510.0, 25.0)]
LAM_LATE_ORD = 1.5              # 454-443.8 Ma, Hounslow et al. (2021)
LAM_SILURIAN = 3.0              # younger than 443.8 Ma, Hounslow et al. (2021)

# reversal-rate fields shown in panel (c)
SUPERCHRON_SCALE_LO, SUPERCHRON_SCALE_HI = 0.01, 0.05
PHANEROZOIC_LO, PHANEROZOIC_HI = 1.0, 5.0
HYPER_LO, HYPER_HI = 10.0, 30.0
LAMBDA_PROBE = 2.0

PANEL_B_PERIODS = (('Pre-MRS', MRS_START, AGE_OLD), ('MRS', MRS_END, MRS_START),
                   ('Post-MRS', AGE_YOUNG, MRS_END))


def load_trout_spectrum(path):
    # element weights w_i proportional to rho_i * dT_i (trapezoidal cell widths)
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=float, encoding='utf-8-sig')
    names = list(data.dtype.names)
    T = np.asarray(data[names[0]], dtype=float).reshape(-1)
    rho = np.asarray(data[names[1]], dtype=float).reshape(-1)
    ok = np.isfinite(T) & np.isfinite(rho)
    T, rho = T[ok], rho[ok]
    o = np.argsort(T)
    T, rho = T[o], rho[o]
    uT, inv = np.unique(T, return_inverse=True)
    if uT.size != T.size:
        s = np.zeros_like(uT, dtype=float)
        n = np.zeros_like(uT, dtype=float)
        np.add.at(s, inv, rho)
        np.add.at(n, inv, 1.0)
        T, rho = uT, s / n
    dT = np.empty_like(T)
    dT[0] = 0.5 * (T[1] - T[0])
    dT[-1] = 0.5 * (T[-1] - T[-2])
    dT[1:-1] = 0.5 * (T[2:] - T[:-2])
    m = rho * dT
    return T, m / m.sum(), dT


tf = np.arange(MODEL_AGE_YOUNG, MODEL_AGE_OLD + 0.5 * MODEL_TIME_STEP, MODEL_TIME_STEP)
age_plot = np.arange(AGE_YOUNG, AGE_OLD + 0.5 * PLOT_AGE_STEP, PLOT_AGE_STEP)
age_render = np.arange(MODEL_AGE_YOUNG, AGE_OLD + 0.5 * PLOT_AGE_STEP, PLOT_AGE_STEP)


def telegraph(s, grid, rng, lo, hi, rate, base=-1.0):
    # symmetric telegraph process on [lo, hi), starting from `base` at the old edge
    s[(grid >= lo) & (grid < hi)] = base
    count = rng.poisson(rate * (hi - lo))
    for edge in np.sort(rng.uniform(lo, hi, count))[::-1]:
        s[(grid >= lo) & (grid < edge)] *= -1.0


def realize(seed, grid=tf, young=None):
    # one polarity history: stochastic Cambrian, documented zones 485-454 Ma,
    # stochastic Late Ordovician and Silurian
    rng = np.random.default_rng(seed)
    s = np.full_like(grid, -1.0, dtype=float)
    edges = [MRS_START]
    for lo, hi, rate in CAMBRIAN_SEGMENTS:
        count = rng.poisson(rate * (hi - lo))
        if count > 0:
            edges.extend(np.sort(rng.uniform(lo, hi, count)).tolist())
    for edge in sorted(edges):
        s[grid >= edge] *= -1.0
    for lo, hi, polarity in DOCUMENTED_ZONES:
        s[(grid >= lo) & (grid < hi)] = polarity
    telegraph(s, grid, rng, ORD_SIL_BOUNDARY, MIX_LO, LAM_LATE_ORD, base=-1.0)
    base = float(s[min(int(np.searchsorted(grid, ORD_SIL_BOUNDARY, side='left')), grid.size - 1)])
    telegraph(s, grid, rng, MODEL_AGE_YOUNG if young is None else young,
              ORD_SIL_BOUNDARY, LAM_SILURIAN, base=base)
    return s


EULER_GAMMA = 0.5772156649015329
GUMBEL_SD_PER_THETA = np.pi / np.sqrt(6.0)


def gumbel_z(age, centres, theta):
    return (centres - age) / theta + EULER_GAMMA


def gumbel_density(grid, centres, theta):
    # Gumbel acquisition kernel of scale theta centred on each blocking age (rows)
    centres = np.asarray(centres, dtype=float)
    th = np.asarray(theta, dtype=float)
    if th.ndim == 0:
        th = np.full(centres.shape, float(th))
    th = np.maximum(th, 1e-12)
    z = gumbel_z(grid[None, :], centres[:, None], th[:, None])
    return np.exp(-z - np.exp(-z)) / th[:, None]


def gumbel_cdf_age(age, centres, theta):
    z = gumbel_z(np.asarray(age, dtype=float), np.asarray(centres, dtype=float),
                 np.asarray(theta, dtype=float))
    return 1.0 - np.exp(-np.exp(-z))


def gumbel_element_ms_coherence(a):
    # <c^2> of a stationary telegraph seen through a Gumbel kernel, a = 2 lambda theta
    a = np.asarray(a, dtype=float)
    out = np.ones_like(a)
    m = a > 1e-12
    out[m] = a[m] * (digamma(0.5 * (a[m] + 1.0)) - digamma(0.5 * a[m])) - 1.0
    return np.clip(out, 0.0, 1.0)


def reference_profile(Ts, wT, pol, rate):
    A, W, theta = B.elements(Ts, wT, rate)
    lost = gumbel_cdf_age(MODEL_AGE_YOUNG, A, theta) + 1.0 - gumbel_cdf_age(MODEL_AGE_OLD, A, theta)
    assert np.sum(W * lost) / W.sum() < 1e-6

    phi = gumbel_density(tf, A, theta)
    phi /= phi.sum(axis=1, keepdims=True)
    coh = pol @ phi.T                       # coherence of element i in history r

    kernel = gumbel_density(age_render, A, np.maximum(theta, MIN_RENDER_SIGMA_MYR / GUMBEL_SD_PER_THETA))
    gross_full = W @ kernel
    peak = np.max(gross_full)
    coh_full = (coh * W[None, :]) @ kernel / peak
    i0 = int(round((AGE_YOUNG - MODEL_AGE_YOUNG) / PLOT_AGE_STEP))
    gross = (gross_full / peak)[i0:i0 + age_plot.size]
    coh_moment = coh_full[:, i0:i0 + age_plot.size]
    support = gross >= GROSS_SUPPORT_THRESHOLD
    frac = np.full_like(coh_moment, np.nan)
    frac[:, support] = coh_moment[:, support] / gross[None, support]
    frac_rms = np.full(age_plot.shape, np.nan)
    frac_rms[support] = np.sqrt(np.mean(frac[:, support] ** 2, axis=0))
    frac_rms[support] = np.clip(frac_rms[support], 0.0, 1.0)
    return dict(A=A, W=W, theta=theta, coh=coh, gross_tf=W @ phi, frac=frac, frac_rms=frac_rms)


def convergence_check(ages, theta, seeds):
    # RMS coherence of 24 elements with 0.01 and 0.005 Myr polarity grids
    idx = np.where(ages - 2.5 * theta >= AGE_YOUNG)[0]
    pick = idx[np.linspace(0, idx.size - 1, min(24, idx.size), dtype=int)]
    out = []
    for step in (0.010, 0.005):
        grid = np.arange(MODEL_AGE_YOUNG, MODEL_AGE_OLD + 0.5 * step, step)
        pol = np.stack([realize(s, grid) for s in seeds], axis=0)
        phi = gumbel_density(grid, ages[pick], theta[pick])
        phi /= phi.sum(axis=1, keepdims=True)
        out.append(np.sqrt(np.mean((pol @ phi.T) ** 2, axis=0)))
    return float(np.max(np.abs(out[1] - out[0])))


def panel_b_efficiencies(npz_path):
    # |sum(W p)| / sum(W) within each interval, central blocking ages, 400 histories
    d = np.load(npz_path)
    ages, masses = d['age_Ma'], d['weight_mean_original_C2']
    grid = PANEL_B_MODEL_AGE_YOUNG + (np.arange(int(round(
        (MODEL_AGE_OLD - PANEL_B_MODEL_AGE_YOUNG) / MODEL_TIME_STEP))) + 0.5) * MODEL_TIME_STEP
    masks = ((ages > MRS_START) & (ages <= AGE_OLD),
             (ages >= MRS_END) & (ages <= MRS_START),
             (ages >= AGE_YOUNG) & (ages < MRS_END))
    mass = np.zeros((3, grid.size + 1))
    for i, mask in enumerate(masks):
        atom = mask & (ages == AGE_OLD)     # elements blocked at 500 Ma
        mass[i, -1] = masses[atom].sum()
        rest = mask & ~atom
        k = np.floor((ages[rest] - PANEL_B_MODEL_AGE_YOUNG) / MODEL_TIME_STEP).astype(int)
        mass[i, :-1] = np.bincount(k, weights=masses[rest], minlength=grid.size)
    gross = mass.sum(axis=1)
    query = np.sort(np.append(grid, AGE_OLD))
    mid = np.searchsorted(query, grid)
    at500 = int(np.searchsorted(query, AGE_OLD))
    signed = np.empty((N_REALIZATIONS, 3))
    for row in range(N_REALIZATIONS):
        p = realize(ENSEMBLE_SEED0 + row, grid=query, young=PANEL_B_MODEL_AGE_YOUNG)
        signed[row] = (mass @ np.append(p[mid], p[at500])) / gross
    signed = np.clip(signed, -1, 1)
    mag = np.abs(signed)
    summary = []
    for i, (name, lo, hi) in enumerate(PANEL_B_PERIODS):
        summary.append({
            'period': name, 'age_old_Ma': hi, 'age_young_Ma': lo, 'duration_Myr': hi - lo,
            'n_realizations': N_REALIZATIONS,
            'gross_mass_of_original_mean_C2': float(gross[i]),
            'gross_fraction_of_500_430_population': float(gross[i] / gross.sum()),
            'rms_efficiency': float(np.sqrt(np.mean(signed[:, i]**2))),
            'median_efficiency': float(np.median(mag[:, i])),
            'q05_efficiency': float(np.quantile(mag[:, i], .05)),
            'q95_efficiency': float(np.quantile(mag[:, i], .95)),
            'mean_signed_efficiency': float(np.mean(signed[:, i])),
        })
    return mag, summary


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'Fig7'
    mag_b, summary_b = panel_b_efficiencies(DATA / 'ensemble31_instantaneous_blocking_2C_Myr.npz')
    Ts, wT, _ = load_trout_spectrum(DATA / 'ensemble31_mean_C2_spectrum.csv')

    s_display = realize(DISPLAY_SEED)
    pol = np.stack([realize(ENSEMBLE_SEED0 + i) for i in range(N_REALIZATIONS)], axis=0)
    ref = reference_profile(Ts, wT, pol, REFERENCE_RATE)

    # share of the net coherent moment acquired within the MRS;
    # g(t) is the gross acquisition density of the reset moment
    g = ref['gross_tf']
    g_tot = float(g.sum())
    in_mrs = (tf >= MRS_END) & (tf <= MRS_START)
    pre = tf > MRS_START
    post = tf < MRS_END
    m_tot = pol @ g
    m_mrs = pol[:, in_mrs] @ g[in_mrs]
    m_pre = pol[:, pre] @ g[pre]
    m_post = pol[:, post] @ g[post]
    share = m_mrs / m_tot
    # same share attributed by central blocking age of each element
    el_mrs = (ref['A'] >= MRS_END) & (ref['A'] <= MRS_START)
    c_mrs = (ref['coh'][:, el_mrs] * ref['W'][None, el_mrs]).sum(axis=1) / ref['W'].sum()
    c_out = (ref['coh'][:, ~el_mrs] * ref['W'][None, ~el_mrs]).sum(axis=1) / ref['W'].sum()
    share_el = c_mrs / (c_mrs + c_out)

    def q(x):
        return [float(np.percentile(x, p)) for p in (2.5, 50.0, 97.5)]

    mrs_share = {
        'definition': 'm_MRS/m_total per realisation; m = sum_t s_r(t) g(t), g = gross acquisition density of the reset moment (physical Gumbel kernels)',
        'cooling_rate_C_per_Myr': REFERENCE_RATE,
        'n_realisations': int(pol.shape[0]),
        'gross_fraction_acquired': {'pre_MRS': float(g[pre].sum() / g_tot),
                                    'MRS': float(g[in_mrs].sum() / g_tot),
                                    'post_MRS': float(g[post].sum() / g_tot)},
        'net_coherent_moment_over_reset_moment_p2.5_p50_p97.5': q(np.abs(m_tot) / g_tot),
        'net_moment_reversed_in_all_realisations': bool(np.all(m_tot < 0)),
        'MRS_share_of_net_coherent_moment_p2.5_p50_p97.5': q(share),
        'MRS_share_mean': float(np.mean(share)),
        'pre_MRS_signed_over_net_p50': float(np.median(m_pre / m_tot)),
        'post_MRS_signed_over_net_p50': float(np.median(m_post / m_tot)),
        'check_element_attribution_MRS_share_p2.5_p50_p97.5': q(share_el),
    }
    print('MRS share of net coherent moment:', json.dumps(mrs_share, indent=1))
    diff = convergence_check(ref['A'], ref['theta'], [ENSEMBLE_SEED0 + i for i in range(32)])
    print(f'time-step convergence: max RMS difference = {diff:.4f}')

    # panel (c): analytical RMS coherence, 31-specimen ensemble
    lambda_grid = np.geomspace(0.01, 50.0, 900)
    run = DATA / 'trout_run'
    members = (run / 'ensemble_membership.txt').read_text().split()
    spectra = [B.load_spec(run / 'C2_SPECTRA' / f'{m}__C2_spectrum.csv') for m in members]
    curves = {}
    for rate in RATES:
        Wl, Tl = [], []
        for T_e, w_e in spectra:
            _, W_e, th_e = B.elements(T_e, w_e, rate)
            Wl.append(W_e / len(spectra)); Tl.append(th_e)
        W = np.concatenate(Wl); theta = np.concatenate(Tl)
        rms = np.empty_like(lambda_grid)
        for j in range(0, lambda_grid.size, 60):
            lam = lambda_grid[j:j + 60]
            el = np.sqrt(gumbel_element_ms_coherence(2.0 * lam[:, None] * theta[None, :]))
            rms[j:j + 60] = (el @ W) / W.sum()
        curves[rate] = rms
        print(f'(c) {rate:g} C/Myr: weighted theta = {np.sum(W * theta) / W.sum():.3f} Myr, '
              f'retained mass = {W.sum():.4f}')

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
                         'font.size': 8.6, 'axes.linewidth': 0.95, 'mathtext.fontset': 'dejavusans'})
    GRN, GRN_FILL, DGRN = '#3E9B6C', '#A7D8BE', '#1F6E47'
    SLATE, SLATE_FILL, DSLATE = '#5E7E9E', '#A6BDD4', '#3A5876'
    GRY, GRY_DARK, LIGHT_GRY = '#A7ADB3', '#80868D', '#D9DDE1'
    INK, BLACK = '#232323', '#1A1A1A'
    SUPERCHRON_BG, PHANEROZOIC_BG, HYPER_BG, HYPER_TEXT = '#DCEFE3', '#E5ECF2', '#F1E3DC', '#8A5B45'
    halo = [pe.Stroke(linewidth=2.8, foreground='white'), pe.Normal()]

    # cooling-rate colours: inferno, lightened; reference rate in green
    inferno = plt.get_cmap('inferno')
    t_inf = {1.0: 0.150, 3.0: 0.285, 5.0: 0.375, 10.0: 0.550, 30.0: 0.650}

    def rate_colour(rate):
        if rate == REFERENCE_RATE:
            return GRN
        rgba = np.array(mcolors.to_rgba(inferno(t_inf[rate])), dtype=float)
        c = (1.0 - 0.32) * rgba + 0.32 * np.array([1.0, 1.0, 1.0, rgba[3]])
        c[3] = rgba[3]
        return tuple(c)

    colour = {rate: rate_colour(rate) for rate in RATES}

    fig = plt.figure(figsize=(7.2, 6.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.17, 0.88], hspace=0.17,
                             left=0.085, right=0.985, top=0.985, bottom=0.075)
    top = outer[0].subgridspec(2, 1, height_ratios=[0.34, 1.0], hspace=0.12)
    ax_pol = fig.add_subplot(top[0])
    ax_a = fig.add_subplot(top[1], sharex=ax_pol)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.27)
    ax_b = fig.add_subplot(bot[0])
    ax_c = fig.add_subplot(bot[1])

    for ax in (ax_pol, ax_a):
        ax.set_xlim(AGE_OLD, AGE_YOUNG)
        for x in (MRS_START, MRS_END, MIX_LO, ORD_SIL_BOUNDARY):
            ax.axvline(x, color=GRY, lw=0.65, ls=(0, (3, 3)), alpha=0.55, zorder=0)

    # polarity bar
    ax_pol.set_ylim(0.0, 1.0)
    h0, h1 = 0.72, 0.99
    segments = [(AGE_OLD, MRS_START, SLATE_FILL, 'Cambrian\n(stochastic)', DSLATE),
                (MRS_START, MRS_END, GRN_FILL, 'Moyero Superchron', DGRN),
                (MRS_END, MIX_LO, SLATE_FILL, 'L. Darriwilian–\nSandbian', DSLATE),
                (MIX_LO, ORD_SIL_BOUNDARY, SLATE_FILL, 'Late Ordovician\n(stochastic)', DSLATE),
                (ORD_SIL_BOUNDARY, AGE_YOUNG, SLATE_FILL, 'Early Silurian\n(stochastic)', DSLATE)]
    for x0, x1, fc, label, tc in segments:
        ax_pol.add_patch(Rectangle((min(x0, x1), h0), abs(x1 - x0), h1 - h0, facecolor=fc,
                                   edgecolor=GRY_DARK, lw=0.85, hatch=None, clip_on=False, zorder=2))
        ax_pol.text(0.5 * (x0 + x1), 0.5 * (h0 + h1), label, ha='center', va='center', fontsize=6.6,
                    color=tc, fontweight='bold', linespacing=0.95, zorder=3)
    s0, s1 = 0.05, 0.53
    ax_pol.add_patch(Rectangle((AGE_YOUNG, s0), AGE_OLD - AGE_YOUNG, s1 - s0,
                               facecolor='white', edgecolor='none', zorder=2))
    shown = (tf >= AGE_YOUNG) & (tf <= AGE_OLD)
    ax_pol.fill_between(tf[shown], s0, s1, where=(s_display[shown] > 0.0), facecolor=BLACK,
                        step='mid', lw=0.0, zorder=3)
    ax_pol.add_patch(Rectangle((AGE_YOUNG, s0), AGE_OLD - AGE_YOUNG, s1 - s0, facecolor='none',
                               edgecolor=INK, lw=1.05, clip_on=False, zorder=6))
    ax_pol.text(AGE_OLD, 0.625, 'Polarity: N = black, R = white', ha='left', va='center',
                fontsize=6.4, color=INK)
    p_pol, p_a = ax_pol.get_position(), ax_a.get_position()
    y_strip = p_pol.y0 + s0 * p_pol.height
    gap = y_strip - p_a.y1
    y_lam = ((0.5 * (y_strip + p_a.y1) - 0.07 * gap) - p_pol.y0) / p_pol.height
    for x, text in [(0.5 * (AGE_OLD + MRS_START), r'$\lambda$ = 25 $\rightarrow$ 12 $\rightarrow$ 3 Myr$^{-1}$'),
                    (0.5 * (MIX_LO + ORD_SIL_BOUNDARY), r'$\lambda$ = 1.5 Myr$^{-1}$'),
                    (0.5 * (ORD_SIL_BOUNDARY + AGE_YOUNG), r'$\lambda$ = 3 Myr$^{-1}$')]:
        ax_pol.text(x, y_lam, text, transform=ax_pol.get_xaxis_transform(), ha='center',
                    va='center', fontsize=6.4, color=INK, clip_on=False)
    ax_pol.set_yticks([])
    ax_pol.tick_params(axis='x', which='both', labelbottom=False, length=0)
    for spine in ax_pol.spines.values():
        spine.set_visible(False)

    # (a)
    ax_a.set_ylim(0.0, 1.0)
    ax_a.axhline(0.0, color=LIGHT_GRY, lw=0.65, ls=':', zorder=0)
    for i in np.linspace(0, N_REALIZATIONS - 1, N_DISPLAY_REALIZATIONS, dtype=int):
        ax_a.plot(age_plot, np.abs(ref['frac'][i]), color=GRY_DARK, lw=0.76, alpha=0.155, zorder=1)
    ax_a.plot(age_plot, ref['frac_rms'], color=GRN, lw=2.55, path_effects=halo, zorder=3)
    ax_a.set_ylabel('Local RMS\ncoherence filter', fontsize=8.9)
    ax_a.set_yticks([0.0, 0.5, 1.0])
    ax_a.set_xlabel('Age (Ma)', fontsize=8.0)
    xb = ax_b.get_position().x0 - 0.165 * ax_b.get_position().width
    ax_pol.text(xb, 0.99, '(a)', transform=mtransforms.blended_transform_factory(fig.transFigure, ax_pol.transAxes),
                ha='left', va='top', fontsize=9.0, fontweight='bold', zorder=10, clip_on=False)
    ax_a.xaxis.set_major_locator(MultipleLocator(10.0))
    ax_a.xaxis.set_minor_locator(MultipleLocator(5.0))
    ax_a.tick_params(labelsize=7.8, length=3.0)
    ax_a.legend(handles=[Line2D([0], [0], color=GRY_DARK, lw=1.0, alpha=0.40,
                                label=f'{N_DISPLAY_REALIZATIONS} stochastic realisations'),
                         Line2D([0], [0], color=GRN, lw=2.55, label='RMS local coherence')],
                loc='upper right', frameon=False, fontsize=6.7, handlelength=2.4)

    # (c)
    ytop = 1.16
    ax_c.set_xscale('log')
    ax_c.set_xlim(0.01, 50.0)
    ax_c.set_ylim(0.0, ytop)
    ax_c.text(-0.165, 1.025, '(c)', transform=ax_c.transAxes, ha='left', va='top',
              fontsize=9.0, fontweight='bold', zorder=10, clip_on=False)
    ax_c.axvspan(SUPERCHRON_SCALE_LO, SUPERCHRON_SCALE_HI, facecolor=SUPERCHRON_BG, alpha=0.82, zorder=0)
    ax_c.axvspan(PHANEROZOIC_LO, PHANEROZOIC_HI, facecolor=PHANEROZOIC_BG, alpha=0.72, zorder=0)
    ax_c.axvspan(HYPER_LO, HYPER_HI, facecolor=HYPER_BG, alpha=0.78, zorder=0)
    ax_c.axhline(1.0, color=GRY_DARK, lw=0.85, zorder=5)
    for x, text, c in [(np.sqrt(SUPERCHRON_SCALE_LO * SUPERCHRON_SCALE_HI), 'Superchron mode\n$\\lambda \\approx 0$', DGRN),
                       (np.sqrt(PHANEROZOIC_LO * PHANEROZOIC_HI), 'Normal reversing\nmode', DSLATE),
                       (np.sqrt(HYPER_LO * HYPER_HI), 'Hyperactive\nmode', HYPER_TEXT)]:
        ax_c.text(x, 0.5 * (1.0 + ytop), text, ha='center', va='center', fontsize=5.6, color=c, zorder=6)
    for rate in RATES:
        ax_c.plot(lambda_grid, curves[rate], color=colour[rate], ls='solid', lw=1.95,
                  label=f'{rate:g} °C Myr$^{{-1}}$', path_effects=None,
                  zorder=4 if rate == REFERENCE_RATE else 3)
    ax_c.grid(axis='y', color=LIGHT_GRY, lw=0.55, alpha=0.85, zorder=0)
    probe = {rate: float(np.interp(LAMBDA_PROBE, lambda_grid, curves[rate])) for rate in RATES}
    y_lo, y_hi = min(probe.values()), max(probe.values())
    ax_c.plot([LAMBDA_PROBE, LAMBDA_PROBE], [0.0, y_hi], color=INK, lw=0.95, ls=(0, (2.6, 2.0)), zorder=6)
    for rate in RATES:
        ax_c.plot([LAMBDA_PROBE], [probe[rate]], marker='o', markersize=6.4, markerfacecolor='white',
                  markeredgecolor=colour[rate], markeredgewidth=1.7, ls='none', zorder=8)
    ax_c.annotate('', xy=(LAMBDA_PROBE * 1.30, y_hi), xytext=(LAMBDA_PROBE * 1.30, y_lo),
                  arrowprops=dict(arrowstyle='-|>', color=INK, lw=1.25, shrinkA=0.0, shrinkB=0.0,
                                  mutation_scale=11,
                                  path_effects=[pe.Stroke(linewidth=3.0, foreground='white'), pe.Normal()]),
                  zorder=9)
    box = dict(boxstyle='round,pad=0.16', facecolor='white', edgecolor='none', alpha=0.90)
    ax_c.text(LAMBDA_PROBE * 1.62, 0.5 * (y_lo + y_hi), 'faster cooling', rotation=90, ha='center',
              va='center', fontsize=6.6, color=INK, bbox=box, zorder=9)
    ax_c.text(LAMBDA_PROBE, 0.035, f'$\\lambda$ = {LAMBDA_PROBE:g} Myr$^{{-1}}$', ha='center', va='bottom',
              fontsize=6.6, color=INK, bbox=box, zorder=9)
    ax_c.set_xlabel('Reversal rate, $\\lambda$ (Myr$^{-1}$)', fontsize=8.0)
    ax_c.set_ylabel(r'Moment-weighted RMS coherence, $\bar{C}_{\mathrm{RMS}}$', fontsize=8.0)
    ax_c.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_c.tick_params(labelsize=7.8, length=3.0)
    ax_c.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
    ax_c.legend(handles=[Line2D([0], [0], color=colour[rate], ls='solid', lw=1.95,
                                label=f'{rate:g} °C Myr$^{{-1}}$' + ('  (reference)' if rate == REFERENCE_RATE else ''))
                         for rate in sorted(RATES)],
                title='Cooling rate', title_fontsize=6.2, loc='lower left', bbox_to_anchor=(0.018, 0.018),
                frameon=True, framealpha=0.88, facecolor='white', edgecolor=LIGHT_GRY, fontsize=5.7,
                handlelength=1.7, labelspacing=0.15, borderpad=0.4)

    # (b)
    rms_b = [summary_b[i]['rms_efficiency'] for i in range(3)]
    jitter = np.random.default_rng(20260722)
    ax_b.text(-0.165, 1.025, '(b)', transform=ax_b.transAxes, ha='left', va='top',
              fontsize=9.0, fontweight='bold', zorder=10, clip_on=False)
    for i, (fc, pc) in enumerate([(SLATE_FILL, SLATE), (GRN_FILL, GRN), (SLATE_FILL, SLATE)]):
        v = mag_b[:, i]
        bp = ax_b.boxplot([v], positions=[i + 1.0], widths=0.48, whis=(5, 95), showfliers=False,
                          patch_artist=True, medianprops=dict(color=INK, linewidth=1.35),
                          whiskerprops=dict(color=GRY_DARK, linewidth=0.95),
                          capprops=dict(color=GRY_DARK, linewidth=0.95),
                          boxprops=dict(edgecolor=GRY_DARK, linewidth=0.95), zorder=3)
        bp['boxes'][0].set_facecolor(fc)
        bp['boxes'][0].set_alpha(0.70)
        ax_b.scatter(np.full(v.size, i + 1.0) + jitter.uniform(-0.15, 0.15, v.size), v, s=10,
                     facecolor=pc, edgecolor='none', alpha=0.22, zorder=2)
    ax_b.set_xlim(0.45, 3.55); ax_b.set_ylim(0.0, 1.08)
    ax_b.set_yticks(np.linspace(0.0, 1.0, 6))
    ax_b.grid(axis='y', color=LIGHT_GRY, lw=0.55, alpha=0.78, zorder=0)
    ax_b.tick_params(axis='y', labelsize=7.5, length=3.0)
    ax_b.tick_params(axis='x', length=0)
    ax_b.set_xticks([1.0, 2.0, 3.0])
    ax_b.set_xticklabels(['Pre-MRS', 'MRS', 'Post-MRS'], fontsize=7.8)
    ax_b.set_ylabel('Coherent-moment efficiency', fontsize=8.0)
    for i, value in enumerate(rms_b):
        v = mag_b[:, i]
        y, va = (float(v.min()) - 0.04, 'top') if i == 1 else (float(v.max()) + 0.03, 'bottom')
        ax_b.text(i + 1.0, y, f'RMS {100*value:.1f}%', ha='center', va=va, fontsize=6.8, color=INK, zorder=6)

    Path(out + '_MRS_SHARE.json').write_text(json.dumps(mrs_share, indent=2), encoding='utf-8')
    with open(out + '_panel_b_summary.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_b[0]))
        w.writeheader()
        w.writerows(summary_b)
    fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
