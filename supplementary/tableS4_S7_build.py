# Tables S4 and S7 from the per-spectrum RSCM fits (fig4_rscm.py) and the per-specimen
# outputs of fig5_pullaiah_reset.py and fig6_blocking_ages.py.
import pandas as pd
import common

D = common.DATA
r = pd.read_csv(D / 'nama_rscm_per_measurement.csv')
r['spectrum'] = r.groupby('sample').cumcount() + 1
r = r.sort_values(['sample'], key=lambda s: s.str[1:].astype(int), kind='stable')
s4 = pd.DataFrame({'Sample': r['sample'], 'Spectrum': r.spectrum,
    'D1_position': r.D1_pos.round(1), 'D1_FWHM': r.D1_fwhm.round(1), 'D3_position': r.D3_pos.round(1),
    'D3_FWHM': r.D3_fwhm.round(1), 'G_position': r.G_pos.round(1), 'G_FWHM': r.G_fwhm.round(1),
    'D2_position': r.D2_pos.round(1), 'D2_FWHM': r.D2_fwhm.round(1), 'R1': r.R1.round(2), 'R2': r.R2.round(3),
    'T_C': r.T_rahl.round(0).astype(int), 'Fit_R2': r.r2fit.round(3)})
s4.to_csv(common.OUT / 'TableS4_raman.csv', index=False)


def short(s):
    for pre, L in (('1G_TT_', 'G'), ('1B_TT_set1_', 'B'), ('1B_TT_set2_', 'B'), ('1A_T_1_', 'A'), ('1A_T_2_', 'A'), ('Dt_', 'D')):
        if s.startswith(pre):
            return f"{L}{int(s.split('_')[-1]):02d}"
    return s


a = pd.read_csv(D / 'Fig5_fractions_per_specimen.csv')
b = pd.read_csv(D / 'Fig6_per_specimen.csv')
m = a.merge(b, on='spec').sort_values('med', ascending=False)
s7 = pd.DataFrame({'Label': m.spec.map(short), 'Laboratory_code': m.spec,
    'Reset_pct': (100 * m.reset_x).round(1), 'Unreset_Po_pct': (100 * m.pyr_not_reset).round(1),
    'Age_10pct_blocked_Ma': m.p90.round(1), 'Age_25pct_blocked_Ma': m.q3.round(1), 'Median_Ma': m.med.round(1),
    'Age_75pct_blocked_Ma': m.q1.round(1), 'Age_90pct_blocked_Ma': m.p10.round(1), 'Reset_moment_in_MRS_pct': m.inm.round(1)})
s7.to_csv(common.OUT / 'TableS7_specimens.csv', index=False)
