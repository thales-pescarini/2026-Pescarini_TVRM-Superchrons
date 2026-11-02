# TROUT two-component fits (Cych et al. 2023) of the thermal demagnetization data.
# The fitting window is the full series, cut only where laboratory acquisition starts
# (three successive gains in m/m0 > 0.002 above 320 C). Seeds are fixed by manifest position.
# Input: MagIC tables of Pescarini et al. (2025) in the sub-directories of DATASET_DIRS.
# Output: trout_status.csv, C2_SPECTRA/, PARAMS/, GATE_DIAGNOSTICS/ (resumable).
# Usage: python trout/trout_batch.py <data root> [output directory]
from pathlib import Path
import json, time, sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import trout as tr
from pmagpy import pmag

if len(sys.argv) < 2:
    sys.exit("Usage: python trout/trout_batch.py <data root> [output directory]")
DATA_ROOT = Path(sys.argv[1]).expanduser().resolve()
DATASET_DIRS = {
    "DT": DATA_ROOT / "_TROUT_MAGIC_STAGING" / "DT",
    "B1": DATA_ROOT / "1B" / "1B_TT_set1_",
    "B2": DATA_ROOT / "1B" / "1B_TT_set2_",
    "G":  DATA_ROOT / "1G",
    "A1": DATA_ROOT / "1A" / "1A_T_1_",
    "A2": DATA_ROOT / "1A" / "1A_T_2_",
}
OUT_ROOT = Path(sys.argv[2] if len(sys.argv) > 2 else "trout_run_new").expanduser().resolve()

N_COMPONENTS = 2
ANCHORED     = False
POP_SIZE     = 2
POLISH       = True
C2_INDEX     = 1
BASE_SEED    = 20260728

HIGH_T_THRESHOLD     = 320.0    # acquisition rule
MIN_SUCCESSIVE_GAINS = 3
GAIN_TOL             = 0.002
M_FLOOR_THRESHOLD    = 0.03     # diagnostic only; does not truncate the fit
MIN_STEPS_AFTER_GATE = 6
SPECTRUM_POINTS      = 10000

# Manifest order fixes the random seed of each specimen.
MANIFEST = [
    (1, "Dt_14", "DT"),
    (2, "Dt_28", "DT"),
    (3, "Dt_29", "DT"),
    (4, "Dt_32", "DT"),
    (5, "Dt_37", "DT"),
    (6, "Dt_38", "DT"),
    (7, "Dt_39", "DT"),
    (8, "Dt_41", "DT"),
    (9, "Dt_42", "DT"),
    (10, "Dt_44", "DT"),
    (11, "Dt_47", "DT"),
    (12, "Dt_55", "DT"),
    (13, "Dt_57", "DT"),
    (14, "Dt_58", "DT"),
    (15, "Dt_60", "DT"),
    (16, "1B_TT_set1_2", "B1"),
    (17, "1B_TT_set1_6", "B1"),
    (18, "1B_TT_set1_8", "B1"),
    (19, "1B_TT_set1_9", "B1"),
    (20, "1B_TT_set1_25", "B1"),
    (21, "1B_TT_set1_47", "B1"),
    (22, "1B_TT_set1_53", "B1"),
    (23, "1B_TT_set1_55", "B1"),
    (24, "1B_TT_set2_66", "B2"),
    (25, "1B_TT_set2_81", "B2"),
    (26, "1B_TT_set2_82", "B2"),
    (27, "1B_TT_set2_83", "B2"),
    (28, "1B_TT_set2_91", "B2"),
    (29, "1B_TT_set2_95", "B2"),
    (30, "1B_TT_set2_98", "B2"),
    (31, "1B_TT_set2_105", "B2"),
    (32, "1B_TT_set2_110", "B2"),
    (33, "1B_TT_set2_111", "B2"),
    (34, "1B_TT_set2_116", "B2"),
    (35, "1B_TT_set2_117", "B2"),
    (36, "1B_TT_set2_119", "B2"),
    (37, "1B_TT_set2_120", "B2"),
    (38, "1B_TT_set2_121", "B2"),
    (39, "1G_TT_3", "G"),
    (40, "1G_TT_6", "G"),
    (41, "1G_TT_8", "G"),
    (42, "1G_TT_9", "G"),
    (43, "1G_TT_10", "G"),
    (44, "1G_TT_12", "G"),
    (45, "1G_TT_13", "G"),
    (46, "1G_TT_15", "G"),
    (47, "1G_TT_17", "G"),
    (48, "1G_TT_18", "G"),
    (49, "1G_TT_19", "G"),
    (50, "1G_TT_20", "G"),
    (51, "1G_TT_21", "G"),
    (52, "1G_TT_22", "G"),
    (53, "1G_TT_29", "G"),
    (54, "1G_TT_31", "G"),
    (55, "1G_TT_32", "G"),
    (56, "1G_TT_37", "G"),
    (57, "1G_TT_38", "G"),
    (58, "1G_TT_39", "G"),
    (59, "1G_TT_46", "G"),
    (60, "1G_TT_47", "G"),
    (61, "1G_TT_49", "G"),
    (62, "1A_T_1_4", "A1"),
    (63, "1A_T_1_7", "A1"),
    (64, "1A_T_1_25", "A1"),
    (65, "1A_T_2_130", "A2"),
    (66, "1A_T_2_131", "A2"),
    (67, "1A_T_2_133", "A2"),
    (68, "1A_T_2_134", "A2"),
    (69, "1A_T_2_135", "A2"),
]

def trout_directory_string(directory):
    import os
    return str(Path(directory)) + os.sep

def load_magic_dataset(name, directory):
    wd = trout_directory_string(directory)
    meas, samp, site = tr.import_direction_data(wd)
    meas = meas.copy()
    meas["specimen"] = meas["specimen"].fillna("").astype(str).str.strip()
    return {"meas_data": meas, "sample_data": samp, "site_data": site}

def prepare_truncated_trout_input(zijd_data, Ts_base):
    # renormalize, rebuild Ms and re-estimate the temperature-axis exponent
    zijd_data = np.asarray(zijd_data, dtype=float)
    Ts_base = np.asarray(Ts_base, dtype=float).reshape(-1)
    vector_norms = np.linalg.norm(zijd_data, axis=1)
    normalization = float(np.nanmax(vector_norms))
    zijd_clean = zijd_data / normalization
    flipped = np.flip(zijd_clean, axis=0)
    diff = np.diff(flipped, axis=0)
    appended = np.vstack([np.zeros(3), diff])
    residual = float(np.linalg.norm(zijd_clean[-1]))
    norm_data = np.linalg.norm(appended, axis=1) + residual
    Ms = np.flip(np.cumsum(norm_data))
    Ms_scaled = Ms / float(np.nanmax(Ms))
    m = minimize(tr.bestfit_tanh, x0=[1.0, 0.1], args=(Ts_base, Ms_scaled),
                 method="L-BFGS-B", bounds=[(1.0, 100.0), (-10.0, 10.0)],
                 options={"eps": np.sqrt(np.finfo("float32").eps)})
    expon = float(m.x[1])
    Ts_data = np.asarray(tr.scale_xs(Ts_base, expon), dtype=float)
    return {"zijd_data": zijd_clean, "Ts_base": Ts_base, "Ts_data": Ts_data,
            "expon": expon, "scale_fit_success": bool(m.success),
            "scale_fit_message": str(m.message), "scale_fit_fun": float(m.fun)}

def calculate_trout_spectra(par_dict, Ts_base, expon):
    # component spectra on a 10000-point axis, with the Jacobian of the axis transform
    cs = np.asarray(par_dict["cs"], dtype=float)
    mus = np.asarray(par_dict["mus"], dtype=float)
    sds = np.asarray(par_dict["sds"], dtype=float)
    p_stars = np.asarray(par_dict["p_stars"], dtype=float)
    q_stars = np.asarray(par_dict["q_stars"], dtype=float)
    Ts_base = np.asarray(Ts_base, dtype=float)
    axis_full = np.linspace(float(np.min(Ts_base)), float(np.max(Ts_base)), SPECTRUM_POINTS)
    scaled_axis_full = np.asarray(tr.scale_xs(axis_full, expon), dtype=float)
    transformation_exponent = np.exp(-abs(expon))
    pdfs = []
    for k in range(len(cs)):
        p_value = np.exp(p_stars[k])
        q_star = float(q_stars[k])
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_q = np.divide(1.0, q_star)
        q_value = 2.0 / np.pi * np.arctan(inv_q)
        pdf = cs[k] * tr.SGG_pdf(scaled_axis_full, mus[k], sds[k], p_value, q_value)
        pdf = np.asarray(pdf, dtype=float)
        h = np.heaviside(expon, 0)
        transform = h * np.max(axis_full) - (h * 2.0 - 1.0) * axis_full
        with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
            pdf *= transformation_exponent * transform ** (transformation_exponent - 1.0)
        pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        pdfs.append(pdf)
    pdfs = np.asarray(pdfs, dtype=float)
    return axis_full[:-1], pdfs[:, :-1]

def integrate_curve(y, x):
    f = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(f(np.asarray(y, float), np.asarray(x, float)))

def fit_window(m_over_m0, Ts):
    m = np.asarray(m_over_m0, float); T = np.asarray(Ts, float)
    gains = (np.diff(m) > GAIN_TOL) & (T[1:] > HIGH_T_THRESHOLD)
    keep = len(T); reason = "full window"
    run = 0
    for i, g in enumerate(gains):
        run = run + 1 if g else 0
        if run >= MIN_SUCCESSIVE_GAINS:
            keep = max((i + 1) - MIN_SUCCESSIVE_GAINS + 1, MIN_STEPS_AFTER_GATE)
            reason = f"acquisition from {T[keep]:.0f} C"
            break
    return keep, reason

def main():
    for sub in ("C2_SPECTRA", "GATE_DIAGNOSTICS", "PARAMS", "LOGS"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)
    status_path = OUT_ROOT / "trout_status.csv"
    rows = []
    if status_path.is_file():
        rows = pd.read_csv(status_path).to_dict("records")
    done = {r["specimen"] for r in rows}

    cache = {}
    t0 = time.perf_counter()
    n_total = len(MANIFEST)
    for position, specimen, dataset_name in MANIFEST:
        if specimen in done and (OUT_ROOT / "C2_SPECTRA" / f"{specimen}__C2_spectrum.csv").is_file():
            print(f"[{position:02d}/{n_total}] {specimen}: already done, skipped")
            continue
        seed = BASE_SEED + position - 1
        if dataset_name not in cache:
            cache[dataset_name] = load_magic_dataset(dataset_name, DATASET_DIRS[dataset_name])
        ds = cache[dataset_name]
        t_item = time.perf_counter()
        try:
            np.random.seed(seed)
            zf, Tf, _td, _ts, datatype, _e = tr.prepare_specimen_data(
                specimen, ds["meas_data"], ds["sample_data"], ds["site_data"])
            zf = np.asarray(zf, float); Tf = np.asarray(Tf, float).reshape(-1)
            norms = np.linalg.norm(zf, axis=1)
            m_over_m0 = norms / norms[0]

            keep, reason = fit_window(m_over_m0, Tf)
            zu, Tu = zf[:keep], Tf[:keep]
            m_end = float(m_over_m0[keep - 1])
            floor_idx = np.where(m_over_m0[:keep] <= M_FLOOR_THRESHOLD)[0]
            floor_T = float(Tf[floor_idx[0]]) if len(floor_idx) else np.nan

            diag = pd.DataFrame({
                "step_index": np.arange(len(Tf)),
                "treatment_C": Tf,
                "m_over_m0": m_over_m0,
                "delta_m_over_m0": np.concatenate([[np.nan], np.diff(m_over_m0)]),
                "above_high_T_threshold": Tf > HIGH_T_THRESHOLD,
                "at_or_below_floor": m_over_m0 <= M_FLOOR_THRESHOLD,
                "kept_for_TROUT": np.arange(len(Tf)) < keep,
            })
            diag.to_csv(OUT_ROOT / "GATE_DIAGNOSTICS" / f"{specimen}__gate_diagnostics.csv",
                        index=False, encoding="utf-8")

            tin = prepare_truncated_trout_input(zu, Tu)
            par_dict, alts = tr.find_best_fit_model(
                tin["zijd_data"], tin["Ts_data"],
                n_components=N_COMPONENTS, anchored=ANCHORED,
                pop_size=POP_SIZE, polish=POLISH)

            axis, pdfs = calculate_trout_spectra(par_dict, tin["Ts_base"], tin["expon"])
            c1_raw = np.nan_to_num(pdfs[0], nan=0.0, posinf=0.0, neginf=0.0)
            c2_raw = np.nan_to_num(pdfs[C2_INDEX], nan=0.0, posinf=0.0, neginf=0.0)
            total_raw = np.nansum(pdfs, axis=0)
            c2_fraction = np.divide(c2_raw, total_raw, out=np.zeros_like(c2_raw),
                                    where=total_raw > 0)
            c2_max = float(np.nanmax(c2_raw))
            if c2_max <= 0:
                raise ValueError("C2 spectrum with zero maximum.")
            c2_area = integrate_curve(c2_raw, axis)
            c2_norm_area = c2_raw / c2_area if (np.isfinite(c2_area) and c2_area > 0) else np.zeros_like(c2_raw)
            pd.DataFrame({
                "temperature_C": axis,
                "C2_spectrum_raw": c2_raw,
                "C2_normalized_to_max": c2_raw / c2_max,
                "C2_normalized_by_area": c2_norm_area,
                "C2_fraction_of_total": c2_fraction,
            }).to_csv(OUT_ROOT / "C2_SPECTRA" / f"{specimen}__C2_spectrum.csv",
                      index=False, encoding="utf-8")

            Bs = np.asarray(par_dict["Bs"], float)
            dirs = np.asarray(pmag.cart2dir(Bs), float)
            cs = np.asarray(par_dict["cs"], float)
            mus = np.asarray(par_dict["mus"], float)
            sds = np.asarray(par_dict["sds"], float)
            ps = np.asarray(par_dict["p_stars"], float)
            qs = np.asarray(par_dict["q_stars"], float)
            c1_peak = float(axis[int(np.nanargmax(c1_raw))]) if np.any(c1_raw > 0) else np.nan
            c2_peak = float(axis[int(np.nanargmax(c2_raw))]) if np.any(c2_raw > 0) else np.nan

            json_payload = {
                "specimen": specimen, "dataset": dataset_name, "seed": seed,
                "protocol": "full_window_acquisition_cut",
                "keep": int(keep), "n_steps": int(len(Tf)), "window_reason": reason,
                "m_end": m_end, "floor_first_T": floor_T,
                "expon": float(tin["expon"]), "logp": float(par_dict["logp"]),
                "cs": cs.tolist(), "mus": mus.tolist(), "sds": sds.tolist(),
                "p_stars": ps.tolist(), "q_stars": qs.tolist(),
                "Bs": Bs.tolist(), "dirs_dec_inc": dirs.tolist(),
                "sigma": float(np.asarray(par_dict.get("sigma", np.nan))),
                "psi": float(np.asarray(par_dict.get("psi", np.nan))),
                "c1_peak_C": c1_peak, "c2_peak_C": c2_peak,
            }
            with open(OUT_ROOT / "PARAMS" / f"{specimen}__params.json", "w") as fh:
                json.dump(json_payload, fh, indent=1)

            rows.append({
                "position": position, "specimen": specimen, "dataset": dataset_name,
                "seed": seed, "status": "ACCEPTED", "gate_reason": reason,
                "original_steps": int(len(Tf)), "used_steps": int(keep),
                "used_max_temperature": float(Tu[-1]), "m_end": m_end,
                "floor_first_T": floor_T,
                "expon_clean_segment": float(tin["expon"]),
                "logp": float(par_dict["logp"]),
                "C1_peak_temperature": c1_peak, "C2_peak_temperature": c2_peak,
                "C2_c": float(cs[C2_INDEX]), "C2_mu": float(mus[C2_INDEX]),
                "C2_sd": float(sds[C2_INDEX]), "C2_p_star": float(ps[C2_INDEX]),
                "C2_q_star": float(qs[C2_INDEX]),
                "C2_dec_deg": float(dirs[C2_INDEX, 0]), "C2_inc_deg": float(dirs[C2_INDEX, 1]),
                "duration_s": round(time.perf_counter() - t_item, 1),
            })
            pd.DataFrame(rows).to_csv(status_path, index=False, encoding="utf-8")
            el = time.perf_counter() - t0
            n_done = len(rows)
            eta = el / max(n_done, 1) * (n_total - position)
            print(f"[{position:02d}/{n_total}] {specimen}: {reason} | k {keep}/{len(Tf)} | "
                  f"expon {tin['expon']:+.4f} | logp {par_dict['logp']:.2f} | "
                  f"C1 {c1_peak:.1f} C2 {c2_peak:.1f} | m_end {m_end:.3f} | "
                  f"{time.perf_counter()-t_item:.0f}s (ETA ~{eta/60:.0f} min)")
        except Exception as exc:
            with open(OUT_ROOT / "LOGS" / f"{specimen}__ERROR.txt", "w") as fh:
                fh.write(repr(exc))
            rows.append({"position": position, "specimen": specimen,
                         "dataset": dataset_name, "seed": seed,
                         "status": f"FAILED: {exc}"})
            pd.DataFrame(rows).to_csv(status_path, index=False, encoding="utf-8")
            print(f"[{position:02d}/{n_total}] {specimen}: FAILED -> {exc}")

    print("Batch finished:", OUT_ROOT)

if __name__ == "__main__":
    main()
