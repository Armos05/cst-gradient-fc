import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, spearmanr, t as tdist
from statsmodels.stats.multitest import multipletests


# ----------------------------
# Subject ID normalization
# ----------------------------
def normalize_to_int_subject(x) -> int | None:
    """
    Convert many formats into a plain integer subject ID.
    Works for:
      - 8, 12, 2245
      - "8", "2245"
      - "sub-0008", "sub_2245", "STRICON_8", etc.
    Returns None if no digits found.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    m = re.search(r"(\d+)", s)
    if not m:
        return None
    return int(m.group(1))


def subdir_to_int_subject(subdir_name: str) -> int | None:
    """
    sub-0008 -> 8
    sub-2245 -> 2245
    """
    return normalize_to_int_subject(subdir_name)


# ----------------------------
# Excel loader
# ----------------------------
def detect_subject_column(df: pd.DataFrame) -> str:
    """
    Heuristic: find a column likely containing subject IDs.
    """
    candidates = [
        "subject", "subject_id", "participant_id", "participant", "sub", "sub_id",
        "id", "rid", "ID", "Subject", "SubjectID", "RID"
    ]
    cols = list(df.columns)

    # exact match first
    for c in candidates:
        if c in cols:
            return c

    # case-insensitive
    lower_map = {c.lower(): c for c in cols}
    for c in [x.lower() for x in candidates]:
        if c in lower_map:
            return lower_map[c]

    # fallback: first column
    return cols[0]


def load_groups_from_excel(excel_path: str, sheet: str, group_col: str) -> dict[int, str]:
    """
    Returns mapping: subject_int -> group_label
    """
    df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
    subj_col = detect_subject_column(df)

    # group col case-insensitive
    if group_col not in df.columns:
        lower = {c.lower(): c for c in df.columns}
        if group_col.lower() in lower:
            group_col = lower[group_col.lower()]
        else:
            raise ValueError(
                f"Sheet '{sheet}' missing group column '{group_col}'. Columns: {list(df.columns)}"
            )

    df = df[[subj_col, group_col]].copy()
    df["subject_int"] = df[subj_col].apply(normalize_to_int_subject)
    df = df.dropna(subset=["subject_int"])
    df["subject_int"] = df["subject_int"].astype(int)
    df["group"] = df[group_col].astype(str).str.strip()

    return dict(zip(df["subject_int"], df["group"]))


# ----------------------------
# FC loader
# ----------------------------
def load_fc_dataset(preproc_root: str) -> tuple[list[int], list[str], np.ndarray]:
    """
    Returns:
      subjects_int: list[int] subject IDs (e.g., 8, 2245) ONLY for subjects that have fc_matrices.npz
      roi_names: list[str]
      mats_r: array (Nsub, R, R) Pearson r matrices
    """
    root = Path(preproc_root)
    sub_dirs = sorted([p for p in root.glob("sub-*") if p.is_dir()])

    subjects_int = []
    mats = []
    roi_names = None

    for sd in sub_dirs:
        npz = sd / "fc_matrices.npz"
        if not npz.exists():
            continue

        sid = subdir_to_int_subject(sd.name)
        if sid is None:
            continue

        dat = np.load(npz, allow_pickle=True)
        r = dat["r"].astype(np.float64)
        rois = dat["roi_names"].tolist()

        if roi_names is None:
            roi_names = rois
        else:
            if list(roi_names) != list(rois):
                raise ValueError(f"ROI order mismatch in {npz}")

        subjects_int.append(sid)
        mats.append(r)

    if roi_names is None or len(mats) == 0:
        raise ValueError(f"No fc_matrices.npz found under {preproc_root}")

    mats = np.stack(mats, axis=0)  # (N, R, R)
    return subjects_int, roi_names, mats


def upper_triangle_edges(roi_names: list[str]) -> pd.DataFrame:
    R = len(roi_names)
    rows = []
    for i in range(R):
        for j in range(i + 1, R):
            rows.append({"i": i, "j": j, "roi_a": roi_names[i], "roi_b": roi_names[j]})
    return pd.DataFrame(rows)


def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def cohen_d(x, y) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    return (np.mean(x) - np.mean(y)) / sp


# ----------------------------
# STRICON: Patients vs Healthy
# ----------------------------
def stats_stricon(subjects_int, mats_r, roi_names, group_map_int):
    """
    group_map_int: dict[int -> str], e.g., 8 -> "Patients"
    Only subjects that exist in mats_r are used (requirement #2).
    """
    edges = upper_triangle_edges(roi_names)

    # align group labels to FC subjects
    groups = [group_map_int.get(sid, None) for sid in subjects_int]
    keep_idx = [k for k, g in enumerate(groups) if g is not None]

    subjects_int = [subjects_int[k] for k in keep_idx]
    groups = [groups[k] for k in keep_idx]
    mats_r = mats_r[keep_idx, :, :]

    mats_z = fisher_z(mats_r)

    idx_pat = [i for i, g in enumerate(groups) if str(g).strip().lower() in {"patients", "patient"}]
    idx_hc  = [i for i, g in enumerate(groups) if str(g).strip().lower() in {"healthy", "hc", "control", "controls"}]

    if len(idx_pat) < 2 or len(idx_hc) < 2:
        present = sorted(set([str(g) for g in groups]))
        raise ValueError(
            f"STRICON needs >=2 per group. Got Patients={len(idx_pat)}, Healthy={len(idx_hc)}. "
            f"Groups present after matching: {present}"
        )

    out_rows = []
    for _, e in edges.iterrows():
        i, j = int(e["i"]), int(e["j"])

        r_pat = mats_r[idx_pat, i, j].astype(float)
        r_hc  = mats_r[idx_hc,  i, j].astype(float)

        z_pat = mats_z[idx_pat, i, j].astype(float)
        z_hc  = mats_z[idx_hc,  i, j].astype(float)

        # Drop non-finite per edge (critical)
        z_pat_f = z_pat[np.isfinite(z_pat)]
        z_hc_f  = z_hc[np.isfinite(z_hc)]
        r_pat_f = r_pat[np.isfinite(r_pat)]
        r_hc_f  = r_hc[np.isfinite(r_hc)]

        # If too few remain, skip stats for this edge
        if len(z_pat_f) < 2 or len(z_hc_f) < 2:
            t, p = np.nan, np.nan
            d = np.nan
        else:
            t, p = ttest_ind(z_pat_f, z_hc_f, equal_var=False, nan_policy="omit")
            d = cohen_d(z_pat_f, z_hc_f)

        mean_pat = float(np.mean(r_pat_f)) if r_pat_f.size else np.nan
        mean_hc  = float(np.mean(r_hc_f))  if r_hc_f.size  else np.nan
        direction = (
            "Patients>Healthy" if (np.isfinite(mean_pat) and np.isfinite(mean_hc) and mean_pat > mean_hc)
            else "Patients<Healthy"
        )

        out_rows.append({
            "roi_a": e["roi_a"],
            "roi_b": e["roi_b"],
            "mean_r_patients": mean_pat,
            "mean_r_healthy": mean_hc,
            "diff_pat_minus_hc": (mean_pat - mean_hc) if (np.isfinite(mean_pat) and np.isfinite(mean_hc)) else np.nan,
            "direction": direction,
            "t_z": float(t) if np.isfinite(t) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
            "q_fdr": np.nan,  # filled after
            "cohen_d_z": float(d) if np.isfinite(d) else np.nan,
            "n_patients": len(idx_pat),
            "n_healthy": len(idx_hc),
            "n_used_pat_edge": int(len(z_pat_f)),
            "n_used_hc_edge": int(len(z_hc_f)),
        })

    res = pd.DataFrame(out_rows)

    # FDR (treat NaN p-values as 1.0 so they don't blow up multipletests)
    p_for_fdr = res["p"].fillna(1.0).values
    res["q_fdr"] = multipletests(p_for_fdr, method="fdr_bh")[1]

    res = res.sort_values(["q_fdr", "p"]).reset_index(drop=True)
    return res


# ----------------------------
# VELAS: monotonic trend ls/hs/patient
# ----------------------------
def stats_velas(subjects_int, mats_r, roi_names, group_map_int):
    """
    Primary test: linear slope on Fisher-z vs ordinal severity (ls=0, hs=1, patient=2)
    Also reports Spearman rho on raw r.
    Applies FDR on slope p-values.
    """
    edges = upper_triangle_edges(roi_names)

    # align group labels to FC subjects (only subjects with .npz are considered)
    groups = [group_map_int.get(sid, None) for sid in subjects_int]
    keep_idx = [k for k, g in enumerate(groups) if g is not None]

    subjects_int = [subjects_int[k] for k in keep_idx]
    groups = [groups[k] for k in keep_idx]
    mats_r = mats_r[keep_idx, :, :]

    enc_map = {"ls": 0, "hs": 1, "patient": 2, "patients": 2}
    x_all = []
    keep2 = []
    for k, g in enumerate(groups):
        gg = str(g).strip().lower()
        if gg in enc_map:
            x_all.append(enc_map[gg])
            keep2.append(k)

    x_all = np.asarray(x_all, dtype=float)
    mats_r = mats_r[keep2, :, :]
    groups = [groups[k] for k in keep2]

    if len(np.unique(x_all)) < 3:
        present = sorted(set([str(g) for g in groups]))
        raise ValueError(f"VELAS needs ls/hs/patient all present. Found (after matching): {present}")

    mats_z = fisher_z(mats_r)

    out_rows = []
    for _, e in edges.iterrows():
        i, j = int(e["i"]), int(e["j"])

        r = mats_r[:, i, j].astype(float)
        z = mats_z[:, i, j].astype(float)
        x = x_all.astype(float)

        # Drop non-finite per edge
        m = np.isfinite(x) & np.isfinite(r) & np.isfinite(z)
        x_e = x[m]
        r_e = r[m]
        z_e = z[m]

        if len(z_e) < 3 or len(np.unique(x_e)) < 2:
            rho, p_rho = np.nan, np.nan
            slope, t_slope, p_slope = np.nan, np.nan, np.nan
        else:
            rho, p_rho = spearmanr(x_e, r_e, nan_policy="omit")

            # closed-form OLS slope (with intercept): z ~ 1 + x
            xbar = np.mean(x_e)
            zbar = np.mean(z_e)
            Sxx = np.sum((x_e - xbar) ** 2)

            if Sxx <= 0:
                slope, t_slope, p_slope = np.nan, np.nan, np.nan
            else:
                slope = np.sum((x_e - xbar) * (z_e - zbar)) / Sxx
                intercept = zbar - slope * xbar

                resid = z_e - (intercept + slope * x_e)
                n = len(z_e)
                dof = n - 2
                if dof <= 0:
                    t_slope, p_slope = np.nan, np.nan
                else:
                    s2 = np.sum(resid ** 2) / dof
                    se_slope = np.sqrt(s2 / Sxx) if s2 >= 0 else np.nan
                    t_slope = slope / se_slope if (np.isfinite(se_slope) and se_slope > 0) else np.nan
                    p_slope = 2 * (1 - tdist.cdf(np.abs(t_slope), df=dof)) if np.isfinite(t_slope) else np.nan

        def mean_or_nan(arr):
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if arr.size > 0 else np.nan

        r_ls = r[x_all == 0]
        r_hs = r[x_all == 1]
        r_pat = r[x_all == 2]
        mean_ls, mean_hs, mean_pat = mean_or_nan(r_ls), mean_or_nan(r_hs), mean_or_nan(r_pat)

        inc = (np.isfinite(mean_ls) and np.isfinite(mean_hs) and np.isfinite(mean_pat) and (mean_ls < mean_hs < mean_pat))
        dec = (np.isfinite(mean_ls) and np.isfinite(mean_hs) and np.isfinite(mean_pat) and (mean_ls > mean_hs > mean_pat))
        direction = "ls<hs<patient" if inc else ("ls>hs>patient" if dec else "non-monotonic")

        out_rows.append({
            "roi_a": e["roi_a"],
            "roi_b": e["roi_b"],
            "mean_r_ls": mean_ls,
            "mean_r_hs": mean_hs,
            "mean_r_patient": mean_pat,
            "direction": direction,
            "spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "p_spearman": float(p_rho) if np.isfinite(p_rho) else np.nan,
            "slope_z": float(slope) if np.isfinite(slope) else np.nan,
            "t_slope_z": float(t_slope) if np.isfinite(t_slope) else np.nan,
            "p_slope_z": float(p_slope) if np.isfinite(p_slope) else np.nan,
            "q_fdr_slope": np.nan,    # filled after
            "q_fdr_spearman": np.nan, # filled after
            "n_ls": int(np.sum(x_all == 0)),
            "n_hs": int(np.sum(x_all == 1)),
            "n_patient": int(np.sum(x_all == 2)),
            "n_used_edge": int(len(z_e)),
        })

    res = pd.DataFrame(out_rows)

    # FDR: treat NaN p-values as 1.0
    res["q_fdr_slope"] = multipletests(res["p_slope_z"].fillna(1.0).values, method="fdr_bh")[1]
    res["q_fdr_spearman"] = multipletests(res["p_spearman"].fillna(1.0).values, method="fdr_bh")[1]

    res = res.sort_values(["q_fdr_slope", "p_slope_z"]).reset_index(drop=True)
    return res

