#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, spearmanr
from statsmodels.stats.multitest import multipletests


# ----------------------------
# Utilities
# ----------------------------
def normalize_subid(x) -> str | None:
    """
    Converts many possible subject id formats into 'sub-XXXX' if possible.
    Examples:
      75 -> sub-0075
      '0075' -> sub-0075
      'sub-0075' -> sub-0075
      'sub_75' -> sub-0075
    Returns None if it can't parse.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()

    # Already sub-xxxx?
    m = re.search(r"sub[-_]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        num = int(m.group(1))
        return f"sub-{num:04d}"

    # Pure digits?
    m = re.fullmatch(r"\d+", s)
    if m:
        num = int(s)
        return f"sub-{num:04d}"

    return None


def detect_subject_column(df: pd.DataFrame) -> str:
    """
    Heuristic: find a column likely containing subject IDs.
    """
    candidates = [
        "subject", "subject_id", "participant_id", "participant", "sub", "sub_id",
        "id", "ID", "Subject", "SubjectID", "RID"
    ]
    cols = list(df.columns)
    # exact match first
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive match
    lower_map = {c.lower(): c for c in cols}
    for c in [x.lower() for x in candidates]:
        if c in lower_map:
            return lower_map[c]
    # fallback: first column
    return cols[0]


def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def cohen_d(x, y) -> float:
    # Cohen's d (pooled SD). For unequal n it's still common to use pooled SD.
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    sx, sy = np.var(x, ddof=1), np.var(y, ddof=1)
    sp = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if sp == 0 or np.isnan(sp):
        return np.nan
    return (np.mean(x) - np.mean(y)) / sp


# ----------------------------
# Load group labels from Excel
# ----------------------------
def load_groups_from_excel(excel_path: str, sheet: str, group_col: str) -> dict[str, str]:
    df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
    subj_col = detect_subject_column(df)

    if group_col not in df.columns:
        # try case-insensitive
        lower = {c.lower(): c for c in df.columns}
        if group_col.lower() in lower:
            group_col = lower[group_col.lower()]
        else:
            raise ValueError(f"Sheet '{sheet}' missing group column '{group_col}'. Columns: {list(df.columns)}")

    df = df[[subj_col, group_col]].copy()
    df["subject_id"] = df[subj_col].apply(normalize_subid)
    df = df.dropna(subset=["subject_id"])
    df["group"] = df[group_col].astype(str).str.strip()

    # map subject -> group
    return dict(zip(df["subject_id"], df["group"]))


# ----------------------------
# Load FC matrices
# ----------------------------
def load_fc_dataset(preproc_root: str) -> tuple[list[str], list[str], np.ndarray]:
    """
    Returns:
      subjects: list[str] subject_id
      roi_names: list[str]
      mats: array shape (Nsub, R, R) of Pearson r
    """
    root = Path(preproc_root)
    sub_dirs = sorted([p for p in root.glob("sub-*") if p.is_dir()])

    subjects = []
    mats = []
    roi_names = None

    for sd in sub_dirs:
        npz = sd / "fc_matrices.npz"
        if not npz.exists():
            continue

        dat = np.load(npz, allow_pickle=True)
        r = dat["r"].astype(np.float64)
        rois = dat["roi_names"].tolist()

        if roi_names is None:
            roi_names = rois
        else:
            # ensure same ordering
            if list(roi_names) != list(rois):
                raise ValueError(f"ROI order mismatch in {npz}")

        subjects.append(sd.name)  # sd.name is 'sub-XXXX'
        mats.append(r)

    if roi_names is None or len(mats) == 0:
        raise ValueError(f"No fc_matrices.npz found under {preproc_root}")

    mats = np.stack(mats, axis=0)  # (N, R, R)
    return subjects, roi_names, mats


def upper_triangle_edges(roi_names: list[str]) -> pd.DataFrame:
    R = len(roi_names)
    rows = []
    for i in range(R):
        for j in range(i + 1, R):
            rows.append({"i": i, "j": j, "roi_a": roi_names[i], "roi_b": roi_names[j]})
    return pd.DataFrame(rows)


# ----------------------------
# STRICON: Patients vs Healthy
# ----------------------------
def stats_stricon(subjects, mats_r, roi_names, group_map):
    edges = upper_triangle_edges(roi_names)

    # Make group vector aligned with subjects
    groups = [group_map.get(s, None) for s in subjects]
    keep_idx = [k for k, g in enumerate(groups) if g is not None]
    subjects = [subjects[k] for k in keep_idx]
    groups = [groups[k] for k in keep_idx]
    mats_r = mats_r[keep_idx, :, :]

    # Fisher-z for testing
    mats_z = fisher_z(mats_r)

    # indices
    idx_pat = [i for i, g in enumerate(groups) if g.lower() == "patients" or g.lower() == "patient"]
    idx_hc = [i for i, g in enumerate(groups) if g.lower() == "healthy" or g.lower() == "control" or g.lower() == "hc"]

    if len(idx_pat) < 2 or len(idx_hc) < 2:
        raise ValueError(f"STRICON needs >=2 per group. Got Patients={len(idx_pat)}, Healthy={len(idx_hc)}")

    out_rows = []
    for _, e in edges.iterrows():
        i, j = int(e["i"]), int(e["j"])

        r_pat = mats_r[idx_pat, i, j]
        r_hc = mats_r[idx_hc, i, j]

        z_pat = mats_z[idx_pat, i, j]
        z_hc = mats_z[idx_hc, i, j]

        # Welch t-test on z
        t, p = ttest_ind(z_pat, z_hc, equal_var=False, nan_policy="omit")

        mean_pat = float(np.mean(r_pat))
        mean_hc = float(np.mean(r_hc))
        direction = "Patients>Healthy" if mean_pat > mean_hc else "Patients<Healthy"

        out_rows.append({
            "roi_a": e["roi_a"],
            "roi_b": e["roi_b"],
            "mean_r_patients": mean_pat,
            "mean_r_healthy": mean_hc,
            "diff_pat_minus_hc": mean_pat - mean_hc,
            "direction": direction,
            "t_z": float(t),
            "p": float(p),
            "cohen_d_z": float(cohen_d(z_pat, z_hc)),
            "n_patients": len(idx_pat),
            "n_healthy": len(idx_hc),
        })

    res = pd.DataFrame(out_rows)
    res["q_fdr"] = multipletests(res["p"].values, method="fdr_bh")[1]
    res = res.sort_values(["q_fdr", "p"]).reset_index(drop=True)
    return res


# ----------------------------
# VELAS: monotonic trend ls/hs/patient
# ----------------------------
def stats_velas(subjects, mats_r, roi_names, group_map):
    edges = upper_triangle_edges(roi_names)

    groups = [group_map.get(s, None) for s in subjects]
    keep_idx = [k for k, g in enumerate(groups) if g is not None]
    subjects = [subjects[k] for k in keep_idx]
    groups = [groups[k] for k in keep_idx]
    mats_r = mats_r[keep_idx, :, :]

    # ordinal encoding
    enc_map = {"ls": 0, "hs": 1, "patient": 2, "patients": 2}
    x = []
    keep2 = []
    for k, g in enumerate(groups):
        gg = str(g).strip().lower()
        if gg in enc_map:
            x.append(enc_map[gg])
            keep2.append(k)
    x = np.asarray(x, dtype=float)
    mats_r = mats_r[keep2, :, :]
    groups = [groups[k] for k in keep2]

    if len(np.unique(x)) < 3:
        raise ValueError(f"VELAS needs ls/hs/patient all present. Found: {sorted(set([str(g) for g in groups]))}")

    mats_z = fisher_z(mats_r)

    out_rows = []
    for _, e in edges.iterrows():
        i, j = int(e["i"]), int(e["j"])
        r = mats_r[:, i, j]
        z = mats_z[:, i, j]

        # Spearman trend on raw r (robust) + also store slope on z via simple regression
        rho, p_rho = spearmanr(x, r, nan_policy="omit")

        # OLS slope on z (x in {0,1,2})
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, z, rcond=None)
        z_hat = X @ beta
        resid = z - z_hat
        dof = len(z) - 2
        s2 = np.sum(resid**2) / dof
        XtX_inv = np.linalg.inv(X.T @ X)
        se_slope = np.sqrt(s2 * XtX_inv[1, 1])
        t_slope = beta[1] / se_slope if se_slope > 0 else np.nan

        # two-sided p-value from t using normal approx if you want no extra deps:
        # better: use scipy.stats.t, but keep minimal:
        from scipy.stats import t as tdist
        p_slope = 2 * (1 - tdist.cdf(np.abs(t_slope), df=dof)) if np.isfinite(t_slope) else np.nan

        # group means in raw r
        r_ls = r[np.array(x) == 0]
        r_hs = r[np.array(x) == 1]
        r_pat = r[np.array(x) == 2]
        mean_ls, mean_hs, mean_pat = float(np.mean(r_ls)), float(np.mean(r_hs)), float(np.mean(r_pat))

        inc = (mean_ls < mean_hs < mean_pat)
        dec = (mean_ls > mean_hs > mean_pat)
        if inc:
            direction = "ls<hs<patient"
        elif dec:
            direction = "ls>hs>patient"
        else:
            direction = "non-monotonic"

        out_rows.append({
            "roi_a": e["roi_a"],
            "roi_b": e["roi_b"],
            "mean_r_ls": mean_ls,
            "mean_r_hs": mean_hs,
            "mean_r_patient": mean_pat,
            "direction": direction,
            "spearman_rho": float(rho),
            "p_spearman": float(p_rho),
            "slope_z": float(beta[1]),
            "t_slope_z": float(t_slope),
            "p_slope_z": float(p_slope),
            "n_ls": int(np.sum(x == 0)),
            "n_hs": int(np.sum(x == 1)),
            "n_patient": int(np.sum(x == 2)),
        })

    res = pd.DataFrame(out_rows)

    # FDR on primary monotonic test p-value (choose one: slope or spearman).
    # I recommend slope on z as primary.
    res["q_fdr_slope"] = multipletests(res["p_slope_z"].values, method="fdr_bh")[1]
    res["q_fdr_spearman"] = multipletests(res["p_spearman"].values, method="fdr_bh")[1]

    res = res.sort_values(["q_fdr_slope", "p_slope_z"]).reset_index(drop=True)
    return res