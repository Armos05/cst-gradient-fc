#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ---------- ID helpers ----------
def normalize_to_int_subject(x):
    if pd.isna(x):
        return None
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else None


def detect_subject_column(df: pd.DataFrame) -> str:
    candidates = [
        "subject", "subject_id", "participant_id", "participant", "sub", "sub_id",
        "id", "rid", "ID", "Subject", "SubjectID", "RID"
    ]
    cols = list(df.columns)
    for c in candidates:
        if c in cols:
            return c
    lower_map = {c.lower(): c for c in cols}
    for c in [x.lower() for x in candidates]:
        if c in lower_map:
            return lower_map[c]
    return cols[0]


# ---------- FC loading ----------
def load_fc_dataset(preproc_root: str):
    """
    Returns:
      subjects_int: list[int]
      roi_names: list[str]
      mats_r: (N, R, R)
    """
    root = Path(preproc_root)
    sub_dirs = sorted([p for p in root.glob("sub-*") if p.is_dir()])

    subjects_int, mats = [], []
    roi_names = None

    for sd in sub_dirs:
        npz = sd / "fc_matrices.npz"
        if not npz.exists():
            continue

        sid = normalize_to_int_subject(sd.name)
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

    return subjects_int, roi_names, np.stack(mats, axis=0)


def edge_index(roi_names):
    R = len(roi_names)
    rows = []
    for i in range(R):
        for j in range(i + 1, R):
            rows.append((i, j, roi_names[i], roi_names[j]))
    return rows


def extract_edge_matrix(mats_r, edges):
    """
    mats_r: (N, R, R)
    edges: list of (i, j, roi_a, roi_b)
    returns E: (N, n_edges)
    """
    N = mats_r.shape[0]
    E = np.zeros((N, len(edges)), dtype=np.float64)
    for k, (i, j, _, _) in enumerate(edges):
        E[:, k] = mats_r[:, i, j]
    return E


# ---------- Behavior loading ----------
def load_panss_from_excel(excel_path: str, sheet: str, cols: list[str]):
    df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
    subj_col = detect_subject_column(df)

    # case-insensitive col mapping
    col_map = {c.lower(): c for c in df.columns}
    subj_col_real = col_map.get(subj_col.lower(), subj_col)

    # resolve PANSS columns case-insensitively
    real_cols = []
    for c in cols:
        if c in df.columns:
            real_cols.append(c)
        elif c.lower() in col_map:
            real_cols.append(col_map[c.lower()])
        else:
            # missing -> will be created as NA
            real_cols.append(None)

    keep = [subj_col_real] + [c for c in real_cols if c is not None]
    out = df[keep].copy()

    out["subject_int"] = out[subj_col_real].apply(normalize_to_int_subject)
    out = out.dropna(subset=["subject_int"])
    out["subject_int"] = out["subject_int"].astype(int)

    # ensure all requested columns exist (missing => NA)
    for wanted, real in zip(cols, real_cols):
        if real is None:
            out[wanted] = np.nan
        elif real != wanted:
            out[wanted] = out[real]

    # force numeric
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out[["subject_int"] + cols]


def compute_panss_total(df: pd.DataFrame, cols: list[str], require_all: bool = False):
    """
    If require_all=True: total is NA unless all 7 are present.
    Else: sum available items; NA if all missing.
    """
    vals = df[cols]
    if require_all:
        total = vals.sum(axis=1, min_count=len(cols))
    else:
        total = vals.sum(axis=1, min_count=1)
    return total


# ---------- Correlations ----------
def corr_one(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return np.nan, np.nan, int(np.sum(m))
    r, p = pearsonr(x[m], y[m])
    return float(r), float(p), int(np.sum(m))


def run_dataset_panss(excel_path, sheet, panss_cols, preproc_root, out_csv, require_all_for_total=False):
    subs, roi_names, mats = load_fc_dataset(preproc_root)
    edges = edge_index(roi_names)
    E = extract_edge_matrix(mats, edges)  # (N, n_edges)

    beh = load_panss_from_excel(excel_path, sheet=sheet, cols=panss_cols)

    beh["PANSS_total"] = compute_panss_total(beh, panss_cols, require_all=require_all_for_total)

    # keep only subjects that have FC
    fc_subs = pd.Series(subs, name="subject_int")
    beh = beh.merge(fc_subs.to_frame(), on="subject_int", how="inner")

    # align behavior order to FC order
    idx = {sid: k for k, sid in enumerate(subs)}
    beh["fc_index"] = beh["subject_int"].map(idx)
    beh = beh.dropna(subset=["fc_index"]).sort_values("fc_index")
    fc_idx = beh["fc_index"].astype(int).values

    # slice edge matrix to matched subjects
    E2 = E[fc_idx, :]

    # Build tidy results
    results = []
    score_list = panss_cols + ["PANSS_total"]

    for score in score_list:
        y = beh[score].to_numpy(dtype=float)

        for k, (i, j, roi_a, roi_b) in enumerate(edges):
            x = E2[:, k]
            r, p, n = corr_one(x, y)
            results.append({
                "sheet": sheet,
                "score": score,
                "roi_a": roi_a,
                "roi_b": roi_b,
                "i": i,
                "j": j,
                "r": r,
                "p": p,
                "n": n,
                "n_subjects_with_fc": int(E2.shape[0]),
            })

    res = pd.DataFrame(results)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (rows={len(res)})")