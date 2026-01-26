#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nb

from nilearn.maskers import NiftiSpheresMasker


def load_rois(path: str) -> pd.DataFrame:
    """
    Accepts CSV with either:
      - roi,x,y,z
      - new_name,x,y,z
    Returns df with columns: roi,x,y,z
    """
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    if "roi" not in df.columns:
        if "new_name" in df.columns:
            df = df.rename(columns={"new_name": "roi"})
        else:
            raise ValueError("ROI CSV must have 'roi' or 'new_name' column.")

    for c in ["x", "y", "z"]:
        if c not in df.columns:
            raise ValueError(f"ROI CSV missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="raise")

    return df[["roi", "x", "y", "z"]].copy()


def find_denoised_bold(sub_dir: Path) -> Path:
    """
    Looks for your denoised output inside each subject folder.
    Expected pattern: *desc-denoised_bold.nii.gz
    """
    cand = sorted(sub_dir.rglob("*desc-denoised_bold.nii.gz"))
    if not cand:
        # fallback: any nii.gz (not ideal, but helps debugging)
        cand = sorted(sub_dir.rglob("*.nii.gz"))
    if not cand:
        raise FileNotFoundError(f"No NIfTI found in {sub_dir}")
    return cand[0]


def extract_sphere_timeseries(img_path: Path, rois: pd.DataFrame, radius_mm: float) -> np.ndarray:
    """
    Returns ts shape (T, R)
    """
    img = nb.load(str(img_path))
    tr = 1.17

    coords = rois[["x", "y", "z"]].to_numpy().tolist()

    masker = NiftiSpheresMasker(
        seeds=coords,
        radius=radius_mm,
        detrend=False,       # already denoised / filtered upstream
        standardize=False,   # keep as-is
        t_r=tr,
        low_pass=None,
        high_pass=None,
        smoothing_fwhm=None,
        allow_overlap=True,
        memory="nilearn_cache",
        memory_level=1,
        verbose=0,
    )
    ts = masker.fit_transform(img)  # (T, R)
    return ts


def pearson_matrix(ts: np.ndarray) -> np.ndarray:
    """
    Pearson correlation between ROI time series.
    ts: (T, R)
    returns r: (R, R) with diagonal = 0
    """
    r = np.corrcoef(ts.T)
    np.fill_diagonal(r, 0.0)
    # safety clip
    r = np.clip(r, -1.0, 1.0)
    return r.astype(np.float32)


def save_fc_npz(sub_dir: Path, roi_names: list[str], r: np.ndarray) -> Path:
    out_path = sub_dir / "fc_matrices.npz"
    np.savez_compressed(out_path, roi_names=np.array(roi_names), r=r)
    return out_path


def iter_subject_dirs(root: Path) -> list[Path]:
    # subject folders are directly under your preproc folder (as you said)
    subs = sorted([p for p in root.glob("sub-*") if p.is_dir()])
    if subs:
        return subs
    # fallback: deeper
    return sorted({p for p in root.rglob("sub-*") if p.is_dir()})


def run_dataset(preproc_root: str, roi_csv: str, overwrite: bool, radius_mm: float):
    root = Path(preproc_root)
    if not root.exists():
        raise FileNotFoundError(f"Preproc folder not found: {root}")

    rois = load_rois(roi_csv)
    roi_names = rois["roi"].tolist()

    subs = iter_subject_dirs(root)
    print(f"[{root.name}] Found {len(subs)} subjects in {root}", flush=True)

    ok, skip, fail = 0, 0, 0
    for sub_dir in subs:
        out_npz = sub_dir / "fc_matrices.npz"
        if out_npz.exists() and not overwrite:
            skip += 1
            continue

        try:
            bold = find_denoised_bold(sub_dir)
            ts = extract_sphere_timeseries(bold, rois, radius_mm=radius_mm)
            r = pearson_matrix(ts)
            save_fc_npz(sub_dir, roi_names, r)
            ok += 1
            print(f"{sub_dir.name}: saved fc_matrices.npz", flush=True)
        except Exception as e:
            fail += 1
            print(f"{sub_dir.name}: FAILED -> {e}", flush=True)

    print(f"Done: OK={ok}, Skipped={skip}, Failed={fail}", flush=True)