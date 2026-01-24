#!/usr/bin/env python3
"""
denoise.py

Resting-state denoising for fMRIPrep outputs:
- Apply brain mask
- Regress out ONLY:
  - 6 motion terms: trans_x, trans_y, trans_z, rot_x, rot_y, rot_z
  - first up to 5 aCompCor components (a_comp_cor_00..04); if fewer exist, use available
- Band-pass filter (default hp=0.008, lp=0.09) in NiftiMasker
- Smooth AFTER denoising (default 4mm FWHM)
- NO FD-based scrubbing/censoring (assumed already done upstream)

Outputs:
- Writes denoised NIfTI files to cfg['stricon_preproc_folder'] / cfg['velas_preproc_folder']
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nb
from nilearn.input_data import NiftiMasker
from nilearn import image
from nilearn.glm.first_level.design_matrix import make_first_level_design_matrix
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import gaussian_filter


MOTION6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", 
           "trans_x_derivative1","trans_y_derivative1","trans_z_derivative1","rot_x_derivative1","rot_y_derivative1","rot_z_derivative1"]


def _fill_na_with_mean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].isnull().any():
            m = out[c].mean()
            out[c] = out[c].fillna(0.0 if np.isnan(m) else m)
    return out


def select_first_n_acompcor(df: pd.DataFrame, n: int = 5) -> list[str]:
    comps = [c for c in df.columns if re.match(r"^a_comp_cor_\d+$", c)]
    comps = sorted(comps, key=lambda x: int(x.split("_")[-1]))
    return comps[:n]


def find_first(paths: list[Path]) -> Path | None:
    return paths[0] if len(paths) > 0 else None

def build_design_matrix(confounds: pd.DataFrame, tr: float, high_pass_hz: float | None):
    """
    Build design matrix with:
      - confounds (motion6 + acompcor)
      - cosine drift regressors for high-pass (if high_pass_hz is not None)
      - intercept
    """
    n_trs = confounds.shape[0]
    frame_times = np.arange(n_trs) * tr

    if high_pass_hz is not None and high_pass_hz > 0:
        dm = make_first_level_design_matrix(
            frame_times=frame_times,
            high_pass=float(high_pass_hz),
            add_regs=confounds.values,
            add_reg_names=confounds.columns.tolist(),
        )
        # dm includes intercept already
        X = dm.values.astype(np.float64)
        colnames = dm.columns.tolist()
    else:
        # confounds + intercept
        X = np.column_stack([confounds.values, np.ones((n_trs, 1))]).astype(np.float64)
        colnames = confounds.columns.tolist() + ["intercept"]

    return X, colnames


def regress_out(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    OLS regression using stable least squares.
    X: (T, K)
    Y: (T, V)
    Returns residuals: (T, V)
    """
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return Y - (X @ beta)


def lowpass_filter(residuals: np.ndarray, tr: float, low_pass_hz: float, order: int = 4) -> np.ndarray:
    """
    Butterworth low-pass applied along time axis.
    residuals: (T, V)
    """
    fs = 1.0 / tr
    nyq = fs / 2.0
    if low_pass_hz >= nyq:
        raise ValueError(f"low_pass_hz={low_pass_hz} is >= Nyquist={nyq}.")
    wn = low_pass_hz / nyq

    sos = butter(order, wn, btype="lowpass", output="sos")
    # sosfiltfilt supports filtering along an axis for 2D arrays
    return sosfiltfilt(sos, residuals, axis=0).astype(np.float32)


def fwhm_to_sigma_vox(fwhm_mm: float, voxel_sizes_mm: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Convert FWHM (mm) to Gaussian sigma in voxel units for each spatial axis.
    sigma_mm = fwhm / (2*sqrt(2*ln2))
    sigma_vox = sigma_mm / voxel_size_mm
    """
    sigma_mm = float(fwhm_mm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return tuple(sigma_mm / vs for vs in voxel_sizes_mm)


def smooth_4d_gaussian(data_4d: np.ndarray, sigma_vox_xyz: tuple[float, float, float]) -> np.ndarray:
    """
    Apply Gaussian smoothing in space only (not across time).
    data_4d: (X, Y, Z, T)
    """
    sx, sy, sz = sigma_vox_xyz
    # sigma for 4D: (x, y, z, t)
    return gaussian_filter(data_4d, sigma=(sx, sy, sz, 0.0)).astype(np.float32)


def find_fmriprep_rest_files(subject_dir: Path) -> tuple[Path, Path, Path]:
    """
    Returns (bold_preproc, brain_mask, confounds_tsv) for a subject.
    Searches broadly but prefers task-rest if present.
    """
    # Prefer task-rest patterns
    bold_candidates = sorted(subject_dir.rglob("*task-rest*desc-preproc_bold.nii.gz"))
    mask_candidates = sorted(subject_dir.rglob("*task-rest*desc-brain_mask.nii.gz"))
    conf_candidates = sorted(subject_dir.rglob("*task-rest*desc-confounds_timeseries.tsv"))

    # Fallback: any desc-preproc_bold, any desc-brain_mask, any confounds
    if not bold_candidates:
        bold_candidates = sorted(subject_dir.rglob("*desc-preproc_bold.nii.gz"))
    if not mask_candidates:
        mask_candidates = sorted(subject_dir.rglob("*desc-brain_mask.nii.gz"))
    if not conf_candidates:
        conf_candidates = sorted(subject_dir.rglob("*desc-confounds_timeseries.tsv"))

    bold = find_first(bold_candidates)
    mask = find_first(mask_candidates)
    conf = find_first(conf_candidates)

    if bold is None or mask is None or conf is None:
        missing = []
        if bold is None:
            missing.append("desc-preproc_bold.nii.gz")
        if mask is None:
            missing.append("desc-brain_mask.nii.gz")
        if conf is None:
            missing.append("desc-confounds_timeseries.tsv")
        raise FileNotFoundError(f"Missing files in {subject_dir}: {', '.join(missing)}")

    return bold, mask, conf


def denoise_one(
    bold_file,
    mask_file,
    confounds_tsv,
    out_file, tr,
    hp: float | None = 0.01,     # high-pass Hz (rest default)
    lp: float | None = 0.08,      # low-pass Hz (rest default); set None to skip
    smooth_fwhm: float = 4.0,     # mm
    n_acompcor: int = 5,
):
    # Load images
    img = nb.load(str(bold_file))
    mask_img = nb.load(str(mask_file))

    tr = tr

    # Load mask and data
    mask = mask_img.get_fdata().astype(bool)
    data = img.get_fdata(dtype=np.float32)  # (X,Y,Z,T)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D BOLD, got shape {data.shape}")

    Xdim, Ydim, Zdim, T = data.shape

    # Load confounds
    df = pd.read_csv(confounds_tsv, sep="\t", na_values="n/a")

    # Motion 12 required
    missing_motion = [c for c in MOTION6 if c not in df.columns]
    if missing_motion:
        raise ValueError(f"Missing motion columns: {missing_motion}")

    # aCompCor: first up to 5; if fewer exist, use available
    acomp = select_first_n_acompcor(df, n=n_acompcor)
    if len(acomp) == 0:
        raise ValueError("No a_comp_cor_* columns found in TSV.")
    if len(acomp) < n_acompcor:
        print(f"  Note: only {len(acomp)} aCompCor found (requested {n_acompcor}); using available.")

    conf = df[MOTION6 + acomp].copy()
    conf = _fill_na_with_mean(conf)

    if conf.shape[0] != T:
        raise ValueError(f"Timepoints mismatch: BOLD has T={T} but confounds has {conf.shape[0]} rows")

    # Extract masked voxel time series: (V,T) -> transpose to (T,V)
    Y = data[mask, :]  # (V, T)
    Y = Y.T.astype(np.float64)     # (T, V)
    voxel_mean = Y.mean(axis=0, keepdims=True)  # (1, V)

    # Build design matrix with HP cosine drifts + intercept
    X, colnames = build_design_matrix(conf, tr=tr, high_pass_hz=hp)

    # Regress out confounds
    residuals = regress_out(X, Y)

    # Optional low-pass (resting-state)
    if lp is not None and lp > 0:
        residuals = lowpass_filter(residuals.astype(np.float32), tr=tr, low_pass_hz=float(lp)).astype(np.float64)

    # Add mean back (preserve baseline)
    clean_Y = (residuals + voxel_mean).astype(np.float32)  # (T, V)

    # Put back into 4D volume
    clean_4d = np.zeros((Xdim, Ydim, Zdim, T), dtype=np.float32)
    clean_4d[mask, :] = clean_Y.T  # (V,T)

    # Smooth AFTER denoise (spatial only)
    if smooth_fwhm is not None and smooth_fwhm > 0:
        voxel_sizes = img.header.get_zooms()[:3]  # (mm,mm,mm)
        sigma_vox = fwhm_to_sigma_vox(smooth_fwhm, voxel_sizes)
        clean_4d = smooth_4d_gaussian(clean_4d, sigma_vox)
        # Re-apply mask to keep outside-brain clean
        clean_4d *= mask[..., None].astype(np.float32)

    # Save
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_img = nb.Nifti1Image(clean_4d, img.affine, header=img.header)
    out_img.to_filename(str(out_file))


def denoise_dataset( dataset_name: str, deriv_root: str, out_root: str, tr,
                    hp: float = 0.008, lp: float | None = 0.09, smooth_fwhm: float = 4.0, n_acompcor: int = 5,
                    overwrite: bool = False):
    

    #deriv_root = Path(deriv_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[{dataset_name}] Found {len(deriv_root)} subjects")

    ok, fail, skipped = 0, 0, 0

    for sub_dir in deriv_root:

        sub_dir = Path(sub_dir)
        sub_id = sub_dir.name  # sub-XXXX
        print(f"Currently processing for {dataset_name}: {sub_id}")
        try:
            bold, mask, conf = find_fmriprep_rest_files(sub_dir)

            # Output name: keep it obvious + stable
            out_file = out_root / sub_id / f"{sub_id}_task-rest_desc-denoised_bold.nii.gz"
            if out_file.exists() and not overwrite:
                skipped += 1
                continue

            print(f"[{dataset_name}] {sub_id}: denoising")
            denoise_one(
                bold_file=bold,
                mask_file=mask,
                confounds_tsv=conf,
                out_file=out_file,
                hp=hp,
                lp=lp,
                smooth_fwhm=smooth_fwhm,
                tr=tr,
                n_acompcor=n_acompcor,
            )
            ok += 1

        except Exception as e:
            fail += 1
            print(f"[{dataset_name}] {sub_id}: FAILED -> {e}")

    print(f"[{dataset_name}] Done. OK={ok}, Skipped={skipped}, Failed={fail}")
