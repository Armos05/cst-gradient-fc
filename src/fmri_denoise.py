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


MOTION6 = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


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
    bold_file: Path,
    mask_file: Path,
    confounds_tsv: Path,
    out_file: Path,
    hp: float = 0.008,
    lp: float | None = 0.09,
    smooth_fwhm: float = 4.0,
    tr: float | None = None,
    n_acompcor: int = 5,
):
    img = nb.load(str(bold_file))
    mask = nb.load(str(mask_file))

    if tr is None:
        tr = float(img.header.get_zooms()[-1])

    df = pd.read_csv(confounds_tsv, sep="\t", na_values="n/a")

    # Validate motion 6
    missing_motion = [c for c in MOTION6 if c not in df.columns]
    if missing_motion:
        raise ValueError(f"{confounds_tsv.name}: missing required motion columns: {missing_motion}")

    acomp_cols = select_first_n_acompcor(df, n=n_acompcor)
    if len(acomp_cols) == 0:
        raise ValueError(f"{confounds_tsv.name}: no a_comp_cor_* columns found")
    if len(acomp_cols) < n_acompcor:
        print(f"  Note: only {len(acomp_cols)} aCompCor comps found (requested {n_acompcor}); using available.")

    used = MOTION6 + acomp_cols
    conf = _fill_na_with_mean(df[used].copy())

    # Ensure timepoints match
    if img.shape[-1] != conf.shape[0]:
        raise ValueError(
            f"Timepoints mismatch: {bold_file.name} has {img.shape[-1]} vols but confounds has {conf.shape[0]} rows"
        )

    masker = NiftiMasker(
        mask_img=mask,
        t_r=tr,
        detrend=True,
        standardize=False,
        high_pass=hp,
        low_pass=lp,
        smoothing_fwhm=None,  # smooth AFTER denoise
    )

    clean_ts = masker.fit_transform(img, confounds=conf.values)
    clean_img = masker.inverse_transform(clean_ts)

    if smooth_fwhm is not None and smooth_fwhm > 0:
        clean_img = image.smooth_img(clean_img, smooth_fwhm)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    clean_img.to_filename(str(out_file))


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
