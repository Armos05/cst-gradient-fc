#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import pandas as pd

# -----------------------
# ROI name -> parent region
# -----------------------
def parent_region(roi: str) -> str:
    """
    Collapse subparts:
      ACC1_left -> ACC
      Caudate3_right -> Caudate
      vlPFC2 -> vlPFC
      NAcc1 -> NAcc
    """
    s = str(roi).strip()

    # Remove laterality suffix if present
    s = re.sub(r"_(left|right|mid)$", "", s, flags=re.IGNORECASE)

    # Take leading letters only (keeps mixed-case like vlPFC)
    m = re.match(r"^([A-Za-z]+)", s)
    base = m.group(1) if m else s.split("_")[0]

    # Normalize common aliases (case-insensitive)
    key = base.lower()
    alias = {
        "nacc": "NAcc",
        "nac": "NAcc",
        "nucleusaccumbens": "NAcc",
        "accumbens": "NAcc",
        "ofc": "OFC",
        "acc": "ACC",
        "hippocampus": "Hippocampus",
        "amygdala": "Amygdala",
        "thalamus": "Thalamus",
        "caudate": "Caudate",
        "putamen": "Putamen",
        "vlpfc": "vlPFC",
        "dlpfc": "dlPFC",
    }
    return alias.get(key, base)


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    ra, rb = parent_region(a), parent_region(b)
    return tuple(sorted([ra, rb]))


# -----------------------
# Literature pattern spec
# -----------------------
# For VELAS trend tables (ls, hs, patient):
#   "reduced"  => expect ls > hs > patient  (monotonic decreasing)
#   "increased"=> expect ls < hs < patient  (monotonic increasing)
VELAS_EXPECTED = {
    # Schizotypy (mostly reduced)
    ("Hippocampus", "Thalamus"): "reduced",
    ("Hippocampus", "Caudate"): "reduced",
    ("OFC", "NAcc"): "reduced",
    ("ACC", "Caudate"): "reduced",
    ("ACC", "NAcc"): "reduced",
    ("vlPFC", "Caudate"): "reduced",
    ("vlPFC", "NAcc"): "reduced",
    ("dlPFC", "Caudate"): "reduced",
    ("dlPFC", "NAcc"): "reduced",

    # At-risk (you listed both increases and decreases)
    ("Amygdala", "vlPFC"): "increased",
    ("Amygdala", "Putamen"): "increased",
    ("Hippocampus", "NAcc"): "reduced",
    ("Thalamus", "Caudate"): "reduced",
    ("ACC", "Thalamus"): "reduced",
    ("ACC", "Caudate"): "reduced",
    ("dlPFC", "Thalamus"): "reduced",
    ("dlPFC", "Caudate"): "reduced",

    # Schizophrenia (mixed)
    ("Amygdala", "NAcc"): "increased",
    ("Amygdala", "OFC"): "reduced",
    ("Amygdala", "ACC"): "reduced",
    ("Amygdala", "vlPFC"): "reduced",
    ("Hippocampus", "ACC"): "reduced",
    ("vlPFC", "Thalamus"): "reduced",
    ("Hippocampus", "NAcc"): "reduced",   # appears above too
    ("Thalamus", "ACC"): "reduced",
    ("Thalamus", "Putamen"): "reduced",
    ("dlPFC", "Thalamus"): "reduced",
    ("dlPFC", "Caudate"): "reduced",
}

# For STRICON Patients vs Healthy:
#   "reduced" => Patients < Healthy
#   "increased" => Patients > Healthy
STRICON_EXPECTED = {
    # If you want to apply the same edge-list expectations to STRICON too:
    # (you can add/remove pairs here; leaving it mirrored for convenience)
    ("Thalamus", "Caudate"): "reduced",
    ("ACC", "Thalamus"): "reduced",
    ("ACC", "Caudate"): "reduced",
    ("dlPFC", "Thalamus"): "reduced",
    ("dlPFC", "Caudate"): "reduced",
    ("Hippocampus", "NAcc"): "reduced",
    ("Amygdala", "vlPFC"): "increased",
    ("Amygdala", "Putamen"): "increased",
}


# -----------------------
# Matching logic
# -----------------------
def velas_direction_ok(row, expected: str) -> bool:
    ls = row.get("mean_r_ls")
    hs = row.get("mean_r_hs")
    pt = row.get("mean_r_patient")

    if pd.isna(ls) or pd.isna(hs) or pd.isna(pt):
        return False

    if expected == "reduced":
        return (ls > hs > pt)
    if expected == "increased":
        return (ls < hs < pt)
    return False


def stricon_direction_ok(row, expected: str) -> bool:
    # these are raw means from stats_stricon
    pat = row.get("mean_r_patients")
    hc = row.get("mean_r_healthy")

    if pd.isna(pat) or pd.isna(hc):
        return False

    if expected == "reduced":
        return (pat < hc)
    if expected == "increased":
        return (pat > hc)
    return False


def match_velas(velas_csv: str, out_csv: str):
    df = pd.read_csv(velas_csv)
    # normalize pairs
    df["pair"] = df.apply(lambda r: canonical_pair(r["roi_a"], r["roi_b"]), axis=1)
    df["pair"] = df["pair"].apply(tuple)

    expected_map = {tuple(sorted(k)): v for k, v in VELAS_EXPECTED.items()}

    df["expected"] = df["pair"].map(expected_map)
    df = df[df["expected"].notna()].copy()

    df["matches_trend"] = df.apply(lambda r: velas_direction_ok(r, r["expected"]), axis=1)
    hit = df[df["matches_trend"]].copy()

    # helpful columns at front
    cols_front = ["roi_a", "roi_b", "pair", "expected", "mean_r_ls", "mean_r_hs", "mean_r_patient",
                 "direction", "p_slope_z", "q_fdr_slope"]
    cols = [c for c in cols_front if c in hit.columns] + [c for c in hit.columns if c not in cols_front]
    hit = hit[cols].sort_values(["q_fdr_slope", "p_slope_z"], na_position="last")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    hit.to_csv(out_csv, index=False)
    print(f"[VELAS] matched edges saved -> {out_csv} (n={len(hit)})")


def match_stricon(stricon_csv: str, out_csv: str):
    df = pd.read_csv(stricon_csv)
    df["pair"] = df.apply(lambda r: canonical_pair(r["roi_a"], r["roi_b"]), axis=1)
    df["pair"] = df["pair"].apply(tuple)

    expected_map = {tuple(sorted(k)): v for k, v in STRICON_EXPECTED.items()}

    df["expected"] = df["pair"].map(expected_map)
    df = df[df["expected"].notna()].copy()

    df["matches_dir"] = df.apply(lambda r: stricon_direction_ok(r, r["expected"]), axis=1)
    hit = df[df["matches_dir"]].copy()

    cols_front = ["roi_a", "roi_b", "pair", "expected", "mean_r_patients", "mean_r_healthy",
                 "direction", "p", "q_fdr"]
    cols = [c for c in cols_front if c in hit.columns] + [c for c in hit.columns if c not in cols_front]
    hit = hit[cols].sort_values(["q_fdr", "p"], na_position="last")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    hit.to_csv(out_csv, index=False)
    print(f"[STRICON] matched edges saved -> {out_csv} (n={len(hit)})")