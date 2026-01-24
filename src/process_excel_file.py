import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

PM_PATTERNS = [
    re.compile(r"^\s*pm\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*±\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\+/-\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE),
    re.compile(r"^\s*\+\/-\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.IGNORECASE),
]

def clean_group_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    return s

def parse_coord_cell(val):
    """
    Returns (values_list, was_pm)
    - If val is 'pm 13' / '±13' / '+/- 13' -> ([-13, +13], True)
    - Else numeric or string numeric -> ([num], False)
    """
    if pd.isna(val):
        raise ValueError("Found NaN coordinate value.")

    if isinstance(val, (int, float, np.integer, np.floating)):
        return [float(val)], False

    s = str(val).strip()

    for pat in PM_PATTERNS:
        m = pat.match(s)
        if m:
            a = float(m.group(1))
            return [-a, +a], True

    try:
        return [float(s)], False
    except ValueError:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)", s)
        if m:
            return [float(m.group(1))], False
        raise ValueError(f"Could not parse coordinate cell: {val!r}")

def build_roi_dataframe(path: str, sheet_name=None, n_acompcor: int = 5) -> pd.DataFrame:
    """
    Reads ROI table (Excel/CSV) with columns: name, x, y, z, group
    Produces dataframe: new_name, x, y, z
    where group subregions become Group1, Group2, ... in row order.
    Expands 'pm/±' in x into left/right rows.
    """
    path = str(path)
    p = Path(path)

    # ---- Load file robustly (CSV or Excel) ----
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(path, na_values="n/a")
    else:
        # IMPORTANT FIX:
        # If sheet_name is None, do NOT pass None to pandas (it returns dict of sheets).
        if sheet_name is None:
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        else:
            df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

        # If something still produced a dict, take the first sheet
        if isinstance(df, dict):
            first_key = next(iter(df))
            df = df[first_key]

    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"name", "x", "y", "z", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ---- Build stable Group1/Group2/... numbering in row order ----
    df["_group_clean"] = df["group"].apply(clean_group_name)
    df["_sub_idx"] = df.groupby("_group_clean", sort=False).cumcount() + 1
    df["_base_name"] = df["_group_clean"] + df["_sub_idx"].astype(str)

    rows = []
    for _, r in df.iterrows():
        base = r["_base_name"]

        x_vals, x_pm = parse_coord_cell(r["x"])
        y_vals, _ = parse_coord_cell(r["y"])
        z_vals, _ = parse_coord_cell(r["z"])

        for x, y, z in product(x_vals, y_vals, z_vals):
            new_name = base

            # Only apply left/right naming if x was ± encoded (your spec)
            if x_pm:
                if x < 0:
                    new_name = f"{base}_left"
                elif x > 0:
                    new_name = f"{base}_right"
                else:
                    new_name = f"{base}_mid"

            rows.append({"new_name": new_name, "x": float(x), "y": float(y), "z": float(z)})

    out = pd.DataFrame(rows)

    # Keep consistent ordering: base then left then right
    def _side_key(n):
        if n.endswith("_left"):
            return 0
        if n.endswith("_right"):
            return 1
        if n.endswith("_mid"):
            return 2
        return 3

    out["_base"] = out["new_name"].str.replace(r"_(left|right|mid)$", "", regex=True)
    out["_side"] = out["new_name"].apply(_side_key)
    out = out.sort_values(["_base", "_side"]).drop(columns=["_base", "_side"]).reset_index(drop=True)

    return out