#!/usr/bin/env python
"""Audit completeness and validity of study metadata without exposing patient IDs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_META = ["Age", "X", "Y", "Z", "Calcification", "mismatch"]
LABELS = ["IDH_mutation", "1p19q_codeletion"]
MISSING_CODES = {"Calcification": [-1.0]}
ALLOWED_VALUES = {
    "IDH_mutation": [0.0, 1.0],
    "1p19q_codeletion": [0.0, 1.0],
    "Calcification": [0.0, 0.5, 1.0],
}

def eq_any(s: pd.Series, values: list[float]) -> pd.Series:
    a = pd.to_numeric(s, errors="coerce").to_numpy(float)
    out = np.zeros(len(a), dtype=bool)
    for v in values:
        out |= np.isclose(a, v, rtol=0, atol=1e-12, equal_nan=False)
    return pd.Series(out, index=s.index)

def audit(path: Path, outdir: Path, expected_n: int | None) -> None:
    df = pd.read_csv(path, low_memory=False)
    id_column = "patient_id" if "patient_id" in df.columns else "No"
    if id_column not in df.columns:
        raise ValueError("The input CSV must contain either 'patient_id' or 'No'.")
    required = [id_column, *LABELS, *DEFAULT_META]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    rows = []
    issues = []

    # The identifier is validated as text and is not included in numeric metadata
    # summaries. This supports both public synthetic IDs and legacy numeric IDs.
    id_text = df[id_column].astype("string").str.strip()
    id_blank = df[id_column].isna() | id_text.isin(["", "NA", "NaN", "nan", "None", "null"])
    if id_blank.any():
        issues.append({"variable": id_column, "issue": "blank_or_na", "n": int(id_blank.sum())})

    variables_to_audit = [*LABELS, *DEFAULT_META]
    for col in variables_to_audit:
        raw = df[col]
        num = pd.to_numeric(raw, errors="coerce")
        blank = raw.isna() | raw.astype(str).str.strip().isin(["", "NA", "NaN", "nan", "None", "null"])
        nonnumeric = (~blank) & num.isna()
        arr = num.to_numpy(float)
        posinf = np.isposinf(arr)
        neginf = np.isneginf(arr)
        special = eq_any(num, MISSING_CODES.get(col, [])) if col in MISSING_CODES else pd.Series(False, index=df.index)
        effective_missing = blank | nonnumeric | pd.Series(posinf | neginf, index=df.index) | special
        observed = num[~effective_missing & np.isfinite(num)]
        allowed_invalid = pd.Series(False, index=df.index)
        if col in ALLOWED_VALUES:
            allowed_invalid = (~effective_missing) & (~eq_any(num, ALLOWED_VALUES[col]))
        rows.append({
            "variable": col,
            "n_total": len(df),
            "n_observed": int((~effective_missing).sum()),
            "n_missing_total": int(effective_missing.sum()),
            "missing_percent": round(100 * effective_missing.mean(), 3),
            "n_blank_or_na": int(blank.sum()),
            "n_nonnumeric": int(nonnumeric.sum()),
            "n_posinf": int(posinf.sum()),
            "n_neginf": int(neginf.sum()),
            "n_special_missing_code": int(special.sum()),
            "n_invalid_allowed_value": int(allowed_invalid.sum()),
            "min_observed": float(observed.min()) if len(observed) else np.nan,
            "max_observed": float(observed.max()) if len(observed) else np.nan,
            "mean_observed": float(observed.mean()) if len(observed) else np.nan,
            "sd_observed": float(observed.std(ddof=1)) if len(observed) > 1 else np.nan,
            "n_unique_observed": int(observed.nunique()),
        })
        for issue_type, mask in [
            ("blank_or_na", blank), ("nonnumeric", nonnumeric),
            ("positive_infinity", pd.Series(posinf, index=df.index)),
            ("negative_infinity", pd.Series(neginf, index=df.index)),
            ("special_missing_code", special),
            ("invalid_allowed_value", allowed_invalid),
        ]:
            if mask.any():
                issues.append({"variable": col, "issue": issue_type, "n": int(mask.sum())})
    summary = pd.DataFrame(rows)
    duplicate_ids = int(id_text[~id_blank].duplicated(keep=False).sum())
    exact_duplicate_rows = int(df.duplicated(keep=False).sum())
    overall = {
        "input_file": path.name, "n_rows": int(len(df)), "n_columns": int(len(df.columns)),
        "expected_n": expected_n, "expected_n_pass": expected_n is None or len(df) == expected_n,
        "identifier_column": id_column, "blank_patient_id_rows": int(id_blank.sum()),
        "duplicate_patient_id_rows": duplicate_ids, "exact_duplicate_rows": exact_duplicate_rows,
        "all_required_columns_present": True,
        "all_non_calcification_meta_complete": bool((summary.loc[summary.variable.isin(["Age","X","Y","Z","mismatch"]), "n_missing_total"] == 0).all()),
        "calcification_missing_n": int(summary.loc[summary.variable.eq("Calcification"), "n_missing_total"].iloc[0]),
    }
    # Value counts for labels and calcification; no patient IDs.
    vc = []
    for col in [*LABELS, "Calcification"]:
        for value, n in df[col].value_counts(dropna=False).items():
            vc.append({"variable": col, "raw_value": str(value), "n": int(n), "percent": round(100*n/len(df),3)})
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "metadata_completeness_summary.csv", index=False)
    pd.DataFrame(vc).to_csv(outdir / "metadata_value_counts.csv", index=False)
    pd.DataFrame(issues, columns=["variable","issue","n"]).to_csv(outdir / "metadata_issue_summary.csv", index=False)
    (outdir / "metadata_audit_overall.json").write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(overall, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--outdir", type=Path, default=Path("metadata_audit_output"))
    p.add_argument("--expected-n", type=int, default=217)
    a = p.parse_args()
    audit(a.csv, a.outdir, a.expected_n)
