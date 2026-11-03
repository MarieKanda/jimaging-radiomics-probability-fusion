#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J. Imaging manuscript table and figure pipeline

Expected directory layout (recommended):

ROOT/
  1p19q/
    all-modality/
    mri-only/
    pet-only/
  IDH/
    all-modality/
    mri-only/
    pet-only/

Each run directory should contain the seven files produced by the existing
probability-fusion pipeline, using either the generic names shown below or
names with an arbitrary suffix after the stem:

  all_predictions_probability_fusion*.csv
  all_fold_metrics_probability_fusion*.csv
  final_summary_probability_fusion*.csv
  radiomics_selected_features_probability_fusion*.csv
  radiomics_inner_tuning_records_probability_fusion*.csv
  fusion_family_selection_records*.csv
  fusion_selected_candidate_frequency*.csv

Analyses:
  01 QC
  02 Patient-level aggregation
  03 Table 3 paired AUC comparisons with stratified paired bootstrap
  04 Calibration analysis
  05 Modality comparisons
  06 Table 2 / Table 4 generation
  07 Feature-selection diagnostics for all-modality runs
  08 ROC, calibration, feature-frequency, and optional binning figures

Optional binning diagnostic input:
  ROOT/binning_diagnostics.csv
Required columns:
  task, modality, image_type, bin_width, occupied_bins
Optional columns:
  patient_id, extraction_failed, constant_feature_count,
  near_constant_feature_count

Important statistical conventions:
  * Patient-level probabilities are the mean of the ten OOS predictions.
  * Bootstrap resampling is stratified by outcome and paired across models.
  * Table 3 uses one Holm family containing eight comparisons.
  * Modality comparisons use one Holm family containing four comparisons.
  * Raw bootstrap p values are two-sided and based on the empirical null-tail
    probability with a +1 finite-sample correction.
  * Inferential results are exploratory. Bold in descriptive tables indicates
    the largest mean only and does not imply statistical significance.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

TASKS = ("1p19q", "IDH")
CONDITIONS = ("all-modality", "mri-only", "pet-only")
EXPECTED_CANDIDATES = (
    "meta_only_prob",
    "radiomics_only_prob",
    "weighted_oof_selected",
    "logit_stacking_oof_selected",
)
# Display order aligned with Table 2; model colors and line styles are unchanged.
FIGURE_MODEL_ORDER = (
    "weighted_oof_selected",
    "logit_stacking_oof_selected",
    "meta_only_prob",
    "radiomics_only_prob",
)
CANDIDATE_LABELS = {
    "meta_only_prob": "Meta-information only",
    "radiomics_only_prob": "Radiomics only",
    "weighted_oof_selected": "Weighted averaging",
    "logit_stacking_oof_selected": "Logit stacking",
}
CANDIDATE_ORDER = {name: i for i, name in enumerate(EXPECTED_CANDIDATES)}
MODALITIES = ("FLAIR", "T1", "T2", "MET", "FDG", "CHO")
MRI_MODALITIES = {"FLAIR", "T1", "T2"}
PET_MODALITIES = {"MET", "FDG", "CHO"}
PROBABILITY_COLUMNS = ("meta_prob", "radiomics_prob", "y_prob")
N_ROC_GRID = 101
# =========================================================
# Publication figure style (v3)
# =========================================================
FIG_TITLE_FONTSIZE = 13
AXIS_LABEL_FONTSIZE = 11
TICK_LABEL_FONTSIZE = 10
LEGEND_FONTSIZE = 9
ANNOTATION_FONTSIZE = 10

MODEL_COLORS = {
    "meta_only_prob": "#E69F00",
    "radiomics_only_prob": "#009E73",
    "weighted_oof_selected": "#5671E9",
    "logit_stacking_oof_selected": "#CC79A7",
}
MODEL_STYLES = {
    "meta_only_prob": (0, (5, 2)),
    "radiomics_only_prob": (0, (5, 2, 1, 2)),
    "weighted_oof_selected": "-",
    "logit_stacking_oof_selected": (0, (1, 1.5)),
}
MODEL_LINEWIDTHS = {
    "meta_only_prob": 1.35,
    "radiomics_only_prob": 1.35,
    "weighted_oof_selected": 2.2,
    "logit_stacking_oof_selected": 2.0,
}
MODALITY_COLORS = {
    "T1": "#E60012",
    "T2": "#F39800",
    "FLAIR": "#FFF100",
    "MET": "#009944",
    "FDG": "#00A0E9",
    "CHO": "#92278F",
    "Unknown": "#9D9D9D",
}
MODALITY_DISPLAY = {"T1": "T1WI", "T2": "T2WI"}
plt.rcParams.update({
    "font.size": TICK_LABEL_FONTSIZE,
    "axes.titlesize": FIG_TITLE_FONTSIZE,
    "axes.labelsize": AXIS_LABEL_FONTSIZE,
    "xtick.labelsize": TICK_LABEL_FONTSIZE,
    "ytick.labelsize": TICK_LABEL_FONTSIZE,
    "legend.fontsize": LEGEND_FONTSIZE,
})

FILE_PATTERNS = {
    "predictions": "all_predictions_probability_fusion*.csv",
    "fold_metrics": "all_fold_metrics_probability_fusion*.csv",
    "summary": "final_summary_probability_fusion*.csv",
    "selected_features": "radiomics_selected_features_probability_fusion*.csv",
    "inner_tuning": "radiomics_inner_tuning_records_probability_fusion*.csv",
    "fusion_records": "fusion_family_selection_records*.csv",
    "fusion_frequency": "fusion_selected_candidate_frequency*.csv",
}
REQUIRED_PRED_COLUMNS = {
    "patient_id", "y_true", "meta_prob", "radiomics_prob", "y_prob",
    "threshold", "y_pred", "seed", "fold", "candidate", "selected_candidate",
}
SUMMARY_METRICS = (
    "auc", "ap", "acc", "bal_acc", "sensitivity", "specificity"
)


@dataclass(frozen=True)
class RunKey:
    task: str
    condition: str

    @property
    def slug(self) -> str:
        return f"{self.task}_{self.condition}"


@dataclass
class RunData:
    key: RunKey
    directory: Path
    files: Dict[str, Optional[Path]]
    predictions: Optional[pd.DataFrame] = None
    fold_metrics: Optional[pd.DataFrame] = None
    summary: Optional[pd.DataFrame] = None
    selected_features: Optional[pd.DataFrame] = None
    inner_tuning: Optional[pd.DataFrame] = None
    fusion_records: Optional[pd.DataFrame] = None
    fusion_frequency: Optional[pd.DataFrame] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze six probability-fusion runs for the J. Imaging revision."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="Root containing the six run directories.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: ROOT/analysis_outputs).")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Number of stratified paired bootstrap replicates.")
    parser.add_argument("--random-state", type=int, default=20261028,
                        help="Random seed for bootstrap analysis.")
    parser.add_argument("--expected-patients", type=int, default=217)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument("--expected-folds", type=int, default=5)
    parser.add_argument("--strict", action="store_true",
                        help="Stop after QC if any ERROR is present.")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize_condition(text: str) -> Optional[str]:
    s = text.lower().replace("_", "-").replace(" ", "-")
    if "mri" in s and "only" in s:
        return "mri-only"
    if "pet" in s and "only" in s:
        return "pet-only"
    if "all" in s and any(x in s for x in ("mod", "six", "modal")):
        return "all-modality"
    return None


def normalize_task(text: str) -> Optional[str]:
    s = text.lower().replace("/", "").replace("-", "")
    if "1p19q" in s:
        return "1p19q"
    if "idh" in s:
        return "IDH"
    return None


def discover_run_directory(root: Path, key: RunKey) -> Optional[Path]:
    direct_candidates = [
        root / key.task / key.condition,
        root / key.task.lower() / key.condition,
        root / f"{key.task}_{key.condition}",
        root / f"{key.task}-{key.condition}",
        root / key.slug,
    ]
    for p in direct_candidates:
        if p.is_dir():
            return p
    matches = []
    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        context = "/".join(part for part in p.parts[-3:])
        if normalize_task(context) == key.task and normalize_condition(context) == key.condition:
            if any(p.glob(FILE_PATTERNS["predictions"])):
                matches.append(p)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple directories matched {key.slug}: {matches}")
    return None


def choose_file(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(pattern))
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    generic = [p for p in files if p.name == pattern.replace("*", "")]
    if len(generic) == 1:
        return generic[0]
    raise RuntimeError(f"Multiple files matched {pattern} in {directory}: {files}")


def load_runs(root: Path) -> Dict[RunKey, RunData]:
    runs: Dict[RunKey, RunData] = {}
    for task in TASKS:
        for condition in CONDITIONS:
            key = RunKey(task, condition)
            directory = discover_run_directory(root, key)
            if directory is None:
                files = {name: None for name in FILE_PATTERNS}
                runs[key] = RunData(key, root / key.slug, files)
                continue
            files = {name: choose_file(directory, pattern)
                     for name, pattern in FILE_PATTERNS.items()}
            run = RunData(key, directory, files)
            for attr, path in files.items():
                if path is not None:
                    setattr(run, attr, read_csv(path))
            runs[key] = run
    return runs


def qc_row(key: RunKey, check: str, status: str, observed, expected="", detail="") -> dict:
    return {
        "task": key.task,
        "condition": key.condition,
        "check": check,
        "status": status,
        "observed": observed,
        "expected": expected,
        "detail": detail,
    }


def run_qc(runs: Mapping[RunKey, RunData], expected_patients: int,
           expected_seeds: int, expected_folds: int) -> pd.DataFrame:
    rows: List[dict] = []
    reference_sets: Dict[str, set] = {}
    reference_labels: Dict[str, pd.Series] = {}

    for key, run in runs.items():
        pred = run.predictions
        if pred is None:
            rows.append(qc_row(key, "all_predictions file", "ERROR", "missing", "present"))
            continue
        missing_cols = sorted(REQUIRED_PRED_COLUMNS - set(pred.columns))
        rows.append(qc_row(key, "required columns", "PASS" if not missing_cols else "ERROR",
                           len(missing_cols), 0, "; ".join(missing_cols)))
        if missing_cols:
            continue

        pred = pred.copy()
        pred["patient_id"] = pred["patient_id"].astype(str)
        unique_patients = pred["patient_id"].nunique()
        rows.append(qc_row(key, "unique patients",
                           "PASS" if unique_patients == expected_patients else "ERROR",
                           unique_patients, expected_patients))

        seeds = sorted(pred["seed"].dropna().unique().tolist())
        rows.append(qc_row(key, "number of seeds",
                           "PASS" if len(seeds) == expected_seeds else "ERROR",
                           len(seeds), expected_seeds, str(seeds)))
        folds = sorted(pred["fold"].dropna().unique().tolist())
        rows.append(qc_row(key, "outer fold labels",
                           "PASS" if len(folds) == expected_folds else "ERROR",
                           len(folds), expected_folds, str(folds)))

        candidates = sorted(pred["candidate"].dropna().unique().tolist())
        missing_candidates = sorted(set(EXPECTED_CANDIDATES) - set(candidates))
        extra_candidates = sorted(set(candidates) - set(EXPECTED_CANDIDATES))
        rows.append(qc_row(key, "four candidate families",
                           "PASS" if not missing_candidates and not extra_candidates else "ERROR",
                           len(candidates), 4,
                           f"missing={missing_candidates}; extra={extra_candidates}"))

        exact_dups = int(pred.duplicated().sum())
        rows.append(qc_row(key, "exact duplicate rows",
                           "PASS" if exact_dups == 0 else "ERROR", exact_dups, 0))

        key_cols = ["patient_id", "seed", "candidate"]
        semantic_dups = int(pred.duplicated(key_cols, keep=False).sum())
        rows.append(qc_row(key, "one held-out prediction per patient-seed-candidate",
                           "PASS" if semantic_dups == 0 else "ERROR",
                           semantic_dups, 0,
                           "Duplicate count is the number of implicated rows."))

        counts = pred.groupby(["patient_id", "candidate"])["seed"].nunique()
        bad_counts = int((counts != expected_seeds).sum())
        rows.append(qc_row(key, "ten OOS repetitions per patient-candidate",
                           "PASS" if bad_counts == 0 else "ERROR",
                           bad_counts, 0,
                           f"Expected {expected_seeds} unique seeds."))

        fold_per_patient_seed = pred.groupby(["patient_id", "seed"])["fold"].nunique()
        bad_fold_assignment = int((fold_per_patient_seed != 1).sum())
        rows.append(qc_row(key, "one held-out fold per patient-seed",
                           "PASS" if bad_fold_assignment == 0 else "ERROR",
                           bad_fold_assignment, 0))

        label_nunique = pred.groupby("patient_id")["y_true"].nunique(dropna=False)
        inconsistent_labels = int((label_nunique != 1).sum())
        binary_labels = set(pd.to_numeric(pred["y_true"], errors="coerce").dropna().unique()) <= {0, 1}
        rows.append(qc_row(key, "within-run y_true consistency",
                           "PASS" if inconsistent_labels == 0 and binary_labels else "ERROR",
                           inconsistent_labels, 0,
                           f"binary={binary_labels}"))

        numeric_cols = [c for c in pred.select_dtypes(include=[np.number]).columns]
        inf_count = int(np.isinf(pred[numeric_cols].to_numpy(dtype=float)).sum()) if numeric_cols else 0
        nan_required = int(pred[list(REQUIRED_PRED_COLUMNS)].isna().sum().sum())
        rows.append(qc_row(key, "NaN/Inf in required prediction data",
                           "PASS" if inf_count == 0 and nan_required == 0 else "ERROR",
                           inf_count + nan_required, 0,
                           f"NaN={nan_required}; Inf={inf_count}"))

        out_of_range = 0
        details = []
        for col in PROBABILITY_COLUMNS:
            vals = pd.to_numeric(pred[col], errors="coerce")
            n = int(((vals < 0) | (vals > 1)).sum())
            out_of_range += n
            details.append(f"{col}={n}")
        rows.append(qc_row(key, "probabilities within [0,1]",
                           "PASS" if out_of_range == 0 else "ERROR",
                           out_of_range, 0, "; ".join(details)))

        threshold_vals = pd.to_numeric(pred["threshold"], errors="coerce")
        invalid_threshold = int(((threshold_vals < 0) | (threshold_vals > 1)).sum())
        rows.append(qc_row(key, "thresholds within [0,1]",
                           "PASS" if invalid_threshold == 0 else "ERROR",
                           invalid_threshold, 0))

        # Cross-condition consistency uses one label per patient.
        labels = pred.groupby("patient_id")["y_true"].first().sort_index()
        pset = set(labels.index)
        if key.task not in reference_sets:
            reference_sets[key.task] = pset
            reference_labels[key.task] = labels
        else:
            set_diff = reference_sets[key.task].symmetric_difference(pset)
            common = reference_labels[key.task].index.intersection(labels.index)
            mismatches = int((reference_labels[key.task].loc[common].astype(float) !=
                              labels.loc[common].astype(float)).sum())
            rows.append(qc_row(key, "patient_id set across modality conditions",
                               "PASS" if not set_diff else "ERROR",
                               len(set_diff), 0))
            rows.append(qc_row(key, "y_true across modality conditions",
                               "PASS" if mismatches == 0 else "ERROR",
                               mismatches, 0))

        # Fold metrics should be 10 seeds x 5 folds x 4 candidates = 200 rows.
        fm = run.fold_metrics
        if fm is None:
            rows.append(qc_row(key, "fold metrics file", "WARNING", "missing", "present"))
        else:
            expected_rows = expected_seeds * expected_folds * len(EXPECTED_CANDIDATES)
            rows.append(qc_row(key, "fold metric rows",
                               "PASS" if len(fm) == expected_rows else "ERROR",
                               len(fm), expected_rows))

    return pd.DataFrame(rows)


def aggregate_patient_predictions(run: RunData) -> pd.DataFrame:
    if run.predictions is None:
        raise ValueError(f"Missing predictions for {run.key.slug}")
    pred = run.predictions.copy()
    pred["patient_id"] = pred["patient_id"].astype(str)
    agg = (pred.groupby(["patient_id", "candidate"], as_index=False)
           .agg(y_true=("y_true", "first"),
                y_prob_mean=("y_prob", "mean"),
                meta_prob_mean=("meta_prob", "mean"),
                radiomics_prob_mean=("radiomics_prob", "mean"),
                n_repetitions=("seed", "nunique")))
    agg.insert(0, "condition", run.key.condition)
    agg.insert(0, "task", run.key.task)
    return agg


def patient_wide(aggregated: pd.DataFrame, task: str, condition: str) -> pd.DataFrame:
    sub = aggregated[(aggregated["task"] == task) &
                     (aggregated["condition"] == condition)].copy()
    labels = sub.groupby("patient_id")["y_true"].first()
    wide = sub.pivot(index="patient_id", columns="candidate", values="y_prob_mean")
    wide.insert(0, "y_true", labels)
    return wide.sort_index()


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(p), np.nan)
    valid = np.where(np.isfinite(p))[0]
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def stratified_paired_bootstrap_auc(y_true: np.ndarray, prob_a: np.ndarray,
                                    prob_b: np.ndarray, n_boot: int,
                                    rng: np.random.Generator) -> dict:
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    if len(np.unique(y)) != 2:
        raise ValueError("Both outcome classes are required for AUC.")
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    auc_a = roc_auc_score(y, a)
    auc_b = roc_auc_score(y, b)
    delta = auc_a - auc_b
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = np.concatenate([
            rng.choice(idx0, size=len(idx0), replace=True),
            rng.choice(idx1, size=len(idx1), replace=True),
        ])
        boot[i] = roc_auc_score(y[sample], a[sample]) - roc_auc_score(y[sample], b[sample])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    p_lo = (np.sum(boot <= 0) + 1) / (n_boot + 1)
    p_hi = (np.sum(boot >= 0) + 1) / (n_boot + 1)
    p_raw = min(1.0, 2.0 * min(p_lo, p_hi))
    return {
        "auc_model": auc_a,
        "auc_reference": auc_b,
        "delta_auc": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_raw": p_raw,
        "bootstrap_valid": n_boot,
    }


def make_table3(aggregated: pd.DataFrame, n_boot: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    comparisons = [
        ("weighted_oof_selected", "meta_only_prob"),
        ("weighted_oof_selected", "radiomics_only_prob"),
        ("logit_stacking_oof_selected", "meta_only_prob"),
        ("logit_stacking_oof_selected", "radiomics_only_prob"),
    ]
    for task in TASKS:
        wide = patient_wide(aggregated, task, "all-modality")
        for model, reference in comparisons:
            result = stratified_paired_bootstrap_auc(
                wide["y_true"].to_numpy(), wide[model].to_numpy(),
                wide[reference].to_numpy(), n_boot, rng)
            rows.append({
                "task": task,
                "fusion_model": CANDIDATE_LABELS[model],
                "fusion_candidate": model,
                "reference_model": CANDIDATE_LABELS[reference],
                "reference_candidate": reference,
                **result,
            })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_raw"])
    out["holm_significant_0.05"] = out["p_holm"] < 0.05
    return out


def calibration_intercept_slope(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float, bool]:
    eps = np.finfo(float).eps
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    x = np.log(p / (1 - p))

    def nll(beta: np.ndarray) -> float:
        eta = beta[0] + beta[1] * x
        q = expit(eta)
        q = np.clip(q, eps, 1 - eps)
        return float(-np.sum(y * np.log(q) + (1 - y) * np.log(1 - q)))

    fit = minimize(nll, x0=np.array([0.0, 1.0]), method="BFGS")
    return float(fit.x[0]), float(fit.x[1]), bool(fit.success)


def calibration_analysis(aggregated: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    curve_rows = []
    for task in TASKS:
        for condition in CONDITIONS:
            wide = patient_wide(aggregated, task, condition)
            y = wide["y_true"].to_numpy(dtype=int)
            for cand in EXPECTED_CANDIDATES:
                p = wide[cand].to_numpy(dtype=float)
                intercept, slope, converged = calibration_intercept_slope(y, p)
                metric_rows.append({
                    "task": task,
                    "condition": condition,
                    "candidate": cand,
                    "model": CANDIDATE_LABELS[cand],
                    "n": len(y),
                    "prevalence": y.mean(),
                    "auc_patient_level": roc_auc_score(y, p),
                    "average_precision_patient_level": average_precision_score(y, p),
                    "brier_score": brier_score_loss(y, p),
                    "calibration_intercept": intercept,
                    "calibration_slope": slope,
                    "calibration_fit_converged": converged,
                })
                frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
                # Reconstruct the exact quantile-bin assignment used by
                # sklearn.calibration.calibration_curve so that calibration bin
                # sizes can be audited. Duplicate percentile edges may reduce
                # the realized number of bins; this is retained and reported.
                quantile_edges = np.percentile(p, np.linspace(0.0, 100.0, 11))
                bin_ids = np.searchsorted(quantile_edges[1:-1], p)
                realized_bins = np.unique(bin_ids)
                if len(realized_bins) != len(mean_pred):
                    raise RuntimeError(
                        f"Calibration-bin reconstruction mismatch for {task}, "
                        f"{condition}, {cand}: reconstructed={len(realized_bins)}, "
                        f"calibration_curve={len(mean_pred)}")
                for i, (bin_id, mp, fp) in enumerate(
                        zip(realized_bins, mean_pred, frac_pos), start=1):
                    in_bin = bin_ids == bin_id
                    curve_rows.append({
                        "task": task, "condition": condition, "candidate": cand,
                        "model": CANDIDATE_LABELS[cand], "bin": i,
                        "n_in_bin": int(in_bin.sum()),
                        "bin_probability_min": float(np.min(p[in_bin])),
                        "bin_probability_max": float(np.max(p[in_bin])),
                        "mean_predicted_probability": mp,
                        "observed_fraction_positive": fp,
                    })
    return pd.DataFrame(metric_rows), pd.DataFrame(curve_rows)



def make_calibration_bin_size_qc(calibration_curves: pd.DataFrame,
                                 all_modality_only: bool = True) -> pd.DataFrame:
    """Summarize realized quantile-bin sizes used in calibration plots.

    The detailed per-bin counts remain in 04_calibration_curve_points.csv.
    This compact QC table reports the minimum and maximum bin size for each
    model, together with the realized number of bins and total patients.
    """
    curves = calibration_curves.copy()
    if all_modality_only:
        curves = curves[curves["condition"] == "all-modality"].copy()
    required = {"task", "condition", "candidate", "model", "bin", "n_in_bin"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"Calibration curve data are missing QC columns: {sorted(missing)}")
    out = (curves.groupby(["task", "condition", "candidate", "model"], dropna=False)
           .agg(number_of_bins=("bin", "nunique"),
                total_patients=("n_in_bin", "sum"),
                minimum_bin_size=("n_in_bin", "min"),
                maximum_bin_size=("n_in_bin", "max"),
                mean_bin_size=("n_in_bin", "mean"),
                bin_size_sd=("n_in_bin", lambda x: float(np.std(x, ddof=0))))
           .reset_index())
    out["bin_size_range"] = (out["minimum_bin_size"].astype(int).astype(str)
                             + "-" + out["maximum_bin_size"].astype(int).astype(str))
    out["maximum_minus_minimum"] = out["maximum_bin_size"] - out["minimum_bin_size"]
    out["ten_bins_realized"] = out["number_of_bins"].eq(10)
    return out

def make_modality_comparison(aggregated: pd.DataFrame, n_boot: int,
                             random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(random_state + 1)
    rows = []
    # Weighted averaging is the primary modality-comparison model specified in the plan.
    cand = "weighted_oof_selected"
    for task in TASKS:
        all_wide = patient_wide(aggregated, task, "all-modality")
        for reference_condition in ("mri-only", "pet-only"):
            ref_wide = patient_wide(aggregated, task, reference_condition)
            merged = all_wide[["y_true", cand]].join(
                ref_wide[["y_true", cand]], how="inner", lsuffix="_all", rsuffix="_ref"
            )
            if not np.array_equal(merged["y_true_all"].to_numpy(),
                                  merged["y_true_ref"].to_numpy()):
                raise ValueError(f"Label mismatch: {task}, all vs {reference_condition}")
            result = stratified_paired_bootstrap_auc(
                merged["y_true_all"].to_numpy(), merged[f"{cand}_all"].to_numpy(),
                merged[f"{cand}_ref"].to_numpy(), n_boot, rng)
            rows.append({
                "task": task,
                "model": CANDIDATE_LABELS[cand],
                "candidate": cand,
                "comparison": f"all-modality minus {reference_condition}",
                "model_condition": "all-modality",
                "reference_condition": reference_condition,
                **result,
            })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_raw"])
    out["holm_significant_0.05"] = out["p_holm"] < 0.05
    return out


def combine_summaries(runs: Mapping[RunKey, RunData]) -> pd.DataFrame:
    rows = []
    for key, run in runs.items():
        if run.summary is None:
            continue
        df = run.summary.copy()
        df.insert(0, "condition", key.condition)
        df.insert(0, "task", key.task)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def metric_display(row: pd.Series, metric: str) -> str:
    return f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_sd']:.3f}"


def add_bold_markers(df: pd.DataFrame, group_cols: Sequence[str], metrics: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for metric in metrics:
        col = f"{metric}_mean"
        maxes = out.groupby(list(group_cols))[col].transform("max")
        out[f"{metric}_bold"] = np.isclose(out[col], maxes, rtol=0, atol=1e-12)
        out[f"{metric}_display"] = [
            (f"**{metric_display(row, metric)}**" if bold else metric_display(row, metric))
            for (_, row), bold in zip(out.iterrows(), out[f"{metric}_bold"])
        ]
    return out


def generate_table2(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary[summary["condition"] == "all-modality"].copy()
    table["model"] = table["candidate"].map(CANDIDATE_LABELS)
    table["order"] = table["candidate"].map(CANDIDATE_ORDER)
    table = table.sort_values(["task", "order"])
    table = add_bold_markers(table, ["task"], SUMMARY_METRICS)
    keep = ["task", "candidate", "model"]
    for m in SUMMARY_METRICS:
        keep += [f"{m}_mean", f"{m}_sd", f"{m}_bold", f"{m}_display"]
    return table[keep]


def generate_table4(summary: pd.DataFrame) -> pd.DataFrame:
    included = {"weighted_oof_selected", "logit_stacking_oof_selected", "radiomics_only_prob"}
    table = summary[summary["candidate"].isin(included)].copy()
    table["model"] = table["candidate"].map(CANDIDATE_LABELS)
    table["condition_order"] = table["condition"].map({c: i for i, c in enumerate(CONDITIONS)})
    table["model_order"] = table["candidate"].map(CANDIDATE_ORDER)
    table = table.sort_values(["task", "condition_order", "model_order"])
    metrics = ("auc", "ap", "bal_acc", "sensitivity", "specificity")
    table = add_bold_markers(table, ["task"], metrics)
    keep = ["task", "condition", "candidate", "model"]
    for m in metrics:
        keep += [f"{m}_mean", f"{m}_sd", f"{m}_bold", f"{m}_display"]
    return table[keep]


def parse_feature(feature: str) -> Tuple[str, str, str]:
    modality = "Unknown"
    match = re.search(r"ROI\d+_(FLAIR|T1|T2|MET|FDG|CHO)_", feature, flags=re.I)
    if match:
        modality = match.group(1).upper()
    feature_class = "other"
    for fc in ("firstorder", "glcm", "gldm", "glrlm", "glszm", "ngtdm", "shape"):
        if f"_{fc}_" in feature.lower():
            feature_class = fc
            break
    group = "MRI" if modality in MRI_MODALITIES else "PET" if modality in PET_MODALITIES else "Unknown"
    return modality, group, feature_class


def feature_diagnostics(runs: Mapping[RunKey, RunData]) -> Tuple[pd.DataFrame, pd.DataFrame,
                                                                         pd.DataFrame, pd.DataFrame,
                                                                         pd.DataFrame]:
    feature_rows = []
    selector_rows = []
    config_rows = []
    tuning_rows = []
    fusion_rows = []
    for task in TASKS:
        run = runs[RunKey(task, "all-modality")]
        sf = run.selected_features
        if sf is not None:
            if "stage" not in sf or "selected_features" not in sf:
                warnings.warn(f"Selected-feature columns missing for {task}")
            else:
                final = sf[sf["stage"].astype(str) == "outer_test_model"].copy()
                # One final outer-test model should exist for each seed-fold pair.
                for _, row in final.iterrows():
                    features = [x for x in str(row["selected_features"]).split("|")
                                if x and x.lower() != "nan"]
                    for feat in features:
                        modality, group, feature_class = parse_feature(feat)
                        feature_rows.append({
                            "task": task, "seed": row.get("seed"), "fold": row.get("fold"),
                            "feature": feat, "modality": modality,
                            "modality_group": group, "feature_class": feature_class,
                        })
                if {"selector_method", "seed", "fold"}.issubset(final.columns):
                    sr = (final.groupby("selector_method", dropna=False).size()
                          .reset_index(name="n_selected_outer_models"))
                    sr.insert(0, "task", task)
                    selector_rows.extend(sr.to_dict("records"))
                if {"feature_config_name", "seed", "fold"}.issubset(final.columns):
                    cr = (final.groupby("feature_config_name", dropna=False).size()
                          .reset_index(name="n_selected_outer_models"))
                    cr.insert(0, "task", task)
                    config_rows.extend(cr.to_dict("records"))

        it = run.inner_tuning
        if it is not None:
            row = {"task": task, "n_rows": len(it)}
            for col in ("valid_fit", "fallback_used", "refit_success"):
                if col in it.columns:
                    vals = it[col].astype(str).str.lower()
                    row[f"{col}_false_or_zero"] = int(vals.isin(["false", "0", "0.0"]).sum())
                    row[f"{col}_true_or_one"] = int(vals.isin(["true", "1", "1.0"]).sum())
            if "error_type" in it.columns:
                row["nonempty_error_type"] = int(it["error_type"].notna().sum())
            tuning_rows.append(row)

        fr = run.fusion_records
        if fr is not None and {"family", "candidate", "selected"}.issubset(fr.columns):
            selected = fr[pd.to_numeric(fr["selected"], errors="coerce") == 1]
            freq = selected.groupby(["family", "candidate"]).size().reset_index(name="n_selected")
            freq.insert(0, "task", task)
            fusion_rows.extend(freq.to_dict("records"))

    features = pd.DataFrame(feature_rows)
    if not features.empty:
        freq = (features.groupby(["task", "feature", "modality", "modality_group", "feature_class"])
                .size().reset_index(name="selection_frequency")
                .sort_values(["task", "selection_frequency", "feature"], ascending=[True, False, True]))
        freq["rank_within_task"] = freq.groupby("task")["selection_frequency"].rank(
            method="first", ascending=False).astype(int)
    else:
        freq = pd.DataFrame()
    return (freq, pd.DataFrame(selector_rows), pd.DataFrame(config_rows),
            pd.DataFrame(tuning_rows), pd.DataFrame(fusion_rows))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def plot_roc(aggregated: pd.DataFrame, outdir: Path) -> None:
    for task in TASKS:
        wide = patient_wide(aggregated, task, "all-modality")
        fig, ax = plt.subplots(figsize=(7, 6))
        y = wide["y_true"].to_numpy()
        for cand in EXPECTED_CANDIDATES:
            p = wide[cand].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            auc = roc_auc_score(y, p)
            ax.plot(fpr, tpr, label=f"{CANDIDATE_LABELS[cand]} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{task}: patient-level ROC curves, all modalities")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"figure_roc_{task}.png", dpi=300)
        plt.close(fig)


def plot_calibration(curves: pd.DataFrame, outdir: Path) -> None:
    for task in TASKS:
        sub = curves[(curves["task"] == task) & (curves["condition"] == "all-modality")]
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1,
                label="Perfect calibration")
        for cand in EXPECTED_CANDIDATES:
            x = sub[sub["candidate"] == cand]
            # Convert Series to NumPy arrays for pandas/matplotlib compatibility.
            x_values = x["mean_predicted_probability"].to_numpy(dtype=float)
            y_values = x["observed_fraction_positive"].to_numpy(dtype=float)
            ax.plot(x_values, y_values, marker="o",
                    label=CANDIDATE_LABELS[cand])
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction positive")
        ax.set_title(f"{task}: calibration, all modalities")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"figure_calibration_{task}.png", dpi=300)
        plt.close(fig)


def plot_feature_frequency(freq: pd.DataFrame, outdir: Path) -> None:
    if freq.empty:
        return
    for task in TASKS:
        top = freq[(freq["task"] == task) & (freq["rank_within_task"] <= 20)].copy()
        top = top.sort_values("selection_frequency", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(top["feature"], top["selection_frequency"])
        ax.set_xlabel("Number of outer-fold models retaining the feature")
        ax.set_title(f"{task}: top 20 retained radiomic features")
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f"figure_feature_frequency_{task}.png", dpi=300)
        plt.close(fig)


def process_binning_diagnostics(root: Path, outdir: Path) -> Optional[pd.DataFrame]:
    path = root / "binning_diagnostics.csv"
    if not path.exists():
        return None
    df = read_csv(path)
    required = {"task", "modality", "image_type", "bin_width", "occupied_bins"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"binning_diagnostics.csv is missing columns: {sorted(missing)}")
    group_cols = ["task", "modality", "image_type", "bin_width"]
    summary = (df.groupby(group_cols)["occupied_bins"]
               .agg(n="size", median="median", q1=lambda x: x.quantile(.25),
                    q3=lambda x: x.quantile(.75), minimum="min", maximum="max")
               .reset_index())
    summary["n_occupied_bins_le_2"] = (df.assign(le2=df["occupied_bins"] <= 2)
                                        .groupby(group_cols)["le2"].sum().to_numpy())
    summary["proportion_occupied_bins_le_2"] = summary["n_occupied_bins_le_2"] / summary["n"]
    for optional in ("extraction_failed", "constant_feature_count", "near_constant_feature_count"):
        if optional in df.columns:
            values = pd.to_numeric(df[optional], errors="coerce").fillna(0)
            temp = df[group_cols].copy()
            temp[optional] = values
            agg = temp.groupby(group_cols)[optional].sum().reset_index()
            summary = summary.merge(agg, on=group_cols, how="left")
    save_csv(summary, outdir / "supplementary_table_binning_diagnostics.csv")

    for task in sorted(df["task"].astype(str).unique()):
        sub = df[df["task"].astype(str) == task]
        labels, data = [], []
        for keys, group in sub.groupby(["modality", "image_type", "bin_width"], sort=True):
            labels.append(f"{keys[0]}\n{keys[1]}\nBW={keys[2]}")
            data.append(group["occupied_bins"].to_numpy())
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * .55), 6))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel("Occupied gray levels")
        ax.set_title(f"{task}: discretization diagnostics")
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        fig.tight_layout()
        fig.savefig(outdir / f"figure_binning_diagnostics_{task}.png", dpi=300)
        plt.close(fig)
    return summary


def write_caption_notes(outdir: Path) -> None:
    text = """Table 2 caption note:
Values are mean ± standard deviation across the 50 held-out outer test folds. Bold values indicate the highest mean within each task and metric and do not indicate statistical significance.

Table 3 caption note:
Out-of-sample probabilities were averaged across ten repetitions for each patient. Differences in AUC and percentile 95% confidence intervals were estimated using 2,000 stratified paired bootstrap replicates with identical resampling applied to both models. Raw two-sided bootstrap p values were adjusted across the eight exploratory comparisons using the Holm method.

Table 4 caption note:
Values are mean ± standard deviation across the 50 held-out outer test folds. Bold values indicate the highest mean within each task and metric and do not indicate statistical significance. Modality-group comparisons are secondary and exploratory; numerical differences do not by themselves establish a statistically significant incremental contribution of a modality group.
"""
    (outdir / "table_caption_notes.txt").write_text(text, encoding="utf-8")


def write_missing_information(runs: Mapping[RunKey, RunData], root: Path, outdir: Path) -> None:
    lines = []
    for key, run in runs.items():
        if run.predictions is None:
            lines.append(f"MISSING: {key.slug}: all_predictions_probability_fusion*.csv")
        for name in ("fold_metrics", "summary", "selected_features", "inner_tuning", "fusion_records"):
            if run.files.get(name) is None:
                lines.append(f"OPTIONAL/REQUIRED FOR SOME OUTPUTS: {key.slug}: {FILE_PATTERNS[name]}")
    if not (root / "binning_diagnostics.csv").exists():
        lines.append("MISSING FOR BINNING FIGURE ONLY: binning_diagnostics.csv")
    if not lines:
        lines.append("No missing files detected for the configured analyses.")
    (outdir / "missing_information.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")




def signed3(x: float) -> str:
    return f"{x:+.3f}"


def p_display(x: float) -> str:
    if not np.isfinite(x):
        return ""
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def make_table3_manuscript(table3: pd.DataFrame) -> pd.DataFrame:
    """Compact main-text Table 3; complete raw results remain in Table S2."""
    out = table3.copy()
    out["AUC, fusion"] = out["auc_model"].map(lambda x: f"{x:.3f}")
    out["AUC, reference"] = out["auc_reference"].map(lambda x: f"{x:.3f}")
    out["Delta AUC (95% CI)"] = out.apply(
        lambda r: f"{signed3(r.delta_auc)} ({r.ci_low:.3f} to {r.ci_high:.3f})", axis=1)
    out["Holm-adjusted p"] = out["p_holm"].map(p_display)
    out["Adjusted interpretation"] = np.where(
        out["p_holm"] < 0.05,
        "Higher AUC after Holm correction",
        "Not statistically significant after Holm correction")
    return out[["task", "fusion_model", "reference_model", "AUC, fusion",
                "AUC, reference", "Delta AUC (95% CI)",
                "Holm-adjusted p", "Adjusted interpretation"]]


def make_table2_manuscript(summary: pd.DataFrame) -> pd.DataFrame:
    raw = generate_table2(summary)
    return raw[["task", "model", "auc_display", "ap_display", "acc_display",
                "bal_acc_display", "sensitivity_display", "specificity_display"]].rename(columns={
        "task": "Task", "model": "Model", "auc_display": "AUC",
        "ap_display": "AP", "acc_display": "Accuracy",
        "bal_acc_display": "Balanced accuracy",
        "sensitivity_display": "Sensitivity", "specificity_display": "Specificity"})


def make_table4_manuscript(summary: pd.DataFrame) -> pd.DataFrame:
    raw = generate_table4(summary)
    return raw[["task", "condition", "model", "auc_display", "ap_display",
                "bal_acc_display", "sensitivity_display", "specificity_display"]].rename(columns={
        "task": "Task", "condition": "Radiomics modality condition",
        "model": "Model", "auc_display": "AUC", "ap_display": "AP",
        "bal_acc_display": "Balanced accuracy",
        "sensitivity_display": "Sensitivity", "specificity_display": "Specificity"})


def make_supplementary_table_s2(modality: pd.DataFrame) -> pd.DataFrame:
    """Return the four weighted-averaging modality comparisons reported in Table S2."""
    out = modality.copy()
    out["multiplicity_family"] = "Four modality-group comparisons"
    out["delta_auc_display"] = out.apply(
        lambda r: f"{signed3(r.delta_auc)} ({r.ci_low:.3f} to {r.ci_high:.3f})", axis=1)
    out["p_raw_display"] = out["p_raw"].map(p_display)
    out["p_holm_display"] = out["p_holm"].map(p_display)
    return out


def make_exploratory_complete_fusion_comparisons(table3: pd.DataFrame) -> pd.DataFrame:
    """Return all eight exploratory fusion-versus-base-model comparisons."""
    out = table3.copy()
    out["multiplicity_family"] = "Eight fusion-versus-base comparisons"
    out["comparison"] = out["fusion_model"] + " minus " + out["reference_model"]
    out["delta_auc_display"] = out.apply(
        lambda r: f"{signed3(r.delta_auc)} ({r.ci_low:.3f} to {r.ci_high:.3f})", axis=1)
    out["p_raw_display"] = out["p_raw"].map(p_display)
    out["p_holm_display"] = out["p_holm"].map(p_display)
    return out

def make_supplementary_table_s3(calibration_metrics: pd.DataFrame) -> pd.DataFrame:
    """Main calibration evidence: all-modality, four families, two tasks."""
    out = calibration_metrics[calibration_metrics["condition"] == "all-modality"].copy()
    out["model_order"] = out["candidate"].map(CANDIDATE_ORDER)
    out = out.sort_values(["task", "model_order"])
    keep = ["task", "model", "n", "prevalence", "auc_patient_level",
            "average_precision_patient_level", "brier_score",
            "calibration_intercept", "calibration_slope",
            "calibration_fit_converged"]
    return out[keep]


def _bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    """Normalize bool-like CSV values without treating NaN as True."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    text = df[column].astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "1.0", "yes", "y"})


def _false_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Identify explicit false values; missing values are not classified as false."""
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    text = df[column].astype(str).str.strip().str.lower()
    return text.isin({"false", "0", "0.0", "no", "n"})


def _nonempty_text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    text = df[column].fillna("").astype(str).str.strip()
    return text.ne("") & text.str.lower().ne("nan")


def _display_task(task: str) -> str:
    return "1p/19q" if str(task) == "1p19q" else str(task)


def _display_condition(condition: str) -> str:
    return {
        "all-modality": "All six modalities",
        "mri-only": "MRI only",
        "pet-only": "PET only",
    }.get(str(condition), str(condition))


def _candidate_context_summary(it: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize five-fold candidate contexts and tuning contexts.

    A candidate context is one candidate evaluated within one radiomics fit.
    Candidate eligibility requires five distinct inner folds and five valid fits.
    """
    tuning = it[it["stage"].astype(str).str.endswith("_inner_tuning")].copy()
    if tuning.empty:
        return pd.DataFrame(), pd.DataFrame()
    tuning["valid_fit_bool"] = _bool_series(tuning, "valid_fit")
    method = tuning.get("selector_method", pd.Series("", index=tuning.index)).fillna("").astype(str).str.lower()
    nonzero = pd.to_numeric(tuning.get("n_nonzero_coefficients", np.nan), errors="coerce")
    error_text = tuning.get("error_type", pd.Series("", index=tuning.index)).fillna("").astype(str).str.strip().str.lower()
    zero_from_count = method.eq("lasso") & nonzero.eq(0)
    zero_from_label = error_text.str.contains(
        r"zero[ _-]?feature|no[ _-]?nonzero|no[ _-]?selected[ _-]?feature",
        regex=True, na=False)
    tuning["zero_feature_failure"] = zero_from_count | zero_from_label
    tuning["other_fit_error"] = _nonempty_text(tuning, "error_type") & ~tuning["zero_feature_failure"]

    # The OOF-specific inner-fold column identifies the radiomics fit that
    # generated one outer-training OOF fold. It is -1/NaN for the final
    # outer-test model fit. Retain it to avoid merging distinct fit contexts.
    context_cols = [c for c in [
        "stage", "seed", "fold", "outer_train_oof_inner_fold"
    ] if c in tuning.columns]
    candidate_cols = [c for c in [
        "candidate_id", "candidate_name", "feature_config_name",
        "selector_name", "selector_method", "lasso_c", "max_features"
    ] if c in tuning.columns]
    group_cols = context_cols + candidate_cols
    if not group_cols or "inner_fold" not in tuning.columns:
        raise ValueError("Inner-tuning records do not contain the columns needed to define candidate contexts.")

    candidate_summary = (tuning.groupby(group_cols, dropna=False)
        .agg(n_inner_folds=("inner_fold", "nunique"),
             n_valid_fits=("valid_fit_bool", "sum"),
             n_zero_feature_fits=("zero_feature_failure", "sum"),
             n_other_errors=("other_fit_error", "sum"))
        .reset_index())
    candidate_summary["eligible"] = (
        candidate_summary["n_inner_folds"].eq(5)
        & candidate_summary["n_valid_fits"].eq(5)
    )
    candidate_summary["partially_invalid"] = (
        candidate_summary["n_valid_fits"].gt(0)
        & candidate_summary["n_valid_fits"].lt(candidate_summary["n_inner_folds"])
    )

    # Tuning context excludes candidate identity. This confirms whether every
    # radiomics fit retained at least one eligible candidate.
    context_summary = (candidate_summary.groupby(context_cols, dropna=False)
        .agg(n_candidates=("eligible", "size"),
             n_eligible_candidates=("eligible", "sum"))
        .reset_index())
    context_summary["no_eligible_candidate"] = context_summary["n_eligible_candidates"].eq(0)
    return candidate_summary, context_summary


def make_supplementary_table_s4(runs: Mapping[RunKey, RunData],
                                selector_freq: pd.DataFrame,
                                config_freq: pd.DataFrame,
                                fusion_freq: pd.DataFrame,
                                outdir: Optional[Path] = None) -> pd.DataFrame:
    """Create publication-oriented Table S4 and detailed audit files.

    Panel A is one row per task-condition. Panel B contains final all-six-
    modality selection frequencies as a single ``Selected, n/50`` field.
    The function never infers an error category from a missing error_type:
    zero-feature LASSO fits are identified from n_nonzero_coefficients == 0
    when available, otherwise from an explicit error_type label.
    """
    panel_a_rows = []
    error_detail_parts = []
    candidate_detail_parts = []
    context_detail_parts = []
    stage_count_parts = []

    for key, run in runs.items():
        it = run.inner_tuning
        base = {"Task": _display_task(key.task),
                "Condition": _display_condition(key.condition)}
        if it is None:
            panel_a_rows.append({**base,
                "Inner-fold candidate evaluations": np.nan,
                "Invalid inner-fold fits": np.nan,
                "Zero-feature LASSO fits": np.nan,
                "Other fitting errors": np.nan,
                "Final refit failures": np.nan,
                "Fallback fits": np.nan})
            continue
        it = it.copy()
        if "stage" not in it.columns:
            raise ValueError(f"stage column is missing for {key.slug}")
        stage = it["stage"].fillna("").astype(str)
        inner_mask = stage.str.endswith("_inner_tuning")
        refit_mask = stage.str.endswith("_refit_attempt")
        valid = _bool_series(it, "valid_fit")
        explicit_invalid = _false_series(it, "valid_fit")
        method = it.get("selector_method", pd.Series("", index=it.index)).fillna("").astype(str).str.lower()
        nonzero = pd.to_numeric(it.get("n_nonzero_coefficients", np.nan), errors="coerce")
        error_text = it.get("error_type", pd.Series("", index=it.index)).fillna("").astype(str).str.strip().str.lower()
        zero_feature = inner_mask & method.eq("lasso") & (
            nonzero.eq(0) | error_text.str.contains(
                r"zero[ _-]?feature|no[ _-]?nonzero|no[ _-]?selected[ _-]?feature",
                regex=True, na=False))
        has_error = _nonempty_text(it, "error_type")
        invalid_inner = inner_mask & (~valid | explicit_invalid | has_error | zero_feature)
        other_error = inner_mask & has_error & ~zero_feature
        refit_failure = refit_mask & _false_series(it, "refit_success")
        fallback_fit = refit_mask & _bool_series(it, "fallback_used")

        cand, contexts = _candidate_context_summary(it)
        if not cand.empty:
            cand.insert(0, "condition", key.condition)
            cand.insert(0, "task", key.task)
            candidate_detail_parts.append(cand)
        if not contexts.empty:
            contexts.insert(0, "condition", key.condition)
            contexts.insert(0, "task", key.task)
            context_detail_parts.append(contexts)

        panel_a_rows.append({**base,
            "Inner-fold candidate evaluations": int(inner_mask.sum()),
            "Invalid inner-fold fits": int(invalid_inner.sum()),
            "Zero-feature LASSO fits": int(zero_feature.sum()),
            "Other fitting errors": int(other_error.sum()),
            "Final refit failures": int(refit_failure.sum()),
            "Fallback fits": int(fallback_fit.sum())})

        detail_mask = invalid_inner | refit_failure | fallback_fit
        if detail_mask.any():
            keep = [c for c in [
                "stage", "seed", "fold", "outer_train_oof_inner_fold",
                "inner_fold", "candidate_id", "candidate_name",
                "feature_config_name", "selector_name", "selector_method",
                "lasso_c", "max_features", "valid_fit",
                "n_nonzero_coefficients", "n_selected", "error_type",
                "error_message", "fallback_used", "refit_candidate_rank",
                "refit_success"
            ] if c in it.columns]
            detail = it.loc[detail_mask, keep].copy()
            detail.insert(0, "condition", key.condition)
            detail.insert(0, "task", key.task)
            detail["classified_zero_feature_lasso"] = zero_feature.loc[detail.index].to_numpy()
            detail["classified_other_fitting_error"] = other_error.loc[detail.index].to_numpy()
            error_detail_parts.append(detail)

        sc = stage.value_counts(dropna=False).rename_axis("stage").reset_index(name="observed_records")
        sc.insert(0, "condition", key.condition)
        sc.insert(0, "task", key.task)
        stage_count_parts.append(sc)

    panel_a = pd.DataFrame(panel_a_rows)

    panel_b_rows = []
    for _, r in selector_freq.iterrows():
        panel_b_rows.append({"Task": _display_task(r["task"]),
            "Selection component": "Selector method",
            "Candidate": str(r["selector_method"]),
            "Selected, n/50": f"{int(r['n_selected_outer_models'])}/50"})
    for _, r in config_freq.iterrows():
        panel_b_rows.append({"Task": _display_task(r["task"]),
            "Selection component": "Feature configuration",
            "Candidate": str(r["feature_config_name"]),
            "Selected, n/50": f"{int(r['n_selected_outer_models'])}/50"})
    for _, r in fusion_freq.iterrows():
        family = str(r["family"])
        component = "Weighted averaging" if "weight" in family.lower() else "Logit stacking"
        panel_b_rows.append({"Task": _display_task(r["task"]),
            "Selection component": component,
            "Candidate": str(r["candidate"]),
            "Selected, n/50": f"{int(r['n_selected'])}/50"})
    panel_b = pd.DataFrame(panel_b_rows)

    # Combined long-format sheet is convenient for the existing workbook writer;
    # publication-ready Panel A and Panel B are also saved separately.
    a = panel_a.copy(); a.insert(0, "Panel", "A. Candidate-selection and fitting diagnostics")
    b = panel_b.copy(); b.insert(0, "Panel", "B. Selected candidate frequencies")
    combined = pd.concat([a, b], ignore_index=True, sort=False)

    if outdir is not None:
        outdir = Path(outdir)
        save_csv(panel_a, outdir / "Table_S4_Panel_A_fitting_diagnostics.csv")
        save_csv(panel_b, outdir / "Table_S4_Panel_B_selected_candidate_frequencies.csv")
        save_csv(pd.concat(stage_count_parts, ignore_index=True) if stage_count_parts else pd.DataFrame(),
                 outdir / "Table_S4_stage_counts_QC.csv")
        save_csv(pd.concat(error_detail_parts, ignore_index=True) if error_detail_parts else pd.DataFrame(),
                 outdir / "Table_S4_error_records_detail.csv")
        save_csv(pd.concat(candidate_detail_parts, ignore_index=True) if candidate_detail_parts else pd.DataFrame(),
                 outdir / "Table_S4_candidate_eligibility_detail.csv")
        save_csv(pd.concat(context_detail_parts, ignore_index=True) if context_detail_parts else pd.DataFrame(),
                 outdir / "Table_S4_tuning_context_eligibility_QC.csv")
    return combined


def make_supplementary_table_s1(root: Path, outdir: Path) -> Optional[pd.DataFrame]:
    """Detailed discretization table. Does not fabricate records if input is absent."""
    path = root / "binning_diagnostics.csv"
    if not path.exists():
        return None
    df = read_csv(path)
    required = {"task", "modality", "image_type", "bin_width", "occupied_bins"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"binning_diagnostics.csv is missing columns: {sorted(missing)}")
    df = df.copy()
    df["occupied_bins"] = pd.to_numeric(df["occupied_bins"], errors="coerce")
    group_cols = ["task", "modality", "image_type", "bin_width"]
    grouped = df.groupby(group_cols, dropna=False)["occupied_bins"]
    out = grouped.agg(
        n_images="count", median_occupied_bins="median",
        q1_occupied_bins=lambda x: x.quantile(.25),
        q3_occupied_bins=lambda x: x.quantile(.75),
        min_occupied_bins="min", max_occupied_bins="max").reset_index()
    le2 = (df.assign(flag=df["occupied_bins"] <= 2)
           .groupby(group_cols, dropna=False)["flag"].sum().reset_index(name="n_images_le_2_bins"))
    out = out.merge(le2, on=group_cols, how="left")
    out["percent_images_le_2_bins"] = 100 * out["n_images_le_2_bins"] / out["n_images"]
    for col in ("extraction_failed", "constant_feature_count", "near_constant_feature_count"):
        if col in df.columns:
            temp = df[group_cols].copy()
            temp[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            counts = temp.groupby(group_cols, dropna=False)[col].sum().reset_index()
            out = out.merge(counts, on=group_cols, how="left")
        else:
            out[col] = np.nan
    out["occupied_bins_median_IQR"] = out.apply(
        lambda r: f"{r.median_occupied_bins:.1f} ({r.q1_occupied_bins:.1f}-{r.q3_occupied_bins:.1f})", axis=1)
    out["occupied_bins_range"] = out.apply(
        lambda r: f"{r.min_occupied_bins:.0f}-{r.max_occupied_bins:.0f}", axis=1)
    out["images_le_2_bins_n_percent"] = out.apply(
        lambda r: f"{int(r.n_images_le_2_bins)} ({r.percent_images_le_2_bins:.1f}%)", axis=1)
    save_csv(out, outdir / "Table_S1_binning_discretization_diagnostics.csv")
    return out


def compute_mean_outer_fold_roc(predictions: pd.DataFrame, candidate: str,
                                n_grid: int = N_ROC_GRID) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """Compute the mean ROC across valid seed-fold outer test sets."""
    sub = predictions[predictions["candidate"] == candidate].copy()
    if sub.empty:
        raise ValueError(f"Candidate not found for mean outer-fold ROC: {candidate}")
    fpr_grid = np.linspace(0.0, 1.0, n_grid)
    interpolated_tprs = []
    fold_aucs = []
    for (_, _), fold_df in sub.groupby(["seed", "fold"], sort=True):
        y = fold_df["y_true"].to_numpy(dtype=int)
        prob = fold_df["y_prob"].to_numpy(dtype=float)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, prob)
        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interpolated_tprs.append(interp_tpr)
        fold_aucs.append(roc_auc_score(y, prob))
    if not interpolated_tprs:
        raise ValueError(f"No valid outer-fold ROC curves for candidate: {candidate}")
    mean_tpr = np.mean(np.vstack(interpolated_tprs), axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    aucs = np.asarray(fold_aucs, dtype=float)
    return fpr_grid, mean_tpr, float(aucs.mean()), float(aucs.std(ddof=0)), len(aucs)


def plot_figure2_roc(runs: Mapping[RunKey, RunData], outdir: Path) -> None:
    """Mean ROC across 50 outer folds for all four prespecified model families."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.3), sharex=True, sharey=True)
    for ax, task, panel in zip(axes, TASKS, ("a", "b")):
        run = runs[RunKey(task, "all-modality")]
        if run.predictions is None:
            raise ValueError(f"Missing all-modality predictions for Figure 2: {task}")
        for cand in FIGURE_MODEL_ORDER:
            fpr, mean_tpr, auc_mean, auc_sd, n_folds = compute_mean_outer_fold_roc(
                run.predictions, cand)
            ax.plot(
                fpr, mean_tpr,
                color=MODEL_COLORS[cand], linestyle=MODEL_STYLES[cand],
                linewidth=MODEL_LINEWIDTHS[cand],
                label=(f"{CANDIDATE_LABELS[cand]} "
                       f"(AUC = {auc_mean:.3f} ± {auc_sd:.3f})"),
            )
            if n_folds != 50:
                warnings.warn(f"Figure 2 used {n_folds} valid outer folds for {task}, {cand}.")
        ax.plot([0, 1], [0, 1], color="0.45", linestyle=(0, (4, 2)),
                linewidth=0.8, label="_nolegend_", zorder=0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"({panel}) {'1p/19q codeletion prediction' if task == '1p19q' else 'IDH mutation prediction'}",
            fontsize=FIG_TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, frameon=True)
    axes[0].set_ylabel("True Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.88, wspace=0.10)
    fig.savefig(outdir / "Figure_2_mean_outer_fold_ROC.png", dpi=400,
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outdir / "Figure_2_mean_outer_fold_ROC.pdf",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_diagnostic_patient_level_roc(aggregated: pd.DataFrame, outdir: Path) -> None:
    """Patient-level ROC after averaging each patient's ten OOS probabilities."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.3), sharex=True, sharey=True)
    for ax, task, panel in zip(axes, TASKS, ("a", "b")):
        wide = patient_wide(aggregated, task, "all-modality")
        y = wide["y_true"].to_numpy(dtype=int)
        for cand in EXPECTED_CANDIDATES:
            prob = wide[cand].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, prob)
            auc_value = roc_auc_score(y, prob)
            ax.plot(
                np.clip(fpr, 0.0, 1.0), np.clip(tpr, 0.0, 1.0),
                color=MODEL_COLORS[cand], linestyle=MODEL_STYLES[cand],
                linewidth=MODEL_LINEWIDTHS[cand],
                label=f"{CANDIDATE_LABELS[cand]} (AUC = {auc_value:.3f})",
            )
        ax.plot([0, 1], [0, 1], color="0.45", linestyle=(0, (4, 2)),
                linewidth=0.8, label="_nolegend_", zorder=0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"({panel}) {'1p/19q codeletion prediction' if task == '1p19q' else 'IDH mutation prediction'}",
            fontsize=FIG_TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE, frameon=True)
    axes[0].set_ylabel("True Positive Rate", fontsize=AXIS_LABEL_FONTSIZE)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.88, wspace=0.10)
    fig.savefig(outdir / "diagnostic_patient_level_aggregated_ROC.png", dpi=400,
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(outdir / "diagnostic_patient_level_aggregated_ROC.pdf",
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

def short_feature_label(feature: str) -> str:
    label = re.sub(r"^ROI\d+_", "", str(feature))
    label = label.replace("LargeDependenceHighGrayLevelEmphasis", "LDHGLE")
    label = label.replace("SmallDependenceLowGrayLevelEmphasis", "SDLGLE")
    parts = label.split("_")
    if parts:
        parts[0] = MODALITY_DISPLAY.get(parts[0], parts[0])
    label = " | ".join(parts)
    label = re.sub(r"log-sigma-(\d+)-(\d+)-mm-3D", r"LoG sigma \1.\2", label)
    label = label.replace("firstorder", "first-order")
    return label

def plot_figure4_features(freq: pd.DataFrame, outdir: Path) -> None:
    if freq.empty:
        return
    modality_order = ("T1", "T2", "FLAIR", "MET", "FDG", "CHO")
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 13.5), dpi=400)
    for ax, task, panel in zip(axes, TASKS, ("a", "b")):
        top = freq[(freq["task"] == task) & (freq["rank_within_task"] <= 20)].copy()
        top = top.sort_values("selection_frequency", ascending=True)
        labels = [short_feature_label(x) for x in top["feature"]]
        colors = [MODALITY_COLORS.get(x, MODALITY_COLORS["Unknown"]) for x in top["modality"]]
        bars = ax.barh(labels, top["selection_frequency"], color=colors,
                       edgecolor="#444444", linewidth=0.5, height=0.76)
        ax.set_xlim(0, 53)
        ax.set_xlabel("Selection frequency across 50 final outer-training models",
                      fontsize=AXIS_LABEL_FONTSIZE, labelpad=7)
        ax.set_title(f"({panel}) {'1p/19q codeletion' if task == '1p19q' else 'IDH mutation'}: Top 20 retained radiomic features",
                     fontsize=FIG_TITLE_FONTSIZE, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=TICK_LABEL_FONTSIZE)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, v in zip(bars, top["selection_frequency"]):
            ax.text(v + 0.6, bar.get_y() + bar.get_height()/2,
                    f"{int(v)*2}% ({int(v)})", va="center", fontsize=10.5)
    present = [m for m in modality_order if m in set(freq["modality"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODALITY_COLORS[m]) for m in present]
    labels = [MODALITY_DISPLAY.get(m, m) for m in present]
    fig.legend(handles, labels, loc="lower center", ncol=len(present),
               frameon=False, fontsize=11, columnspacing=1.6, handletextpad=0.6)
    fig.subplots_adjust(left=0.47, right=0.94, top=0.96, bottom=0.065, hspace=0.20)
    fig.savefig(outdir / "Figure_4_feature_selection_frequency.png", dpi=400,
                bbox_inches="tight", pad_inches=0.08)
    fig.savefig(outdir / "Figure_4_feature_selection_frequency.pdf",
                bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

def plot_figure3_calibration(curves: pd.DataFrame, metrics: pd.DataFrame,
                             outdir: Path) -> None:
    """Publication-ready calibration plots using the Figure 2 family styles."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.3), sharex=True, sharey=True)
    for ax, task, panel in zip(axes, TASKS, ("a", "b")):
        sub = curves[(curves["task"] == task) &
                     (curves["condition"] == "all-modality")]
        met = metrics[(metrics["task"] == task) &
                      (metrics["condition"] == "all-modality")]
        ax.plot([0, 1], [0, 1], color="0.45", linestyle=(0, (4, 2)),
                linewidth=0.9, label="Perfect calibration", zorder=0)
        for cand in FIGURE_MODEL_ORDER:
            pts = sub[sub["candidate"] == cand]
            mm = met[met["candidate"] == cand].iloc[0]
            ax.plot(
                pts["mean_predicted_probability"].to_numpy(float),
                pts["observed_fraction_positive"].to_numpy(float),
                marker="o", markersize=5.5, markeredgewidth=0.7,
                color=MODEL_COLORS[cand], linestyle=MODEL_STYLES[cand],
                linewidth=MODEL_LINEWIDTHS[cand], clip_on=False,
                label=f"{CANDIDATE_LABELS[cand]} ({mm.brier_score:.3f})",
            )
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.035)
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_aspect(1.0 / 1.035, adjustable="box")
        ax.set_title(
            f"({panel}) {'1p/19q codeletion prediction' if task == '1p19q' else 'IDH mutation prediction'}",
            fontsize=FIG_TITLE_FONTSIZE, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
        ax.legend(
            loc="upper left",
            fontsize=7.0,
            frameon=True,
            framealpha=0.88,
            title="Model family (Brier score)",
            title_fontsize=7.5,
            borderaxespad=0.55,
            borderpad=0.45,
            labelspacing=0.35,
            handlelength=2.5,
            handletextpad=0.55,
            markerscale=0.85,
        )
    axes[0].set_ylabel("Observed Proportion Positive", fontsize=AXIS_LABEL_FONTSIZE)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.88, wspace=0.10)
    fig.savefig(outdir / "Figure_3_calibration_plots.png", dpi=400,
                bbox_inches="tight", pad_inches=0.06)
    fig.savefig(outdir / "Figure_3_calibration_plots.pdf",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

def write_publication_captions(outdir: Path) -> None:
    text = """Table 2. Predictive performance across the 50 held-out outer test folds. Values are mean ± standard deviation. Bold values indicate the highest mean within each task and metric and do not indicate statistical significance. AP, average precision; AUC, area under the receiver operating characteristic curve.\n\nTable 3. Patient-level paired comparisons of AUC. Out-of-sample probabilities were averaged across ten repetitions for each patient. Delta AUC and percentile 95% confidence intervals were estimated using 2,000 stratified paired bootstrap replicates with identical resampling applied to both models. Two-sided bootstrap p values were adjusted across the eight exploratory comparisons using the Holm method. The confidence intervals are pointwise percentile intervals and were not multiplicity-adjusted.\n\nTable 4. Predictive performance according to radiomics modality condition. Values are mean ± standard deviation across 50 held-out outer test folds. Bold values indicate the highest mean within each task and metric and do not indicate statistical significance. The paired modality comparisons are reported in Table S2.\n\nFigure 2. Mean receiver operating characteristic curves across the 50 held-out outer test folds for the four prespecified model families using all modalities: (a) 1p/19q codeletion and (b) IDH mutation. For each model family, fold-specific curves were interpolated on a common false-positive-rate grid and averaged; legend values show mean ± standard deviation of the 50 outer-fold AUCs.\n\nFigure 4. Twenty most frequently retained radiomic features across 50 final outer-training models for (a) 1p/19q codeletion and (b) IDH mutation. Selection frequency describes model-selection stability and does not establish causal or biological importance.\n\nFigure 3. Patient-level calibration plots for the four model families using all modalities: (a) 1p/19q codeletion and (b) IDH mutation. Points represent ten quantile-based probability bins. Out-of-sample probabilities were averaged across ten repetitions for each patient. Legend values are Brier scores.\n\nDiagnostic figure. Patient-level receiver operating characteristic curves for the four prespecified model families using all modalities: (a) 1p/19q codeletion and (b) IDH mutation. Out-of-sample probabilities were averaged across ten repetitions for each patient before curve construction.\n\nTable S1. Image-level binning and discretization diagnostics according to modality, image type, and fixed bin width.\n\nTable S2. Paired modality-group AUC comparisons for weighted averaging. The four comparisons were adjusted as one Holm family.\n\nTable S3. Patient-level calibration metrics for the four all-modality model families.\n\nTable S4. Radiomics candidate-fit diagnostics and final model-selection frequencies.\n"""
    (outdir / "publication_captions.txt").write_text(text, encoding="utf-8")

def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    outdir = (args.output or (root / "analysis_outputs")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(root)
    write_missing_information(runs, root, outdir)

    qc = run_qc(runs, args.expected_patients, args.expected_seeds, args.expected_folds)
    save_csv(qc, outdir / "01_qc_report.csv")
    errors = qc[qc["status"] == "ERROR"]
    if not errors.empty and args.strict:
        print(f"QC failed with {len(errors)} ERROR checks. See 01_qc_report.csv", file=sys.stderr)
        return 2

    missing_predictions = [key.slug for key, run in runs.items() if run.predictions is None]
    if missing_predictions:
        print("Cannot proceed without all six prediction files: " + ", ".join(missing_predictions),
              file=sys.stderr)
        return 3
    if not errors.empty:
        warnings.warn(f"QC contains {len(errors)} ERROR checks; outputs should not be used until resolved.")

    aggregated_parts = []
    for key, run in runs.items():
        agg = aggregate_patient_predictions(run)
        aggregated_parts.append(agg)
        save_csv(agg, outdir / f"02_patient_level_aggregated_{key.slug}.csv")
    aggregated = pd.concat(aggregated_parts, ignore_index=True)
    save_csv(aggregated, outdir / "02_patient_level_aggregated_all_runs.csv")

    table3 = make_table3(aggregated, args.bootstrap, args.random_state)
    save_csv(table3, outdir / "03_table3_paired_auc_comparisons.csv")
    table3_manuscript = make_table3_manuscript(table3)
    save_csv(table3_manuscript, outdir / "Table_3_manuscript.csv")

    calibration_metrics, calibration_curves = calibration_analysis(aggregated)
    save_csv(calibration_metrics, outdir / "04_calibration_metrics.csv")
    # Detailed calibration-plot QC: one row per realized calibration bin, including n.
    save_csv(calibration_curves, outdir / "04_calibration_curve_points_with_bin_counts.csv")
    calibration_bin_qc = make_calibration_bin_size_qc(
        calibration_curves, all_modality_only=True)
    save_csv(calibration_bin_qc,
             outdir / "04_calibration_bin_size_summary_QC.csv")

    modality = make_modality_comparison(aggregated, args.bootstrap, args.random_state)
    save_csv(modality, outdir / "05_modality_paired_comparisons.csv")

    summary = combine_summaries(runs)
    if not summary.empty:
        save_csv(summary, outdir / "06_all_final_summaries_unrounded.csv")
        table2 = make_table2_manuscript(summary)
        table4 = make_table4_manuscript(summary)
        save_csv(table2, outdir / "Table_2_manuscript.csv")
        save_csv(table4, outdir / "Table_4_manuscript.csv")

    feature_freq, selector_freq, config_freq, tuning_qc, fusion_freq = feature_diagnostics(runs)
    save_csv(feature_freq, outdir / "07_feature_selection_frequency.csv")
    if not feature_freq.empty:
        save_csv(feature_freq[feature_freq["rank_within_task"] <= 20],
                 outdir / "07_feature_selection_top20.csv")
        modal = (feature_freq.groupby(["task", "modality_group", "modality"], as_index=False)
                 ["selection_frequency"].sum())
        classes = (feature_freq.groupby(["task", "feature_class"], as_index=False)
                   ["selection_frequency"].sum())
        save_csv(modal, outdir / "07_modality_selection_frequency.csv")
        save_csv(classes, outdir / "07_feature_class_selection_frequency.csv")
    save_csv(selector_freq, outdir / "07_selector_frequency.csv")
    save_csv(config_freq, outdir / "07_configuration_frequency.csv")
    save_csv(tuning_qc, outdir / "07_inner_tuning_qc_summary.csv")
    save_csv(fusion_freq, outdir / "07_fusion_candidate_frequency.csv")

    # Publication-ready main figures and supplementary tables.
    plot_figure2_roc(runs, outdir)
    plot_diagnostic_patient_level_roc(aggregated, outdir)
    plot_figure4_features(feature_freq, outdir)
    plot_figure3_calibration(calibration_curves, calibration_metrics, outdir)
    table_s1 = make_supplementary_table_s1(root, outdir)
    table_s2 = make_supplementary_table_s2(modality)
    exploratory_complete = make_exploratory_complete_fusion_comparisons(table3)
    table_s3 = make_supplementary_table_s3(calibration_metrics)
    table_s4 = make_supplementary_table_s4(
        runs, selector_freq, config_freq, fusion_freq, outdir)
    save_csv(table_s2, outdir / "Table_S2_modality_comparisons.csv")
    save_csv(exploratory_complete, outdir / "exploratory_complete_fusion_comparisons.csv")
    save_csv(table_s3, outdir / "Table_S3_calibration_metrics.csv")
    save_csv(table_s4, outdir / "Table_S4_candidate_selection_failure_diagnostics.csv")
    write_publication_captions(outdir)
    write_caption_notes(outdir)

    manifest = {
        "root": str(root), "output": str(outdir), "bootstrap": args.bootstrap,
        "random_state": args.random_state, "expected_patients": args.expected_patients,
        "expected_seeds": args.expected_seeds, "expected_folds": args.expected_folds,
        "qc_error_checks": int((qc["status"] == "ERROR").sum()),
        "qc_warning_checks": int((qc["status"] == "WARNING").sum()),
        "calibration_bin_qc_file": "04_calibration_bin_size_summary_QC.csv",
        "runs": {key.slug: {name: str(path) if path else None
                             for name, path in run.files.items()}
                 for key, run in runs.items()},
    }
    (outdir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Analysis complete. Outputs: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
