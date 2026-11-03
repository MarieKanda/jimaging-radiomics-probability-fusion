# -*- coding: utf-8 -*-
"""Configuration for Radiomics extraction and Meta + Radiomics probability fusion.

Replace the placeholder paths before execution. No patient data, medical images,
or institutional identifiers are included in the public repository.
"""

from pathlib import Path
import random
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data_not_included"
OUTPUT_ROOT = PROJECT_ROOT / "results"

# Reproducibility and cross-validation
QUICK_TEST = False
SEEDS = [42] if QUICK_TEST else [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
OUTER_FOLDS = 2 if QUICK_TEST else 5
INNER_FOLDS = 2 if QUICK_TEST else 5


def set_seed(seed):
    """Set random seeds used by the NumPy/scikit-learn pipeline."""
    random.seed(seed)
    np.random.seed(seed)


# Task and tabular inputs
TASK = "IDH"  # "IDH" or "1p19q"
ANALYSIS_COHORT_NAME = "primary_217_fold_imputation"
LABEL_CSV = DATA_ROOT / "labels.csv"
DATA_DIR = DATA_ROOT / "images"  # Retained for entry-point compatibility only.
MODALITIES = ["FLAIR", "T1", "T2", "MET", "FDG", "CHO"]

# Probability-fusion mode
MODEL_NAME = "R1_PROB_FUSION_CALIBRATED_NESTED_THRESHOLD_BALACC"
FUSION_ARCHITECTURE = "probability"
PROB_FUSION_INPUTS = "meta_rad"
PROB_FUSION_THRESHOLD_MODE = "balanced_accuracy"
PROB_FUSION_FIXED_THRESHOLD = 0.5
PROB_FUSION_THRESHOLD_MIN = 0.10
PROB_FUSION_THRESHOLD_MAX = 0.90
PROB_FUSION_THRESHOLD_STEP = 0.01
PROB_FUSION_SELECTION_METRIC = "auc"
PROB_FUSION_STACKING_SELECTION_FOLDS = 5
PROB_FUSION_METHODS = [
    {"name": "weighted_m0.4_r0.6", "type": "weighted_average", "w_meta": 0.4, "w_rad": 0.6},
    {"name": "weighted_m0.5_r0.5", "type": "weighted_average", "w_meta": 0.5, "w_rad": 0.5},
    {"name": "weighted_m0.6_r0.4", "type": "weighted_average", "w_meta": 0.6, "w_rad": 0.4},
    {"name": "logit_stacking_C0.01", "type": "logit_stacking", "C": 0.01},
    {"name": "logit_stacking_C0.1", "type": "logit_stacking", "C": 0.1},
]

# Meta-information model
META_FEATURE_MODE = "full"
META_MODEL_TYPE = "logistic"
USE_CLASS_WEIGHT = True
USE_CALIBRATION = True
FEATURE_SELECTION_METHOD = "none"
USE_FIXED_FEATURES = True
FIXED_FEATURE_NAMES = ["Age", "X", "Y", "Z", "Calcification", "mismatch"]
META_MAX_ITER = 50000
META_TOL = 1e-3
META_RANDOM_STATE = 42

# Training-fold-only missing-data handling
META_MISSING_MODE = "fold_imputation"
META_MISSING_CODE_MAP = {"Calcification": [-1.0]}
META_IMPUTATION_STRATEGY_MAP = {"Calcification": "most_frequent"}
META_ALLOW_MISSING_COLUMNS = ["Calcification"]
META_ALLOWED_VALUES_MAP = {"Calcification": [0.0, 0.5, 1.0]}
META_ADD_MISSING_INDICATOR = False
META_MISSING_AUDIT = True

# Radiomics feature-table input
RADIOMICS_CSV = DATA_ROOT / "radiomics_features.csv"
RADIOMICS_ID_COL = "patient_id"
RADIOMICS_LABEL_COL_IDH = "IDH_mutation"
RADIOMICS_LABEL_COL_1P19Q = "1p19q_codeletion"
RADIOMICS_VARIANCE_THRESHOLD = 0.0
RADIOMICS_CORR_THRESHOLD = 0.90
RADIOMICS_SCALER = "standard"
USE_RADIOMICS_COLUMN_FILTER = True
RADIOMICS_FILTER_VERBOSE = True
RADIOMICS_MODEL_ROI_PREFIXES = ["ROI25"]
RADIOMICS_MODEL_MODALITIES = ["FLAIR", "T1", "T2", "MET", "FDG", "CHO"]
RADIOMICS_CACHE_TABLE = True
RADIOMICS_NUMERIC_DTYPE = "float32"
RADIOMICS_MEMORY_EFFICIENT_CORR = True
RADIOMICS_CORR_BLOCK_SIZE = 128
RADIOMICS_FORCE_GC = True

# Radiomics feature selection and inner-CV tuning
RADIOMICS_FS_STRATEGY = "per_fold"
RADIOMICS_FS_METHOD = "lasso"
RADIOMICS_ELASTICNET_L1_RATIO = 0.5
RADIOMICS_SELECTOR_MAX_ITER = 10000
RADIOMICS_SELECTOR_TOL = 1e-4
RADIOMICS_ALLOW_SELECTOR_FALLBACK = False
RADIOMICS_USE_INNER_TUNING = True
RADIOMICS_INNER_SELECTION_METRIC = "auc"
RADIOMICS_INNER_FOLDS = INNER_FOLDS
RADIOMICS_FEATURE_CONFIGS_FULL = [
    {
        "name": "A_firstorder_sigma1",
        "feature_classes": ["firstorder"],
        "image_families": ["original", "log", "wavelet"],
        "log_sigmas": [1.0],
        "wavelet_types": ["H"],
    },
    {
        "name": "B_firstorder_glcm_gldm_multisigma",
        "feature_classes": ["firstorder", "glcm", "gldm"],
        "image_families": ["original", "log", "wavelet"],
        "log_sigmas": [0.5, 1.0, 1.5, 2.0, 3.0],
        "wavelet_types": ["H"],
    },
]
RADIOMICS_FEATURE_CONFIGS_QUICK = [RADIOMICS_FEATURE_CONFIGS_FULL[0]]
RADIOMICS_FEATURE_CONFIGS = RADIOMICS_FEATURE_CONFIGS_QUICK if QUICK_TEST else RADIOMICS_FEATURE_CONFIGS_FULL
RADIOMICS_SELECTOR_CANDIDATES_FULL = [
    {"name": "lasso_C0.1_max10", "method": "lasso", "C": 0.1, "max_features": 10},
    {"name": "lasso_C0.1_max15", "method": "lasso", "C": 0.1, "max_features": 15},
    {"name": "lasso_C0.5_max10", "method": "lasso", "C": 0.5, "max_features": 10},
    {"name": "lasso_C0.5_max15", "method": "lasso", "C": 0.5, "max_features": 15},
    {"name": "anova_F_top10", "method": "anova_f", "max_features": 10},
    {"name": "anova_F_top15", "method": "anova_f", "max_features": 15},
]
RADIOMICS_SELECTOR_CANDIDATES_QUICK = [
    RADIOMICS_SELECTOR_CANDIDATES_FULL[0],
    RADIOMICS_SELECTOR_CANDIDATES_FULL[4],
]
RADIOMICS_SELECTOR_CANDIDATES = (
    RADIOMICS_SELECTOR_CANDIDATES_QUICK if QUICK_TEST else RADIOMICS_SELECTOR_CANDIDATES_FULL
)
RADIOMICS_LASSO_C = 0.1
RADIOMICS_MAX_FEATURES_FOR_FUSION = 15
RADIOMICS_CLASSIFIER = "logistic"
RADIOMICS_MODEL_TYPE = RADIOMICS_CLASSIFIER
RADIOMICS_CLASSIFIER_C = 0.01
RADIOMICS_CLASS_WEIGHT = "balanced"
RADIOMICS_CALIBRATION_METHOD = "sigmoid"
USE_RADIOMICS_CALIBRATION = True

# Radiomics extraction inputs and preprocessing
TUMOR_ROI = DATA_ROOT / "roi_coordinates"
RADIOMICS_ROI_MODE = "roi25"
RADIOMICS_DATA_DIR_25 = DATA_ROOT / "patches_25x25"
RADIOMICS_DATA_DIR_32 = DATA_ROOT / "patches_32x32"
RADIOMICS_EXTRACT_MODALITIES = ["FLAIR", "T1", "T2", "MET", "FDG", "CHO"]
RADIOMICS_YAML_TEMPLATE = PROJECT_ROOT / "radiomics_extraction" / "pyradiomics_template_probability_fusion.yaml"
RADIOMICS_OUTPUT_DIR = OUTPUT_ROOT / "radiomics"
MRI_CLIP_LOW = 1
MRI_CLIP_HIGH = 99
PET_CLIP_LOW = 1
PET_CLIP_HIGH = 99
USE_CHO_STRONG_CLIP = True
CHO_CLIP_LOW = 5
CHO_CLIP_HIGH = 95

# The extraction script supports one bin width per experiment. To reproduce the
# study input with MRI bin width 0.25 and PET bin width 10.0, run the two modality
# groups separately and merge their feature columns by patient_id before modeling.
RADIOMICS_EXPERIMENTS = [
    {
        "name": "MRI_bin025",
        "output_csv": RADIOMICS_OUTPUT_DIR / "radiomics_MRI_bin025.csv",
        "modalities": ["FLAIR", "T1", "T2"],
        "pet_preprocess": "clip_only",
        "mri_preprocess": "per_patch_z",
        "bin_width": 0.25,
        "feature_classes": ["firstorder", "glcm", "gldm"],
    },
    {
        "name": "PET_bin10",
        "output_csv": RADIOMICS_OUTPUT_DIR / "radiomics_PET_bin10.csv",
        "modalities": ["MET", "FDG", "CHO"],
        "pet_preprocess": "clip_only",
        "mri_preprocess": "per_patch_z",
        "bin_width": 10.0,
        "feature_classes": ["firstorder", "glcm", "gldm"],
    },
]
