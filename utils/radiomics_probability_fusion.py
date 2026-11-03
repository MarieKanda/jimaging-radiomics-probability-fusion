"""
Radiomics pipeline used by Meta + Radiomics probability-level fusion.

Retained: CSV table loading/cache, configuration-based column filtering,
training-fold-only imputation/scaling/filtering/feature selection, inner-CV
candidate selection, calibration, refitting, and probability prediction.

Removed: the helper used only by the former neural-network late-fusion path.
"""
import inspect
import re
import numpy as np
import pandas as pd
import gc
import os
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_selection import f_classif
from types import SimpleNamespace
_RADIOMICS_TABLE_CACHE = {}
KNOWN_FEATURE_CLASSES = ['firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'shape', 'shape2D']

def _safe_cv_splits(y, requested_cv=3):
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0
    return int(max(0, min(int(requested_cv), int(np.min(counts)))))

def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)

def _make_calibrated_classifier(base_model, method='sigmoid', cv=3):
    sig = inspect.signature(CalibratedClassifierCV)
    if 'estimator' in sig.parameters:
        return CalibratedClassifierCV(estimator=base_model, method=method, cv=cv)
    return CalibratedClassifierCV(base_estimator=base_model, method=method, cv=cv)

def _collect_pid_y_from_indices(dataset, indices):
    pids = []
    ys = []
    for i in indices:
        _, _, y, pid = dataset[i]
        pids.append(str(pid))
        ys.append(int(y.item() if hasattr(y, 'item') else y))
    return (np.asarray(pids).astype(str), np.asarray(ys).astype(int))

def _copy_config_with_radiomics_params(config, feature_config, max_features, selector_method, lasso_c=None):
    """Create an independent configuration object for one feature-selector candidate."""
    cfg_dict = {}
    for name in dir(config):
        if name.startswith('_'):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        if hasattr(value, '__package__') and hasattr(value, '__name__'):
            continue
        cfg_dict[name] = value
    cfg = SimpleNamespace(**cfg_dict)
    cfg.RADIOMICS_MODEL_FEATURE_CLASSES = list(feature_config['feature_classes'])
    cfg.RADIOMICS_MODEL_IMAGE_FAMILIES = list(feature_config['image_families'])
    cfg.RADIOMICS_MODEL_LOG_SIGMAS = list(feature_config['log_sigmas'])
    cfg.RADIOMICS_MODEL_WAVELET_TYPES = list(feature_config['wavelet_types'])
    cfg.RADIOMICS_FEATURE_CONFIG_NAME = str(feature_config['name'])
    cfg.RADIOMICS_MAX_FEATURES_FOR_FUSION = int(max_features)
    cfg.RADIOMICS_SELECTOR_METHOD = str(selector_method).lower()
    if cfg.RADIOMICS_SELECTOR_METHOD == 'lasso':
        if lasso_c is None:
            raise ValueError('lasso candidate requires lasso_c.')
        cfg.RADIOMICS_LASSO_C = float(lasso_c)
    elif cfg.RADIOMICS_SELECTOR_METHOD == 'anova_f':
        cfg.RADIOMICS_LASSO_C = np.nan
    else:
        raise ValueError(f"selector_method must be 'lasso' or 'anova_f'. Received: {selector_method}")
    return cfg

def _get_radiomics_feature_config_grid(config):
    """Build the grid of feature configurations and explicit selector candidates."""
    feature_configs = getattr(config, 'RADIOMICS_FEATURE_CONFIGS', None)
    if feature_configs is None:
        feature_configs = [{'name': 'current_config', 'feature_classes': getattr(config, 'RADIOMICS_MODEL_FEATURE_CLASSES', ['firstorder']), 'image_families': getattr(config, 'RADIOMICS_MODEL_IMAGE_FAMILIES', ['original', 'log', 'wavelet']), 'log_sigmas': getattr(config, 'RADIOMICS_MODEL_LOG_SIGMAS', [1.0]), 'wavelet_types': getattr(config, 'RADIOMICS_MODEL_WAVELET_TYPES', ['H'])}]
    selector_candidates = getattr(config, 'RADIOMICS_SELECTOR_CANDIDATES', None)
    if selector_candidates is None:
        selector_candidates = [{'name': 'lasso_C0.1_max10', 'method': 'lasso', 'C': 0.1, 'max_features': 10}, {'name': 'lasso_C0.1_max15', 'method': 'lasso', 'C': 0.1, 'max_features': 15}, {'name': 'lasso_C0.5_max10', 'method': 'lasso', 'C': 0.5, 'max_features': 10}, {'name': 'lasso_C0.5_max15', 'method': 'lasso', 'C': 0.5, 'max_features': 15}, {'name': 'anova_F_top10', 'method': 'anova_f', 'max_features': 10}, {'name': 'anova_F_top15', 'method': 'anova_f', 'max_features': 15}]
    grid = []
    for feature_config in feature_configs:
        for selector in selector_candidates:
            method = str(selector['method']).lower()
            max_features = int(selector['max_features'])
            lasso_c = float(selector['C']) if method == 'lasso' else None
            selector_name = str(selector.get('name', method))
            grid.append({'feature_config': dict(feature_config), 'selector_method': method, 'selector_name': selector_name, 'max_features': max_features, 'lasso_c': lasso_c, 'name': f"{feature_config['name']}__{selector_name}"})
    return grid

def parse_radiomics_feature_name(col):
    parts = str(col).split('_')
    if len(parts) < 4:
        return None
    roi = parts[0]
    modality = parts[1]
    class_pos = None
    feature_class = None
    for i, p in enumerate(parts):
        if p in KNOWN_FEATURE_CLASSES:
            class_pos = i
            feature_class = p
            break
    if class_pos is None or class_pos < 2:
        return None
    image_family = '_'.join(parts[2:class_pos])
    feature_name = '_'.join(parts[class_pos + 1:])
    if image_family == 'original':
        image_family_group = 'original'
    elif image_family.startswith('log'):
        image_family_group = 'log'
    elif image_family.startswith('wavelet'):
        image_family_group = 'wavelet'
    else:
        image_family_group = image_family
    log_sigma = None
    if image_family_group == 'log':
        m = re.search('log-sigma-([0-9]+)-([0-9]+)-mm', image_family)
        if m is not None:
            log_sigma = float(f'{m.group(1)}.{m.group(2)}')
    wavelet_type = None
    if image_family_group == 'wavelet':
        m = re.search('wavelet-([^_]+)', image_family)
        if m is not None:
            wavelet_type = m.group(1)
    return {'roi': roi, 'modality': modality, 'image_family': image_family, 'image_family_group': image_family_group, 'log_sigma': log_sigma, 'wavelet_type': wavelet_type, 'feature_class': feature_class, 'feature_name': feature_name}

def filter_radiomics_columns_by_config(feature_cols, config):
    feature_cols = list(feature_cols)
    if not bool(getattr(config, 'USE_RADIOMICS_COLUMN_FILTER', True)):
        raise ValueError('The public probability-fusion pipeline requires USE_RADIOMICS_COLUMN_FILTER=True.')
    roi_prefixes = _as_list(getattr(config, 'RADIOMICS_MODEL_ROI_PREFIXES', []))
    modalities = _as_list(getattr(config, 'RADIOMICS_MODEL_MODALITIES', []))
    feature_classes = _as_list(getattr(config, 'RADIOMICS_MODEL_FEATURE_CLASSES', []))
    image_families = _as_list(getattr(config, 'RADIOMICS_MODEL_IMAGE_FAMILIES', []))
    log_sigmas = _as_list(getattr(config, 'RADIOMICS_MODEL_LOG_SIGMAS', []))
    wavelet_types = _as_list(getattr(config, 'RADIOMICS_MODEL_WAVELET_TYPES', []))
    log_sigmas_float = [float(x) for x in log_sigmas]
    selected = []
    for col in feature_cols:
        info = parse_radiomics_feature_name(col)
        if info is None:
            continue
        if roi_prefixes and info['roi'] not in roi_prefixes:
            continue
        if modalities and info['modality'] not in modalities:
            continue
        if feature_classes and info['feature_class'] not in feature_classes:
            continue
        if image_families and info['image_family_group'] not in image_families:
            continue
        if info['image_family_group'] == 'log' and log_sigmas_float:
            if info['log_sigma'] not in log_sigmas_float:
                continue
        if info['image_family_group'] == 'wavelet' and wavelet_types:
            if info['wavelet_type'] not in wavelet_types:
                continue
        selected.append(col)
    if getattr(config, 'RADIOMICS_FILTER_VERBOSE', True):
        print('[RadiomicsFilter]')
        print('  input columns :', len(feature_cols))
        print('  output columns:', len(selected))
        print('  modalities    :', modalities)
        print('  classes       :', feature_classes)
        print('  families      :', image_families)
        print('  log sigmas    :', log_sigmas)
        print('  wavelet types :', wavelet_types)
    if len(selected) == 0:
        raise ValueError('No radiomics feature columns remained after filtering. Check modalities, feature classes, image families, sigmas, and CSV column names.')
    return selected

class RadiomicsTable:
    """
    Memory-efficient Radiomics table.

    Main changes
    ------------
    1. The CSV is read only once for each file path.
    2. Patient IDs are mapped directly to row positions.
    3. Numeric Radiomics data are stored as one float32 matrix.
    4. get_X() no longer creates a new indexed DataFrame.
    5. get_X() no longer calls apply(pd.to_numeric) repeatedly.
    """

    def __init__(self, config):
        self.config = config
        self.id_col = getattr(config, 'RADIOMICS_ID_COL', 'patient_id')
        self.csv_path = os.path.abspath(str(getattr(config, 'RADIOMICS_CSV')))
        dtype_name = str(getattr(config, 'RADIOMICS_NUMERIC_DTYPE', 'float32')).lower()
        if dtype_name != 'float32':
            raise ValueError("The public probability-fusion pipeline supports RADIOMICS_NUMERIC_DTYPE='float32' only.")
        self.numeric_dtype = np.float32
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f'Radiomics CSV was not found: {self.csv_path}')
        print('[RadiomicsTable] Loading CSV once:', self.csv_path)
        df = pd.read_csv(self.csv_path, low_memory=False)
        if self.id_col not in df.columns:
            raise ValueError(f"RADIOMICS_ID_COL='{self.id_col}' was not found in {self.csv_path}")
        pid_keys = df[self.id_col].astype(str).str.strip().values
        duplicated_mask = pd.Series(pid_keys).duplicated(keep=False).values
        if np.any(duplicated_mask):
            duplicated_ids = sorted(np.unique(pid_keys[duplicated_mask]).tolist())
            raise ValueError(f'Duplicated patient IDs were found in the Radiomics CSV. Examples: {duplicated_ids[:10]}')
        label_cols = {self.id_col, getattr(config, 'RADIOMICS_LABEL_COL_IDH', 'IDH_mutation'), getattr(config, 'RADIOMICS_LABEL_COL_1P19Q', '1p19q_codeletion'), 'IDH_mutation', '1p19q_codeletion', 'subtyped', 'Grade'}
        candidate_feature_cols = []
        for column in df.columns:
            if column in label_cols:
                continue
            if str(column).startswith('diagnostics_'):
                continue
            candidate_feature_cols.append(column)
        if len(candidate_feature_cols) == 0:
            raise ValueError('No candidate Radiomics columns were found.')
        print(f'[RadiomicsTable] Converting {len(candidate_feature_cols)} candidate columns to numeric...')
        numeric_df = df[candidate_feature_cols].apply(pd.to_numeric, errors='coerce')
        valid_numeric_mask = numeric_df.notna().sum(axis=0) > 0
        feature_cols = numeric_df.columns[valid_numeric_mask].astype(str).tolist()
        if len(feature_cols) == 0:
            raise ValueError('No numeric Radiomics feature columns were found.')
        numeric_df = numeric_df.loc[:, feature_cols]
        X_all = numeric_df.to_numpy(dtype=self.numeric_dtype, copy=True)
        self.feature_cols = list(feature_cols)
        self.feature_to_position = {name: position for position, name in enumerate(self.feature_cols)}
        self.pid_to_row = {str(pid): position for position, pid in enumerate(pid_keys)}
        self.pid_keys = np.asarray(pid_keys, dtype=str)
        self.X_all = np.ascontiguousarray(X_all, dtype=self.numeric_dtype)
        print('[RadiomicsTable] Loaded:')
        print('  patients :', self.X_all.shape[0])
        print('  features :', self.X_all.shape[1])
        print('  dtype    :', self.X_all.dtype)
        print('  matrix MB:', round(self.X_all.nbytes / 1024 ** 2, 2))
        del numeric_df
        del df
        gc.collect()

    def get_X(self, pids, feature_cols=None):
        """
        Retrieve the requested patients and features.

        No CSV read, DataFrame indexing, or numeric conversion
        is performed here.
        """
        pid_list = [str(pid).strip() for pid in pids]
        if feature_cols is None:
            selected_feature_cols = list(self.feature_cols)
        else:
            selected_feature_cols = [str(column) for column in feature_cols]
        missing_pids = [pid for pid in pid_list if pid not in self.pid_to_row]
        if len(missing_pids) > 0:
            raise KeyError(f'{len(missing_pids)} patient_id values were not found in the Radiomics CSV. Examples: {missing_pids[:10]}')
        missing_columns = [column for column in selected_feature_cols if column not in self.feature_to_position]
        if len(missing_columns) > 0:
            raise KeyError(f'{len(missing_columns)} Radiomics feature columns were not found. Examples: {missing_columns[:10]}')
        row_positions = np.fromiter((self.pid_to_row[pid] for pid in pid_list), dtype=np.int64, count=len(pid_list))
        column_positions = np.fromiter((self.feature_to_position[column] for column in selected_feature_cols), dtype=np.int64, count=len(selected_feature_cols))
        X = self.X_all[np.ix_(row_positions, column_positions)]
        X = np.asarray(X, dtype=self.numeric_dtype, order='C').copy()
        return (X, selected_feature_cols)

def get_cached_radiomics_table(config):
    csv_path = os.path.abspath(str(getattr(config, 'RADIOMICS_CSV')))
    id_col = str(getattr(config, 'RADIOMICS_ID_COL', 'patient_id'))
    dtype_name = str(getattr(config, 'RADIOMICS_NUMERIC_DTYPE', 'float32')).lower()
    if not bool(getattr(config, 'RADIOMICS_CACHE_TABLE', True)):
        raise ValueError('The public probability-fusion pipeline requires RADIOMICS_CACHE_TABLE=True.')
    cache_key = (csv_path, id_col, dtype_name)
    if cache_key not in _RADIOMICS_TABLE_CACHE:
        _RADIOMICS_TABLE_CACHE[cache_key] = RadiomicsTable(config)
    else:
        print('[RadiomicsTable] Reusing cached table.')
    return _RADIOMICS_TABLE_CACHE[cache_key]

def fit_variance_filter(X_train, feature_names, config):
    threshold = float(getattr(config, 'RADIOMICS_VARIANCE_THRESHOLD', 0.0))
    feature_names = list(feature_names)
    variances = np.var(X_train, axis=0)
    keep_idx = np.where(variances > threshold)[0]
    if len(keep_idx) == 0:
        raise ValueError('Variance filter removed all radiomics features.')
    kept_names = [feature_names[i] for i in keep_idx]
    print(f'[Radiomics VarianceFilter] kept {len(kept_names)} / {len(feature_names)}')
    return (keep_idx.astype(int), kept_names)

def fit_correlation_filter(X_train, feature_names, config):
    """
    Memory-efficient greedy correlation filter.

    The previous implementation constructed a complete
    p x p correlation matrix. With 2,352 features, that
    required a 2,352 x 2,352 float64 array in addition
    to the already allocated Radiomics tables.

    This implementation processes one feature at a time
    and compares it only with features already retained.
    Therefore, it does not construct the full correlation
    matrix.

    The filtering rule remains:
        keep feature j if its absolute Pearson correlation
        with every previously retained feature is <= threshold.

    The result preserves the original feature order.
    """
    threshold = getattr(config, 'RADIOMICS_CORR_THRESHOLD', 0.9)
    feature_names = list(feature_names)
    X = np.asarray(X_train, dtype=np.float32, order='C')
    if X.ndim != 2:
        raise ValueError(f'X_train must be a two-dimensional array. Received shape={X.shape}')
    if X.shape[1] != len(feature_names):
        raise ValueError(f'The number of Radiomics features does not match the number of feature names: X.shape[1]={X.shape[1]}, names={len(feature_names)}')
    if not np.all(np.isfinite(X)):
        raise ValueError('Non-finite values remain before the Radiomics correlation filter.')
    if threshold is None or float(threshold) >= 1.0:
        keep_idx = np.arange(X.shape[1], dtype=int)
        return (keep_idx, feature_names)
    threshold = float(threshold)
    if threshold < 0.0:
        raise ValueError('RADIOMICS_CORR_THRESHOLD must be non-negative.')
    n_samples = X.shape[0]
    n_features = X.shape[1]
    if n_samples < 2:
        raise ValueError('At least two training samples are required for correlation filtering.')
    means = np.mean(X, axis=0, dtype=np.float64)
    centered_norms = np.empty(n_features, dtype=np.float64)
    for j in range(n_features):
        column = X[:, j].astype(np.float64, copy=False) - means[j]
        centered_norms[j] = np.sqrt(np.dot(column, column))
    keep = []
    for j in range(n_features):
        norm_j = centered_norms[j]
        if not np.isfinite(norm_j) or norm_j <= 1e-12:
            continue
        if len(keep) == 0:
            keep.append(j)
            continue
        xj = X[:, j].astype(np.float64, copy=False) - means[j]
        should_keep = True
        for kept_j in keep:
            norm_kept = centered_norms[kept_j]
            if not np.isfinite(norm_kept) or norm_kept <= 1e-12:
                continue
            x_kept = X[:, kept_j].astype(np.float64, copy=False) - means[kept_j]
            denominator = norm_j * norm_kept
            if denominator <= 1e-12:
                continue
            corr = float(np.dot(xj, x_kept) / denominator)
            if abs(corr) > threshold:
                should_keep = False
                break
        if should_keep:
            keep.append(j)
    keep_idx = np.asarray(keep, dtype=int)
    if len(keep_idx) == 0:
        raise ValueError('Correlation filter removed all Radiomics features.')
    kept_names = [feature_names[i] for i in keep_idx]
    print(f'[Radiomics CorrFilter] kept {len(kept_names)} / {len(feature_names)} (memory-efficient mode)')
    return (keep_idx, kept_names)

class NoRadiomicsFeaturesError(ValueError):
    """Raised when LASSO or Elastic Net selects no Radiomics features."""
    pass

def fit_feature_selector(X_train, y_train, feature_names, config, seed):
    """Select Radiomics features using training data only. LASSO ranks nonzero coefficients by absolute magnitude and invalidates zero-feature candidates without ANOVA fallback. ANOVA F retains the highest-ranked training-only F statistics."""
    feature_names = list(feature_names)
    y_train = np.asarray(y_train).astype(int)
    strategy = str(getattr(config, 'RADIOMICS_FS_STRATEGY', 'per_fold')).lower()
    if strategy != 'per_fold':
        raise ValueError("The public probability-fusion pipeline supports RADIOMICS_FS_STRATEGY='per_fold' only.")
    max_features = int(getattr(config, 'RADIOMICS_MAX_FEATURES_FOR_FUSION', 15))
    actual_max = min(max_features, X_train.shape[1])
    if X_train.shape[1] == 0:
        raise NoRadiomicsFeaturesError('No input radiomics features remained.')
    method = str(getattr(config, 'RADIOMICS_SELECTOR_METHOD', getattr(config, 'RADIOMICS_FS_METHOD', 'lasso'))).lower()
    if method == 'lasso':
        C = float(getattr(config, 'RADIOMICS_LASSO_C', 0.1))
        cw = _get_radiomics_class_weight(config)
        model = LogisticRegression(penalty='l1', C=C, solver='liblinear', class_weight=cw, max_iter=int(getattr(config, 'RADIOMICS_SELECTOR_MAX_ITER', 10000)), tol=float(getattr(config, 'RADIOMICS_SELECTOR_TOL', 0.0001)), random_state=seed)
        model.fit(X_train, y_train)
        coef_abs = np.abs(model.coef_).ravel()
        nonzero_idx = np.where(coef_abs > 1e-12)[0]
        n_nonzero = int(len(nonzero_idx))
        if n_nonzero == 0:
            print(f'[INVALID CANDIDATE] LASSO selected no features (C={C}); ANOVA fallback is not used.')
            raise NoRadiomicsFeaturesError(f'LASSO selected no features (C={C}, input_features={X_train.shape[1]}).')
        ranked = nonzero_idx[np.argsort(coef_abs[nonzero_idx])[::-1]]
        idx = np.asarray(ranked[:min(actual_max, n_nonzero)], dtype=int)
        reason = 'nonzero_lasso'
        info = {'selector_method': 'lasso', 'selector_reason': reason, 'fallback_used': False, 'n_nonzero_coefficients': n_nonzero, 'n_selected': int(len(idx)), 'input_features': int(X_train.shape[1]), 'lasso_c': C}
    elif method in ['anova', 'anova_f', 'f_classif']:
        scores, _ = f_classif(X_train, y_train)
        scores = np.asarray(scores, dtype=float)
        scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        finite_or_posinf = np.where(scores > -np.inf)[0]
        if len(finite_or_posinf) == 0:
            raise NoRadiomicsFeaturesError('ANOVA F produced no valid feature scores.')
        ranked = np.argsort(scores)[::-1]
        ranked = ranked[scores[ranked] > -np.inf]
        idx = np.asarray(ranked[:min(actual_max, len(ranked))], dtype=int)
        if len(idx) == 0:
            raise NoRadiomicsFeaturesError('ANOVA F selected no features.')
        reason = 'anova_f_topk'
        info = {'selector_method': 'anova_f', 'selector_reason': reason, 'fallback_used': False, 'n_nonzero_coefficients': np.nan, 'n_selected': int(len(idx)), 'input_features': int(X_train.shape[1]), 'lasso_c': np.nan}
    else:
        raise ValueError(f'Unsupported selector method: {method}')
    names = [feature_names[i] for i in idx]
    print(f"[Radiomics FS] method={info['selector_method']}, max_features={max_features}, input_features={X_train.shape[1]}, n_nonzero={info['n_nonzero_coefficients']}, selected={len(idx)}, reason={reason}, fallback_used=False")
    return (idx, names, info)

def _get_radiomics_class_weight(config):
    cw = str(getattr(config, 'RADIOMICS_CLASS_WEIGHT', 'balanced')).lower()
    if cw != 'balanced':
        raise ValueError("The public probability-fusion pipeline supports RADIOMICS_CLASS_WEIGHT='balanced' only.")
    return 'balanced'

def _get_radiomics_calibration_method(config):
    method = str(getattr(config, 'RADIOMICS_CALIBRATION_METHOD', 'sigmoid')).lower()
    if method != 'sigmoid' or not bool(getattr(config, 'USE_RADIOMICS_CALIBRATION', True)):
        raise ValueError('The public probability-fusion pipeline requires sigmoid calibration with USE_RADIOMICS_CALIBRATION=True.')
    return 'sigmoid'

def _make_sample_weight_if_needed(config, y_train):
    cw = _get_radiomics_class_weight(config)
    if cw == 'balanced':
        return compute_sample_weight(class_weight='balanced', y=y_train)
    return None

def _fit_model_with_optional_sample_weight(model, X, y, config):
    sample_weight = _make_sample_weight_if_needed(config, y)
    if sample_weight is None:
        model.fit(X, y)
    else:
        try:
            model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            model.fit(X, y)
    return model

def get_radiomics_classifier(config, y_train=None, seed=42):
    model_type = str(getattr(config, 'RADIOMICS_MODEL_TYPE', getattr(config, 'RADIOMICS_CLASSIFIER', 'logistic'))).lower()
    if model_type != 'logistic':
        raise ValueError("The public probability-fusion pipeline supports RADIOMICS_MODEL_TYPE='logistic' only.")
    class_weight = _get_radiomics_class_weight(config)
    classifier_c = float(getattr(config, 'RADIOMICS_CLASSIFIER_C', 0.01))
    return LogisticRegression(penalty='l2', C=classifier_c, solver='liblinear', class_weight=class_weight, max_iter=5000, random_state=seed)

def _fit_radiomics_pipeline_fixed_config(dataset, config, train_indices, seed):
    """
    Fit one fixed Radiomics configuration using training data only.

    Memory management
    -----------------
    - Reuse one cached Radiomics table.
    - Use float32 working arrays.
    - Delete large intermediate arrays as soon as they
      are no longer required.
    - Use the memory-efficient correlation filter.

    Leakage control
    ---------------
    Imputer, scaler, variance filter, correlation filter,
    feature selector, classifier, and calibration are fitted
    using only train_indices.
    """
    rad_table = get_cached_radiomics_table(config)
    pids_train, y_train = _collect_pid_y_from_indices(dataset, train_indices)
    y_train = np.asarray(y_train, dtype=int)
    if len(np.unique(y_train)) < 2:
        raise ValueError('Radiomics training fold contains only one class.')
    feature_names0 = filter_radiomics_columns_by_config(rad_table.feature_cols, config)
    X_raw, feature_names0 = rad_table.get_X(pids_train, feature_cols=feature_names0)
    X_raw = np.asarray(X_raw, dtype=np.float32, order='C')
    print('[Radiomics Matrix] raw shape=', X_raw.shape, 'dtype=', X_raw.dtype, 'MB=', round(X_raw.nbytes / 1024 ** 2, 2))
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_raw)
    X_imp = np.asarray(X_imp, dtype=np.float32, order='C')
    del X_raw
    scaler_type = str(getattr(config, 'RADIOMICS_SCALER', 'standard')).lower()
    if scaler_type != 'standard':
        raise ValueError("The public probability-fusion pipeline supports RADIOMICS_SCALER='standard' only.")
    scaler = StandardScaler(copy=True)
    X_scaled = scaler.fit_transform(X_imp)
    X_scaled = np.asarray(X_scaled, dtype=np.float32, order='C')
    del X_imp
    variance_keep_idx, feature_names1 = fit_variance_filter(X_scaled, feature_names0, config)
    X_var = np.asarray(X_scaled[:, variance_keep_idx], dtype=np.float32, order='C').copy()
    del X_scaled
    corr_keep_idx, feature_names2 = fit_correlation_filter(X_var, feature_names1, config)
    X_corr = np.asarray(X_var[:, corr_keep_idx], dtype=np.float32, order='C').copy()
    del X_var
    print(f'[Radiomics Matrix] after correlation shape={X_corr.shape}, dtype={X_corr.dtype}, MB={X_corr.nbytes / 1024 ** 2:.2f}')
    selected_idx, selected_feature_names, selector_info = fit_feature_selector(X_corr, y_train, feature_names2, config, seed)
    X_sel = np.asarray(X_corr[:, selected_idx], dtype=np.float32, order='C').copy()
    del X_corr
    base_model = get_radiomics_classifier(config, y_train=y_train, seed=seed)
    model = _fit_model_with_optional_sample_weight(base_model, X_sel, y_train, config)
    calibration_method = _get_radiomics_calibration_method(config)
    cv_calib = _safe_cv_splits(y_train, requested_cv=3)
    if cv_calib >= 2:
        calibrated = _make_calibrated_classifier(clone(model), method=calibration_method, cv=cv_calib)
        calibrated.fit(X_sel, y_train)
        final_model = calibrated
    else:
        print('[WARNING] Calibration skipped because too few samples per class.')
        final_model = model
    pipeline = {'rad_table': rad_table, 'feature_names0': feature_names0, 'feature_names1': feature_names1, 'feature_names2': feature_names2, 'selected_feature_names': selected_feature_names, 'imputer': imputer, 'scaler': scaler, 'variance_keep_idx': variance_keep_idx, 'corr_keep_idx': corr_keep_idx, 'selected_idx': selected_idx, 'selector_method': selector_info.get('selector_method', ''), 'selector_reason': selector_info.get('selector_reason', ''), 'fallback_used': bool(selector_info.get('fallback_used', False)), 'n_nonzero_coefficients': selector_info.get('n_nonzero_coefficients', np.nan), 'n_selected': int(selector_info.get('n_selected', len(selected_feature_names))), 'selector_input_features': int(selector_info.get('input_features', len(feature_names2))), 'model': final_model, 'train_pids': pids_train, 'train_y': y_train, 'feature_config_name': getattr(config, 'RADIOMICS_FEATURE_CONFIG_NAME', 'current_config'), 'lasso_c': float(getattr(config, 'RADIOMICS_LASSO_C', np.nan)), 'max_features': int(getattr(config, 'RADIOMICS_MAX_FEATURES_FOR_FUSION', -1)), 'feature_classes': '|'.join(_as_list(getattr(config, 'RADIOMICS_MODEL_FEATURE_CLASSES', []))), 'log_sigmas': '|'.join(map(str, _as_list(getattr(config, 'RADIOMICS_MODEL_LOG_SIGMAS', [])))), 'wavelet_types': '|'.join(_as_list(getattr(config, 'RADIOMICS_MODEL_WAVELET_TYPES', []))), 'feature_selection_method': str(selector_info.get('selector_method', '')), 'elasticnet_l1_ratio': float(getattr(config, 'RADIOMICS_ELASTICNET_L1_RATIO', 0.5)) if str(getattr(config, 'RADIOMICS_FS_METHOD', 'lasso')).lower() == 'elasticnet' else np.nan, 'numeric_dtype': 'float32'}
    del X_sel
    if bool(getattr(config, 'RADIOMICS_FORCE_GC', True)):
        gc.collect()
    return pipeline

def predict_radiomics_prob(dataset, config, pred_indices, pipeline):
    """
    Apply a training-fitted Radiomics pipeline to validation/test data.

    No component is fitted here.
    """
    pids, y_true = _collect_pid_y_from_indices(dataset, pred_indices)
    X_raw, _ = pipeline['rad_table'].get_X(pids, feature_cols=pipeline['feature_names0'])
    X_raw = np.asarray(X_raw, dtype=np.float32, order='C')
    X_imp = pipeline['imputer'].transform(X_raw)
    X_imp = np.asarray(X_imp, dtype=np.float32, order='C')
    del X_raw
    scaler = pipeline['scaler']
    if scaler is not None:
        X_scaled = scaler.transform(X_imp)
        X_scaled = np.asarray(X_scaled, dtype=np.float32, order='C')
    else:
        X_scaled = np.asarray(X_imp, dtype=np.float32, order='C').copy()
    del X_imp
    X_var = np.asarray(X_scaled[:, pipeline['variance_keep_idx']], dtype=np.float32, order='C').copy()
    del X_scaled
    X_corr = np.asarray(X_var[:, pipeline['corr_keep_idx']], dtype=np.float32, order='C').copy()
    del X_var
    X_sel = np.asarray(X_corr[:, pipeline['selected_idx']], dtype=np.float32, order='C').copy()
    del X_corr
    model = pipeline['model']
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(X_sel)[:, 1]
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(X_sel)
        prob = 1.0 / (1.0 + np.exp(-score))
    else:
        raise TypeError('Radiomics model has neither predict_proba nor decision_function.')
    prob = np.asarray(prob, dtype=float)
    if not np.all(np.isfinite(prob)):
        raise ValueError('Radiomics prediction contains NaN or Inf.')
    X_selected_output = X_sel.copy()
    del X_sel
    if bool(getattr(config, 'RADIOMICS_FORCE_GC', True)):
        gc.collect()
    return (prob, np.asarray(y_true).astype(int), np.asarray(pids).astype(str), X_selected_output)

def _score_inner_predictions(y_true, y_prob, metric='auc'):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    if metric == 'auc':
        return float(roc_auc_score(y_true, y_prob))
    if metric in ['ap', 'average_precision']:
        return float(average_precision_score(y_true, y_prob))
    raise ValueError(f'Unsupported RADIOMICS_INNER_SELECTION_METRIC: {metric}')

def select_radiomics_params_by_inner_cv(dataset, config, train_indices, seed):
    """Compare mean scores only among candidates that succeed in every inner fold."""
    _, y = _collect_pid_y_from_indices(dataset, train_indices)
    y = np.asarray(y).astype(int)
    inds = np.asarray(train_indices)
    requested = int(getattr(config, 'RADIOMICS_INNER_FOLDS', getattr(config, 'INNER_FOLDS', 5)))
    n_splits = _safe_cv_splits(y, requested_cv=requested)
    if n_splits < 2:
        raise RuntimeError('Radiomics inner CV requires at least 2 folds.')
    metric = getattr(config, 'RADIOMICS_INNER_SELECTION_METRIC', 'auc')
    grid = _get_radiomics_feature_config_grid(config)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 991)
    splits = list(skf.split(np.zeros(len(inds)), y))
    all_records = []
    valid_candidates = []
    print(f'[Radiomics InnerCV] n_splits={n_splits}, candidates={len(grid)}, metric={metric}')
    for candidate_id, cand in enumerate(grid):
        scores = []
        records = []
        for inner_fold, (tr_pos, va_pos) in enumerate(splits):
            cfg = _copy_config_with_radiomics_params(config=config, feature_config=cand['feature_config'], max_features=cand['max_features'], selector_method=cand['selector_method'], lasso_c=cand['lasso_c'])
            rec = {'candidate_id': candidate_id, 'candidate_name': cand['name'], 'selector_name': cand['selector_name'], 'selector_method': cand['selector_method'], 'inner_fold': inner_fold, 'feature_config_name': cand['feature_config']['name'], 'max_features': cand['max_features'], 'lasso_c': cand['lasso_c'] if cand['lasso_c'] is not None else np.nan, 'valid_fit': 0, 'score': np.nan, 'n_nonzero_coefficients': 0 if cand['selector_method'] == 'lasso' else np.nan, 'n_selected': 0, 'selector_reason': 'fit_failed', 'fallback_used': False, 'error_type': '', 'error_message': ''}
            try:
                pipe = _fit_radiomics_pipeline_fixed_config(dataset, cfg, inds[tr_pos].tolist(), seed + candidate_id * 10000 + inner_fold)
                prob, y_valid, _, _ = predict_radiomics_prob(dataset, cfg, inds[va_pos].tolist(), pipe)
                score = _score_inner_predictions(y_valid, prob, metric)
                if not np.isfinite(score):
                    raise ValueError('Non-finite inner score.')
                scores.append(float(score))
                rec.update({'valid_fit': 1, 'score': float(score), 'n_nonzero_coefficients': pipe.get('n_nonzero_coefficients', np.nan), 'n_selected': pipe.get('n_selected', len(pipe.get('selected_feature_names', []))), 'selector_reason': pipe.get('selector_reason', ''), 'fallback_used': False})
            except Exception as exc:
                rec.update({'selector_reason': 'zero_features' if isinstance(exc, NoRadiomicsFeaturesError) else 'fit_failed', 'error_type': type(exc).__name__, 'error_message': str(exc)})
                print(f"[Radiomics InnerCV INVALID] candidate={cand['name']}, fold={inner_fold}, error={type(exc).__name__}: {exc}")
            finally:
                if 'pipe' in locals():
                    del pipe
                if bool(getattr(config, 'RADIOMICS_FORCE_GC', True)):
                    gc.collect()
            records.append(rec)
            all_records.append(rec)
        n_valid = int(sum((r['valid_fit'] for r in records)))
        is_valid = n_valid == n_splits
        mean_score = float(np.mean(scores)) if is_valid else np.nan
        sd_score = float(np.std(scores)) if is_valid else np.nan
        item = {**cand, 'inner_score_mean': mean_score, 'inner_score_sd': sd_score, 'candidate_valid': is_valid, 'n_valid_folds': n_valid, 'n_required_folds': n_splits, 'inner_records': records}
        print(f"[Radiomics InnerCV] {cand['name']}: status={('VALID' if is_valid else 'INVALID')}, valid_folds={n_valid}/{n_splits}, {metric}_mean={mean_score}, sd={sd_score}")
        if is_valid:
            valid_candidates.append(item)
    if not valid_candidates:
        raise RuntimeError('Radiomics inner CV failed: no candidate was valid in all inner folds.')
    ranked = sorted(valid_candidates, key=lambda x: x['inner_score_mean'], reverse=True)
    best = dict(ranked[0])
    best['all_inner_records'] = all_records
    best['ranked_valid_candidates'] = [{k: r[k] for k in ['feature_config', 'selector_method', 'selector_name', 'max_features', 'lasso_c', 'name', 'inner_score_mean', 'inner_score_sd', 'candidate_valid', 'n_valid_folds', 'n_required_folds']} for r in ranked]
    print(f"[Radiomics InnerCV] BEST: {best['name']} mean={best['inner_score_mean']:.4f} sd={best['inner_score_sd']:.4f}")
    return best

def fit_radiomics_pipeline(dataset, config, train_indices, seed):
    """Refit candidates in inner-CV rank order and return the highest-ranked successful candidate."""
    if not bool(getattr(config, 'RADIOMICS_USE_INNER_TUNING', True)):
        raise ValueError('The public probability-fusion pipeline requires RADIOMICS_USE_INNER_TUNING=True.')
    selection = select_radiomics_params_by_inner_cv(dataset, config, train_indices, seed)
    attempts = []
    for rank, cand in enumerate(selection['ranked_valid_candidates'], start=1):
        cfg = _copy_config_with_radiomics_params(config=config, feature_config=cand['feature_config'], max_features=cand['max_features'], selector_method=cand['selector_method'], lasso_c=cand['lasso_c'])
        rec = {'refit_candidate_rank': rank, 'candidate_name': cand['name'], 'selector_name': cand['selector_name'], 'selector_method': cand['selector_method'], 'max_features': cand['max_features'], 'lasso_c': cand['lasso_c'] if cand['lasso_c'] is not None else np.nan, 'inner_score_mean': cand['inner_score_mean'], 'inner_score_sd': cand['inner_score_sd'], 'refit_success': 0, 'n_nonzero_coefficients': 0 if cand['selector_method'] == 'lasso' else np.nan, 'n_selected': 0, 'selector_reason': 'refit_failed', 'fallback_used': False, 'error_type': '', 'error_message': ''}
        try:
            pipe = _fit_radiomics_pipeline_fixed_config(dataset, cfg, train_indices, seed + 700000 + rank)
            rec.update({'refit_success': 1, 'n_nonzero_coefficients': pipe.get('n_nonzero_coefficients', np.nan), 'n_selected': pipe.get('n_selected', 0), 'selector_reason': pipe.get('selector_reason', '')})
            attempts.append(rec)
            pipe.update({'inner_best_candidate_name': cand['name'], 'inner_score_mean': cand['inner_score_mean'], 'inner_score_sd': cand['inner_score_sd'], 'inner_records': selection['all_inner_records'], 'ranked_valid_candidates': selection['ranked_valid_candidates'], 'refit_candidate_rank': rank, 'refit_attempt_records': attempts, 'selector_method': cand['selector_method'], 'selector_name': cand['selector_name']})
            print(f"[Radiomics Refit SELECTED] candidate={cand['name']}, rank={rank}, method={cand['selector_method']}, selected={pipe.get('n_selected')}, reason={pipe.get('selector_reason')}, fallback_used=False")
            return pipe
        except Exception as exc:
            rec.update({'selector_reason': 'zero_features' if isinstance(exc, NoRadiomicsFeaturesError) else 'refit_failed', 'error_type': type(exc).__name__, 'error_message': str(exc)})
            attempts.append(rec)
            print(f"[Radiomics Refit INVALID] rank={rank}, candidate={cand['name']}, error={type(exc).__name__}: {exc}. Try next candidate.")
    raise RuntimeError('Radiomics outer-training refit failed for all valid candidates.')
