# -*- coding: utf-8 -*-
"""Utilities required by Meta + Radiomics probability-level fusion."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

class NumericColumnImputer:
    """Small NumPy-based imputer for one numeric column.

    This avoids version-dependent behavior in older combinations of
    scikit-learn and SciPy while preserving training-fold-only fitting.
    """
    SUPPORTED_STRATEGIES = {'most_frequent', 'median', 'mean'}

    def __init__(self, strategy: str='most_frequent'):
        strategy = str(strategy).lower()
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(f'Unsupported imputation strategy: {strategy}. Choose from {sorted(self.SUPPORTED_STRATEGIES)}.')
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X):
        values = np.asarray(X, dtype=np.float64).reshape(-1)
        observed = values[np.isfinite(values)]
        if observed.size == 0:
            raise ValueError('Cannot fit an imputer because no observed values remain.')
        if self.strategy == 'most_frequent':
            unique, counts = np.unique(observed, return_counts=True)
            statistic = float(unique[np.argmax(counts)])
        elif self.strategy == 'median':
            statistic = float(np.median(observed))
        else:
            statistic = float(np.mean(observed))
        if not np.isfinite(statistic):
            raise ValueError('The learned imputation statistic is not finite.')
        self.statistics_ = np.asarray([statistic], dtype=np.float64)
        return self

    def transform(self, X):
        if self.statistics_ is None:
            raise RuntimeError('The imputer must be fitted before transform().')
        array = np.asarray(X, dtype=np.float64)
        original_shape = array.shape
        output = array.copy().reshape(-1)
        output[~np.isfinite(output)] = float(self.statistics_[0])
        return output.reshape(original_shape)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

@dataclass
class MetaMissingPipeline:
    meta_columns: List[str]
    missing_code_map: Dict[str, List[float]]
    strategy_map: Dict[str, str]
    allowed_missing_columns: List[str]
    allowed_values_map: Dict[str, List[float]]
    imputers: Dict[str, 'NumericColumnImputer']
    missing_count_train: Dict[str, int]
    imputation_values: Dict[str, float]

    def transform(self, X_raw: np.ndarray, stage: str='prediction') -> np.ndarray:
        """Apply missing-code conversion and training-fitted imputers."""
        X = _validate_matrix(X_raw, self.meta_columns, stage)
        X = _replace_missing_codes(X, self.meta_columns, self.missing_code_map)
        _validate_allowed_values(X=X, meta_columns=self.meta_columns, allowed_values_map=self.allowed_values_map, stage=stage, allow_nan=True)
        _reject_unexpected_missing(X=X, meta_columns=self.meta_columns, allowed_missing_columns=self.allowed_missing_columns, stage=stage)
        for column, imputer in self.imputers.items():
            j = _column_index(self.meta_columns, column)
            X[:, j] = imputer.transform(X[:, j].reshape(-1, 1)).ravel()
        if not np.all(np.isfinite(X)):
            raise ValueError(f"Non-finite meta values remain after imputation at stage='{stage}': {_nonfinite_summary(X, self.meta_columns)}")
        _validate_allowed_values(X=X, meta_columns=self.meta_columns, allowed_values_map=self.allowed_values_map, stage=stage, allow_nan=False)
        return X.astype(np.float64, copy=False)

    def audit_dict(self) -> Dict[str, Any]:
        return {'meta_columns': list(self.meta_columns), 'missing_code_map': {k: list(v) for k, v in self.missing_code_map.items()}, 'strategy_map': dict(self.strategy_map), 'allowed_missing_columns': list(self.allowed_missing_columns), 'allowed_values_map': {k: list(v) for k, v in self.allowed_values_map.items()}, 'missing_count_train': dict(self.missing_count_train), 'imputation_values': dict(self.imputation_values)}

def fit_meta_missing_pipeline(X_train_raw: np.ndarray, meta_columns: Sequence[str], config: Any) -> MetaMissingPipeline:
    """Fit missing-data handling using the supplied training fold only."""
    meta_columns = list(meta_columns)
    X_train = _validate_matrix(X_train_raw, meta_columns, 'training')
    missing_code_map = _normalize_missing_code_map(getattr(config, 'META_MISSING_CODE_MAP', {'Calcification': [-1.0]}))
    strategy_map = dict(getattr(config, 'META_IMPUTATION_STRATEGY_MAP', {'Calcification': 'most_frequent'}))
    allowed_missing_columns = list(getattr(config, 'META_ALLOW_MISSING_COLUMNS', ['Calcification']))
    allowed_values_map = _normalize_allowed_values_map(getattr(config, 'META_ALLOWED_VALUES_MAP', {'Calcification': [0.0, 0.5, 1.0]}))
    for group in (missing_code_map, strategy_map, allowed_values_map):
        for column in group:
            _column_index(meta_columns, column)
    for column in allowed_missing_columns:
        _column_index(meta_columns, column)
    if set(strategy_map) - set(allowed_missing_columns):
        raise ValueError(f'Imputation strategies were specified for columns not listed in META_ALLOW_MISSING_COLUMNS: {sorted(set(strategy_map) - set(allowed_missing_columns))}')
    X_train_nan = _replace_missing_codes(X_train, meta_columns, missing_code_map)
    _validate_allowed_values(X_train_nan, meta_columns, allowed_values_map, 'training', allow_nan=True)
    _reject_unexpected_missing(X_train_nan, meta_columns, allowed_missing_columns, 'training')
    missing_count_train = {column: int(np.isnan(X_train_nan[:, j]).sum()) for j, column in enumerate(meta_columns)}
    imputers: Dict[str, 'NumericColumnImputer'] = {}
    imputation_values: Dict[str, float] = {}
    for column in allowed_missing_columns:
        j = _column_index(meta_columns, column)
        values = X_train_nan[:, j].reshape(-1, 1)
        if int(np.isfinite(values).sum()) == 0:
            raise ValueError(f"Column '{column}' has no observed values in this training fold.")
        strategy = strategy_map.get(column, 'most_frequent')
        if strategy not in {'most_frequent', 'median', 'mean'}:
            raise ValueError(f"Unsupported imputation strategy for '{column}': {strategy}")
        imputer = NumericColumnImputer(strategy=strategy)
        imputer.fit(values)
        statistic = float(imputer.statistics_[0])
        if not np.isfinite(statistic):
            raise ValueError(f"Non-finite imputation value for '{column}'.")
        _validate_single_allowed_value(column=column, value=statistic, allowed_values_map=allowed_values_map, stage='imputer_fit')
        imputers[column] = imputer
        imputation_values[column] = statistic
    pipeline = MetaMissingPipeline(meta_columns=meta_columns, missing_code_map=missing_code_map, strategy_map=strategy_map, allowed_missing_columns=allowed_missing_columns, allowed_values_map=allowed_values_map, imputers=imputers, missing_count_train=missing_count_train, imputation_values=imputation_values)
    pipeline.transform(X_train_raw, stage='training_post_fit_check')
    return pipeline

def _validate_matrix(X_raw, meta_columns, stage):
    X = np.asarray(X_raw, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Meta matrix must be two-dimensional at stage='{stage}', received shape={X.shape}.")
    if X.shape[1] != len(meta_columns):
        raise ValueError(f"Meta column mismatch at stage='{stage}': matrix has {X.shape[1]} columns but {len(meta_columns)} names were supplied.")
    return X.copy()

def _column_index(meta_columns, column):
    if column not in meta_columns:
        raise KeyError(f"Meta column '{column}' was not found. Available columns: {list(meta_columns)}")
    return list(meta_columns).index(column)

def _normalize_missing_code_map(source: Dict[str, Iterable[float]]):
    normalized = {}
    for column, codes in source.items():
        if np.isscalar(codes):
            codes = [codes]
        normalized[column] = [float(code) for code in codes]
    return normalized

def _normalize_allowed_values_map(source: Dict[str, Iterable[float]]):
    normalized = {}
    for column, values in source.items():
        if np.isscalar(values):
            values = [values]
        vals = [float(value) for value in values]
        if not vals:
            raise ValueError(f"Allowed-value list for '{column}' is empty.")
        if not np.all(np.isfinite(vals)):
            raise ValueError(f"Allowed values for '{column}' must be finite: {vals}")
        normalized[column] = vals
    return normalized

def _replace_missing_codes(X, meta_columns, missing_code_map):
    out = np.asarray(X, dtype=np.float64).copy()
    for column, codes in missing_code_map.items():
        j = _column_index(meta_columns, column)
        for code in codes:
            out[np.isclose(out[:, j], code, rtol=0.0, atol=1e-12, equal_nan=False), j] = np.nan
    return out

def _reject_unexpected_missing(X, meta_columns, allowed_missing_columns, stage):
    allowed = set(allowed_missing_columns)
    unexpected = {column: int(np.isnan(X[:, j]).sum()) for j, column in enumerate(meta_columns) if np.isnan(X[:, j]).any() and column not in allowed}
    if unexpected:
        raise ValueError(f"Unexpected missing meta values at stage='{stage}': {unexpected}. Allowed missing columns: {sorted(allowed)}")
    inf_summary = {column: int(np.isinf(X[:, j]).sum()) for j, column in enumerate(meta_columns) if np.isinf(X[:, j]).any()}
    if inf_summary:
        raise ValueError(f"Infinite meta values at stage='{stage}': {inf_summary}")

def _validate_allowed_values(X, meta_columns, allowed_values_map, stage, allow_nan):
    """Validate categorical/ordinal observed codes, including 0.5."""
    for column, allowed_values in allowed_values_map.items():
        j = _column_index(meta_columns, column)
        values = np.asarray(X[:, j], dtype=np.float64)
        if allow_nan:
            values_to_check = values[np.isfinite(values)]
        else:
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Column '{column}' contains NaN or Inf at stage='{stage}'.")
            values_to_check = values
        valid = np.zeros(values_to_check.shape, dtype=bool)
        for allowed in allowed_values:
            valid |= np.isclose(values_to_check, allowed, rtol=0.0, atol=1e-12)
        if not np.all(valid):
            invalid_values = sorted(np.unique(values_to_check[~valid]).astype(float).tolist())
            raise ValueError(f"Column '{column}' contains unexpected observed values at stage='{stage}': {invalid_values}. Allowed values: {list(allowed_values)}")

def _validate_single_allowed_value(column, value, allowed_values_map, stage):
    if column not in allowed_values_map:
        return
    allowed = allowed_values_map[column]
    if not any((np.isclose(value, v, rtol=0.0, atol=1e-12) for v in allowed)):
        raise ValueError(f"Learned imputation value {value} for '{column}' is invalid at stage='{stage}'. Allowed values: {allowed}")

def _nonfinite_summary(X, meta_columns):
    return {column: int((~np.isfinite(X[:, j])).sum()) for j, column in enumerate(meta_columns) if (~np.isfinite(X[:, j])).any()}
