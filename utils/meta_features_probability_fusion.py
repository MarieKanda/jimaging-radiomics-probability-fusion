"""Utilities required by Meta + Radiomics probability-level fusion."""
from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
import numpy as np
VALID_MODES = {'base', 'interaction', 'full'}

def transform_meta(meta, mode: str='base', feature_names: Optional[Sequence[str]]=None) -> Tuple[np.ndarray, List[str]]:
    """Transform one patient's meta-information into model features.

Parameters
----------
meta
    One-dimensional array-like meta-information for one patient.
mode
    ``"base"`` returns original features only. ``"interaction"`` adds all
    pairwise interactions. ``"full"`` additionally adds squared terms.
feature_names
    Names corresponding to the input values; the counts must match.

Returns
-------
transformed
    One-dimensional transformed array with ``float32`` dtype.
transformed_names
    Names of the transformed features in output order.

Notes
-----
This function does not impute missing values. NaN or infinite values must be
handled by the training-fold-fitted missing-data pipeline before transformation."""
    mode = str(mode).lower()
    if mode not in VALID_MODES:
        raise ValueError(f'Invalid META_FEATURE_MODE: {mode}. Choose from {sorted(VALID_MODES)}.')
    meta_array = np.asarray(meta, dtype=np.float64).reshape(-1)
    if meta_array.size == 0:
        raise ValueError('meta is empty.')
    if feature_names is None:
        input_names = [f'f{i}' for i in range(meta_array.size)]
    else:
        input_names = [str(name) for name in feature_names]
    if len(input_names) != meta_array.size:
        raise ValueError(f'The number of feature names does not match the number of meta values: n_values={meta_array.size}, n_names={len(input_names)}.')
    if len(set(input_names)) != len(input_names):
        duplicated = sorted({name for name in input_names if input_names.count(name) > 1})
        raise ValueError(f'Duplicated input feature names were found: {duplicated}')
    nonfinite_mask = ~np.isfinite(meta_array)
    if np.any(nonfinite_mask):
        invalid = {input_names[i]: float(meta_array[i]) for i in np.where(nonfinite_mask)[0]}
        raise ValueError(f'NaN or Inf remains in meta-information before feature transformation. Missing values must be imputed using the training-fold-fitted pipeline. Invalid values: {invalid}')
    features = [meta_array.copy()]
    output_names = list(input_names)
    if mode == 'base':
        transformed = meta_array.copy()
    elif mode == 'interaction':
        for i in range(meta_array.size):
            for j in range(i + 1, meta_array.size):
                value = meta_array[i] * meta_array[j]
                features.append(np.asarray([value], dtype=np.float64))
                output_names.append(f'{input_names[i]}*{input_names[j]}')
        transformed = np.concatenate([feature.reshape(-1) for feature in features])
    elif mode == 'full':
        for i in range(meta_array.size):
            value = meta_array[i] ** 2
            features.append(np.asarray([value], dtype=np.float64))
            output_names.append(f'{input_names[i]}^2')
        for i in range(meta_array.size):
            for j in range(i + 1, meta_array.size):
                value = meta_array[i] * meta_array[j]
                features.append(np.asarray([value], dtype=np.float64))
                output_names.append(f'{input_names[i]}*{input_names[j]}')
        transformed = np.concatenate([feature.reshape(-1) for feature in features])
    else:
        raise RuntimeError(f'Unexpected mode: {mode}')
    transformed = np.asarray(transformed, dtype=np.float64).reshape(-1)
    if transformed.size != len(output_names):
        raise RuntimeError(f'Feature transformation produced inconsistent output lengths: n_values={transformed.size}, n_names={len(output_names)}.')
    if len(set(output_names)) != len(output_names):
        duplicated = sorted({name for name in output_names if output_names.count(name) > 1})
        raise RuntimeError(f'Duplicated transformed feature names were found: {duplicated}')
    if not np.all(np.isfinite(transformed)):
        invalid_indices = np.where(~np.isfinite(transformed))[0]
        invalid = {output_names[i]: float(transformed[i]) for i in invalid_indices}
        raise ValueError(f'Feature transformation produced NaN or Inf: {invalid}')
    return (transformed.astype(np.float32, copy=False), output_names)
