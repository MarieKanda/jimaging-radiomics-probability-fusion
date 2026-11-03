"""Tabular dataset for Meta + Radiomics probability-level fusion.

The probability-fusion pipeline needs metadata, the target label, and a patient
identifier. Radiomics features are read separately from the configured CSV by
``radiomics_probability_fusion.py``. No image loading, image normalization,
augmentation or debug-image output is performed here.
"""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd

class _ArrayValue:
    """Minimal NumPy-backed value with the methods used by the CV pipeline."""

    def __init__(self, value):
        self._value = np.asarray(value)

    def numpy(self):
        return self._value.copy()

    def item(self):
        return self._value.item()

class BrainDataset:
    """Provide metadata, labels, and patient IDs for probability fusion.

    The constructor keeps the original call signature so that
    ``main_probability_fusion.py`` can use this class without behavioral changes
    outside the removal of image processing.
    """
    BASE_META_COLUMNS = ['Age', 'X', 'Y', 'Z', 'Calcification', 'mismatch']

    def __init__(self, csv_file, data_dir=None, modalities: Sequence[str] | None=None, config=None, use_edge_full: bool=False, task: str='1p19q', training: bool=True):
        del data_dir, modalities, use_edge_full, training
        if config is None:
            raise ValueError('config is required.')
        self.config = config
        self.task = str(task)
        self.df = pd.read_csv(Path(csv_file), low_memory=False)
        if 'patient_id' not in self.df.columns:
            if 'No' not in self.df.columns:
                raise ValueError("The label CSV must contain 'patient_id' or 'No'.")
            patient_numbers = pd.to_numeric(self.df['No'], errors='raise').astype(int)
            self.df['patient_id'] = patient_numbers.map(lambda value: f'patient{value:03d}')
        else:
            self.df['patient_id'] = self.df['patient_id'].astype(str).str.strip()
        if self.df['patient_id'].duplicated().any():
            duplicated = self.df.loc[self.df['patient_id'].duplicated(keep=False), 'patient_id'].astype(str).unique().tolist()
            raise ValueError(f'Duplicated patient IDs were found: {duplicated[:10]}')
        self.meta_cols = list(self.BASE_META_COLUMNS)
        target_column = self._target_column()
        required = ['patient_id', target_column, *self.meta_cols]
        missing = [column for column in required if column not in self.df.columns]
        if missing:
            raise ValueError(f'Required columns are missing from the label CSV: {missing}')
        self._meta_matrix = self.df[self.meta_cols].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32, copy=True)
        self._labels = pd.to_numeric(self.df[target_column], errors='raise').to_numpy(dtype=np.float32, copy=True)
        self._patient_ids = self.df['patient_id'].astype(str).to_numpy(copy=True)
        valid_labels = np.isin(self._labels, [0.0, 1.0])
        if not np.all(valid_labels):
            invalid = np.unique(self._labels[~valid_labels]).tolist()
            raise ValueError(f"Target column '{target_column}' contains values other than 0/1: {invalid}")
        print('Using meta columns:', self.meta_cols)
        print('Dataset mode: tabular Meta + Radiomics probability fusion')

    def _target_column(self) -> str:
        if self.task == 'IDH':
            return 'IDH_mutation'
        if self.task == '1p19q':
            return '1p19q_codeletion'
        raise ValueError("task must be 'IDH' or '1p19q'.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        empty_image_placeholder = np.empty(0, dtype=np.float32)
        meta = _ArrayValue(self._meta_matrix[idx].copy())
        label = _ArrayValue(np.float32(self._labels[idx]))
        patient_id = str(self._patient_ids[idx])
        return (empty_image_placeholder, meta, label, patient_id)
