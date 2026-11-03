"""Leakage-aware repeated nested cross-validation for Meta and Radiomics probability-level fusion. The public version supports logistic regression for the metadata model only."""
import os
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import f1_score, average_precision_score, balanced_accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from utils.meta_features_probability_fusion import transform_meta
from utils.radiomics_probability_fusion import fit_radiomics_pipeline, predict_radiomics_prob, _safe_cv_splits as _safe_cv_splits_radiomics
from utils.meta_missing_probability_fusion import fit_meta_missing_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

def _prob_fusion_logit(p, eps=1e-05):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def _prob_fusion_compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    if len(np.unique(y_true)) >= 2:
        out['auc'] = float(roc_auc_score(y_true, y_prob))
        out['ap'] = float(average_precision_score(y_true, y_prob))
    else:
        out['auc'] = np.nan
        out['ap'] = np.nan
    out['acc'] = float(accuracy_score(y_true, y_pred))
    out['bal_acc'] = float(balanced_accuracy_score(y_true, y_pred))
    out['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    out['f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    out['tn'] = int(tn)
    out['fp'] = int(fp)
    out['fn'] = int(fn)
    out['tp'] = int(tp)
    out['sensitivity'] = float(tp / (tp + fn + 1e-12))
    out['specificity'] = float(tn / (tn + fp + 1e-12))
    out['threshold'] = float(threshold)
    return out

def _prob_fusion_select_threshold(y_true, y_prob, mode='fixed_0.5', fixed_threshold=0.5, th_min=0.1, th_max=0.9, th_step=0.01):
    """Select a decision threshold using outer-training pooled OOF predictions only."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError('y_true and y_prob must be one-dimensional arrays.')
    if len(y_true) != len(y_prob):
        raise ValueError('y_true and y_prob must have the same length.')
    if len(y_true) == 0:
        raise ValueError('Threshold selection received no samples.')
    if len(np.unique(y_true)) < 2:
        raise ValueError('Threshold selection requires both classes.')
    if not np.all(np.isfinite(y_prob)):
        raise ValueError('Threshold selection probabilities contain NaN/Inf.')
    if th_step <= 0:
        raise ValueError('th_step must be positive.')
    if th_min > th_max:
        raise ValueError('th_min must not exceed th_max.')
    if mode in ['fixed', 'fixed_0.5']:
        return (float(fixed_threshold), np.nan)
    n_steps = int(np.floor((th_max - th_min) / th_step + 1e-12))
    grid = th_min + np.arange(n_steps + 1, dtype=float) * th_step
    if grid[-1] < th_max - 1e-12:
        grid = np.append(grid, float(th_max))
    grid = np.unique(np.round(grid, 12))
    scored = []
    for th in grid:
        y_pred = (y_prob >= th).astype(int)
        if mode == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        elif mode == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif mode in ['f1', 'binary_f1']:
            score = f1_score(y_true, y_pred, zero_division=0)
        elif mode in ['macro_f1', 'f1_macro']:
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif mode == 'youden':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            sens = tp / (tp + fn + 1e-12)
            spec = tn / (tn + fp + 1e-12)
            score = sens + spec - 1.0
        else:
            raise ValueError(f'Unknown threshold mode: {mode}')
        scored.append((float(th), float(score)))
    best_score = max((score for _, score in scored))
    tied_thresholds = [th for th, score in scored if np.isclose(score, best_score, rtol=0.0, atol=1e-12)]
    best_th = min(tied_thresholds, key=lambda th: (abs(th - float(fixed_threshold)), th))
    return (float(best_th), float(best_score))

def _fit_predict_meta_probs_for_outer_fold(dataset, config, outer_train_idx, outer_test_idx, seed, fold):
    """
    Generate leak-free meta probabilities for one outer fold.

    outer train:
        OOF probabilities. Each meta imputer/model is fitted
        without the corresponding validation samples.

    outer test:
        Probabilities from a meta imputer/model fitted using
        the complete outer-training fold only.
    """
    _, y_outer_train, pid_outer_train = _collect_meta_arrays_from_indices(dataset, outer_train_idx)
    y_outer_train = np.asarray(y_outer_train).astype(int)
    pid_outer_train = np.asarray(pid_outer_train)
    n_splits = min(5, _safe_cv_splits(y_outer_train, requested_cv=5))
    if n_splits < 2:
        raise ValueError('Meta OOF probability requires at least 2 valid CV splits.')
    oof_prob = np.full(len(outer_train_idx), np.nan, dtype=float)
    outer_train_idx_arr = np.asarray(outer_train_idx)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + fold * 1000 + 501)
    for inner_fold, (tr_pos, va_pos) in enumerate(skf.split(np.zeros(len(outer_train_idx_arr)), y_outer_train)):
        tr_idx = outer_train_idx_arr[tr_pos].tolist()
        va_idx = outer_train_idx_arr[va_pos].tolist()
        model, scaler, selected_idx, selected_names, _, _, missing_pipeline = _fit_meta_model_leak_free(dataset=dataset, config=config, train_indices=tr_idx, seed=seed + fold * 1000 + inner_fold)
        prob_va, _, _, _ = _predict_meta_prob_with_pipeline(dataset=dataset, config=config, pred_indices=va_idx, fitted_model=model, scaler=scaler, selected_idx=selected_idx, meta_missing_pipeline=missing_pipeline)
        oof_prob[va_pos] = prob_va
    if np.isnan(oof_prob).any():
        raise RuntimeError('Meta OOF probability contains NaN.')
    final_model, final_scaler, final_selected_idx, final_selected_names, _, _, final_missing_pipeline = _fit_meta_model_leak_free(dataset=dataset, config=config, train_indices=outer_train_idx, seed=seed + fold * 1000 + 777)
    test_prob, y_test, pid_test, _ = _predict_meta_prob_with_pipeline(dataset=dataset, config=config, pred_indices=outer_test_idx, fitted_model=final_model, scaler=final_scaler, selected_idx=final_selected_idx, meta_missing_pipeline=final_missing_pipeline)
    return {'train_pid': pid_outer_train, 'train_y': y_outer_train, 'train_prob': oof_prob, 'test_pid': np.asarray(pid_test), 'test_y': np.asarray(y_test).astype(int), 'test_prob': np.asarray(test_prob, dtype=float), 'selected_names': final_selected_names, 'missing_count_train': final_missing_pipeline.missing_count_train if final_missing_pipeline is not None else {}, 'imputation_values': final_missing_pipeline.imputation_values if final_missing_pipeline is not None else {}}

def _unpack_radiomics_prediction(pred_result):
    """Normalize supported Radiomics prediction return formats."""
    if not isinstance(pred_result, tuple):
        raise TypeError('predict_radiomics_prob() must return a tuple: (prob, y_true, pids) or (prob, y_true, pids, X_selected).')
    if len(pred_result) == 3:
        prob, y_true, pids = pred_result
        X_selected = None
    elif len(pred_result) == 4:
        prob, y_true, pids, X_selected = pred_result
    else:
        raise ValueError(f'Unexpected number of return values from predict_radiomics_prob(): {len(pred_result)}. Expected 3 or 4.')
    return (prob, y_true, pids, X_selected)

def _fit_predict_radiomics_probs_for_outer_fold(dataset, config, outer_train_idx, outer_test_idx, seed, fold):
    """Generate leak-free Radiomics OOF and outer-test probabilities."""
    pids_train, y_train = ([], [])
    for i in outer_train_idx:
        _, _, y, pid = dataset[i]
        pids_train.append(str(pid))
        y_train.append(int(y.item() if hasattr(y, 'item') else y))
    pids_train = np.asarray(pids_train).astype(str)
    y_train = np.asarray(y_train).astype(int)
    n_splits = min(5, _safe_cv_splits_radiomics(y_train, requested_cv=5))
    if n_splits < 2:
        raise ValueError('Radiomics OOF probability requires at least 2 valid CV splits.')
    oof_prob = np.full(len(outer_train_idx), np.nan, dtype=float)
    outer_train_idx_arr = np.asarray(outer_train_idx)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + fold * 1000 + 101)
    selected_records = []
    inner_tuning_records = []
    for inner_fold, (tr_pos, va_pos) in enumerate(skf.split(np.zeros(len(outer_train_idx_arr)), y_train)):
        tr_idx = outer_train_idx_arr[tr_pos].tolist()
        va_idx = outer_train_idx_arr[va_pos].tolist()
        pipe = fit_radiomics_pipeline(dataset=dataset, config=config, train_indices=tr_idx, seed=seed + fold * 1000 + inner_fold)
        pred_result = predict_radiomics_prob(dataset=dataset, config=config, pred_indices=va_idx, pipeline=pipe)
        prob_va, y_va, pid_va, _ = _unpack_radiomics_prediction(pred_result)
        oof_prob[va_pos] = np.asarray(prob_va, dtype=float)
        selected_records.append({'seed': seed, 'fold': fold, 'inner_fold': inner_fold, 'stage': 'outer_train_oof', 'selected_features': '|'.join(pipe.get('selected_feature_names', [])), 'n_selected': len(pipe.get('selected_feature_names', [])), 'feature_config_name': pipe.get('feature_config_name', ''), 'inner_best_candidate_name': pipe.get('inner_best_candidate_name', ''), 'selector_method': pipe.get('selector_method', ''), 'selector_name': pipe.get('selector_name', ''), 'inner_score_mean': pipe.get('inner_score_mean', np.nan), 'inner_score_sd': pipe.get('inner_score_sd', np.nan), 'lasso_c': pipe.get('lasso_c', np.nan), 'max_features': pipe.get('max_features', np.nan), 'feature_classes': pipe.get('feature_classes', ''), 'log_sigmas': pipe.get('log_sigmas', ''), 'wavelet_types': pipe.get('wavelet_types', ''), 'n_nonzero_coefficients': pipe.get('n_nonzero_coefficients', np.nan), 'selector_reason': pipe.get('selector_reason', ''), 'fallback_used': bool(pipe.get('fallback_used', False)), 'refit_candidate_rank': pipe.get('refit_candidate_rank', np.nan)})
        for rec in pipe.get('inner_records', []):
            r = dict(rec)
            r.update({'seed': seed, 'fold': fold, 'outer_train_oof_inner_fold': inner_fold, 'stage': 'outer_train_oof_inner_tuning'})
            inner_tuning_records.append(r)
        for rec in pipe.get('refit_attempt_records', []):
            r = dict(rec)
            r.update({'seed': seed, 'fold': fold, 'outer_train_oof_inner_fold': inner_fold, 'stage': 'outer_train_oof_refit_attempt'})
            inner_tuning_records.append(r)
    if np.isnan(oof_prob).any():
        raise RuntimeError('Radiomics OOF probability contains NaN.')
    final_pipe = fit_radiomics_pipeline(dataset=dataset, config=config, train_indices=outer_train_idx, seed=seed + fold * 1000 + 701)
    pred_result = predict_radiomics_prob(dataset=dataset, config=config, pred_indices=outer_test_idx, pipeline=final_pipe)
    test_prob, y_test, pid_test, _ = _unpack_radiomics_prediction(pred_result)
    selected_records.append({'seed': seed, 'fold': fold, 'inner_fold': -1, 'stage': 'outer_test_model', 'selected_features': '|'.join(final_pipe.get('selected_feature_names', [])), 'n_selected': len(final_pipe.get('selected_feature_names', [])), 'feature_config_name': final_pipe.get('feature_config_name', ''), 'inner_best_candidate_name': final_pipe.get('inner_best_candidate_name', ''), 'selector_method': final_pipe.get('selector_method', ''), 'selector_name': final_pipe.get('selector_name', ''), 'inner_score_mean': final_pipe.get('inner_score_mean', np.nan), 'inner_score_sd': final_pipe.get('inner_score_sd', np.nan), 'lasso_c': final_pipe.get('lasso_c', np.nan), 'max_features': final_pipe.get('max_features', np.nan), 'feature_classes': final_pipe.get('feature_classes', ''), 'log_sigmas': final_pipe.get('log_sigmas', ''), 'wavelet_types': final_pipe.get('wavelet_types', ''), 'n_nonzero_coefficients': final_pipe.get('n_nonzero_coefficients', np.nan), 'selector_reason': final_pipe.get('selector_reason', ''), 'fallback_used': bool(final_pipe.get('fallback_used', False)), 'refit_candidate_rank': final_pipe.get('refit_candidate_rank', np.nan)})
    for rec in final_pipe.get('inner_records', []):
        r = dict(rec)
        r.update({'seed': seed, 'fold': fold, 'outer_train_oof_inner_fold': -1, 'stage': 'outer_test_model_inner_tuning'})
        inner_tuning_records.append(r)
    for rec in final_pipe.get('refit_attempt_records', []):
        r = dict(rec)
        r.update({'seed': seed, 'fold': fold, 'outer_train_oof_inner_fold': -1, 'stage': 'outer_test_model_refit_attempt'})
        inner_tuning_records.append(r)
    return {'train_pid': pids_train, 'train_y': y_train, 'train_prob': oof_prob, 'test_pid': np.asarray(pid_test).astype(str), 'test_y': np.asarray(y_test).astype(int), 'test_prob': np.asarray(test_prob, dtype=float), 'selected_feature_names': final_pipe.get('selected_feature_names', []), 'selected_records': selected_records, 'inner_tuning_records': inner_tuning_records}

def _make_probability_fusion_candidates(config):
    """Return the configured weighted-average and logit-stacking candidates."""
    candidates = getattr(config, 'PROB_FUSION_METHODS', None)
    if candidates is None:
        candidates = [{'name': 'weighted_m0.4_r0.6', 'type': 'weighted_average', 'w_meta': 0.4, 'w_rad': 0.6}, {'name': 'weighted_m0.5_r0.5', 'type': 'weighted_average', 'w_meta': 0.5, 'w_rad': 0.5}, {'name': 'weighted_m0.6_r0.4', 'type': 'weighted_average', 'w_meta': 0.6, 'w_rad': 0.4}, {'name': 'logit_stacking_C0.01', 'type': 'logit_stacking', 'C': 0.01}, {'name': 'logit_stacking_C0.1', 'type': 'logit_stacking', 'C': 0.1}]
    allowed_types = {'weighted_average', 'logit_stacking'}
    cleaned = []
    for candidate in candidates:
        candidate = dict(candidate)
        if candidate.get('type') not in allowed_types:
            raise ValueError('Invalid configuration or input detected in the public probability-fusion workflow.')
        cleaned.append(candidate)
    return cleaned

def _fit_apply_probability_fusion_candidate(candidate, y_train, meta_train, rad_train, meta_test, rad_test, seed):
    """Fit or apply one probability-fusion candidate without using outer-test labels."""
    y_train = np.asarray(y_train).astype(int)
    meta_train = np.asarray(meta_train, dtype=float)
    rad_train = np.asarray(rad_train, dtype=float)
    meta_test = np.asarray(meta_test, dtype=float)
    rad_test = np.asarray(rad_test, dtype=float)
    candidate_type = candidate['type']
    if candidate_type == 'weighted_average':
        w_meta = float(candidate['w_meta'])
        w_rad = float(candidate['w_rad'])
        denominator = w_meta + w_rad
        if denominator <= 0:
            raise ValueError('w_meta + w_rad must be positive.')
        w_meta = w_meta / denominator
        w_rad = w_rad / denominator
        train_prob = w_meta * meta_train + w_rad * rad_train
        test_prob = w_meta * meta_test + w_rad * rad_test
        return (train_prob, test_prob, None)
    if candidate_type == 'logit_stacking':
        X_train = np.column_stack([_prob_fusion_logit(meta_train), _prob_fusion_logit(rad_train)])
        X_test = np.column_stack([_prob_fusion_logit(meta_test), _prob_fusion_logit(rad_test)])
        model = LogisticRegression(penalty='l2', C=float(candidate['C']), solver='liblinear', class_weight='balanced', max_iter=2000, random_state=seed)
        model.fit(X_train, y_train)
        train_prob = model.predict_proba(X_train)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        return (train_prob, test_prob, model)
    raise ValueError(f'Unknown probability fusion type: {candidate_type}')

def _score_probability_fusion_candidate(y_true, y_prob, metric='auc'):
    """Score a fusion candidate using the configured training-only metric."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    if metric == 'auc':
        return float(roc_auc_score(y_true, y_prob))
    if metric in ['ap', 'average_precision']:
        return float(average_precision_score(y_true, y_prob))
    raise ValueError(f'Unknown selection metric: {metric}')

def _make_logit_stacking_oof_probability(candidate, y_train, meta_train_prob, rad_train_prob, seed, requested_folds=5):
    """Generate stacker-level OOF probabilities within the outer-training set."""
    y_train = np.asarray(y_train).astype(int)
    meta_train_prob = np.asarray(meta_train_prob, dtype=float)
    rad_train_prob = np.asarray(rad_train_prob, dtype=float)
    _, class_counts = np.unique(y_train, return_counts=True)
    n_splits = min(int(requested_folds), int(np.min(class_counts)))
    if n_splits < 2:
        raise ValueError('logit stacking OOF selection requires at least 2 folds.')
    X_stack = np.column_stack([_prob_fusion_logit(meta_train_prob), _prob_fusion_logit(rad_train_prob)])
    oof_prob = np.full(len(y_train), np.nan, dtype=float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for subfold, (train_pos, valid_pos) in enumerate(skf.split(X_stack, y_train)):
        model = LogisticRegression(penalty='l2', C=float(candidate['C']), solver='liblinear', class_weight='balanced', max_iter=2000, random_state=seed + subfold + 1)
        model.fit(X_stack[train_pos], y_train[train_pos])
        oof_prob[valid_pos] = model.predict_proba(X_stack[valid_pos])[:, 1]
    if np.isnan(oof_prob).any():
        raise RuntimeError('logit stacking OOF probability contains NaN.')
    return oof_prob

def _select_probability_fusion_candidate_within_family(family_name, family_candidates, y_train, meta_train_prob, rad_train_prob, config, seed):
    """Select one candidate within a prespecified fusion family."""
    metric = getattr(config, 'PROB_FUSION_SELECTION_METRIC', 'auc')
    stacking_folds = int(getattr(config, 'PROB_FUSION_STACKING_SELECTION_FOLDS', 5))
    best_candidate = None
    best_score = -np.inf
    selection_records = []
    for candidate_order, candidate in enumerate(family_candidates):
        if family_name == 'weighted':
            selection_prob, _, _ = _fit_apply_probability_fusion_candidate(candidate=candidate, y_train=y_train, meta_train=meta_train_prob, rad_train=rad_train_prob, meta_test=meta_train_prob, rad_test=rad_train_prob, seed=seed + candidate_order)
        elif family_name == 'logit_stacking':
            selection_prob = _make_logit_stacking_oof_probability(candidate=candidate, y_train=y_train, meta_train_prob=meta_train_prob, rad_train_prob=rad_train_prob, seed=seed + 100 + candidate_order, requested_folds=stacking_folds)
        else:
            raise ValueError(f'Unknown family_name: {family_name}')
        score = _score_probability_fusion_candidate(y_true=y_train, y_prob=selection_prob, metric=metric)
        selection_records.append({'family': family_name, 'candidate': candidate['name'], 'selection_metric': metric, 'selection_score_oof': score})
        if np.isfinite(score) and score > best_score + 1e-12:
            best_score = score
            best_candidate = candidate
    if best_candidate is None:
        raise RuntimeError(f'No valid candidate for family={family_name}')
    for record in selection_records:
        record['selected'] = int(record['candidate'] == best_candidate['name'])
    return (best_candidate, float(best_score), selection_records)

def _append_probability_fusion_result(display_name, selected_candidate_name, candidate_type, y_train, y_test, pid_test, meta_test_prob, rad_test_prob, train_prob, test_prob, config, seed, fold, pred_dir):
    """Evaluate and save one model result for an outer fold."""
    threshold_mode = getattr(config, 'PROB_FUSION_THRESHOLD_MODE', 'fixed_0.5')
    fixed_threshold = float(getattr(config, 'PROB_FUSION_FIXED_THRESHOLD', 0.5))
    threshold, threshold_score = _prob_fusion_select_threshold(y_true=y_train, y_prob=train_prob, mode=threshold_mode, fixed_threshold=fixed_threshold, th_min=float(getattr(config, 'PROB_FUSION_THRESHOLD_MIN', 0.1)), th_max=float(getattr(config, 'PROB_FUSION_THRESHOLD_MAX', 0.9)), th_step=float(getattr(config, 'PROB_FUSION_THRESHOLD_STEP', 0.01)))
    metrics = _prob_fusion_compute_metrics(y_true=y_test, y_prob=test_prob, threshold=threshold)
    metric_row = {'model_type': 'probability_fusion', 'candidate': display_name, 'candidate_type': candidate_type, 'selected_candidate': selected_candidate_name, 'seed': seed, 'fold': fold, 'threshold_mode': threshold_mode, 'threshold_selection_source': 'outer_training_pooled_oof', 'threshold_score_oof': threshold_score}
    metric_row.update(metrics)
    prediction_df = pd.DataFrame({'patient_id': np.asarray(pid_test).astype(str), 'y_true': np.asarray(y_test).astype(int), 'meta_prob': np.asarray(meta_test_prob, dtype=float), 'radiomics_prob': np.asarray(rad_test_prob, dtype=float), 'y_prob': np.asarray(test_prob, dtype=float), 'threshold': threshold, 'threshold_mode': threshold_mode, 'threshold_selection_source': 'outer_training_pooled_oof', 'threshold_score_oof': threshold_score, 'y_pred': (np.asarray(test_prob) >= threshold).astype(int), 'seed': seed, 'fold': fold, 'candidate': display_name, 'selected_candidate': selected_candidate_name})
    prediction_path = os.path.join(pred_dir, f'pred_{display_name}_seed{seed}_fold{fold}.csv')
    prediction_df.to_csv(prediction_path, index=False)
    return (metric_row, prediction_df)

def run_probability_late_fusion_cv(dataset, config, save_dir):
    """Run repeated nested CV for Meta and Radiomics probability fusion."""
    os.makedirs(save_dir, exist_ok=True)
    pred_dir = os.path.join(save_dir, 'predictions_probability_fusion')
    os.makedirs(pred_dir, exist_ok=True)
    candidates = _make_probability_fusion_candidates(config)
    weighted_candidates = [c for c in candidates if c['type'] == 'weighted_average']
    logit_candidates = [c for c in candidates if c['type'] == 'logit_stacking']
    if len(weighted_candidates) != 3:
        raise ValueError('weighted_average candidates must be exactly 3.')
    if len(logit_candidates) != 2:
        raise ValueError('logit_stacking candidates must be exactly 2.')
    all_metric_rows = []
    all_prediction_rows = []
    all_selection_rows = []
    all_selected_rad_rows = []
    all_inner_tuning_rows = []
    X = np.arange(len(dataset))
    if dataset.task == 'IDH':
        y = dataset.df['IDH_mutation'].values.astype(int)
    else:
        y = dataset.df['1p19q_codeletion'].values.astype(int)
    for seed in config.SEEDS:
        config.set_seed(seed)
        skf_outer = StratifiedKFold(n_splits=config.OUTER_FOLDS, shuffle=True, random_state=seed)
        for fold, (outer_train_idx, outer_test_idx) in enumerate(skf_outer.split(X, y)):
            print(f'\n===== Probability Fusion | seed={seed}, fold={fold} =====')
            outer_train_idx = list(outer_train_idx)
            outer_test_idx = list(outer_test_idx)
            meta_info = _fit_predict_meta_probs_for_outer_fold(dataset=dataset, config=config, outer_train_idx=outer_train_idx, outer_test_idx=outer_test_idx, seed=seed, fold=fold)
            rad_info = _fit_predict_radiomics_probs_for_outer_fold(dataset=dataset, config=config, outer_train_idx=outer_train_idx, outer_test_idx=outer_test_idx, seed=seed, fold=fold)
            all_selected_rad_rows.extend(rad_info.get('selected_records', []))
            all_inner_tuning_rows.extend(rad_info.get('inner_tuning_records', []))
            if not np.array_equal(meta_info['train_pid'].astype(str), rad_info['train_pid'].astype(str)):
                raise RuntimeError('Train PID order mismatch.')
            if not np.array_equal(meta_info['test_pid'].astype(str), rad_info['test_pid'].astype(str)):
                raise RuntimeError('Test PID order mismatch.')
            if not np.array_equal(meta_info['train_y'].astype(int), rad_info['train_y'].astype(int)):
                raise RuntimeError('Train labels mismatch.')
            if not np.array_equal(meta_info['test_y'].astype(int), rad_info['test_y'].astype(int)):
                raise RuntimeError('Test labels mismatch.')
            y_train = meta_info['train_y']
            y_test = meta_info['test_y']
            pid_test = meta_info['test_pid'].astype(str)
            meta_train_prob = meta_info['train_prob']
            rad_train_prob = rad_info['train_prob']
            meta_test_prob = meta_info['test_prob']
            rad_test_prob = rad_info['test_prob']
            baseline_data = [('meta_only_prob', meta_train_prob, meta_test_prob), ('radiomics_only_prob', rad_train_prob, rad_test_prob)]
            for name, train_prob, test_prob in baseline_data:
                row, pred_df = _append_probability_fusion_result(display_name=name, selected_candidate_name=name, candidate_type='baseline', y_train=y_train, y_test=y_test, pid_test=pid_test, meta_test_prob=meta_test_prob, rad_test_prob=rad_test_prob, train_prob=train_prob, test_prob=test_prob, config=config, seed=seed, fold=fold, pred_dir=pred_dir)
                all_metric_rows.append(row)
                all_prediction_rows.append(pred_df)
            selected_weighted, weighted_score, weighted_records = _select_probability_fusion_candidate_within_family(family_name='weighted', family_candidates=weighted_candidates, y_train=y_train, meta_train_prob=meta_train_prob, rad_train_prob=rad_train_prob, config=config, seed=seed + fold * 1000 + 2000)
            for record in weighted_records:
                record.update({'seed': seed, 'fold': fold})
                all_selection_rows.append(record)
            weighted_train_prob, weighted_test_prob, _ = _fit_apply_probability_fusion_candidate(candidate=selected_weighted, y_train=y_train, meta_train=meta_train_prob, rad_train=rad_train_prob, meta_test=meta_test_prob, rad_test=rad_test_prob, seed=seed + fold * 1000 + 2100)
            row, pred_df = _append_probability_fusion_result(display_name='weighted_oof_selected', selected_candidate_name=selected_weighted['name'], candidate_type='weighted_average', y_train=y_train, y_test=y_test, pid_test=pid_test, meta_test_prob=meta_test_prob, rad_test_prob=rad_test_prob, train_prob=weighted_train_prob, test_prob=weighted_test_prob, config=config, seed=seed, fold=fold, pred_dir=pred_dir)
            row['family_selection_score_oof'] = weighted_score
            all_metric_rows.append(row)
            all_prediction_rows.append(pred_df)
            selected_logit, logit_score, logit_records = _select_probability_fusion_candidate_within_family(family_name='logit_stacking', family_candidates=logit_candidates, y_train=y_train, meta_train_prob=meta_train_prob, rad_train_prob=rad_train_prob, config=config, seed=seed + fold * 1000 + 3000)
            for record in logit_records:
                record.update({'seed': seed, 'fold': fold})
                all_selection_rows.append(record)
            logit_train_prob, logit_test_prob, _ = _fit_apply_probability_fusion_candidate(candidate=selected_logit, y_train=y_train, meta_train=meta_train_prob, rad_train=rad_train_prob, meta_test=meta_test_prob, rad_test=rad_test_prob, seed=seed + fold * 1000 + 3100)
            logit_threshold_oof_prob = _make_logit_stacking_oof_probability(candidate=selected_logit, y_train=y_train, meta_train_prob=meta_train_prob, rad_train_prob=rad_train_prob, seed=seed + fold * 1000 + 3200, requested_folds=int(getattr(config, 'PROB_FUSION_STACKING_SELECTION_FOLDS', 5)))
            row, pred_df = _append_probability_fusion_result(display_name='logit_stacking_oof_selected', selected_candidate_name=selected_logit['name'], candidate_type='logit_stacking', y_train=y_train, y_test=y_test, pid_test=pid_test, meta_test_prob=meta_test_prob, rad_test_prob=rad_test_prob, train_prob=logit_threshold_oof_prob, test_prob=logit_test_prob, config=config, seed=seed, fold=fold, pred_dir=pred_dir)
            row['family_selection_score_oof'] = logit_score
            all_metric_rows.append(row)
            all_prediction_rows.append(pred_df)
            print(f"[weighted] selected={selected_weighted['name']}, OOF score={weighted_score:.4f}")
            print(f"[logit stacking] selected={selected_logit['name']}, OOF score={logit_score:.4f}")
    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_path = os.path.join(save_dir, 'all_fold_metrics_probability_fusion.csv')
    metrics_df.to_csv(metrics_path, index=False)
    pred_all = pd.concat(all_prediction_rows, ignore_index=True)
    pred_all_path = os.path.join(save_dir, 'all_predictions_probability_fusion.csv')
    pred_all.to_csv(pred_all_path, index=False)
    selection_df = pd.DataFrame(all_selection_rows)
    selection_path = os.path.join(save_dir, 'fusion_family_selection_records.csv')
    selection_df.to_csv(selection_path, index=False)
    if len(all_selected_rad_rows) > 0:
        pd.DataFrame(all_selected_rad_rows).to_csv(os.path.join(save_dir, 'radiomics_selected_features_probability_fusion.csv'), index=False)
    if len(all_inner_tuning_rows) > 0:
        pd.DataFrame(all_inner_tuning_rows).to_csv(os.path.join(save_dir, 'radiomics_inner_tuning_records_probability_fusion.csv'), index=False)
    metric_cols = ['auc', 'ap', 'acc', 'bal_acc', 'f1', 'f1_macro', 'sensitivity', 'specificity', 'threshold', 'threshold_score_oof']
    summary_rows = []
    for candidate_name, group in metrics_df.groupby('candidate'):
        summary_row = {'candidate': candidate_name, 'n': int(len(group))}
        for metric_name in metric_cols:
            summary_row[f'{metric_name}_mean'] = float(np.nanmean(group[metric_name].values))
            summary_row[f'{metric_name}_sd'] = float(np.nanstd(group[metric_name].values))
        summary_rows.append(summary_row)
    summary_df = pd.DataFrame(summary_rows).sort_values('auc_mean', ascending=False)
    summary_path = os.path.join(save_dir, 'final_summary_probability_fusion.csv')
    summary_df.to_csv(summary_path, index=False)
    selected_frequency_df = metrics_df[metrics_df['candidate'].isin(['weighted_oof_selected', 'logit_stacking_oof_selected'])].groupby(['candidate', 'selected_candidate']).size().reset_index(name='n_selected')
    selected_frequency_df.to_csv(os.path.join(save_dir, 'fusion_selected_candidate_frequency.csv'), index=False)
    print('Saved:', metrics_path)
    print('Saved:', pred_all_path)
    print('Saved:', selection_path)
    print('Saved:', summary_path)
    print('\n===== Probability Fusion Summary =====')
    print(summary_df.to_string(index=False))
    return (metrics_df, summary_df)

def get_model_and_param_grid(config, y=None):
    model_type = str(config.META_MODEL_TYPE).lower()
    if model_type != 'logistic':
        raise ValueError("The public probability-fusion pipeline supports META_MODEL_TYPE='logistic' only.")
    class_weight = 'balanced' if getattr(config, 'USE_CLASS_WEIGHT', False) else None
    model = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=int(getattr(config, 'META_MAX_ITER', 50000)), tol=float(getattr(config, 'META_TOL', 0.001)), class_weight=class_weight, random_state=int(getattr(config, 'META_RANDOM_STATE', 42)))
    param_grid = {'C': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.3, 0.5, 0.7]}
    return (model, param_grid)

def _safe_cv_splits(y, requested_cv=3):
    """Return a class-count-aware number of cross-validation splits."""
    y = np.asarray(y).astype(int)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return 0
    min_count = int(np.min(counts))
    return int(max(0, min(requested_cv, min_count)))

def _collect_meta_arrays_from_indices(dataset, indices):
    """Collect metadata, labels, and patient IDs for selected rows."""
    X_raw, y_list, pid_list = ([], [], [])
    for i in indices:
        _, meta, y_val, pid = dataset[i]
        X_raw.append(meta.numpy())
        y_list.append(y_val.item())
        pid_list.append(str(pid))
    return (np.asarray(X_raw), np.asarray(y_list).astype(int), np.asarray(pid_list))

def _transform_meta_matrix(X_raw, dataset, config):
    """Apply metadata feature transformation row by row."""
    X_feat = []
    feature_names = None
    for x in X_raw:
        x_feat, feature_names = transform_meta(x, mode=config.META_FEATURE_MODE, feature_names=dataset.meta_cols)
        X_feat.append(x_feat)
    return (np.asarray(X_feat), feature_names)

def _select_meta_features_train_only(X_train_scaled, y_train, feature_names, config, seed):
    if not getattr(config, 'USE_FIXED_FEATURES', False):
        if str(getattr(config, 'FEATURE_SELECTION_METHOD', 'none')).lower() == 'none':
            return np.arange(X_train_scaled.shape[1], dtype=int)
        raise ValueError("The public pipeline supports fixed metadata features or FEATURE_SELECTION_METHOD='none' only.")
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    selected_idx = [name_to_idx[name] for name in config.FIXED_FEATURE_NAMES if name in name_to_idx]
    missing_names = [name for name in config.FIXED_FEATURE_NAMES if name not in name_to_idx]
    if missing_names:
        print(f'[WARNING] Fixed metadata features not found: {missing_names}')
    if not selected_idx:
        raise ValueError('No fixed metadata features were found. Check FIXED_FEATURE_NAMES.')
    return np.asarray(selected_idx, dtype=int)

def _fit_meta_model_leak_free(dataset, config, train_indices, seed):
    """
    Specified training indices only are used to fit:

    1. Missing-code conversion
    2. Meta-information imputation
    3. Feature engineering
    4. StandardScaler
    5. Feature selection
    6. Hyperparameter selection
    7. Calibration

    No validation/test feature distributions or labels are used.
    """
    X_train_raw, y_train, _ = _collect_meta_arrays_from_indices(dataset, train_indices)
    missing_mode = str(getattr(config, 'META_MISSING_MODE', 'none')).lower()
    if missing_mode == 'fold_imputation':
        meta_missing_pipeline = fit_meta_missing_pipeline(X_train_raw=X_train_raw, meta_columns=dataset.meta_cols, config=config)
        X_train_clean = meta_missing_pipeline.transform(X_train_raw, stage='meta_model_training')
    elif missing_mode == 'none':
        meta_missing_pipeline = None
        X_train_clean = np.asarray(X_train_raw, dtype=np.float64)
        if not np.all(np.isfinite(X_train_clean)):
            raise ValueError("META_MISSING_MODE='none', but the raw meta matrix contains NaN or Inf.")
    else:
        raise ValueError(f'Unsupported META_MISSING_MODE: {missing_mode}')
    X_train_feat, feature_names = _transform_meta_matrix(X_train_clean, dataset, config)
    X_train_feat = np.asarray(X_train_feat, dtype=np.float64)
    if not np.all(np.isfinite(X_train_feat)):
        raise ValueError('Non-finite values were produced by transform_meta() after missing-data handling.')
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    selected_idx = _select_meta_features_train_only(X_train_scaled=X_train_scaled, y_train=y_train, feature_names=feature_names, config=config, seed=seed)
    selected_feature_names = [feature_names[i] for i in selected_idx]
    X_train_sel = X_train_scaled[:, selected_idx]
    base_model, param_grid = get_model_and_param_grid(config, y_train)
    cv_grid = _safe_cv_splits(y_train, requested_cv=3)
    if len(param_grid) > 0 and cv_grid >= 2:
        grid = GridSearchCV(base_model, param_grid, cv=cv_grid, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_sel, y_train)
        best_model = clone(grid.best_estimator_)
        print('[LeakFree Meta] best params:', grid.best_params_)
    else:
        best_model = base_model
        if len(param_grid) > 0:
            print('[WARNING] GridSearchCV skipped because too few samples per class.')
    best_model.fit(X_train_sel, y_train)
    if getattr(config, 'USE_CALIBRATION', False):
        cv_calib = _safe_cv_splits(y_train, requested_cv=3)
        if cv_calib >= 2:
            final_model = CalibratedClassifierCV(clone(best_model), method='sigmoid', cv=cv_calib)
            final_model.fit(X_train_sel, y_train)
        else:
            print('[WARNING] Calibration skipped because too few samples per class.')
            final_model = best_model
    else:
        final_model = best_model
    if meta_missing_pipeline is not None:
        print('[Meta Missing] training missing counts:', meta_missing_pipeline.missing_count_train)
        print('[Meta Missing] imputation values:', meta_missing_pipeline.imputation_values)
    return (final_model, scaler, selected_idx, selected_feature_names, X_train_sel, y_train, meta_missing_pipeline)

def _predict_meta_prob_with_pipeline(dataset, config, pred_indices, fitted_model, scaler, selected_idx, meta_missing_pipeline):
    """
    Use the fitted meta pipeline for prediction.

    The imputer, scaler, selected features, and model were all
    fitted using the corresponding training fold only.
    """
    X_pred_raw, y_pred, pid_pred = _collect_meta_arrays_from_indices(dataset, pred_indices)
    missing_mode = str(getattr(config, 'META_MISSING_MODE', 'none')).lower()
    if missing_mode == 'fold_imputation':
        if meta_missing_pipeline is None:
            raise ValueError("META_MISSING_MODE='fold_imputation', but meta_missing_pipeline is None.")
        X_pred_clean = meta_missing_pipeline.transform(X_pred_raw, stage='meta_model_prediction')
    elif missing_mode == 'none':
        X_pred_clean = np.asarray(X_pred_raw, dtype=np.float64)
        if not np.all(np.isfinite(X_pred_clean)):
            raise ValueError("Non-finite values found in prediction meta data while META_MISSING_MODE='none'.")
    else:
        raise ValueError(f'Unsupported META_MISSING_MODE: {missing_mode}')
    X_pred_feat, _ = _transform_meta_matrix(X_pred_clean, dataset, config)
    X_pred_feat = np.asarray(X_pred_feat, dtype=np.float64)
    if not np.all(np.isfinite(X_pred_feat)):
        raise ValueError('Non-finite values were produced by transform_meta() during prediction.')
    X_pred_scaled = scaler.transform(X_pred_feat)
    X_pred_sel = X_pred_scaled[:, selected_idx]
    prob = fitted_model.predict_proba(X_pred_sel)[:, 1]
    return (prob, y_pred, pid_pred, X_pred_sel)

def run_nested_cv(dataset, config, save_dir):
    """Compatibility entry point restricted to probability-level fusion."""
    architecture = str(getattr(config, 'FUSION_ARCHITECTURE', '')).lower()
    if architecture != 'probability':
        raise ValueError("This pruned nested_cv module supports only FUSION_ARCHITECTURE='probability'.")
    return run_probability_late_fusion_cv(dataset, config, save_dir)
