"""Standalone checkpoint-enabled Radiomics extraction for the public probability-fusion workflow. MRI and PET experiments may use different fixed bin widths; their outputs are merged by patient_id after strict identifier and label checks."""
import os
import copy
import yaml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import config
import csv
import gc
import json
import time
import traceback
from pathlib import Path
LABEL_CSV = config.LABEL_CSV
ROI_CENTER_DIR = config.TUMOR_ROI
ROI_MODE = config.RADIOMICS_ROI_MODE
DATA_DIR_25 = config.RADIOMICS_DATA_DIR_25
DATA_DIR_32 = config.RADIOMICS_DATA_DIR_32
YAML_TEMPLATE_PATH = getattr(config, 'RADIOMICS_YAML_TEMPLATE', 'radiomics_extraction/pyradiomics_template_probability_fusion.yaml')
RADIOMICS_EXPERIMENTS = config.RADIOMICS_EXPERIMENTS
if ROI_MODE == 'roi25':
    roi_settings = [('ROI25', DATA_DIR_25)]
elif ROI_MODE == 'roi32':
    roi_settings = [('ROI32', DATA_DIR_32)]
elif ROI_MODE == 'both':
    roi_settings = [('ROI25', DATA_DIR_25), ('ROI32', DATA_DIR_32)]
else:
    raise ValueError(f'Unknown RADIOMICS_ROI_MODE: {ROI_MODE}')

def load_label_dataframe(label_csv):
    """Load labels and create patient_id from No when necessary."""
    df = pd.read_csv(label_csv)
    if 'patient_id' not in df.columns:
        if 'No' in df.columns:
            df['patient_id'] = df['No'].apply(lambda x: f'patient{int(x):03d}')
        else:
            raise ValueError('Invalid configuration or input detected in the public probability-fusion workflow.')
    if 'IDH_mutation' not in df.columns:
        if 'IDH' in df.columns:
            df['IDH_mutation'] = df['IDH']
    if '1p19q_codeletion' not in df.columns:
        if '1p19q' in df.columns:
            df['1p19q_codeletion'] = df['1p19q']
    required_cols = ['patient_id', 'IDH_mutation', '1p19q_codeletion']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Invalid configuration or input detected in the public probability-fusion workflow.{col}')
    return df

def load_center_slice(patient_id):
    """
    ROI txt format:
        line1: y
        line2: x
        line3: slice index
    """
    pid_num = int(str(patient_id).replace('patient', ''))
    txt_path = os.path.join(ROI_CENTER_DIR, f'{pid_num}.txt')
    if not os.path.exists(txt_path):
        print(f'[WARNING] ROI txt not found: {txt_path}')
        return None
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    if len(lines) < 3:
        print(f'[WARNING] invalid ROI txt: {txt_path}')
        return None
    slice_idx = int(lines[2])
    return slice_idx

def load_npy_image(npy_path):
    arr = np.load(npy_path).astype(np.float32)
    return arr

def create_full_square_mask(shape):
    """Create a full-square ROI mask for the tumor-centered crop."""
    mask = np.ones(shape, dtype=np.uint8)
    return mask

def to_sitk(arr):
    """Convert a two-dimensional NumPy array to a SimpleITK image."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((1.0, 1.0))
    return img

def safe_percentile_clip(img_arr, low_p, high_p):
    """Apply within-image percentile clipping."""
    low = np.percentile(img_arr, low_p)
    high = np.percentile(img_arr, high_p)
    if high <= low:
        return img_arr.astype(np.float32)
    img_arr = np.clip(img_arr, low, high)
    return img_arr.astype(np.float32)

def standard_z_score(img_arr):
    """Apply within-image z-score standardization."""
    mean = np.mean(img_arr)
    std = np.std(img_arr) + 1e-06
    img_arr = (img_arr - mean) / std
    return img_arr.astype(np.float32)

def preprocess_for_radiomics(img_arr, modality, pet_preprocess, mri_preprocess):
    img_arr = np.asarray(img_arr, dtype=np.float32)
    mri_modalities = {'FLAIR', 'T1', 'T2'}
    pet_modalities = {'MET', 'FDG', 'CHO'}
    if modality in mri_modalities:
        if mri_preprocess != 'per_patch_z':
            raise ValueError("The public extraction pipeline supports mri_preprocess='per_patch_z' only.")
        img_arr = safe_percentile_clip(img_arr, getattr(config, 'MRI_CLIP_LOW', 1), getattr(config, 'MRI_CLIP_HIGH', 99))
        img_arr = standard_z_score(img_arr)
    elif modality in pet_modalities:
        if pet_preprocess != 'clip_only':
            raise ValueError("The public extraction pipeline supports pet_preprocess='clip_only' only.")
        if modality == 'CHO' and getattr(config, 'USE_CHO_STRONG_CLIP', False):
            low_p = getattr(config, 'CHO_CLIP_LOW', 5)
            high_p = getattr(config, 'CHO_CLIP_HIGH', 95)
        else:
            low_p = getattr(config, 'PET_CLIP_LOW', 1)
            high_p = getattr(config, 'PET_CLIP_HIGH', 99)
        img_arr = safe_percentile_clip(img_arr, low_p, high_p)
    else:
        raise ValueError(f'Unknown modality: {modality}')
    return np.nan_to_num(img_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def load_yaml_template(yaml_path):
    """Load the PyRadiomics YAML template."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f'YAML template not found: {yaml_path}')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)
    return params

def make_experiment_yaml(template_params, bin_width, feature_classes, save_path):
    """Create an experiment-specific YAML file."""
    params = copy.deepcopy(template_params)
    if 'setting' not in params:
        params['setting'] = {}
    params['setting']['binWidth'] = float(bin_width)
    params['setting']['normalize'] = False
    params['featureClass'] = {}
    for cls in feature_classes:
        params['featureClass'][cls] = []
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(params, f, sort_keys=False, allow_unicode=True)
    return save_path

def create_extractor_from_yaml(yaml_path):
    """Create a PyRadiomics extractor from a YAML file."""
    extractor = featureextractor.RadiomicsFeatureExtractor(yaml_path)
    print('\nEnabled features:')
    print(extractor.enabledFeatures)
    print('\nEnabled image types:')
    print(extractor.enabledImagetypes)
    print('\nSettings:')
    print(extractor.settings)
    return extractor
CHECKPOINT_VERSION = 1
EXTRACTOR_RESET_EVERY = 10

def get_rss_mb():
    """Return current process memory use in MB when psutil is available."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 2
    except Exception:
        return float('nan')

def safe_patient_filename(patient_id):
    """
    patient001 -> patient001.csv
    """
    text = str(patient_id)
    safe = ''.join((char for char in text if char.isalnum() or char in ('-', '_')))
    return safe + '.csv'

def atomic_write_json(data, output_path):
    """Write JSON atomically through a temporary file."""
    output_path = Path(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, output_path)

def atomic_write_patient_row(row, output_path):
    """Write one patient checkpoint atomically."""
    output_path = Path(output_path)
    temp_path = output_path.with_suffix(output_path.suffix + '.tmp')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame([row])
    dataframe.to_csv(temp_path, index=False, encoding='utf-8')
    os.replace(temp_path, output_path)

def append_failure(failure_csv, patient_id, modality, stage, error):
    """Append one extraction failure to the failure log."""
    failure_csv = Path(failure_csv)
    failure_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failure_csv.exists()
    record = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'patient_id': patient_id, 'modality': modality, 'stage': stage, 'error_type': type(error).__name__, 'error': str(error), 'rss_mb': get_rss_mb(), 'traceback': traceback.format_exc()}
    with open(failure_csv, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)
        file.flush()
        os.fsync(file.fileno())

def build_manifest(experiment, yaml_output_path):
    """Build a manifest that identifies the extraction settings."""
    return {'checkpoint_version': CHECKPOINT_VERSION, 'experiment_name': experiment['name'], 'output_csv': str(experiment['output_csv']), 'modalities': list(experiment.get('modalities', [])), 'pet_preprocess': experiment.get('pet_preprocess', 'clip_only'), 'mri_preprocess': experiment.get('mri_preprocess', 'per_patch_z'), 'bin_width': float(experiment.get('bin_width', 10.0)), 'feature_classes': list(experiment.get('feature_classes', ['firstorder', 'glcm', 'gldm'])), 'roi_mode': ROI_MODE, 'yaml_used': str(yaml_output_path)}

def validate_or_create_manifest(manifest, manifest_path):
    """Validate an existing checkpoint manifest or create it."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as file:
            existing = json.load(file)
        if existing != manifest:
            raise RuntimeError(f'Checkpoint manifest does not match the current experiment settings.\nExisting:\n{existing}\nCurrent:\n{manifest}\nUse a different output CSV or remove the checkpoint folder after confirming that it is no longer needed.')
    else:
        atomic_write_json(manifest, manifest_path)

def merge_patient_checkpoints(patient_ids, checkpoint_patient_dir, output_csv):
    """Merge patient checkpoints in label-table order."""
    checkpoint_patient_dir = Path(checkpoint_patient_dir)
    output_csv = Path(output_csv)
    temp_output = output_csv.with_suffix(output_csv.suffix + '.tmp')
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    number_written = 0
    header_written = False
    with open(temp_output, 'w', newline='', encoding='utf-8') as output_file:
        for patient_id in patient_ids:
            patient_file = checkpoint_patient_dir / safe_patient_filename(patient_id)
            if not patient_file.exists():
                continue
            patient_dataframe = pd.read_csv(patient_file, low_memory=False)
            patient_dataframe.to_csv(output_file, index=False, header=not header_written)
            header_written = True
            number_written += len(patient_dataframe)
        output_file.flush()
        os.fsync(output_file.fileno())
    if number_written == 0:
        if temp_output.exists():
            temp_output.unlink()
        print('[WARNING] No patient checkpoints were available for merging.')
        return 0
    os.replace(temp_output, output_csv)
    return number_written

def create_extractor(yaml_output_path):
    """Create or recreate the PyRadiomics extractor."""
    extractor = create_extractor_from_yaml(str(yaml_output_path))
    return extractor

def extract_one_experiment_resumable(label_dataframe, experiment, template_parameters):
    """Run one checkpoint-enabled Radiomics extraction experiment."""
    experiment_name = experiment['name']
    output_csv = Path(experiment['output_csv'])
    modalities = experiment.get('modalities', getattr(config, 'RADIOMICS_EXTRACT_MODALITIES', config.MODALITIES))
    pet_preprocess = experiment.get('pet_preprocess', 'clip_only')
    mri_preprocess = experiment.get('mri_preprocess', 'per_patch_z')
    bin_width = float(experiment.get('bin_width', 10.0))
    feature_classes = experiment.get('feature_classes', ['firstorder', 'glcm', 'gldm'])
    print('\n' + '=' * 70)
    print(f'Experiment: {experiment_name}')
    print(f'Output CSV: {output_csv}')
    print(f'Modalities: {modalities}')
    print(f'PET preprocess: {pet_preprocess}')
    print(f'MRI preprocess: {mri_preprocess}')
    print(f'binWidth: {bin_width}')
    print(f'featureClasses: {feature_classes}')
    print(f'Extractor reset every: {EXTRACTOR_RESET_EVERY} patients')
    print('=' * 70)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    yaml_output_dir = output_csv.parent / 'yaml_used'
    yaml_output_path = yaml_output_dir / ('pyradiomics_' + experiment_name + '.yaml')
    make_experiment_yaml(template_params=template_parameters, bin_width=bin_width, feature_classes=feature_classes, save_path=str(yaml_output_path))
    checkpoint_root = output_csv.parent / (output_csv.stem + '_checkpoints')
    checkpoint_patient_dir = checkpoint_root / 'patients'
    checkpoint_patient_dir.mkdir(parents=True, exist_ok=True)
    failure_csv = checkpoint_root / 'failures.csv'
    manifest_path = checkpoint_root / 'manifest.json'
    manifest = build_manifest(experiment, yaml_output_path)
    validate_or_create_manifest(manifest, manifest_path)
    patient_ids = label_dataframe['patient_id'].tolist()
    completed = {path.stem for path in checkpoint_patient_dir.glob('*.csv')}
    print(f'Previously completed patients: {len(completed)}')
    extractor = None
    patients_since_reset = EXTRACTOR_RESET_EVERY
    success_count = 0
    skip_count = 0
    failure_count = 0
    for patient_index, patient_id in enumerate(patient_ids, start=1):
        patient_file = checkpoint_patient_dir / safe_patient_filename(patient_id)
        if patient_file.exists():
            print(f'[RESUME-SKIP] {patient_id} ({patient_index}/{len(patient_ids)})')
            skip_count += 1
            continue
        print(f'\nProcessing: {patient_id} ({patient_index}/{len(patient_ids)})')
        print(f'Current RSS: {get_rss_mb():.1f} MB')
        if extractor is None or patients_since_reset >= EXTRACTOR_RESET_EVERY:
            if extractor is not None:
                del extractor
            gc.collect()
            print('[MEMORY] Recreating PyRadiomics extractor')
            extractor = create_extractor(yaml_output_path)
            patients_since_reset = 0
        center_slice = load_center_slice(patient_id)
        if center_slice is None:
            error = RuntimeError('Center slice was missing.')
            append_failure(failure_csv, patient_id, '', 'center_slice', error)
            failure_count += 1
            continue
        patient_label_row = label_dataframe[label_dataframe['patient_id'] == patient_id]
        if len(patient_label_row) == 0:
            error = RuntimeError('Label row was missing.')
            append_failure(failure_csv, patient_id, '', 'label', error)
            failure_count += 1
            continue
        idh = int(patient_label_row['IDH_mutation'].values[0])
        codeletion = int(patient_label_row['1p19q_codeletion'].values[0])
        patient_output_row = {'patient_id': patient_id, 'IDH_mutation': idh, '1p19q_codeletion': codeletion}
        success = True
        failed_modality = ''
        try:
            for roi_prefix, data_directory in roi_settings:
                patient_directory = Path(data_directory) / patient_id
                for modality in modalities:
                    failed_modality = modality
                    npy_path = patient_directory / modality / f'slice_{center_slice:03d}.npy'
                    if not npy_path.exists():
                        raise FileNotFoundError(str(npy_path))
                    image_array = load_npy_image(str(npy_path))
                    image_array = preprocess_for_radiomics(img_arr=image_array, modality=modality, pet_preprocess=pet_preprocess, mri_preprocess=mri_preprocess)
                    mask_array = create_full_square_mask(image_array.shape)
                    image_sitk = to_sitk(image_array)
                    mask_sitk = to_sitk(mask_array)
                    features = extractor.execute(image_sitk, mask_sitk)
                    for key, value in features.items():
                        if key.startswith('diagnostics'):
                            continue
                        feature_name = f'{roi_prefix}_{modality}_{key}'
                        try:
                            patient_output_row[feature_name] = float(value)
                        except Exception:
                            patient_output_row[feature_name] = value
                    del features
                    del image_sitk
                    del mask_sitk
                    del image_array
                    del mask_array
                    gc.collect()
        except Exception as error:
            success = False
            print(f'[ERROR] feature extraction failed: {patient_id}, {failed_modality}')
            print(error)
            append_failure(failure_csv, patient_id, failed_modality, 'feature_extraction', error)
            failure_count += 1
        if success:
            atomic_write_patient_row(patient_output_row, patient_file)
            success_count += 1
            print(f'[CHECKPOINT OK] {patient_id}')
            print(f'Saved: {patient_file}')
        else:
            print(f'[FAILED] {patient_id}')
        del patient_output_row
        gc.collect()
        patients_since_reset += 1
        if patient_index % 10 == 0:
            rows_merged = merge_patient_checkpoints(patient_ids, checkpoint_patient_dir, output_csv)
            print(f'[PERIODIC MERGE] {rows_merged} patients written to {output_csv}')
    if extractor is not None:
        del extractor
    gc.collect()
    final_rows = merge_patient_checkpoints(patient_ids, checkpoint_patient_dir, output_csv)
    missing_checkpoints = [patient_id for patient_id in patient_ids if not (checkpoint_patient_dir / safe_patient_filename(patient_id)).exists()]
    summary = {'experiment_name': experiment_name, 'total_label_patients': len(patient_ids), 'success_in_this_run': success_count, 'resume_skipped': skip_count, 'failures_in_this_run': failure_count, 'final_rows': final_rows, 'missing_checkpoint_count': len(missing_checkpoints), 'missing_patients': missing_checkpoints, 'final_csv': str(output_csv), 'checkpoint_root': str(checkpoint_root)}
    atomic_write_json(summary, checkpoint_root / 'summary.json')
    print('\n' + '=' * 70)
    print('EXTRACTION SUMMARY')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('=' * 70)
    return output_csv

def merge_experiment_feature_tables(experiments, output_csv):
    """Merge experiment CSVs column-wise after strict patient and label validation."""
    if len(experiments) < 2:
        raise ValueError('At least two experiments are required for a combined feature table.')
    id_cols = ['patient_id', 'IDH_mutation', '1p19q_codeletion']
    merged = None
    reference_ids = None
    reference_labels = None
    feature_names = set()
    for experiment in experiments:
        csv_path = Path(experiment['output_csv'])
        if not csv_path.exists():
            raise FileNotFoundError(f'Experiment output was not found: {csv_path}')
        frame = pd.read_csv(csv_path, low_memory=False)
        missing = [c for c in id_cols if c not in frame.columns]
        if missing:
            raise ValueError(f'{csv_path} is missing required columns: {missing}')
        if frame['patient_id'].astype(str).duplicated().any():
            raise ValueError(f'Duplicated patient_id values were found in {csv_path}.')
        frame['patient_id'] = frame['patient_id'].astype(str)
        frame = frame.sort_values('patient_id').reset_index(drop=True)
        ids = frame['patient_id'].tolist()
        labels = frame[id_cols].copy()
        if reference_ids is None:
            reference_ids = ids
            reference_labels = labels
            merged = frame
            feature_names.update((c for c in frame.columns if c not in id_cols))
            continue
        if ids != reference_ids:
            raise ValueError(f'Patient IDs do not match across experiment outputs: {csv_path}')
        if not labels.equals(reference_labels):
            raise ValueError(f'Labels do not match across experiment outputs: {csv_path}')
        new_features = [c for c in frame.columns if c not in id_cols]
        duplicates = sorted(feature_names.intersection(new_features))
        if duplicates:
            raise ValueError(f'Duplicated feature columns were found across experiments: {duplicates[:10]}')
        merged = merged.merge(frame, on=id_cols, how='inner', validate='one_to_one', sort=False)
        feature_names.update(new_features)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_csv.with_suffix(output_csv.suffix + '.tmp')
    merged.to_csv(temp_path, index=False)
    os.replace(temp_path, output_csv)
    print(f'Combined feature table saved: {output_csv}')
    print(f'Rows: {len(merged)}; feature columns: {len(merged.columns) - len(id_cols)}')
    return output_csv

def main():
    print('Loaded config.')
    print('LABEL_CSV:', LABEL_CSV)
    print('ROI_MODE:', ROI_MODE)
    print('YAML_TEMPLATE_PATH:', YAML_TEMPLATE_PATH)
    print('Number of experiments:', len(RADIOMICS_EXPERIMENTS))
    label_dataframe = load_label_dataframe(LABEL_CSV)
    print('\nLoaded label CSV:', label_dataframe.shape)
    template_parameters = load_yaml_template(YAML_TEMPLATE_PATH)
    for experiment in RADIOMICS_EXPERIMENTS:
        extract_one_experiment_resumable(label_dataframe=label_dataframe, experiment=experiment, template_parameters=template_parameters)
    merge_experiment_feature_tables(RADIOMICS_EXPERIMENTS, config.RADIOMICS_CSV)
    print('\nAll radiomics experiments finished.')
if __name__ == '__main__':
    main()
