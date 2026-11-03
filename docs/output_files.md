# Output files

## Probability-fusion outputs

The probability-fusion analysis creates a timestamped directory under `results/`.

- `all_fold_metrics_probability_fusion.csv`: Fold-level metrics for each evaluated model family.
- `all_predictions_probability_fusion.csv`: Patient-level outer-test probabilities and predictions.
- `fusion_family_selection_records.csv`: Outer-training OOF candidate-selection records.
- `radiomics_selected_features_probability_fusion.csv`: Selected Radiomics features and selector metadata.
- `radiomics_inner_tuning_records_probability_fusion.csv`: Inner-CV tuning and refit-attempt records.
- `final_summary_probability_fusion.csv`: Mean and standard deviation across repeated outer folds.
- `fusion_selected_candidate_frequency.csv`: Frequency of selected weighted-average and logit-stacking candidates.
- `predictions_probability_fusion/`: Fold-specific prediction CSV files.
- `log.txt`: Captured console output.
- `config.py`: Copy of the configuration used for the run.

## Radiomics extraction outputs

The extraction workflow writes experiment-specific MRI and PET feature tables, checkpoint folders, failure logs, manifests, generated YAML files, summary JSON files, and the merged feature table configured by `RADIOMICS_CSV`.

## Statistical-analysis outputs

`analysis/jimaging_revision_analysis.py` produces the following main publication files in addition to detailed QC CSVs:

- `Table_2_manuscript.csv`
- `Table_3_manuscript.csv`
- `Table_4_manuscript.csv`
- `Table_S1_binning_discretization_diagnostics.csv`, when `binning_diagnostics.csv` is available
- `Table_S2_modality_comparisons.csv`
- `Table_S3_calibration_metrics.csv`
- `Table_S4_candidate_selection_failure_diagnostics.csv`
- `exploratory_complete_fusion_comparisons.csv`
- `Figure_2_mean_outer_fold_ROC.png` and `.pdf`
- `Figure_3_calibration_plots.png` and `.pdf`
- `Figure_4_feature_selection_frequency.png` and `.pdf`
- `diagnostic_patient_level_aggregated_ROC.png` and `.pdf`
- `analysis_manifest.json`
- `publication_captions.txt`
- `table_caption_notes.txt`

No Excel workbook is created, and `openpyxl` is not required.

## Metadata-audit outputs

`analysis/metadata_completeness_audit.py` writes:

- `metadata_completeness_summary.csv`
- `metadata_value_counts.csv`
- `metadata_issue_summary.csv`
- `metadata_audit_overall.json`

## Synthetic examples

The two files in `examples/` contain four fully synthetic records with matching `patient_id` values and labels. They are schema examples only, are not derived from study participants, are insufficient for five-fold nested cross-validation, and cannot reproduce reported results.
