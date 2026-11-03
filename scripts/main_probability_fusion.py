# -*- coding: utf-8 -*-
"""Entry point for Meta + Radiomics probability-level fusion only."""

import os
import shutil
import sys
from datetime import datetime

import config
from config import DATA_DIR, LABEL_CSV, MODALITIES, TASK
from data.dataset_probability_fusion import BrainDataset
from cv.nested_cv_probability_fusion import run_nested_cv


class Logger:
    """Write standard output to both the terminal and a UTF-8 log file."""

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    """Run the probability-level Meta + Radiomics fusion analysis."""
    architecture = str(getattr(config, "FUSION_ARCHITECTURE", "")).lower()
    if architecture != "probability":
        raise ValueError(
            "This entry point supports only "
            "FUSION_ARCHITECTURE='probability'."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modality_string = "_".join(MODALITIES)
    save_dir = os.path.join(
        "results",
        f"results_{timestamp}_{TASK}_{modality_string}",
    )
    os.makedirs(save_dir, exist_ok=True)

    config_source = os.path.abspath(config.__file__)
    shutil.copy2(config_source, os.path.join(save_dir, "config.py"))

    log_path = os.path.join(save_dir, "log.txt")
    logger = Logger(log_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("Logging to:", log_path)
        print("Save dir:", save_dir)
        print("Analysis: Meta + Radiomics probability-level fusion")

        dataset = BrainDataset(
            LABEL_CSV,
            DATA_DIR,
            MODALITIES,
            config=config,
            use_edge_full=False,
            task=TASK,
        )

        metrics_df, summary_df = run_nested_cv(
            dataset=dataset,
            config=config,
            save_dir=save_dir,
        )

        print("Final probability-fusion summary:")
        print(summary_df)
        return metrics_df, summary_df
    finally:
        sys.stdout = original_stdout
        logger.close()


if __name__ == "__main__":
    main()
