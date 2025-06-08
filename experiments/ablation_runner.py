#!/usr/bin/env python

import subprocess

def run_experiments(config_paths):
    for cfg in config_paths:
        print(f"=== Running ablation: {cfg} ===")
        subprocess.run(["python", "train.py", "--config", cfg], check=True)

if __name__ == "__main__":
    # List of config files for ablation studies
    ablations = [
        "config/config_lmvt.yaml",
        "config/config_lmvt_no_cbam.yaml",
        "config/config_lmvt_no_sgldm.yaml",
        "config/config_lmvt_no_mhsa.yaml"
    ]
    run_experiments(ablations)
