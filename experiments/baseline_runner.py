import subprocess

def run_baselines(config_paths):
    for cfg in config_paths:
        print(f"=== Running baseline: {cfg} ===")
        subprocess.run(["python", "train.py", "--config", cfg], check=True)

if __name__ == "__main__":
    # List of config files for baseline models
    baselines = [
        "config/config_resnet50.yaml",
        "config/config_swin_b.yaml",
        "config/config_convnextv2_large.yaml",
        "config/config_inception_resnet_v2.yaml",
        "config/config_efficientnet_b4.yaml",
        "config/config_xception.yaml",
        "config/config_capsulenet.yaml",
        "config/config_squeezenet.yaml",
        "config/config_shufflenet.yaml"
    ]
    run_baselines(baselines)
