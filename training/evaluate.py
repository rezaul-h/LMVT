import os
import yaml
import torch
import numpy as np
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.train.metrics import compute_metrics
from src.models import get_model

def evaluate(config_path, checkpoint_path, fold=None):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    set_seed(config.get("seed", 42))
    logger = setup_logger(
        config["logging"]["log_dir"], config["logging"]["log_file"]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config["dataset"]["num_classes"]
    # Data loader for test (use fold or train+val combined)
    _, test_loader = build_dataloaders(**config["dataset"], fold=fold)
    # Model
    model = get_model(config["model"]["name"], num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    all_true, all_pred, all_scores = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds)
            all_scores.extend(probs)
    metrics = compute_metrics(all_true, all_pred, np.array(all_scores))
    logger.info(f"Test Metrics: {metrics}")
    return metrics
