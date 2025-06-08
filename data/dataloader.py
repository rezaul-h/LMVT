import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from src.data.datasets import LungCancerDataset
from src.utils.seed import seed_everything


def get_stratified_kfold_loaders(csv_path, img_dir, transform, batch_size, num_folds=10, seed=42):
    """
    Prepare stratified k-fold DataLoaders for lung cancer classification.

    Args:
        csv_path (str): Path to CSV with 'image' and 'label' columns.
        img_dir (str): Path to image directory.
        transform (callable): Albumentations transformation.
        batch_size (int): Batch size.
        num_folds (int): Number of folds for cross-validation.
        seed (int): Random seed.

    Returns:
        List of tuples: (train_loader, val_loader) for each fold.
    """
    seed_everything(seed)
    df = pd.read_csv(csv_path)
    labels = df['label'].values
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    fold_loaders = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"[INFO] Preparing Fold {fold + 1}/{num_folds}...")

        train_subset = Subset(LungCancerDataset(csv_path, img_dir, transform), train_idx)
        val_subset = Subset(LungCancerDataset(csv_path, img_dir, transform), val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        fold_loaders.append((train_loader, val_loader))

    return fold_loaders
