import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl

from src.data.datasets import LungCancerDataset
from src.data.augmentation import get_transforms


class LungCancerDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, fold_index, batch_size=32, num_workers=4, image_size=224):
        super().__init__()
        self.csv_file = csv_file
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.train_transform = get_transforms(phase='train', image_size=self.image_size)
        self.valid_transform = get_transforms(phase='valid', image_size=self.image_size)

        self.dataset = None
        self.train_idx, self.val_idx = None, None

    def prepare_data(self):
        # Already prepared via CSV — can extend for download
        self.dataset = LungCancerDataset(self.csv_file)

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        y = df['label'].values
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(skf.split(df, y)):
            if fold == self.fold_index:
                self.train_idx = train_idx
                self.val_idx = val_idx
                break

        self.train_dataset = LungCancerDataset(
            csv_file=self.csv_file,
            transform=self.train_transform
        )
        self.val_dataset = LungCancerDataset(
            csv_file=self.csv_file,
            transform=self.valid_transform
        )

        # Subset sampling based on stratified indices
        self.train_dataset = Subset(self.train_dataset, self.train_idx)
        self.val_dataset = Subset(self.val_dataset, self.val_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
