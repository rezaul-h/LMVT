import os
import yaml
import torch
import numpy as np
from torch import nn, optim
from src.utils.logger import setup_logger
from src.utils.seed import set_seed
from src.data.dataloader import build_dataloaders
from src.train.metrics import compute_metrics
from src.train.losses import FocalLoss
from src.models import get_model

class Trainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # Setup
        set_seed(self.config.get("seed", 42))
        self.logger = setup_logger(
            self.config["logging"]["log_dir"], self.config["logging"]["log_file"]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = self.config["dataset"]["num_classes"]

    def train_fold(self, fold):
        # Data loaders
        train_loader, valid_loader = build_dataloaders(
            **self.config["dataset"], fold=fold
        )
        # Model
        model = get_model(self.config["model"]["name"], self.num_classes)
        model.to(self.device)
        # Loss
        if self.config["training"].get("use_focal", False):
            criterion = FocalLoss(gamma=self.config["training"]["gamma"])
        else:
            criterion = nn.CrossEntropyLoss()
        # Optimizer & Scheduler
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config["training"]["step_size"],
            gamma=self.config["training"]["gamma"],
        )
        best_val_loss = np.inf
        patience = self.config["training"].get("patience", 5)
        wait = 0

        for epoch in range(self.config["training"]["epochs"]):
            # Training
            model.train()
            train_losses = []
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            # Validation
            model.eval()
            val_losses = []
            all_true, all_pred, all_scores = [], [], []
            with torch.no_grad():
                for imgs, labels in valid_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_losses.append(loss.item())
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)
                    all_true.extend(labels.cpu().numpy())
                    all_pred.extend(preds)
                    all_scores.extend(probs)
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            metrics = compute_metrics(
                all_true, all_pred, np.array(all_scores)
            )
            self.logger.info(
                f"Fold {fold}, Epoch [{epoch+1}/{self.config['training']['epochs']}], "
                f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {metrics['accuracy']:.4f}"
            )
            # Checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(
                    self.config["logging"]["log_dir"], f"best_model_fold{fold}.pth"
                ))
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    self.logger.info("Early stopping triggered")
                    break
            scheduler.step()

    def run(self):
        folds = self.config["cross_validation"]["folds"]
        for fold in range(folds):
            self.logger.info(f"Starting fold {fold}")
            self.train_fold(fold)
