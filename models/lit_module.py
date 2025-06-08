import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    Accuracy, F1Score, PrecisionRecallCurve,
    Specificity, MatthewsCorrCoef, AUROC
)
from src.utils.losses import FocalLoss
from src.utils.metrics import pr_auc_score

class LitLMVTModule(pl.LightningModule):
    def __init__(self, model, num_classes, lr=1e-4, loss_fn='cross_entropy'):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.loss_fn_type = loss_fn

        # Define loss function
        if loss_fn == 'focal':
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.train_mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        self.train_spec = Specificity(task="multiclass", num_classes=num_classes)
        self.train_auc = AUROC(task="multiclass", num_classes=num_classes, average="macro")

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.val_mcc = MatthewsCorrCoef(task="multiclass", num_classes=num_classes)
        self.val_spec = Specificity(task="multiclass", num_classes=num_classes)
        self.val_auc = AUROC(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def step(self, batch, stage="train"):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        if stage == "train":
            self.train_acc.update(preds, y)
            self.train_f1.update(preds, y)
            self.train_mcc.update(preds, y)
            self.train_spec.update(preds, y)
            self.train_auc.update(logits.softmax(dim=-1), y)
        else:
            self.val_acc.update(preds, y)
            self.val_f1.update(preds, y)
            self.val_mcc.update(preds, y)
            self.val_spec.update(preds, y)
            self.val_auc.update(logits.softmax(dim=-1), y)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="val")

    def on_train_epoch_end(self):
        self.log("train/acc", self.train_acc.compute(), prog_bar=True)
        self.log("train/f1", self.train_f1.compute(), prog_bar=True)
        self.log("train/mcc", self.train_mcc.compute(), prog_bar=True)
        self.log("train/spec", self.train_spec.compute(), prog_bar=True)
        self.log("train/auc", self.train_auc.compute(), prog_bar=True)
        self.train_acc.reset()
        self.train_f1.reset()
        self.train_mcc.reset()
        self.train_spec.reset()
        self.train_auc.reset()

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.log("val/mcc", self.val_mcc.compute(), prog_bar=True)
        self.log("val/spec", self.val_spec.compute(), prog_bar=True)
        self.log("val/auc", self.val_auc.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()
        self.val_mcc.reset()
        self.val_spec.reset()
        self.val_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]
