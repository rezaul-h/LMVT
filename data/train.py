import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils.losses import FocalLoss
from src.utils.metrics import pr_auc_score, specificity_score, mcc_score
from src.utils.logger import create_logger
from src.config import CFG

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)

        all_preds.append(preds)
        all_labels.append(labels)
        all_probs.append(probs)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    acc = (all_preds == all_labels).float().mean().item()
    f1 = torchmetrics.functional.f1_score(all_preds, all_labels, average='macro', num_classes=CFG.num_classes)
    pr_auc = pr_auc_score(all_labels, all_probs)
    specificity = specificity_score(all_labels, all_preds)
    mcc = mcc_score(all_labels, all_preds)

    return total_loss / len(loader), acc, f1, pr_auc, specificity, mcc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(preds)
            all_labels.append(labels)
            all_probs.append(probs)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    acc = (all_preds == all_labels).float().mean().item()
    f1 = torchmetrics.functional.f1_score(all_preds, all_labels, average='macro', num_classes=CFG.num_classes)
    pr_auc = pr_auc_score(all_labels, all_probs)
    specificity = specificity_score(all_labels, all_preds)
    mcc = mcc_score(all_labels, all_preds)

    return total_loss / len(loader), acc, f1, pr_auc, specificity, mcc