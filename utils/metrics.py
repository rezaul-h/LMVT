import torch
import torch.nn.functional as F

import torch
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef
import numpy as np

def pr_auc_score(y_true, y_prob, average='macro'):
    y_true = y_true.cpu().numpy()
    y_prob = y_prob.cpu().numpy()
    return average_precision_score(y_true, y_prob, average=average)

def specificity_score(y_true, y_pred, average='macro'):
    cm = confusion_matrix(y_true, y_pred)
    tn = cm.sum(axis=1) - np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    specificity_per_class = tn / (tn + fp + 1e-10)
    if average == 'macro':
        return np.mean(specificity_per_class)
    return specificity_per_class

def mcc_score(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    return matthews_corrcoef(y_true, y_pred)