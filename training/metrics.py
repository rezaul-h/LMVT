import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize

def compute_metrics(y_true, y_pred, y_scores, average="weighted"):
    """
    Compute classification metrics.
    y_true: list or array of true labels
    y_pred: list or array of predicted labels
    y_scores: array of shape (n_samples, n_classes) with probability scores
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_classes = y_scores.shape[1]

    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["f1_score"] = f1_score(y_true, y_pred, average=average)
    # PR AUC
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_scores[:, 1])
        metrics["pr_auc"] = auc(recall, precision)
    else:
        # macro-average PR AUC for multiclass
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        pr_aucs = []
        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
            pr_aucs.append(auc(recall, precision))
        metrics["pr_auc"] = np.mean(pr_aucs)

    # Specificity (mean over classes)
    cm = confusion_matrix(y_true, y_pred)
    spec_per_class = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    metrics["specificity"] = np.mean(spec_per_class)

    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    return metrics
