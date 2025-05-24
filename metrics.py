import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_fscore_support,
)


def aucPerformance(score, labels):

    roc_auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)
    return roc_auc, ap


def F1Performance(score, target):

    normal_ratio = (target == 0).sum() / len(target)
    score = np.squeeze(score)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, pred, average="binary"
    )
    return f1, threshold


def observation_indicators(y_pred, y_true, threshold=0.5):

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred = (y_pred >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)


    print(
        f"+------------+----------+\n"
        f"| Metric     | Value    |\n"
        f"+------------+----------+\n"
        f"| Recall     | {recall:.4f}  |\n"
        f"| Precision  | {precision:.4f}  |\n"
        f"| FPR        | {fpr:.4f}  |\n"  
        f"| TPR        | {tpr:.4f}  |\n"
        f"+------------+----------+"
    )
