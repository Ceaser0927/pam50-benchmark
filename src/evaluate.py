"""Metrics, bootstrap confidence intervals and the corrected resampled t-test."""
import numpy as np
from scipy import stats
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

METRICS = ["Balanced accuracy", "Macro precision", "Macro recall", "Macro F1", "Macro OvR ROC-AUC"]


def compute_metrics(y, pred, proba, n_classes):
    labels = np.arange(n_classes)
    return {
        "Balanced accuracy": balanced_accuracy_score(y, pred),
        "Macro precision": precision_score(y, pred, average="macro", labels=labels, zero_division=0),
        "Macro recall": recall_score(y, pred, average="macro", labels=labels, zero_division=0),
        "Macro F1": f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "Macro OvR ROC-AUC": roc_auc_score(y, proba, multi_class="ovr", average="macro",
                                           labels=labels),
    }


def bootstrap_indices(y, n_boot, seed):
    """Stratified bootstrap: resample within each class so every class is always present.
    The same indices are reused for every model, giving paired comparisons."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in np.unique(y)]
    return [np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
            for _ in range(n_boot)]


def bootstrap_metrics(y, pred, proba, n_classes, idx_list):
    out = {m: [] for m in METRICS}
    for idx in idx_list:
        for m, v in compute_metrics(y[idx], pred[idx], proba[idx], n_classes).items():
            out[m].append(v)
    return {m: np.asarray(v) for m, v in out.items()}


def ci(values, alpha=0.05):
    return np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def corrected_resampled_ttest(diffs, n_train, n_test):
    """Nadeau & Bengio (2003) corrected repeated-CV t-test for paired fold differences.
    Plain t-tests on CV folds are anti-conservative because training sets overlap."""
    diffs = np.asarray(diffs)
    k = len(diffs)
    var = diffs.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = diffs.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)
