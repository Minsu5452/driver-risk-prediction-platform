import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, matthews_corrcoef
from sklearn.calibration import calibration_curve
from src.core.constants import CLIP_EPS

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, CLIP_EPS, 1 - CLIP_EPS)
    return np.log(p / (1 - p))

def ece_fast(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    N = y_prob.shape[0]

    bin_counts = np.bincount(bin_ids, minlength=n_bins)
    bin_conf_sum = np.bincount(bin_ids, weights=y_prob, minlength=n_bins)
    bin_true_sum = np.bincount(bin_ids, weights=y_true, minlength=n_bins)

    mask = bin_counts > 0
    bin_conf = bin_conf_sum[mask] / bin_counts[mask]
    bin_acc = bin_true_sum[mask] / bin_counts[mask]

    ece = np.sum((bin_counts[mask] / N) * np.abs(bin_acc - bin_conf))
    return float(ece)

def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    y_prob = np.clip(np.asarray(y_prob, dtype=float), CLIP_EPS, 1 - CLIP_EPS)
    y_true = np.asarray(y_true)
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_totals, _ = np.histogram(y_prob, bins=bin_edges, density=False)
    non_empty_mask = bin_totals > 0
    bin_weights = bin_totals[non_empty_mask] / y_prob.shape[0]
    m = min(len(bin_weights), len(prob_true), len(prob_pred))
    if m == 0:
        return 0.0
    ece = np.sum(bin_weights[:m] * np.abs(prob_true[:m] - prob_pred[:m]))
    return float(ece)

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_prob_clip = np.clip(y_prob, CLIP_EPS, 1 - CLIP_EPS)
    auc = roc_auc_score(y_true, y_prob)
    brier = mean_squared_error(y_true, y_prob_clip)
    ece = expected_calibration_error(y_true, y_prob_clip)
    score = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece
    y_pred_binary = (y_prob >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    return {
        "auc": float(auc),
        "brier": float(brier),
        "ece": float(ece),
        "score": float(score),
        "mcc": float(mcc),
    }
