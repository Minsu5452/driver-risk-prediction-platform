import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, mean_squared_error
from src.core.constants import SEED, N_JOBS, ENSEMBLE_DIRICHLET_SAMPLES
from src.models.metrics import sigmoid, logit, ece_fast, compute_metrics


def golden_section_minimize(fn, a, b, n_iter=60):
    """Golden section search to minimize *fn* over [a, b]."""
    phi = (1 + 5**0.5) / 2
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    fc = fn(c)
    fd = fn(d)
    for _ in range(n_iter):
        if fc < fd:
            b, fd = d, fc
        else:
            a, fc = c, fd
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        fc = fn(c)
        fd = fn(d)
    return (a + b) / 2

def optimize_weights_and_temperature(
    y_true: np.ndarray, cal_oof: dict, n_samples: int = ENSEMBLE_DIRICHLET_SAMPLES, n_jobs: int = N_JOBS
):
    rng = np.random.default_rng(SEED)
    names = list(cal_oof.keys())
    P = np.vstack([cal_oof[n] for n in names])
    K, N = P.shape

    def score_with_weights(w: np.ndarray):
        z = np.sum(w[:, None] * logit(P), axis=0)
        fixed_auc = roc_auc_score(y_true, z)

        def obj_T(T: float) -> float:
            p = sigmoid(z / T)
            brier = mean_squared_error(y_true, p)
            ece = ece_fast(y_true, p)
            return 0.5 * (1 - fixed_auc) + 0.25 * brier + 0.25 * ece

        T_best = golden_section_minimize(obj_T, 0.5, 10.0)
        p_final = sigmoid(z / T_best)
        m = compute_metrics(y_true, p_final)
        return m["score"], float(T_best), m

    cand_ws = [np.ones(K) / K]
    for i in range(K):
        e = np.zeros(K)
        e[i] = 1.0
        cand_ws.append(e)
    for _ in range(n_samples):
        w = rng.dirichlet(alpha=np.ones(K))
        cand_ws.append(w)

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(score_with_weights)(w) for w in cand_ws
    )
    best_idx = int(np.argmin([r[0] for r in results]))
    best_w = cand_ws[best_idx]
    best_T = results[best_idx][1]
    best_m = results[best_idx][2]
    return names, best_w.astype(float), float(best_T), best_m
