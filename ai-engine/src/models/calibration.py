import numpy as np
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from src.core.constants import SEED, CLIP_EPS
from src.models.metrics import sigmoid, logit, ece_fast
from src.models.ensemble import golden_section_minimize

class CalibModel:
    def __init__(self, kind: str, params: Dict):
        self.kind = kind
        self.params = params

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, CLIP_EPS, 1 - CLIP_EPS)
        if self.kind == "platt":
            w = self.params["w"]
            b = self.params["b"]
            return sigmoid(w * logit(p) + b)
        elif self.kind == "beta":
            a = self.params["a"]
            b0 = self.params["b"]
            c = self.params["c"]
            x1 = np.log(np.clip(p, CLIP_EPS, 1))
            x2 = np.log(np.clip(1 - p, CLIP_EPS, 1))
            return sigmoid(a * x1 + b0 * x2 + c)
        elif self.kind == "isotonic":
            xs = np.asarray(self.params["xs"], dtype=float)
            ys = np.asarray(self.params["ys"], dtype=float)
            p2 = np.clip(p, xs.min(), xs.max())
            return np.interp(p2, xs, ys)
        elif self.kind == "temperature":
            T = self.params["T"]
            return sigmoid(logit(p) / T)
        elif self.kind == "identity":
            return p
        else:
            raise ValueError(f"Unknown calibrator kind: {self.kind}")

class CalibratorSelector:
    def __init__(self, random_state: int = SEED):
        self.random_state = random_state
        self.best_: CalibModel | None = None

    @staticmethod
    def _fit_platt(p: np.ndarray, y: np.ndarray) -> CalibModel:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=SEED)
        lr.fit(logit(p).reshape(-1, 1), y)
        return CalibModel(
            "platt", {"w": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        )

    @staticmethod
    def _fit_beta(p: np.ndarray, y: np.ndarray) -> CalibModel:

        p = np.clip(p, CLIP_EPS, 1 - CLIP_EPS)
        X = np.column_stack([np.log(p), np.log(1 - p)])
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000, random_state=SEED)
        lr.fit(X, y)
        a = float(lr.coef_[0, 0])
        b = float(lr.coef_[0, 1])
        c = float(lr.intercept_[0])
        return CalibModel("beta", {"a": a, "b": b, "c": c})

    @staticmethod
    def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> CalibModel:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p, y)
        return CalibModel(
            "isotonic",
            {"xs": ir.X_thresholds_.tolist(), "ys": ir.y_thresholds_.tolist()},
        )

    @staticmethod
    def _fit_temperature(p: np.ndarray, y: np.ndarray) -> CalibModel:
        def objective(T):
            p_adj = sigmoid(logit(p) / T)
            return (
                0.5 * (1 - roc_auc_score(y, p_adj))
                + 0.25 * mean_squared_error(y, p_adj)
                + 0.25 * ece_fast(y, p_adj)
            )

        T_best = golden_section_minimize(objective, 0.5, 10.0)
        return CalibModel("temperature", {"T": float(T_best)})

    def fit_select(self, p_oof: np.ndarray, y: np.ndarray) -> CalibModel:
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        def eval_model(fit_fn):
            scores = []
            for tr, va in kf.split(p_oof, y):
                calib = fit_fn(p_oof[tr], y[tr])
                p_hat = calib.apply(p_oof[va])
                s = (
                    0.5 * (1 - roc_auc_score(y[va], p_hat))
                    + 0.25 * mean_squared_error(y[va], p_hat)
                    + 0.25 * ece_fast(y[va], p_hat)
                )
                scores.append(s)
            return np.mean(scores)

        cands = {
            "platt": self._fit_platt,
            "isotonic": self._fit_isotonic,
            "beta": self._fit_beta,
            "temperature": self._fit_temperature,
        }
        best_name, best_score = None, 1e9
        for name, fit_fn in cands.items():
            s = eval_model(fit_fn)
            if s < best_score:
                best_name, best_score = name, s
        self.best_ = cands[best_name](p_oof, y)
        return self.best_
