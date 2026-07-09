from typing import Dict, Any
import copy
from dataclasses import dataclass
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from src.core.constants import SEED, N_JOBS, DOMAIN_CFG

@dataclass
class ModelConfig:
    cls: Any
    params: Dict[str, Any]

def _deep_update(base: dict, override: dict):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

def make_models(domain: str) -> Dict[str, ModelConfig]:

    hgb_params = dict(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=0.5,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=15,
        class_weight="balanced",
        random_state=SEED,
    )

    xgb_params = dict(
        n_estimators=2500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        tree_method="hist",
        random_state=SEED,
        n_jobs=N_JOBS,
        eval_metric="auc",
        max_delta_step=1,
    )

    cat_params = dict(
        iterations=4000,
        learning_rate=0.05,
        depth=4,
        l2_leaf_reg=3.0,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )

    xgb_alt_params = dict(
        n_estimators=3000,
        learning_rate=0.0065,
        max_depth=4,
        min_child_weight=7,
        subsample=0.69,
        colsample_bytree=0.62,
        reg_lambda=1e-6,
        reg_alpha=7.4,
        objective="binary:logistic",
        tree_method="hist",
        random_state=SEED,
        n_jobs=N_JOBS,
        eval_metric="auc",
        max_delta_step=1,
    )

    models = {
        "hgb_stack_v0": ModelConfig(cls=HistGradientBoostingClassifier, params=hgb_params),
        "xgb_stack_v0": ModelConfig(cls=xgb.XGBClassifier, params=xgb_params),
        "cat_stack_v0": ModelConfig(cls=CatBoostClassifier, params=cat_params),

        "xgb_alt": ModelConfig(cls=xgb.XGBClassifier, params=xgb_alt_params),

        "hgb_stack_v1": ModelConfig(
            cls=HistGradientBoostingClassifier,
            params=_deep_update(
                copy.deepcopy(hgb_params), {"random_state": SEED + 1}
            ),
        ),
        "xgb_stack_v1": ModelConfig(
            cls=xgb.XGBClassifier,
            params=_deep_update(
                copy.deepcopy(xgb_params), {"random_state": SEED + 1}
            ),
        ),
        "cat_stack_v1": ModelConfig(
            cls=CatBoostClassifier,
            params=_deep_update(
                copy.deepcopy(cat_params), {"random_seed": SEED + 1}
            ),
        ),
    }

    overrides = DOMAIN_CFG.get(domain, {}).get("MODELS", {})
    for name, ov in overrides.items():
        if name in models and isinstance(ov, dict):
            _deep_update(models[name].params, ov)

    return models
