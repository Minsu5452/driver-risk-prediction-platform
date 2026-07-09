import os
import gc
import json
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Tuple
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, mean_squared_error

from src.core.constants import (
    SEED,
    N_JOBS,
    CLIP_EPS,
    DOMAIN_CFG,
    EARLY_STOPPING_ROUNDS,
    ENSEMBLE_DIRICHLET_SAMPLES,
    TRAIN_DOMAIN_WORKERS,
)
from src.data.preprocessor import (
    encode_age,
)
from src.data.features import (
    build_personal_timecausal_features,
    build_cross_prior_features_timecausal_by_domain,
    build_cohort_timecausal_features,
    build_cohort_prior_features_timecausal_by_domain,
)
from src.data.loader import load_domain_train
from src.models.factory import make_models

logger = logging.getLogger("ai-engine")
from src.models.calibration import CalibratorSelector
from src.models.ensemble import optimize_weights_and_temperature
from src.models.metrics import logit, sigmoid, compute_metrics


def load_and_merge_domain(
    domain: str,
    df_loaded: pd.DataFrame,
    model_dir: str = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str], pd.DataFrame]:
    _mdir = model_dir

    df = df_loaded
    df = df.loc[:, ~df.columns.duplicated()]

    df = pd.concat([df, encode_age(df["Age"])], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]

    pfe_path = os.path.join(
        _mdir, "stack", "personal", f"{domain}_timecausal_features.parquet"
    )
    pfe = pd.read_parquet(pfe_path)
    if "Test_id" in pfe.columns:
        pfe["Test_id"] = pfe["Test_id"].astype(str)
    df = df.merge(pfe, on="Test_id", how="left", validate="1:1")
    del pfe

    if domain == "A":
        cross_path = os.path.join(
            _mdir, "stack", "personal", "A_cross_from_B_timecausal.parquet"
        )
        cross = pd.read_parquet(cross_path)
        cross["Test_id"] = cross["Test_id"].astype(str)
        df = df.merge(cross, on="Test_id", how="left", validate="1:1")
        del cross
        if "cross_from_B_pi_w" in df.columns:
            df["cross_from_B_pi_w"] = (
                df["cross_from_B_pi_w"]
                .fillna(df["drv_prior_w"])
                .fillna(df["drv_prior"])
            )
            df["cross_from_B_n_eff_w"] = (
                df["cross_from_B_n_eff_w"].fillna(0.0).astype("float32")
            )
        df["cross_from_B_raw_pi"] = df["cross_from_B_raw_pi"].fillna(df["drv_prior"])
    else:
        cross_path = os.path.join(
            _mdir, "stack", "personal", "B_cross_from_A_timecausal.parquet"
        )
        cross = pd.read_parquet(cross_path)
        cross["Test_id"] = cross["Test_id"].astype(str)
        df = df.merge(cross, on="Test_id", how="left", validate="1:1")
        del cross
        if "cross_from_A_pi_w" in df.columns:
            df["cross_from_A_pi_w"] = (
                df["cross_from_A_pi_w"]
                .fillna(df["drv_prior_w"])
                .fillna(df["drv_prior"])
            )
            df["cross_from_A_n_eff_w"] = (
                df["cross_from_A_n_eff_w"].fillna(0.0).astype("float32")
            )
        df["cross_from_A_raw_pi"] = df["cross_from_A_raw_pi"].fillna(df["drv_prior"])

    coh_path = os.path.join(
        _mdir, "stack", "personal", f"{domain}_cohort_timecausal_features.parquet"
    )
    if os.path.exists(coh_path):
        coh = pd.read_parquet(coh_path)
        coh["Test_id"] = coh["Test_id"].astype(str)
        df = df.merge(coh, on="Test_id", how="left", validate="1:1")
        del coh
    else:
        logger.warning(f"코호트 피처 파일 누락: {coh_path}")

    drop_cols = ["Test_id", "Test", "PrimaryKey", "Age", "TestDate", "Label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_base = df[feature_cols].replace([np.inf, -np.inf], np.nan).astype("float32")
    n_cols = X_base.shape[1]
    na_cnt = X_base.isna().sum(axis=1).astype("float32")
    X_base["__NA_COUNT__"] = na_cnt
    X_base["__NA_RATIO__"] = (na_cnt / max(n_cols, 1)).astype("float32")

    y = df["Label"].astype(int).values
    final_feature_cols = list(X_base.columns)

    meta_cols = ["Test_id", "PrimaryKey", "month_idx"]
    if domain == "A":
        meta_cols += [
            "cross_from_B_raw_pi",
            "cross_from_B_n",
            "cross_from_B_last_gap",
            "cross_from_B_has_hist",
            "cross_from_B_pi_w",
            "cross_from_B_n_eff_w",
        ]
    else:
        meta_cols += [
            "cross_from_A_raw_pi",
            "cross_from_A_n",
            "cross_from_A_last_gap",
            "cross_from_A_has_hist",
            "cross_from_A_pi_w",
            "cross_from_A_n_eff_w",
        ]

    meta_cols = [c for c in meta_cols if c in df.columns]
    meta_df = df[meta_cols].copy()
    meta_df["PrimaryKey"] = meta_df["PrimaryKey"].astype(str)

    del df
    gc.collect()
    return X_base, y, feature_cols, final_feature_cols, meta_df


def train_domain(domain_name: str, df_loaded: pd.DataFrame | None = None, model_dir: str = None):
    _mdir = model_dir
    logger.info(f"===== 도메인 {domain_name} 학습 (PrimaryKey 기준 그룹 계층화) =====")
    X_base, y, feature_cols, base_feature_cols, meta_df = load_and_merge_domain(
        domain_name, df_loaded, model_dir=model_dir
    )

    domain_dir = os.path.join(_mdir, "stack", domain_name)
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, "features.json"), "w") as f:

        json.dump(
            {"feature_cols": base_feature_cols, "clip_eps": CLIP_EPS, "seed": SEED},
            f,
            indent=2,
        )

    groups = meta_df["PrimaryKey"].astype(str).values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    models_cfg = make_models(domain_name)
    oof_preds: Dict[str, np.ndarray] = {
        name: np.zeros(len(X_base), dtype=float) for name in models_cfg
    }

    from src.services.training_service import check_cancelled

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X_base, y, groups=groups)):
        check_cancelled()  # 폴드 시작 전 취소 확인
        g_tr = set(groups[tr_idx].tolist())
        g_va = set(groups[va_idx].tolist())
        assert g_tr.isdisjoint(
            g_va
        ), "[BUG] Group leakage detected: train/valid share PrimaryKey!"
        pos_ratio_fold = float((y[tr_idx] == 1).mean())
        logger.info(
            f"[Fold {fold}] train={len(tr_idx)} valid={len(va_idx)}  "
            f"groups(train)={len(g_tr)} groups(valid)={len(g_va)}  "
            f"train_pos_ratio={pos_ratio_fold:.6f}"
        )

        X_tr = X_base.iloc[tr_idx]
        X_va = X_base.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pos_ratio = (y_tr == 1).mean()
        neg_ratio = 1 - pos_ratio
        spw = float(neg_ratio / max(pos_ratio, 1e-6))
        class_weights_cat = [1.0, float(spw)]
        sample_weight_vec = np.where(y_tr == 1, spw, 1.0).astype("float32")

        for name, cfg in models_cfg.items():
            check_cancelled()  # 모델 학습 전 취소 확인
            logger.info(f"  - Training {name}")
            cls = cfg.cls
            params = cfg.params.copy()

            p_va = None
            model = None

            if "xgb" in name:

                params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
                params["scale_pos_weight"] = spw
                model = cls(**params)
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                p_va = model.predict_proba(X_va)[:, 1]

            elif "hgb" in name:
                model = cls(**params)
                model.fit(X_tr, y_tr)
                p_va = model.predict_proba(X_va)[:, 1]

            elif "cat" in name:
                model = cls(**{**params, "class_weights": class_weights_cat})
                model.fit(
                    X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=EARLY_STOPPING_ROUNDS
                )
                p_va = model.predict_proba(X_va)[:, 1]

            else:

                logger.warning(f"알 수 없는 모델 유형 {name}, 범용 fit 시도.")
                model = cls(**params)
                model.fit(X_tr, y_tr)
                p_va = model.predict_proba(X_va)[:, 1]

            mdir = os.path.join(domain_dir, name)
            os.makedirs(mdir, exist_ok=True)

            if hasattr(model, "to_cpu_inplace"):
                model.to_cpu_inplace()

            joblib.dump(model, os.path.join(mdir, f"fold{fold}.pkl"))

            oof_preds[name][va_idx] = p_va
            auc_fold = roc_auc_score(y_va, p_va)
            brier_fold = mean_squared_error(y_va, p_va)
            logger.info(
                f"    -> Fold {fold} {name} AUC={auc_fold:.6f} Brier={brier_fold:.6f}"
            )

        del X_tr, X_va, y_tr, y_va
        gc.collect()

    for name in models_cfg:
        auc_raw = roc_auc_score(y, oof_preds[name])
        brier_raw = mean_squared_error(y, oof_preds[name])
        logger.info(f"[OOF-RAW] {domain_name} {name} AUC={auc_raw:.6f} Brier={brier_raw:.6f}")

    calib_info = {}
    cal_oof = {}
    for name, p in oof_preds.items():
        selector = CalibratorSelector(random_state=SEED)
        best_calib = selector.fit_select(p, y)
        calib_info[name] = {"kind": best_calib.kind, "params": best_calib.params}
        cal_oof[name] = best_calib.apply(p)
        m_raw = compute_metrics(y, p)
        m_cal = compute_metrics(y, cal_oof[name])
        logger.info(
            f"[Calib] {domain_name} {name}: {best_calib.kind} | RAW={m_raw['score']:.6f} -> CAL={m_cal['score']:.6f}"
        )

    with open(os.path.join(domain_dir, "calibrators.json"), "w") as f:
        json.dump(calib_info, f, indent=2)

    names, w_best, T_final, m_final = optimize_weights_and_temperature(
        y, cal_oof, n_samples=ENSEMBLE_DIRICHLET_SAMPLES, n_jobs=N_JOBS
    )
    ens_cfg = {
        "model_names": names,
        "weights": w_best.tolist(),
        "T_final": float(T_final),
        "clip_eps": CLIP_EPS,
        "seed": SEED,
    }
    with open(os.path.join(domain_dir, "ensemble.json"), "w") as f:
        json.dump(ens_cfg, f, indent=2)

    oof_cal_df = pd.DataFrame({"test_id": meta_df["Test_id"].values, "y_true": y})
    for n in oof_preds:
        oof_cal_df[f"{n}_cal"] = cal_oof[n]

    Z = np.zeros_like(y, dtype=float)
    for n, w in zip(names, w_best):
        Z += w * logit(cal_oof[n])
    oof_cal_df["ensemble_temp_scaled"] = sigmoid(Z / T_final)

    oof_cal_df.to_parquet(
        os.path.join(domain_dir, "oof_predictions_calibrated.parquet"), index=False
    )
    with open(os.path.join(domain_dir, "oof_score.json"), "w") as f:
        json.dump(
            {
                "final_ensemble": m_final,
                "models": {
                    name: {
                        "raw": compute_metrics(y, oof_preds[name]),
                        "calibrated": compute_metrics(y, cal_oof[name]),
                        "calibrator": calib_info[name]["kind"],
                    }
                    for name in names
                },
            },
            f,
            indent=2,
        )

    logger.info(
        f"[OOF-FINAL] {domain_name} ENSEMBLE(w={np.round(w_best,4).tolist()}, T={T_final:.3f}) "
        f"Score={m_final['score']:.6f} | AUC={m_final['auc']:.6f} Brier={m_final['brier']:.6f} ECE={m_final['ece']:.6f}"
    )
    del oof_preds, cal_oof, oof_cal_df
    gc.collect()
    logger.info(f"===== 도메인 {domain_name} 학습 완료 =====")


def run_all_training(model_dir: str = None, data_dir: str = None):
    _mdir = model_dir

    M_BY = {d: float(DOMAIN_CFG[d]["M_SMOOTH"]) for d in ["A", "B"]}
    HL_BY = {d: float(DOMAIN_CFG[d]["DECAY_HALF_LIFE"]) for d in ["A", "B"]}

    os.makedirs(os.path.join(_mdir, "stack", "personal"), exist_ok=True)

    logger.info("원시 학습 데이터 로딩 (공유)...")
    A_raw = load_domain_train("A", base_dir=data_dir)
    B_raw = load_domain_train("B", base_dir=data_dir)

    for domain, df_raw in [("A", A_raw), ("B", B_raw)]:
        out_path = os.path.join(
            _mdir, "stack", "personal", f"{domain}_timecausal_features.parquet"
        )

        if os.path.exists(out_path):
            logger.info(
                f"[Stage-A] Skipping personal features for {domain} (Found artifacts)"
            )
        else:
            logger.info(f"[Stage-A] Building time-causal personal features for {domain} ...")
            personal_tc_df = build_personal_timecausal_features(
                domain, m_smooth=M_BY[domain], half_life_months=HL_BY[domain], df=df_raw,
                model_dir=model_dir,
            )
            personal_tc_df.to_parquet(out_path, index=False)
            logger.info(f"Saved: {out_path} shape={personal_tc_df.shape}")
            gc.collect()

    common_cfg = {
        "M_SMOOTH_BY_DOMAIN": M_BY,
        "DECAY_HALF_LIFE_BY_DOMAIN": HL_BY,
        "SEED": SEED,
    }
    with open(os.path.join(_mdir, "stack", "personal", "config.json"), "w") as f:
        json.dump(common_cfg, f, indent=2)

    cross_path_A = os.path.join(
        _mdir, "stack", "personal", "A_cross_from_B_timecausal.parquet"
    )
    if os.path.exists(cross_path_A):
        logger.info("[Stage-A] Skipping cross priors (Found artifacts)")
    else:
        logger.info("[Stage-A] Building cross priors (per-domain smoothing & half-life) ...")
        build_cross_prior_features_timecausal_by_domain(
            m_by_domain=M_BY, hl_by_domain=HL_BY, model_dir=model_dir,
            A_raw=A_raw, B_raw=B_raw,
        )

    for domain, df_raw in [("A", A_raw), ("B", B_raw)]:
        coh_out = os.path.join(
            _mdir, "stack", "personal", f"{domain}_cohort_timecausal_features.parquet"
        )
        if os.path.exists(coh_out):
            logger.info(f"[Stage-A] Skipping cohort features for {domain} (Found artifacts)")
        else:
            logger.info(f"[Stage-A] Building time-causal COHORT features for {domain} ...")
            coh_df = build_cohort_timecausal_features(
                domain, m_smooth=M_BY[domain], half_life_months=HL_BY[domain], df=df_raw,
                model_dir=model_dir,
            )
            coh_df.to_parquet(coh_out, index=False)
            logger.info(f"Saved: {coh_out} shape={coh_df.shape}")
            gc.collect()

    snap_check = os.path.join(_mdir, "stack", "personal", "snap_cohort_A.parquet")
    if os.path.exists(snap_check):
        logger.info("[Stage-A] Skipping cohort snapshot & prior artifacts (Found artifacts)")
    else:
        logger.info("[Stage-A] Building cohort snapshot & prior artifacts (per-domain) ...")
        build_cohort_prior_features_timecausal_by_domain(
            m_by_domain=M_BY, hl_by_domain=HL_BY, model_dir=model_dir,
            A_raw=A_raw, B_raw=B_raw,
        )

    logger.info("[Stage-A] Done. Personal/Cross/Cohort artifacts are ready.")

    # A/B 도메인 병렬 학습 (피처 생성은 위에서 순차 완료, 모델 학습은 독립)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _train_A():
        train_domain("A", df_loaded=A_raw, model_dir=model_dir)

    def _train_B():
        train_domain("B", df_loaded=B_raw, model_dir=model_dir)

    with ThreadPoolExecutor(max_workers=TRAIN_DOMAIN_WORKERS) as pool:
        futures = {pool.submit(_train_A): "A", pool.submit(_train_B): "B"}
        for future in as_completed(futures):
            domain = futures[future]
            future.result()  # 예외 전파
            logger.info(f"도메인 {domain} 학습 완료")

    del A_raw, B_raw
    gc.collect()
    logger.info(
        "전체 학습 완료. 아티팩트 저장 위치: ./model/<A|B>/ 및 ./model/personal/"
    )


if __name__ == "__main__":
    run_all_training()
