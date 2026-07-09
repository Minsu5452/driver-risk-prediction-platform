import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.core.constants import MAX_SHAP_SAMPLES
from src.inference.loader import get_artifacts

logger = logging.getLogger(__name__)
from src.data.preprocessor import (
    preprocess_A,
    preprocess_B,
    encode_testdate,
    encode_age,
)
from src.data.features import cohort_key_from_age
from src.inference.feature_construction import (
    _stitch_by_pk_strict_past,
    _stitch_by_key_strict_past,
)
from src.models.metrics import sigmoid, logit


def _get_tree_explainer(domain: str, name: str, fold_idx: int, model):
    """도메인·모델·폴드별 SHAP TreeExplainer를 캐시에서 가져오거나 1회 생성한다.

    TreeExplainer는 모델 트리 구조에만 의존하고 입력 X와 무관(stateless)하므로,
    한 번 만들어 두면 모든 요청에서 안전하게 재사용할 수 있다. 매 호출마다
    재생성하던 비용(트리 앙상블 파싱)을 제거하며 SHAP 값은 동일하다.
    """
    import shap

    artifacts = get_artifacts()
    dcache = artifacts.explainers.setdefault(domain, {})
    lst = dcache.setdefault(name, [])
    while len(lst) <= fold_idx:
        lst.append(None)
    if lst[fold_idx] is None:
        lst[fold_idx] = shap.TreeExplainer(model)
    return lst[fold_idx]


def _build_personal_features_mem(domain: str, df_proc: pd.DataFrame) -> pd.DataFrame:
    artifacts = get_artifacts()
    cfg = artifacts.personal_configs.get(domain)
    if not cfg:
        raise ValueError(f"Personal config for {domain} not found.")

    pi0 = float(cfg["pi0"])
    m_s = float(cfg["m_smooth"])
    gamma = float(cfg["gamma"])
    alpha = pi0 * m_s
    beta = (1.0 - pi0) * m_s

    q = df_proc[["Test_id", "PrimaryKey", "month_idx"]].rename(
        columns={"month_idx": "mi_left"}
    )
    q = q.assign(
        Test_id=q["Test_id"].astype(str),
        PrimaryKey=q["PrimaryKey"].astype(str),
        mi_left=pd.to_numeric(q["mi_left"], errors="coerce").astype("float64"),
    )

    s = artifacts.personal_snapshots.get(domain)
    flookup = artifacts.fast_personal_lookups.get(domain)

    if flookup is not None:
        keys = q["PrimaryKey"].values.astype(str)
        times = q["mi_left"].values.astype(float)
        stitched = flookup.query_batch(keys, times)
        # 호환성 별칭
        if "mi_snap" in stitched:
            stitched["snap_mi"] = stitched["mi_snap"]
    elif s is not None:
        stitched = _stitch_by_pk_strict_past(
            q, s, left_key="mi_left", right_key="mi_snap"
        )
    else:
        raise ValueError(f"Personal snapshot/lookup for {domain} not found.")

    mi_left = q["mi_left"].to_numpy(np.float64)
    mi_snap = stitched["snap_mi"]
    delta = np.maximum(np.nan_to_num(mi_left - mi_snap, nan=0.0), 0.0)
    decay = np.power(gamma, delta)

    S_adj = stitched["S_snap"] * decay
    C_adj = stitched["C_snap"] * decay
    denom = (alpha + beta) + np.maximum(C_adj, 1e-15)

    pi_cum = stitched["pi_cum"]
    drv_prior = np.full_like(mi_left, np.nan, dtype=np.float64)
    use_pic = ~np.isnan(pi_cum)
    drv_prior[use_pic] = pi_cum[use_pic]
    use_sc = ~use_pic & (~np.isnan(S_adj)) & (~np.isnan(C_adj))
    drv_prior[use_sc] = (alpha + S_adj[use_sc]) / np.maximum(denom[use_sc], 1e-15)

    drv_prior_w = np.full_like(mi_left, np.nan, dtype=np.float64)
    ok_w = (~np.isnan(S_adj)) & (~np.isnan(C_adj))
    drv_prior_w[ok_w] = (alpha + S_adj[ok_w]) / np.maximum(denom[ok_w], 1e-15)
    n_eff_w = np.maximum(C_adj, 0.0)

    out = pd.DataFrame(
        {
            "Test_id": q["Test_id"].values.astype(str),
            "drv_prior": drv_prior.astype("float32"),
            "drv_prior_w": drv_prior_w.astype("float32"),
            "drv_n_eff_w": n_eff_w.astype("float32"),
            "drv_last_gap_mon": (mi_left - mi_snap).astype("float32"),
            "drv_n_hist": np.nan_to_num(stitched.get("n_cum"), nan=0.0).astype("int32"),
            "drv_gamma": np.float32(gamma),
        }
    )

    pp = artifacts.personal_priors.get(domain)
    if pp is not None:
        aux = q[["Test_id", "PrimaryKey", "mi_left"]].merge(
            pp[["PrimaryKey", "first_mi", "pi_last_w", "pi"]],
            on="PrimaryKey",
            how="left",
            validate="m:1",
        )
        first_mi = pd.to_numeric(aux["first_mi"], errors="coerce").astype("float64")
        elapsed = (aux["mi_left"].astype("float64") - first_mi).astype("float32")
        out = out.merge(
            pd.DataFrame(
                {
                    "Test_id": aux["Test_id"].values.astype(str),
                    "drv_first_mi": first_mi.astype("float32"),
                    "drv_elapsed_since_first": elapsed,
                }
            ),
            on="Test_id",
            how="left",
            validate="1:1",
        )

        p_fb = pd.to_numeric(aux["pi_last_w"], errors="coerce").astype("float64")
        p_fb = np.where(
            np.isnan(p_fb),
            pd.to_numeric(aux["pi"], errors="coerce").astype("float64"),
            p_fb,
        )
        p_fb = np.where(np.isnan(p_fb), float(pi0), p_fb)

        m1 = out["drv_prior_w"].isna().to_numpy()
        if np.any(m1):
            out.loc[m1, "drv_prior_w"] = p_fb[m1].astype("float32")
        m2 = out["drv_prior"].isna().to_numpy()
        if np.any(m2):
            out.loc[m2, "drv_prior"] = p_fb[m2].astype("float32")
    else:
        out["drv_first_mi"] = np.nan
        out["drv_elapsed_since_first"] = np.nan

    return df_proc.merge(out, on="Test_id", how="left", validate="1:1")


def _attach_cross_prior_mem(domain: str, df: pd.DataFrame) -> pd.DataFrame:
    artifacts = get_artifacts()

    if domain == "A":
        other = "B"
        raw_pi_col = "cross_from_B_raw_pi"
        n_col = "cross_from_B_n"
        last_gap = "cross_from_B_last_gap"
        has_hist = "cross_from_B_has_hist"
        piw_col = "cross_from_B_pi_w"
        nw_col = "cross_from_B_n_eff_w"
    else:
        other = "A"
        raw_pi_col = "cross_from_A_raw_pi"
        n_col = "cross_from_A_n"
        last_gap = "cross_from_A_last_gap"
        has_hist = "cross_from_A_has_hist"
        piw_col = "cross_from_A_pi_w"
        nw_col = "cross_from_A_n_eff_w"

    cfg = artifacts.personal_configs.get(other)
    if not cfg:
        raise ValueError(f"Personal config for {other} not found.")
    pi0 = float(cfg["pi0"])
    m_s = float(cfg["m_smooth"])
    gamma = float(cfg["gamma"])
    alpha = pi0 * m_s
    beta = (1.0 - pi0) * m_s

    s = artifacts.personal_snapshots.get(other)
    flookup = artifacts.fast_personal_lookups.get(other)

    q = df[["Test_id", "PrimaryKey", "month_idx"]].rename(
        columns={"month_idx": "mi_left"}
    )
    q = q.assign(
        Test_id=q["Test_id"].astype(str),
        PrimaryKey=q["PrimaryKey"].astype(str),
        mi_left=pd.to_numeric(q["mi_left"], errors="coerce").astype("float64"),
    )

    if flookup is not None:
        keys = q["PrimaryKey"].values.astype(str)
        times = q["mi_left"].values.astype(float)
        stitched = flookup.query_batch(keys, times)
        if "mi_snap" in stitched:
            stitched["snap_mi"] = stitched["mi_snap"]
    elif s is not None:
        stitched = _stitch_by_pk_strict_past(
            q, s, left_key="mi_left", right_key="mi_snap"
        )
    else:
        raise ValueError(f"Personal snapshot/lookup for {other} not found.")
    mi_left = q["mi_left"].to_numpy(np.float64)
    mi_snap = stitched["snap_mi"]
    delta = np.maximum(np.nan_to_num(mi_left - mi_snap, nan=0.0), 0.0)
    decay = np.power(gamma, delta)
    S_adj = stitched["S_snap"] * decay
    C_adj = stitched["C_snap"] * decay
    denom = alpha + beta + np.maximum(C_adj, 1e-15)
    pi_w = (alpha + S_adj) / denom
    n_eff_w = np.maximum(C_adj, 0.0)

    raw_pi = np.full_like(S_adj, np.nan, dtype=np.float64)
    if not np.all(np.isnan(stitched.get("pi_cum", []))):
        raw_pi = stitched["pi_cum"].astype(np.float64)
    need_sc = np.isnan(raw_pi)
    raw_pi[need_sc] = (alpha + S_adj[need_sc]) / np.maximum(
        alpha + beta + np.maximum(C_adj[need_sc], 0.0), 1e-15
    )

    out = pd.DataFrame(
        {
            "Test_id": q["Test_id"].values.astype(str),
            raw_pi_col: raw_pi.astype("float32"),
            n_col: np.array(stitched.get("n_cum"), dtype=np.float64).astype("float32"),
            last_gap: (mi_left - mi_snap).astype("float32"),
            piw_col: pi_w.astype("float32"),
            nw_col: n_eff_w.astype("float32"),
            has_hist: (~np.isnan(stitched.get("n_cum"))).astype("int8"),
        }
    )

    df = df.merge(out, on="Test_id", how="left", validate="1:1")
    if "drv_prior_w" in df.columns:
        df[piw_col] = df[piw_col].fillna(df["drv_prior_w"])
    if "drv_prior" in df.columns:
        df[piw_col] = df[piw_col].fillna(df["drv_prior"])
        df[raw_pi_col] = df[raw_pi_col].fillna(df["drv_prior"])
    df[nw_col] = df[nw_col].fillna(0.0).astype("float32")
    return df


def _build_cohort_features_mem(domain: str, df_proc: pd.DataFrame) -> pd.DataFrame:
    artifacts = get_artifacts()
    cfg = artifacts.cohort_configs.get(domain)
    if not cfg:
        raise ValueError(f"Cohort config for {domain} not found.")
    pi0 = float(cfg["pi0"])
    m_s = float(cfg["m_smooth"])
    gamma = float(cfg["gamma"])
    alpha = pi0 * m_s
    beta = (1.0 - pi0) * m_s

    cohort_keys = (
        df_proc["CohortKey"] if "CohortKey" in df_proc.columns
        else cohort_key_from_age(df_proc["Age"])
    )
    yi_left = pd.to_numeric(df_proc["year"], errors="coerce").astype("float64")
    q = pd.DataFrame(
        {"CohortKey": cohort_keys.astype(str).values, "yi_left": yi_left.values}
    )

    s = artifacts.cohort_snapshots.get(domain)
    clookup = artifacts.fast_cohort_lookups.get(domain)

    if clookup is not None:
        keys = q["CohortKey"].values.astype(str)
        times = q["yi_left"].values.astype(float)
        stitched = clookup.query_batch(keys, times)
        if "yi_snap" in stitched:
            stitched["snap_mi"] = stitched["yi_snap"]
    elif s is not None:
        stitched = _stitch_by_key_strict_past(
            q, s, key_col="CohortKey", left_key="yi_left", right_key="yi_snap"
        )
        # 코호트용 키 매핑 보정
        if "mi_snap" in stitched:
            stitched["snap_mi"] = stitched["mi_snap"]
    else:
        raise ValueError(f"Cohort snapshot/lookup for {domain} not found.")
    snap_yi = np.asarray(stitched["snap_mi"], dtype=np.float64)
    delta = np.maximum(yi_left.to_numpy(np.float64) - snap_yi, 0.0)
    decay = np.power(gamma, delta)
    S_adj = np.asarray(stitched["S_snap"], dtype=np.float64) * decay
    C_adj = np.asarray(stitched["C_snap"], dtype=np.float64) * decay
    denom = (alpha + beta) + np.maximum(C_adj, 1e-15)
    coh_prior_w = ((alpha + S_adj) / denom).astype("float32")
    coh_n_eff_w = np.maximum(C_adj, 0.0).astype("float32")
    coh_n_hist = np.nan_to_num(stitched.get("n_cum"), nan=0.0).astype("int32")

    raw_pi = np.asarray(stitched.get("pi_cum"), dtype=np.float64)
    if raw_pi.dtype == object or raw_pi.shape[0] == 0:
        raw_pi = np.full_like(S_adj, np.nan, dtype=np.float64)
    need_sc = np.isnan(raw_pi)
    raw_pi[need_sc] = (alpha + S_adj[need_sc]) / np.maximum(
        (alpha + beta) + np.maximum(C_adj[need_sc], 0.0), 1e-15
    )
    coh_prior = raw_pi.astype("float32")

    pp = artifacts.cohort_priors.get(domain)
    if pp is not None:
        aux = pd.DataFrame(
            {
                "CohortKey": cohort_keys.astype(str).values,
                "Test_id": df_proc["Test_id"].astype(str).values,
                "yi_left": yi_left.values,
            }
        ).merge(
            pp[["CohortKey", "first_yi"]], on="CohortKey", how="left", validate="m:1"
        )
        first_yi = pd.to_numeric(aux["first_yi"], errors="coerce").astype("float64")
        elapsed = (aux["yi_left"].astype("float64") - first_yi).astype("float32")
    else:
        first_yi = np.full(len(df_proc), np.nan)
        elapsed = np.full(len(df_proc), np.nan)

    coh_prior_w = np.where(np.isnan(coh_prior_w), pi0, coh_prior_w).astype("float32")
    coh_prior = np.where(np.isnan(coh_prior), pi0, coh_prior).astype("float32")
    coh_last_gap_year = np.where(np.isnan(snap_yi), np.nan, delta).astype("float32")

    out = pd.DataFrame(
        {
            "Test_id": df_proc["Test_id"].astype(str).values,
            "coh_prior": coh_prior,
            "coh_prior_w": coh_prior_w,
            "coh_n_eff_w": coh_n_eff_w,
            "coh_n_hist": coh_n_hist,
            "coh_last_gap_year": coh_last_gap_year,
            "coh_first_yi": first_yi.astype("float32"),
            "coh_elapsed_since_first_year": elapsed.astype("float32"),
            "coh_gamma": np.full(len(df_proc), float(gamma), dtype=np.float32),
        }
    )
    return df_proc.merge(out, on="Test_id", how="left", validate="1:1")


def _prepare_feature_matrix(domain: str, df_input: pd.DataFrame):
    artifacts = get_artifacts()

    df = df_input.copy()
    if "Test_id" not in df.columns:
        df["Test_id"] = [f"TEST_{i}" for i in range(len(df))]

    if "PrimaryKey" not in df.columns:
        raise ValueError("Input must contain PrimaryKey")

    df = preprocess_A(df) if domain == "A" else preprocess_B(df)
    df = pd.concat([df, encode_testdate(df["TestDate"])], axis=1)
    df = pd.concat([df, encode_age(df["Age"])], axis=1)

    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    df = _build_personal_features_mem(domain, df)
    df = _attach_cross_prior_mem(domain, df)
    df = _build_cohort_features_mem(domain, df)

    feat_cols = artifacts.feature_cols.get(domain)
    if not feat_cols:
        raise ValueError(f"Feature cols for {domain} not found.")

    cols_to_use = [c for c in feat_cols if c not in ["__NA_COUNT__", "__NA_RATIO__"]]
    present = [c for c in cols_to_use if c in df.columns]
    X_base = df[present].replace([np.inf, -np.inf], np.nan).astype("float32")
    for c in cols_to_use:
        if c not in X_base.columns:
            X_base[c] = np.nan
    n_cols = len(cols_to_use)
    na_cnt = X_base.isna().sum(axis=1).astype("float32")
    X_base["__NA_COUNT__"] = na_cnt
    X_base["__NA_RATIO__"] = (na_cnt / max(n_cols, 1)).astype("float32")
    X = X_base.reindex(columns=feat_cols, fill_value=np.nan).astype("float32")

    return df, X


def predict_dataframe(domain: str, df_input: pd.DataFrame) -> pd.DataFrame:
    artifacts = get_artifacts()

    df, X = _prepare_feature_matrix(domain, df_input)

    models_dict = artifacts.models.get(domain, {})
    calibrators = artifacts.calibrators.get(domain, {})
    ens_cfg = artifacts.ensemble_configs.get(domain, {})
    model_names = ens_cfg.get("model_names", [])
    weights = np.asarray(ens_cfg.get("weights", []), dtype=float)
    T_final = float(ens_cfg.get("T_final", 1.0))

    cal_probs = {}
    for name in model_names:
        folds = models_dict.get(name, [])
        if not folds:
            continue

        ps = []
        for model in folds:
            p = model.predict_proba(X)[:, 1]
            ps.append(p)
        p_raw = np.mean(ps, axis=0)

        calib = calibrators.get(name)
        p_cal = p_raw if calib is None else calib.apply(p_raw)
        cal_probs[name] = p_cal.astype(float)

    Z = np.zeros(len(X), dtype=float)
    for w, name in zip(weights, model_names):
        if name in cal_probs:
            Z += w * logit(cal_probs[name])

    y_hat = sigmoid(Z / T_final)

    return pd.DataFrame(
        {"Test_id": df["Test_id"].values, "Label": y_hat.astype("float32")}
    )


def explain_dataframe(
    domain: str, df_input: pd.DataFrame, detailed: bool = False, sample: bool = True
) -> Dict[str, Any]:
    """예측에 대한 SHAP 값을 계산한다.

    Args:
        sample: True(기본)면 글로벌 평균용으로 MAX_SHAP_SAMPLES 초과 시 층화 샘플링한다.
            개인별 보고서처럼 전원의 실제 SHAP이 필요한 경우 False로 호출해야 한다.
            (False일 때 대량 입력은 호출측에서 청크로 나눠 메모리를 관리할 것)

    Returns:
        다음 키를 포함하는 Dict:
            - shap_values: 피처별 SHAP 값 배열
            - base_values: 베이스 값 목록
            - feature_names: 피처명 목록
            - data: 전처리된 입력 피처 행렬 (X)의 레코드
            - Test_id: Test_id 목록
    """
    import shap

    artifacts = get_artifacts()

    df, X = _prepare_feature_matrix(domain, df_input)

    # 글로벌 SHAP은 샘플링으로 충분 — 대량 데이터 시 성능 문제 방지
    if sample and len(X) > MAX_SHAP_SAMPLES:
        # Stratified sampling: age_ord5 컬럼으로 연령대 비율을 유지하며 sampling
        # 고령층·소수 연령대도 적절히 반영
        if "age_ord5" in df.columns:
            rng = np.random.RandomState(42)
            group_col = df["age_ord5"]
            sampled_indices = []
            for _, g_idx in df.groupby(group_col).groups.items():
                n_take = max(1, int(MAX_SHAP_SAMPLES * len(g_idx) / len(df)))
                chosen = rng.choice(g_idx, size=min(len(g_idx), n_take), replace=False)
                sampled_indices.extend(chosen)
            # 오차 보정: 목표 수보다 많으면 trim, 적으면 보충
            if len(sampled_indices) > MAX_SHAP_SAMPLES:
                sampled_indices = list(rng.choice(sampled_indices, MAX_SHAP_SAMPLES, replace=False))
            elif len(sampled_indices) < MAX_SHAP_SAMPLES:
                remaining = list(set(range(len(df))) - set(sampled_indices))
                extra = rng.choice(remaining, size=min(len(remaining), MAX_SHAP_SAMPLES - len(sampled_indices)), replace=False)
                sampled_indices.extend(extra)
            sample_idx = np.array(sampled_indices)
        else:
            sample_idx = np.random.RandomState(42).choice(len(X), MAX_SHAP_SAMPLES, replace=False)
        logger.info(f"SHAP stratified sampling: {len(X)} → {len(sample_idx)} records")
        X = X.iloc[sample_idx].reset_index(drop=True)
        df = df.iloc[sample_idx].reset_index(drop=True)

    ens_cfg = artifacts.ensemble_configs.get(domain, {})
    model_names = ens_cfg.get("model_names", [])
    weights = np.asarray(ens_cfg.get("weights", []), dtype=float)

    if len(weights) > 0:
        weights = weights / np.sum(weights)

    total_shap_values = np.zeros(X.shape, dtype=float)
    total_base_value = 0.0

    model_details = {}

    valid_models_count = 0

    for idx, name in enumerate(model_names):
        models_dict = artifacts.models.get(domain, {})
        folds = models_dict.get(name, [])

        weight = weights[idx] if len(weights) > idx else 1.0 / len(model_names)

        if not folds or weight == 0:
            continue

        fold_shap_values = np.zeros(X.shape, dtype=float)
        fold_base_value = 0.0
        n_folds = 0

        for fold_idx, model in enumerate(folds):
            try:

                explainer = _get_tree_explainer(domain, name, fold_idx, model)
                shap_vals = explainer.shap_values(X, check_additivity=False)

                if isinstance(shap_vals, list):

                    if len(shap_vals) == 2:
                        shap_vals = shap_vals[1]
                    else:
                        shap_vals = shap_vals[0]

                fold_shap_values += shap_vals

                ev = explainer.expected_value
                if isinstance(ev, list) or isinstance(ev, np.ndarray):
                    if len(ev) == 2:
                        ev = ev[1]
                    else:
                        ev = ev[0]
                fold_base_value += ev
                n_folds += 1
            except Exception as e:
                logger.debug(f"SHAP calculation skipped: {e}")

        if n_folds > 0:

            avg_fold_shap = fold_shap_values / n_folds
            avg_fold_base = fold_base_value / n_folds

            total_shap_values += weight * avg_fold_shap
            total_base_value += weight * avg_fold_base
            valid_models_count += 1

            if detailed:
                model_details[name] = {
                    "shap_values": avg_fold_shap.tolist(),
                    "base_value": (
                        float(avg_fold_base)
                        if isinstance(avg_fold_base, (float, np.floating))
                        else 0.0
                    ),
                    "weight": float(weight),
                }

    if valid_models_count == 0:
        total_shap_values = np.zeros(X.shape)

    return {
        "shap_values": total_shap_values.tolist(),
        "base_value": (
            float(total_base_value)
            if isinstance(total_base_value, (float, np.floating))
            else 0.0
        ),
        "model_details": model_details if detailed else None,
        "feature_names": X.columns.tolist(),
        "data": X.to_dict(orient="records"),
    }
