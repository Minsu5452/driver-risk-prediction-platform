import gc
import os
import logging
import warnings
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Mean of empty slice"
)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Degrees of freedom")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="A column-vector y was passed"
)

from src.utils.optimization_utils import (
    str_to_num_array,
    fast_seq_mean,
    fast_seq_std,
    fast_masked_mean,
    fast_masked_mean_in_set,
)
from src.core.constants import (
    SEED, N_JOBS, BASE_DIR, A_SEQ_COLS, A_INT_COLS, A_RAW_COLS,
    B_SEQ_COLS, B_SCORE_COLS, B_RAW_COLS, SEQ_PARSE_JOBS, SEQ_DIFF_CHUNK,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from src.models.metrics import compute_metrics

logger = logging.getLogger("ai-engine")
import joblib
from src.data.loader import load_domain_train


def preprocess_A(train_A):
    df = train_A  # copy 불필요: 함수 내에서 df 컬럼을 읽기만 함

    cols_to_parse = A_SEQ_COLS

    logger.info("Step 1: A 컬럼 파싱 (Parallel)...")
    from joblib import Parallel, delayed

    def _parse_helper(col, series):
        return col, str_to_num_array(series)

    valid_cols = [c for c in cols_to_parse if c in df.columns]

    parsed_items = Parallel(n_jobs=SEQ_PARSE_JOBS)(
        delayed(_parse_helper)(col, df[col]) for col in valid_cols
    )
    parsed = dict(parsed_items)
    del parsed_items

    feats = {}

    logger.info("Step 2: A1 feature 생성...")

    def get_rate(col, target="1"):
        """특정 값의 비율 계산"""
        arr = parsed[col]
        target_val = float(target)
        matches = np.sum(arr == target_val, axis=1)
        totals = np.sum(~np.isnan(arr), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return matches / np.where(totals == 0, np.nan, totals)

    def get_mean(col):
        """평균 계산"""
        return fast_seq_mean(parsed[col])

    def get_std(col):
        """표준편차 계산"""
        return fast_seq_std(parsed[col])

    def get_masked_mean(cond_col, val_col, mask_val):
        """조건부 마스킹 평균 계산"""
        return fast_masked_mean(parsed[cond_col], parsed[val_col], mask_val)

    def get_masked_mean_set(cond_col, val_col, mask_set):
        """집합 조건 마스킹 평균 계산"""
        return fast_masked_mean_in_set(parsed[cond_col], parsed[val_col], mask_set)

    feats["A1_resp_rate"] = get_rate("A1-3", "1")
    feats["A1_left_resp_rate"] = get_masked_mean("A1-1", "A1-3", 1)
    feats["A1_right_resp_rate"] = get_masked_mean("A1-1", "A1-3", 2)
    feats["A1_resp_rate_side_diff"] = (
        feats["A1_left_resp_rate"] - feats["A1_right_resp_rate"]
    )
    feats["A1_slow_resp_rate"] = get_masked_mean("A1-2", "A1-3", 1)
    feats["A1_normal_resp_rate"] = get_masked_mean("A1-2", "A1-3", 2)
    feats["A1_fast_resp_rate"] = get_masked_mean("A1-2", "A1-3", 3)
    feats["A1_resp_rate_diff"] = feats["A1_slow_resp_rate"] - feats["A1_fast_resp_rate"]

    feats["A1_rt_mean"] = get_mean("A1-4")
    feats["A1_rt_std"] = get_std("A1-4")
    feats["A1_rt_left"] = get_masked_mean("A1-1", "A1-4", 1)
    feats["A1_rt_right"] = get_masked_mean("A1-1", "A1-4", 2)
    feats["A1_rt_side_diff"] = feats["A1_rt_left"] - feats["A1_rt_right"]
    feats["A1_rt_slow"] = get_masked_mean("A1-2", "A1-4", 1)
    feats["A1_rt_normal"] = get_masked_mean("A1-2", "A1-4", 2)
    feats["A1_rt_fast"] = get_masked_mean("A1-2", "A1-4", 3)
    feats["A1_rt_speed_diff"] = feats["A1_rt_slow"] - feats["A1_rt_fast"]

    logger.info("Step 3: A2 feature 생성...")
    feats["A2_resp_rate"] = get_rate("A2-3", "1")
    feats["A2_slow_resp_rate_cond1"] = get_masked_mean("A2-1", "A2-3", 1)
    feats["A2_normal_resp_rate_cond1"] = get_masked_mean("A2-1", "A2-3", 2)
    feats["A2_fast_resp_rate_cond1"] = get_masked_mean("A2-1", "A2-3", 3)
    feats["A2_resp_rate_cond1_diff"] = (
        feats["A2_slow_resp_rate_cond1"] - feats["A2_fast_resp_rate_cond1"]
    )
    feats["A2_slow_resp_rate_cond2"] = get_masked_mean("A2-2", "A2-3", 1)
    feats["A2_normal_resp_rate_cond2"] = get_masked_mean("A2-2", "A2-3", 2)
    feats["A2_fast_resp_rate_cond2"] = get_masked_mean("A2-2", "A2-3", 3)
    feats["A2_resp_rate_cond2_diff"] = (
        feats["A2_slow_resp_rate_cond2"] - feats["A2_fast_resp_rate_cond2"]
    )

    feats["A2_rt_mean"] = get_mean("A2-4")
    feats["A2_rt_std"] = get_std("A2-4")
    feats["A2_rt_slow_cond1"] = get_masked_mean("A2-1", "A2-4", 1)
    feats["A2_rt_normal_cond1"] = get_masked_mean("A2-1", "A2-4", 2)
    feats["A2_rt_fast_cond1"] = get_masked_mean("A2-1", "A2-4", 3)
    feats["A2_rt_cond1_diff"] = feats["A2_rt_slow_cond1"] - feats["A2_rt_fast_cond1"]
    feats["A2_rt_slow_cond2"] = get_masked_mean("A2-2", "A2-4", 1)
    feats["A2_rt_normal_cond2"] = get_masked_mean("A2-2", "A2-4", 2)
    feats["A2_rt_fast_cond2"] = get_masked_mean("A2-2", "A2-4", 3)
    feats["A2_rt_cond2_diff"] = feats["A2_rt_slow_cond2"] - feats["A2_rt_fast_cond2"]

    logger.info("Step 4: A3 feature 생성...")

    a3_5 = parsed["A3-5"]

    total = np.sum(~np.isnan(a3_5), axis=1)
    valid = np.sum((a3_5 == 1.0) | (a3_5 == 2.0), axis=1)
    invalid = np.sum((a3_5 == 3.0) | (a3_5 == 4.0), axis=1)
    correct = np.sum((a3_5 == 1.0) | (a3_5 == 3.0), axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        feats["A3_valid_ratio"] = valid / total
        feats["A3_invalid_ratio"] = invalid / total
        feats["A3_correct_ratio"] = correct / total

    for k in ["A3_valid_ratio", "A3_invalid_ratio", "A3_correct_ratio"]:
        feats[k][np.isinf(feats[k])] = np.nan

    feats["A3_resp2_rate"] = get_rate("A3-6", "1")
    feats["A3_resp2_small_rate"] = get_masked_mean("A3-1", "A3-6", 1)
    feats["A3_resp2_big_rate"] = get_masked_mean("A3-1", "A3-6", 2)
    feats["A3_resp2_size_diff"] = (
        feats["A3_resp2_big_rate"] - feats["A3_resp2_small_rate"]
    )
    feats["A3_resp2_left_rate"] = get_masked_mean("A3-3", "A3-6", 1)
    feats["A3_resp2_right_rate"] = get_masked_mean("A3-3", "A3-6", 2)
    feats["A3_resp2_side_diff"] = (
        feats["A3_resp2_left_rate"] - feats["A3_resp2_right_rate"]
    )

    feats["A3_rt_mean"] = get_mean("A3-7")
    feats["A3_rt_std"] = get_std("A3-7")
    feats["A3_rt_small_mean"] = get_masked_mean("A3-1", "A3-7", 1)
    feats["A3_rt_big_mean"] = get_masked_mean("A3-1", "A3-7", 2)
    feats["A3_rt_size_diff"] = feats["A3_rt_small_mean"] - feats["A3_rt_big_mean"]

    feats["A3_rt_left_mean"] = get_masked_mean("A3-3", "A3-7", 1)
    feats["A3_rt_right_mean"] = get_masked_mean("A3-3", "A3-7", 2)
    feats["A3_rt_side_diff"] = feats["A3_rt_left_mean"] - feats["A3_rt_right_mean"]

    logger.info("Step 5: A4 feature 생성...")
    feats["A4_acc_rate"] = get_rate("A4-3", "1")

    feats["A4_con_acc_rate"] = np.abs(get_masked_mean("A4-1", "A4-3", 1) - 2)
    feats["A4_incon_acc_rate"] = np.abs(get_masked_mean("A4-1", "A4-3", 2) - 2)
    feats["A4_incon_acc_rate_diff"] = (
        feats["A4_con_acc_rate"] - feats["A4_incon_acc_rate"]
    )

    feats["A4_red_acc_rate"] = np.abs(get_masked_mean("A4-2", "A4-3", 1) - 2)
    feats["A4_green_acc_rate"] = np.abs(get_masked_mean("A4-2", "A4-3", 2) - 2)
    feats["A4_color_acc_rate_diff"] = (
        feats["A4_red_acc_rate"] - feats["A4_green_acc_rate"]
    )

    feats["A4_resp2_rate"] = get_rate("A4-4", "1")
    feats["A4_rt_mean"] = get_mean("A4-5")
    feats["A4_rt_std"] = get_std("A4-5")
    feats["A4_rt_con_stroop"] = get_masked_mean("A4-1", "A4-5", 1)
    feats["A4_rt_incon_stroop"] = get_masked_mean("A4-1", "A4-5", 2)
    feats["A4_stroop_diff"] = feats["A4_rt_incon_stroop"] - feats["A4_rt_con_stroop"]

    feats["A4_rt_red"] = get_masked_mean("A4-2", "A4-5", 1)
    feats["A4_rt_green"] = get_masked_mean("A4-2", "A4-5", 2)
    feats["A4_rt_color_diff"] = feats["A4_rt_red"] - feats["A4_rt_green"]

    logger.info("Step 6: A5 feature 생성...")
    feats["A5_acc_rate"] = get_rate("A5-2", "1")
    feats["A5_resp2_rate"] = get_rate("A5-3", "1")
    feats["A5_acc_nonchange"] = get_masked_mean("A5-1", "A5-2", 1)
    feats["A5_acc_change"] = get_masked_mean_set("A5-1", "A5-2", {2, 3, 4})
    feats["A5_resp2_nonchange"] = get_masked_mean("A5-1", "A5-3", 1)
    feats["A5_resp2_change"] = get_masked_mean_set("A5-1", "A5-3", {2, 3, 4})

    del parsed
    gc.collect()
    return pd.DataFrame(feats)


def preprocess_B(train_B):
    df = train_B  # copy 불필요: 함수 내에서 df 컬럼을 읽기만 함

    cols_to_parse = B_SEQ_COLS

    logger.info("Step 1 (B): B 컬럼 파싱 (Parallel)...")
    from joblib import Parallel, delayed

    def _parse_helper(col, series):
        return col, str_to_num_array(series)

    valid_cols = [c for c in cols_to_parse if c in df.columns]

    parsed_items = Parallel(n_jobs=SEQ_PARSE_JOBS)(
        delayed(_parse_helper)(col, df[col]) for col in valid_cols
    )
    parsed = dict(parsed_items)
    del parsed_items

    def get_rate(col, target="1"):
        """특정 값의 비율 계산"""
        arr = parsed[col]
        target_val = float(target)
        matches = np.sum(arr == target_val, axis=1)
        totals = np.sum(~np.isnan(arr), axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return matches / np.where(totals == 0, np.nan, totals)

    def get_mean(col):
        """평균 계산"""
        return fast_seq_mean(parsed[col])

    def get_std(col):
        """표준편차 계산"""
        return fast_seq_std(parsed[col])

    feats = {}

    logger.info("Step 2: B1 feature 생성...")
    feats["B1_acc_task1"] = get_rate("B1-1", "1")
    feats["B1_rt_mean"] = get_mean("B1-2")
    feats["B1_rt_std"] = get_std("B1-2")
    feats["B1_acc_task2"] = get_rate("B1-3", "1")
    feats["B1_acc_task2_nc_ic"] = get_rate("B1-3", "4")

    logger.info("Step 3: B2 feature 생성...")
    feats["B2_acc_task1"] = get_rate("B2-1", "1")
    feats["B2_rt_mean"] = get_mean("B2-2")
    feats["B2_rt_std"] = get_std("B2-2")
    feats["B2_acc_task2"] = get_rate("B2-3", "1")
    feats["B2_acc_task2_nc_ic"] = get_rate("B2-3", "4")

    logger.info("Step 4: B3 feature 생성...")
    feats["B3_acc_rate"] = get_rate("B3-1", "1")
    feats["B3_rt_mean"] = get_mean("B3-2")
    feats["B3_rt_std"] = get_std("B3-2")

    logger.info("Step 5: B4 feature 생성...")

    b4_1 = parsed["B4-1"]
    total = np.sum(~np.isnan(b4_1), axis=1)

    con_mask = (b4_1 == 1.0) | (b4_1 == 2.0)
    con = np.sum(con_mask, axis=1)

    con_correct = np.sum(b4_1 == 1.0, axis=1)

    incon_mask = (b4_1 == 3.0) | (b4_1 == 4.0) | (b4_1 == 5.0) | (b4_1 == 6.0)
    incon = np.sum(incon_mask, axis=1)

    incon_correct = np.sum((b4_1 == 3.0) | (b4_1 == 5.0), axis=1)

    feats["B4_acc_rate"] = get_rate("B4-1", "1")
    with np.errstate(divide="ignore", invalid="ignore"):
        feats["B4_con_acc_rate"] = con_correct / con
        feats["B4_incon_acc_rate"] = incon_correct / incon

    feats["B4_con_acc_rate"][np.isinf(feats["B4_con_acc_rate"])] = np.nan
    feats["B4_incon_acc_rate"][np.isinf(feats["B4_incon_acc_rate"])] = np.nan

    feats["B4_acc_rate_diff"] = feats["B4_con_acc_rate"] - feats["B4_incon_acc_rate"]
    feats["B4_rt_mean"] = get_mean("B4-2")
    feats["B4_rt_std"] = get_std("B4-2")

    logger.info("Step 6: B5 feature 생성...")
    feats["B5_acc_rate"] = get_rate("B5-1", "1")
    feats["B5_rt_mean"] = get_mean("B5-2")
    feats["B5_rt_std"] = get_std("B5-2")

    logger.info("Step 7: B6~B8 feature 생성...")
    feats["B6_acc_rate"] = get_rate("B6", "1")
    feats["B7_acc_rate"] = get_rate("B7", "1")
    feats["B8_acc_rate"] = get_rate("B8", "1")
    feats["B6_8_acc_rate_mean"] = (
        feats["B6_acc_rate"] + feats["B7_acc_rate"] + feats["B8_acc_rate"]
    ) / 3

    del parsed
    gc.collect()
    return pd.DataFrame(feats)


def _safe_div(a, b, eps=1e-6):
    """안전한 나눗셈 (0으로 나눅 방지)"""
    return a / (b + eps)


def add_features_A(df: pd.DataFrame) -> pd.DataFrame:
    """속도-정확도 트레이드오프, 변동계수 등 A 파생 피처 생성"""
    feats = df.copy()
    eps = 1e-6

    feats["A1_speed_acc_tradeoff"] = _safe_div(
        feats["A1_rt_mean"], feats["A1_resp_rate"], eps
    )
    feats["A1_slow_speed_acc_tradeoff"] = _safe_div(
        feats["A1_rt_slow"], feats["A1_slow_resp_rate"], eps
    )
    feats["A1_normal_speed_acc_tradeoff"] = _safe_div(
        feats["A1_rt_normal"], feats["A1_normal_resp_rate"], eps
    )
    feats["A1_fast_speed_acc_tradeoff"] = _safe_div(
        feats["A1_rt_fast"], feats["A1_fast_resp_rate"], eps
    )

    feats["A2_speed_acc_tradeoff"] = _safe_div(
        feats["A2_rt_mean"], feats["A2_resp_rate"], eps
    )
    feats["A2_slow_speed_acc_tradeoff_cond1"] = _safe_div(
        feats["A2_rt_slow_cond1"], feats["A2_slow_resp_rate_cond1"], eps
    )
    feats["A2_normal_speed_acc_tradeoff_cond1"] = _safe_div(
        feats["A2_rt_normal_cond1"], feats["A2_normal_resp_rate_cond1"], eps
    )
    feats["A2_fast_speed_acc_tradeoff_cond1"] = _safe_div(
        feats["A2_rt_fast_cond1"], feats["A2_fast_resp_rate_cond1"], eps
    )
    feats["A2_slow_speed_acc_tradeoff_cond2"] = _safe_div(
        feats["A2_rt_slow_cond2"], feats["A2_slow_resp_rate_cond2"], eps
    )
    feats["A2_normal_speed_acc_tradeoff_cond2"] = _safe_div(
        feats["A2_rt_normal_cond2"], feats["A2_normal_resp_rate_cond2"], eps
    )
    feats["A2_fast_speed_acc_tradeoff_cond2"] = _safe_div(
        feats["A2_rt_fast_cond2"], feats["A2_fast_resp_rate_cond2"], eps
    )

    feats["A3_speed_acc_tradeoff"] = _safe_div(
        feats["A3_rt_mean"], feats["A3_resp2_rate"], eps
    )
    feats["A3_small_speed_acc_tradeoff"] = _safe_div(
        feats["A3_rt_small_mean"], feats["A3_resp2_small_rate"], eps
    )
    feats["A3_big_speed_acc_tradeoff"] = _safe_div(
        feats["A3_rt_big_mean"], feats["A3_resp2_big_rate"], eps
    )
    feats["A3_left_speed_acc_tradeoff"] = _safe_div(
        feats["A3_rt_left_mean"], feats["A3_resp2_left_rate"], eps
    )
    feats["A3_right_speed_acc_tradeoff"] = _safe_div(
        feats["A3_rt_right_mean"], feats["A3_resp2_right_rate"], eps
    )

    feats["A4_speed_acc_tradeoff"] = _safe_div(
        feats["A4_rt_mean"], feats["A4_acc_rate"], eps
    )
    feats["A4_incon_speed_acc_tradeoff"] = _safe_div(
        feats["A4_rt_incon_stroop"], feats["A4_incon_acc_rate"], eps
    )
    feats["A4_con_speed_acc_tradeoff"] = _safe_div(
        feats["A4_rt_con_stroop"], feats["A4_con_acc_rate"], eps
    )
    feats["A4_red_speed_acc_tradeoff"] = _safe_div(
        feats["A4_rt_red"], feats["A4_red_acc_rate"], eps
    )
    feats["A4_green_speed_acc_tradeoff"] = _safe_div(
        feats["A4_rt_green"], feats["A4_green_acc_rate"], eps
    )

    for k in ["A1", "A2", "A3", "A4"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    for name, base in [
        ("A1_rt_side_gap_abs", "A1_rt_side_diff"),
        ("A1_rt_speed_gap_abs", "A1_rt_speed_diff"),
        ("A2_rt_cond1_gap_abs", "A2_rt_cond1_diff"),
        ("A2_rt_cond2_gap_abs", "A2_rt_cond2_diff"),
        ("A4_stroop_gap_abs", "A4_stroop_diff"),
        ("A4_color_gap_abs", "A4_rt_color_diff"),
    ]:
        if base in feats.columns:
            feats[name] = feats[base].abs()

    feats["A3_valid_invalid_gap"] = feats["A3_valid_ratio"] - feats["A3_invalid_ratio"]
    feats["A3_correct_invalid_gap"] = (
        feats["A3_correct_ratio"] - feats["A3_invalid_ratio"]
    )
    feats["A5_change_nonchange_gap"] = (
        feats["A5_acc_change"] - feats["A5_acc_nonchange"]
    )
    feats["A5_resp2_nonchange_gap"] = (
        feats["A5_resp2_change"] - feats["A5_resp2_nonchange"]
    )

    parts = []
    if "A4_stroop_gap_abs" in feats:
        parts.append(0.30 * feats["A4_stroop_gap_abs"].fillna(0))
    if "A4_acc_rate" in feats:
        parts.append(0.20 * (1 - feats["A4_acc_rate"].fillna(0)))
    if "A3_valid_invalid_gap" in feats:
        parts.append(0.20 * feats["A3_valid_invalid_gap"].fillna(0).abs())
    if "A1_rt_cv" in feats:
        parts.append(0.20 * feats["A1_rt_cv"].fillna(0))
    if "A2_rt_cv" in feats:
        parts.append(0.10 * feats["A2_rt_cv"].fillna(0))
    if parts:
        feats["RiskScore"] = sum(parts)

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats


def add_features_B(df: pd.DataFrame) -> pd.DataFrame:
    """속도-정확도 트레이드오프, 변동계수 등 B 파생 피처 생성"""
    feats = df.copy()
    eps = 1e-6

    feats["B1to2_acc_diff"] = feats["B1_acc_task1"] - feats["B2_acc_task1"]
    feats["B1to2_correct_diff"] = feats["B1_acc_task2"] - feats["B2_acc_task2"]

    for k, acc_col, rt_col in [
        ("B1", "B1_acc_task1", "B1_rt_mean"),
        ("B2", "B2_acc_task1", "B2_rt_mean"),
        ("B3", "B3_acc_rate", "B3_rt_mean"),
        ("B4", "B4_acc_rate", "B4_rt_mean"),
        ("B5", "B5_acc_rate", "B5_rt_mean"),
    ]:
        feats[f"{k}_speed_acc_tradeoff"] = _safe_div(feats[rt_col], feats[acc_col], eps)

    for k in ["B1", "B2", "B3", "B4", "B5"]:
        m, s = f"{k}_rt_mean", f"{k}_rt_std"
        feats[f"{k}_rt_cv"] = _safe_div(feats[s], feats[m], eps)

    parts = []
    for k in ["B4", "B5"]:
        parts.append(0.25 * feats[f"{k}_rt_cv"].fillna(0))
    for k in ["B3", "B4", "B5"]:
        acc = f"{k}_acc_rate" if k != "B1" and k != "B2" else None
        if k in ["B1", "B2"]:
            acc = f"{k}_acc_task1"
        if acc in feats:
            parts.append(0.25 * (1 - feats[acc].fillna(0)))
    for k in ["B1", "B2"]:
        tcol = f"{k}_speed_acc_tradeoff"
        if tcol in feats:
            parts.append(0.25 * feats[tcol].fillna(0))
    if parts:
        feats["RiskScore_B"] = sum(parts)

    feats["B9to10_aud_hit"] = (feats["B9-1"] / 50 + feats["B10-1"] / 80) / 2
    feats["B9to10_vis_err"] = (
        feats["B9-5"] / 32 + feats["B10-5"] / 52 + (1 - feats["B10-6"] / 20)
    ) / 3
    feats["B9to10_visaud_err_diff"] = (1 - feats["B9to10_aud_hit"]) - feats[
        "B9to10_vis_err"
    ]

    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feats



def train_seq_cv_no_test(model_dir=None, data_dir=None):
    """테스트 데이터 없이 Rank 7 CV 학습 수행 (관리자 재학습 파이프라인용)."""
    _mdir = model_dir
    _data_dir = data_dir or BASE_DIR

    logger.info("Running Rank 7 CV Training (no test)...")

    from src.data.loader import _read_train_file
    _meta_path = os.path.join(_data_dir, "train.parquet")
    if not os.path.exists(_meta_path):
        _meta_path = os.path.join(_data_dir, "train.csv")
    train = _read_train_file(_meta_path)

    _a_path = os.path.join(_data_dir, "train", "A.parquet")
    if not os.path.exists(_a_path):
        _a_path = os.path.join(_data_dir, "train", "A.csv")
    A_df = _read_train_file(_a_path)

    _b_path = os.path.join(_data_dir, "train", "B.parquet")
    if not os.path.exists(_b_path):
        _b_path = os.path.join(_data_dir, "train", "B.csv")
    B_df = _read_train_file(_b_path)
    train = train.merge(
        pd.concat(
            [A_df[["Test_id", "Age"]], B_df[["Test_id", "Age"]]], axis=0
        ).drop_duplicates(),
        on="Test_id"
    )

    train_df = train["Test_id"].str.split("_", n=2, expand=True)
    train_df.columns = ["PrimaryKey", "Test_split", "date"]
    train_df = (
        train_df.assign(
            Test_id=train["Test_id"],
            Label=train["Label"],
            Test=train["Test"],
            Age=train["Age"],
            rank=lambda x: x.groupby("PrimaryKey")["date"].transform("rank"),
        )
        .sort_values(["PrimaryKey", "rank"])
        .reset_index(drop=True)
    )
    train_df["before_label"] = train_df.groupby("PrimaryKey")["Label"].shift(1).fillna(2)
    train_df["before_test"] = train_df.groupby("PrimaryKey")["Test"].shift(1).fillna("new")
    # Age 코드 → 정수 변환 (빈 문자열/NaN 안전 처리)
    age_str = train_df["Age"].fillna("").astype(str)
    age_base = pd.to_numeric(age_str.str.extract(r"(\d+)", expand=False), errors="coerce").fillna(0).astype(int)
    age_suffix = np.where(age_str.str.contains("b", na=False), 5, 0)
    train_df["Age"] = age_base + age_suffix

    # 증강: rank > 1인 행을 before_label=3으로 복사하여 추가
    aug = (
        train_df[train_df["rank"] > 1]
        .copy()
        .reset_index(drop=True)
        .assign(before_label=3)
    )
    aug["PrimaryKey"] = aug["PrimaryKey"] + "_aug"

    merged = pd.concat([train_df, aug], axis=0).reset_index(drop=True)
    merged = merged.sort_values(["PrimaryKey", "rank"]).reset_index(drop=True)
    merged = merged.assign(
        before_label_cummean=lambda x: np.where(
            (x["before_label"] == 2) | (x["before_label"] == 3),
            np.nan,
            x["before_label"],
        )
    ).assign(
        before_label_cummean=lambda x: x.groupby("PrimaryKey")["before_label_cummean"]
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )

    logger.info("base preprocess done (no test)")

    tab_A_df = preprocess_A(A_df)
    a_extra = A_df[[c for c in A_RAW_COLS if c in A_df.columns]].copy()
    for c in A_INT_COLS:
        if c in a_extra.columns:
            a_extra[c] = pd.to_numeric(a_extra[c], errors="coerce").fillna(0)
    tab_A_df = pd.concat([a_extra, tab_A_df], axis=1)

    tab_B_df = preprocess_B(B_df)
    b_extra = B_df[[c for c in B_RAW_COLS if c in B_df.columns]].copy()
    for c in B_SCORE_COLS:
        if c in b_extra.columns:
            b_extra[c] = pd.to_numeric(b_extra[c], errors="coerce").fillna(0)
    tab_B_df = pd.concat([b_extra, tab_B_df], axis=1)

    merged["date"] = pd.to_numeric(merged["date"], errors="coerce").fillna(0).astype(int)
    merged = (
        merged.merge(tab_A_df, how="left")
        .merge(tab_B_df, how="left")
    )

    del A_df, B_df, tab_A_df, tab_B_df
    gc.collect()

    merged = add_features_A(merged)
    merged = add_features_B(merged)
    merged = merged.sort_values(["PrimaryKey", "rank"]).reset_index(drop=True)

    # ── float32 변환: 메모리 절반 절약 (LightGBM은 내부적으로 float32 사용) ──
    non_numeric_cols = {"Label", "Test", "Test_id", "Test_split", "before_label",
                        "before_test", "PrimaryKey", "rank"}
    numeric_cols = [c for c in merged.columns if c not in non_numeric_cols]
    for c in numeric_cols:
        if merged[c].dtype == np.float64:
            merged[c] = merged[c].astype(np.float32)
    gc.collect()

    # numeric_cols 안에 'X'와 'X_diff'(예: A1_resp_rate / A1_resp_rate_diff)가
    # 모두 있으면, 'X' + '_diff_before' == 'X_diff' + '_before' 같은 컬럼명이 되어
    # 충돌한다. 'X_diff'는 cross-sectional 차이(예: slow - fast)이므로 시간차 처리에서
    # 제외한다 — 그래야 'X_diff_before' = X.diff() 의미가 정확히 보존된다.
    _num_set = set(numeric_cols)
    diff_feat = sorted([
        c for c in _num_set
        if not (c.endswith("_diff") and c[:-5] in _num_set)
    ])

    # ── 청크 단위 diff 피처 생성: 메모리 피크 방지 ──
    logger.info("Generating diff features (memory-optimized, chunked)...")
    if diff_feat:
        CHUNK = SEQ_DIFF_CHUNK
        grouped = merged.groupby("PrimaryKey")
        for i in range(0, len(diff_feat), CHUNK):
            chunk_cols = diff_feat[i:i + CHUNK]
            shifted = grouped[chunk_cols].shift(1)
            for col in chunk_cols:
                merged[f"{col}_before"] = shifted[col].astype(np.float32)
                merged[f"{col}_diff_before"] = (merged[col] - shifted[col]).astype(np.float32)
            del shifted
        del grouped
        gc.collect()

    merged = merged.loc[:, ~merged.columns.duplicated()]

    train_df = merged
    del merged

    train_df = train_df.sort_values(["PrimaryKey", "rank"]).reset_index(drop=True)

    feat_cat = ["Test", "before_label", "before_test"]
    feat_con = sorted(
        list(
            set(list(train_df.columns))
            - set(feat_cat)
            - set(["PrimaryKey", "Test_split", "Test_id", "Label"])
        )
    )

    encoder = {}
    train_encoding_out = []
    for x in feat_cat:
        encoder[x] = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        foo = encoder[x].fit_transform(train_df[[x]])
        train_encoding_out.append(
            pd.DataFrame(foo, columns=encoder[x].get_feature_names_out([x]))
        )

    train_X = pd.concat(
        [pd.concat(train_encoding_out, axis=1).astype(np.float32), train_df[feat_con]], axis=1
    )
    train_y = train_df["Label"].values
    # SGKF 그룹은 원본 PrimaryKey로 통일 — augmentation으로 추가된 "_aug" suffix를
    # 그대로 두면 PK1과 PK1_aug를 다른 그룹으로 인식해 같은 사람의 원본/증강 행이
    # 다른 fold로 분리됨 → patient memorization → AUC inflation. 반드시 strip.
    groups = train_df["PrimaryKey"].str.replace("_aug", "", regex=False).values
    # OOF metric은 augmented 행을 제외하고 원본 행에 대해서만 계산 (artificial input 영향 차단)
    is_aug = train_df["PrimaryKey"].str.endswith("_aug").values

    # train_df 메모리 해제 (train_X, train_y, groups, is_aug로 충분)
    test_ids = train_df["Test_id"].copy()
    del train_df, train_encoding_out
    gc.collect()
    seq_dir = os.path.join(_mdir, "seq")
    os.makedirs(seq_dir, exist_ok=True)

    # ── 라벨 분포 확인 ──
    unique_classes, class_counts = np.unique(train_y, return_counts=True)
    class_dist = dict(zip(unique_classes.astype(int), class_counts))
    logger.info(f"Label distribution: {class_dist}")

    if len(unique_classes) < 2:
        logger.warning(f"Only one class ({int(unique_classes[0])}) in training data. "
              f"Skipping Rank 7 CV training.")
        return {"oof_auc": 0.0, "skipped": True, "reason": "single_class"}

    min_class_count = int(min(class_counts))
    n_unique_groups = len(np.unique(groups))
    n_splits = min(5, min_class_count, n_unique_groups)
    if n_splits < 2:
        n_splits = 2
    if n_splits != 5:
        logger.info(f"Adjusted n_splits from 5 to {n_splits} "
              f"(min_class={min_class_count}, groups={n_unique_groups})")

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    # skip된 fold의 행은 0.5(neutral)로 남도록 초기화 — Rank 1처럼 모든 행이 평가에 포함됨
    oof_preds = np.full(len(train_X), 0.5, dtype=float)

    logger.info(f"Starting StratifiedGroupKFold CV (no test, {n_splits} folds)...")

    from src.services.training_service import check_cancelled

    trained_folds = 0
    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(train_X, train_y, groups=groups)
    ):
        check_cancelled()  # 폴드 시작 전 취소 확인
        # Group leakage 방어 — Rank 1 (stack_trainer.py:178-180)과 동일한 패턴.
        # train과 valid가 PrimaryKey를 공유하면 patient memorization으로 metric inflation 발생.
        g_tr = set(groups[tr_idx].tolist())
        g_va = set(groups[va_idx].tolist())
        assert g_tr.isdisjoint(g_va), "[BUG] Group leakage detected in Rank 7: train/valid share PrimaryKey!"

        X_tr, X_va = train_X.iloc[tr_idx], train_X.iloc[va_idx]
        y_tr, y_va = train_y[tr_idx], train_y[va_idx]

        if len(np.unique(y_tr)) < 2:
            logger.warning(f"Fold {fold} training set has only one class, skipping...")
            continue

        model = LGBMClassifier(
            objective="binary",
            metric="auc",
            learning_rate=0.03142274194992809,
            n_estimators=210,
            num_leaves=43,
            max_depth=10,
            min_child_weight=2.4157024864125147,
            subsample=0.8252111894498689,
            colsample_bytree=0.7010909151513369,
            reg_alpha=4.740155157673996,
            reg_lambda=4.655806837391666,
            scale_pos_weight=0.9722795027915864,
            random_state=SEED,
            deterministic=True,
            force_col_wise=True,
            n_jobs=N_JOBS,
            verbose=-1,
        )

        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])

        val_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[va_idx] = val_pred

        if len(np.unique(y_va)) >= 2:
            score = roc_auc_score(y_va, val_pred)
            logger.info(f"Fold {fold} AUC: {score:.6f}")
        else:
            logger.info(f"Fold {fold} trained (validation has single class, AUC skipped)")

        joblib.dump(model, os.path.join(seq_dir, f"model_fold{fold}.pkl"))
        del model, X_tr, X_va, y_tr, y_va
        gc.collect()
        trained_folds += 1

    if trained_folds == 0:
        logger.error("No fold had sufficient class diversity for training.")
        oof_df = pd.DataFrame(
            {"Test_id": test_ids, "prob": np.zeros(len(train_y)), "Label": train_y}
        )
        return {"oof_auc": 0.0, "skipped": True, "reason": "no_valid_folds"}

    # OOF 메트릭은 augmented 행을 제외한 원본 행에 대해서만 계산.
    # augmented 행은 before_label=3이라는 artificial input이라 실제 추론 시나리오를 반영하지 않음.
    real_mask = ~is_aug
    if real_mask.sum() > 0 and len(np.unique(train_y[real_mask])) >= 2:
        oof_metrics = compute_metrics(train_y[real_mask], oof_preds[real_mask])
    else:
        oof_metrics = {"auc": 0.0, "brier": 0.0, "ece": 0.0, "score": 0.0, "mcc": 0.0}
    logger.info(
        f"Rank 7 OOF (no test, {trained_folds}/{n_splits} folds, real={real_mask.sum()}/{len(train_y)}): {oof_metrics}"
    )

    logger.info(f"Rank 7 artifacts saved to {seq_dir} (no test)")
    # oof_auc는 하위 호환성을 위해 유지
    return {"oof_auc": float(oof_metrics["auc"]), **oof_metrics}


if __name__ == "__main__":
    train_seq_cv_no_test()
