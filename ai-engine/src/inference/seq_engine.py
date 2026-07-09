import json
import joblib
import numpy as np
import pandas as pd
import logging
from typing import List, Any
from src.core.constants import A_INT_COLS, A_RAW_COLS, B_SCORE_COLS, B_RAW_COLS
from src.training.seq_trainer import (
    preprocess_A,
    preprocess_B,
    add_features_A,
    add_features_B,
)

logger = logging.getLogger("ai-engine")

_MODELS: List[Any] = []


def load_seq_artifacts(run_id: int = None, **_kwargs):
    """Rank 7용 LightGBM 모델을 DB에서 로드한다."""
    global _MODELS
    expected_folds = 5
    if _MODELS and len(_MODELS) >= expected_folds:
        return

    if run_id is None:
        logger.warning("No run_id provided for Rank 7 model loading.")
        return

    from io import BytesIO
    from src.core.database import list_artifact_keys, load_artifact

    pkl_keys = [k for k in list_artifact_keys(run_id, "seq/") if k.endswith(".pkl")]

    if not pkl_keys:
        logger.warning(f"No Rank 7 models found in DB for run_id={run_id}.")
        return

    logger.info(f"Loading {len(pkl_keys)} Rank 7 models from DB (run_id={run_id})...")
    for key in pkl_keys:
        try:
            data = load_artifact(run_id, key)
            model = joblib.load(BytesIO(data))
            _MODELS.append(model)
        except Exception as e:
            logger.error(f"Failed to load Rank 7 model {key}: {e}")

    logger.info("Rank 7 artifacts loaded.")


def reload_seq_artifacts(run_id: int = None, **_kwargs):
    global _MODELS
    _MODELS = []
    load_seq_artifacts(run_id=run_id)


def _convert_age_like_training(age_series: pd.Series) -> np.ndarray:
    """학습 코드(seq_trainer.py:527-530)와 동일한 Age 정수 변환.
    "30a" → 30, "30b" → 35.
    """
    age_str = age_series.fillna("").astype(str)
    age_base = pd.to_numeric(
        age_str.str.extract(r"(\d+)", expand=False), errors="coerce"
    ).fillna(0).astype(int)
    age_suffix = np.where(age_str.str.contains("b", na=False), 5, 0)
    return (age_base.values + age_suffix).astype(np.int64)


def _build_features_for_domain(domain: str, df: pd.DataFrame) -> pd.DataFrame:
    """현재 검사 데이터의 피처를 구성한다 (학습과 동일)."""
    if domain == "A":
        raw_cols, numeric_cols = A_RAW_COLS, A_INT_COLS
        features_df = preprocess_A(df)
    else:
        raw_cols, numeric_cols = B_RAW_COLS, B_SCORE_COLS
        features_df = preprocess_B(df)

    base_df = df[[c for c in raw_cols if c in df.columns]].copy()
    for c in numeric_cols:
        if c in base_df.columns:
            base_df[c] = pd.to_numeric(base_df[c], errors="coerce").fillna(0)

    merged = pd.concat(
        [base_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
    )
    if domain == "A":
        merged = add_features_A(merged)
    else:
        merged = add_features_B(merged)

    # Age 컬럼 추가 — 학습과 동일한 정수 변환 적용 (seq_trainer.py:527-530)
    if "Age" in df.columns:
        merged["Age"] = _convert_age_like_training(df["Age"].reset_index(drop=True))

    return merged


def _apply_diff_vectorized(merged, prev_features, orig_indices):
    """과거 피처의 before/diff를 벡터화로 계산한다. numpy 배열 + concat 1회."""
    _skip = {"Test_id", "Test", "PrimaryKey", "Test_split"}
    numeric_cols = [c for c in prev_features.columns
                    if c in merged.columns and c not in _skip
                    and prev_features[c].dtype.kind in ('f', 'i')]
    # 학습(seq_trainer.py:599-603)과 동일: 'X'와 'X_diff'가 모두 있으면
    # 'X_diff'는 cross-sectional 차이라 시간차 처리에서 제외 (그래야 X_diff_before == X.diff() 의미 보존).
    _num_set = set(numeric_cols)
    numeric_cols = [
        c for c in numeric_cols
        if not (c.endswith("_diff") and c[:-5] in _num_set)
    ]
    if not numeric_cols:
        return merged

    n_rows = len(merged)
    prev_vals = prev_features[numeric_cols].values.astype(np.float32)
    cur_vals = merged.loc[orig_indices, numeric_cols].values.astype(np.float32)
    diff_vals = cur_vals - prev_vals

    before_cols = [f"{c}_before" for c in numeric_cols]
    diff_cols = [f"{c}_diff_before" for c in numeric_cols]

    # numpy 배열로 한번에 구성 → DataFrame 1회 생성 → concat 1회
    before_arr = np.full((n_rows, len(numeric_cols)), np.nan, dtype=np.float32)
    diff_arr = np.full((n_rows, len(numeric_cols)), np.nan, dtype=np.float32)
    before_arr[orig_indices] = prev_vals
    diff_arr[orig_indices] = diff_vals

    new_df = pd.concat([
        pd.DataFrame(before_arr, index=merged.index, columns=before_cols),
        pd.DataFrame(diff_arr, index=merged.index, columns=diff_cols),
    ], axis=1)

    # merged에 같은 이름의 _before/_diff_before 컬럼이 이미 있으면 새 값으로 교체
    existing = [c for c in new_df.columns if c in merged.columns]
    if existing:
        merged = merged.drop(columns=existing)
    merged = pd.concat([merged, new_df], axis=1)

    return merged


def _get_history_features(df: pd.DataFrame, merged: pd.DataFrame):
    """DB에서 과거 이력을 조회하여 before_label, before_test, *_before, *_diff_before,
    before_label_cummean을 계산한다.
    """
    from src.core.database import get_latest_exam_by_pks, get_label_cummean_by_pks

    # PrimaryKey 추출
    pks = df["PrimaryKey"].unique().tolist() if "PrimaryKey" in df.columns else []
    if not pks:
        merged["before_label_cummean"] = np.nan
        return merged, pd.Series(2.0, index=merged.index), pd.Series("new", index=merged.index)

    # 추론 대상 test_id (self-prev 방지) — DB에 동일 test_id가 있어도 latest로 가져오지 않음
    current_test_ids = df["Test_id"].astype(str).tolist() if "Test_id" in df.columns else []

    # DB에서 각 PK의 최근 검사 기록 배치 조회 (self 제외)
    history_map = get_latest_exam_by_pks(pks, exclude_test_ids=current_test_ids)

    # before_label_cummean (학습 expanding mean의 추론 등가 — PK당 0/1 라벨 평균, self 제외)
    cummean_map = get_label_cummean_by_pks(pks, exclude_test_ids=current_test_ids)

    # 현재 데이터의 PK → index 매핑
    pk_col = df["PrimaryKey"].values if "PrimaryKey" in df.columns else [""] * len(df)

    before_labels = np.full(len(merged), 2.0)   # 기본: 신규 (2)
    before_tests = np.full(len(merged), "new", dtype=object)

    # before_label_cummean 컬럼 채우기 (없으면 NaN — 학습 시 신규/증강이 NaN인 것과 동일)
    cummean_arr = np.array(
        [cummean_map[pk] if pk in cummean_map else np.nan for pk in pk_col],
        dtype=np.float64,
    )
    merged["before_label_cummean"] = cummean_arr

    # 이력이 있는 PK별로 처리
    has_history_mask = np.array([pk in history_map for pk in pk_col])

    if not any(has_history_mask):
        return merged, pd.Series(before_labels, index=merged.index), pd.Series(before_tests, index=merged.index)

    # 과거 기록의 피처를 파싱하여 DataFrame 구성
    prev_rows_a = []
    prev_rows_b = []

    for i, pk in enumerate(pk_col):
        if pk not in history_map:
            continue
        rec = history_map[pk]
        prev_domain = rec["domain"]
        prev_test_id = rec["test_id"]
        prev_age = rec.get("age", "")

        # before_test
        before_tests[i] = prev_domain

        # before_label: DB에 저장된 실제 Label 사용
        prev_label = rec.get("label")
        if prev_label is not None:
            before_labels[i] = float(prev_label)  # 0.0 or 1.0
        else:
            before_labels[i] = 2.0  # 라벨 없음 = 신규 (학습과 동일)

        # 과거 features 파싱
        try:
            feats = json.loads(rec["features_json"]) if rec.get("features_json") else {}
        except (json.JSONDecodeError, TypeError):
            feats = {}

        if feats:
            feats["Test_id"] = prev_test_id
            feats["Age"] = prev_age  # _build_features_for_domain이 정수 변환
            if prev_domain == "A":
                prev_rows_a.append((i, feats))
            else:
                prev_rows_b.append((i, feats))

    # 과거 A 기록을 배치 전처리
    if prev_rows_a:
        prev_a_df = pd.DataFrame([f for _, f in prev_rows_a])
        prev_a_features = _build_features_for_domain("A", prev_a_df)
        orig_indices_a = [idx for idx, _ in prev_rows_a]
        merged = _apply_diff_vectorized(merged, prev_a_features, orig_indices_a)

    # 과거 B 기록을 배치 전처리
    if prev_rows_b:
        prev_b_df = pd.DataFrame([f for _, f in prev_rows_b])
        prev_b_features = _build_features_for_domain("B", prev_b_df)
        orig_indices_b = [idx for idx, _ in prev_rows_b]
        merged = _apply_diff_vectorized(merged, prev_b_features, orig_indices_b)

    return merged, pd.Series(before_labels, index=merged.index), pd.Series(before_tests, index=merged.index)


def predict_dataframe(domain: str, df: pd.DataFrame) -> pd.DataFrame:
    """Rank 7 실시간 추론. DB 이력으로 temporal 피처를 재구성한다."""
    global _MODELS
    if not _MODELS:
        from src.core.database import get_run_id_for_active_version
        load_seq_artifacts(run_id=get_run_id_for_active_version())

    if not _MODELS:
        raise RuntimeError("Rank 7 models not loaded!")

    df = df.reset_index(drop=True)

    # 1. 현재 검사 피처 구성 (학습과 동일)
    merged = _build_features_for_domain(domain, df)

    # 2. DB 이력 기반 temporal 피처 재구성
    merged, before_labels, before_tests = _get_history_features(df, merged)

    # 3. 모델이 기대하는 피처로 정렬
    expected_features = _MODELS[0].feature_name_
    # NaN default — 학습 시 NaN이던 컬럼(도메인 cross 등)을 학습과 동일하게 NaN으로 보존.
    # LightGBM은 NaN을 학습 시 본 분포대로 처리한다.
    final_X = pd.DataFrame(np.nan, index=merged.index, columns=expected_features)

    # OneHot 컬럼은 학습 시 매칭 안 되는 것이 0이므로, NaN 대신 0으로 default
    onehot_prefixes = ("Test_", "before_label_", "before_test_")
    onehot_cols = [c for c in expected_features if c.startswith(onehot_prefixes)]
    if onehot_cols:
        final_X[onehot_cols] = 0.0

    # 일치하는 피처 채우기
    for c in merged.columns:
        if c in final_X.columns:
            final_X[c] = merged[c]

    # Test 원-핫 처리
    if "Test" in df.columns:
        test_val = df["Test"].astype(str)
        for idx, val in test_val.items():
            col_name = f"Test_{val}"
            if col_name in final_X.columns:
                final_X.at[idx, col_name] = 1.0

    # before_label 원-핫 처리
    for idx, val in before_labels.items():
        col_name = f"before_label_{val}"
        if col_name in final_X.columns:
            final_X.at[idx, col_name] = 1.0

    # before_test 원-핫 처리
    for idx, val in before_tests.items():
        col_name = f"before_test_{val}"
        if col_name in final_X.columns:
            final_X.at[idx, col_name] = 1.0

    # 4. 폴드 평균 예측
    preds = np.zeros(len(final_X))
    for model in _MODELS:
        preds += model.predict_proba(final_X)[:, 1]
    preds /= len(_MODELS)

    return pd.DataFrame({"Test_id": df["Test_id"].values, "Label": preds})
