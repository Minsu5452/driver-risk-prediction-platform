from typing import List
import logging

import numpy as np
import pandas as pd
from fastapi import HTTPException

from src.config.ensemble_config import WEIGHT_STACK as W_R1, WEIGHT_SEQ as W_R7, BIAS
from src.inference.stack_engine import predict_dataframe
from src.inference.seq_engine import predict_dataframe as predict_seq
from src.schemas import DriverInput, PredictionResponse

logger = logging.getLogger("ai-engine")


def predict_domain_df(domain: str, df: pd.DataFrame) -> pd.DataFrame:
    """도메인별 DataFrame을 직접 받아 예측을 수행한다.

    Args:
        domain: "A" 또는 "B"
        df: Test_id, TestDate, Age, PrimaryKey + 피처 컬럼을 포함하는 DataFrame

    Returns:
        Test_id, final_score, riskGroup 컬럼을 포함하는 DataFrame
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["Test_id"], keep="last")
    if len(df) < n_before:
        logger.info(f"Deduplicated {n_before - len(df)} duplicate Test_ids for domain {domain}")

    logger.info(f"Predicting for domain {domain} with {len(df)} records")

    # ── Rank 1 추론 ──
    pred_df_r1 = predict_dataframe(domain, df)

    # ── Rank 7 추론 ──
    r7_fallback = False
    try:
        pred_df_r7 = predict_seq(domain, df)
    except (FileNotFoundError, ValueError, RuntimeError) as e7:
        logger.warning(
            f"Rank 7 inference failed: {e7}. Falling back to Rank 1 only.",
            exc_info=True,
        )
        pred_df_r7 = None
        r7_fallback = True

    # ── 앙상블 블렌딩 ──
    if r7_fallback:
        merged = pred_df_r1[["Test_id", "Label"]].rename(columns={"Label": "r1"})
        merged = merged.assign(final_score=merged["r1"])
    else:
        merged = (
            pred_df_r1[["Test_id", "Label"]]
            .rename(columns={"Label": "r1"})
            .merge(
                pred_df_r7[["Test_id", "Label"]].rename(columns={"Label": "r7"}),
                on="Test_id",
                how="left",
            )
        )
        merged["r7"] = merged["r7"].fillna(merged["r1"])
        merged = merged.assign(
            final_score=(merged["r1"] * W_R1 + merged["r7"] * W_R7 + BIAS)
        )

    merged["final_score"] = merged["final_score"].clip(0, 1)

    # 위험 그룹 벡터화 계산
    scores = merged["final_score"].values
    merged["riskGroup"] = np.where(
        scores >= 0.7, "고위험", np.where(scores >= 0.5, "중위험", "저위험")
    )

    return merged[["Test_id", "final_score", "riskGroup"]]


def predict_service(inputs: List[DriverInput]) -> List[PredictionResponse]:
    """DriverInput 리스트를 받아 예측 결과를 반환한다 (/predict 엔드포인트용)."""
    results = []

    # 도메인별로 그룹화
    inputs_by_domain: dict[str, list] = {"A": [], "B": []}

    for inp in inputs:
        if inp.domain not in inputs_by_domain:
            continue
        data = {k: v for k, v in inp.features.items() if k != "_meta"}
        data["Test_id"] = inp.Test_id
        data["TestDate"] = inp.TestDate
        data["Age"] = inp.Age
        data["PrimaryKey"] = inp.PrimaryKey
        inputs_by_domain[inp.domain].append(data)

    for domain, data_list in inputs_by_domain.items():
        if not data_list:
            continue

        try:
            df = pd.DataFrame(data_list)
            merged = predict_domain_df(domain, df)

            for _, row in merged.iterrows():
                results.append(
                    PredictionResponse(
                        Test_id=str(row["Test_id"]),
                        score=float(row["final_score"]),
                        result=float(row["final_score"]),
                        riskGroup=row["riskGroup"],
                        domain=domain,
                    )
                )

        except Exception as e:
            logger.exception(f"Error predicting domain {domain}: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return results
