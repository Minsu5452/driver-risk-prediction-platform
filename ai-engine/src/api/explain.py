import logging
import math
from typing import List

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.constants import EXPLAIN_BATCH_LIMIT
from src.inference.stack_engine import explain_dataframe
from src.schemas import DriverInput

logger = logging.getLogger("ai-engine")

router = APIRouter()


class ExplainByIdsRequest(BaseModel):
    test_ids: List[str]


def sanitize_floats(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj


def _explain_single(inp: DriverInput) -> dict:
    """단건 SHAP 설명 (배치에서도 재사용)."""
    domain = inp.domain
    data = inp.features.copy()
    data.update(
        {
            "Test_id": inp.Test_id,
            "TestDate": inp.TestDate,
            "Age": inp.Age,
            "PrimaryKey": inp.PrimaryKey,
        }
    )

    df = pd.DataFrame([data])
    explanation = explain_dataframe(domain, df, detailed=False)

    shap_vals = explanation.get("shap_values", [[]])
    if isinstance(shap_vals, list) and len(shap_vals) > 0:
        if isinstance(shap_vals[0], list):
            shap_vals = shap_vals[0]
    elif hasattr(shap_vals, "tolist"):
        shap_vals = shap_vals.tolist()
        if isinstance(shap_vals[0], list):
            shap_vals = shap_vals[0]

    feature_names = explanation.get("feature_names", [])
    formatted_shap = [
        {"feature": name, "value": float(val), "code": name}
        for name, val in zip(feature_names, shap_vals)
    ]
    formatted_shap.sort(key=lambda x: abs(x["value"]), reverse=True)

    return sanitize_floats(
        {
            "Test_id": inp.Test_id,
            "PrimaryKey": inp.PrimaryKey,
            "shap_values": formatted_shap,
            "base_value": explanation.get("base_value", 0.0),
            "feature_names": feature_names,
        }
    )


@router.post("/explain")
def explain(inp: DriverInput):
    try:
        return _explain_single(inp)
    except Exception as e:
        logger.exception(f"Error explaining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _explain_group(domain: str, items: list) -> list:
    """같은 도메인 운전자 묶음을 단 1회의 explain_dataframe 호출로 계산한다.

    SHAP TreeExplainer와 피처 생성은 모두 행(사람) 단위로 독립적이므로,
    N명을 한 번에 넣어도 각 사람의 결과는 1명씩 계산할 때와 동일하다.
    explain_dataframe는 입력 df의 행 순서를 보존하므로 row i ↔ items[i] 로 매핑한다.

    주의: explain_dataframe는 MAX_SHAP_SAMPLES 초과 시 샘플링하지만,
    배치 청크는 EXPLAIN_BATCH_LIMIT(<= MAX_SHAP_SAMPLES) 이하라 샘플링되지 않는다.
    → 전원이 자기 자신의 실제 SHAP 값을 받는다.
    """
    flat = []
    for inp in items:
        data = inp.features.copy()
        data.update(
            {
                "Test_id": inp.Test_id,
                "TestDate": inp.TestDate,
                "Age": inp.Age,
                "PrimaryKey": inp.PrimaryKey,
            }
        )
        flat.append(data)

    df = pd.DataFrame(flat)
    explanation = explain_dataframe(domain, df, detailed=False)

    shap_matrix = explanation.get("shap_values", []) or []
    feature_names = explanation.get("feature_names", [])
    base_value = explanation.get("base_value", 0.0)

    out = []
    for i, inp in enumerate(items):
        row = shap_matrix[i] if i < len(shap_matrix) else []
        formatted_shap = [
            {"feature": name, "value": float(val), "code": name}
            for name, val in zip(feature_names, row)
        ]
        formatted_shap.sort(key=lambda x: abs(x["value"]), reverse=True)
        out.append(
            sanitize_floats(
                {
                    "Test_id": inp.Test_id,
                    "PrimaryKey": inp.PrimaryKey,
                    "shap_values": formatted_shap,
                    "base_value": base_value,
                    "feature_names": feature_names,
                }
            )
        )
    return out


@router.post("/explain/batch")
def explain_batch(inputs: List[DriverInput]):
    """여러 운전자 SHAP 일괄 계산. 프론트에서 청크 호출.

    도메인별로 묶어 각 도메인당 explain_dataframe를 1회만 호출한다(1인씩 호출 제거).
    """
    if len(inputs) > EXPLAIN_BATCH_LIMIT:
        raise HTTPException(status_code=400, detail=f"1회 최대 {EXPLAIN_BATCH_LIMIT}건입니다. 프론트에서 청크 분할해주세요.")

    if not inputs:
        return {"results": [], "errors": []}

    # 도메인별 그룹화 (원본 순서/인덱스 보존)
    groups: dict[str, list] = {}
    indexed: dict[str, list] = {}
    for idx, inp in enumerate(inputs):
        groups.setdefault(inp.domain, []).append(inp)
        indexed.setdefault(inp.domain, []).append(idx)

    results = []
    errors = []
    for domain, items in groups.items():
        try:
            results.extend(_explain_group(domain, items))
        except Exception as e:
            logger.warning(f"Batch explain error [domain={domain}, n={len(items)}]: {e}")
            for j, inp in enumerate(items):
                errors.append({"index": indexed[domain][j], "Test_id": inp.Test_id, "error": str(e)})

    return {"results": results, "errors": errors}


@router.post("/explain/batch_by_ids")
def explain_batch_by_ids(payload: ExplainByIdsRequest):
    """test_ids만 받아 캐시에서 개인별 SHAP을 조회한다.

    - 캐시 히트: 즉시 반환(재계산 없음).
    - 캐시 미스: analysis_cache의 features로 일괄 계산 → 캐시 적재 후 반환.
    프론트는 features 전송 없이 test_ids만 보내므로 네트워크도 크게 절감된다.
    응답 형식은 /explain/batch와 동일하다({results, errors}).
    """
    from src.core import shap_cache, analysis_cache
    from src.services.shap_service import compute_for_records

    test_ids = payload.test_ids or []
    if not test_ids:
        return {"results": [], "errors": []}

    missing = shap_cache.missing_ids(test_ids)
    if missing:
        records = analysis_cache.get_by_ids(missing)
        if records:
            try:
                n = compute_for_records(records)
                logger.info("explain_batch_by_ids: 캐시 미스 %d건 계산 (요청 %d건)", n, len(test_ids))
            except Exception as e:
                logger.exception("on-demand SHAP 계산 실패: %s", e)
                raise HTTPException(status_code=500, detail="SHAP 계산 중 오류가 발생했습니다.")

    results = shap_cache.get_formatted(test_ids)
    return {"results": results, "errors": []}
