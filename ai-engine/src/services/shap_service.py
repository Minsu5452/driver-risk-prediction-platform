"""개인별 SHAP을 일괄 계산하여 shap_cache에 적재하는 서비스.

- 도메인별로 묶고, 청크 단위(SHAP_PRECOMPUTE_CHUNK)로 explain_dataframe(sample=False)를 호출한다.
  → 전원이 자기 실제 SHAP을 받으며(샘플링 없음), 청크로 메모리를 제어한다.
- 결과는 shap_cache에 압축 저장(피처명 1회 + 사람별 값 배열)한다.
"""

import logging

import numpy as np
import pandas as pd

from src.core import shap_cache
from src.inference.stack_engine import explain_dataframe

logger = logging.getLogger("ai-engine")

# 한 번의 explain_dataframe 호출에 넣을 최대 행 수.
# MAX_SHAP_SAMPLES(기본 5000) 이하로 두어 sample=False여도 메모리 급증을 방지한다.
SHAP_PRECOMPUTE_CHUNK = 2000


def _records_to_inputs(records: list) -> dict:
    """analysis_cache 레코드를 도메인별 입력 리스트로 그룹화한다."""
    by_domain: dict[str, list] = {}
    for r in records:
        domain = r.get("domain")
        if domain not in ("A", "B"):
            continue
        tid = r.get("Test_id")
        if not tid:
            continue
        by_domain.setdefault(domain, []).append(r)
    return by_domain


def compute_for_records(records: list) -> int:
    """records(analysis_cache 형식)의 개인별 SHAP을 계산하여 캐시에 적재한다.

    이미 캐시에 있는 test_id는 건너뛴다. 반환: 이번에 새로 계산한 건수.
    """
    by_domain = _records_to_inputs(records)
    total_new = 0

    for domain, items in by_domain.items():
        # 미계산분만 추림
        todo = [r for r in items if not shap_cache.has(r["Test_id"])]
        if not todo:
            continue

        for i in range(0, len(todo), SHAP_PRECOMPUTE_CHUNK):
            chunk = todo[i : i + SHAP_PRECOMPUTE_CHUNK]
            flat = []
            for r in chunk:
                data = dict(r.get("features") or {})
                data.update(
                    {
                        "Test_id": r.get("Test_id"),
                        "TestDate": r.get("TestDate", ""),
                        "Age": r.get("Age", "0"),
                        "PrimaryKey": r.get("PrimaryKey", "UNKNOWN"),
                    }
                )
                flat.append(data)

            df = pd.DataFrame(flat)
            explanation = explain_dataframe(domain, df, detailed=False, sample=False)

            shap_matrix = explanation.get("shap_values", []) or []
            feature_names = explanation.get("feature_names", [])
            base_value = explanation.get("base_value", 0.0)

            if not feature_names:
                logger.warning("compute_for_records: domain=%s feature_names 비어있음", domain)
                continue

            shap_cache.set_domain_meta(domain, feature_names, base_value)

            arr = np.asarray(shap_matrix, dtype=float)
            n_feat = len(feature_names)
            for j, r in enumerate(chunk):
                if j < len(arr):
                    # NaN/Inf 방어 후 float32 배열로 저장(메모리 절감).
                    # astype은 복사본을 만들어 청크 행렬(arr) 전체가 잔존하지 않게 한다.
                    # float32 정밀도(~7자리)는 보고서 표시(%p 소수 2자리)보다 4자릿수 이상 정밀.
                    vals = np.nan_to_num(arr[j], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                else:
                    vals = np.zeros(n_feat, dtype=np.float32)
                shap_cache.put_row(r["Test_id"], domain, r.get("PrimaryKey", ""), vals)
                total_new += 1

        logger.info("SHAP precompute: domain=%s, %d명 계산 완료", domain, len(todo))

    return total_new
