"""개인별 SHAP 결과 메모리 캐시.

업로드(분석) 시점 또는 최초 요청 시 1회 계산한 개인별 SHAP을 보관한다.
다운로드/단건진단/비교분석이 매번 재계산하지 않고 결과만 조회하기 위함이다.

메모리 절약:
  - 피처명(feature_names)은 도메인당 1회만 저장한다.
  - 사람별로는 값 배열(list[float])만 보관한다. (피처명 문자열 반복 저장 회피)

analysis_cache와 동일하게 새 업로드 시 전체 교체된다(무한 적재 없음).
"""

import logging

logger = logging.getLogger("ai-engine")

# {domain: {"feature_names": [...], "base_value": float}}
_meta: dict[str, dict] = {}
# {test_id: {"domain": str, "pk": str, "values": list[float]}}
_rows: dict[str, dict] = {}


def clear() -> None:
    _meta.clear()
    _rows.clear()


def set_domain_meta(domain: str, feature_names: list, base_value: float) -> None:
    _meta[domain] = {
        "feature_names": list(feature_names),
        "base_value": float(base_value),
    }


def put_row(test_id: str, domain: str, primary_key: str, values: list) -> None:
    _rows[test_id] = {"domain": domain, "pk": primary_key, "values": values}


def has(test_id: str) -> bool:
    return test_id in _rows


def missing_ids(test_ids: list) -> list:
    """캐시에 없는 test_id만 반환."""
    return [tid for tid in test_ids if tid not in _rows]


def get_formatted(test_ids: list) -> list:
    """test_ids에 대한 SHAP을 프론트가 기대하는 형식으로 복원하여 반환한다.

    형식: [{"Test_id", "PrimaryKey", "shap_values": [{feature,value,code}...],
            "base_value", "feature_names"}]
    캐시에 없는 test_id는 건너뛴다.
    """
    out = []
    for tid in test_ids:
        row = _rows.get(tid)
        if not row:
            continue
        meta = _meta.get(row["domain"])
        if not meta:
            continue
        names = meta["feature_names"]
        formatted = [
            {"feature": n, "value": float(v), "code": n}
            for n, v in zip(names, row["values"])
        ]
        formatted.sort(key=lambda x: abs(x["value"]), reverse=True)
        out.append(
            {
                "Test_id": tid,
                "PrimaryKey": row["pk"],
                "shap_values": formatted,
                "base_value": meta["base_value"],
                "feature_names": names,
            }
        )
    return out


def size() -> int:
    return len(_rows)
