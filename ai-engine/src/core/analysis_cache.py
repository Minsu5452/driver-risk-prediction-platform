"""분석용 데이터 메모리 캐시.

/predict/upload으로 올린 데이터의 features를 서버 메모리에 보관한다.
학습 DB(exam_records)와 완전 분리되어 무한 적재 없음.
새 업로드 시 이전 캐시를 교체한다.
"""

import logging

logger = logging.getLogger("ai-engine")

# key: test_id, value: {Test_id, PrimaryKey, Age, TestDate, domain, features: {...}}
_cache: dict[str, dict] = {}


def replace_all(records: list[dict]) -> int:
    """캐시를 새 records로 전체 교체. 반환: 저장된 건수."""
    _cache.clear()
    for r in records:
        tid = r.get("Test_id")
        if tid:
            _cache[tid] = r
    logger.info("Analysis cache replaced: %d records", len(_cache))
    return len(_cache)


def get_by_ids(test_ids: list[str]) -> list[dict]:
    """test_ids에 매칭되는 records 반환. 캐시에 없는 건은 스킵."""
    return [_cache[tid] for tid in test_ids if tid in _cache]


def size() -> int:
    return len(_cache)
