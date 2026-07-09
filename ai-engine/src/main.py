from typing import List
import logging
import logging.handlers
import os
import warnings

# .env 파일은 src/core/constants.py에서 로드됨 (모든 entry point 지원)
import src.core.constants  # noqa: F401 — .env 로드 보장

import numpy as np
import pandas as pd

# ── 경고 억제: 무해한 RuntimeWarning 정리 ──
# numpy 2.0: int8/int16 배열에 NaN 채울 때 (0으로 변환됨, 정상 동작)
warnings.filterwarnings("ignore", message="invalid value encountered in cast",
                        category=RuntimeWarning)
# sklearn 캘리브레이션: LogisticRegression 최적화 중 극단값 (결과에 영향 없음)
warnings.filterwarnings("ignore", message="divide by zero encountered in matmul",
                        category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in matmul",
                        category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in matmul",
                        category=RuntimeWarning)
# pandas DataFrame fragmentation (diff 피처 생성 시, 기능상 문제 없음)
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented",
                        category=pd.errors.PerformanceWarning)
# joblib resource_tracker: 임시 메모리매핑 파일 정리 타이밍 (무해)
warnings.filterwarnings("ignore", message="resource_tracker:",
                        category=UserWarning)
from fastapi import FastAPI, HTTPException
from src.inference.loader import load_all_artifacts
from src.schemas import DriverInput, PredictionResponse, GlobalExplainRequest, GlobalExplainByIdsRequest
from src.services.prediction_service import predict_service

# ── 로깅 설정 ──
_log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log", "ai_engine.log")
_log_fmt = logging.Formatter(
    "%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 1) root 로거: 기존 핸들러 전부 제거 (--reload 시 누적 방지)
_root = logging.getLogger()
_root.handlers.clear()
_root.setLevel(logging.WARNING)  # 라이브러리 잡음 차단

# 2) ai-engine 로거: 앱 로그 전용
logger = logging.getLogger("ai-engine")
logger.handlers.clear()  # --reload 시 핸들러 누적 방지
logger.setLevel(logging.INFO)
logger.propagate = False  # root 전파 차단

_sh = logging.StreamHandler()
_sh.setFormatter(_log_fmt)
logger.addHandler(_sh)

_log_dir = os.path.dirname(_log_path)
os.makedirs(_log_dir, exist_ok=True)

# 날짜별 로그 파일 (backend와 동일 패턴: ai_engine.2026-04-04.log)
import datetime as _dt
_today = _dt.datetime.now().strftime("%Y-%m-%d")
_daily_log_path = os.path.join(_log_dir, f"ai_engine.{_today}.log")
_fh = logging.handlers.RotatingFileHandler(
    _daily_log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8",
)
_fh.setFormatter(_log_fmt)
logger.addHandler(_fh)

# 3) uvicorn 접근 로그: 파일에 안 쓰고 콘솔에만 (앱 로그와 분리)
_uv_access = logging.getLogger("uvicorn.access")
_uv_access.handlers.clear()
_uv_access.propagate = False  # root/파일 전파 차단 → 콘솔 터미널에만 표시

# OOF 예측 점수 캐시 (test_id → score). 서버 시작 시 + 학습 후 로드.
_oof_scores: dict = {}


def _load_oof_scores():
    """활성 모델의 OOF 예측 점수를 로드한다. DB BLOB → BytesIO → Parquet."""
    global _oof_scores
    from io import BytesIO
    from src.core.database import get_active_model_version, load_artifact

    scores = {}
    try:
        active = get_active_model_version()
        if not active or not active.get("run_id"):
            _oof_scores = scores
            return

        run_id = active["run_id"]
        for domain in ["A", "B"]:
            key = f"stack/{domain}/oof_predictions_calibrated.parquet"
            data = load_artifact(run_id, key)
            if data is None:
                continue
            oof_df = pd.read_parquet(BytesIO(data))
            if "test_id" in oof_df.columns and "ensemble_temp_scaled" in oof_df.columns:
                for tid, score in zip(oof_df["test_id"], oof_df["ensemble_temp_scaled"]):
                    scores[str(tid)] = float(score)
                logger.info("OOF 점수 로드: 도메인 %s, %d건", domain, len(oof_df))
            elif "ensemble_temp_scaled" in oof_df.columns:
                logger.warning("OOF 파일에 test_id 없음 (구버전) — 학습 재실행 필요")
    except Exception as e:
        logger.warning("OOF 점수 로드 실패: %s", e)

    _oof_scores = scores
    logger.info("OOF 점수 캐시: %d건", len(scores))


def _get_driver_history_from_db(primary_key: str) -> list[dict]:
    """exam_records DB에서 운전자의 검사 이력을 조회한다. OOF 점수가 있으면 함께 반환."""
    import json
    from datetime import datetime, timedelta, timezone
    from src.core.database import get_exam_records_by_pk
    from src.utils.age_calc import compute_current_age_from_yyyymmdd

    records = get_exam_records_by_pk(primary_key)
    if not records:
        return []

    KST = timezone(timedelta(hours=9))
    today_kst = datetime.now(KST)

    result = []
    for r in records:
        features = {}
        if r.get("features_json"):
            try:
                features = json.loads(r["features_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        test_date = str(r.get("test_date", "") or "").strip()
        if len(test_date) == 6 and test_date.isdigit():
            test_date = test_date + "01"

        tid = r["test_id"]
        score = _oof_scores.get(tid)

        exam_age_val = r.get("exam_age")
        birth_yyyymmdd = (r.get("birth_yyyymmdd") or "").strip()
        current_age_val = None

        if birth_yyyymmdd and len(birth_yyyymmdd) == 8:
            # 정확 계산: 출생일 + 오늘(KST) 비교
            current_age_val = compute_current_age_from_yyyymmdd(
                birth_yyyymmdd, now_kst=today_kst
            )
        elif exam_age_val is not None and test_date:
            # Fallback: 구 데이터(birth 컬럼 없음)는 연도차 근사
            try:
                test_year = int(test_date[:4])
                current_age_val = exam_age_val + (today_kst.year - test_year)
                if current_age_val < 0:
                    current_age_val = None
            except (ValueError, TypeError):
                pass

        result.append({
            "Test_id": tid,
            "TestDate": test_date,
            "domain": r["domain"],
            "Age": r.get("age", ""),
            "exam_age": str(exam_age_val) if exam_age_val is not None else None,
            "current_age": str(current_age_val) if current_age_val is not None else None,
            "birth_yyyymmdd": birth_yyyymmdd or None,
            "score": score,
            "features": features,
        })

    return result


async def lifespan(app: FastAPI):
    # ── 시스템 사양 자동 감지 결과 로깅 ──
    from src.core.constants import (
        _CPU_COUNT, _RAM_GB, N_JOBS, DB_CACHE_SIZE, DB_MMAP_SIZE,
        MAX_SHAP_SAMPLES, SEQ_DIFF_CHUNK, EXPLAIN_BATCH_WORKERS,
    )
    import platform as _pf
    _ram_note = "" if _RAM_GB != 8.0 else " (감지 실패, 기본값)"
    logger.info(
        "시스템: %s %s, CPU %d threads, RAM %.0fGB%s | "
        "N_JOBS=%d, DB_CACHE=%dMB, DB_MMAP=%dMB, SHAP=%d, DIFF_CHUNK=%d",
        _pf.system(), _pf.machine(), _CPU_COUNT, _RAM_GB, _ram_note,
        N_JOBS, abs(DB_CACHE_SIZE) // 1000, DB_MMAP_SIZE // (1024 * 1024),
        MAX_SHAP_SAMPLES, SEQ_DIFF_CHUNK,
    )

    # 고아 버전 디렉터리 정리
    try:
        from src.core.database import cleanup_orphan_version_dirs
        deleted = cleanup_orphan_version_dirs()
        if deleted:
            logger.info("Cleaned up %d orphan version dirs: %s", len(deleted), deleted)
    except Exception as e:
        logger.warning("Orphan cleanup skipped: %s", e)

    # 이전 서버 충돌/재시작으로 'running' 상태에 멈춘 학습 실행 복구
    try:
        from src.core.database import get_db
        from datetime import datetime, timezone
        db = get_db()
        cur = db.execute(
            "UPDATE training_runs SET status='failed', step_detail='failed', "
            "error_message='서버 재시작으로 학습 중단됨', completed_at=? "
            "WHERE status='running'",
            (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),),
        )
        if cur.rowcount > 0:
            logger.warning("Fixed %d orphaned training run(s) stuck in 'running'", cur.rowcount)
        db.commit()
    except Exception as e:
        logger.warning("Orphan training cleanup skipped: %s", e)

    # 시작 시 모델 아티팩트 로드 (활성 모델 버전이 존재하는 경우에만)
    try:
        from src.core.database import get_active_model_version
        from src.core.constants import PROJECT_ROOT
        active = get_active_model_version()

        if active and active.get("run_id"):
            run_id = active["run_id"]
            logger.info("Loading AI artifacts from DB (run_id=%s)", run_id)
            load_all_artifacts(["A", "B"], run_id=run_id)
            from src.inference.seq_engine import load_seq_artifacts
            load_seq_artifacts(run_id=run_id)
            logger.info("Artifacts loaded successfully.")
        else:
            logger.info("활성 모델 버전 없음 — 관리자 대시보드에서 학습을 실행하면 모델이 로드됩니다.")
    except Exception as e:
        logger.error("Failed to load artifacts: %s", e)

    # OOF 예측 점수 로드 (이력 조회 시 사고 위험도 표시용)
    try:
        _load_oof_scores()
    except Exception as e:
        logger.warning("OOF 점수 로드 실패: %s", e)

    from src.core.database import get_data_summary
    try:
        summary = get_data_summary()
        logger.info("검사 이력 DB 현황: A %d건, B %d건, 사고 %d건",
                     summary["a_exam"], summary["b_exam"], summary["sago"])
    except Exception as e:
        logger.warning("DB summary check failed: %s", e)

    yield

    # 서버 종료 시 DB 커넥션 정리
    from src.core.database import close_db
    close_db()


app = FastAPI(title="Driver Risk AI Engine", lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from src.api.upload import router as upload_router
from src.api.explain import router as explain_router
from src.api.admin import router as admin_router

app.include_router(upload_router)
app.include_router(explain_router)
app.include_router(admin_router)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "risk-ai-engine"}


@app.post("/predict", response_model=List[PredictionResponse])
async def predict(inputs: List[DriverInput]):
    import asyncio
    return await asyncio.to_thread(predict_service, inputs)


@app.post("/predict/explain_global")
def explain_global(payload: GlobalExplainRequest):
    domain = payload.domain
    items = payload.items

    from src.inference.stack_engine import explain_dataframe

    if items:
        try:
            # items를 평탄화하여 DataFrame 생성
            flat_items = []
            for item in items:
                base = item.model_dump()
                feats = base.pop("features", {})
                if isinstance(feats, dict):
                    base.update(feats)
                flat_items.append(base)

            df = pd.DataFrame(flat_items)
            if "Test_id" in df.columns:
                df = df.drop_duplicates(subset=["Test_id"], keep="last")

            # SHAP 계산
            explanation = explain_dataframe(domain, df, detailed=payload.detailed)

            shap_vals = explanation.get("shap_values", [])
            feature_names = explanation.get("feature_names", [])

            if not shap_vals or not feature_names:
                return {"shap_values": [], "feature_names": []}

            # 방향성 글로벌 평균 집계 (shap_vals: N x M 행렬)
            arr = np.array(shap_vals)
            if arr.ndim == 2:
                # 샘플 간 평균
                global_shap = np.mean(arr, axis=0)
                return {
                    "shap_values": global_shap.tolist(),
                    "feature_names": feature_names,
                    "base_value": explanation.get("base_value", 0.0),
                }
            else:
                return {"shap_values": [], "feature_names": []}

        except Exception as e:
            logger.error("Internal error: %s", e)
            raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")

    return {"shap_values": [], "feature_names": []}


@app.post("/predict/explain_global_by_ids")
def explain_global_by_ids(payload: GlobalExplainByIdsRequest):
    """서버 캐시에서 features를 로드하여 SHAP 1회 계산.

    프론트에서 test_ids만 전달 → 캐시에서 features 로드 → 1회 feature construction + SHAP.
    기존 explain_global 대비 네트워크 83배 절감, 처리 시간 40분→~30초.
    """
    from src.core.analysis_cache import get_by_ids
    from src.inference.stack_engine import explain_dataframe

    records = get_by_ids(payload.test_ids)
    if not records:
        return {"shap_values": [], "feature_names": []}

    # DataFrame 구성 (기존 explain_global과 동일한 형태)
    flat_items = []
    for r in records:
        base = {k: v for k, v in r.items() if k != "features"}
        feats = r.get("features")
        if isinstance(feats, dict):
            base.update(feats)
        flat_items.append(base)

    df = pd.DataFrame(flat_items)
    if "Test_id" in df.columns:
        df = df.drop_duplicates(subset=["Test_id"], keep="last")

    logger.info("explain_global_by_ids: domain=%s, cache_hit=%d/%d, df=%d rows",
                payload.domain, len(records), len(payload.test_ids), len(df))

    explanation = explain_dataframe(payload.domain, df, detailed=False)

    shap_vals = explanation.get("shap_values", [])
    feature_names = explanation.get("feature_names", [])

    if not shap_vals or not feature_names:
        return {"shap_values": [], "feature_names": []}

    arr = np.array(shap_vals)
    if arr.ndim == 2:
        global_shap = np.mean(arr, axis=0)
        return {
            "shap_values": global_shap.tolist(),
            "feature_names": feature_names,
            "base_value": explanation.get("base_value", 0.0),
        }
    return {"shap_values": [], "feature_names": []}


@app.get("/predict/history/{primary_key}")
def get_driver_history(primary_key: str):
    """exam_records DB에서 운전자의 검사 이력을 조회한다."""
    return _get_driver_history_from_db(primary_key)


if __name__ == "__main__":
    import uvicorn

    workers = int(os.environ.get("WORKERS", "1"))
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, workers=workers)
