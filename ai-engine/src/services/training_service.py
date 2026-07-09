import asyncio
import gc
import os
import json
import logging
import shutil
import tempfile
import threading
import time
import pandas as pd
from datetime import datetime, timezone

from src.core.constants import PROJECT_ROOT, BASE_DIR, VERSIONS_DIR
from src.core.database import (
    update_training_run,
    insert_model_version,
    set_active_model_version,
)

logger = logging.getLogger("ai-engine")

_TRAINING_LOCK = asyncio.Lock()
_CURRENT_RUN_ID = None
_CANCEL_EVENT = threading.Event()


class TrainingCancelled(Exception):
    """사용자 요청에 의해 학습이 취소될 때 발생하는 예외."""
    pass


def is_training_running() -> bool:
    return _TRAINING_LOCK.locked()


def get_current_run_id():
    return _CURRENT_RUN_ID


def request_cancel():
    """현재 실행 중인 학습의 취소를 요청한다."""
    _CANCEL_EVENT.set()


def check_cancelled():
    """취소 요청 여부를 확인한다. 단계 내부에서 자유롭게 호출 가능."""
    if _CANCEL_EVENT.is_set():
        raise TrainingCancelled("Training cancelled by user")


def _check_cancel(run_id: int):
    """취소 요청 여부를 확인하고, 요청된 경우 예외를 발생시킨다."""
    check_cancelled()


def _calc_dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


async def run_training_pipeline(run_id: int):
    """비동기 학습 파이프라인 오케스트레이터.

    단계:
    1. loading: DB에서 학습 데이터 로드
    2. labeling: 사고 데이터로 라벨 생성
    3. generating_files: train.csv, train/A.csv, train/B.csv 파일 생성
    4. training_stack: Rank 1 학습 실행
    5. training_seq: Rank 7 학습 실행 (테스트 데이터 없이)
    6. reloading_models: 모델 아티팩트 리로드
    7. completed: 메트릭 저장 + 버전 등록
    """
    global _CURRENT_RUN_ID

    if _TRAINING_LOCK.locked():
        update_training_run(
            run_id,
            status="failed",
            error_message="Another training is already running",
            completed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        )
        return

    async with _TRAINING_LOCK:
        _CURRENT_RUN_ID = run_id
        loop = asyncio.get_event_loop()

        # 버전 디렉터리 생성
        version_label = f"v_{run_id}"
        version_dir = os.path.join(VERSIONS_DIR, version_label)
        os.makedirs(version_dir, exist_ok=True)

        try:
            pipeline_start = time.time()

            # 1단계: DB에서 데이터 로드
            update_training_run(run_id, status="running", step_detail="loading")
            logger.info(f"[Training {run_id}] Step 1/6: Loading data from DB...")
            t0 = time.time()

            a_exam_df, b_exam_df, a_sago_df, b_sago_df = await loop.run_in_executor(
                None, _step_load_from_db
            )

            logger.info(f"[Training {run_id}] Step 1/6 완료: DB 데이터 로드 ({time.time() - t0:.1f}s)")
            _check_cancel(run_id)

            # 2단계: 라벨 생성
            update_training_run(run_id, step_detail="labeling")
            logger.info(f"[Training {run_id}] Step 2/6: Creating labels...")
            t0 = time.time()

            a_labeled, b_labeled = await loop.run_in_executor(
                None, _step_label, a_exam_df, b_exam_df, a_sago_df, b_sago_df
            )

            logger.info(f"[Training {run_id}] Step 2/6 완료: 라벨 생성 ({time.time() - t0:.1f}s)")
            _check_cancel(run_id)

            # 2-1: 라벨을 exam_records에 저장 (추론 시 before_label로 사용)
            from src.core.database import update_exam_labels
            label_pairs = []
            for labeled_df in [a_labeled, b_labeled]:
                if labeled_df is not None and "Test_id" in labeled_df.columns and "Label" in labeled_df.columns:
                    for tid, lbl in zip(labeled_df["Test_id"], labeled_df["Label"]):
                        label_pairs.append((str(tid), int(lbl)))
            if label_pairs:
                await loop.run_in_executor(None, update_exam_labels, label_pairs)
                logger.info(f"[Training {run_id}] exam_records label 저장: {len(label_pairs)}건")

            # 3단계: 임시 디렉토리에 학습 파일 생성 (Parquet)
            update_training_run(run_id, step_detail="generating_files")
            logger.info(f"[Training {run_id}] Step 3/6: Generating training files (Parquet)...")
            t0 = time.time()

            tmp_data_dir = tempfile.mkdtemp(prefix="risk_train_")
            await loop.run_in_executor(None, _step_generate_files, a_labeled, b_labeled, tmp_data_dir)

            logger.info(f"[Training {run_id}] Step 3/6 완료: 파일 생성 ({time.time() - t0:.1f}s) → {tmp_data_dir}")

            # 1~3단계 데이터 메모리 해제
            del a_exam_df, b_exam_df, a_sago_df, b_sago_df, a_labeled, b_labeled
            gc.collect()

            _check_cancel(run_id)

            # 4단계: Rank 1 학습 (version_dir에 저장)
            update_training_run(run_id, step_detail="training_stack")
            logger.info(f"[Training {run_id}] Step 4/6: Training Rank 1 models...")
            t0 = time.time()

            await loop.run_in_executor(None, _step_train_stack, version_dir, tmp_data_dir)

            logger.info(f"[Training {run_id}] Step 4/6 완료: Rank 1 학습 ({time.time() - t0:.1f}s)")

            # Rank 1 학습 메모리 해제 → Rank 7 시작 전 메모리 확보
            gc.collect()

            _check_cancel(run_id)

            # 5단계: Rank 7 학습
            update_training_run(run_id, step_detail="training_seq")
            logger.info(f"[Training {run_id}] Step 5/6: Training Rank 7 models...")
            t0 = time.time()

            seq_metrics = await loop.run_in_executor(None, _step_train_seq, version_dir, tmp_data_dir)

            logger.info(f"[Training {run_id}] Step 5/6 완료: Rank 7 학습 ({time.time() - t0:.1f}s)")

            # 임시 학습 데이터 삭제
            shutil.rmtree(tmp_data_dir, ignore_errors=True)
            logger.info(f"[Training {run_id}] 임시 학습 데이터 삭제: {tmp_data_dir}")
            _check_cancel(run_id)

            # 6단계: 학습 산출물 정리 + DB 이전 + 버전 등록 + 리로드
            update_training_run(run_id, step_detail="reloading_models")

            # 중간 산출물 삭제 (timecausal_features 등)
            _cleanup_training_artifacts(version_dir)

            # 메트릭 수집 (oof_score.json이 아직 디스크에 있을 때)
            metrics = _collect_metrics(seq_metrics, version_dir)
            update_training_run(
                run_id,
                status="completed",
                step_detail="completed",
                completed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                metrics_json=json.dumps(metrics),
            )

            # 용량 계산 (DB 저장 전에 — 저장 후 디렉토리 삭제됨)
            artifact_path = os.path.relpath(version_dir, PROJECT_ROOT)
            size_bytes = _calc_dir_size(version_dir)

            # 모든 아티팩트 → DB BLOB 이전 + 디스크 삭제
            logger.info(f"[Training {run_id}] Storing artifacts to DB...")
            t0 = time.time()
            _store_artifacts_to_db(run_id, version_dir)
            logger.info(f"[Training {run_id}] Artifacts stored to DB ({time.time() - t0:.1f}s)")
            version_id = insert_model_version(
                run_id=run_id,
                version_label=version_label,
                artifact_path=artifact_path,
                size_bytes=size_bytes,
                metrics_json=json.dumps(metrics),
                description=f"학습 Run #{run_id}",
            )
            set_active_model_version(version_id)

            # 모델 리로드 (JSON/Parquet은 DB에서, PKL은 디스크에서)
            logger.info(f"[Training {run_id}] Reloading model artifacts...")
            t0 = time.time()
            await loop.run_in_executor(None, _step_reload_models, version_dir, run_id)
            logger.info(f"[Training {run_id}] Model reload 완료 ({time.time() - t0:.1f}s)")

            # OOF 점수 캐시 갱신 (이력 조회용)
            try:
                from src.main import _load_oof_scores
                _load_oof_scores()
            except Exception as e:
                logger.warning(f"OOF scores reload failed: {e}")

            total_time = time.time() - pipeline_start
            logger.info(f"[Training {run_id}] 학습 완료! Version: {version_label} (총 {total_time:.1f}s / {total_time/60:.1f}min)")

        except TrainingCancelled:
            logger.info(f"[Training {run_id}] Training cancelled by user.")
            update_training_run(
                run_id,
                status="cancelled",
                step_detail="cancelled",
                completed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            )
            # 미완성 아티팩트 정리 (디스크 + DB)
            if os.path.isdir(version_dir):
                shutil.rmtree(version_dir, ignore_errors=True)
            from src.core.database import delete_artifacts_by_run
            delete_artifacts_by_run(run_id)
            logger.info(f"[Training {run_id}] Cleaned up artifacts (disk + DB)")

        except Exception as e:
            logger.exception(f"[Training {run_id}] Training failed: {e}")
            update_training_run(
                run_id,
                status="failed",
                step_detail="failed",
                error_message=str(e),
                completed_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            )
            # 실패 시 아티팩트 정리
            if os.path.isdir(version_dir):
                shutil.rmtree(version_dir, ignore_errors=True)
            from src.core.database import delete_artifacts_by_run
            delete_artifacts_by_run(run_id)
        finally:
            _CURRENT_RUN_ID = None
            _CANCEL_EVENT.clear()


def _read_excel_fast(path: str) -> pd.DataFrame:
    """calamine 엔진으로 Excel 읽기 (openpyxl 대비 3~10배 빠름), 실패 시 openpyxl 폴백."""
    try:
        return pd.read_excel(path, dtype=str, engine="calamine")
    except Exception:
        return pd.read_excel(path, dtype=str)


def _step_load_from_db():
    """DB에서 학습 데이터를 로드한다. dedup 불필요 (DB UNIQUE 제약이 보장)."""
    from src.data.db_loader import load_exam_df_from_db, load_sago_df_from_db

    a_exam_df = load_exam_df_from_db("A")
    check_cancelled()
    b_exam_df = load_exam_df_from_db("B")
    check_cancelled()
    sago_df = load_sago_df_from_db()

    # 사고 데이터를 도메인별로 분리 (labeler가 기대하는 형식)
    a_sago_df = None
    b_sago_df = None
    if len(sago_df) > 0:
        # sago는 이미 DB에서 중복 제거됨 — 그대로 사용
        a_sago_df = sago_df
        b_sago_df = None  # labeler에서 A+B 사고를 병합하므로 한쪽만 전달

    logger.info(
        f"[DB Load] A exam: {len(a_exam_df) if a_exam_df is not None else 0}, "
        f"B exam: {len(b_exam_df) if b_exam_df is not None else 0}, "
        f"Sago: {len(sago_df)}"
    )

    return a_exam_df, b_exam_df, a_sago_df, b_sago_df


def _step_label(a_exam_df, b_exam_df, a_sago_df, b_sago_df):
    from src.data.labeler import create_labels

    # 사고 데이터 병합 (A/B 도메인 중복 유지 — USB 레거시 동일 동작)
    # NOTE: dedup 하지 않음! A sago + B sago의 동일 사고 레코드가 모두 유지되어야
    # build_event_cumsum_monthly()에서 올바른 S_post 분포가 생성됨.
    sago_dfs = [df for df in [a_sago_df, b_sago_df] if df is not None and len(df) > 0]
    if sago_dfs:
        sago_df = pd.concat(sago_dfs, axis=0, ignore_index=True)
        logger.info(f"[Label] 사고 데이터 병합: {len(sago_df)}행")
    else:
        sago_df = pd.DataFrame()
        logger.info("[Label] 사고 데이터 없음")

    t0 = time.time()
    a_labeled, b_labeled = create_labels(a_exam_df, b_exam_df, sago_df)
    elapsed = time.time() - t0
    logger.info(
        f"[Label] 라벨 생성 완료 ({elapsed:.1f}s): "
        f"A={len(a_labeled) if a_labeled is not None else 0}행, "
        f"B={len(b_labeled) if b_labeled is not None else 0}행"
    )
    return a_labeled, b_labeled


def _step_generate_files(a_labeled, b_labeled, tmp_dir: str):
    """임시 디렉토리에 Parquet 형식으로 학습 데이터를 생성한다.

    CSV 대비 3~5배 빠르고 50~70% 작은 Parquet 형식을 사용한다.
    학습 완료 후 tmp_dir은 자동 삭제된다.
    """
    train_dir = os.path.join(tmp_dir, "train")
    os.makedirs(train_dir, exist_ok=True)

    all_records = []

    if a_labeled is not None and len(a_labeled) > 0:
        t0 = time.time()
        a_detail = a_labeled.drop(columns=["Label"], errors="ignore")
        a_detail.to_parquet(os.path.join(train_dir, "A.parquet"), index=False)
        logger.info(f"[Generate] train/A.parquet 저장: {len(a_detail)}행 ({time.time() - t0:.1f}s)")
        all_records.append(
            a_labeled[["Test_id", "Test", "PrimaryKey", "TestDate", "Label"]]
        )

    if b_labeled is not None and len(b_labeled) > 0:
        t0 = time.time()
        b_detail = b_labeled.drop(columns=["Label"], errors="ignore")
        b_detail.to_parquet(os.path.join(train_dir, "B.parquet"), index=False)
        logger.info(f"[Generate] train/B.parquet 저장: {len(b_detail)}행 ({time.time() - t0:.1f}s)")
        all_records.append(
            b_labeled[["Test_id", "Test", "PrimaryKey", "TestDate", "Label"]]
        )

    if all_records:
        t0 = time.time()
        train_df = pd.concat(all_records, axis=0, ignore_index=True)
        train_df.to_parquet(os.path.join(tmp_dir, "train.parquet"), index=False)
        logger.info(f"[Generate] train.parquet 저장: {len(train_df)}행 ({time.time() - t0:.1f}s)")


def _step_train_stack(version_dir: str, data_dir: str = None):
    """version_dir에 Rank 1 학습 파이프라인을 실행한다."""
    from src.training.stack_trainer import run_all_training

    run_all_training(model_dir=version_dir, data_dir=data_dir)


def _step_train_seq(version_dir: str, data_dir: str = None):
    """version_dir에 Rank 7 학습 파이프라인을 실행한다 (테스트 데이터 없이)."""
    from src.training.seq_trainer import train_seq_cv_no_test

    return train_seq_cv_no_test(model_dir=version_dir, data_dir=data_dir)


def _step_reload_models(version_dir: str, run_id: int = None):
    """모델 아티팩트를 DB에서 리로드한다."""
    from src.inference.loader import reload_all_artifacts
    from src.inference.seq_engine import reload_seq_artifacts

    reload_all_artifacts(["A", "B"], run_id=run_id)
    reload_seq_artifacts(run_id=run_id)
    logger.info(f"All model artifacts reloaded from DB (run_id={run_id})")


def _collect_metrics(seq_metrics, version_dir: str):
    """학습 아티팩트에서 메트릭을 수집한다."""
    metrics = {"seq": seq_metrics or {}}

    for domain in ["A", "B"]:
        score_path = os.path.join(version_dir, "stack", domain, "oof_score.json")
        if os.path.exists(score_path):
            with open(score_path, "r") as f:
                metrics[f"stack_{domain}"] = json.load(f)

    return metrics


def _cleanup_training_artifacts(version_dir: str):
    """추론에 불필요한 학습 중간 산출물을 삭제한다.

    보존: 모델 pkl, config json, snapshot/prior, calibrated oof, oof_score.json
    삭제: timecausal_features, cohort_timecausal_features (학습 중 디스크 경유 전달용)
    참고: oof_raw, train_stats, seq/oof는 생성 자체를 제거함 (불필요)
    """
    deletable_patterns = [
        # stack/personal/ 중간 산출물 (학습 중 디스크 경유 전달용)
        ("stack", "personal", "A_timecausal_features.parquet"),
        ("stack", "personal", "B_timecausal_features.parquet"),
        ("stack", "personal", "A_cohort_timecausal_features.parquet"),
        ("stack", "personal", "B_cohort_timecausal_features.parquet"),
        # 레거시 CSV 버전
        ("stack", "personal", "A_timecausal_features.csv"),
        ("stack", "personal", "B_timecausal_features.csv"),
        ("stack", "personal", "A_cohort_timecausal_features.csv"),
        ("stack", "personal", "B_cohort_timecausal_features.csv"),
    ]

    deleted_size = 0
    for parts in deletable_patterns:
        fpath = os.path.join(version_dir, *parts)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            os.remove(fpath)
            deleted_size += size
            logger.debug(f"[Cleanup] 삭제: {os.path.join(*parts)} ({size / 1024 / 1024:.1f}MB)")

    if deleted_size > 0:
        logger.info(f"[Cleanup] 불필요 산출물 삭제 완료: {deleted_size / 1024 / 1024:.1f}MB 절약")


def _store_artifacts_to_db(run_id: int, version_dir: str):
    """모든 아티팩트(JSON/Parquet/PKL)를 DB BLOB으로 저장하고 version_dir을 삭제한다."""
    from src.core.database import store_artifacts_batch

    _EXT_MAP = {".json": "json", ".parquet": "parquet", ".pkl": "pkl"}
    artifacts = []

    for dirpath, _, filenames in os.walk(version_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname)[1].lower()
            artifact_type = _EXT_MAP.get(ext)
            if artifact_type is None:
                continue

            rel_key = os.path.relpath(fpath, version_dir).replace(os.sep, "/")
            with open(fpath, "rb") as f:
                data = f.read()
            artifacts.append((rel_key, artifact_type, data))

    if artifacts:
        stored = store_artifacts_batch(run_id, artifacts)
        logger.info(f"[Training {run_id}] DB에 아티팩트 {stored}건 저장 완료 (PKL 포함)")

    # 디스크에서 전체 삭제
    shutil.rmtree(version_dir, ignore_errors=True)
    logger.info(f"[Training {run_id}] version_dir 삭제: {version_dir}")
