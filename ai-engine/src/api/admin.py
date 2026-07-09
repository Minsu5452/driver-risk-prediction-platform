import asyncio
import hashlib
import json
import logging
import os
import secrets
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.upload import COLUMN_MAPPING_A, COLUMN_MAPPING_B
from src.core.constants import PROJECT_ROOT, VERSIONS_DIR
from src.core.database import (
    insert_upload,
    insert_training_run,
    get_training_run,
    get_latest_training_run,
    get_all_training_runs,
    get_all_model_versions,
    get_model_version,
    get_active_model_version,
    set_active_model_version,
    delete_model_version,
    insert_upload_metadata,
    get_all_uploads_with_metadata,
    hard_delete_upload,
    bulk_hard_delete_uploads,
    get_active_uploads_by_date_range,
    reset_all_uploads,
    check_duplicate_hash,
    reset_all_training_runs,
    reset_all_model_versions,
    upsert_exam_records,
    upsert_sago_records,
    get_data_summary,
)
from src.services.training_service import (
    run_training_pipeline,
    is_training_running,
    get_current_run_id,
    request_cancel,
)

logger = logging.getLogger("ai-engine")

router = APIRouter(prefix="/admin")

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

TYPE_LABELS_KR = {
    "a_exam": "신규검사 데이터",
    "b_exam": "자격유지검사 데이터",
    "a_sago": "신규검사 사고 데이터",
    "b_sago": "자격유지검사 사고 데이터",
}

def _load_admin_config():
    """
    관리자 계정 설정을 로드한다.

    우선순위:
        1) admin.conf 파일 (PROJECT_ROOT/admin.conf 또는 그 상위 디렉토리)
        2) 환경변수 ADMIN_USERNAME / ADMIN_PASSWORD
        3) 기본값 (admin / 1)

    파일 기반 설정이 1순위인 이유:
        Windows cmd 의 'setlocal enabledelayedexpansion' 상태에서는 '!' 문자가
        지연 확장 마커로 취급되어 `set "ADMIN_PASSWORD=change-this!"` 의 '!' 가
        잘려나가는 버그가 발생한다. 파일에 직접 기록하면 cmd escaping 과 무관해
        '!' 를 포함한 비밀번호도 안전하게 전달된다.

    admin.conf 포맷 (한 줄씩, `#` 는 주석):
        username=admin
        password=change-this!
    """
    conf_candidates = []
    env_conf = os.environ.get("RISK_ADMIN_CONF")
    if env_conf:
        conf_candidates.append(Path(env_conf))
    conf_candidates.append(Path(PROJECT_ROOT) / "admin.conf")
    conf_candidates.append(Path(PROJECT_ROOT).parent / "admin.conf")  # C:\DriverRisk-Platform\admin.conf

    file_username = None
    file_password = None
    loaded_from = None

    for candidate in conf_candidates:
        try:
            if not candidate.exists():
                continue
            # 여러 인코딩 시도 (Windows 메모장 저장 시 UTF-16 BOM 가능)
            raw = None
            for enc in ("utf-8-sig", "utf-8", "utf-16", "cp949"):
                try:
                    raw = candidate.read_text(encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if raw is None:
                logger.warning(f"관리자 설정 파일 인코딩 파악 실패: {candidate}")
                continue

            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().lower()
                v = v.strip()
                # 따옴표로 감싸진 값 허용
                if len(v) >= 2 and (
                    (v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")
                ):
                    v = v[1:-1]
                if k in ("username", "user", "id"):
                    file_username = v
                elif k in ("password", "pass", "pw"):
                    file_password = v

            loaded_from = str(candidate)
            break
        except Exception as e:
            logger.warning(f"관리자 설정 파일 읽기 실패 ({candidate}): {e}")

    # 환경변수 fallback
    env_username = os.environ.get("ADMIN_USERNAME")
    env_password = os.environ.get("ADMIN_PASSWORD")

    # 최종 우선순위: 파일 > 환경변수 > 기본값
    username = file_username or env_username or "admin"
    password = file_password or env_password or "1"

    # 기동 로그 — 비밀번호는 일부만 노출해 설정 경로 추적 가능
    if file_password:
        src = f"file ({loaded_from})"
    elif env_password:
        src = "env"
    else:
        src = "default"
    if password:
        masked = password[:2] + "*" * max(0, len(password) - 2)
    else:
        masked = "(empty)"
    logger.info(f"관리자 계정 로드: username={username!r} password={masked!r} source={src}")

    return username, password


ADMIN_USERNAME, ADMIN_PASSWORD = _load_admin_config()

# 활성 관리자 토큰 (서버 재시작 시 초기화)
_active_tokens: set[str] = set()


class LoginRequest(BaseModel):
    username: str
    password: str


class BulkDeleteRequest(BaseModel):
    date_from: str  # "YYYY-MM"
    date_to: str  # "YYYY-MM"


class ResetRequest(BaseModel):
    confirm: bool


def verify_admin_token(authorization: str = Header(None)):
    """관리자 토큰 검증 미들웨어."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    token = authorization.split(" ", 1)[1]
    if token not in _active_tokens:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다. 다시 로그인해주세요.")


@router.post("/login")
async def admin_login(req: LoginRequest):
    if req.username == ADMIN_USERNAME and req.password == ADMIN_PASSWORD:
        token = secrets.token_hex(32)
        _active_tokens.add(token)
        return {"success": True, "message": "로그인 성공", "token": token}
    raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")


@router.get("/verify")
async def verify_session(authorization: str = Header(None)):
    """현재 토큰이 유효한지 확인한다."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증이 필요합니다.")
    token = authorization.split(" ", 1)[1]
    if token not in _active_tokens:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
    return {"valid": True}


@router.post("/logout")
async def admin_logout(authorization: str = Header(None)):
    """로그아웃 (토큰 무효화)."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        _active_tokens.discard(token)
    return {"success": True}


def _validate_file_columns(cols: set, file_type: str, filename: str) -> dict:
    """파일 컬럼 집합이 해당 유형에 필요한 모든 컬럼을 포함하는지 검증한다."""
    # 검사 파일 필수 공통 컬럼
    common_exam = {"이름", "주민번호", "주민번호_hash", "만나이", "수검일"}
    # 사고/위반 파일 필수 컬럼
    required_sago = {"이름", "주민번호", "주민번호_hash", "사고(위반)구분", "사고(위반)일자", "검사명"}

    if file_type == "a_exam":
        required = set(COLUMN_MAPPING_A.keys()) | common_exam
    elif file_type == "b_exam":
        required = set(COLUMN_MAPPING_B.keys()) | common_exam
    elif file_type in ("a_sago", "b_sago"):
        required = required_sago
    else:
        return {"filename": filename, "type": file_type, "valid": False,
                "error": "알 수 없는 파일 유형", "missing": []}

    missing = sorted(list(required - cols))

    if missing:
        return {"filename": filename, "type": file_type, "valid": False,
                "missing": missing, "error": None}
    return {"filename": filename, "type": file_type, "valid": True,
            "missing": [], "error": None}


from src.core.constants import UPLOAD_CHUNK_BYTES, MAX_UPLOAD_FILE_SIZE
CHUNK_SIZE = UPLOAD_CHUNK_BYTES


MAX_UPLOAD_SIZE = MAX_UPLOAD_FILE_SIZE


async def _stream_save_and_hash(upload_file: UploadFile, dest_path: str) -> str:
    """파일을 청크 단위로 디스크에 스트리밍 저장하며 동시에 SHA256 해시를 계산한다.

    파일 크기와 무관하게 메모리 사용량이 ~1MB로 유지된다.
    """
    h = hashlib.sha256()
    file_size = 0
    with open(dest_path, "wb") as f:
        while True:
            chunk = await upload_file.read(CHUNK_SIZE)
            if not chunk:
                break
            file_size += len(chunk)
            if file_size > MAX_UPLOAD_SIZE:
                os.remove(dest_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"파일 '{upload_file.filename}'이 10GB를 초과합니다.",
                )
            f.write(chunk)
            h.update(chunk)
    return h.hexdigest()


def _read_excel_once(file_path: str) -> pd.DataFrame:
    """Excel 파일을 1회만 읽어 DataFrame으로 반환한다 (calamine 우선)."""
    try:
        return pd.read_excel(file_path, dtype=str, engine="calamine")
    except Exception:
        return pd.read_excel(file_path, dtype=str)


def _detect_and_validate(file_path: str, filename: str):
    """파일 유형 감지 + 컬럼 검증 + 데이터 캐시를 1회 읽기로 수행한다."""
    from src.data.transform import detect_file_type

    file_type = detect_file_type(file_path)

    try:
        df = _read_excel_once(file_path)
    except Exception as e:
        validation = {"filename": filename, "type": file_type, "valid": False,
                      "error": f"파일을 읽을 수 없습니다: {str(e)}", "missing": []}
        return file_type, validation, None

    cols = set(df.columns)
    validation = _validate_file_columns(cols, file_type, filename)

    # 빈 파일 검증
    if validation["valid"] and len(df) == 0:
        validation["valid"] = False
        validation["error"] = "업로드된 파일에 데이터가 없습니다."

    # 검증 통과 시 DataFrame을 캐시로 반환
    cached_df = df if validation["valid"] else None
    return file_type, validation, cached_df


@router.post("/training/upload")
async def upload_training_files(files: List[UploadFile] = File(...), _admin=Depends(verify_admin_token)):
    """학습용 Excel 파일 업로드 (최대 4개, 컬럼 검증 포함)."""
    if len(files) > 4:
        raise HTTPException(status_code=400, detail="최대 4개 파일만 업로드 가능합니다.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_subdir = os.path.join(UPLOAD_DIR, timestamp)
    os.makedirs(upload_subdir, exist_ok=True)

    t_upload_start = time.monotonic()

    file_paths = {}
    hash_parts = []
    detected_types = []
    validation_results = []
    saved_files = []  # (filename, file_path) for Phase 2

    # 1단계: 각 파일을 디스크에 병렬 스트리밍 저장 + 해시 계산 (파일당 ~1MB 메모리)
    t0 = time.monotonic()

    async def _save_one(f):
        fp = os.path.join(upload_subdir, f.filename)
        fh = await _stream_save_and_hash(f, fp)
        return f.filename, fp, fh

    save_results = await asyncio.gather(*[_save_one(f) for f in files])
    for filename, file_path, file_hash in save_results:
        hash_parts.append(file_hash)
        saved_files.append((filename, file_path))

    logger.info(f"[Upload] Phase 1 파일 저장 완료: {time.monotonic() - t0:.1f}s ({len(files)}개 파일)")

    # 2단계: 스레드 풀에서 유형 감지 + 검증 + 1회 읽기 동시 실행
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()
    detect_futures = []
    cached_dfs = {}  # file_type -> DataFrame (검증 통과한 파일 캐시)
    for filename, file_path in saved_files:
        future = loop.run_in_executor(None, _detect_and_validate, file_path, filename)
        detect_futures.append((filename, file_path, future))

    for filename, file_path, future in detect_futures:
        try:
            file_type, validation, cached_df = await future
            file_paths[file_type] = file_path
            if cached_df is not None:
                cached_dfs[file_type] = cached_df
            detected_types.append({"filename": filename, "type": file_type})
            validation_results.append(validation)
            logger.info(f"Uploaded {filename} -> detected as {file_type}")
        except Exception as e:
            detected_types.append(
                {"filename": filename, "type": "unknown", "error": str(e)}
            )
            validation_results.append({
                "filename": filename, "type": "unknown", "valid": False,
                "missing": [], "error": str(e),
            })
            logger.warning(f"Could not detect type for {filename}: {e}")

    logger.info(f"[Upload] Phase 2 감지+검증+읽기 완료: {time.monotonic() - t0:.1f}s")

    # 검증 실패 확인
    failed = [v for v in validation_results if not v["valid"]]
    if failed:
        # 업로드된 파일 정리
        shutil.rmtree(upload_subdir, ignore_errors=True)

        error_lines = []
        for f in failed:
            if f.get("error"):
                error_lines.append(f"[{f['filename']}] {f['error']}")
            elif f["missing"]:
                error_lines.append(
                    f"[{f['filename']}] ({f['type']}) 누락 컬럼: {', '.join(f['missing'])}"
                )
        raise HTTPException(status_code=400, detail="\n".join(error_lines))

    file_hash = hashlib.sha256("".join(sorted(hash_parts)).encode()).hexdigest()

    # 중복 업로드 확인 — 동일 파일이면 저장하지 않고 거부
    duplicates = check_duplicate_hash(file_hash)
    if duplicates:
        # 중복 파일의 임시 저장 파일 제거
        shutil.rmtree(upload_subdir, ignore_errors=True)

        dup = duplicates[0]
        # UTC → KST 변환
        KST = timezone(timedelta(hours=9))
        raw_dt = dup.get("created_at", "")
        try:
            utc_dt = datetime.fromisoformat(raw_dt).replace(tzinfo=timezone.utc)
            dup_date = utc_dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")
        except Exception:
            dup_date = raw_dt[:16].replace("T", " ") if raw_dt else "알 수 없음"
        raise HTTPException(
            status_code=409,
            detail=(
                f"동일한 파일이 이미 업로드되어 있습니다.\n"
                f"\n"
                f"기존 업로드: 데이터 관리 #{dup['id']} ({dup_date})\n"
                f"파일 내용을 변경하거나, 기존 업로드를 삭제 후 다시 시도하세요."
            ),
        )

    # 파일명만 저장 (디스크 path 아님 — Excel 원본은 즉시 삭제됨, UI 표시용 파일명만 보존)
    upload_id = insert_upload(
        a_exam_path=os.path.basename(file_paths["a_exam"]) if file_paths.get("a_exam") else None,
        b_exam_path=os.path.basename(file_paths["b_exam"]) if file_paths.get("b_exam") else None,
        a_sago_path=os.path.basename(file_paths["a_sago"]) if file_paths.get("a_sago") else None,
        b_sago_path=os.path.basename(file_paths["b_sago"]) if file_paths.get("b_sago") else None,
        file_hash=file_hash,
    )

    # 데이터 변환 + DB UPSERT + 메타데이터 추출
    t0 = time.monotonic()
    loop = asyncio.get_running_loop()
    try:
        record_counts = await loop.run_in_executor(
            None, _transform_and_upsert, upload_id, file_paths, cached_dfs
        )
    except ValueError as e:
        # 모든 파일 변환 실패 → 업로드 레코드 삭제 + 파일 정리
        hard_delete_upload(upload_id)
        shutil.rmtree(upload_subdir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))

    logger.info(f"[Upload] Phase 3 변환+DB 완료: {time.monotonic() - t0:.1f}s")

    # DB 인제스트 성공 → 원본 Excel 파일 즉시 삭제 (보안: 민감 데이터 디스크 잔존 방지)
    shutil.rmtree(upload_subdir, ignore_errors=True)
    logger.info(f"[Upload] 전체 완료: {time.monotonic() - t_upload_start:.1f}s — Excel 파일 삭제: {upload_subdir}")

    return {
        "upload_id": upload_id,
        "files": detected_types,
        "validation": validation_results,
        "record_counts": record_counts,
    }


def _safe_int(val, default=0):
    """NaN-safe 정수 변환. float('nan'), None, 빈 문자열 등을 안전하게 처리한다."""
    try:
        v = float(val)
        return int(v) if not pd.isna(v) else default
    except (ValueError, TypeError):
        return default


def _transform_and_upsert(upload_id: int, file_paths: dict, cached_dfs: dict | None = None) -> dict:
    """업로드된 Excel 파일을 변환하고 DB에 UPSERT한다.

    Args:
        upload_id: 업로드 ID.
        file_paths: 파일 유형별 경로 dict.
        cached_dfs: _detect_and_validate에서 미리 읽은 DataFrame 캐시 (1회 읽기 최적화).

    Returns:
        각 파일 유형별 {"total", "new", "existing"} dict.

    Raises:
        ValueError: 모든 파일의 변환이 실패한 경우.
    """
    from src.data.transform import (
        transform_exam_to_train_format,
        transform_sago_to_train_format,
    )
    from src.core.database import get_db

    if cached_dfs is None:
        cached_dfs = {}

    record_counts = {}
    errors = []

    # 벌크 INSERT 최적화: 전체 파일 처리 전에 보조 인덱스 DROP (1회)
    # 단, idx_exam_pk(primary_key)는 유지한다 — 업로드 도중 들어오는 예측 요청이
    # exam_records를 primary_key로 조회하므로, 이 인덱스를 내리면 그 동안 풀스캔이 된다.
    # (idx_exam_domain·idx_sago_pk는 예측 경로가 쓰지 않으므로 DROP해도 무방.)
    conn = get_db()
    conn.execute("DROP INDEX IF EXISTS idx_exam_domain")
    conn.execute("DROP INDEX IF EXISTS idx_sago_pk")

    type_config = {
        "a_exam": ("A", "exam"),
        "b_exam": ("B", "exam"),
        "a_sago": ("A", "sago"),
        "b_sago": ("B", "sago"),
    }

    for file_type, (domain, data_kind) in type_config.items():
        path = file_paths.get(file_type)
        if not path or not os.path.exists(path):
            continue

        try:
            t_file = time.monotonic()

            # 캐시된 DataFrame 사용 (없으면 1회 읽기)
            raw = cached_dfs.pop(file_type, None)
            if raw is None:
                raw = _read_excel_once(path)

            if data_kind == "exam":
                t0 = time.monotonic()
                df = transform_exam_to_train_format(raw, domain)

                # 주민번호_hash 누락 행 제외
                empty_pk = df["PrimaryKey"].str.strip() == ""
                n_empty = int(empty_pk.sum())
                skipped_rows_info = ""
                if n_empty > 0:
                    empty_indices = list(df.index[empty_pk] + 2)  # Excel 행번호 (헤더+1-indexed)
                    if len(empty_indices) <= 10:
                        skipped_rows_info = f" — 행 {', '.join(str(r) for r in empty_indices)}"
                    else:
                        skipped_rows_info = f" — 행 {', '.join(str(r) for r in empty_indices[:10])} 외 {len(empty_indices)-10}건"
                    df = df[~empty_pk].reset_index(drop=True)
                    logger.warning(f"[Upload] {file_type}: 주민번호 누락 {n_empty}건 제외{skipped_rows_info}")

                t_transform = time.monotonic() - t0

                feature_codes = list(
                    (COLUMN_MAPPING_A if domain == "A" else COLUMN_MAPPING_B).values()
                )

                # C레벨 pd.to_json으로 벡터화
                t0 = time.monotonic()
                feat_df = df[feature_codes].fillna("").astype(str)
                features_json_lines = feat_df.to_json(
                    orient="records", lines=True, force_ascii=False
                ).rstrip("\n").split("\n") if len(feat_df) > 0 else []

                n = len(df)
                age_col = df.get("Age", pd.Series([""] * n)).fillna("").astype(str)
                exam_age_raw = df.get("exam_age", pd.Series([pd.NA] * n))
                exam_age_list = [int(v) if pd.notna(v) else None for v in exam_age_raw]
                birth_col = df.get("birth_yyyymmdd", pd.Series([""] * n)).fillna("").astype(str)
                td_col = df.get("TestDate", pd.Series([""] * n)).fillna("").astype(str)
                params = list(zip(
                    df["Test_id"].tolist(),
                    [domain] * n,
                    df["PrimaryKey"].tolist(),
                    age_col.tolist(),
                    exam_age_list,
                    birth_col.tolist(),
                    td_col.tolist(),
                    features_json_lines,
                    [upload_id] * n,
                ))
                t_serialize = time.monotonic() - t0

                t0 = time.monotonic()
                result = upsert_exam_records(params, upload_id, tuple_mode=True)
                t_db = time.monotonic() - t0

                result["skipped_empty_pk"] = n_empty
                result["skipped_rows_info"] = skipped_rows_info
                record_counts[file_type] = result
                logger.info(
                    f"[Upload] {file_type}: {result['total']} total, "
                    f"{result['new']} new, {result['existing']} existing"
                    f"{f', 주민번호 누락 {n_empty}건 제외' if n_empty else ''} "
                    f"(transform {t_transform:.1f}s, serialize {t_serialize:.1f}s, db {t_db:.1f}s, "
                    f"total {time.monotonic() - t_file:.1f}s)"
                )

                _store_metadata_from_df(upload_id, domain, "exam", df, "TestDate", record_count=result["new"])

            else:  # sago
                t0 = time.monotonic()
                df = transform_sago_to_train_format(raw, domain)

                # 주민번호_hash 누락 행 제외
                empty_pk = df["PrimaryKey"].str.strip() == ""
                n_empty = int(empty_pk.sum())
                skipped_rows_info = ""
                if n_empty > 0:
                    empty_indices = list(df.index[empty_pk] + 2)
                    if len(empty_indices) <= 10:
                        skipped_rows_info = f" — 행 {', '.join(str(r) for r in empty_indices)}"
                    else:
                        skipped_rows_info = f" — 행 {', '.join(str(r) for r in empty_indices[:10])} 외 {len(empty_indices)-10}건"
                    df = df[~empty_pk].reset_index(drop=True)
                    logger.warning(f"[Upload] {file_type}: 주민번호 누락 {n_empty}건 제외{skipped_rows_info}")

                t_transform = time.monotonic() - t0

                # 벡터화: iterrows() 대신 DataFrame 연산으로 일괄 변환
                t0 = time.monotonic()
                records = [
                    {
                        "primary_key": pk,
                        "acc_date": str(ad) if pd.notna(ad) else "",
                        "acc_type": str(at) if pd.notna(at) else "",
                        "domain": str(t) if pd.notna(t) else domain,
                        "count_1": _safe_int(c1),
                        "count_2": _safe_int(c2),
                        "count_3": _safe_int(c3),
                        "count_4": _safe_int(c4),
                        "count_5": _safe_int(c5),
                        "count_6": _safe_int(c6),
                    }
                    for pk, ad, at, t, c1, c2, c3, c4, c5, c6 in zip(
                        df["PrimaryKey"],
                        df.get("AccDate", pd.Series([""] * len(df))),
                        df.get("AccType", pd.Series([""] * len(df))),
                        df.get("Test", pd.Series([domain] * len(df))),
                        df.get("Count_1", pd.Series([0] * len(df))),
                        df.get("Count_2", pd.Series([0] * len(df))),
                        df.get("Count_3", pd.Series([0] * len(df))),
                        df.get("Count_4", pd.Series([0] * len(df))),
                        df.get("Count_5", pd.Series([0] * len(df))),
                        df.get("Count_6", pd.Series([0] * len(df))),
                    )
                ]
                t_serialize = time.monotonic() - t0

                t0 = time.monotonic()
                result = upsert_sago_records(records, upload_id)
                t_db = time.monotonic() - t0

                result["skipped_empty_pk"] = n_empty
                result["skipped_rows_info"] = skipped_rows_info
                record_counts[file_type] = result
                logger.info(
                    f"[Upload] {file_type}: {result['total']} total, "
                    f"{result['new']} new, {result['existing']} existing"
                    f"{f', 주민번호 누락 {n_empty}건 제외' if n_empty else ''} "
                    f"(transform {t_transform:.1f}s, serialize {t_serialize:.1f}s, db {t_db:.1f}s, "
                    f"total {time.monotonic() - t_file:.1f}s)"
                )

                _store_metadata_from_df(upload_id, domain, "sago", df, "AccDate", record_count=result["new"])

        except Exception as e:
            logger.error(f"[Upload] Failed to transform+upsert {file_type}: {e}")
            errors.append(f"{file_type}: {str(e)}")

    # 벌크 INSERT 완료 후 보조 인덱스 재생성 (1회) + WAL 체크포인트
    t0 = time.monotonic()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exam_domain ON exam_records(domain)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_exam_pk ON exam_records(primary_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sago_pk ON sago_records(primary_key)")
    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    logger.info(f"[Upload] 인덱스 재생성 + 체크포인트: {time.monotonic() - t0:.1f}s")

    if errors and not record_counts:
        raise ValueError(f"데이터 변환 실패: {'; '.join(errors)}")

    if errors:
        for err_msg in errors:
            ft = err_msg.split(":")[0]
            record_counts[ft] = {"total": 0, "new": 0, "existing": 0, "error": err_msg}

    return record_counts


def _store_metadata_from_df(
    upload_id: int, domain: str, file_type: str,
    df: pd.DataFrame, date_col: str,
    record_count: int | None = None,
):
    """변환된 DataFrame에서 메타데이터(건수, 날짜 범위)를 추출하여 저장한다."""
    if record_count is None:
        record_count = len(df)
    date_from = None
    date_to = None

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if len(dates) > 0:
            date_from = dates.min().strftime("%Y%m")
            date_to = dates.max().strftime("%Y%m")

    insert_upload_metadata(
        upload_id=upload_id,
        domain=domain,
        file_type=file_type,
        record_count=record_count,
        date_from=date_from,
        date_to=date_to,
    )


def _extract_and_store_metadata(upload_id: int, file_paths: dict):
    """업로드된 파일에서 날짜 범위와 레코드 수를 추출한다."""
    type_map = {
        "a_exam": ("A", "exam"),
        "b_exam": ("B", "exam"),
        "a_sago": ("A", "sago"),
        "b_sago": ("B", "sago"),
    }

    for key, (domain, file_type) in type_map.items():
        path = file_paths.get(key)
        if not path or not os.path.exists(path):
            continue

        try:
            # 1단계: 헤더만 읽어 날짜 컬럼 탐색
            try:
                header_df = pd.read_excel(path, nrows=0, dtype=str, engine="calamine")
            except Exception:
                header_df = pd.read_excel(path, nrows=0, dtype=str)
            columns = set(header_df.columns)

            date_col = None
            if file_type == "exam":
                for col_name in ["수검일", "수검일자", "검사일", "검사일자"]:
                    if col_name in columns:
                        date_col = col_name
                        break
            else:
                for col_name in ["사고(위반)일자", "사고일자", "위반일자", "사고일"]:
                    if col_name in columns:
                        date_col = col_name
                        break

            # 2단계: 날짜 컬럼만 로드 (없으면 첫 컬럼으로 행 수 계산)
            date_from = None
            date_to = None
            try:
                if date_col:
                    col_df = pd.read_excel(path, usecols=[date_col], dtype=str, engine="calamine")
                else:
                    col_df = pd.read_excel(path, usecols=[0], dtype=str, engine="calamine")
            except Exception:
                if date_col:
                    col_df = pd.read_excel(path, usecols=[date_col], dtype=str)
                else:
                    col_df = pd.read_excel(path, usecols=[0], dtype=str)

            record_count = len(col_df)

            if date_col and date_col in col_df.columns:
                dates = pd.to_datetime(col_df[date_col], errors="coerce").dropna()
                if len(dates) > 0:
                    date_from = dates.min().strftime("%Y%m")
                    date_to = dates.max().strftime("%Y%m")

            insert_upload_metadata(
                upload_id=upload_id,
                domain=domain,
                file_type=file_type,
                record_count=record_count,
                date_from=date_from,
                date_to=date_to,
            )
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {key}: {e}")


@router.post("/training/start")
async def start_training(_admin=Depends(verify_admin_token)):
    """DB에 저장된 데이터를 사용하여 학습을 시작한다."""
    if is_training_running():
        current_run = get_current_run_id()
        raise HTTPException(
            status_code=409,
            detail=f"학습이 이미 진행 중입니다. (Run ID: {current_run})",
        )

    # DB에서 데이터 건수 확인
    summary = get_data_summary()
    if summary["a_exam"] == 0 and summary["b_exam"] == 0:
        raise HTTPException(
            status_code=400,
            detail="검사 데이터가 없습니다. 데이터를 먼저 업로드해주세요.",
        )

    run_id = insert_training_run(upload_id=None)

    # 백그라운드에서 학습 시작 (DB에서 데이터 로드)
    asyncio.create_task(run_training_pipeline(run_id))

    return {
        "run_id": run_id,
        "status": "started",
        "data_summary": summary,
    }


@router.post("/training/cancel")
async def cancel_training(_admin=Depends(verify_admin_token)):
    """현재 진행 중인 학습의 중단을 요청한다."""
    if not is_training_running():
        raise HTTPException(status_code=400, detail="현재 진행 중인 학습이 없습니다.")

    request_cancel()
    return {"success": True, "message": "학습 중단이 요청되었습니다."}


@router.get("/training/status")
async def get_training_status(_admin=Depends(verify_admin_token)):
    """현재 또는 가장 최근 학습 상태를 반환한다."""
    current_run_id = get_current_run_id()

    if current_run_id:
        run = get_training_run(current_run_id)
        if run:
            return dict(run)

    latest = get_latest_training_run()
    if latest:
        return dict(latest)

    return {"status": "no_runs", "step_detail": None}


@router.get("/training/metrics")
async def get_training_metrics(_admin=Depends(verify_admin_token)):
    """최근 완료된 학습의 메트릭을 반환한다."""
    latest = get_latest_training_run()
    if not latest or latest["status"] != "completed":
        raise HTTPException(status_code=404, detail="완료된 학습이 없습니다.")

    metrics = {}
    if latest["metrics_json"]:
        metrics = json.loads(latest["metrics_json"])

    return {"run_id": latest["id"], "metrics": metrics}


@router.get("/training/history")
async def get_training_history(_admin=Depends(verify_admin_token)):
    """전체 학습 실행 이력을 반환한다."""
    runs = get_all_training_runs()
    return {"runs": [dict(r) for r in runs]}


@router.post("/training/reset")
async def reset_training_history(req: ResetRequest, _admin=Depends(verify_admin_token)):
    """전체 학습 이력을 초기화한다 (모든 학습 실행 완전 삭제)."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="confirm이 true여야 합니다.")

    if is_training_running():
        raise HTTPException(status_code=409, detail="학습이 진행 중일 때는 이력을 초기화할 수 없습니다.")

    deleted_count = reset_all_training_runs()
    return {
        "success": True,
        "deleted_count": deleted_count,
        "message": f"학습 이력 {deleted_count}건이 초기화되었습니다.",
    }


@router.post("/models/reload")
async def reload_models(_admin=Depends(verify_admin_token)):
    """모델 아티팩트를 수동으로 리로드한다."""
    try:
        from src.inference.loader import reload_all_artifacts
        from src.inference.seq_engine import reload_seq_artifacts
        from src.core.database import get_run_id_for_active_version

        run_id = get_run_id_for_active_version()
        if run_id is None:
            raise HTTPException(status_code=404, detail="활성 모델 버전이 없습니다. 학습을 먼저 실행하세요.")
        reload_all_artifacts(["A", "B"], run_id=run_id)
        reload_seq_artifacts(run_id=run_id)
        return {"success": True, "message": "모델이 성공적으로 리로드되었습니다."}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 모델 버전 엔드포인트 ──

@router.get("/models/versions")
async def list_model_versions(_admin=Depends(verify_admin_token)):
    """전체 모델 버전 목록을 반환한다."""
    versions = get_all_model_versions()
    result = []
    for v in versions:
        item = dict(v)
        if item.get("metrics_json"):
            try:
                item["metrics"] = json.loads(item["metrics_json"])
            except (json.JSONDecodeError, TypeError):
                item["metrics"] = None
        else:
            item["metrics"] = None
        result.append(item)
    return {"versions": result}


@router.post("/models/versions/{version_id}/activate")
async def activate_model_version(version_id: int, _admin=Depends(verify_admin_token)):
    """특정 모델 버전을 활성화한다 (롤백)."""
    version = get_model_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="버전을 찾을 수 없습니다.")

    run_id = version.get("run_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="해당 버전에 run_id가 없습니다.")

    # DB에 아티팩트가 있는지 확인
    from src.core.database import load_artifact
    if load_artifact(run_id, "stack/A/features.json") is None:
        raise HTTPException(status_code=400, detail="해당 버전의 아티팩트가 존재하지 않습니다.")

    # 이전 활성 버전 기록 (롤백용)
    prev_active = get_active_model_version()
    prev_id = prev_active["id"] if prev_active else None

    # 활성 버전 설정
    set_active_model_version(version_id)

    # 활성화된 버전에서 모델 리로드
    try:
        from src.inference.loader import reload_all_artifacts
        from src.inference.seq_engine import reload_seq_artifacts

        reload_all_artifacts(["A", "B"], run_id=run_id)
        reload_seq_artifacts(run_id=run_id)
    except Exception as e:
        logger.error(f"Failed to reload models after activation: {e}")
        # 리로드 실패 → 이전 버전으로 DB 롤백
        if prev_id is not None:
            set_active_model_version(prev_id)
            logger.info(f"Rolled back to previous version id={prev_id}")
        raise HTTPException(status_code=500, detail=f"모델 리로드 실패: {str(e)}")

    return {"success": True, "message": f"버전 {version['version_label']}이(가) 활성화되었습니다."}


@router.delete("/models/versions/{version_id}")
async def delete_version(version_id: int, _admin=Depends(verify_admin_token)):
    """비활성 모델 버전을 삭제한다."""
    version = get_model_version(version_id)
    if not version:
        raise HTTPException(status_code=404, detail="버전을 찾을 수 없습니다.")

    if version["is_active"]:
        raise HTTPException(status_code=400, detail="활성 버전은 삭제할 수 없습니다.")

    # DB 아티팩트 BLOB 삭제
    if version.get("run_id"):
        from src.core.database import delete_artifacts_by_run
        delete_artifacts_by_run(version["run_id"])

    # DB 레코드 삭제
    delete_model_version(version_id)

    return {"success": True, "message": f"버전 {version['version_label']}이(가) 삭제되었습니다."}


@router.get("/models/disk-usage")
async def get_disk_usage(_admin=Depends(verify_admin_token)):
    """전체 모델 아티팩트의 디스크 사용량을 반환한다."""

    def _dir_size(path: str) -> int:
        total = 0
        if os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
        return total

    from src.core.database import get_artifacts_total_size

    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    versions_disk = _dir_size(VERSIONS_DIR)
    artifacts_disk = _dir_size(artifacts_dir)
    db_blob_size = get_artifacts_total_size()

    return {
        "versions": versions_disk + db_blob_size,
        "original": artifacts_disk - versions_disk,
        "total": artifacts_disk + db_blob_size,
    }


# ── 데이터셋 엔드포인트 ──

@router.get("/datasets")
async def list_datasets(_admin=Depends(verify_admin_token)):
    """전체 업로드 목록을 메타데이터와 함께 반환한다."""
    uploads = get_all_uploads_with_metadata()
    return {"datasets": uploads}


@router.get("/datasets/summary")
async def datasets_summary(_admin=Depends(verify_admin_token)):
    """학습에 사용될 데이터의 정확한 건수를 반환한다 (중복 제거 후)."""
    return get_data_summary()


@router.get("/datasets/{upload_id}")
async def get_dataset_detail(upload_id: int, _admin=Depends(verify_admin_token)):
    """데이터셋 상세 정보를 반환한다."""
    from src.core.database import get_db
    db = get_db()
    row = db.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")
    result = dict(row)
    meta_rows = db.execute(
        "SELECT * FROM upload_metadata WHERE upload_id = ?", (upload_id,)
    ).fetchall()
    result["metadata"] = [dict(m) for m in meta_rows]
    return result


@router.delete("/datasets/{upload_id}")
async def delete_dataset(upload_id: int, _admin=Depends(verify_admin_token)):
    """데이터셋을 삭제한다 (업로드 + 종속 exam/sago/metadata 모두 한 트랜잭션으로)."""
    from src.core.database import get_db
    db = get_db()
    row = db.execute("SELECT * FROM uploads WHERE id = ?", (upload_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다.")

    try:
        records_deleted = hard_delete_upload(upload_id)
    except Exception as e:
        logger.exception(f"delete_dataset({upload_id}) failed")
        raise HTTPException(status_code=500, detail=f"데이터셋 삭제 실패: {e}")

    return {
        "success": True,
        "message": "데이터셋이 삭제되었습니다.",
        "records_deleted": records_deleted,
    }


@router.post("/datasets/bulk-delete/preview")
async def bulk_delete_preview(req: BulkDeleteRequest, _admin=Depends(verify_admin_token)):
    """업로드 날짜 범위로 삭제될 데이터셋을 미리 확인한다."""
    uploads = get_active_uploads_by_date_range(req.date_from, req.date_to)
    return {
        "count": len(uploads),
        "upload_ids": [u["id"] for u in uploads],
        "uploads": [
            {"id": u["id"], "created_at": u.get("created_at")}
            for u in uploads
        ],
    }


@router.post("/datasets/bulk-delete")
async def bulk_delete_datasets(req: BulkDeleteRequest, _admin=Depends(verify_admin_token)):
    """업로드 날짜 범위로 데이터셋을 일괄 삭제한다 (한 트랜잭션, 부분 실패 없음)."""
    uploads = get_active_uploads_by_date_range(req.date_from, req.date_to)

    if not uploads:
        return {"success": True, "deleted_count": 0, "message": "삭제할 데이터가 없습니다."}

    upload_ids = [u["id"] for u in uploads]
    try:
        deleted_count = bulk_hard_delete_uploads(upload_ids)
    except Exception as e:
        logger.exception(f"bulk_delete_datasets({upload_ids}) failed")
        raise HTTPException(status_code=500, detail=f"일괄 삭제 실패: {e}")

    return {
        "success": True,
        "deleted_count": deleted_count,
        "message": f"{deleted_count}건의 데이터가 삭제되었습니다.",
    }


@router.post("/datasets/reset")
async def reset_datasets(req: ResetRequest, _admin=Depends(verify_admin_token)):
    """전체 업로드 + 종속 데이터를 초기화한다 (한 트랜잭션)."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="confirm이 true여야 합니다.")

    try:
        # reset_all_uploads는 이제 exam_records, sago_records, upload_metadata도 함께 비움
        deleted_count = reset_all_uploads()
    except Exception as e:
        logger.exception("reset_datasets failed")
        raise HTTPException(status_code=500, detail=f"데이터셋 초기화 실패: {e}")

    return {
        "success": True,
        "deleted_count": deleted_count,
        "message": f"전체 {deleted_count}건의 데이터가 초기화되었습니다.",
    }


@router.post("/models/versions/reset")
async def reset_model_versions(req: ResetRequest, _admin=Depends(verify_admin_token)):
    """전체 모델 버전을 초기화한다 (모든 버전 + 아티팩트 파일 삭제)."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="confirm이 true여야 합니다.")

    if is_training_running():
        raise HTTPException(status_code=409, detail="학습이 진행 중일 때는 모델을 초기화할 수 없습니다.")

    deleted_count = reset_all_model_versions()

    # 비정상 종료 잔존물(고아 버전 디렉터리) 정리 — 안전망
    from src.core.database import cleanup_orphan_version_dirs
    orphans = cleanup_orphan_version_dirs()

    return {
        "success": True,
        "deleted_count": deleted_count,
        "orphans_cleaned": len(orphans),
        "message": f"모델 버전 {deleted_count}건이 초기화되었습니다.",
    }


@router.post("/system/reset-all")
async def reset_all_system(req: ResetRequest, _admin=Depends(verify_admin_token)):
    """전체 시스템 초기화: 학습 이력 + 모델 버전 + 데이터셋."""
    if not req.confirm:
        raise HTTPException(status_code=400, detail="confirm이 true여야 합니다.")

    if is_training_running():
        raise HTTPException(status_code=409, detail="학습이 진행 중일 때는 초기화할 수 없습니다.")

    results = {}

    # 1. 업로드 + 종속 데이터 (exam/sago/metadata) 한꺼번에 삭제
    results["datasets"] = reset_all_uploads()

    # 2. 모델 버전 + 아티팩트 BLOB (FK 종속 테이블 정리)
    results["model_versions"] = reset_all_model_versions()

    # 3. 학습 이력 마지막 — model_artifacts/model_versions 정리 후 FK 참조 없음
    results["training_runs"] = reset_all_training_runs()

    # 4. 비정상 종료 잔존물(고아 버전 디렉터리) 정리 — 안전망
    from src.core.database import cleanup_orphan_version_dirs
    orphans = cleanup_orphan_version_dirs()
    results["orphans_cleaned"] = len(orphans)

    return {
        "success": True,
        "results": results,
        "message": "전체 시스템이 초기화되었습니다.",
    }
