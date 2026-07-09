from fastapi import APIRouter, File, UploadFile, HTTPException
import asyncio
import pandas as pd
import os
import tempfile
from typing import List
from src.schemas import DriverInput, PredictionResponse
from src.services.prediction_service import predict_service as predict, predict_domain_df
from src.utils.age_calc import (
    age_to_code,
    build_birth_yyyymmdd,
    compute_current_age,
    compute_exam_age,
    normalize_test_date,
    parse_rrn_birth,
)
import logging

logger = logging.getLogger("ai-engine")

router = APIRouter()

# 한글 컬럼명 -> 학습 스키마 피처 코드 매핑
# 업로드된 Excel 파일의 한글 헤더를 모델이 기대하는 코드로 변환할 때 사용
# A형 (신규검사): 속도예측, 주의전환, 반응조절 등 9개 검사 영역
COLUMN_MAPPING_A = {
    "속도예측검사(출현방향)": "A1-1",
    "속도예측검사(이동속도)": "A1-2",
    "속도예측검사(오반응)": "A1-3",
    "속도예측검사(편차거리)": "A1-4",
    "정지거리예측검사(이동속도)": "A2-1",
    "정지거리예측검사(이동가속도)": "A2-2",
    "정지거리예측검사(오반응)": "A2-3",
    "정지거리예측검사(편차거리)": "A2-4",
    "주의전환검사(자극크기)": "A3-1",
    "주의전환검사(타겟위치)": "A3-2",
    "주의전환검사(자극좌우)": "A3-3",
    "주의전환검사(자극위치)": "A3-4",
    "주의전환검사(타겟자극조건)": "A3-5",
    "주의전환검사(오반응)": "A3-6",
    "주의전환검사(반응시간)": "A3-7",
    "반응조절검사(일치조건)": "A4-1",
    "반응조절검사(색상조건)": "A4-2",
    "반응조절검사(정오)": "A4-3",
    "반응조절검사(무반응)": "A4-4",
    "반응조절검사(반응시간)": "A4-5",
    "변화탐지검사(변화)": "A5-1",
    "변화탐지검사(정오)": "A5-2",
    "변화탐지검사(무반응)": "A5-3",
    "인지능력_정답수": "A6-1",
    "지각성향_정답수": "A7-1",
    "긍정왜곡_정답수": "A8-1",
    "반응일관_정답수": "A8-2",
    "정서안정성_정답수": "A9-1",
    "행동안정성_정답수": "A9-2",
    "현실판단력_정답수": "A9-3",
    "정신적민첩성_정답수": "A9-4",
    "생활스트레스_정답수": "A9-5",
}

# B형 (자격유지검사): 시야각, 신호등, 복합기능 등 10개 검사 영역
COLUMN_MAPPING_B = {
    "시야각검사A_위치정오": "B1-1",
    "시야각검사A_반응시간": "B1-2",
    "시야각검사A_반응정확": "B1-3",
    "시야각검사B_위치정오": "B2-1",
    "시야각검사B_반응시간": "B2-2",
    "시야각검사B_반응정확": "B2-3",
    "신호등_정오": "B3-1",
    "신호등_반응시간": "B3-2",
    "화살표_정오": "B4-1",
    "화살표_시간": "B4-2",
    "도로찾기_정오": "B5-1",
    "도로찾기_시간": "B5-2",
    "표지판검사A_정오": "B6",
    "표지판검사B_정오": "B7",
    "추적검사_정오": "B8",
    "복합기능검사A_청각_HIT": "B9-1",
    "복합기능검사A_청각_MISS": "B9-2",
    "복합기능검사A_청각_FA": "B9-3",
    "복합기능검사A_청각_CR": "B9-4",
    "복합기능검사A_청각_충돌횟수": "B9-5",
    "복합기능검사B_청각_HIT": "B10-1",
    "복합기능검사B_청각_MISS": "B10-2",
    "복합기능검사B_청각_FA": "B10-3",
    "복합기능검사B_청각_CR": "B10-4",
    "복합기능검사B_청각_충돌횟수": "B10-5",
    "복합기능검사B_시각정답총합": "B10-6",
}


def resolve_pk(gov_hash_val) -> str:
    """주민번호_hash 값에서 PrimaryKey를 추출한다.

    '0x'/'0X' 접두어를 제거하고 소문자 hex 문자열로 변환한다.
    PrimaryKey 생성의 유일한 원본(Single Source of Truth).
    """
    if pd.isna(gov_hash_val):
        return ""
    h = str(gov_hash_val).strip()
    if h.startswith("0x") or h.startswith("0X"):
        h = h[2:]
    return h.lower()


def mask_name(name: str) -> str:
    """이름 마스킹 규칙:
    - 2글자: 첫 글자 + * (김구 -> 김*)
    - 3글자 이상: 앞 2글자 + * (김안전 -> 김안*)
    """
    s = str(name).strip()
    if len(s) < 2:
        return s + "*"

    if len(s) == 2:
        return s[0] + "*"
    else:
        return s[:2] + "*"


def mask_rrn(rrn: str) -> str:
    """주민번호 마스킹 규칙: 99****-1******
    앞 2자리 노출, 하이픈 뒤 1자리 노출, 나머지 마스킹.
    """
    if not rrn:
        return ""

    s = str(rrn).strip().replace(" ", "")

    if "-" in s:
        parts = s.split("-")
        front = parts[0]
        back = parts[1] if len(parts) > 1 else ""
    else:
        front = s[:6]
        back = s[6:]

    if len(front) >= 2:
        m_front = front[:2] + "*" * (len(front) - 2)
    else:
        m_front = front + "*"

    if len(back) >= 1:
        m_back = back[0] + "*" * (len(back) - 1)
    else:
        m_back = "*"

    return f"{m_front}-{m_back}"


def transform_excel_val(val):
    if pd.isna(val):
        return ""
    return str(val).strip()


def _validate_columns(filename: str, columns, keys_a_stripped, keys_b_stripped):
    """단일 시트의 컬럼을 검증한다. (에러메시지, None) 또는 (None, 검출 유형)을 반환."""
    df_cols_stripped = {
        str(c).replace(" ", "").replace("\n", "").strip(): c
        for c in columns
    }
    columns_stripped = set(df_cols_stripped.keys())

    required_common_stripped = [
        "이름", "주민번호", "주민번호_hash", "업종상세", "업종", "만나이", "지사명", "수검일",
    ]

    missing_common = [req for req in required_common_stripped if req not in df_cols_stripped]
    if missing_common:
        return f"[{filename}] 필수 공통 컬럼이 누락되었습니다: {', '.join(missing_common)}", None

    missing_a_set = set(keys_a_stripped.keys()) - columns_stripped
    missing_b_set = set(keys_b_stripped.keys()) - columns_stripped

    if len(missing_a_set) == 0:
        return None, "A"
    elif len(missing_b_set) == 0:
        return None, "B"
    else:
        if len(missing_a_set) < len(missing_b_set):
            missing_orig = [keys_a_stripped[k] for k in missing_a_set]
            error_msg = f"[{filename}] 신규 검사(A유형) 양식과 일치하지 않습니다. 누락된 항목: {', '.join(missing_orig[:5])}"
        else:
            missing_orig = [keys_b_stripped[k] for k in missing_b_set]
            error_msg = f"[{filename}] 자격유지 검사(B유형) 양식과 일치하지 않습니다. 누락된 항목: {', '.join(missing_orig[:5])}"
        if len(missing_orig) > 5:
            error_msg += "..."
        return error_msg, None


def _read_excel_calamine(path_or_buf, **kwargs):
    """calamine 엔진으로 Excel 읽기 (3~10배 빠름), 실패 시 openpyxl 폴백."""
    try:
        return pd.read_excel(path_or_buf, engine="calamine", **kwargs)
    except Exception:
        if isinstance(path_or_buf, str):
            return pd.read_excel(path_or_buf, **kwargs)
        path_or_buf.seek(0)
        return pd.read_excel(path_or_buf, **kwargs)


def _process_sheet(filename: str, sheet_name: str, df: pd.DataFrame,
                   test_type: str, mapping: dict,
                   keys_a_stripped: dict, keys_b_stripped: dict):
    """단일 시트를 처리하여 (features_df, meta_df)를 반환한다.

    DriverInput 객체 생성 없이 DataFrame을 직접 구성한다.
    """
    # 컬럼명 정규화 (공백/줄바꿈 제거)
    df_cols_stripped = {
        str(c).replace(" ", "").replace("\n", "").strip(): c
        for c in df.columns
    }

    # 공통 컬럼 매핑
    required_common_stripped = [
        "이름", "주민번호", "주민번호_hash", "업종상세", "업종", "만나이", "지사명", "수검일",
    ]
    common_col_map = {}
    for req in required_common_stripped:
        if req in df_cols_stripped:
            common_col_map[req] = df_cols_stripped[req]

    n_rows = len(df)
    logger.info(f"Processing File: {filename}, Sheet: {sheet_name}, Type: {test_type}, Rows: {n_rows}")

    # ── 공통 컬럼 벡터화 처리 ──
    def _safe_col(key):
        col = common_col_map.get(key)
        if col is None:
            return pd.Series("", index=df.index)
        return df[col].fillna("").astype(str).str.strip()

    names_s = _safe_col("이름")
    rrns_s = _safe_col("주민번호")
    industry_s = _safe_col("업종")
    industry_detail_s = _safe_col("업종상세")
    branch_s = _safe_col("지사명")
    test_date_raw = _safe_col("수검일")

    # 수검일 / RRN 파싱 + age 계산: utils/age_calc 공통 유틸 사용
    test_date_s = normalize_test_date(test_date_raw)
    dob_years, dob_mm, dob_dd, genders = parse_rrn_birth(rrns_s)
    dob_mmdd = (dob_mm.fillna(1) * 100 + dob_dd.fillna(1)).astype(int)

    exam_ages = compute_exam_age(dob_years, dob_mmdd, test_date_s)
    current_ages = compute_current_age(dob_years, dob_mmdd)
    # 출생연도 미상(0)인 경우 exam_age로 폴백
    current_ages = current_ages.where(current_ages != 0, exam_ages)

    birth_yyyymmdd_s = build_birth_yyyymmdd(dob_years, dob_mm, dob_dd)

    # PrimaryKey: 벡터화
    gov_hash_col = df[df_cols_stripped["주민번호_hash"]].fillna("").astype(str).str.strip()
    unique_ids = gov_hash_col.str.replace(r"^0[xX]", "", regex=True).str.lower()

    # 마스킹 벡터화
    masked_names = names_s.map(mask_name)
    masked_rrns = rrns_s.map(mask_rrn)

    # 생년월일 마스킹 (YYYY-MM-** / YYYY-MM-DD)
    mm_str = dob_mm.fillna(0).astype("Int64").astype(str).str.zfill(2)
    dd_str = dob_dd.fillna(0).astype("Int64").astype(str).str.zfill(2)
    yy_str = dob_years.fillna(0).astype("Int64").astype(str).str.zfill(4)
    has_mm = dob_years.notna() & dob_mm.notna()
    has_dd = has_mm & dob_dd.notna()
    masked_dobs = pd.Series("", index=df.index)
    orig_dobs = pd.Series("", index=df.index)
    masked_dobs[has_mm] = yy_str[has_mm] + "-" + mm_str[has_mm] + "-**"
    orig_dobs[has_dd] = yy_str[has_dd] + "-" + mm_str[has_dd] + "-" + dd_str[has_dd]

    # Age 코드: "30a"/"30b" 형식 (계산된 exam_age 기반)
    ea_codes = age_to_code(exam_ages)

    # Test_id 생성 (중복 시 _2, _3 접미사 — transform.py와 동일 방식)
    test_type_str = str(test_type)
    base_ids = unique_ids + "_" + test_type_str + "_" + test_date_s
    dup_num = base_ids.groupby(base_ids).cumcount() + 1
    test_ids = base_ids.where(dup_num == 1, base_ids + "_" + dup_num.astype(str))

    # 유효한 행만 필터 (PrimaryKey가 있는 행)
    valid_mask = (unique_ids != "") & (unique_ids != "nan")
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning(f"[{filename}|{sheet_name}] {n_invalid}행 건너뜀: 주민번호_hash 누락")

    # ── 피처 DataFrame 구성 ──
    feature_cols = {col_name: model_code for col_name, model_code in mapping.items()}
    feat_df = df[list(feature_cols.keys())].rename(columns=feature_cols)
    feat_df = feat_df.assign(
        Test_id=test_ids,
        TestDate=test_date_s,
        Age=ea_codes,
        PrimaryKey=unique_ids,
    )
    feat_df = feat_df[valid_mask].reset_index(drop=True)

    # ── 메타 DataFrame 구성 (응답 매핑용) ──
    meta_df = pd.DataFrame({
        "Test_id": test_ids,
        "PrimaryKey": unique_ids,
        "domain": test_type_str,
        "masked_name": masked_names,
        "masked_rrn": masked_rrns,
        "original_name": names_s,
        "original_rrn": rrns_s,
        "gender": genders,
        "industry": industry_s,
        "industry_detail": industry_detail_s,
        "branch": branch_s,
        "exam_age": exam_ages.astype(str),
        "current_age": current_ages.astype(str),
        "TestDate": test_date_s,
        "Age": ea_codes,
        "masked_dob": masked_dobs,
        "original_dob": orig_dobs,
        "birth_yyyymmdd": birth_yyyymmdd_s,
    })
    meta_df = meta_df[valid_mask].reset_index(drop=True)

    return feat_df, meta_df


@router.post("/predict/upload", response_model=List[PredictionResponse])
async def predict_from_upload(files: List[UploadFile] = File(...)):
    """다중 파일 업로드 -> 유형 자동 감지 -> 파싱 -> 예측

    최적화:
    - 임시 파일로 스트리밍 (메모리 절약)
    - calamine 엔진으로 1회 읽기 (3-10배 빠름)
    - DriverInput 생성 없이 DataFrame 직접 전달
    """
    try:
        keys_a_stripped = {k.replace(" ", ""): k for k in COLUMN_MAPPING_A.keys()}
        keys_b_stripped = {k.replace(" ", ""): k for k in COLUMN_MAPPING_B.keys()}

        all_errors = []
        # (filename, sheet_name, df, test_type, mapping) 튜플 리스트
        validated_sheets = []
        temp_paths = []

        from src.core.constants import MAX_UPLOAD_FILE_SIZE, UPLOAD_CHUNK_BYTES
        MAX_FILE_SIZE = MAX_UPLOAD_FILE_SIZE

        # ── 1단계: 파일 스트리밍 + 단일 읽기 + 검증 ──
        for file in files:
            # 임시 파일로 스트리밍 (대용량 파일 메모리 절약)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx")
            temp_paths.append(tmp_path)
            try:
                file_size = 0
                with os.fdopen(tmp_fd, "wb") as tmp_f:
                    while True:
                        chunk = await file.read(UPLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        file_size += len(chunk)
                        if file_size > MAX_FILE_SIZE:
                            raise HTTPException(
                                status_code=413,
                                detail=f"파일 '{file.filename}'이 500MB를 초과합니다.",
                            )
                        tmp_f.write(chunk)

                # calamine 엔진으로 1회만 읽기 (헤더+데이터 동시)
                xls = _read_excel_calamine(
                    tmp_path, sheet_name=None, header=0, dtype=str
                )
            except Exception as e:
                logger.error(f"[{file.filename}] Excel 읽기 실패: {e}")
                all_errors.append(f"[{file.filename}] 파일을 읽을 수 없습니다: {e}")
                continue

            # 읽은 데이터로 검증 + 수집
            for sheet_name, df in xls.items():
                error_msg, detected_type = _validate_columns(
                    file.filename, df.columns, keys_a_stripped, keys_b_stripped
                )
                if error_msg:
                    logger.error(f"[{file.filename}|{sheet_name}] {error_msg}")
                    all_errors.append(error_msg)
                else:
                    # 매핑 구성
                    df_cols_stripped = {
                        str(c).replace(" ", "").replace("\n", "").strip(): c
                        for c in df.columns
                    }
                    if detected_type == "A":
                        mapping = {df_cols_stripped[k_strip]: COLUMN_MAPPING_A[orig_k]
                                   for k_strip, orig_k in keys_a_stripped.items()}
                    else:
                        mapping = {df_cols_stripped[k_strip]: COLUMN_MAPPING_B[orig_k]
                                   for k_strip, orig_k in keys_b_stripped.items()}
                    validated_sheets.append(
                        (file.filename, sheet_name, df, detected_type, mapping)
                    )

        # 임시 파일 즉시 삭제
        for tmp_path in temp_paths:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if all_errors:
            raise HTTPException(status_code=400, detail="\n".join(all_errors))

        if not validated_sheets:
            raise HTTPException(status_code=400, detail="유효한 데이터가 없습니다.")

        # ── 2단계: 시트 처리 → 도메인별 DataFrame 수집 ──
        domain_feats: dict[str, list] = {"A": [], "B": []}
        domain_metas: dict[str, list] = {"A": [], "B": []}

        for filename, sheet_name, df, test_type, mapping in validated_sheets:
            feat_df, meta_df = _process_sheet(
                filename, sheet_name, df, test_type, mapping,
                keys_a_stripped, keys_b_stripped,
            )
            if len(feat_df) > 0:
                domain_feats[test_type].append(feat_df)
                domain_metas[test_type].append(meta_df)

        # ── 3단계: 도메인별 예측 + 응답 구성 ──
        results = []

        for domain in ["A", "B"]:
            if not domain_feats[domain]:
                continue

            combined_feat = pd.concat(domain_feats[domain], ignore_index=True)
            combined_meta = pd.concat(domain_metas[domain], ignore_index=True)

            # predict_domain_df로 직접 예측 (DriverInput 변환 없음)
            pred_df = predict_domain_df(domain, combined_feat)

            # 메타 데이터 병합
            response_df = pred_df.merge(combined_meta, on="Test_id", how="left")

            # 피처 딕셔너리 준비 (응답에 포함)
            feat_only_cols = [
                c for c in combined_feat.columns
                if c not in ("Test_id", "TestDate", "Age", "PrimaryKey")
            ]
            feat_records = (
                combined_feat.drop_duplicates(subset=["Test_id"], keep="last")
                .set_index("Test_id")[feat_only_cols]
                .to_dict("index")
            )

            # 응답 딕셔너리 구성 (to_dict + 벡터화)
            response_df["domain"] = domain
            response_df["score"] = response_df["final_score"]
            response_df["result"] = response_df["final_score"]
            resp_cols = [
                "Test_id", "score", "result", "riskGroup", "domain",
                "PrimaryKey", "masked_name", "masked_rrn",
                "original_name", "original_rrn", "gender",
                "industry", "industry_detail", "branch",
                "exam_age", "current_age", "masked_dob", "original_dob",
                "birth_yyyymmdd",
                "TestDate", "Age",
            ]
            present_cols = [c for c in resp_cols if c in response_df.columns]
            resp_records = response_df[present_cols].to_dict("records")
            for r in resp_records:
                r["score"] = float(r["score"])
                r["result"] = float(r["result"])
                tid = r["Test_id"]
                if tid in feat_records:
                    r["features"] = feat_records[tid]
                results.append(r)

        if not results:
            raise HTTPException(
                status_code=400, detail="유효한 데이터를 처리할 수 없습니다."
            )

        # 분석 캐시에 features 보관 (비교 분석 SHAP에서 test_ids로 조회)
        from src.core.analysis_cache import replace_all as cache_replace
        cache_items = []
        for r in results:
            cache_items.append({
                "Test_id": r.get("Test_id", ""),
                "PrimaryKey": r.get("PrimaryKey", ""),
                "Age": r.get("Age", "0"),
                "TestDate": r.get("TestDate", ""),
                "domain": r.get("domain", ""),
                "features": r.get("features", {}),
            })
        cache_replace(cache_items)

        # 개인별 SHAP 사전계산: 새 업로드이므로 이전 SHAP 캐시를 비우고
        # 백그라운드 스레드에서 미리 계산해 둔다(다운로드 시 "조회만" 하도록).
        # 응답을 막지 않으며, 계산 전 다운로드가 오면 endpoint가 on-demand로 계산한다.
        try:
            from src.core import shap_cache
            from src.services.shap_service import compute_for_records
            shap_cache.clear()

            # 다운로드 보고서는 사람(PrimaryKey)별 '최신 검사' 1건만 사용하므로
            # 사전계산도 그 1건으로 제한한다(중복/과거 검사 제외 → 메모리·시간 절감).
            # 누락분이 있어도 다운로드 시 on-demand로 계산되므로 안전하다.
            latest_by_pk = {}
            for it in cache_items:
                pk = it.get("PrimaryKey")
                if not pk:
                    continue
                cur = latest_by_pk.get(pk)
                if cur is None or str(it.get("TestDate", "")) > str(cur.get("TestDate", "")):
                    latest_by_pk[pk] = it
            warm_items = list(latest_by_pk.values())

            async def _warm_shap(items):
                try:
                    n = await asyncio.to_thread(compute_for_records, items)
                    logger.info("[Upload] SHAP 사전계산 완료: %d명", n)
                except Exception as e:
                    logger.warning("[Upload] SHAP 사전계산 실패(무시, on-demand로 대체): %s", e)

            asyncio.create_task(_warm_shap(warm_items))
        except Exception as e:
            logger.warning("[Upload] SHAP 사전계산 스케줄 실패(무시): %s", e)

        return results

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Internal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="서버 내부 오류가 발생했습니다.")
