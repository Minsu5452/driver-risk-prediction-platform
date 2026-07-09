"""
데이터 변환 모듈: 정부 real_data 한글 형식을
AI 엔진 학습 모델이 사용하는 학습 스키마(train format)으로 변환한다.
"""

import pandas as pd

from src.api.upload import (
    COLUMN_MAPPING_A,
    COLUMN_MAPPING_B,
    resolve_pk,
)
from src.utils.age_calc import (
    age_to_code,
    build_birth_yyyymmdd,
    compute_exam_age,
    normalize_test_date,
    parse_rrn_birth,
)


# 역방향 매핑: 스키마 코드 -> 한글 컬럼명
_REVERSE_A = {v: k for k, v in COLUMN_MAPPING_A.items()}
_REVERSE_B = {v: k for k, v in COLUMN_MAPPING_B.items()}


def _normalize_a_sequence(val) -> str:
    """A 도메인 시퀀스 데이터를 쉼표 구분 형식으로 정규화한다."""
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # 쉼표가 있으면 쉼표로 분리
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        return ",".join(parts)
    # 공백이 있으면 공백으로 분리
    if " " in s:
        parts = s.split()
        return ",".join(parts)
    # 모두 숫자이고 2자리 이상이면 한 글자씩 분리
    if s.isdigit() and len(s) > 1:
        return ",".join(list(s))
    # 단일 값
    return s


def _normalize_a_sequence_vectorized(series: pd.Series) -> pd.Series:
    """A 도메인 시퀀스 데이터를 벡터화로 정규화한다.

    _normalize_a_sequence의 벡터화 버전 — .apply() 대신 pandas str 연산 사용.
    675K행 × 24컬럼 기준 ~20배 빠름 (1620만 Python 함수 호출 제거).
    """
    s = series.fillna("").astype(str).str.strip()
    result = s.copy()
    non_empty = s != ""

    # 쉼표 있음 → 공백 제거 + 빈 파트 제거 + 중복 쉼표 정리
    has_comma = non_empty & s.str.contains(",", na=False)
    if has_comma.any():
        cleaned = s[has_comma].str.replace(r"\s*,\s*", ",", regex=True)
        cleaned = cleaned.str.replace(r",+", ",", regex=True).str.strip(",")
        result[has_comma] = cleaned

    # 공백 있음 (쉼표 없음) → 공백을 쉼표로
    remaining = non_empty & (~has_comma)
    has_space = remaining & s.str.contains(" ", na=False)
    if has_space.any():
        result[has_space] = s[has_space].str.replace(r"\s+", ",", regex=True)

    # 2자리 이상 숫자만 → 한 글자씩 쉼표 분리
    remaining2 = remaining & (~has_space)
    is_multidigit = remaining2 & s.str.match(r"^\d{2,}$", na=False)
    if is_multidigit.any():
        result[is_multidigit] = s[is_multidigit].str.replace(
            r"(\d)(?=\d)", r"\1,", regex=True
        )

    return result


def _resolve_pk(row) -> str:
    """주민번호_hash 컬럼에서 PrimaryKey를 추출한다."""
    return resolve_pk(row.get("주민번호_hash"))


def transform_exam_to_train_format(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """정부 검사 Excel 데이터를 학습 스키마 CSV 형식으로 변환한다.

    수검 당시 만나이는 RRN(주민번호 앞자리) + 수검일에서 재계산한다.
    Excel의 '만나이' 컬럼은 덤프 시점 현재값 단일 스냅샷이라 검사별 연령으로
    신뢰할 수 없음 — upload.py와 동일 로직을 utils/age_calc로 공유.

    Args:
        df: 정부 검사 Excel의 한글 컬럼명 DataFrame.
        domain: "A" 또는 "B".

    Returns:
        PrimaryKey, Age, exam_age, birth_yyyymmdd, TestDate, Test, Test_id 및
        도메인별 피처 컬럼을 포함하는 학습 스키마 DataFrame.
    """
    mapping = COLUMN_MAPPING_A if domain == "A" else COLUMN_MAPPING_B
    feature_codes = list(mapping.values())

    result = pd.DataFrame()

    # PrimaryKey: 벡터화
    pk_raw = df["주민번호_hash"].fillna("").astype(str).str.strip()
    pk_raw = pk_raw.str.replace(r"^0[xX]", "", regex=True).str.lower()
    result["PrimaryKey"] = pk_raw

    # TestDate: 정규화 (점/하이픈/공백 제거 → 8자리 YYYYMMDD)
    td_raw = df.get("수검일", pd.Series("", index=df.index))
    test_date_s = normalize_test_date(td_raw)
    result["TestDate"] = test_date_s

    # 수검 당시 만나이: RRN + 수검일 재계산 (Excel '만나이' 무시)
    rrn_col = df.get("주민번호", pd.Series("", index=df.index))
    dob_year, dob_mm, dob_dd, _ = parse_rrn_birth(rrn_col)
    dob_mmdd = (dob_mm.fillna(1) * 100 + dob_dd.fillna(1)).astype(int)

    exam_ages = compute_exam_age(dob_year, dob_mmdd, test_date_s)
    result["Age"] = age_to_code(exam_ages)
    result["exam_age"] = exam_ages.astype("Int64")
    result["birth_yyyymmdd"] = build_birth_yyyymmdd(dob_year, dob_mm, dob_dd)

    result["Test"] = domain

    # Test_id 생성: 벡터화
    base_id = result["PrimaryKey"] + "_" + domain + "_" + result["TestDate"]
    dup_num = base_id.groupby(base_id).cumcount() + 1
    result["Test_id"] = base_id.where(dup_num == 1, base_id + "_" + dup_num.astype(str))

    # 한글 컬럼을 스키마 코드로 매핑 — 원본 값 그대로 저장 (정규화 불필요)
    # downstream 파서(seq_matrix, str_to_num_array)가 원본 형식을 직접 처리
    for kor_col, code in mapping.items():
        if kor_col not in df.columns:
            result[code] = ""
        else:
            result[code] = df[kor_col].fillna("").astype(str).str.strip()

    # 컬럼 순서 보장
    col_order = [
        "PrimaryKey", "Age", "exam_age", "birth_yyyymmdd",
        "TestDate", "Test", "Test_id",
    ] + feature_codes
    for c in col_order:
        if c not in result.columns:
            result[c] = ""
    result = result[col_order]

    return result


def _safe_int_parse(val) -> int:
    """값을 안전하게 정수로 변환한다. 실패 시 0 반환."""
    if pd.isna(val):
        return 0
    s = str(val).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return 0


def _map_exam_name(val, fallback: str) -> str:
    """검사명에서 검사 유형(A/B)을 판별한다."""
    s = str(val).strip() if pd.notna(val) else ""
    if s in ("신규검사", "신규"):
        return "A"
    if s in ("자격유지검사", "자격유지"):
        return "B"
    return fallback


def _map_acc_type(val) -> str:
    """사고(위반)구분에서 AccType을 결정한다."""
    s = str(val).strip() if pd.notna(val) else ""
    if s == "사고":
        return "A"
    if s == "위반":
        return "B"
    return ""


def transform_sago_to_train_format(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """사고/위반 데이터를 학습 스키마으로 변환한다.

    Args:
        df: 사고/위반 Excel의 한글 컬럼명 DataFrame.
        domain: "A" 또는 "B" (폴백용; 주요 감지는 검사명 컬럼으로).

    Returns:
        PrimaryKey, Test, AccType, AccDate, Count_1..Count_6 DataFrame.
    """
    # 카운트 컬럼명 변형 (real_data는 다른 이름 사용)
    count_col_map = {
        "Count_1": ["사망자수(30일이내사망자포함)", "사망자수", "사망"],
        "Count_2": ["중상자수", "중상"],
        "Count_3": ["경상자수", "경상"],
        "Count_4": ["부상자수", "부상신고자수", "부상신고"],
        "Count_5": ["벌점", "점수"],
    }

    # DataFrame에서 실제 컬럼명 탐색
    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    count_actual = {}
    for count_key, candidates in count_col_map.items():
        count_actual[count_key] = find_col(candidates)

    result = pd.DataFrame()

    # PrimaryKey: 벡터화
    pk_raw = df["주민번호_hash"].fillna("").astype(str).str.strip()
    pk_raw = pk_raw.str.replace(r"^0[xX]", "", regex=True).str.lower()
    result["PrimaryKey"] = pk_raw

    # Test: 벡터화 (컬럼 누락 시 빈 시리즈는 df.index에 정렬해 boolean 인덱싱 오류 방지)
    exam_s = df.get("검사명", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    test_col = pd.Series(domain, index=df.index)
    test_col[exam_s.isin(["신규검사", "신규"])] = "A"
    test_col[exam_s.isin(["자격유지검사", "자격유지"])] = "B"
    result["Test"] = test_col

    # AccType: 벡터화
    acc_s = df.get("사고(위반)구분", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
    acc_type = pd.Series("", index=df.index)
    acc_type[acc_s == "사고"] = "A"
    acc_type[acc_s == "위반"] = "B"
    result["AccType"] = acc_type

    # AccDate: 검사일(TestDate)과 동일하게 normalize_test_date로 8자리 정규화한다.
    # 기존 split(".")[0] 방식은 "2023.01.01" 같은 점 구분 날짜를 "2023"으로 잘라버리는
    # 잠재 버그가 있었다. 현재 데이터("20150115")는 출력 동일, 점/하이픈/".0" 케이스만 교정.
    result["AccDate"] = normalize_test_date(df.get("사고(위반)일자", pd.Series("", index=df.index)))

    # Count 컬럼: 벡터화
    for count_key, actual_col in count_actual.items():
        if actual_col is not None:
            result[count_key] = pd.to_numeric(df[actual_col], errors="coerce").fillna(0).astype(int)
        else:
            result[count_key] = 0

    result["Count_6"] = 0

    col_order = ["PrimaryKey", "Test", "AccType", "AccDate",
                 "Count_1", "Count_2", "Count_3", "Count_4", "Count_5", "Count_6"]
    for c in col_order:
        if c not in result.columns:
            result[c] = 0
    result = result[col_order]

    return result


def detect_file_type(file_path: str) -> str:
    """Excel 컬럼 헤더로부터 파일 유형을 자동 감지한다.

    반환값: 'a_exam', 'b_exam', 'a_sago', 'b_sago' 중 하나.

    Raises:
        ValueError: 파일 유형을 판별할 수 없는 경우.
    """
    try:
        df = pd.read_excel(file_path, nrows=50, dtype=str, engine="calamine")
    except Exception:
        df = pd.read_excel(file_path, nrows=50, dtype=str)
    cols = set(df.columns)

    # A 도메인 검사 컬럼 확인
    a_exam_keys = set(COLUMN_MAPPING_A.keys())
    # B 도메인 검사 컬럼 확인
    b_exam_keys = set(COLUMN_MAPPING_B.keys())

    if a_exam_keys.issubset(cols):
        return "a_exam"
    if b_exam_keys.issubset(cols):
        return "b_exam"

    # 사고/위반 데이터 확인
    if "사고(위반)구분" in cols:
        # 검사명 다수결로 A/B 결정
        if "검사명" in cols:
            exam_names = df["검사명"].dropna().astype(str).str.strip()
            new_count = exam_names.isin(["신규검사", "신규"]).sum()
            maint_count = exam_names.isin(["자격유지검사", "자격유지"]).sum()
            if new_count >= maint_count:
                return "a_sago"
            else:
                return "b_sago"
        return "a_sago"

    # 가장 가까운 유형 추정 후 누락 컬럼 안내
    missing_a = sorted(a_exam_keys - cols)
    missing_b = sorted(b_exam_keys - cols)
    sago_required = {"이름", "주민번호", "사고(위반)구분", "사고(위반)일자", "검사명"}
    missing_sago = sorted(sago_required - cols)

    hints = []
    if len(missing_a) <= len(missing_b):
        hints.append(
            f"신규검사(A) 양식 기준 누락 컬럼 ({len(missing_a)}개): "
            f"{', '.join(missing_a[:8])}"
            f"{'...' if len(missing_a) > 8 else ''}"
        )
    else:
        hints.append(
            f"자격유지검사(B) 양식 기준 누락 컬럼 ({len(missing_b)}개): "
            f"{', '.join(missing_b[:8])}"
            f"{'...' if len(missing_b) > 8 else ''}"
        )
    if missing_sago:
        hints.append(
            f"사고/위반 양식 기준 누락 컬럼 ({len(missing_sago)}개): "
            f"{', '.join(missing_sago)}"
        )

    raise ValueError(
        f"파일의 컬럼이 어떤 양식과도 일치하지 않습니다.\n"
        + "\n".join(hints)
    )
