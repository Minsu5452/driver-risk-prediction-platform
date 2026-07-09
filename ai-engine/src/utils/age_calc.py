"""연령 계산 공통 유틸.

upload.py(predict 경로)와 transform.py(admin 학습 경로)가 동일 로직을 공유하도록
한 곳에서 정의한다. 이전에 두 파일이 따로 구현돼 있어 한쪽만 수정하다 버그가 반복됐음.

핵심 규칙:
- 한국 주민번호 = "YYMMDD-Gxxxxxx", 7번째 자리(=하이픈 뒤 첫자리)가 성별/세기
  · 1, 2, 5, 6 → 1900년대 출생
  · 3, 4, 7, 8 → 2000년대 출생
  · 9, 0       → 1800년대 출생 (희귀)
- 만나이 = 수검년도 - 출생년도, 단 (수검 MMDD < 출생 MMDD) 이면 -1 (생일 전)
- 정부 원본 Excel의 '만나이' 컬럼은 덤프 시점 현재값 단일 스냅샷이므로
  검사별 연령으로 신뢰할 수 없음 → 항상 RRN+수검일로 재계산할 것.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

KST = timezone(timedelta(hours=9))


# ── 1. RRN 파싱 ──────────────────────────────────────────────────────────────


def parse_rrn_birth(rrn_series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """주민번호 시리즈에서 (출생년도, 출생월, 출생일, 성별) 을 벡터화로 추출.

    Args:
        rrn_series: 주민번호 문자열 Series. "YYMMDD-Gxxxxxx" / "YYMMDDGxxxxxx" / 공백 등 허용.

    Returns:
        (dob_year, dob_mm, dob_dd, gender)
        - dob_year: Int64 Series (NaN 가능)
        - dob_mm:   Int64 Series (1~12, NaN 가능)
        - dob_dd:   Int64 Series (1~31, NaN 가능)
        - gender:   문자열 Series ("남"/"여"/"Unknown")
    """
    s = rrn_series.fillna("").astype(str).str.strip()
    clean = s.str.replace("-", "", regex=False).str.replace(" ", "", regex=False)
    rrn_lens = clean.str.len()

    y_prefix = pd.to_numeric(clean.str[:2], errors="coerce")
    gender_digit = pd.to_numeric(clean.str[6], errors="coerce")

    valid_g = gender_digit.isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    century = pd.Series(pd.NA, index=s.index, dtype="Int64")
    century[valid_g & gender_digit.isin([1, 2, 5, 6])] = 1900
    century[valid_g & gender_digit.isin([3, 4, 7, 8])] = 2000
    century[valid_g & gender_digit.isin([9, 0])] = 1800

    dob_year = (century + y_prefix.astype("Int64")).where(rrn_lens >= 7)

    dob_mm = pd.to_numeric(clean.str[2:4], errors="coerce").astype("Int64")
    dob_dd = pd.to_numeric(clean.str[4:6], errors="coerce").astype("Int64")
    # 유효 범위 외 값은 NA
    dob_mm = dob_mm.where((dob_mm >= 1) & (dob_mm <= 12))
    dob_dd = dob_dd.where((dob_dd >= 1) & (dob_dd <= 31))

    gender = pd.Series("Unknown", index=s.index)
    gender[(gender_digit % 2 != 0) & gender_digit.notna()] = "남"
    gender[(gender_digit % 2 == 0) & gender_digit.notna()] = "여"

    return dob_year, dob_mm, dob_dd, gender


# ── 2. 수검일 정규화 ─────────────────────────────────────────────────────────


def normalize_test_date(test_date_series: pd.Series) -> pd.Series:
    """수검일 시리즈를 8자리 YYYYMMDD 문자열로 정규화.

    "2023.03.15", "2023-03-15", "2023/03/15", "20230315.0" 등 다양한 입력 처리.
    """
    s = test_date_series.fillna("").astype(str).str.strip()
    return s.str.replace(r"[^\d]", "", regex=True).str[:8]


# ── 3. exam_age (수검 당시 만나이) ──────────────────────────────────────────


def compute_exam_age(
    dob_year: pd.Series,
    dob_mmdd: pd.Series,
    test_date_yyyymmdd: pd.Series,
) -> pd.Series:
    """수검 당시 만나이를 벡터화로 계산.

    Args:
        dob_year: 출생년도 Int64 Series.
        dob_mmdd: 출생 MMDD int Series (예: 0315).
        test_date_yyyymmdd: 정규화된 8자리 수검일 문자열 Series.

    Returns:
        int Series (음수는 0으로 클립).
    """
    test_years = pd.to_numeric(test_date_yyyymmdd.str[:4], errors="coerce")
    test_mm = pd.to_numeric(test_date_yyyymmdd.str[4:6], errors="coerce")
    test_dd = pd.to_numeric(test_date_yyyymmdd.str[6:8], errors="coerce")
    # 월/일 누락 시 연말로 폴백 (생일 지난 것으로 처리 → year_diff 그대로)
    test_mmdd = (test_mm.fillna(12) * 100 + test_dd.fillna(31)).astype(int)

    year_diff = (test_years - dob_year).fillna(0).astype(int)
    ages = year_diff - (test_mmdd < dob_mmdd).astype(int)
    return ages.clip(lower=0)


# ── 4. current_age (현재 만나이, 벡터) ──────────────────────────────────────


def compute_current_age(
    dob_year: pd.Series,
    dob_mmdd: pd.Series,
    now_kst: Optional[datetime] = None,
) -> pd.Series:
    """오늘 KST 기준 현재 만나이를 벡터화로 계산."""
    if now_kst is None:
        now_kst = datetime.now(KST)
    cur_year = now_kst.year
    cur_mmdd = int(now_kst.strftime("%m%d"))

    year_diff = (cur_year - dob_year).fillna(0).astype(int)
    ages = year_diff - (cur_mmdd < dob_mmdd).astype(int)
    return ages.clip(lower=0)


# ── 5. current_age (스칼라, DB 조회용) ─────────────────────────────────────


def compute_current_age_from_yyyymmdd(
    birth_yyyymmdd: str,
    now_kst: Optional[datetime] = None,
) -> Optional[int]:
    """birth_yyyymmdd 8자리 문자열에서 오늘 KST 기준 만나이를 계산.

    Args:
        birth_yyyymmdd: "19670315" 등 8자리 문자열.
        now_kst: 테스트용 시점 주입.

    Returns:
        만나이 int 또는 파싱 불가시 None.
    """
    if not birth_yyyymmdd or len(str(birth_yyyymmdd)) != 8:
        return None
    s = str(birth_yyyymmdd)
    if not s.isdigit():
        return None
    try:
        by, bm, bd = int(s[:4]), int(s[4:6]), int(s[6:8])
    except ValueError:
        return None
    if not (1800 <= by <= 2200 and 1 <= bm <= 12 and 1 <= bd <= 31):
        return None

    if now_kst is None:
        now_kst = datetime.now(KST)
    age = now_kst.year - by
    if (now_kst.month, now_kst.day) < (bm, bd):
        age -= 1
    return max(0, age)


# ── 6. age code 변환 ────────────────────────────────────────────────────────


def age_to_code(age_int_series: pd.Series) -> pd.Series:
    """정수 만나이 → 연령 코드 ("30a"=30~34, "30b"=35~39).

    age == 0 인 행은 빈 문자열 (출생연도 미상) — 학습 데이터 일관성 유지.
    """
    a = age_int_series.fillna(0).astype(int)
    decades = (a // 10) * 10
    suffixes = a.mod(10).lt(5).map({True: "a", False: "b"})
    code = decades.astype(str) + suffixes
    return code.where(a > 0, "")


# ── 7. birth_yyyymmdd 8자리 문자열 빌드 ──────────────────────────────────────


def build_birth_yyyymmdd(
    dob_year: pd.Series,
    dob_mm: pd.Series,
    dob_dd: pd.Series,
) -> pd.Series:
    """(year, mm, dd) Int64 Series → "YYYYMMDD" 문자열 Series. 결측은 빈 문자열."""
    valid = dob_year.notna() & dob_mm.notna() & dob_dd.notna()
    y = dob_year.fillna(0).astype("Int64").astype(str).str.zfill(4)
    m = dob_mm.fillna(0).astype("Int64").astype(str).str.zfill(2)
    d = dob_dd.fillna(0).astype("Int64").astype(str).str.zfill(2)
    return (y + m + d).where(valid, "")
