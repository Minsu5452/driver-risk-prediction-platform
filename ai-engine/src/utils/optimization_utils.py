import numpy as np
import pandas as pd


def _detect_delimiter(sample: str) -> str:
    """첫 번째 유효 값에서 구분자를 자동 감지한다."""
    if "," in sample:
        return ","
    if " " in sample:
        return " "
    # 숫자만 2자리 이상 → 퓨즈드 (문자 단위 분리)
    return ""


def str_to_num_array(series):
    """문자열 Series를 2D numpy 배열로 파싱한다.

    포맷 자동 감지: 쉼표 구분("1,2,3"), 공백 구분("0.5 0.6"), 퓨즈드("10110").
    """
    s_filled = series.fillna("")
    empty_mask = (s_filled == "").values
    valid_series = s_filled[~empty_mask]

    if len(valid_series) == 0:
        return np.full((len(series), 0), np.nan)

    # 포맷 자동 감지
    delimiter = _detect_delimiter(str(valid_series.iloc[0]))

    if delimiter == "":
        # 퓨즈드 숫자: "10110" → [1,0,1,1,0]
        return _parse_fused(series, valid_series, empty_mask)

    # 쉼표 또는 공백 구분
    try:
        import pyarrow as pa
        import pyarrow.compute as pc

        pa_arr = pa.Array.from_pandas(valid_series)
        split = pc.split_pattern(pa_arr, pattern=delimiter)

        # 행별 길이 검증 — 불균일하면 Python 폴백 (NaN 패딩)
        lengths = pc.list_value_length(split).to_pylist()
        if len(set(lengths)) > 1:
            return _str_to_num_array_python(series, delimiter)

        flat_nums = pc.cast(split.flatten(), pa.float64())
        valid_flat = flat_nums.to_numpy()

        n = len(valid_series)
        m = lengths[0] if lengths else 0
        valid_arr = valid_flat.reshape(n, m)
    except Exception:
        return _str_to_num_array_python(series, delimiter)

    if m == 0:
        return np.full((len(series), 0), np.nan)

    full_arr = np.full((len(series), m), np.nan)
    full_arr[~empty_mask] = valid_arr
    return full_arr


def _parse_fused(series, valid_series, empty_mask):
    """퓨즈드 숫자 문자열을 문자 단위로 분리한다: "10110" → [1,0,1,1,0]"""
    # 공백/쉼표 제거 (혼합 형식 대응)
    s_clean = valid_series.astype(str).str.replace(r"[\s,]+", "", regex=True)
    str_lens = s_clean.str.len().fillna(0).astype(int)
    max_len = int(str_lens.max()) if len(str_lens) > 0 else 0

    if max_len == 0:
        return np.full((len(series), 1), np.nan)

    valid_arr = np.full((len(valid_series), max_len), np.nan)
    for pos in range(max_len):
        chars = s_clean.str[pos]
        valid_arr[:, pos] = pd.to_numeric(chars, errors="coerce").values

    full_arr = np.full((len(series), max_len), np.nan)
    full_arr[~empty_mask] = valid_arr
    return full_arr


def _str_to_num_array_python(series, delimiter=","):
    """Python 폴백 파서 — 행별 길이가 달라도 NaN 패딩으로 처리."""
    s_filled = series.fillna("")
    empty_mask = s_filled == ""
    valid_vals = s_filled[~empty_mask].values
    if len(valid_vals) == 0:
        return np.array([[]] * len(series), dtype=float)

    parsed_list = [x.split(delimiter) if delimiter else list(x) for x in valid_vals]

    try:
        valid_arr = np.array(parsed_list, dtype=float)
    except ValueError:
        # 행별 길이 불균일 → NaN 패딩
        max_len = max(len(row) for row in parsed_list)
        valid_arr = np.full((len(valid_vals), max_len), np.nan)
        for i, row in enumerate(parsed_list):
            nums = pd.to_numeric(pd.Series(row), errors="coerce").values
            valid_arr[i, :len(nums)] = nums

    full_arr = np.full((len(series), valid_arr.shape[1]), np.nan)
    full_arr[~empty_mask] = valid_arr
    return full_arr


def fast_seq_mean(arr):
    """2D 배열의 행별 평균 계산 (NaN 무시)."""
    return np.nanmean(arr, axis=1)

def fast_seq_std(arr):
    """2D 배열의 행별 표준편차 계산 (NaN 무시, ddof=0)."""
    return np.nanstd(arr, axis=1, ddof=0)

def fast_seq_rate(series, target="1"):
    """시퀀스 내 특정 값의 비율을 계산한다."""
    arr = str_to_num_array(series)
    target_val = float(target)

    matches = np.sum(arr == target_val, axis=1)

    totals = np.sum(~np.isnan(arr), axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = matches / totals

    return rate

def fast_masked_mean(cond_arr, val_arr, mask_val):
    """조건 배열에서 mask_val과 일치하는 위치의 값 배열 평균을 계산한다."""
    # 열 수가 다를 경우 짧은 쪽에 맞춤 (A1-1: 18열, A1-4: 76열 등)
    min_cols = min(cond_arr.shape[1], val_arr.shape[1])
    cond_arr = cond_arr[:, :min_cols]
    val_arr = val_arr[:, :min_cols]

    mask = cond_arr == mask_val

    masked_vals = np.where(mask, val_arr, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = np.nansum(masked_vals, axis=1)
        row_counts = np.sum(
            ~np.isnan(masked_vals), axis=1
        )

        row_counts = np.sum(mask, axis=1)
        out = row_sums / np.where(row_counts == 0, np.nan, row_counts)

    return out

def fast_masked_mean_in_set(cond_arr, val_arr, mask_set):
    """조건 배열에서 mask_set에 포함된 값 위치의 값 배열 평균을 계산한다."""
    min_cols = min(cond_arr.shape[1], val_arr.shape[1])
    cond_arr = cond_arr[:, :min_cols]
    val_arr = val_arr[:, :min_cols]

    mask_list = [float(x) for x in mask_set]
    mask = np.isin(cond_arr, mask_list)

    with np.errstate(divide="ignore", invalid="ignore"):
        sums = np.nansum(np.where(mask, val_arr, np.nan), axis=1)
        counts = np.sum(mask, axis=1)
        out = sums / np.where(counts == 0, np.nan, counts)
    return out
