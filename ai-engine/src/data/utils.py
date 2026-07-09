from typing import Optional
import numpy as np
import pandas as pd


def seq_matrix(
    series: pd.Series,
    length: Optional[int] = None,
    dtype=np.float32,
    cache: bool = True,
    delimiter: str = ",",
) -> np.ndarray:
    cache_key = f"__seqmat__::{series.name}::{str(length)}::{str(dtype)}::{delimiter}"
    if cache and hasattr(series, "attrs") and cache_key in series.attrs:
        return series.attrs[cache_key]

    try:
        import pyarrow as pa
        import pyarrow.compute as pc

        s_filled = series.fillna("")

        if delimiter == "":
            # 벡터화 문자 분할: Python per-row 루프 대비 5-10배 빠름
            # Python 폴백과 동일 동작: 공백/쉼표 제거 후 한 글자씩 분리
            s_str = s_filled.astype(str)
            s_clean = s_str.str.replace(r"[\s,]+", "", regex=True)
            str_lens = s_clean.str.len().fillna(0).astype(int)
            max_len = int(str_lens.max()) if len(str_lens) > 0 else 0
            L = length if length is not None else max_len

            N = len(series)
            if L == 0 or N == 0:
                out = np.full((N, max(L, 1)), np.nan, dtype=dtype)
            else:
                out = np.full((N, L), np.nan, dtype=dtype)
                for pos in range(L):
                    chars = s_clean.str[pos]
                    out[:, pos] = pd.to_numeric(chars, errors="coerce").astype(dtype)
        else:
            arr = pa.Array.from_pandas(s_filled)
            split = pc.split_pattern(arr, pattern=delimiter)

            pa_type = pa.float32() if np.dtype(dtype) == np.float32 else pa.float64()

            flat_vals = pc.cast(split.flatten(), pa_type).to_numpy()

            lengths = pc.list_value_length(split).to_numpy()
            max_len_data = lengths.max() if len(lengths) > 0 else 0
            L = length if length is not None else max_len_data

            if len(series) == 0:
                out = np.array([], dtype=dtype).reshape(0, L)
            elif len(lengths) > 0 and lengths.min() == lengths.max():
                width = lengths[0]
                reshaped = flat_vals.reshape(len(series), width)

                if width == L:
                    out = reshaped.astype(dtype)
                elif width < L:
                    out = np.full((len(series), L), np.nan, dtype=dtype)
                    out[:, :width] = reshaped
                else:
                    out = reshaped[:, :L].astype(dtype)
            else:
                offsets = split.offsets.to_numpy()
                out = np.full((len(series), L), np.nan, dtype=dtype)

                for i in range(len(series)):
                    l_i = lengths[i]
                    if l_i > 0:
                        start = offsets[i]
                        end = offsets[i + 1]
                        n = min(l_i, L)
                        out[i, :n] = flat_vals[start : start + n]

    except Exception:
        # Fallback (Python loop)
        vals = series.values
        if delimiter == "":
            # Split by character, ignoring whitespace because some Fused columns might come as "1 0 1"
            parsed = [
                (
                    np.array([c for c in str(x) if c.strip() and c != ","], dtype=dtype)
                    if isinstance(x, str) and len(str(x)) > 0
                    else np.array([], dtype=dtype)
                )
                for x in vals
            ]
        else:
            # Fix for Mixed Delimiters: If we expect space, but get comma, replace it.
            # This handles cases where Excel/CSV export puts "0.5, 0.6" into a single cell.
            if delimiter == " ":
                vals = [
                    str(x).replace(",", " ") if isinstance(x, str) else x for x in vals
                ]

            parsed = [
                (
                    np.fromstring(x, sep=delimiter, dtype=dtype)
                    if isinstance(x, str)
                    else np.asarray(x, dtype=dtype)
                )
                for x in vals
            ]

        L = length if length is not None else (max((len(a) for a in parsed), default=0))
        out = np.full((len(parsed), L), np.nan, dtype=dtype)
        for i, a in enumerate(parsed):
            n = min(L, len(a))
            if n > 0:
                out[i, :n] = a[:n]

    if cache:
        try:
            series.attrs[cache_key] = out
        except Exception:
            pass
    return out


def _mask_in(mat: np.ndarray, values) -> np.ndarray:
    if np.isscalar(values) or isinstance(values, (np.generic, str, bytes, int, float)):
        return mat == values
    try:
        return np.isin(mat, np.asarray(list(values)))
    except Exception:
        v = np.atleast_1d(values)
        m = mat == v[0]
        for val in v[1:]:
            m |= mat == val
        return m


def _apply_optional_miss(
    mask: Optional[np.ndarray], miss_mat: Optional[np.ndarray], miss_values
):
    if miss_mat is None or miss_values is None:
        return mask
    miss_mask = _mask_in(miss_mat, miss_values)
    return miss_mask if mask is None else (mask & miss_mask)


def _resolve_empty_value(empty_value):
    return np.nan if empty_value is None else empty_value


def _rowwise_mean(
    values: np.ndarray,
    mask: Optional[np.ndarray] = None,
    absolute: bool = False,
    empty_value=None,
):
    ev = _resolve_empty_value(empty_value)
    v = np.abs(values) if absolute else values
    valid = (~np.isnan(v)) if mask is None else (mask & ~np.isnan(v))
    sums = np.sum(np.where(valid, v, 0.0), axis=1)
    counts = np.sum(valid, axis=1)
    out = np.full(sums.shape, ev, dtype=np.float32)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out, counts


def _rowwise_std(
    values: np.ndarray,
    mask: Optional[np.ndarray] = None,
    absolute: bool = False,
    empty_value=None,
):
    ev = _resolve_empty_value(empty_value)
    v = np.abs(values) if absolute else values
    valid = (~np.isnan(v)) if mask is None else (mask & ~np.isnan(v))
    sums = np.sum(np.where(valid, v, 0.0), axis=1)
    sq_sums = np.sum(np.where(valid, v * v, 0.0), axis=1)
    counts = np.sum(valid, axis=1)
    mean_vec = np.zeros_like(sums, dtype=np.float32)
    np.divide(sums, counts, out=mean_vec, where=counts > 0)
    ex2_vec = np.zeros_like(sq_sums, dtype=np.float32)
    np.divide(sq_sums, counts, out=ex2_vec, where=counts > 0)
    var_vec = (ex2_vec - mean_vec * mean_vec).astype(np.float32)
    var_vec = np.maximum(var_vec, 0.0)
    std_vec = np.sqrt(var_vec).astype(np.float32)
    std_vec[counts == 0] = ev
    return std_vec, counts.astype(np.int32)


def _as_frame(
    name: str,
    mean: np.ndarray,
    counts: Optional[np.ndarray],
    return_flag: bool,
    return_count: bool,
) -> pd.DataFrame:
    cols = {name: mean}
    if return_flag:
        cols[f"{name}_missing"] = (counts == 0).astype(np.int8)
    if return_count:
        cols[f"{name}_cnt"] = counts.astype(np.int16)
    return pd.DataFrame(cols)


def _as_frame_std(
    name: str,
    std_vec: np.ndarray,
    counts: np.ndarray,
    return_flag: bool,
    return_count: bool,
) -> pd.DataFrame:
    cols = {f"{name}_std": std_vec}
    if return_flag:
        cols[f"{name}_missing"] = (counts == 0).astype(np.int8)
    if return_count:
        cols[f"{name}_cnt"] = counts.astype(np.int16)
    return pd.DataFrame(cols)


def seq_abs_mean(
    value_mat: np.ndarray,
    miss_mat: Optional[np.ndarray] = None,
    miss_values=None,
    empty_value=None,
    name: Optional[str] = None,
    return_flag: bool = False,
    return_count: bool = False,
):
    mask = _apply_optional_miss(None, miss_mat, miss_values)
    mean, cnt = _rowwise_mean(
        value_mat, mask=mask, absolute=True, empty_value=empty_value
    )
    if return_flag or return_count:
        if name is None:
            name = "seq_abs_mean"
        return _as_frame(name, mean, cnt, return_flag, return_count)
    return mean


def seq_cond_mean(
    cond_mat: np.ndarray,
    value_mat: np.ndarray,
    cond_values,
    absolute: bool = False,
    miss_mat: Optional[np.ndarray] = None,
    miss_values=None,
    empty_value=None,
    name: Optional[str] = None,
    return_flag: bool = False,
    return_count: bool = False,
):
    mask = _mask_in(cond_mat, cond_values)
    mask = _apply_optional_miss(mask, miss_mat, miss_values)
    mean, cnt = _rowwise_mean(
        value_mat, mask=mask, absolute=absolute, empty_value=empty_value
    )
    if return_flag or return_count:
        if name is None:
            name = "seq_cond_mean"
        return _as_frame(name, mean, cnt, return_flag, return_count)
    return mean


def seq_std(
    value_mat: np.ndarray,
    miss_mat: Optional[np.ndarray] = None,
    miss_values=None,
    absolute: bool = False,
    empty_value=None,
    name: Optional[str] = None,
    return_flag: bool = False,
    return_count: bool = False,
):
    mask = _apply_optional_miss(None, miss_mat, miss_values)
    std_vec, cnt_vec = _rowwise_std(
        value_mat, mask=mask, absolute=absolute, empty_value=empty_value
    )
    if return_flag or return_count:
        if name is None:
            name = "seq_std"
        return _as_frame_std(name, std_vec, cnt_vec, return_flag, return_count)
    return std_vec


def seq_cond_std(
    cond_mat: np.ndarray,
    value_mat: np.ndarray,
    cond_values,
    absolute: bool = False,
    miss_mat: Optional[np.ndarray] = None,
    miss_values=None,
    empty_value=None,
    name: Optional[str] = None,
    return_flag: bool = False,
    return_count: bool = False,
):
    mask = _mask_in(cond_mat, cond_values)
    mask = _apply_optional_miss(mask, miss_mat, miss_values)
    std_vec, cnt_vec = _rowwise_std(
        value_mat, mask=mask, absolute=absolute, empty_value=empty_value
    )
    if return_flag or return_count:
        if name is None:
            name = "seq_cond_std"
        return _as_frame_std(name, std_vec, cnt_vec, return_flag, return_count)
    return std_vec


def seq_count_equals(
    mat: np.ndarray, target, miss_mat: Optional[np.ndarray] = None, miss_values=None
) -> np.ndarray:
    mask = _mask_in(mat, target)
    mask = _apply_optional_miss(mask, miss_mat, miss_values)
    return np.sum(mask, axis=1)


