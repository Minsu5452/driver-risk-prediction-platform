import logging
import numpy as np
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)


def _stitch_by_key_strict_past(
    q: pd.DataFrame,
    s: pd.DataFrame,
    key_col: str,
    left_key: str = "mi_left",
    right_key: str = "mi_snap",
) -> Dict[str, np.ndarray]:
    q = q.copy()
    s = s.copy()
    q[key_col] = q[key_col].astype(str)
    s[key_col] = s[key_col].astype(str)
    if right_key not in s.columns:
        alt = "snap_mi" if right_key == "mi_snap" else "mi_snap"
        if alt in s.columns:
            s = s.rename(columns={alt: right_key})
    q[left_key] = pd.to_numeric(q[left_key], errors="coerce").astype("float64")
    s[right_key] = pd.to_numeric(s[right_key], errors="coerce").astype("float64")

    cats = pd.Index(np.unique(np.concatenate([q[key_col].values, s[key_col].values])))
    gid_map = pd.Series(np.arange(len(cats), dtype=np.int64), index=cats)
    gid_left = gid_map.reindex(q[key_col].values).to_numpy(dtype=np.int64, copy=False)
    gid_right = gid_map.reindex(s[key_col].values).to_numpy(dtype=np.int64, copy=False)

    miL_i = np.rint(q[left_key].to_numpy(np.float64)).astype(np.int64)
    miR_i = np.rint(s[right_key].to_numpy(np.float64)).astype(np.int64)
    OFFSET = np.int64(1 << 31)
    key_left = (gid_left << 32) + (miL_i + OFFSET)
    key_right = (gid_right << 32) + (miR_i + OFFSET)

    ordL = np.argsort(key_left, kind="mergesort")
    ordR = np.argsort(key_right, kind="mergesort")
    key_left_sorted = key_left[ordL]
    left_pos_sorted = np.arange(len(q), dtype=np.int64)[ordL]
    gid_left_sorted = gid_left[ordL]
    key_right_sorted = key_right[ordR]
    r_gid_sorted = gid_right[ordR]

    idx = np.searchsorted(key_right_sorted, key_left_sorted, side="left") - 1
    has = idx >= 0

    valid_mask = np.zeros_like(has, dtype=bool)
    if has.any():
        same_key = r_gid_sorted[idx[has]] == gid_left_sorted[has]
        valid_mask[has] = same_key

    lp_valid = left_pos_sorted[valid_mask]
    ri_valid = idx[valid_mask]

    def _col(name, default_nan=True):
        if name in s.columns:
            arr = s[name].to_numpy(np.float64)[ordR]
        else:
            arr = np.full(len(s), np.nan if default_nan else 0.0, dtype=np.float64)
        return arr

    r_snap_mi = s[right_key].to_numpy(np.float64)[ordR]
    r_n_cum = _col("n_cum")
    r_pi_cum = _col("pi_cum")
    r_S_snap = _col("S_snap")
    r_C_snap = _col("C_snap")

    N = len(q)
    out = {
        "snap_mi": np.full(N, np.nan, dtype=np.float64),
        "n_cum": np.full(N, np.nan, dtype=np.float64),
        "pi_cum": np.full(N, np.nan, dtype=np.float64),
        "S_snap": np.full(N, np.nan, dtype=np.float64),
        "C_snap": np.full(N, np.nan, dtype=np.float64),
    }
    if len(lp_valid) > 0:
        out["snap_mi"][lp_valid] = r_snap_mi[ri_valid]
        out["n_cum"][lp_valid] = r_n_cum[ri_valid]
        out["pi_cum"][lp_valid] = r_pi_cum[ri_valid]
        out["S_snap"][lp_valid] = r_S_snap[ri_valid]
        out["C_snap"][lp_valid] = r_C_snap[ri_valid]
    return out

def _stitch_by_pk_strict_past(
    q: pd.DataFrame,
    s: pd.DataFrame,
    left_key: str = "mi_left",
    right_key: str = "mi_snap",
) -> Dict[str, np.ndarray]:
    q = q.copy()
    s = s.copy()
    q["PrimaryKey"] = q["PrimaryKey"].astype(str)
    s["PrimaryKey"] = s["PrimaryKey"].astype(str)
    if right_key not in s.columns:
        alt = "snap_mi" if right_key == "mi_snap" else "mi_snap"
        if alt in s.columns:
            s = s.rename(columns={alt: right_key})
    q[left_key] = pd.to_numeric(q[left_key], errors="coerce").astype("float64")
    s[right_key] = pd.to_numeric(s[right_key], errors="coerce").astype("float64")

    all_pks = np.unique(
        np.concatenate([q["PrimaryKey"].values, s["PrimaryKey"].values])
    )
    cats = pd.Index(all_pks)
    gid_map = pd.Series(np.arange(len(cats), dtype=np.int64), index=cats)
    gid_left = gid_map.reindex(q["PrimaryKey"].values).to_numpy(
        dtype=np.int64, copy=False
    )
    gid_right = gid_map.reindex(s["PrimaryKey"].values).to_numpy(
        dtype=np.int64, copy=False
    )

    miL_i = np.rint(q[left_key].to_numpy(np.float64)).astype(np.int64)
    miR_i = np.rint(s[right_key].to_numpy(np.float64)).astype(np.int64)
    OFFSET = np.int64(1 << 31)
    key_left = (gid_left << 32) + (miL_i + OFFSET)
    key_right = (gid_right << 32) + (miR_i + OFFSET)

    ordL = np.argsort(key_left, kind="mergesort")
    ordR = np.argsort(key_right, kind="mergesort")
    key_left_sorted = key_left[ordL]
    left_pos_sorted = np.arange(len(q), dtype=np.int64)[ordL]
    gid_left_sorted = gid_left[ordL]
    key_right_sorted = key_right[ordR]
    r_gid_sorted = gid_right[ordR]

    idx = np.searchsorted(key_right_sorted, key_left_sorted, side="left") - 1
    has = idx >= 0

    valid_mask = np.zeros_like(has, dtype=bool)
    if has.any():
        same_pk = r_gid_sorted[idx[has]] == gid_left_sorted[has]
        valid_mask[has] = same_pk

    lp_valid = left_pos_sorted[valid_mask]
    ri_valid = idx[valid_mask]

    def _col(name, default_nan=True):
        if name in s.columns:
            arr = s[name].to_numpy(np.float64)[ordR]
        else:
            arr = np.full(len(s), np.nan if default_nan else 0.0, dtype=np.float64)
        return arr

    s_snap = s[right_key].to_numpy(np.float64)[ordR]
    s_ncum = _col("n_cum")
    s_pic = _col("pi_cum")
    s_S = _col("S_snap")
    s_C = _col("C_snap")

    N = len(q)
    out = {
        "snap_mi": np.full(N, np.nan, dtype=np.float64),
        "n_cum": np.full(N, np.nan, dtype=np.float64),
        "pi_cum": np.full(N, np.nan, dtype=np.float64),
        "S_snap": np.full(N, np.nan, dtype=np.float64),
        "C_snap": np.full(N, np.nan, dtype=np.float64),
    }
    if len(lp_valid) > 0:
        out["snap_mi"][lp_valid] = s_snap[ri_valid]
        out["n_cum"][lp_valid] = s_ncum[ri_valid]
        out["pi_cum"][lp_valid] = s_pic[ri_valid]
        out["S_snap"][lp_valid] = s_S[ri_valid]
        out["C_snap"][lp_valid] = s_C[ri_valid]
    return out
