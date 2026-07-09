import numpy as np
import pandas as pd
from src.core.constants import SEQ_LENGTHS
from src.data.utils import (
    seq_matrix,
    seq_abs_mean,
    seq_std,
    seq_count_equals,
    seq_cond_mean,
    seq_cond_std,
)


def encode_age(series: pd.Series, use_midpoint: bool = True) -> pd.DataFrame:
    s = series.astype(str)
    m = s.str.extract(r"(?P<dec>\d+)(?P<band>[ab])", expand=True)
    dec = pd.to_numeric(m["dec"], errors="coerce")
    band = m["band"].map({"a": 0, "b": 1}).astype("float32")
    age_low = dec + 5 * band
    age_ord5 = age_low + (2.5 if use_midpoint else 0.0)
    return pd.DataFrame({"age_ord5": age_ord5.astype("float32")})


def encode_testdate(series: pd.Series) -> pd.DataFrame:
    s = series.astype(str).str.strip()
    m = s.str.extract(r"(?P<y>\d{4})(?P<m>\d{2})", expand=True)
    year = pd.to_numeric(m["y"], errors="coerce")
    month = pd.to_numeric(m["m"], errors="coerce")
    month = month.where((month >= 1) & (month <= 12))
    month_abs = (year * 12 + (month - 1)).astype("float64")
    anchor_abs = float(2000 * 12 + (1 - 1))
    month_idx = (month_abs - anchor_abs).astype("float32")
    return pd.DataFrame({"year": year.astype("float32"), "month_idx": month_idx})


def preprocess_A(
    df: pd.DataFrame,
    seq_lengths: dict = SEQ_LENGTHS,
    drop_seq_cols: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    out = df
    eps = 1e-6
    cols = ["A1-1", "A1-2", "A1-3", "A1-4"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    A1_L = seq_lengths["A1"]
    A1_MR = seq_matrix(out["A1-3"], length=A1_L, dtype=np.int8, delimiter="")
    A1_RT = seq_matrix(out["A1-4"], length=A1_L, dtype=np.float32, delimiter=" ")
    A1_abs_mean = seq_abs_mean(A1_RT, miss_mat=A1_MR, miss_values=0)
    out["A1_rt_abs_mean_miss0"] = A1_abs_mean
    out["A1_rt_abs_cv_miss0"] = seq_std(
        A1_RT, absolute=True, miss_mat=A1_MR, miss_values=0
    ) / (A1_abs_mean + eps)
    out["A1_miss1_cnt"] = seq_count_equals(A1_MR, 1)
    out["A1_miss1_ratio"] = out["A1_miss1_cnt"] / A1_L
    out["A1_miss_over"] = (out["A1_miss1_cnt"] >= 3).astype("int8")

    cols = ["A2-1", "A2-2", "A2-3", "A2-4"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    A2_L = seq_lengths["A2"]
    A2_MR = seq_matrix(out["A2-3"], length=A2_L, dtype=np.int8, delimiter="")
    A2_RT = seq_matrix(out["A2-4"], length=A2_L, dtype=np.float32, delimiter=" ")
    A2_abs_mean = seq_abs_mean(A2_RT, miss_mat=A2_MR, miss_values=0)
    out["A2_rt_abs_mean_miss0"] = A2_abs_mean
    out["A2_rt_abs_cv_miss0"] = seq_std(
        A2_RT, absolute=True, miss_mat=A2_MR, miss_values=0
    ) / (A2_abs_mean + eps)
    out["A2_miss1_cnt"] = seq_count_equals(A2_MR, 1)
    out["A2_miss1_ratio"] = out["A2_miss1_cnt"] / A2_L
    out["A2_miss_over"] = (out["A2_miss1_cnt"] >= 3).astype("int8")

    cols = ["A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    A3_L = seq_lengths["A3"]
    A3_R = seq_matrix(out["A3-5"], length=A3_L, dtype=np.int8, delimiter="")
    A3_MR = seq_matrix(out["A3-6"], length=A3_L, dtype=np.int8, delimiter="")
    A3_RT = seq_matrix(out["A3-7"], length=A3_L, dtype=np.float32, delimiter=" ")
    A3_cond_mean = seq_cond_mean(A3_R, A3_RT, [1, 3], miss_mat=A3_MR, miss_values=0)
    out["A3_rt_correct_mean_miss0"] = A3_cond_mean
    out["A3_rt_correct_cv_miss0"] = seq_cond_std(
        A3_R, A3_RT, [1, 3], miss_mat=A3_MR, miss_values=0
    ) / (A3_cond_mean + eps)
    out["A3_correct_cnt_miss0"] = seq_count_equals(
        A3_R, [1, 3], miss_mat=A3_MR, miss_values=0
    )
    out["A3_correct_ratio_miss0"] = out["A3_correct_cnt_miss0"] / A3_L
    out["A3_miss1_cnt"] = seq_count_equals(A3_MR, 1)
    out["A3_miss1_ratio"] = out["A3_miss1_cnt"] / A3_L
    out["A3_miss_over"] = (out["A3_miss1_cnt"] >= 5).astype("int8")

    cols = ["A4-1", "A4-2", "A4-3", "A4-4", "A4-5"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    A4_L = seq_lengths["A4"]
    A4_R = seq_matrix(out["A4-3"], length=A4_L, dtype=np.int8, delimiter="")
    A4_MR = seq_matrix(out["A4-4"], length=A4_L, dtype=np.int8, delimiter="")
    A4_RT = seq_matrix(out["A4-5"], length=A4_L, dtype=np.float32, delimiter=" ")
    A4_cond_mean = seq_cond_mean(A4_R, A4_RT, 1, miss_mat=A4_MR, miss_values=0)
    out["A4_rt_correct_mean_miss0"] = A4_cond_mean
    out["A4_rt_correct_cv_miss0"] = seq_cond_std(
        A4_R, A4_RT, 1, miss_mat=A4_MR, miss_values=0
    ) / (A4_cond_mean + eps)
    out["A4_correct_cnt_miss0"] = seq_count_equals(
        A4_R, 1, miss_mat=A4_MR, miss_values=0
    )
    out["A4_correct_ratio_miss0"] = out["A4_correct_cnt_miss0"] / A4_L
    out["A4_miss1_cnt"] = seq_count_equals(A4_MR, 1)
    out["A4_miss1_ratio"] = out["A4_miss1_cnt"] / A4_L
    out["A4_miss_over"] = (out["A4_miss1_cnt"] >= 11).astype("int8")

    cols = ["A5-1", "A5-2", "A5-3"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    A5_L = seq_lengths["A5"]
    A5_R = seq_matrix(out["A5-2"], length=A5_L, dtype=np.int8, delimiter="")
    A5_MR = seq_matrix(out["A5-3"], length=A5_L, dtype=np.int8, delimiter="")
    out["A5_correct_cnt_miss0"] = seq_count_equals(
        A5_R, 1, miss_mat=A5_MR, miss_values=0
    )
    out["A5_correct_ratio_miss0"] = out["A5_correct_cnt_miss0"] / A5_L
    out["A5_miss1_cnt"] = seq_count_equals(A5_MR, 1)
    out["A5_miss1_ratio"] = out["A5_miss1_cnt"] / A5_L

    if "A6-1" not in out.columns:
        out["A6-1"] = np.nan
    else:
        out["A6-1"] = pd.to_numeric(out["A6-1"], errors="coerce")

    if "A7-1" not in out.columns:
        out["A7-1"] = np.nan
    else:
        out["A7-1"] = pd.to_numeric(out["A7-1"], errors="coerce")

    out["A6_correct_ratio"] = out["A6-1"] / 14
    out["A7_correct_ratio"] = out["A7-1"] / 18

    a9_cols = ["A9-1", "A9-2", "A9-3", "A9-5"]
    if set(a9_cols).issubset(out.columns):
        for c in a9_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        out["A9_emotional_behavioral"] = out["A9-1"] + out["A9-2"]
        out["A9_behavioral_judgment"] = out["A9-2"] + out["A9-3"]
        out["A9_emotional_stress"] = out["A9-1"] + out["A9-5"]
        out["A9_comprehensive_stability"] = out["A9-1"] + out["A9-2"] + out["A9-3"]

    if drop_seq_cols:
        seq_cols = [
            "A1-1",
            "A1-2",
            "A1-3",
            "A1-4",
            "A2-1",
            "A2-2",
            "A2-3",
            "A2-4",
            "A3-1",
            "A3-2",
            "A3-3",
            "A3-4",
            "A3-5",
            "A3-6",
            "A3-7",
            "A4-1",
            "A4-2",
            "A4-3",
            "A4-4",
            "A4-5",
            "A5-1",
            "A5-2",
            "A5-3",
        ]
        out = out.drop(columns=seq_cols, errors="ignore")
    return out


def preprocess_B(
    df: pd.DataFrame,
    seq_lengths: dict = SEQ_LENGTHS,
    drop_seq_cols: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    out = df
    cols = ["B1-1", "B1-2", "B1-3", "B2-1", "B2-2", "B2-3"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    B1_L = seq_lengths["B1"]
    B1_R1 = seq_matrix(out["B1-1"], length=B1_L, dtype=np.int8)
    B1_R2 = seq_matrix(out["B1-3"], length=B1_L, dtype=np.int8)
    out["B1_loc_correct_cnt"] = seq_count_equals(B1_R1, 1)
    out["B1_color_correct_cnt"] = seq_count_equals(B1_R2, [1, 3])

    B2_L = seq_lengths["B2"]
    B2_R1 = seq_matrix(out["B2-1"], length=B2_L, dtype=np.int8)
    B2_R2 = seq_matrix(out["B2-3"], length=B2_L, dtype=np.int8)
    out["B2_loc_correct_cnt"] = seq_count_equals(B2_R1, 1)
    out["B2_color_correct_cnt"] = seq_count_equals(B2_R2, [1, 3])

    cols = ["B3-1", "B3-2"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    B3_L = seq_lengths["B3"]
    B3_R = seq_matrix(out["B3-1"], length=B3_L, dtype=np.int8)
    B3_RT = seq_matrix(out["B3-2"], length=B3_L, dtype=np.float32)
    out["B3_rt_correct_mean"] = seq_cond_mean(B3_R, B3_RT, 1)
    out["B3_correct_cnt"] = seq_count_equals(B3_R, 1)

    cols = ["B4-1", "B4-2"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    B4_L = seq_lengths["B4"]
    B4_R = seq_matrix(out["B4-1"], length=B4_L, dtype=np.int8)
    B4_RT = seq_matrix(out["B4-2"], length=B4_L, dtype=np.float32)
    out["B4_rt_con_correct_mean"] = seq_cond_mean(B4_R, B4_RT, 1)
    out["B4_rt_incon_correct_mean"] = seq_cond_mean(B4_R, B4_RT, [3, 5])
    out["B4_con_correct_cnt"] = seq_count_equals(B4_R, 1)
    out["B4_incon_correct_cnt"] = seq_count_equals(B4_R, [3, 5])

    cols = ["B5-1", "B5-2"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    B5_L = seq_lengths["B5"]
    B5_R = seq_matrix(out["B5-1"], length=B5_L, dtype=np.int8)
    out["B5_correct_cnt"] = seq_count_equals(B5_R, 1)

    cols = ["B6", "B7"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    B6_L = seq_lengths["B6"]
    B7_L = seq_lengths["B7"]
    B6_R = seq_matrix(out["B6"], length=B6_L, dtype=np.int8)
    B7_R = seq_matrix(out["B7"], length=B7_L, dtype=np.int8)
    out["B6_correct_cnt"] = seq_count_equals(B6_R, 1)
    out["B7_correct_cnt"] = seq_count_equals(B7_R, 1)

    cols = ["B8"]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out[cols] = out[cols].fillna("")
    B8_L = seq_lengths["B8"]
    B8_R = seq_matrix(out["B8"], length=B8_L, dtype=np.int8)
    out["B8_correct_cnt"] = seq_count_equals(B8_R, 1)

    for c in ["B9-1", "B9-4", "B9-5", "B10-1", "B10-4", "B10-5", "B10-6"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    if set(["B9-1", "B9-4", "B9-5"]).issubset(out.columns):
        out["B9_score"] = out["B9-1"] + out["B9-4"] - out["B9-5"]
    if set(["B10-1", "B10-4", "B10-5", "B10-6"]).issubset(out.columns):
        out["B10_score"] = out["B10-1"] + out["B10-4"] - out["B10-5"] + out["B10-6"]

    if drop_seq_cols:
        seq_cols = [
            "B1-1",
            "B1-2",
            "B1-3",
            "B2-1",
            "B2-2",
            "B2-3",
            "B3-1",
            "B3-2",
            "B4-1",
            "B4-2",
            "B5-1",
            "B5-2",
            "B6",
            "B7",
            "B8",
        ]
        out = out.drop(columns=seq_cols, errors="ignore")
    return out
