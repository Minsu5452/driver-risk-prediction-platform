import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional
from src.core.constants import SEED
from src.data.loader import load_domain_train

logger = logging.getLogger(__name__)


def _compute_time_decay_stats(
    df: pd.DataFrame,
    key_col: str = "PrimaryKey",
    month_col: str = "month_idx",
    y_col: str = "Label",
    half_life_months: float = 12.0,
):
    tmp_cols = [key_col, month_col, y_col]
    has_tid = "Test_id" in df.columns
    if has_tid:
        tmp_cols.append("Test_id")
    tmp = df[tmp_cols].copy()
    tmp["__orig_idx__"] = np.arange(len(tmp), dtype=np.int32)
    sort_keys = (
        [key_col, month_col, "Test_id"]
        if has_tid
        else [key_col, month_col, "__orig_idx__"]
    )
    tmp = tmp.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

    mi = tmp[month_col].astype("float64")
    y = tmp[y_col].astype("float64")
    gobj = tmp.groupby(key_col, sort=False)

    month_prev = gobj[month_col].shift(1)
    delta = (mi - month_prev).where(month_prev.notna(), 0.0)
    delta = delta.clip(lower=0.0).astype("float64")
    H = max(float(half_life_months), 1.0)
    gamma64 = 0.5 ** (1.0 / H)

    g_series = np.power(gamma64, delta.to_numpy())
    s_g = pd.Series(g_series, index=tmp.index)
    s_G = s_g.groupby(tmp[key_col], sort=False).cumprod()

    invG = 1.0 / np.maximum(s_G.to_numpy(), 1e-15)
    y_over_G = y.to_numpy() * invG
    csum_y_over_G = (
        pd.Series(y_over_G, index=tmp.index)
        .groupby(tmp[key_col], sort=False)
        .cumsum()
        .to_numpy()
    )
    csum_invG = (
        pd.Series(invG, index=tmp.index)
        .groupby(tmp[key_col], sort=False)
        .cumsum()
        .to_numpy()
    )
    S = s_G.to_numpy() * csum_y_over_G
    C = s_G.to_numpy() * csum_invG

    S_prev = S - y.to_numpy()
    C_prev = C - 1.0

    N = len(df)
    out = {
        k: np.empty(N, dtype=np.float32)
        for k in ["S", "C", "S_prev", "C_prev", "delta", "g", "G", "month_prev"]
    }
    orig_idx = tmp["__orig_idx__"].to_numpy()
    out["S"][orig_idx] = S.astype("float32")
    out["C"][orig_idx] = C.astype("float32")
    out["S_prev"][orig_idx] = S_prev.astype("float32")
    out["C_prev"][orig_idx] = C_prev.astype("float32")
    out["delta"][orig_idx] = delta.to_numpy().astype("float32")
    out["g"][orig_idx] = s_g.to_numpy().astype("float32")
    out["G"][orig_idx] = s_G.to_numpy().astype("float32")
    out["month_prev"][orig_idx] = month_prev.to_numpy(
        dtype="float64", na_value=np.nan
    ).astype("float32")
    out["gamma"] = np.float32(gamma64)
    return out


def cohort_key_from_age(age_series: pd.Series) -> pd.Series:
    s = age_series.astype(str).str.replace(r"\s+", "", regex=True)
    valid = s.str.contains(r"^\d{1,3}[ab]$", regex=True)
    return s.where(valid, "UNK").astype(str)


def build_personal_timecausal_features(
    domain: str,
    m_smooth: float,
    half_life_months: float,
    seed: int = SEED,
    freeze_within_month: bool = True,
    df: Optional[pd.DataFrame] = None,
    model_dir: str = None,
) -> pd.DataFrame:
    df0 = df if df is not None else load_domain_train(domain)
    df = df0[["Test_id", "PrimaryKey", "Label", "month_idx"]].copy()
    df["PrimaryKey"] = df["PrimaryKey"].astype(str)
    df["Test_id"] = df["Test_id"].astype(str)

    pi0 = float((df["Label"] == 1).mean())
    alpha = pi0 * m_smooth
    beta = (1.0 - pi0) * m_smooth

    gobj = df.groupby("PrimaryKey", sort=False)
    n_hist = gobj.cumcount().astype("int32")
    sum_y_cum = gobj["Label"].cumsum().astype("int32")
    sum_y_hist = (sum_y_cum - df["Label"]).astype("int32")
    denom = (alpha + beta) + n_hist
    drv_prior_row = np.where(denom > 0, (alpha + sum_y_hist) / denom, pi0).astype(
        "float32"
    )

    month_prev = gobj["month_idx"].shift(1).astype("float32")
    last_gap = (df["month_idx"].astype("float32") - month_prev).astype("float32")
    last_gap[n_hist == 0] = np.nan
    first_mi = gobj["month_idx"].transform("min").astype("float32")
    elapsed = (df["month_idx"].astype("float32") - first_mi).astype("float32")

    td = _compute_time_decay_stats(
        df,
        key_col="PrimaryKey",
        month_col="month_idx",
        y_col="Label",
        half_life_months=half_life_months,
    )
    p_w_row = (
        (alpha + td["S_prev"]) / np.maximum(alpha + beta + td["C_prev"], 1e-12)
    ).astype("float32")
    n_eff_row = np.clip(td["C_prev"], 0.0, None).astype("float32")

    if freeze_within_month:
        tmp = pd.DataFrame(
            {
                "PrimaryKey": df["PrimaryKey"].values,
                "month_idx": df["month_idx"].values,
                "Test_id": df["Test_id"].values,
                "drv_prior_row": drv_prior_row,
                "p_w_row": p_w_row,
                "n_eff_row": n_eff_row,
                "n_hist_row": n_hist.values,
                "last_gap_row": last_gap.values,
            }
        )
        tmp_sorted = tmp.sort_values(
            ["PrimaryKey", "month_idx", "Test_id"], kind="mergesort"
        )
        first_map = tmp_sorted.groupby(
            ["PrimaryKey", "month_idx"], sort=False, as_index=False
        ).first()[
            [
                "PrimaryKey",
                "month_idx",
                "drv_prior_row",
                "p_w_row",
                "n_eff_row",
                "n_hist_row",
                "last_gap_row",
            ]
        ]
        frozen = tmp[["PrimaryKey", "month_idx"]].merge(
            first_map, on=["PrimaryKey", "month_idx"], how="left", validate="m:1"
        )
        drv_prior = frozen["drv_prior_row"].astype("float32").to_numpy()
        drv_prior_w = frozen["p_w_row"].astype("float32").to_numpy()
        drv_n_eff_w = frozen["n_eff_row"].astype("float32").to_numpy()
        drv_n_hist = frozen["n_hist_row"].astype("int32").to_numpy()
        drv_last_gap = frozen["last_gap_row"].astype("float32").to_numpy()
    else:
        drv_prior = drv_prior_row
        drv_prior_w = p_w_row
        drv_n_eff_w = n_eff_row
        drv_n_hist = n_hist.to_numpy(dtype="int32")
        drv_last_gap = last_gap.to_numpy(dtype="float32")

    out = pd.DataFrame(
        {
            "Test_id": df["Test_id"].values.astype(str),
            "drv_prior": drv_prior,
            "drv_n_hist": drv_n_hist,
            "drv_last_gap_mon": drv_last_gap,
            "drv_first_mi": first_mi.values.astype("float32"),
            "drv_elapsed_since_first": elapsed.values.astype("float32"),
            "drv_prior_w": drv_prior_w,
            "drv_n_eff_w": drv_n_eff_w,
            "drv_gamma": td["gamma"],
        }
    )

    cfg = {
        "domain": domain,
        "m_smooth": m_smooth,
        "pi0": pi0,
        "half_life_months": float(half_life_months),
        "gamma": float(td["gamma"]),
        "freeze_within_month": bool(freeze_within_month),
        "note": "Deterministic month-level freezing (sorted by PrimaryKey, month, Test_id).",
    }
    _mdir = model_dir
    os.makedirs(os.path.join(_mdir, "stack", "personal"), exist_ok=True)
    with open(
        os.path.join(_mdir, "stack", "personal", f"{domain}_config.json"), "w"
    ) as f:
        json.dump(cfg, f, indent=2)
    return out



def build_cross_prior_features_timecausal_by_domain(
    m_by_domain: Dict[str, float], hl_by_domain: Dict[str, float],
    model_dir: str = None,
    A_raw: Optional[pd.DataFrame] = None,
    B_raw: Optional[pd.DataFrame] = None,
):
    _mdir = model_dir
    logger.info("[Cross-TC] Load with preprocessing...")
    _a = A_raw if A_raw is not None else load_domain_train("A")
    _b = B_raw if B_raw is not None else load_domain_train("B")
    A_df = _a[
        ["Test_id", "PrimaryKey", "Label", "month_idx"]
    ].copy()
    B_df = _b[
        ["Test_id", "PrimaryKey", "Label", "month_idx"]
    ].copy()
    del _a, _b
    for df in (A_df, B_df):
        df["PrimaryKey"] = df["PrimaryKey"].astype(str)
        df["Test_id"] = df["Test_id"].astype(str)
        df["month_idx"] = pd.to_numeric(df["month_idx"], errors="coerce").astype(
            "float64"
        )

    def _person_prior_full(
        df_src: pd.DataFrame, domain_tag: str, m_s: float, half_life: float
    ):
        df_sorted = df_src.sort_values(
            ["PrimaryKey", "month_idx", "Test_id"], kind="mergesort"
        ).reset_index(drop=True)
        g = df_sorted.groupby("PrimaryKey", sort=False)
        td = _compute_time_decay_stats(
            df_sorted, "PrimaryKey", "month_idx", "Label", half_life
        )
        last_mask = (g.cumcount() == (g["Label"].transform("size") - 1)).to_numpy()
        out = g.agg(
            n=("Label", "size"),
            sum_y=("Label", "sum"),
            last_mi=("month_idx", "max"),
            first_mi=("month_idx", "min"),
        ).reset_index()
        out["S_last_w"] = td["S"][last_mask].astype("float32")
        out["C_last_w"] = td["C"][last_mask].astype("float32")
        pi0 = float((df_sorted["Label"] == 1).mean())
        alpha = pi0 * m_s
        beta = (1.0 - pi0) * m_s
        denom = (alpha + beta) + np.maximum(out["C_last_w"].astype(float), 0.0)
        out["pi_last_w"] = (
            (alpha + out["S_last_w"].astype(float)) / np.maximum(denom, 1e-15)
        ).astype("float32")
        out["pi_mle"] = (
            out["sum_y"].astype(float) / np.maximum(out["n"].astype(float), 1e-15)
        ).astype("float32")
        out["pi"] = out["pi_last_w"].astype("float32")
        cols = [
            "PrimaryKey",
            "n",
            "sum_y",
            "last_mi",
            "first_mi",
            "S_last_w",
            "C_last_w",
            "pi_last_w",
            "pi_mle",
            "pi",
        ]
        out = out[cols].sort_values("PrimaryKey").reset_index(drop=True)
        out.to_parquet(
            os.path.join(
                _mdir, "stack", "personal", f"person_prior_{domain_tag}.parquet"
            ),
            index=False,
        )

        n_cum = (g.cumcount() + 1).astype("int32")
        sum_cum = g["Label"].cumsum().astype("int32")
        pi_cum = ((alpha + sum_cum) / (alpha + beta + n_cum)).astype("float32")
        snap = pd.DataFrame(
            {
                "PrimaryKey": df_sorted["PrimaryKey"].values.astype(str),
                "mi_snap": df_sorted["month_idx"].values.astype("float64"),
                "n_cum": n_cum.values.astype("float32"),
                "pi_cum": pi_cum.values.astype("float32"),
                "S_snap": td["S"].astype("float32"),
                "C_snap": td["C"].astype("float32"),
            }
        )
        snap = (
            snap.sort_values(["PrimaryKey", "mi_snap", "n_cum"], kind="mergesort")
            .groupby(["PrimaryKey", "mi_snap"], as_index=False, sort=False)
            .last()
        )
        snap = (
            snap[snap["mi_snap"].notna()]
            .copy()
            .sort_values(["PrimaryKey", "mi_snap"], kind="mergesort")
            .reset_index(drop=True)
        )
        snap.to_parquet(
            os.path.join(_mdir, "stack", "personal", f"snap_{domain_tag}.parquet"),
            index=False,
        )
        return pi0

    pi0_A = _person_prior_full(A_df, "A", m_by_domain["A"], hl_by_domain["A"])
    pi0_B = _person_prior_full(B_df, "B", m_by_domain["B"], hl_by_domain["B"])

    alpha_A, beta_A = pi0_A * m_by_domain["A"], (1.0 - pi0_A) * m_by_domain["A"]
    alpha_B, beta_B = pi0_B * m_by_domain["B"], (1.0 - pi0_B) * m_by_domain["B"]

    def _stitch_and_eval_vectorized(
        target_df: pd.DataFrame,
        snap_src: pd.DataFrame,
        alpha: float,
        beta: float,
        gamma: float,
    ) -> pd.DataFrame:
        q = target_df[["Test_id", "PrimaryKey", "month_idx"]].copy()
        q["PrimaryKey"] = q["PrimaryKey"].astype(str)
        q["Test_id"] = q["Test_id"].astype(str)
        q["mi_left"] = pd.to_numeric(q["month_idx"], errors="coerce").astype("float64")

        s = snap_src.rename(columns={"mi_snap": "snap_mi"}).copy()
        s["PrimaryKey"] = s["PrimaryKey"].astype(str)
        s["snap_mi"] = pd.to_numeric(s["snap_mi"], errors="coerce").astype("float64")
        s = s[s["snap_mi"].notna()].copy()

        cats = pd.Index(
            np.unique(np.concatenate([q["PrimaryKey"].values, s["PrimaryKey"].values]))
        )
        gid_map = pd.Series(np.arange(len(cats), dtype=np.int64), index=cats)
        gid_left = gid_map.reindex(q["PrimaryKey"].values).to_numpy().astype(np.int64)
        gid_right = gid_map.reindex(s["PrimaryKey"].values).to_numpy().astype(np.int64)

        OFFSET = np.int64(1 << 31)
        miL_i = np.rint(q["mi_left"].to_numpy(np.float64)).astype(np.int64)
        miR_i = np.rint(s["snap_mi"].to_numpy(np.float64)).astype(np.int64)
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

        r_snap_mi = s["snap_mi"].to_numpy(np.float64)[ordR]
        r_n_cum = _col("n_cum")
        r_pi_cum = _col("pi_cum")
        r_S_snap = _col("S_snap")
        r_C_snap = _col("C_snap")

        N = len(q)
        out_snap_mi = np.full(N, np.nan, dtype=np.float64)
        out_n_cum = np.full(N, np.nan, dtype=np.float64)
        out_pi_cum = np.full(N, np.nan, dtype=np.float64)
        out_S_adj = np.full(N, np.nan, dtype=np.float64)
        out_C_adj = np.full(N, np.nan, dtype=np.float64)

        if len(lp_valid) > 0:
            out_snap_mi[lp_valid] = r_snap_mi[ri_valid]
            out_n_cum[lp_valid] = r_n_cum[ri_valid]
            out_pi_cum[lp_valid] = r_pi_cum[ri_valid]
            delta = np.maximum(
                q["mi_left"].to_numpy(np.float64)[lp_valid] - out_snap_mi[lp_valid], 0.0
            )
            decay = np.power(float(gamma), delta)
            out_S_adj[lp_valid] = r_S_snap[ri_valid] * decay
            out_C_adj[lp_valid] = r_C_snap[ri_valid] * decay

        pi_w = np.full(N, np.nan, dtype=np.float32)
        denom = float(alpha) + float(beta) + np.maximum(out_C_adj, 1e-15)
        ok = ~np.isnan(out_S_adj) & ~np.isnan(out_C_adj)
        pi_w[ok] = ((float(alpha) + out_S_adj[ok]) / denom[ok]).astype("float32")
        n_eff_w = np.zeros(N, dtype=np.float32)
        n_eff_w[ok] = np.maximum(out_C_adj[ok], 0.0).astype("float32")

        out = pd.DataFrame(
            {
                "Test_id": q["Test_id"].values,
                "raw_pi": out_pi_cum.astype("float32"),
                "n_raw": out_n_cum.astype("float32"),
                "last_gap": (q["mi_left"].to_numpy(np.float64) - out_snap_mi).astype(
                    "float32"
                ),
                "pi_w": pi_w,
                "n_eff_w": n_eff_w,
            }
        )
        out["has_hist"] = (~np.isnan(out["n_raw"].values)).astype("int8")
        return out

    snapA = pd.read_parquet(os.path.join(_mdir, "stack", "personal", "snap_A.parquet"))
    snapB = pd.read_parquet(os.path.join(_mdir, "stack", "personal", "snap_B.parquet"))

    gamma_A = 0.5 ** (1.0 / max(float(hl_by_domain["A"]), 1.0))
    gamma_B = 0.5 ** (1.0 / max(float(hl_by_domain["B"]), 1.0))

    A_feat = (
        _stitch_and_eval_vectorized(A_df, snapB, alpha_B, beta_B, gamma_B)
        .sort_values("Test_id", kind="mergesort")
        .rename(
            columns={
                "raw_pi": "cross_from_B_raw_pi",
                "n_raw": "cross_from_B_n",
                "last_gap": "cross_from_B_last_gap",
                "has_hist": "cross_from_B_has_hist",
                "pi_w": "cross_from_B_pi_w",
                "n_eff_w": "cross_from_B_n_eff_w",
            }
        )
    )
    A_feat.to_parquet(
        os.path.join(_mdir, "stack", "personal", "A_cross_from_B_timecausal.parquet"),
        index=False,
    )

    B_feat = (
        _stitch_and_eval_vectorized(B_df, snapA, alpha_A, beta_A, gamma_A)
        .sort_values("Test_id", kind="mergesort")
        .rename(
            columns={
                "raw_pi": "cross_from_A_raw_pi",
                "n_raw": "cross_from_A_n",
                "last_gap": "cross_from_A_last_gap",
                "has_hist": "cross_from_A_has_hist",
                "pi_w": "cross_from_A_pi_w",
                "n_eff_w": "cross_from_A_n_eff_w",
            }
        )
    )
    B_feat.to_parquet(
        os.path.join(_mdir, "stack", "personal", "B_cross_from_A_timecausal.parquet"),
        index=False,
    )
    logger.info("[Cross-TC] Done.")


def build_cohort_timecausal_features(
    domain: str,
    m_smooth: float,
    half_life_months: float,
    seed: int = SEED,
    freeze_within_month: bool = True,
    df: Optional[pd.DataFrame] = None,
    model_dir: str = None,
) -> pd.DataFrame:
    df0 = df if df is not None else load_domain_train(domain)
    df = df0[["Test_id", "Age", "Label", "year"]].copy()
    df["Test_id"] = df["Test_id"].astype(str)
    df["CohortKey"] = cohort_key_from_age(df["Age"])
    df["year_idx"] = pd.to_numeric(df["year"], errors="coerce").astype("float64")

    pi0 = float((df["Label"] == 1).mean())
    alpha = pi0 * m_smooth
    beta = (1.0 - pi0) * m_smooth

    g = df.groupby("CohortKey", sort=False)
    n_hist = g.cumcount().astype("int32")
    sum_y_cum = g["Label"].cumsum().astype("int32")
    sum_y_hist = (sum_y_cum - df["Label"]).astype("int32")
    denom = (alpha + beta) + n_hist
    coh_prior_row = np.where(denom > 0, (alpha + sum_y_hist) / denom, pi0).astype(
        "float32"
    )

    half_life_years = max(float(half_life_months) / 12.0, 1.0)
    td = _compute_time_decay_stats(
        df[["CohortKey", "year_idx", "Label", "Test_id"]].copy(),
        key_col="CohortKey",
        month_col="year_idx",
        y_col="Label",
        half_life_months=half_life_years,
    )
    coh_prior_w_row = (
        (alpha + td["S_prev"]) / np.maximum(alpha + beta + td["C_prev"], 1e-12)
    ).astype("float32")
    n_eff_row = np.clip(td["C_prev"], 0.0, None).astype("float32")

    first_yi = g["year_idx"].transform("min").astype("float32")
    elapsed_years = (df["year_idx"].astype("float32") - first_yi).astype("float32")
    last_gap_year = td["delta"].astype("float32")
    last_gap_year[n_hist.to_numpy(dtype=np.int32) == 0] = np.nan

    if freeze_within_month:
        tmp = pd.DataFrame(
            {
                "CohortKey": df["CohortKey"].astype(str).to_numpy(),
                "year_idx": df["year_idx"].to_numpy(),
                "Test_id": df["Test_id"].astype(str).to_numpy(),
                "coh_prior_row": coh_prior_row,
                "coh_prior_w_row": coh_prior_w_row,
                "n_eff_row": n_eff_row,
                "n_hist_row": n_hist.to_numpy(),
                "last_gap_row": last_gap_year,
            }
        )
        tmp_sorted = tmp.sort_values(
            ["CohortKey", "year_idx", "Test_id"], kind="mergesort"
        )
        first_map = tmp_sorted.groupby(
            ["CohortKey", "year_idx"], sort=False, as_index=False
        ).first()[
            [
                "CohortKey",
                "year_idx",
                "coh_prior_row",
                "coh_prior_w_row",
                "n_eff_row",
                "n_hist_row",
                "last_gap_row",
            ]
        ]
        frozen = tmp[["CohortKey", "year_idx"]].merge(
            first_map, on=["CohortKey", "year_idx"], how="left", validate="m:1"
        )
        coh_prior = frozen["coh_prior_row"].to_numpy(dtype="float32")
        coh_prior_w = frozen["coh_prior_w_row"].to_numpy(dtype="float32")
        coh_n_eff_w = frozen["n_eff_row"].to_numpy(dtype="float32")
        coh_n_hist = frozen["n_hist_row"].to_numpy(dtype="int32")
        coh_last_gap = frozen["last_gap_row"].to_numpy(dtype="float32")
    else:
        coh_prior = coh_prior_row
        coh_prior_w = coh_prior_w_row
        coh_n_eff_w = n_eff_row
        coh_n_hist = n_hist.to_numpy(dtype="int32")
        coh_last_gap = last_gap_year

    gamma_year = np.float32(0.5 ** (1.0 / half_life_years))
    out = pd.DataFrame(
        {
            "Test_id": df["Test_id"].values.astype(str),
            "coh_prior": coh_prior,
            "coh_prior_w": coh_prior_w,
            "coh_n_eff_w": coh_n_eff_w,
            "coh_n_hist": coh_n_hist,
            "coh_last_gap_year": coh_last_gap,
            "coh_first_yi": first_yi.to_numpy(dtype="float32"),
            "coh_elapsed_since_first_year": elapsed_years,
            "coh_gamma": gamma_year,
        }
    )

    cfg = {
        "domain": domain,
        "m_smooth": float(m_smooth),
        "pi0": float(pi0),
        "half_life_years": float(half_life_years),
        "gamma": float(gamma_year),
        "group_key": "Age(5y bin)",
        "freeze_within_year": bool(freeze_within_month),
        "time_bucket": "year",
    }
    _mdir = model_dir
    with open(
        os.path.join(_mdir, "stack", "personal", f"{domain}_cohort_config.json"),
        "w",
    ) as f:
        json.dump(cfg, f, indent=2)

    return out


def build_cohort_prior_features_timecausal_by_domain(
    m_by_domain: Dict[str, float], hl_by_domain: Dict[str, float],
    model_dir: str = None,
    A_raw: Optional[pd.DataFrame] = None,
    B_raw: Optional[pd.DataFrame] = None,
):
    _mdir = model_dir

    def _make(
        df_src: pd.DataFrame, domain_tag: str, m_s: float, half_life_months: float
    ):
        df = df_src[["Test_id", "Age", "Label", "year"]].copy()
        df["Test_id"] = df["Test_id"].astype(str)
        df["CohortKey"] = cohort_key_from_age(df["Age"])
        df["year_idx"] = pd.to_numeric(df["year"], errors="coerce").astype("float64")

        pi0 = float((df["Label"] == 1).mean())
        alpha = pi0 * m_s
        beta = (1.0 - pi0) * m_s

        df_sorted = df.sort_values(
            ["CohortKey", "year_idx", "Test_id"], kind="mergesort"
        ).reset_index(drop=True)
        g = df_sorted.groupby("CohortKey", sort=False)
        n_cum = (g.cumcount() + 1).astype("int32")
        sum_cum = g["Label"].cumsum().astype("int32")
        pi_cum = ((alpha + sum_cum) / (alpha + beta + n_cum)).astype("float32")

        half_life_years = max(float(half_life_months) / 12.0, 1.0)
        td = _compute_time_decay_stats(
            df_sorted,
            key_col="CohortKey",
            month_col="year_idx",
            y_col="Label",
            half_life_months=half_life_years,
        )

        snap = pd.DataFrame(
            {
                "CohortKey": df_sorted["CohortKey"].values.astype(str),
                "yi_snap": df_sorted["year_idx"].values.astype("float64"),
                "n_cum": n_cum.values.astype("float32"),
                "pi_cum": pi_cum.values.astype("float32"),
                "S_snap": td["S"].astype("float32"),
                "C_snap": td["C"].astype("float32"),
            }
        )
        snap = (
            snap.sort_values(["CohortKey", "yi_snap", "n_cum"], kind="mergesort")
            .groupby(["CohortKey", "yi_snap"], as_index=False, sort=False)
            .last()
            .sort_values(["CohortKey", "yi_snap"], kind="mergesort")
            .reset_index(drop=True)
        )
        snap.to_parquet(
            os.path.join(
                _mdir, "stack", "personal", f"snap_cohort_{domain_tag}.parquet"
            ),
            index=False,
        )

        last_mask = (g.cumcount() == (g["Label"].transform("size") - 1)).to_numpy()
        prior = g.agg(
            n=("Label", "size"),
            sum_y=("Label", "sum"),
            last_yi=("year_idx", "max"),
            first_yi=("year_idx", "min"),
        ).reset_index()
        prior["S_last_w"] = td["S"][last_mask].astype("float32")
        prior["C_last_w"] = td["C"][last_mask].astype("float32")
        denom = (alpha + beta) + np.maximum(prior["C_last_w"].astype(float), 0.0)
        prior["pi_last_w"] = (
            (alpha + prior["S_last_w"].astype(float)) / np.maximum(denom, 1e-15)
        ).astype("float32")
        prior["pi_mle"] = (
            prior["sum_y"].astype(float) / np.maximum(prior["n"].astype(float), 1e-15)
        ).astype("float32")
        prior["pi"] = prior["pi_last_w"].astype("float32")

        cols = [
            "CohortKey",
            "n",
            "sum_y",
            "last_yi",
            "first_yi",
            "S_last_w",
            "C_last_w",
            "pi_last_w",
            "pi_mle",
            "pi",
        ]
        prior[cols].sort_values("CohortKey").to_parquet(
            os.path.join(
                _mdir, "stack", "personal", f"cohort_prior_{domain_tag}.parquet"
            ),
            index=False,
        )

    _a = A_raw if A_raw is not None else load_domain_train("A")
    _b = B_raw if B_raw is not None else load_domain_train("B")
    A_df = _a[["Test_id", "Age", "Label", "year"]].copy()
    B_df = _b[["Test_id", "Age", "Label", "year"]].copy()
    del _a, _b
    _make(A_df, "A", m_by_domain["A"], hl_by_domain["A"])
    _make(B_df, "B", m_by_domain["B"], hl_by_domain["B"])
