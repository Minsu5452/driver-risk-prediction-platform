import logging
import re
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger("ai-engine")


def yyyymm_to_month_index(val) -> int:
    """YYYYMM 값을 정수 월 인덱스로 변환한다.

    엣지 케이스 처리: NaN, 비숫자 문자열, 짧은 문자열.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if not s.isdigit():
        digs = re.findall(r"\d+", s)
        if not digs:
            return np.nan
        s = digs[0]
    if len(s) < 6:
        return np.nan
    year = int(s[:4])
    month = int(s[4:6])
    if month < 1 or month > 12:
        return np.nan
    return year * 12 + month


def parse_age_band(x) -> str | None:
    """연령 코드 (예: '30a', '65b', '70a')를 코호트 그룹핑용 연령대로 변환한다.

    '30a' = 30~34 (초반), '30b' = 35~39 (후반).
    반환값: 'lt65', '65_69', '70p', 또는 None.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        mid = float(x)
    else:
        s = str(x).strip().lower()
        m = re.match(r"^\s*(\d+)\s*([ab])\s*$", s)
        if m:
            base = int(m.group(1))
            ab = m.group(2)
            lo, hi = (base, base + 4) if ab == "a" else (base + 5, base + 9)
            mid = (lo + hi) / 2.0
        else:
            digs = re.findall(r"\d+", s)
            mid = float(digs[0]) if digs else np.nan
    if np.isnan(mid):
        return None
    if mid < 65:
        return "lt65"
    if 65 <= mid <= 69:
        return "65_69"
    return "70p"


def prepare_events(sago_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """사고 레코드로부터 이벤트 DataFrame을 생성한다.

    반환: (events_df, data_end_m) 여기서 data_end_m은 최대 월 인덱스.
    S_part는 사고 심각도 가중합: 사망 1.0 + 중상 0.7 + 경상 0.3 + 부상 0.1 + 벌점 0.02
    """
    if sago_df is None or len(sago_df) == 0:
        return pd.DataFrame(
            columns=["PrimaryKey", "event_m", "S_part", "KSI_evt"]
        ), -10**9

    ev = pd.DataFrame({
        "PrimaryKey": sago_df["PrimaryKey"].astype(str),
        "event_m": sago_df["AccDate"].map(yyyymm_to_month_index),
    })

    for c in ["Count_1", "Count_2", "Count_3", "Count_4", "Count_5"]:
        ev[c] = pd.to_numeric(sago_df[c], errors="coerce").fillna(0).astype(float)

    ev["S_part"] = (
        1.0 * ev["Count_1"]
        + 0.7 * ev["Count_2"]
        + 0.3 * ev["Count_3"]
        + 0.1 * ev["Count_4"]
        + 0.02 * ev["Count_5"]
    ).astype("float32")
    ev["KSI_evt"] = ((ev["Count_1"] + ev["Count_2"]) > 0).astype(int)

    ev = ev.dropna(subset=["event_m"]).reset_index(drop=True)
    data_end_m = int(ev["event_m"].max()) if not ev.empty else -10**9

    return ev, data_end_m


def build_event_cumsum_monthly(events: pd.DataFrame) -> pd.DataFrame:
    """(PrimaryKey, 월) 단위로 이벤트를 집계하고 누적합을 계산한다.

    반환 컬럼: PrimaryKey, m, S_cum (심각도 누적), K_cum (KSI 누적)
    """
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=["PrimaryKey", "m", "S_cum", "K_cum"])

    events["m"] = events["event_m"].astype("int32")

    monthly = events.groupby(["PrimaryKey", "m"], as_index=False).agg(
        S_month=("S_part", "sum"),
        K_month=("KSI_evt", "sum"),
    )
    monthly = monthly.sort_values(["PrimaryKey", "m"]).reset_index(drop=True)
    monthly["m"] = monthly["m"].astype("int32")
    grp = monthly.groupby("PrimaryKey")
    monthly["S_cum"] = grp["S_month"].cumsum().astype("float32")
    monthly["K_cum"] = grp["K_month"].cumsum().astype("int32")
    return monthly[["PrimaryKey", "m", "S_cum", "K_cum"]]


def prepare_exam_meta(exam_df: pd.DataFrame, exam_type: str) -> pd.DataFrame:
    """라벨링을 위한 검사 메타데이터를 추출한다."""
    meta = pd.DataFrame({
        "PrimaryKey": exam_df["PrimaryKey"].astype(str),
        "TestDate": exam_df["TestDate"],
        "Test_id": exam_df["Test_id"].astype(str) if "Test_id" in exam_df.columns else "",
    })

    if "Test" in exam_df.columns:
        test_col = exam_df["Test"].astype(str)
    else:
        test_col = pd.Series(exam_type, index=exam_df.index, dtype="object")
    meta["exam_type"] = str(exam_type)

    meta["exam_m"] = meta["TestDate"].map(yyyymm_to_month_index)
    valid_mask = meta["exam_m"].notna()
    orig_indices = meta.index[valid_mask]
    meta = meta.loc[valid_mask].reset_index(drop=True)
    if meta.empty:
        return pd.DataFrame(columns=["test_id", "PrimaryKey", "exam_type", "exam_m", "age_band", "TestDate"])
    meta["exam_m"] = meta["exam_m"].astype("int32")

    if "Age" in exam_df.columns:
        meta["age_band"] = exam_df.loc[orig_indices, "Age"].values
        meta["age_band"] = meta["age_band"].map(parse_age_band)
    else:
        meta["age_band"] = None

    # test_id가 없으면 생성
    if "Test_id" not in exam_df.columns or (len(meta) > 0 and meta["Test_id"].iloc[0] == ""):
        meta["test_id"] = (
            meta["PrimaryKey"].astype(str) + "_"
            + test_col.iloc[orig_indices].values.astype(str) + "_"
            + meta["TestDate"].astype(str)
        )
    else:
        meta["test_id"] = meta["Test_id"]

    return meta[["test_id", "PrimaryKey", "exam_type", "exam_m", "age_band", "TestDate"]]


def window_sum_via_group_search(
    exams_meta: pd.DataFrame,
    ev_cs: pd.DataFrame,
    H: int,
    data_end_m: int,
) -> pd.DataFrame:
    """각 검사에 대해 H개월 관찰 윈도우 내 S_post와 KSI를 계산한다.

    윈도우: [exam_m, exam_m + H - 1] 포함.
    has_window: exam_m + (H-1) <= data_end_m 이면 True.

    누적합과 searchsorted를 활용한 효율적 조회.
    """
    out = exams_meta.copy()
    out["exam_m"] = out["exam_m"].astype("int32")
    out = out.sort_values(["PrimaryKey", "exam_m"]).reset_index(drop=True)

    out["end_m"] = (out["exam_m"] + (H - 1)).astype("int32")
    out["start_prev_m"] = (out["exam_m"] - 1).astype("int32")
    out["has_window"] = out["end_m"] <= int(data_end_m)

    out["S_post"] = 0.0
    out["KSI"] = False

    if ev_cs is None or ev_cs.empty or out["has_window"].sum() == 0:
        return out

    # ev_cs는 build_event_cumsum_monthly에서 이미 정렬 + int32 — copy 제거
    ev = ev_cs.sort_values(["PrimaryKey", "m"]).reset_index(drop=True)

    valid = out[out["has_window"]].copy()

    S_end = np.zeros(len(valid), dtype=np.float32)
    S_start = np.zeros(len(valid), dtype=np.float32)
    K_end = np.zeros(len(valid), dtype=np.int32)
    K_start = np.zeros(len(valid), dtype=np.int32)

    valid_idx = valid.index.to_numpy()
    pk_valid = valid["PrimaryKey"].to_numpy()
    end_arr = valid["end_m"].to_numpy()
    start_arr = valid["start_prev_m"].to_numpy()

    grp = ev.groupby("PrimaryKey", sort=False)

    locs = defaultdict(list)
    for i, pk in enumerate(pk_valid):
        locs[pk].append(i)

    for pk, idx_list in locs.items():
        idx = np.array(idx_list, dtype=np.int32)
        try:
            g = grp.get_group(pk)
        except KeyError:
            continue

        m_arr = g["m"].to_numpy(dtype=np.int32, copy=False)
        S_arr = g["S_cum"].to_numpy(dtype=np.float32, copy=False)
        K_arr = g["K_cum"].to_numpy(dtype=np.int32, copy=False)

        ends = end_arr[idx]
        starts = start_arr[idx]

        idx_end = np.searchsorted(m_arr, ends, side="right") - 1
        idx_start = np.searchsorted(m_arr, starts, side="right") - 1

        valid_end_mask = idx_end >= 0
        valid_start_mask = idx_start >= 0

        tmp_S_end = np.zeros(len(idx), dtype=np.float32)
        tmp_K_end = np.zeros(len(idx), dtype=np.int32)
        tmp_S_start = np.zeros(len(idx), dtype=np.float32)
        tmp_K_start = np.zeros(len(idx), dtype=np.int32)

        tmp_S_end[valid_end_mask] = S_arr[idx_end[valid_end_mask]]
        tmp_K_end[valid_end_mask] = K_arr[idx_end[valid_end_mask]]
        tmp_S_start[valid_start_mask] = S_arr[idx_start[valid_start_mask]]
        tmp_K_start[valid_start_mask] = K_arr[idx_start[valid_start_mask]]

        S_end[idx] = tmp_S_end
        K_end[idx] = tmp_K_end
        S_start[idx] = tmp_S_start
        K_start[idx] = tmp_K_start

    S_win = (S_end - S_start).astype(np.float64)
    K_win = (K_end - K_start) > 0

    valid["S_post"] = S_win
    valid["KSI"] = K_win

    out.loc[valid.index, "S_post"] = valid["S_post"].to_numpy()
    out.loc[valid.index, "KSI"] = valid["KSI"].to_numpy()

    return out


def label_by_cohort_percentile(df_meta: pd.DataFrame, q: float = 0.90) -> pd.Series:
    """KSI 상태와 코호트 퍼센타일 임계값 기반으로 라벨을 부여한다.

    규칙:
        1. KSI >= 1 -> Label = 1 (자동 양성)
        2. KSI == 0일 때, 각 (age_band, exam_type) 코호트 내에서:
           - tau = percentile(S_post | S_post > 0, q), n >= 20인 경우
           - 폴백: tau = percentile(S_post | KSI==0, q), n >= 20인 경우
           - 폴백: tau = inf (추가 양성 없음)
        3. S_post >= tau -> Label = 1, 아니면 Label = 0
        4. has_window == True인 레코드에만 라벨 부여
    """
    labels = pd.Series(0, index=df_meta.index, dtype=int)
    labels[df_meta["KSI"]] = 1

    def cohort_tau(g: pd.DataFrame) -> float:
        g0 = g[(~g["KSI"]) & (g["S_post"] > 0)]
        if len(g0) >= 20:
            return float(np.quantile(g0["S_post"].values, q))
        g1 = g[~g["KSI"]]
        if len(g1) >= 20:
            return float(np.quantile(g1["S_post"].values, q))
        return float("inf")

    # tau map 생성 후 merge (루프 + .loc[] 반복 할당 대비 빠름)
    tau_map = {}
    for (band, etype), g in df_meta.groupby(["age_band", "exam_type"]):
        tau_map[(band, etype)] = cohort_tau(g)

    taus = df_meta.set_index(["age_band", "exam_type"]).index.map(
        lambda k: tau_map.get(k, float("inf"))
    )
    taus = pd.Series(taus, index=df_meta.index, dtype=np.float64)

    labels[(~df_meta["KSI"]) & (df_meta["S_post"] >= taus)] = 1
    return labels


def create_labels(
    a_comp_df: pd.DataFrame,
    b_comp_df: pd.DataFrame,
    sago_df: pd.DataFrame,
    H: int = 24,
    q: float = 0.90,
):
    """A/B 도메인 데이터셋에 라벨을 생성한다.

    Args:
        a_comp_df: 학습 스키마 A 검사 DataFrame
        b_comp_df: 학습 스키마 B 검사 DataFrame
        sago_df: 학습 스키마 사고 DataFrame (컬럼:
            PrimaryKey, Test, AccType, AccDate (YYYYMM), Count_1..Count_6)
        H: 관찰 윈도우 월수 (기본값 24)
        q: 코호트 퍼센타일 임계값 (기본값 0.90)

    Returns:
        (a_labeled, b_labeled): 원본 컬럼 + Label이 추가된 DataFrame 튜플
    """
    # 1. 전체 사고 데이터로 이벤트 준비 (날짜 필터링 없음!)
    events, data_end_m = prepare_events(sago_df)
    ev_cs = build_event_cumsum_monthly(events)

    # 2. 각 도메인 처리
    results = []
    for exam_df, exam_type in [(a_comp_df, "A"), (b_comp_df, "B")]:
        if exam_df is None or len(exam_df) == 0:
            results.append(pd.DataFrame())
            logger.info("Skipping domain %s: empty exam DataFrame", exam_type)
            continue

        # 검사 메타데이터 준비
        meta = prepare_exam_meta(exam_df, exam_type)

        # 윈도우 합계 계산
        meta2 = window_sum_via_group_search(meta, ev_cs, H, data_end_m)

        # 전체 관찰 윈도우가 있는 레코드에만 라벨 부여
        meta2["Label"] = label_by_cohort_percentile(meta2, q)

        # has_window=True인 레코드만 유지 — 필터 먼저, copy 나중에 (메모리 절감)
        keep_ids = set(meta2.loc[meta2["has_window"], "test_id"])
        id_col = "Test_id"
        if "Test_id" in exam_df.columns:
            id_series = exam_df["Test_id"]
        else:
            # test_id 생성 (exam_df 변경 없이 별도 Series로)
            if "Test" in exam_df.columns:
                test_col = exam_df["Test"].astype(str)
            else:
                test_col = pd.Series(exam_type, index=exam_df.index, dtype="object")
            id_series = (
                exam_df["PrimaryKey"].astype(str) + "_"
                + test_col + "_"
                + exam_df["TestDate"].astype(str)
            )

        mask = id_series.isin(keep_ids)
        out = exam_df.loc[mask].copy()
        if "Test_id" not in out.columns:
            out["Test_id"] = id_series.loc[mask].values

        # 라벨 병합
        out = out.merge(
            meta2[["test_id", "Label"]].rename(columns={"test_id": id_col}),
            on=id_col,
            how="left",
        )
        out["Label"] = out["Label"].fillna(0).astype(int)

        logger.info(
            "Domain %s: %d exams -> %d labeled (%.1f%% positive)",
            exam_type,
            len(exam_df),
            len(out),
            100.0 * out["Label"].mean() if len(out) > 0 else 0.0,
        )
        results.append(out)

    return results[0], results[1]
