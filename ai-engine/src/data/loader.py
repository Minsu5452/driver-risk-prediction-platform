import os
import pandas as pd
from src.core.constants import BASE_DIR
from src.data.preprocessor import preprocess_A, preprocess_B, encode_testdate


def _read_train_file(path: str) -> pd.DataFrame:
    """Parquet 또는 CSV 파일을 자동 감지하여 읽는다."""
    if path.endswith(".parquet") and os.path.exists(path):
        return pd.read_parquet(path)
    csv_path = path.replace(".parquet", ".csv") if path.endswith(".parquet") else path
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    # Parquet 우선 시도 (확장자 없는 경우)
    parquet_path = path + ".parquet" if not path.endswith((".csv", ".parquet")) else path
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {path}")


def load_domain_train(domain: str, base_dir: str = BASE_DIR) -> pd.DataFrame:
    # Parquet 우선, CSV 폴백
    meta_path = os.path.join(base_dir, "train.parquet")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(base_dir, "train.csv")
    train_meta = _read_train_file(meta_path)

    detail_path = os.path.join(base_dir, "train", f"{domain}.parquet")
    if not os.path.exists(detail_path):
        detail_path = os.path.join(base_dir, "train", f"{domain}.csv")
    df_detail = _read_train_file(detail_path)

    df = df_detail.merge(
        train_meta[["Test_id", "Test", "Label"]], on=["Test_id", "Test"], how="left", validate="1:1"
    )
    df["PrimaryKey"] = df["PrimaryKey"].astype(str)
    df = preprocess_A(df) if domain == "A" else preprocess_B(df)
    df = pd.concat([df, encode_testdate(df["TestDate"])], axis=1)
    return df
