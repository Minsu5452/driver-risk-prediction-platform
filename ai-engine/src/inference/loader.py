import json
import logging
import joblib
import pandas as pd
from io import BytesIO
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from src.models.calibration import CalibModel
from src.inference.fast_lookup import FastSnapshotLookup

logger = logging.getLogger(__name__)


def _load_json_from_db(run_id: int, key: str) -> dict | None:
    """DB에서 JSON 아티팩트를 로딩한다."""
    from src.core.database import load_artifact
    data = load_artifact(run_id, key)
    if data is None:
        return None
    return json.loads(data)


def _load_parquet_from_db(run_id: int, key: str) -> pd.DataFrame | None:
    """DB에서 Parquet 아티팩트를 BytesIO로 로딩한다."""
    from src.core.database import load_artifact
    data = load_artifact(run_id, key)
    if data is None:
        return None
    return pd.read_parquet(BytesIO(data))


def _load_pkl_from_db(run_id: int, key: str):
    """DB에서 PKL 아티팩트를 BytesIO로 로딩한다."""
    from src.core.database import load_artifact
    data = load_artifact(run_id, key)
    if data is None:
        return None
    return joblib.load(BytesIO(data))


@dataclass
class ModelArtifacts:

    personal_configs: Dict[str, Dict] = field(default_factory=dict)
    cohort_configs: Dict[str, Dict] = field(default_factory=dict)

    personal_snapshots: Dict[str, pd.DataFrame] = field(default_factory=dict)
    personal_priors: Dict[str, pd.DataFrame] = field(default_factory=dict)
    cohort_snapshots: Dict[str, pd.DataFrame] = field(default_factory=dict)
    cohort_priors: Dict[str, pd.DataFrame] = field(default_factory=dict)

    feature_cols: Dict[str, List[str]] = field(default_factory=dict)

    models: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)
    calibrators: Dict[str, Dict[str, CalibModel]] = field(default_factory=dict)
    ensemble_configs: Dict[str, Dict] = field(default_factory=dict)
    fast_personal_lookups: Dict[str, Any] = field(default_factory=dict)
    fast_cohort_lookups: Dict[str, Any] = field(default_factory=dict)

    # SHAP TreeExplainer 캐시: {domain: {model_name: [fold0_explainer, ...]}}
    # 모델 로드 시점이 아니라 최초 SHAP 계산 시 lazy 생성 후 재사용한다.
    # 모델과 1:1 결정적이므로 캐싱이 SHAP 값을 바꾸지 않는다.
    # artifacts가 재학습/리로드로 교체되면 캐시도 함께 폐기된다.
    explainers: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)


_GLOBAL_ARTIFACTS: Optional[ModelArtifacts] = None


def get_artifacts() -> ModelArtifacts:
    global _GLOBAL_ARTIFACTS
    if _GLOBAL_ARTIFACTS is None:
        raise RuntimeError("Artifacts not loaded. Call load_all_artifacts() first.")
    return _GLOBAL_ARTIFACTS


def load_all_artifacts(
    domains: List[str] = ["A", "B"],
    run_id: int = None,
) -> ModelArtifacts:
    global _GLOBAL_ARTIFACTS

    if run_id is None:
        logger.warning("[ModelLoader] run_id 없음 — 아티팩트 로딩 불가")
        return ModelArtifacts()

    from src.core.database import list_artifact_keys

    logger.info("[ModelLoader] DB에서 아티팩트 로딩 중... (run_id=%s)", run_id)
    artifacts = ModelArtifacts()

    for d in domains:
        # personal/cohort configs
        cfg = _load_json_from_db(run_id, f"stack/personal/{d}_config.json")
        if cfg is not None:
            artifacts.personal_configs[d] = cfg

        cfg = _load_json_from_db(run_id, f"stack/personal/{d}_cohort_config.json")
        if cfg is not None:
            artifacts.cohort_configs[d] = cfg

        # personal snapshots
        df = _load_parquet_from_db(run_id, f"stack/personal/snap_{d}.parquet")
        if df is not None:
            if "mi_snap" not in df.columns and "snap_mi" in df.columns:
                df = df.rename(columns={"snap_mi": "mi_snap"})
            df["PrimaryKey"] = df["PrimaryKey"].astype(str)
            df["mi_snap"] = pd.to_numeric(df["mi_snap"], errors="coerce").astype("float64")
            artifacts.personal_snapshots[d] = df

            flookup = FastSnapshotLookup(
                "PrimaryKey", "mi_snap", ["n_cum", "pi_cum", "S_snap", "C_snap"]
            )
            flookup.build(df)
            artifacts.fast_personal_lookups[d] = flookup

        df = _load_parquet_from_db(run_id, f"stack/personal/person_prior_{d}.parquet")
        if df is not None:
            df["PrimaryKey"] = df["PrimaryKey"].astype(str)
            artifacts.personal_priors[d] = df

        # cohort snapshots
        df = _load_parquet_from_db(run_id, f"stack/personal/snap_cohort_{d}.parquet")
        if df is not None:
            df["CohortKey"] = df["CohortKey"].astype(str)
            df["yi_snap"] = pd.to_numeric(df["yi_snap"], errors="coerce").astype("float64")
            artifacts.cohort_snapshots[d] = df

            clookup = FastSnapshotLookup(
                "CohortKey", "yi_snap", ["n_cum", "pi_cum", "S_snap", "C_snap"]
            )
            clookup.build(df)
            artifacts.fast_cohort_lookups[d] = clookup

        df = _load_parquet_from_db(run_id, f"stack/personal/cohort_prior_{d}.parquet")
        if df is not None:
            df["CohortKey"] = df["CohortKey"].astype(str)
            artifacts.cohort_priors[d] = df

    # domain-level configs (features, ensemble, calibrators) + PKL models
    for d in domains:
        fcfg = _load_json_from_db(run_id, f"stack/{d}/features.json")
        if fcfg:
            feat_cols = fcfg.get("feature_cols") or fcfg.get("final_feature_cols")
            if feat_cols:
                artifacts.feature_cols[d] = list(feat_cols)

        ens = _load_json_from_db(run_id, f"stack/{d}/ensemble.json")
        if ens:
            artifacts.ensemble_configs[d] = ens
            model_names = ens.get("model_names", [])
        else:
            model_names = []

        cal_info = _load_json_from_db(run_id, f"stack/{d}/calibrators.json")
        calib_map = {}
        if cal_info:
            for name, item in cal_info.items():
                calib_map[name] = CalibModel(kind=item["kind"], params=item["params"])
        artifacts.calibrators[d] = calib_map

        # PKL 모델: DB에서 로딩
        artifacts.models[d] = {}
        for name in model_names:
            prefix = f"stack/{d}/{name}/"
            pkl_keys = [k for k in list_artifact_keys(run_id, prefix) if k.endswith(".pkl")]
            models_list = []
            for key in pkl_keys:
                try:
                    m = _load_pkl_from_db(run_id, key)
                    if m is not None:
                        if hasattr(m, "to_cpu_inplace"):
                            m.to_cpu_inplace()
                        models_list.append(m)
                except Exception as e:
                    logger.warning(f"모델 로드 실패 {key}: {e}")
            artifacts.models[d][name] = models_list

    _GLOBAL_ARTIFACTS = artifacts
    logger.info("[ModelLoader] 아티팩트 로딩 완료.")
    return artifacts


def reload_all_artifacts(domains=["A", "B"], run_id: int = None, **_kwargs):
    return load_all_artifacts(domains, run_id=run_id)
