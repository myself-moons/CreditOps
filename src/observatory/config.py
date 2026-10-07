"""
src/observatory/config.py — CreditOps v2 Observatory

Loads observatory.yaml and exposes a typed ObservatoryConfig dataclass.
Single source of truth for all observatory settings.

Usage:
    from src.observatory.config import load_config
    cfg = load_config()
    seed = cfg.seed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OBS_YAML = ROOT_DIR / "observatory.yaml"


@dataclass
class DriftInjectionConfig:
    scenario: str = "control"          # control|covariate_shift|fraud_rate_shift|concept_drift|novel_pattern
    shape: str = "sudden"              # sudden|gradual
    onset_window: int = 15             # 0-indexed window index
    ramp_windows: int = 5              # for gradual only
    magnitude: float = 1.0            # scenario-specific
    affected_categories: List[str] = field(default_factory=list)

    VALID_SCENARIOS = frozenset(
        {"control", "covariate_shift", "fraud_rate_shift", "concept_drift", "novel_pattern"}
    )
    VALID_SHAPES = frozenset({"sudden", "gradual"})

    def __post_init__(self) -> None:
        if self.scenario not in self.VALID_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{self.scenario}'. Valid: {sorted(self.VALID_SCENARIOS)}"
            )
        if self.shape not in self.VALID_SHAPES:
            raise ValueError(
                f"Unknown shape '{self.shape}'. Valid: {sorted(self.VALID_SHAPES)}"
            )
        if self.onset_window < 0:
            raise ValueError(f"onset_window must be >= 0, got {self.onset_window}")
        if self.ramp_windows < 1:
            raise ValueError(f"ramp_windows must be >= 1, got {self.ramp_windows}")


@dataclass
class CostModelConfig:
    fn_cost: float = 500.0
    fp_cost: float = 5.0
    retrain_cost: float = 200.0
    review_cost: float = 10.0


@dataclass
class MonitoringConfig:
    reference_type: str = "stream_windows"
    reference_windows: int = 10
    k_prev_windows: int = 3
    label_delay_windows: int = 2
    k_perf_windows: int = 5
    min_fraud_rows: int = 5
    threshold_data_drift: float = 1.000
    threshold_novelty: float = 0.240
    threshold_isolation_outlier: float = 0.130
    threshold_unseen_category: int = 3
    threshold_fraud_score_drift: float = 0.020
    threshold_performance_recall: float = 0.500
    threshold_pr_auc_drop: float = 0.180


@dataclass
class GovernanceConfig:
    k_val_windows: int = 3
    margin_recall: float = 0.02
    margin_cost: float = 100.0
    rollback_window_m: int = 5
    fp_budget: Optional[float] = None
    min_val_rows: int = 20


@dataclass
class ObservatoryConfig:
    seed: int = 42
    data_dir: str = "data"
    raw_train: str = "data/fraudTrain.csv"
    raw_test: str = "data/fraudTest.csv"
    window_size_days: int = 2
    stream_start: str = "2020-09-01"
    stream_end: str = "2020-12-31"
    min_fraud_per_window: int = 5
    min_windows: int = 10
    drift_injection: DriftInjectionConfig = field(default_factory=DriftInjectionConfig)
    cost_model: CostModelConfig = field(default_factory=CostModelConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    # Raw dict preserved for any future Phase extensions
    _raw: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.drift_injection.onset_window < self.monitoring.reference_windows:
            raise ValueError(
                f"onset_window ({self.drift_injection.onset_window}) must be >= "
                f"reference window count (R={self.monitoring.reference_windows})"
            )


def load_config(path: Optional[Path] = None) -> ObservatoryConfig:
    """
    Load observatory.yaml from root (or custom path) and return ObservatoryConfig.
    Falls back to defaults if the file is missing (useful in tests).
    """
    yaml_path = path or OBS_YAML
    if not yaml_path.exists():
        logger.warning("observatory.yaml not found at %s — using defaults", yaml_path)
        return ObservatoryConfig()

    with yaml_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    obs = raw.get("observatory", {})
    drift_raw = obs.get("drift_injection", {})
    cost_raw = raw.get("cost_model", {})

    drift_cfg = DriftInjectionConfig(
        scenario=drift_raw.get("scenario", "control"),
        shape=drift_raw.get("shape", "sudden"),
        onset_window=int(drift_raw.get("onset_window", 15)),
        ramp_windows=int(drift_raw.get("ramp_windows", 5)),
        magnitude=float(drift_raw.get("magnitude", 1.0)),
        affected_categories=list(drift_raw.get("affected_categories") or []),
    )

    cost_cfg = CostModelConfig(
        fn_cost=float(cost_raw.get("fn_cost", 500.0)),
        fp_cost=float(cost_raw.get("fp_cost", 5.0)),
        retrain_cost=float(cost_raw.get("retrain_cost", 200.0)),
        review_cost=float(cost_raw.get("review_cost", 10.0)),
    )

    mon_raw = obs.get("monitoring", {})
    thresh_raw = mon_raw.get("thresholds", {})

    mon_cfg = MonitoringConfig(
        reference_type=str(mon_raw.get("reference_type", "stream_windows")),
        reference_windows=int(mon_raw.get("reference_windows", 10)),
        k_prev_windows=int(mon_raw.get("k_prev_windows", 3)),
        label_delay_windows=int(mon_raw.get("label_delay_windows", 2)),
        k_perf_windows=int(mon_raw.get("k_perf_windows", 5)),
        min_fraud_rows=int(mon_raw.get("min_fraud_rows", 5)),
        threshold_data_drift=float(thresh_raw.get("data_drift", 1.000)),
        threshold_novelty=float(thresh_raw.get("novelty", 0.240)),
        threshold_isolation_outlier=float(thresh_raw.get("isolation_outlier", 0.130)),
        threshold_unseen_category=int(thresh_raw.get("unseen_category", 3)),
        threshold_fraud_score_drift=float(thresh_raw.get("fraud_score_drift", 0.020)),
        threshold_performance_recall=float(thresh_raw.get("performance_recall", 0.500)),
        threshold_pr_auc_drop=float(thresh_raw.get("pr_auc_drop", 0.180)),
    )

    gov_raw = obs.get("governance", {})
    gov_cfg = GovernanceConfig(
        k_val_windows=int(gov_raw.get("k_val_windows", 3)),
        margin_recall=float(gov_raw.get("margin_recall", 0.02)),
        margin_cost=float(gov_raw.get("margin_cost", 100.0)),
        rollback_window_m=int(gov_raw.get("rollback_window_m", 5)),
        fp_budget=float(gov_raw["fp_budget"]) if "fp_budget" in gov_raw and gov_raw["fp_budget"] is not None else None,
        min_val_rows=int(gov_raw.get("min_val_rows", 20)),
    )

    cfg = ObservatoryConfig(
        seed=int(obs.get("seed", 42)),
        data_dir=str(obs.get("data_dir", "data")),
        raw_train=str(obs.get("raw_train", "data/fraudTrain.csv")),
        raw_test=str(obs.get("raw_test", "data/fraudTest.csv")),
        window_size_days=int(obs.get("window_size_days", 2)),
        stream_start=str(obs.get("stream_start", "2020-10-04")),
        stream_end=str(obs.get("stream_end", "2020-12-31")),
        min_fraud_per_window=int(obs.get("min_fraud_per_window", 5)),
        min_windows=int(obs.get("min_windows", 10)),
        drift_injection=drift_cfg,
        cost_model=cost_cfg,
        monitoring=mon_cfg,
        governance=gov_cfg,
        _raw=raw,
    )

    logger.info(
        "Observatory config loaded: seed=%d, window_size_days=%d, scenario=%s, shape=%s",
        cfg.seed, cfg.window_size_days,
        cfg.drift_injection.scenario, cfg.drift_injection.shape,
    )
    return cfg
