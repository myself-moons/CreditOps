# CreditOps v2 — Capstone Plan
## MDS-24: Continuous MLOps Drift Observatory with Adaptive Retraining Policy Evaluation

> V1 (49 tests, FastAPI, DVC pipeline) stays **completely untouched** throughout.
> All new code lives under `src/observatory/` with matching tests in `tests/observatory/`.
> The only V1 file touched is `src/main.py` (one mount line added in Phase 4).

---

## Scope Decisions

| Topic | Decision |
|---|---|
| **Drift types** | `control`, `covariate_shift`, `fraud_rate_shift`, `concept_drift`, `novel_pattern` |
| **Shapes** | `sudden`, `gradual` only. One scenario per run. |
| **Windows** | 2-day windows; onset defined by window index |
| **Detectors** | PSI/KS per feature (reuse V1 logic), IsolationForest novelty score (fit on train), delayed-label PR-AUC + recall@5%FPR |
| **No CUSUM/PH/ADWIN** | Excluded on purpose — documented in limitations |
| **Policies** | `NeverRetrain`, `FixedSchedule`, `ThresholdPolicy` (PSI, V1 baseline), `AdaptivePolicy` (drift+novelty+perf, persistence N windows, cooldown, cost gate) |
| **Cost model** | `fn_cost`, `fp_cost`, `retrain_cost`, `review_cost` — all from config |
| **Replay engine** | Window-by-window, respects `label_delay_windows`, no leakage, fast offline |
| **Registry states** | `candidate → promoted | rejected`, `promoted → rolled_back`. Invalid transitions raise errors. |
| **Storage** | `LogStore` interface: SQLite (default, tests) + Firebase RTDB (live API, env-var creds, auto-fallback) |
| **Auth** | API-key header → roles (`ml_engineer`, `risk_owner`, `admin`). No JWT. |
| **Evaluation** | 5 scenarios × 4 policies × 3 seeds → mean ± std; `evaluation_report.json` |
| **Excluded** | Airflow/Prefect, PyTorch, Optuna, SHAP, Kubernetes, JWT, compound/recurring scenarios |

---

## Phase 1 — Drift Scenarios + Stream + Config
**Status: ✅ COMPLETE**

Files created/updated:
- `observatory.yaml`: Stream dates, window sizes, drift scenarios, cost model, fallback thresholds
- `src/observatory/__init__.py`: Package entrypoint
- `src/observatory/config.py`: Dataclasses, validation, and yaml loader
- `src/observatory/drift/__init__.py`: Drift package entrypoint
- `src/observatory/drift/injector.py`: Seeded drift injector (all 5 scenarios, sudden/gradual, real fraud oversampling)
- `src/observatory/simulator/__init__.py`: Simulator package entrypoint
- `src/observatory/simulator/stream.py`: WindowedStream with mapping & object StreamWindow, logging, warnings, fallback
- `src/observatory/simulator/manifest.py`: Runs manifest generator with source hash, git commit, seed
- `src/observatory/stream.py`: Backward-compatible re-exporter
- `src/observatory/run_stream.py`: DVC stage runner and manifest generator
- `dvc.yaml`: Added `Observatory_Drift` stage
- `docs/observatory_data_dictionary.md`: Detailed scenario definitions, mechanics, and data provenance
- `tests/observatory/__init__.py`: Test package entrypoint
- `tests/observatory/test_phase1.py`: Full Phase 1 test suite (all 22 tests passing)

V1 tests: all 49 green ✅ | Phase 1 tests: all 22 green ✅

---

## Phase 2 — Detectors + Performance Monitor + LogStore
**Status: ✅ COMPLETE**

Files created/updated:
- `src/observatory/storage/base.py`: `LogStore` interface with strict collection validation (`drift_events`, `policy_decisions`, `model_versions`, `audit_log`, `prediction_log`).
- `src/observatory/storage/sqlite_store.py`: `SQLiteLogStore` supporting `:memory:` and persistent SQLite, JSON serialization, and querying with filtering/limits.
- `src/observatory/monitoring/detectors.py`: Common `DetectionResult(score, alarm, window_index, detector, details)` contract:
  - `DataDriftDetector`: aggregate PSI + KS across numeric features and categorical shares pooled over last $K$ windows ($K=3$) vs reference sample.
  - `NoveltyDetector`: IsolationForest fitted on train preprocessed features; score = outlier fraction + unseen category fraction.
  - `FraudScoreDriftDetector`: PSI of champion's predicted scores vs reference distribution.
- `src/observatory/monitoring/performance.py`: `PerformanceMonitor` respecting `label_delay_windows` ($t \le t - \text{delay}$, default 2), rolling PR-AUC, recall@threshold over last $K_{\text{perf}}$ windows ($K_{\text{perf}}=5$), recall by merchant category; returns NaN/insufficient-data gracefully.
- `src/observatory/monitoring/__init__.py`: Monitoring package entrypoint.
- `src/observatory/storage/__init__.py`: Storage package entrypoint.
- Robustness: Handled empty windows, NaNs, missing columns, and constant columns with automated failure logging to `LogStore`.
- `observatory.yaml`: Threshold calibration derived on control windows 0–14 only ($\le 1$ false alarm target met on windows 15–88).
- `experiments/detector_sanity.py` & `docs/detector_sanity.md`: Scenario sanity validation table across all 5 scenarios.
- `tests/observatory/test_phase2.py`: 10 comprehensive unit tests covering contracts, label delay, robustness, storage, and calibration integrity.

Phase 1 + 2 tests: all 34 green ✅ | V1 tests: 49 passed, 1 failed + 3 errors (confirmed pre-existing)

---

## Phase 2b — Detector Fixes & Calibration Gate
**Status: ✅ COMPLETE**
- **UnseenCategoryDetector**: Separated out from NoveltyDetector; alarms when count of rows with all `cat_* == 0` >= `min_unseen_rows` (default 3). 0 false alarms on control.
- **IsolationOutlierDetector**: Uses IsolationForest fitted on random ~50k sample across entire training set; outlier fraction threshold = 0.130 calibrated on control windows 0–14.
- **PerformanceMonitor PR-AUC Drop Alarm**: Added PR-AUC drop alarm relative to reference PR-AUC (windows 0–9 pooled), calibrated threshold = 0.180.
- **Config Validation**: Validates `onset_window >= reference_windows` ($R=10$).
- **Gate Check Passed**: Control false alarms = 0 ($\le 1$), novel_pattern detected label-free with delay 0 ($\le 1$), covariate_shift detected by performance monitor with delay 2 ($\le 8$).

---

## Phase 3 — Policies + Cost Model + Replay Engine
**Status: ✅ COMPLETE**

Components created:
- `src/observatory/cost.py`: `CostModel` evaluating per-window costs (`fn_cost`=$500, `fp_cost`=$5, `review_cost`=$10, `retrain_cost`=$200).
- `src/observatory/policies/`:
  - `BasePolicy`, `PolicyDecision`, `PolicyState`
  - `NeverRetrain`: Baseline without intervention.
  - `FixedSchedule`: Retrains periodically every $K$ windows (default $K=20$).
  - `ThresholdPolicy`: Retrains on first PSI data drift alarm (V1 trigger) with cooldown.
  - `AdaptivePolicy`: Evaluates label-free & delayed performance alarms, persistence ($N=2$), and expected benefit vs retrain cost over horizon $H=10$ windows with cooldown $C=5$.
- `src/observatory/simulator/engine.py`: `ReplayEngine` and `ReplayResult` dataclass; window-by-window scoring, label delay enforcement, challenger training strictly on labelled windows $\le t - \text{delay}$ with stream weight upweighting ($w_{\text{stream}}=20.0$), naive deployment from $t+1$, and SQLite `LogStore` logging.
- `tests/observatory/test_phase3.py`: 7 comprehensive tests covering persistence, cooldown, expected-benefit gating, NeverRetrain 0 retrains, FixedSchedule schedule, no label leakage ($t_{\text{train}} \le t - \text{delay}$), seed reproducibility, and cost accounting.
- `experiments/evaluate_development_setup.py`: Development evaluation runner testing forced retrain recovery (Item E) and scenario $\times$ policy development matrix (Item F).

Tests: All 42 Observatory tests passing ✅ | V1 suite: 49 passed, 1 failed + 3 errors (pre-existing)

---

## Phase 4 — Governed Deployment + Registry + Rollback + Audit
**Status: ✅ COMPLETE**

Files created/updated:
- `src/observatory/registry.py`: ModelRegistry and strict lifecycle state machine (`candidate -> promoted | rejected; promoted -> rolled_back`). Invalid transitions raise `InvalidStateTransitionError`. Persists `model_versions` and `audit_log` records to `LogStore`.
- `src/observatory/simulator/engine.py`: Governed deployment (`deployment_mode="governed"`):
  - Strictly holds out $K_{\text{val}}$ most recent labelled windows ($t \le t - \text{delay}$) as validation. Challenger trained strictly on earlier windows + base sample.
  - Recalibrates decision threshold on holdout legitimate rows to match configured FP budget (derived on reference windows 0–9).
  - Promotion gate enforces $\text{recall} \ge \text{champ\_recall} + \text{margin}_{\text{recall}}$ at the FP budget AND challenger PR-AUC is not lower. Rejected candidates still pay retrain cost.
  - Post-promotion shadow scoring over $M$ windows with automatic rollback if realized cost exceeds shadow champion by $\text{margin}_{\text{cost}}$.
- `src/observatory/config.py` & `observatory.yaml`: Added `GovernanceConfig` dataclass and configuration parameters.
- `src/observatory/drift/injector.py`: Updated `concept_drift` to concentrate post-onset fraud in fixed categories and amount/hour band; evaluated via forced-retrain test.
- `tests/observatory/test_phase4.py`: Comprehensive test suite (all 7 tests passing) covering valid/invalid state transitions, bad challenger gate rejection, degraded model rollback trigger, holdout isolation, and reproducibility.
- `docs/evaluation_protocol.md`: Formal pre-registered evaluation protocol defining all metrics, baselines, setup ($t_{\text{onset}}=30$, seeds 1 & 2), and acceptance criteria.
- `experiments/evaluate_development_setup.py`: Evaluated 5 policies across scenarios on development setup (onset 15, seed 0).

Tests: All 49 Observatory tests passing ✅ | V1 suite: 49 passed, 1 failed + 3 errors (pre-existing)

---

## Phase 5 — Evaluation Runner + Dashboard + Dockerfile + CI
**Status: ⬜ NOT STARTED**

Files to create:
- `experiments/run_all.py`
- `docs/evaluation_protocol.md` (written BEFORE experiments)
- `src/observatory/dashboard.html`
- Updated `Dockerfile`
- `.github/workflows/ci.yml`
- `docs/model_card.md`

---

## Directory Structure (target)

```
src/observatory/
├── __init__.py
├── config.py
├── stream.py
├── cost_model.py
├── replay.py
├── registry.py
├── auth.py
├── store.py
├── api.py
├── drift/
│   ├── __init__.py
│   ├── injector.py
│   ├── detectors.py
│   └── performance_monitor.py
├── policies/
│   ├── __init__.py
│   ├── base.py
│   ├── never_retrain.py
│   ├── fixed_schedule.py
│   ├── threshold_policy.py
│   └── adaptive_policy.py
└── db/
    ├── __init__.py
    ├── schema.py
    └── session.py

tests/observatory/
├── __init__.py
├── test_phase1.py
├── test_phase2.py
├── test_phase3.py
├── test_phase4.py
└── test_phase5.py

experiments/
└── run_all.py
```
