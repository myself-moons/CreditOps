# CreditOps v2 Observatory — Data Dictionary & Drift Specification

## 1. Provenance of Base Data

- **Source**: Sparkov Credit Card Transaction Simulation (Brandon Harris / GitHub `JoshData/sparkov_data_generation`).
- **Nature (Real vs. Synthetic)**: The base dataset is **100% synthetically generated** via agent-based simulation. Customer profiles, merchant locations, timestamps, and fraud tags are simulated. No genuine financial PII or real cardholders exist in the dataset.
- **Split Baseline**:
  - **Train**: `2019-01-01` to `2020-06-21` (1,296,675 rows; baseline fraud rate = **0.5789% / ~0.579%**).
  - **Validation**: `2020-06-21` to `2020-10-03` (277,859 rows; fraud rate = 0.4394%).
  - **Test**: `2020-10-03` to `2020-12-31` (277,860 rows; fraud rate = 0.3325%).
- **Observatory Stream Slice**: Covers the tail of validation and the entire test period (`2020-09-01` to `2020-12-31`). At 2-day resolution, this yields **61 consecutive temporal windows**, ensuring sufficient pre-onset baseline and post-onset monitoring windows.

---

## 2. Drift Scenarios & Injection Mechanics

All injections are seeded (`np.random.default_rng(seed + window_index)`), fully deterministic, and applied after V1 feature derivation via `build_features()`.

| Scenario | What Changes | Ground Truth Labels ($Y$) | Target Mechanism |
|---|---|---|---|
| **`control`** | None | Unchanged | Establishes false alarm rate under natural Sparkov variation. |
| **`covariate_shift`** | $P(X)$ changes: `log_amt` and `haversine_km` shifted upward by `magnitude` + noise | Unchanged | Simulates macroeconomic inflation or a surge in affluent/long-distance shoppers. |
| **`fraud_rate_shift`** | Fraud proportion scales to `magnitude × baseline` (0.579%) | Resampled via oversampling real fraud | Oversamples **real fraud rows** with replacement. **Never fabricates labels on legitimate rows**. |
| **`concept_drift`** | $P(Y \mid X)$ changes: fraud migrates to different merchant categories | Conserved count, reassigned | Swaps fraud labels to legitimate rows of target or alternate categories. Overall $P(X)$ and total fraud count remain identical. |
| **`novel_pattern`** | Unseen typology introduced: all 14 `cat_* = 0`, extreme `log_amt`, weekend night | $Y=1$ for novel fraud | Fraudsters invent a new transaction channel outside training distribution. Preprocessor passes zero one-hot vectors without error. |

---

## 3. Drift Shapes

- **`sudden`**: Effective magnitude jumps from `0.0` to `magnitude` immediately at `onset_window`.
- **`gradual`**: Effective magnitude ramps linearly:
  $$\text{effective\_magnitude} = \text{magnitude} \times \min\left(\frac{\text{window\_index} - \text{onset\_window}}{\text{ramp\_windows} - 1}, 1.0\right)$$
  Reaches full magnitude at `onset_window + ramp_windows - 1`.

---

## 4. Unseen Merchant Category & Preprocessing Integrity

- In V1 `src/features.py`, category encoding checks exact membership against 14 training categories:
  ```python
  for cat in CATEGORIES:
      df[f"cat_{cat}"] = (df["category"] == cat).astype(np.int8)
  ```
- When a novel merchant category appears, all 14 one-hot columns evaluate to `0`.
- In `src/data_preprocessing.py`, the `ColumnTransformer` scales continuous features and passes through binary/one-hot features. All column names and data types remain perfectly aligned with `feature_names()`. Thus, `preprocessor.transform()` evaluates cleanly without crashing.

---

## 5. Assumptions and Limitations

1. **Window-based streaming**: Simulation assumes batched 2-day arrival rather than real-time tick-by-tick message streams.
2. **Deterministic noise**: Row-level jitter uses `np.random.default_rng(seed + window_index)` to ensure exact replayability across experiment runs.
3. **Single scenario per run**: To maintain rigorous experimental control and attribution, compound/overlapping multi-scenario injections are excluded.
4. **Oversampling fidelity**: In `fraud_rate_shift`, oversampling duplicates genuine fraud cases from the same window, preserving real multivariate correlation structures without synthetic fabrication.
