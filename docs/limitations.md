# CreditOps — Limitations

## 1. Data is Simulated

All results in CreditOps are based on **Sparkov-simulated** credit card transactions.
The simulation produces statistically realistic patterns but does NOT reflect real-world fraud behaviour.
No claims about real-world fraud detection performance can be drawn from this project.

See `docs/data_disclosure.md` for the full data statement.

## 2. Near-Stationary Simulation

Sparkov generates data from fixed distributions. As measured by `src/stationarity_check.py`:

- The fraud rate is approximately constant across the simulation period.
- PSI values for model features across months are expected to be below 0.10 (stable).
- PR-AUC and recall on monthly val/test windows should show minimal variation.

**Implication:** The monitoring system and retrain trigger in CreditOps have limited opportunity to
demonstrate concept drift detection on this dataset in its natural state.

## 3. Injected Drift for Capstone

Any drift scenarios in later capstone work **MUST be:**

- Explicitly **scripted and seeded** (deterministic, reproducible)
- Clearly **labelled "injected"** in code, docs, and experiment names
- Configured via `params.yaml` (not hardcoded)
- Accompanied by a **known onset window** (start date, end date) so that detection delay can be measured
- **Not** described as "real-world drift" — always "simulated/injected drift"

Example in code:
```python
# params.yaml
drift_injection:
  enabled: true
  type: "fraud_rate_shift"        # label: "injected"
  onset_date: "2020-09-01"        # known start for delay measurement
  end_date: "2020-12-31"
  target_fraud_rate: 0.015        # injected change: 3x baseline
  seed: 42
```

## 4. Temporal Split and Imbalance

- The temporal split prevents random-split leakage but does not eliminate all temporal correlations
  within the Sparkov simulation (e.g., recurring merchant patterns).
- Majority-class downsampling is applied during training (30% kept). Full val and test splits are
  always evaluated without downsampling.
- PR-AUC on very rare classes (<1% fraud rate) has high variance for small test windows.

## 5. Feature Limitations

- **Merchant name** is dropped (very high cardinality, ~700 merchants). A future version
  could use target-encoding with proper cross-validation, but this risks leakage on simulated data.
- **City, state, job** are dropped for privacy and cardinality reasons. Haversine distance
  partially captures the geographic signal.
- **Haversine distance** is computed from cardholder home location (not current location),
  which may differ from the card's actual travel pattern in real use.

## 6. Known Issues

See `docs/known_issues.md` for any stubbed or deferred features.
