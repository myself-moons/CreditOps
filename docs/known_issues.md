# CreditOps — Known Issues & Operational Notes

No critical issues as of v2.0 (Phase 5 Capstone Complete).

## Operational & Architectural Notes

- **Capstone v2 Verification** — All 5 acceptance criteria (AC-1 through AC-5) passed under frozen evaluation tag `eval-v1` across 48 experimental runs (4 scenarios $\times$ 6 policies $\times$ 2 seeds).
- **Test Isolation & Security** — Pytest execution strictly isolates the persistence store to in-memory/local SQLite (`tests/conftest.py` overrides `STORE_BACKEND=sqlite` and strips `FIREBASE_*` variables prior to importing application modules). Tests never contact live cloud endpoints.
- **Credential Protection in Audit Logs** — The audit log and model registry sanitize all client tokens by storing actors as `role:sha256(key)[:8]`. No raw API keys or key prefixes are persisted in database records or logs.
- **Firebase Realtime Database Fallback** — In production or local running, if Firebase credentials (`FIREBASE_CREDENTIALS_PATH` / `FIREBASE_DATABASE_URL`) are omitted or unresolvable, the system automatically falls back to SQLite with an explicit logged warning.
- **Production Deployment on Render** — Multi-stage `Dockerfile` uses non-root execution and dynamic port binding (`${PORT:-8000}`). Live deployment verified at https://creditops.onrender.com.
- **Stationarity Data and Charts** — Pipeline outputs `stationarity_report.json` and `stationarity_monthly.csv` are fully generated and served via `/api/stationarity/report` and `/api/stationarity/monthly`.
- **LightGBM is optional** — If `lightgbm` is not installed, it is gracefully excluded from candidate model sweeps. Install with `pip install lightgbm` to include it.
