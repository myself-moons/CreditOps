# CreditOps — Known Issues

No critical issues as of v1.0.

## Notes

- **Production Deployment on Render** — Dockerfile is configured with dynamic port binding (`${PORT:-8000}`), `.dockerignore`, and verified with live deployment at https://creditops.onrender.com.

- **Stationarity Data and Charts** — Pipeline outputs `stationarity_report.json` and `stationarity_monthly.csv` are fully generated and served via `/api/stationarity/report` and `/api/stationarity/monthly`. The `/dataset` and `/monitor` views display stability metrics directly.

- **LightGBM is optional** — If `lightgbm` is not installed, it is silently excluded from model
  candidates. Install with `pip install lightgbm` to include it.
