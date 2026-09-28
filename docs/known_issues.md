# CreditOps — Known Issues

No critical issues as of v1.0.

## Notes

- **Docker build not validated on this machine** — Docker is not confirmed available in the current
  environment. The Dockerfile is written and ready; run `docker build -t creditops .` after
  `dvc repro` completes.

- **Stationarity chart on /dataset page** — Currently displays a static text explanation.
  A future version will fetch the stationarity_report.json from the API and render a Chart.js line chart
  directly on the page. The data is already computed and stored.

- **LightGBM is optional** — If `lightgbm` is not installed, it is silently excluded from model
  candidates. Install with `pip install lightgbm` to include it.
