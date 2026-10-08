# scripts/run_demo.ps1 — CreditOps v2 Demo Launcher
# Seeds the demo store if configured, then launches uvicorn and prints URLs

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       CreditOps v2 — Continuous MLOps Drift Observatory  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Ensure virtual environment python is used if present
$PythonCmd = "python"
if (Test-Path "venv\Scripts\python.exe") {
    $PythonCmd = "venv\Scripts\python.exe"
}

# 1. Seed demo store
Write-Host "`n[1/2] Seeding Observatory LogStore demo data..." -ForegroundColor Yellow
& $PythonCmd scripts/seed_demo.py

# 2. Display Service URLs
Write-Host "`n[2/2] Starting CreditOps Uvicorn Server..." -ForegroundColor Green
Write-Host "`nAccess Services at:" -ForegroundColor Cyan
Write-Host "  • Observatory Dashboard: http://127.0.0.1:8000/observatory" -ForegroundColor White
Write-Host "  • Interactive Swagger Docs: http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  • V1 Experiment Dashboard: http://127.0.0.1:8000/dashboard" -ForegroundColor White
Write-Host "  • Live Predict Endpoint:   POST http://127.0.0.1:8000/predict" -ForegroundColor White
Write-Host "`nPress Ctrl+C to terminate the server.`n" -ForegroundColor DarkGray

& $PythonCmd -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
