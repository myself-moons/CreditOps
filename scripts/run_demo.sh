#!/usr/bin/env bash
# scripts/run_demo.sh — CreditOps v2 Demo Launcher
set -e

echo "=========================================================="
echo "       CreditOps v2 — Continuous MLOps Drift Observatory  "
echo "=========================================================="

PYTHON_CMD="python"
if [ -f "venv/bin/python" ]; then
    PYTHON_CMD="venv/bin/python"
fi

# 1. Seed demo store
echo ""
echo "[1/2] Seeding Observatory LogStore demo data..."
$PYTHON_CMD scripts/seed_demo.py

# 2. Display Service URLs
echo ""
echo "[2/2] Starting CreditOps Uvicorn Server..."
echo ""
echo "Access Services at:"
echo "  • Observatory Dashboard: http://127.0.0.1:8000/observatory"
echo "  • Interactive Swagger Docs: http://127.0.0.1:8000/docs"
echo "  • V1 Experiment Dashboard: http://127.0.0.1:8000/dashboard"
echo "  • Live Predict Endpoint:   POST http://127.0.0.1:8000/predict"
echo ""
echo "Press Ctrl+C to terminate the server."
echo ""

$PYTHON_CMD -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
