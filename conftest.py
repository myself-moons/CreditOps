"""
conftest.py — CreditOps

Pytest configuration.
Ensures src/ is importable from the project root.
"""

import sys
from pathlib import Path

# Make the src package importable when running pytest from the project root
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
