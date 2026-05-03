"""
Root conftest.py — makes the project root importable as a package root.
This allows `from src.features.exogenous import ...` in all test files
without requiring an installed package or a PYTHONPATH export.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
