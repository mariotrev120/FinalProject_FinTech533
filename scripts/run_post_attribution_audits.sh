#!/usr/bin/env bash
# Single chain to run all post-attribution audits in one pass.
# Triggered manually after component_attribution.py finishes.

set -e

cd "$(dirname "$0")/.."

echo "================================================================="
echo "Tier 3: paired bootstrap on (ml_only - naked) trade returns"
echo "================================================================="
PYTHONPATH=. .venv/bin/python scripts/audit_paired_bootstrap.py 2>&1 | tail -25

echo
echo "================================================================="
echo "Tier 3: manual replay of 5 trades from 2019"
echo "================================================================="
PYTHONPATH=. .venv/bin/python scripts/audit_manual_replay.py 2>&1 | tail -30

echo
echo "================================================================="
echo "Tier 3: PutWrite reference comparison"
echo "================================================================="
PYTHONPATH=. .venv/bin/python scripts/audit_putwrite_compare.py 2>&1 | tail -25

echo
echo "================================================================="
echo "Consistency check (with new strict-equality assertions)"
echo "================================================================="
PYTHONPATH=. .venv/bin/python scripts/consistency_check.py 2>&1 | tail -10
