#!/usr/bin/env python3
"""Train the XGBoost scorer on the CPU: bounded search, calibration, locked test evaluation."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cpu_only  # noqa: F401
cpu_only()
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.scorer.train import train_scorer  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--output", default=str(paths.SCORER))
parser.add_argument("--threads", type=int, default=8)
args = parser.parse_args()
metrics = train_scorer(Path(args.csv), Path(args.manifest), Path(args.output), args.threads)
print(json.dumps({k: metrics[k] for k in ("selected_candidate", "selected_trees", "calibration_method", "test_metrics")}, indent=2))
