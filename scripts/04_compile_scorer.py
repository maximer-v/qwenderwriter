#!/usr/bin/env python3
"""Compile the booster into exact fixed-point tensors and prove parity on all 30,000 rows."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cpu_only  # noqa: F401
cpu_only()
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.scorer.compile import compile_scorer  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scorer", default=str(paths.SCORER))
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--output", default=str(paths.COMPILED_SCORER))
args = parser.parse_args()
scorer = Path(args.scorer)
metrics = compile_scorer(scorer / "xgboost_model.json", Path(args.csv), Path(args.manifest), scorer / "metrics.json", Path(args.output))
print(json.dumps({"threshold_equivalence": metrics["threshold_equivalence"], "test": {k: metrics["test"][k] for k in ("max_absolute_margin_error", "sign_inversions", "roc_auc_difference", "ks_difference")}}, indent=2))
