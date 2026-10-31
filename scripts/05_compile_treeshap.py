#!/usr/bin/env python3
"""Compile exact TreeSHAP into per-tree region lookup tables and check them against the native booster."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cpu_only  # noqa: F401
cpu_only()
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.scorer.treeshap import compile_treeshap  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scorer", default=str(paths.SCORER))
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--compiled-scorer", default=str(paths.COMPILED_SCORER))
parser.add_argument("--output", default=str(paths.COMPILED_TREESHAP))
args = parser.parse_args()
result = compile_treeshap(Path(args.scorer) / "xgboost_model.json", Path(args.csv), Path(args.compiled_scorer), Path(args.output))
print(json.dumps({"regions": result["regions"], "global_max_absolute_error": result["parity"]["global_max_absolute_error"],
                  "compiled_local_accuracy_residual": result["parity"]["compiled_local_accuracy_residual"]}, indent=2))
