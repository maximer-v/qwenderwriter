#!/usr/bin/env python3
"""Compile UCI_DEMO_POLICY_v1 into tensor buffers and audit it against the NumPy oracle."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cpu_only  # noqa: F401
cpu_only()
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.policy.compile import compile_policy  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scorer", default=str(paths.SCORER))
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--compiled-scorer", default=str(paths.COMPILED_SCORER))
parser.add_argument("--output", default=str(paths.COMPILED_POLICY))
args = parser.parse_args()
scorer = Path(args.scorer)
result = compile_policy(Path(args.csv), Path(args.manifest), Path(args.compiled_scorer), scorer / "xgboost_model.json", scorer / "calibrator.json", Path(args.output))
print(json.dumps({"policy_state_sha256": result["receipt"]["policy_state_sha256"], "test_ranking": result["audit"]["test_ranking"],
                  "outcomes_test": result["audit"]["policy_distribution"]["test"]["outcomes"]}, indent=2))
