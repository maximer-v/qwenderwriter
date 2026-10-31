#!/usr/bin/env python3
"""Tamper and determinism tests on the final checkpoint (CPU, plus a CPU-versus-GPU comparison when a GPU is present)."""
import argparse
import json
import tempfile
from pathlib import Path

from _common import ROOT  # noqa: F401
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.package.tamper import tamper_and_determinism  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(paths.FINAL_CHECKPOINT))
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--rows", default=str(paths.EXPLANATIONS / "rows.jsonl"))
parser.add_argument("--n-rows", type=int, default=1000)
parser.add_argument("--scratch", default=None)
parser.add_argument("--skip-gpu", action="store_true")
args = parser.parse_args()
smoke = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l and json.loads(l)["partition"] == "smoke"]
scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="ccllm-tamper-"))
report = tamper_and_determinism(Path(args.checkpoint), Path(args.csv), Path(args.manifest), smoke, scratch, args.n_rows, not args.skip_gpu)
for test in report["tests"]:
    print(f"{test['test']:45s} {'PASS' if test['pass'] else 'FAIL'}")
print("all_pass", report["all_pass"])
raise SystemExit(0 if report["all_pass"] else 1)
