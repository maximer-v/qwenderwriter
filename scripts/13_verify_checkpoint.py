#!/usr/bin/env python3
"""CPU verification of the final checkpoint: strict reload, compiled identity, 30,000-row parity, oracle parity, refusals."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cpu_only  # noqa: F401
cpu_only()
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.package.verify import verify_checkpoint  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(paths.FINAL_CHECKPOINT))
parser.add_argument("--reference", default=str(paths.BASE_CHECKPOINT))
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--rows", default=str(paths.EXPLANATIONS / "rows.jsonl"))
args = parser.parse_args()
smoke_rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l and json.loads(l)["partition"] == "smoke"]
smoke = next((r for r in smoke_rows if r["variant"] == "N"), smoke_rows[0])  # a natural row, not the invalid-input case
reference = Path(args.reference) if args.reference and Path(args.reference).exists() else None
report = verify_checkpoint(Path(args.checkpoint), reference, Path(args.csv), Path(args.manifest), smoke)
print(json.dumps({k: report[k] for k in ("status", "compiled_state_sha256", "tensor_inventory", "policy_oracle_parity", "deterministic_refusals") if k in report}, indent=2, default=str)[:3000])
