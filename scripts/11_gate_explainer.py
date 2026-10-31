#!/usr/bin/env python3
"""Free-generation gate on the real route for one partition (smoke, rare_eval or validation), with the adapter or merged weights."""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
import torch  # noqa: E402
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.data.requests import RawTable  # noqa: E402
from compiled_credit_llm.explain.gate import load_model, run_gate  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(paths.BASE_CHECKPOINT))
parser.add_argument("--adapter", default=None)
parser.add_argument("--partition", default="smoke", choices=("smoke", "rare_eval", "validation", "train"))
parser.add_argument("--dataset", default=str(paths.EXPLANATIONS))
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--output", default=None)
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()
rows = [json.loads(l) for l in (Path(args.dataset) / f"{args.partition}.jsonl").read_text().splitlines() if l]
if args.limit:
    rows = rows[:args.limit]
model, tokenizer = load_model(Path(args.checkpoint), Path(args.adapter) if args.adapter else None, args.device)
report = run_gate(model, tokenizer, rows, RawTable(Path(args.csv), Path(args.manifest)))
label = ("adapter" if args.adapter else Path(args.checkpoint).name)
output = Path(args.output) if args.output else paths.EVALUATION / f"gate_{label}_{args.partition}.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
if not args.adapter and Path(args.checkpoint).resolve() == paths.FINAL_CHECKPOINT.resolve():
    # gates run on the merged weights travel inside the checkpoint as merged_generation_gate/<partition>.json
    inside = paths.FINAL_CHECKPOINT / "merged_generation_gate" / f"{args.partition}.json"
    inside.parent.mkdir(exist_ok=True)
    inside.write_text(output.read_text())
print(json.dumps({k: report[k] for k in ("rows", "verified", "verified_rate", "violations", "gate")}, indent=2), "refusals", report["refusals"]["classes_passed"], "/ 7")
print("written", output)
