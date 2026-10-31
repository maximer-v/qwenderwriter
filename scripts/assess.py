#!/usr/bin/env python3
"""Assess one or more raw requests with the checkpoint and print the JSON assessment(s).

    python scripts/assess.py examples/requests/decline_close_to_approval.json
    python scripts/assess.py --no-explanation examples/requests/*.json
"""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
import torch  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from compiled_credit_llm import paths  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("requests", nargs="+", help="JSON files, each {\"action\": \"ASSESS\", \"credit\": {...23 fields...}}")
parser.add_argument("--checkpoint", default=str(paths.FINAL_CHECKPOINT))
parser.add_argument("--no-explanation", action="store_true", help="skip the language model; deterministic fields only")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--output-dir", default=None, help="write one <name>.assessment.json per request")
args = parser.parse_args()
model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, trust_remote_code=True, dtype=torch.bfloat16).to(args.device).eval()
tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
for path in args.requests:
    request = json.loads(Path(path).read_text())
    result = model.assess_credit(request, tokenizer, explain=not args.no_explanation)
    print(f"# {path}: {result.status}")
    print(result.output)
    if args.output_dir:
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / (Path(path).stem + ".assessment.json")).write_text(result.output + "\n")
