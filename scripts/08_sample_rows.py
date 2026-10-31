#!/usr/bin/env python3
"""Choose the development rows (and variants) the explanation writer learns from and is tested on."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cpu_only  # noqa: F401
cpu_only()
from transformers import AutoModelForImageTextToText  # noqa: E402
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.data.requests import RawTable  # noqa: E402
from compiled_credit_llm.explain.rows import sample_rows, write_rows  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", default=str(paths.BASE_CHECKPOINT))
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--output", default=str(paths.EXPLANATIONS / "rows.jsonl"))
args = parser.parse_args()
model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, trust_remote_code=True).eval()
rows = sample_rows(model, RawTable(Path(args.csv), Path(args.manifest)))
summary = write_rows(rows, Path(args.output))
(Path(args.output).parent / "rows_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k in ("train", "validation", "rare_eval", "smoke")}, indent=2))
