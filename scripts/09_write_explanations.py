#!/usr/bin/env python3
"""Write verified explanation targets with a teacher model (GPU). Every target passes the checkpoint's verifier."""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
import torch  # noqa: E402
from transformers import AutoModelForImageTextToText  # noqa: E402
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.data.requests import RawTable  # noqa: E402
from compiled_credit_llm.explain.dataset import prepare_prompts, write_targets  # noqa: E402
from compiled_credit_llm.explain.rows import read_rows  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
parser.add_argument("--checkpoint", default=str(paths.BASE_CHECKPOINT))
parser.add_argument("--rows", default=str(paths.EXPLANATIONS / "rows.jsonl"))
parser.add_argument("--output", default=str(paths.EXPLANATIONS))
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--tries", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--limit", type=int, default=None, help="rows per partition (smoke test)")
args = parser.parse_args()
scorer = AutoModelForImageTextToText.from_pretrained(args.checkpoint, trust_remote_code=True).eval()  # CPU: compiled scorer and recourse only
table = RawTable(Path(args.csv), Path(args.manifest))
rows = {name: prepare_prompts(items, scorer, table) for name, items in read_rows(Path(args.rows)).items()}
del scorer
manifest = write_targets(rows, args.teacher, Path(args.output), tries=args.tries, batch_size=args.batch_size, limit=args.limit)
print(json.dumps({"counts": manifest["counts"], "stats": manifest["stats"]}, indent=2))
