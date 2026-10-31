#!/usr/bin/env python3
"""Merge the gated adapter into the base checkpoint and write the final self-contained checkpoint (CPU)."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cpu_only  # noqa: F401
cpu_only()
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.package.merge import merge_checkpoint  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", default=str(paths.BASE_CHECKPOINT))
parser.add_argument("--adapter", default=str(paths.ADAPTERS / "explainer-lora" / "final_adapter"))
parser.add_argument("--gate", default=str(paths.EVALUATION / "gate_adapter_smoke.json"))
parser.add_argument("--output", default=str(paths.FINAL_CHECKPOINT))
args = parser.parse_args()
docs = ROOT / "docs"
result = merge_checkpoint(Path(args.base), Path(args.adapter), Path(args.gate), Path(args.output), ROOT / "compiled_credit_llm" / "checkpoint",
                          paths.MODEL_NAME, (docs / "CHECKPOINT_README.md").read_text(), (docs / "MODEL_CARD.md").read_text())
print(json.dumps(result, indent=2))
