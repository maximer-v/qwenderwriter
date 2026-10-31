#!/usr/bin/env python3
"""LoRA-adapt the language backbone on the verified explanation targets (GPU)."""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.explain.train import train_explainer  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default=str(paths.CONFIGS / "explainer_lora.json"))
parser.add_argument("--checkpoint", default=str(paths.BASE_CHECKPOINT))
parser.add_argument("--dataset", default=str(paths.EXPLANATIONS))
parser.add_argument("--output", default=str(paths.ADAPTERS / "explainer-lora"))
args = parser.parse_args()
report = train_explainer(json.loads(Path(args.config).read_text()), Path(args.checkpoint), Path(args.dataset), Path(args.output))
print(json.dumps({k: report[k] for k in ("train_rows", "steps", "trainable_parameter_count", "target_module_count", "runtime_seconds", "thermal_pause_seconds")}, indent=2))
print("final eval loss:", [h.get("eval_loss") for h in report["log_history"] if "eval_loss" in h][-1:])
