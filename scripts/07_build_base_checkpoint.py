#!/usr/bin/env python3
"""Assemble the base checkpoint: official Qwen3.5-0.8B weights plus the compiled scorer, TreeSHAP and policy buffers."""
import argparse
import json
from pathlib import Path

from _common import ROOT, cpu_only  # noqa: F401
cpu_only()
from compiled_credit_llm import paths  # noqa: E402
from compiled_credit_llm.package.base import BASE_REPO, build_base_checkpoint  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", default=BASE_REPO, help="local directory with the official Qwen3.5-0.8B weights, or the Hub id")
parser.add_argument("--compiled-scorer", default=str(paths.COMPILED_SCORER))
parser.add_argument("--compiled-treeshap", default=str(paths.COMPILED_TREESHAP))
parser.add_argument("--compiled-policy", default=str(paths.COMPILED_POLICY))
parser.add_argument("--output", default=str(paths.BASE_CHECKPOINT))
args = parser.parse_args()
receipt = build_base_checkpoint(args.base, Path(args.compiled_scorer), Path(args.compiled_treeshap), Path(args.compiled_policy),
                                Path(args.output), ROOT / "compiled_credit_llm" / "checkpoint")
print(json.dumps({k: receipt[k] for k in ("compiled_state_sha256", "official_tensor_count", "compiled_tensor_count", "packaging_runtime_seconds")}, indent=2))
