"""Default locations inside the repository. Every CLI accepts overrides; these are the conventions."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_CSV = DATA / "default_of_credit_card_clients.csv"
SPLIT_MANIFEST = DATA / "split_manifest.csv"
SPLIT_SUMMARY = DATA / "split_summary.json"
EXPLANATIONS = DATA / "explanations"
ARTIFACTS = ROOT / "artifacts"
SCORER = ARTIFACTS / "scorer"
COMPILED_SCORER = ARTIFACTS / "compiled" / "scorer"
COMPILED_TREESHAP = ARTIFACTS / "compiled" / "treeshap"
COMPILED_POLICY = ARTIFACTS / "compiled" / "policy"
CHECKPOINTS = ARTIFACTS / "checkpoints"
MODEL_NAME = "CompiledCreditLLM-0.8B-UCI"
BASE_CHECKPOINT = CHECKPOINTS / f"{MODEL_NAME}-base"
FINAL_CHECKPOINT = CHECKPOINTS / MODEL_NAME
ADAPTERS = ARTIFACTS / "adapters"
EVALUATION = ARTIFACTS / "evaluation"
CONFIGS = ROOT / "configs"
