"""Compile the trained XGBoost booster into exact fixed-point tensors and prove parity on every row."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from ..compiler.compiler import compile_manifest, load_compiled, save_compiled
from ..compiler.ir import XGBoostJSONAdapter
from ..compiler.parity import parity_statistics
from ..compiler.schema import CompiledManifest
from ..data.uci import FEATURE_NAMES, TARGET, load_table

FIXED_POINT_SCALE = 1_000_000_000
BATCH_SIZE = 512
MODEL_VERSION = "xgboost-uci-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _leaves(node: dict):
    if node["type"] == "leaf":
        yield node["value"]
    else:
        yield from _leaves(node["left"])
        yield from _leaves(node["right"])


def threshold_proof(raw: dict, X: np.ndarray, scales: tuple[int, ...]) -> dict[str, int]:
    """Every observed value and five integers around every split boundary compare identically in float32 and on the lattice."""
    observed_checks = observed_mismatches = boundary_checks = boundary_mismatches = 0
    for tree in raw["learner"]["gradient_booster"]["model"]["trees"]:
        for node, left in enumerate(tree["left_children"]):
            if left == -1:
                continue
            feature = tree["split_indices"][node]
            native = np.float32(tree["split_conditions"][node])
            boundary = math.ceil(float(tree["split_conditions"][node]) * scales[feature])
            observed = np.unique(X[:, feature].astype(np.int64))
            observed_checks += len(observed)
            observed_mismatches += int(np.count_nonzero((observed.astype(np.float32) / np.float32(scales[feature]) < native) != (observed < boundary)))
            candidates = np.arange(boundary - 2, boundary + 3, dtype=np.int64)
            boundary_checks += len(candidates)
            boundary_mismatches += int(np.count_nonzero((candidates.astype(np.float32) / np.float32(scales[feature]) < native) != (candidates < boundary)))
    return {"observed_integer_comparisons": observed_checks, "observed_integer_mismatches": observed_mismatches,
            "boundary_integer_comparisons": boundary_checks, "boundary_integer_mismatches": boundary_mismatches}


def compiled_predict(module, X: np.ndarray) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(X), BATCH_SIZE):
            batch = torch.from_numpy(X[start:start + BATCH_SIZE])
            values.append(module(batch, torch.zeros_like(batch, dtype=torch.bool)).score.cpu().numpy().astype(np.int64))
    return np.concatenate(values).astype(np.float64) / FIXED_POINT_SCALE


def compile_scorer(model_json: Path, data_csv: Path, split_manifest: Path, baseline_metrics: Path, output: Path) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    raw = json.loads(model_json.read_text("utf-8"))
    frame = load_table(data_csv)
    splits = pd.read_csv(split_manifest)
    if not np.array_equal(frame[TARGET].to_numpy(), splits["target"].to_numpy()):
        raise RuntimeError("data and split labels are not aligned")
    feature_names = tuple(raw["learner"]["feature_names"])
    if feature_names != FEATURE_NAMES:
        raise RuntimeError("booster feature order differs from the data contract")
    X = frame[list(feature_names)].to_numpy(dtype=np.int64)
    y = frame[TARGET].to_numpy(dtype=np.int8)
    scales = (1,) * len(feature_names)
    adapted = XGBoostJSONAdapter.adapt(raw, feature_scales=scales, output_scale=FIXED_POINT_SCALE)
    ir = adapted.normalized_ir
    conservative = abs(ir["base_score"]) + sum(max(abs(v) for v in _leaves(tree)) for tree in ir["trees"])
    if conservative >= 2 ** 52:
        raise RuntimeError("compiled conservative bound exceeds signed 53-bit range")
    bound = conservative + 1
    manifest = CompiledManifest.from_dict({
        "schema_version": "1.0", "model_version": MODEL_VERSION, "policy_version": "raw-margin-zero-v1",
        "features": [{"name": name, "index": i, "dtype": "fixed_int", "scale": 1, "minimum": int(X[:, i].min()),
                      "maximum": int(X[:, i].max()), "missing_allowed": True} for i, name in enumerate(feature_names)],
        "experts": [{"id": "xgboost-gbtree", "kind": "tree_ensemble", "ir_path": "normalized_ir.json",
                     "output_scale": FIXED_POINT_SCALE, "reason_namespace": "none"}],
        "routing": {"kind": "static_weighted", "weight_numerators": [1], "denominator": 1, "attention_layers": 0},
        "calibration": {"input_scale": FIXED_POINT_SCALE, "output_scale": FIXED_POINT_SCALE, "multiplier": 1, "divisor": 1,
                        "offset": 0, "minimum": -bound, "maximum": bound},
        "policy": {"decision_threshold": 0, "approve_when": "lte", "grade_bands": [
            {"grade": "NEGATIVE_MARGIN", "minimum": -bound, "maximum": -1}, {"grade": "NONNEGATIVE_MARGIN", "minimum": 0, "maximum": bound}]},
        "reasons": [],
        "precision": {"feature_bits": 32, "accumulator_bits": 53, "output_bits": 53, "max_abs_accumulator": bound,
                      "max_trees": len(ir["trees"]), "max_depth": adapted.metadata["max_depth"], "max_leaves": adapted.metadata["leaf_count"]},
        "receipt": {"algorithm": "sha256", "manifest_sha256": "", "ir_sha256": [""], "compiled_sha256": ""},
    })
    module = compile_manifest(manifest, [ir])
    output.mkdir(parents=True, exist_ok=True)
    receipt = save_compiled(output, manifest, [ir], module)
    metadata = dict(adapted.metadata)
    metadata.update({"source_model_sha256": sha256(model_json), "feature_scales": list(scales), "conservative_fixed_point_accumulator_bound": conservative})
    (output / "adapter_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _, loaded, loaded_receipt = load_compiled(output)
    if loaded_receipt != receipt:
        raise RuntimeError("save/reload receipt changed")
    booster = xgb.Booster(); booster.load_model(model_json)
    native_margin = booster.predict(xgb.DMatrix(frame[list(feature_names)], feature_names=list(feature_names)), output_margin=True).astype(np.float64)
    compiled_margin = compiled_predict(loaded, X)
    proof = threshold_proof(raw, X, scales)
    if proof["observed_integer_mismatches"] or proof["boundary_integer_mismatches"]:
        raise RuntimeError(f"integer-lattice threshold proof failed: {proof}")
    test_mask = splits["split"].eq("test").to_numpy()
    stored = json.loads(baseline_metrics.read_text("utf-8"))["test_metrics"]["uncalibrated"]
    test_stats = parity_statistics(native_margin[test_mask], compiled_margin[test_mask], y[test_mask])
    metrics = {
        "scope": "raw XGBoost margins; the compiled scorer reproduces the booster's ranking exactly",
        "fixed_point_scale": FIXED_POINT_SCALE,
        "raw_margin_quantization_bound": adapted.metadata["raw_margin_absolute_quantization_bound"],
        "threshold_equivalence": proof, "test": test_stats, "all_rows": parity_statistics(native_margin, compiled_margin, y),
        "stored_test_roc_auc": stored["roc_auc"], "stored_test_ks": stored["ks"],
        "runtime_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "torch": torch.__version__, "xgboost": xgb.__version__},
    }
    (output / "parity_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    pd.DataFrame({"opaque_row_id": splits["opaque_row_id"], "split": splits["split"], "target": y, "native_raw_margin": native_margin,
                  "compiled_raw_margin": compiled_margin, "absolute_margin_error": np.abs(compiled_margin - native_margin)}).to_csv(
        output / "row_predictions.csv", index=False, float_format="%.12g", lineterminator="\n")
    (output / "artifact_hashes.json").write_text(json.dumps({p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}, indent=2, sort_keys=True) + "\n")
    return metrics
