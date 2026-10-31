"""Compile exact TreeSHAP into fixed-point region lookup tables, one table per tree.

TreeSHAP is not constant over a prediction leaf (cold subtrees matter), but it is constant over every region of
the Cartesian product of a tree's split thresholds. Every region is queried once from the native booster at
compile time; the checkpoint only computes region indices and gathers."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from ..compiler.compiler import load_compiled
from ..compiler.receipts import canonical_json_bytes, hash_state_dict, sha256_json
from ..data.uci import FEATURE_NAMES, load_table

SCALE = 1_000_000_000
BATCH_SIZE = 512
FEATURE_LABELS = {
    "LIMIT_BAL": "Credit limit amount", "SEX": "Recorded sex category", "EDUCATION": "Recorded education category",
    "MARRIAGE": "Recorded marital-status category", "AGE": "Age in years", "PAY_0": "Most recent repayment status",
    "PAY_2": "Repayment status two months ago", "PAY_3": "Repayment status three months ago",
    "PAY_4": "Repayment status four months ago", "PAY_5": "Repayment status five months ago",
    "PAY_6": "Repayment status six months ago", "BILL_AMT1": "Most recent statement balance",
    "BILL_AMT2": "Statement balance two months ago", "BILL_AMT3": "Statement balance three months ago",
    "BILL_AMT4": "Statement balance four months ago", "BILL_AMT5": "Statement balance five months ago",
    "BILL_AMT6": "Statement balance six months ago", "PAY_AMT1": "Most recent payment amount",
    "PAY_AMT2": "Payment amount two months ago", "PAY_AMT3": "Payment amount three months ago",
    "PAY_AMT4": "Payment amount four months ago", "PAY_AMT5": "Payment amount five months ago",
    "PAY_AMT6": "Payment amount six months ago",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cover_expectation(tree: dict, node: int = 0) -> float:
    left, right = tree["left_children"][node], tree["right_children"][node]
    if left == -1:
        return float(tree["split_conditions"][node])
    cover = tree["sum_hessian"]
    parent = float(cover[node])
    if not math.isfinite(parent) or parent <= 0:
        raise RuntimeError("invalid nonpositive node cover")
    if abs(float(cover[left]) + float(cover[right]) - parent) > max(1e-3, 2e-6 * parent):
        raise RuntimeError("child covers do not reproduce parent cover within JSON precision")
    return float(cover[left]) / parent * cover_expectation(tree, left) + float(cover[right]) / parent * cover_expectation(tree, right)


def region_definition(tree: dict):
    thresholds = defaultdict(set)
    for node, left in enumerate(tree["left_children"]):
        if left == -1:
            continue
        if tree["default_left"][node] != 0:
            raise RuntimeError("region encoding requires default-right splits (no missing values in training)")
        threshold = float(tree["split_conditions"][node])
        if not threshold.is_integer():
            raise RuntimeError("integer-feature model unexpectedly has a fractional threshold")
        thresholds[int(tree["split_indices"][node])].add(int(threshold))
    features = sorted(thresholds)
    values = [sorted(thresholds[index]) for index in features]
    radices = [len(value) + 1 for value in values]
    strides, product = [], 1
    for radix in radices:
        strides.append(product); product *= radix
    return features, values, radices, strides, product


def representative_rows(feature_count, features, thresholds, radices, strides, count):
    rows = np.zeros((count, feature_count), dtype=np.int64)
    ids = np.arange(count, dtype=np.int64)
    for feature, splits, radix, stride in zip(features, thresholds, radices, strides):
        bins = (ids // stride) % radix
        rows[:, feature] = np.asarray([splits[0] - 1, *splits], dtype=np.int64)[bins]
    return rows


def evaluate_normalized_tree(node: dict, rows: np.ndarray, result: np.ndarray, indices: np.ndarray):
    if node["type"] == "leaf":
        result[indices] = node["value"]
        return
    left_mask = rows[indices, node["feature"]] < node["threshold"]
    evaluate_normalized_tree(node["left"], rows, result, indices[left_mask])
    evaluate_normalized_tree(node["right"], rows, result, indices[~left_mask])


def compiled_scores(module, X):
    values = []
    with torch.inference_mode():
        for start in range(0, len(X), BATCH_SIZE):
            features = torch.from_numpy(X[start:start + BATCH_SIZE])
            values.append(module(features, torch.zeros_like(features, dtype=torch.bool)).score.to(torch.int64).numpy())
    return np.concatenate(values)


def runtime_attributions(state, tree_specs, X):
    total = np.zeros((len(X), X.shape[1]), dtype=np.int64)
    for tree_index, tree_spec in enumerate(tree_specs):
        region = np.zeros(len(X), dtype=np.int64)
        for feature_spec in tree_spec["features"]:
            thresholds = np.asarray(feature_spec["thresholds"], dtype=np.int64)
            region += (X[:, feature_spec["feature_index"], None] >= thresholds[None, :]).sum(axis=1) * feature_spec["stride"]
        total += state[f"trees.{tree_index}.contribution_table"].numpy()[region]
    return total


def compile_treeshap(model_json: Path, data_csv: Path, compiled_scorer: Path, output: Path) -> dict:
    started = time.perf_counter()
    raw = json.loads(model_json.read_text("utf-8"))
    native_trees = raw["learner"]["gradient_booster"]["model"]["trees"]
    feature_names = raw["learner"]["feature_names"]
    if tuple(feature_names) != FEATURE_NAMES:
        raise RuntimeError("booster feature order differs from the data contract")
    normalized = json.loads((compiled_scorer / "normalized_ir.json").read_text("ascii"))[0]
    if len(normalized["trees"]) != len(native_trees):
        raise RuntimeError("compiled scorer and booster tree counts differ")
    booster = xgb.Booster(); booster.load_model(model_json)
    frame = load_table(data_csv)
    X = frame[feature_names].to_numpy(dtype=np.int64)
    dmatrix_all = xgb.DMatrix(frame[feature_names], feature_names=feature_names)
    native_bias = float(booster.predict(xgb.DMatrix(frame[feature_names].iloc[:1], feature_names=feature_names), pred_contribs=True, strict_shape=True)[0, 0][-1])
    bias_fixed = round(native_bias * SCALE)
    base_fixed = int(normalized["base_score"])
    cover_expected = [cover_expectation(tree) for tree in native_trees]
    expected_fixed = [round(value * SCALE) for value in cover_expected]
    expected_fixed[-1] += (bias_fixed - base_fixed) - sum(expected_fixed)

    state = {"bias_fixed": torch.tensor(bias_fixed, dtype=torch.int64)}
    tree_specs, compile_metrics, total_regions = [], [], 0
    for tree_index, (native_tree, normalized_tree) in enumerate(zip(native_trees, normalized["trees"])):
        features, thresholds, radices, strides, region_count = region_definition(native_tree)
        total_regions += region_count
        rows = representative_rows(len(feature_names), features, thresholds, radices, strides, region_count)
        sliced = booster[tree_index:tree_index + 1]
        native_contributions = sliced.predict(xgb.DMatrix(pd.DataFrame(rows, columns=feature_names), feature_names=feature_names),
                                              pred_contribs=True, strict_shape=True)[:, 0, :-1].astype(np.float64)
        table = np.rint(native_contributions * SCALE).astype(np.int64)
        leaf_fixed = np.empty(region_count, dtype=np.int64)
        evaluate_normalized_tree(normalized_tree, rows, leaf_fixed, np.arange(region_count))
        target_sum = leaf_fixed - expected_fixed[tree_index]
        residual = target_sum - table.sum(axis=1)
        largest = np.argmax(np.abs(native_contributions), axis=1)
        table[np.arange(region_count), largest] += residual
        if not np.array_equal(table.sum(axis=1), target_sum):
            raise RuntimeError("fixed-point TreeSHAP local-accuracy correction failed")
        spec_features = []
        for position, (feature, split_values, stride) in enumerate(zip(features, thresholds, strides)):
            prefix = f"trees.{tree_index}.features.{position}"
            state[f"{prefix}.feature_index"] = torch.tensor(feature, dtype=torch.int64)
            state[f"{prefix}.thresholds"] = torch.tensor(split_values, dtype=torch.int64)
            state[f"{prefix}.stride"] = torch.tensor(stride, dtype=torch.int64)
            spec_features.append({"feature_index": feature, "thresholds": split_values, "stride": stride})
        state[f"trees.{tree_index}.contribution_table"] = torch.from_numpy(table)
        tree_specs.append({"features": spec_features, "region_count": region_count})
        compile_metrics.append({"tree": tree_index, "regions": region_count, "used_features": len(features),
                                "cover_expected_value": cover_expected[tree_index], "fixed_expected_value": expected_fixed[tree_index],
                                "maximum_local_accuracy_adjustment_fixed_units": int(np.max(np.abs(residual))),
                                "mean_local_accuracy_adjustment_fixed_units": float(np.mean(np.abs(residual)))})

    output.mkdir(parents=True, exist_ok=True)
    torch.save(state, output / "treeshap_weights.pt")
    metadata = {
        "schema_version": "compiled-treeshap/v1", "source_model_sha256": sha256_file(model_json),
        "algorithm": "exact XGBoost TreeSHAP region compilation using all per-tree split-decision regions",
        "native_oracle": "Booster.predict(pred_contribs=True, strict_shape=True)",
        "feature_names": feature_names, "feature_count": len(feature_names), "tree_count": len(native_trees),
        "fixed_point_scale": SCALE, "bias_fixed": bias_fixed, "native_bias": native_bias,
        "total_regions": total_regions, "tree_specs": tree_specs, "local_accuracy_tolerance_fixed_units": 0,
        "fixed_point_adjustment": "per-region residual assigned to the largest-absolute native component so integer local accuracy is exact",
        "semantics": "model-specific TreeSHAP contributions; associative, not causal or regulatory adverse-action reasons",
        "higher_risk_rule": "when the compiled policy declines, rank strictly positive contributions descending",
        "max_runtime_reasons": 4,
    }
    (output / "treeshap_metadata.json").write_bytes(canonical_json_bytes(metadata) + b"\n")
    reason_metadata = [{"feature_index": i, "raw_feature_name": name, "reason_code": f"MODEL_CONTRIB_{name}",
                        "factual_label": FEATURE_LABELS[name],
                        "positive_direction": "increases model raw margin (higher modeled default risk)",
                        "negative_direction": "decreases model raw margin (lower modeled default risk)",
                        "causal": False, "regulatory_adverse_action_reason": False} for i, name in enumerate(feature_names)]
    (output / "reason_metadata.json").write_text(json.dumps(reason_metadata, indent=2) + "\n")

    compiled_attr_fixed = runtime_attributions(state, tree_specs, X)
    _, score_module, _ = load_compiled(compiled_scorer)
    score_fixed = compiled_scores(score_module, X)
    local_fixed = compiled_attr_fixed.sum(axis=1) + bias_fixed - score_fixed
    if np.any(local_fixed):
        raise RuntimeError("compiled attribution does not exactly reproduce the compiled margin")
    native = booster.predict(dmatrix_all, pred_contribs=True, strict_shape=True)[:, 0, :].astype(np.float64)
    compiled = np.column_stack((compiled_attr_fixed.astype(np.float64) / SCALE, np.full(len(X), bias_fixed / SCALE)))
    error = compiled - native
    native_margin = booster.predict(dmatrix_all, output_margin=True).astype(np.float64)
    parity = {
        "rows": len(X),
        "feature_metrics": [{"feature_index": i, "feature_name": name, "max_absolute_error": float(np.max(np.abs(error[:, i]))),
                             "mean_absolute_error": float(np.mean(np.abs(error[:, i])))} for i, name in enumerate([*feature_names, "BIAS"])],
        "global_max_absolute_error": float(np.max(np.abs(error))), "global_mean_absolute_error": float(np.mean(np.abs(error))),
        "native_local_accuracy_residual": {"max_absolute": float(np.max(np.abs(native.sum(axis=1) - native_margin)))},
        "compiled_local_accuracy_residual": {"max_absolute_fixed_units": int(np.max(np.abs(local_fixed)))},
        "compiled_margin_vs_native_margin": {"max_absolute_error": float(np.max(np.abs(score_fixed / SCALE - native_margin)))},
        "total_regions": total_regions, "runtime_seconds": time.perf_counter() - started,
    }
    (output / "attribution_parity.json").write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n")
    receipt = {"algorithm": "sha256", "weights_logical_sha256": hash_state_dict(state), "metadata_sha256": sha256_json(metadata),
               "reason_metadata_sha256": sha256_json(reason_metadata), "source_model_sha256": sha256_file(model_json),
               "compiled_scorer_sha256": json.loads((compiled_scorer / "receipt.json").read_text())["compiled_sha256"]}
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    (output / "tree_compile_metrics.json").write_text(json.dumps(compile_metrics, indent=2) + "\n")
    return {"regions": total_regions, "weights_logical_sha256": receipt["weights_logical_sha256"], "parity": parity}
