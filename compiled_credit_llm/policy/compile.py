"""Compile UCI_DEMO_POLICY_v1 into tensor buffers, audit it against the NumPy oracle and record its specification."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

from ..compiler.compiler import load_compiled
from ..compiler.receipts import canonical_json_bytes, hash_state_dict, sha256_json
from ..data.uci import FEATURE_NAMES, load_table
from .oracle import CALCULATION_NAMES, FLAG_NAMES, OUTCOME_NAMES, RISK_BAND_NAMES, THRESHOLDS, TRIGGER_NAMES, calculate_policy

POLICY_ID = "UCI_DEMO_POLICY_v1"
BATCH = 512
WARNING = "Not a real bank policy, credit policy recommendation, lending standard, or regulatory adverse-action framework."
PROBABILITY_WARNING = "Both are model outputs on a historical benchmark, not validated real-world probability of default."
FEATURE_SEMANTICS = [
    {"name": "LIMIT_BAL", "meaning": "Granted consumer/family credit limit in New Taiwan dollars."},
    {"name": "SEX", "meaning": "Recorded sex category: one male, two female."},
    {"name": "EDUCATION", "meaning": "Recorded education: one graduate school, two university, three high school, four other; observed zero/five/six are undocumented."},
    {"name": "MARRIAGE", "meaning": "Recorded marital status: one married, two single, three other; observed zero is undocumented."},
    {"name": "AGE", "meaning": "Age in years."},
    {"name": "PAY_0..PAY_6", "meaning": "Repayment status from September through April 2005; UCI documents minus one paid duly and one through nine months delay; observed minus two and zero are undocumented."},
    {"name": "BILL_AMT1..BILL_AMT6", "meaning": "Statement amounts from September through April 2005 in New Taiwan dollars."},
    {"name": "PAY_AMT1..PAY_AMT6", "meaning": "Previous-payment amounts from September through April 2005 in New Taiwan dollars."},
]
CALCULATION_SPECS = [
    {"name": "current_utilization_bp", "formula": "max(BILL_AMT1,0)*10000//max(LIMIT_BAL,1)", "unit": "basis points"},
    {"name": "maximum_utilization_bp", "formula": "max(max(BILL_AMT1..6,0))*10000//max(LIMIT_BAL,1)", "unit": "basis points"},
    {"name": "average_utilization_bp", "formula": "sum(max(BILL_AMT1..6,0))*10000//(6*max(LIMIT_BAL,1))", "unit": "basis points"},
    {"name": "delinquency_count", "formula": "count(PAY_0..PAY_6 > 0)", "unit": "months"},
    {"name": "repeated_delinquency", "formula": "delinquency_count >= 2", "unit": "binary"},
    {"name": "maximum_delinquency_severity", "formula": "max(max(PAY_0..PAY_6),0)", "unit": "months delay code"},
    {"name": "payment_coverage_bp", "formula": "sum(PAY_AMT1..6)*10000//max(sum(max(BILL_AMT1..6,0)),1)", "unit": "basis points"},
    {"name": "bill_trend_bp", "formula": "(BILL_AMT1-BILL_AMT6)*10000//max(LIMIT_BAL,1)", "unit": "basis points"},
    {"name": "bill_mean_absolute_deviation", "formula": "sum(abs(6*bill_i-sum(bills)))//36", "unit": "NT dollars, floor mean absolute deviation"},
    {"name": "payment_mean_absolute_deviation", "formula": "sum(abs(6*payment_i-sum(payments)))//36", "unit": "NT dollars, floor mean absolute deviation"},
    {"name": "zero_payment_high_utilization", "formula": "sum(payments)==0 and current_utilization_bp>=8000", "unit": "binary"},
    {"name": "positive_bill_total", "formula": "sum(max(BILL_AMT1..6,0))", "unit": "NT dollars"},
    {"name": "payment_total", "formula": "sum(PAY_AMT1..6)", "unit": "NT dollars"},
]
THRESHOLD_SPECS = [
    {"name": "current_high_utilization_bp", "value": 8000, "status": "demonstration assumption", "rationale": "Current positive statement balance at or above eighty percent of the limit is a synthetic stress indicator."},
    {"name": "maximum_high_utilization_bp", "value": 9000, "status": "demonstration assumption", "rationale": "Any of the six positive statement balances at or above ninety percent of the limit."},
    {"name": "maximum_extreme_utilization_bp", "value": 15000, "status": "demonstration assumption", "rationale": "Utilization above one hundred fifty percent combined with delinquency is a synthetic hard trigger."},
    {"name": "low_payment_coverage_bp", "value": 1000, "status": "demonstration assumption", "rationale": "Aggregate payments below ten percent of aggregate positive bills."},
    {"name": "rising_bill_trend_bp", "value": 3000, "status": "demonstration assumption", "rationale": "Current-minus-oldest bill growth of at least thirty percent of the limit."},
    {"name": "soft_delinquency_severity", "value": 2, "status": "demonstration assumption", "rationale": "Two or more months of delay counts as repeated-delinquency context."},
    {"name": "hard_delinquency_severity", "value": 3, "status": "demonstration assumption", "rationale": "A maximum status of at least three months delay is a synthetic decline trigger."},
    {"name": "hard_delinquency_count", "value": 4, "status": "demonstration assumption", "rationale": "Four delayed monthly statuses is a synthetic decline trigger."},
    {"name": "risk_band_boundaries", "value": [-1000000000, 0, 1000000000], "status": "demonstration assumption", "rationale": "Raw-margin bands are descriptive model bands, not probability or policy calibration thresholds."},
    {"name": "development_feature_bounds", "value": "per-feature min/max", "status": "development-distribution boundary", "rationale": "Minima and maxima use train, validation and calibration rows only."},
    {"name": "typed_input_contract", "value": "23 integer features plus Boolean missing mask", "status": "exact data contract", "rationale": "Non-integer and non-finite values fail before policy evaluation; missing values produce INVALID_INPUT."},
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ks(y, score):
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def compile_policy(data_csv: Path, split_manifest: Path, compiled_scorer: Path, model_json: Path, calibrator_json: Path, output: Path) -> dict:
    started = time.perf_counter()
    frame = load_table(data_csv)
    splits = pd.read_csv(split_manifest)
    X = frame[list(FEATURE_NAMES)].to_numpy(dtype=np.int64)
    development = ~splits["split"].eq("test").to_numpy()
    dev_minimum, dev_maximum = X[development].min(axis=0), X[development].max(axis=0)
    _, scorer, _ = load_compiled(compiled_scorer)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(X), BATCH):
            features = torch.from_numpy(X[start:start + BATCH])
            parts.append(scorer(features, torch.zeros_like(features, dtype=torch.bool)).score.to(torch.int64).cpu().numpy())
    score_fixed = np.concatenate(parts)
    oracle = calculate_policy(X, score_fixed, dev_minimum, dev_maximum)
    calibrator = json.loads(calibrator_json.read_text())
    if calibrator["method"] != "platt":
        raise RuntimeError("the checkpoint embeds a Platt calibrator; the frozen rule selected a different method")
    threshold_order = list(THRESHOLDS)
    state = {
        "dev_minimum": torch.from_numpy(dev_minimum.astype(np.int64)),
        "dev_maximum": torch.from_numpy(dev_maximum.astype(np.int64)),
        "thresholds": torch.tensor([THRESHOLDS[name] for name in threshold_order], dtype=torch.int64),
        "platt": torch.tensor([calibrator["coefficient"], calibrator["intercept"]], dtype=torch.float64),
        "declared_margin_bounds_fixed": scorer.output_bounds.to(torch.int64).cpu(),
    }
    output.mkdir(parents=True, exist_ok=True)
    torch.save(state, output / "policy_weights.pt")
    specification = {
        "policy_id": POLICY_ID, "policy_version": "1.0.0", "status": "SYNTHETIC_DEMONSTRATION_ONLY", "production": False,
        "regulatory": False, "warning": WARNING,
        "source": {"dataset": "UCI Default of Credit Card Clients", "uci_id": 350, "doi": "10.24432/C55S3H", "paper_doi": "10.1016/j.eswa.2007.12.020"},
        "feature_semantics": FEATURE_SEMANTICS, "calculation_specs": CALCULATION_SPECS,
        "calculation_names": list(CALCULATION_NAMES), "flag_names": list(FLAG_NAMES), "risk_band_names": list(RISK_BAND_NAMES),
        "outcome_names": list(OUTCOME_NAMES), "trigger_names": list(TRIGGER_NAMES), "threshold_order": threshold_order,
        "threshold_specs": THRESHOLD_SPECS, "decision_priority": list(OUTCOME_NAMES),
        "probability": {"uncalibrated": "sigmoid(compiled raw margin)", "frozen_platt": "sigmoid(intercept + coefficient * compiled raw margin)",
                        "real_world_pd": False, "warning": PROBABILITY_WARNING},
        "threshold_selection_labels_used": False, "development_bounds_use_test_rows": False,
    }
    (output / "policy_specification.json").write_text(json.dumps(specification, indent=2, sort_keys=True) + "\n")
    (output / "development_bounds.json").write_text(json.dumps({name: {"minimum": int(dev_minimum[i]), "maximum": int(dev_maximum[i])}
                                                                 for i, name in enumerate(FEATURE_NAMES)}, indent=2) + "\n")
    margins = torch.from_numpy(score_fixed).to(torch.float64) / 1_000_000_000.0
    uncalibrated = torch.sigmoid(margins)
    calibrated = torch.sigmoid(state["platt"][1] + state["platt"][0] * margins)
    booster = xgb.Booster(); booster.load_model(model_json)
    dmatrix = xgb.DMatrix(frame[list(FEATURE_NAMES)], feature_names=list(FEATURE_NAMES))
    native_probability = booster.predict(dmatrix).astype(np.float64)
    native_margin = booster.predict(dmatrix, output_margin=True).astype(np.float64)
    native_platt = 1.0 / (1.0 + np.exp(-(calibrator["intercept"] + calibrator["coefficient"] * native_margin)))
    distribution = {}
    for split in ("train", "validation", "calibration", "test"):
        mask = splits["split"].eq(split).to_numpy()
        distribution[split] = {"rows": int(mask.sum()),
                               "outcomes": {name: int(np.sum(oracle["outcome"][mask] == i)) for i, name in enumerate(OUTCOME_NAMES)},
                               "risk_bands": {name: int(np.sum(oracle["risk_band"][mask] == i)) for i, name in enumerate(RISK_BAND_NAMES)},
                               "triggers": {name: int(oracle["triggers"][mask, i].sum()) for i, name in enumerate(TRIGGER_NAMES)},
                               "flags": {name: int(oracle["flags"][mask, i].sum()) for i, name in enumerate(FLAG_NAMES)}}
    test = splits["split"].eq("test").to_numpy(); y = splits["target"].to_numpy(np.int8)
    audit = {
        "rows": len(X), "policy_distribution": distribution,
        "probability": {"native_xgboost_probability_max_error": float(np.max(np.abs(uncalibrated.numpy() - native_probability))),
                        "frozen_platt_native_margin_max_error": float(np.max(np.abs(calibrated.numpy() - native_platt))),
                        "minimum": float(uncalibrated.min()), "maximum": float(uncalibrated.max()),
                        "declared_margin_bounds_fixed": state["declared_margin_bounds_fixed"].tolist()},
        "test_ranking": {"margin_auc": float(roc_auc_score(y[test], score_fixed[test])), "probability_auc": float(roc_auc_score(y[test], uncalibrated.numpy()[test])),
                         "margin_ks": _ks(y[test], score_fixed[test]), "probability_ks": _ks(y[test], uncalibrated.numpy()[test])},
        "threshold_tuning_test_labels_used": False, "runtime_seconds": time.perf_counter() - started,
    }
    (output / "compile_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    receipt = {"algorithm": "sha256", "policy_state_sha256": hash_state_dict(state), "specification_sha256": sha256_json(specification),
               "source_data_sha256": sha256_file(data_csv), "split_manifest_sha256": sha256_file(split_manifest),
               "compiled_scorer_sha256": json.loads((compiled_scorer / "receipt.json").read_text())["compiled_sha256"]}
    (output / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return {"receipt": receipt, "audit": audit}
