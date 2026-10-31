"""CPU verification of a checkpoint: strict reload, compiled identity, all-row scorer parity against the base checkpoint,
tensor inventory, deterministic refusals with the language model forbidden, and unit-test style policy oracle parity."""
from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ..checkpoint.routing import evidence_from_score, refusal, validate_raw_request
from ..data.requests import RawTable, malformed_mutations
from ..data.uci import FEATURE_NAMES, load_table
from ..policy.oracle import calculate_policy
from .base import logical_state_hash, sha256_file

TENSOR_FIELDS = ("score_fixed", "raw_margin", "expert_scores_fixed", "policy_decision", "grade_index", "feature_contributions_fixed",
                 "feature_contributions", "attribution_bias_fixed", "attribution_bias", "policy_calculations", "policy_flags",
                 "risk_band_index", "policy_triggers", "policy_outcome_index", "model_probability", "frozen_platt_probability", "verifier_passed")


def tensor_fields(result) -> dict[str, torch.Tensor]:
    return {name: getattr(result, name) for name in TENSOR_FIELDS}


def normalize_loading(info) -> dict:
    return {key: sorted(info[key]) if isinstance(info[key], set) else info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}


def load_strict(path: Path):
    model, info = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True, low_cpu_mem_usage=True, output_loading_info=True)
    loading = normalize_loading(info)
    if any(loading.values()):
        raise RuntimeError(f"strict load failed for {path.name}: {loading}")
    return model.eval(), loading


def score_all(model, features: np.ndarray, batch: int = 1000) -> tuple[dict, dict]:
    tensors: dict[str, list] = {}
    labels: dict[str, list] = {"grade_labels": [], "risk_band_labels": [], "policy_outcome_labels": [], "policy_reason_codes": [], "model_contribution_reasons": []}
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            block = torch.from_numpy(features[start:start + batch])
            result = model.score_credit(block, torch.zeros_like(block, dtype=torch.bool))
            for name, value in tensor_fields(result).items():
                tensors.setdefault(name, []).append(np.atleast_1d(value.detach().cpu().numpy()))  # bias is a scalar per batch
            for name in labels:
                value = getattr(result, name)
                if name == "model_contribution_reasons":
                    value = [[dataclasses.asdict(reason) for reason in row] for row in value]
                labels[name].extend(value)
    return {k: np.concatenate(v) for k, v in tensors.items()}, labels


def verify_checkpoint(checkpoint: Path, reference: Path | None, data_csv: Path, split_manifest: Path, smoke_row: dict) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    model, loading = load_strict(checkpoint)
    if not hasattr(model, "assess_credit") or type(model).__name__ != "CompiledCreditLLMForConditionalGeneration":
        raise RuntimeError("checkpoint class does not expose assess_credit")
    expected = model.config.compiled_credit["compiled_sha256"]
    compiled_hash = logical_state_hash(model.compiled_credit.state_dict())
    if compiled_hash != expected:
        raise RuntimeError("compiled-state identity mismatch")
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as stream:
        keys = list(stream.keys())
    inventory = {"total_tensor_count": len(keys), "compiled_tensor_count": sum(k.startswith("compiled_credit.") for k in keys),
                 "language_tensor_count": sum(not k.startswith("compiled_credit.") and not k.startswith("mtp.") for k in keys),
                 "mtp_tensor_count": sum(k.startswith("mtp.") for k in keys), "lora_tensor_count": sum("lora_" in k or "base_layer" in k for k in keys)}
    if inventory["lora_tensor_count"]:
        raise RuntimeError("adapter tensors remain in the checkpoint")
    frame = load_table(data_csv)
    features = frame[list(FEATURE_NAMES)].to_numpy(np.int64)
    tensors, labels = score_all(model, features)
    report = {"status": "VERIFIED_CPU_SCOPE", "checkpoint": checkpoint.name, "strict_load": loading, "compiled_state_sha256": compiled_hash,
              "weight_sha256": sha256_file(checkpoint / "model.safetensors"),
              "artifact_size_bytes": sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file()), "tensor_inventory": inventory}
    if not bool(tensors["verifier_passed"].all()):
        raise RuntimeError("tensor verifier failed on at least one row")
    policy = model.compiled_credit.demo_policy
    oracle = calculate_policy(features, tensors["score_fixed"].astype(np.int64), policy.dev_minimum.numpy(), policy.dev_maximum.numpy())
    oracle_parity = {"calculations": bool(np.array_equal(tensors["policy_calculations"], oracle["calculations"])),
                     "flags": bool(np.array_equal(tensors["policy_flags"], oracle["flags"])),
                     "risk_band": bool(np.array_equal(tensors["risk_band_index"], oracle["risk_band"])),
                     "triggers": bool(np.array_equal(tensors["policy_triggers"], oracle["triggers"])),
                     "outcome": bool(np.array_equal(tensors["policy_outcome_index"], oracle["outcome"]))}
    if not all(oracle_parity.values()):
        raise RuntimeError(f"policy oracle parity failed: {oracle_parity}")
    report["policy_oracle_parity"] = {"rows": int(len(features)), **oracle_parity}
    report["local_accuracy_exact"] = bool(np.all(tensors["feature_contributions_fixed"].sum(axis=1) + tensors["attribution_bias_fixed"][0] == tensors["score_fixed"].astype(np.int64)))
    if reference is not None:
        ref_model, ref_loading = load_strict(reference)
        if logical_state_hash(ref_model.compiled_credit.state_dict()) != compiled_hash:
            raise RuntimeError("reference checkpoint carries a different compiled state")
        ref_tensors, ref_labels = score_all(ref_model, features)
        tensor_parity = {k: bool(np.array_equal(tensors[k], ref_tensors[k])) for k in ref_tensors}
        label_parity = {k: labels[k] == ref_labels[k] for k in ref_labels}
        if not all(tensor_parity.values()) or not all(label_parity.values()):
            raise RuntimeError(f"all-row parity failed: {tensor_parity} {label_parity}")
        report["all_row_scorer_parity"] = {"rows": int(len(features)), "reference": reference.name, "tensor_fields": tensor_parity,
                                           "label_and_reason_fields": label_parity, "all_exact": True}
        del ref_model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    table = RawTable(data_csv, split_manifest)
    request = table.request(smoke_row, int(policy.dev_maximum[4].item()) + 1)
    _, values, missing = validate_raw_request(request)
    evidence = evidence_from_score(model.score_credit(torch.tensor([values]), torch.tensor([missing])))
    calls = {"generate": 0}
    original_generate = model.generate

    def forbidden(*a, **k):
        calls["generate"] += 1
        raise RuntimeError("language model invoked for malformed input")

    model.generate = forbidden
    refusals = []
    for cls, bad in malformed_mutations(request, evidence):
        result = model.assess_credit(bad, tokenizer)
        refusals.append({"class": cls, "status": result.status, "refusal_class": result.refusal_class,
                         "exact_text": result.output == refusal(cls), "generation_invoked": result.generation_invoked})
    no_explanation = model.assess_credit(request, tokenizer, explain=False)
    model.generate = original_generate
    if calls["generate"] or any(not (r["status"] == "REFUSE" and r["refusal_class"] == r["class"] and r["exact_text"]) for r in refusals):
        raise RuntimeError(f"deterministic refusal check failed: {refusals}")
    if json.loads(no_explanation.output)["explanation_status"] != "NOT_GENERATED":
        raise RuntimeError("explain=False still produced an explanation")
    report["deterministic_refusals"] = {"classes_passed": len(refusals), "language_model_calls": calls["generate"], "cases": refusals}
    report["assessment_without_language_model"] = json.loads(no_explanation.output)["decision"]
    report["self_contained"] = {"separate_adapter_required": False, "external_runtime_services": []}
    report["runtime_seconds"] = time.perf_counter() - started
    (checkpoint / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    return report
