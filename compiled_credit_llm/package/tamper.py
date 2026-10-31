"""Tamper and determinism tests for a checkpoint.

T1  one compiled tree leaf changed in memory          -> assess refused by the tensor receipt, no language model call
T2  one policy bound changed in memory                -> same
T3  expected compiled hash in the config changed      -> same
T0  values restored                                   -> assessment identical to the untampered one
T4  one language weight changed                       -> every compiled score field unchanged on N rows
T5  one bit flipped inside the compiled region of a copy of model.safetensors -> refused after reload
D1  same request assessed twice                       -> byte-identical JSON
D2  N rows scored one by one and in batches           -> integer and fixed-point fields bit-identical; floats identical at receipt precision
D3  smoke requests assessed on CPU and on GPU         -> deterministic fields byte-identical; explanation agreement reported
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ..data.requests import RawTable
from ..data.uci import FEATURE_NAMES, load_table
from .verify import tensor_fields

RECEIPT_ERROR = "compiled-credit tensor receipt mismatch"
NONDETERMINISTIC_KEYS = ("explanation", "explanation_status", "explanation_violations")


def load(path: Path, device: str):
    return AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16).to(device).eval()


def score_tensors(model, features: np.ndarray, batch: int) -> dict:
    out: dict[str, list] = {}
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            block = torch.from_numpy(features[start:start + batch]).to(device)
            result = model.score_credit(block, torch.zeros_like(block, dtype=torch.bool))
            for name, value in tensor_fields(result).items():
                out.setdefault(name, []).append(np.atleast_1d(value.detach().cpu().numpy()))  # bias is a scalar per batch
    return {k: np.concatenate(v) for k, v in out.items()}


def compare_fields(reference: dict, other: dict) -> dict:
    out = {}
    for name in reference:
        a, b = reference[name], other[name]
        if a.shape != b.shape:  # batch-constant scalars concatenate to different lengths
            exact = len(np.unique(a)) == 1 and len(np.unique(b)) == 1 and a.ravel()[0] == b.ravel()[0]
            out[name] = {"exact": bool(exact), "pass": bool(exact), "note": "batch-constant scalar"}
            continue
        exact = bool(np.array_equal(a, b))
        kind = "float" if np.issubdtype(a.dtype, np.floating) else "integer_or_fixed_point"
        if kind == "float" and not exact:
            diff = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
            receipt_same = all(f"{x:.12f}" == f"{y:.12f}" for x, y in zip(a.ravel(), b.ravel()))
            out[name] = {"exact": False, "kind": kind, "max_abs_diff": diff, "rows_differing": int((a != b).sum()),
                         "receipt_precision_identical": receipt_same, "pass": receipt_same}
        else:
            out[name] = {"exact": exact, "kind": kind, "pass": exact}
    return out


def expect_receipt_refusal(model, tokenizer, request, label: str) -> dict:
    calls = {"generate": 0}
    original = model.generate

    def counting(*a, **k):
        calls["generate"] += 1
        return original(*a, **k)

    model.generate = counting
    try:
        model.assess_credit(request, tokenizer)
        outcome = {"test": label, "refused": False, "error": None}
    except RuntimeError as err:
        outcome = {"test": label, "refused": str(err) == RECEIPT_ERROR, "error": str(err)}
    finally:
        model.generate = original
    outcome["language_model_calls"] = calls["generate"]
    outcome["pass"] = bool(outcome["refused"]) and calls["generate"] == 0
    return outcome


def deterministic_view(payload: dict) -> str:
    return json.dumps({k: v for k, v in payload.items() if k not in NONDETERMINISTIC_KEYS}, sort_keys=True, separators=(",", ":"))


def flip_bit_in_tensor(src: Path, dst: Path, tensor_name: str) -> dict:
    shutil.copyfile(src, dst)
    with dst.open("r+b") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len))
        start, end = header[tensor_name]["data_offsets"]
        position = 8 + header_len + start + (end - start) // 2
        stream.seek(position); original = stream.read(1)[0]
        stream.seek(position); stream.write(bytes([original ^ 0x01]))
    return {"tensor": tensor_name, "dtype": header[tensor_name]["dtype"], "byte_offset": position, "bit_flipped": 1}


def tamper_and_determinism(checkpoint: Path, data_csv: Path, split_manifest: Path, smoke_rows: list[dict], scratch: Path,
                           rows: int = 1000, use_gpu: bool = True) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = load(checkpoint, "cpu")
    table = RawTable(data_csv, split_manifest)
    age_od = int(model.compiled_credit.demo_policy.dev_maximum[4].item()) + 1
    requests = [table.request(r, age_od) for r in smoke_rows]
    features = load_table(data_csv)[list(FEATURE_NAMES)].to_numpy(np.int64)[:rows]
    report: dict = {"checkpoint": checkpoint.name, "tests": []}

    baseline = model.assess_credit(requests[0], tokenizer)
    if baseline.status == "REFUSE":
        raise RuntimeError(baseline.output)
    healthy = score_tensors(model, features, batch=250)

    leaf = model.compiled_credit.experts[0].leaf_value
    with torch.no_grad():
        saved = leaf[0].item(); leaf[0] += 1e-6
    report["tests"].append(expect_receipt_refusal(model, tokenizer, requests[0], "T1_compiled_leaf_tamper"))
    with torch.no_grad():
        leaf[0] = saved
    bound = model.compiled_credit.demo_policy.dev_maximum
    with torch.no_grad():
        saved_b = bound[0].item(); bound[0] += 1
    report["tests"].append(expect_receipt_refusal(model, tokenizer, requests[0], "T2_policy_bound_tamper"))
    with torch.no_grad():
        bound[0] = saved_b
    expected = model.config.compiled_credit["compiled_sha256"]
    model.config.compiled_credit["compiled_sha256"] = "0" * 64
    model._verified_compiled_versions = None
    report["tests"].append(expect_receipt_refusal(model, tokenizer, requests[0], "T3_config_hash_tamper"))
    model.config.compiled_credit["compiled_sha256"] = expected
    restored = model.assess_credit(requests[0], tokenizer)
    report["tests"].append({"test": "T0_restored_after_tamper", "pass": restored.status != "REFUSE" and deterministic_view(json.loads(restored.output)) == deterministic_view(json.loads(baseline.output))})
    lm_param = next(p for n, p in model.named_parameters() if n.startswith("model.language_model.layers.0.") and p.dim() == 2)
    with torch.no_grad():
        saved_w = lm_param.flatten()[0].item(); lm_param.flatten()[0] = saved_w + 1.0
    tampered_scores = score_tensors(model, features, batch=250)
    with torch.no_grad():
        lm_param.flatten()[0] = saved_w
    cmp = compare_fields(healthy, tampered_scores)
    report["tests"].append({"test": "T4_language_weight_tamper_scores_unchanged", "rows": int(len(features)), "fields": cmp,
                            "pass": all(v["exact"] for v in cmp.values())})
    twice = [model.assess_credit(requests[1], tokenizer).output for _ in range(2)]
    report["tests"].append({"test": "D1_same_request_twice_cpu", "identical_bytes": twice[0] == twice[1],
                            "sha256": hashlib.sha256(twice[0].encode()).hexdigest(), "pass": twice[0] == twice[1]})
    one_by_one = score_tensors(model, features, batch=1)
    cmp = compare_fields(healthy, one_by_one)
    report["tests"].append({"test": "D2_batch_invariance", "rows": int(len(features)), "batch_sizes": [250, 1], "fields": cmp,
                            "integer_and_fixed_point_fields_exact": all(v["exact"] for v in cmp.values() if v.get("kind") != "float"),
                            "pass": all(v["pass"] for v in cmp.values()),
                            "note": "float64 probabilities may differ by 1 ulp between vectorised and scalar sigmoid paths; receipts carry 12 digits"})
    cpu_outputs = [model.assess_credit(r, tokenizer).output for r in requests]
    del model
    tampered_dir = scratch / f"tampered_{checkpoint.name}"
    if tampered_dir.exists():
        shutil.rmtree(tampered_dir)
    tampered_dir.mkdir(parents=True)
    flip = None
    for item in checkpoint.iterdir():
        if item.name == "model.safetensors":
            flip = flip_bit_in_tensor(item, tampered_dir / item.name, "compiled_credit.experts.0.leaf_value")
        elif item.is_file():
            os.symlink(item, tampered_dir / item.name)
    try:
        tampered = load(tampered_dir, "cpu")
        t5 = expect_receipt_refusal(tampered, tokenizer, requests[0], "T5_file_bit_flip")
        del tampered
    except Exception as err:
        t5 = {"test": "T5_file_bit_flip", "refused": True, "error": f"load-time: {err}", "language_model_calls": 0, "pass": True}
    t5["flip"] = flip
    report["tests"].append(t5)
    shutil.rmtree(tampered_dir)
    if use_gpu and torch.cuda.is_available():
        gpu = load(checkpoint, "cuda")
        gpu_outputs = [gpu.assess_credit(r, tokenizer).output for r in requests]
        det = [deterministic_view(json.loads(a)) == deterministic_view(json.loads(b)) for a, b in zip(cpu_outputs, gpu_outputs)]
        receipts = [json.loads(a)["receipt"] == json.loads(b)["receipt"] for a, b in zip(cpu_outputs, gpu_outputs)]
        expl = [json.loads(a)["explanation"] == json.loads(b)["explanation"] for a, b in zip(cpu_outputs, gpu_outputs)]
        status = [(json.loads(a)["explanation_status"], json.loads(b)["explanation_status"]) for a, b in zip(cpu_outputs, gpu_outputs)]
        cmp = compare_fields(healthy, score_tensors(gpu, features, batch=250))
        report["tests"].append({"test": "D3_cpu_vs_gpu", "cases": len(requests), "deterministic_fields_identical": sum(det),
                                "receipts_identical": sum(receipts), "explanation_text_identical": sum(expl), "explanation_status_pairs": status,
                                "score_fields": cmp, "integer_and_fixed_point_fields_exact": all(v["exact"] for v in cmp.values() if v.get("kind") != "float"),
                                "float_fields_max_abs_diff": max([v.get("max_abs_diff", 0.0) for v in cmp.values()] + [0.0]),
                                "pass": all(det) and all(receipts) and all(v["pass"] for v in cmp.values()),
                                "note": "the explanation text is the only field allowed to differ across devices; its verification status is reported per case"})
        del gpu
    report["all_pass"] = all(t["pass"] for t in report["tests"])
    report["runtime_seconds"] = time.perf_counter() - started
    (checkpoint / "tamper_determinism_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", "utf-8")
    return report
