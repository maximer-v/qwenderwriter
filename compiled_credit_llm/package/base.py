"""Assemble the base checkpoint: the official Qwen3.5-0.8B tensors, untouched, plus the compiled scorer, TreeSHAP
and policy buffers under ``compiled_credit.*`` in the same safetensors file, with the checkpoint code beside it."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ..checkpoint import modeling_compiled_credit_llm as modeling

CHECKPOINT_FILES = ("configuration_compiled_credit_llm.py", "modeling_compiled_credit_llm.py", "routing.py")
BASE_REPO = "Qwen/Qwen3.5-0.8B"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes(order="C")).hexdigest()


def logical_state_hash(state) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def resolve_base(base: str | Path) -> Path:
    """A local directory with the official weights, or a Hub id to download."""
    path = Path(base)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=str(base)))


def _weight_files(base: Path) -> list[Path]:
    files = sorted(base.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors weights in {base}")
    return files


def _buffer_specs(compiled_state):
    top, experts = {}, {}
    for name, tensor in compiled_state.items():
        spec = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        if name.startswith("experts."):
            _, index, buffer_name = name.split(".", 2)
            experts.setdefault(int(index), {})[buffer_name] = spec
        else:
            top[name] = spec
    top["experts"] = [experts[index] for index in sorted(experts)]
    return top


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_base_checkpoint(base: str | Path, compiled_scorer: Path, compiled_treeshap: Path, compiled_policy: Path,
                          output: Path, checkpoint_code_dir: Path) -> dict:
    started = time.perf_counter()
    base = resolve_base(base)
    manifest = json.loads((compiled_scorer / "manifest.json").read_text("ascii"))
    scorer_receipt = json.loads((compiled_scorer / "receipt.json").read_text("ascii"))
    scoring_state = torch.load(compiled_scorer / "compiled_weights.pt", map_location="cpu", weights_only=True)
    if logical_state_hash(scoring_state) != scorer_receipt["compiled_sha256"]:
        raise RuntimeError("compiled scorer receipt mismatch")
    attribution_metadata = json.loads((compiled_treeshap / "treeshap_metadata.json").read_text("ascii"))
    attribution_receipt = json.loads((compiled_treeshap / "receipt.json").read_text("ascii"))
    reason_metadata = json.loads((compiled_treeshap / "reason_metadata.json").read_text("utf-8"))
    attribution_state = torch.load(compiled_treeshap / "treeshap_weights.pt", map_location="cpu", weights_only=True)
    if logical_state_hash(attribution_state) != attribution_receipt["weights_logical_sha256"]:
        raise RuntimeError("compiled TreeSHAP receipt mismatch")
    policy_specification = json.loads((compiled_policy / "policy_specification.json").read_text("utf-8"))
    policy_receipt = json.loads((compiled_policy / "receipt.json").read_text("ascii"))
    policy_state = torch.load(compiled_policy / "policy_weights.pt", map_location="cpu", weights_only=True)
    if logical_state_hash(policy_state) != policy_receipt["policy_state_sha256"]:
        raise RuntimeError("compiled policy receipt mismatch")
    if len(manifest["experts"]) != 1:
        raise RuntimeError("the checkpoint carries exactly one compiled tree-ensemble scorer")
    compiled_state = dict(scoring_state)
    compiled_state.update({f"attribution.{name}": tensor for name, tensor in attribution_state.items()})
    compiled_state.update({f"demo_policy.{name}": tensor for name, tensor in policy_state.items()})
    compiled_sha256 = logical_state_hash(compiled_state)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    weight_files = _weight_files(base)
    skip = {"config.json", "README.md", "model.safetensors.index.json", *(p.name for p in weight_files)}
    for source in base.iterdir():
        if source.is_file() and source.name not in skip:
            shutil.copy2(source, output / source.name)
    if (base / "README.md").exists():
        shutil.copy2(base / "README.md", output / "BASE_MODEL_CARD.md")
    shutil.copy2(base / "config.json", output / "base_config.json")
    for name in CHECKPOINT_FILES:
        shutil.copy2(checkpoint_code_dir / name, output / name)
    (output / "__init__.py").write_text("")

    grades = sorted(manifest["policy"]["grade_bands"], key=lambda value: value["minimum"])
    metadata = {
        "checkpoint_format_version": modeling.CHECKPOINT_FORMAT, "authority": modeling.AUTHORITY,
        "model_version": manifest["model_version"], "policy_version": policy_specification["policy_id"],
        "manifest_sha256": scorer_receipt["manifest_sha256"], "ir_sha256": scorer_receipt["ir_sha256"],
        "compiled_sha256": compiled_sha256, "scoring_compiled_sha256": scorer_receipt["compiled_sha256"],
        "attribution_sha256": attribution_receipt["weights_logical_sha256"], "evidence_contract": modeling.EVIDENCE_CONTRACT,
        "expert_count": len(manifest["experts"]), "feature_count": len(manifest["features"]),
        "feature_names": [value["name"] for value in manifest["features"]],
        "feature_minimums": [value["minimum"] for value in manifest["features"]],
        "feature_maximums": [value["maximum"] for value in manifest["features"]],
        "feature_scales": [value["scale"] for value in manifest["features"]],
        "missing_allowed": [value["missing_allowed"] for value in manifest["features"]],
        "output_scale": manifest["calibration"]["output_scale"], "grade_labels": [value["grade"] for value in grades],
        "buffer_specs": _buffer_specs(scoring_state),
        "attribution_tree_specs": [{"region_count": tree["region_count"], "features": [{"threshold_count": len(f["thresholds"])} for f in tree["features"]]}
                                   for tree in attribution_metadata["tree_specs"]],
        "attribution_local_tolerance_fixed": attribution_metadata["local_accuracy_tolerance_fixed_units"],
        "max_model_reasons": attribution_metadata["max_runtime_reasons"], "reason_metadata": reason_metadata,
        "policy_id": policy_specification["policy_id"], "policy_status": policy_specification["status"],
        "policy_sha256": policy_receipt["policy_state_sha256"],
        "policy_calculation_names": policy_specification["calculation_names"], "policy_flag_names": policy_specification["flag_names"],
        "risk_band_names": policy_specification["risk_band_names"], "policy_outcome_names": policy_specification["outcome_names"],
        "policy_trigger_names": policy_specification["trigger_names"], "policy_warning": policy_specification["warning"],
        "probability_warning": policy_specification["probability"]["warning"],
    }
    config = json.loads((base / "config.json").read_text("utf-8"))
    config.update({
        "model_type": "compiled_credit_llm", "architectures": ["CompiledCreditLLMForConditionalGeneration"],
        "auto_map": {"AutoConfig": "configuration_compiled_credit_llm.CompiledCreditLLMConfig",
                     "AutoModelForCausalLM": "modeling_compiled_credit_llm.CompiledCreditLLMForConditionalGeneration",
                     "AutoModelForImageTextToText": "modeling_compiled_credit_llm.CompiledCreditLLMForConditionalGeneration"},
        "compiled_credit": metadata,
    })
    write_json(output / "config.json", config)

    merged, identity = {}, {}
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as source:
            for key in source.keys():
                tensor = source.get_tensor(key)
                merged[key] = tensor
                identity[key] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": tensor_sha256(tensor)}
    if any(f"compiled_credit.{name}" in merged for name in compiled_state):
        raise RuntimeError("compiled-credit namespace collides with official tensors")
    for name, tensor in compiled_state.items():
        merged[f"compiled_credit.{name}"] = tensor.detach().cpu().contiguous()
    output_weight = output / "model.safetensors"
    save_file(merged, output_weight, metadata={"format": "pt"})
    del merged
    with safe_open(output_weight, framework="pt", device="cpu") as packaged:
        keys = set(packaged.keys())
        for key, expected in identity.items():
            tensor = packaged.get_tensor(key)
            if {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": tensor_sha256(tensor)} != expected:
                raise RuntimeError(f"official tensor identity changed: {key}")
    if keys != set(identity) | {f"compiled_credit.{name}" for name in compiled_state}:
        raise RuntimeError("packaged safetensor key set mismatch")
    write_json(output / "base_tensor_identity.json", {"algorithm": "sha256 over raw tensor bytes", "base_model": str(base),
                                                       "official_tensor_count": len(identity), "all_official_tensors_identical": True, "tensors": identity})
    shutil.copy2(compiled_scorer / "manifest.json", output / "compiled_scorer_manifest.json")
    shutil.copy2(compiled_scorer / "receipt.json", output / "compiled_scorer_receipt.json")
    shutil.copy2(compiled_scorer / "normalized_ir.json", output / "compiled_scorer_normalized_ir.json")
    shutil.copy2(compiled_treeshap / "treeshap_metadata.json", output / "compiled_treeshap_metadata.json")
    shutil.copy2(compiled_treeshap / "reason_metadata.json", output / "feature_labels.json")
    shutil.copy2(compiled_treeshap / "attribution_parity.json", output / "compiled_treeshap_parity.json")
    shutil.copy2(compiled_treeshap / "receipt.json", output / "compiled_treeshap_receipt.json")
    shutil.copy2(compiled_policy / "policy_specification.json", output / "policy_specification.json")
    shutil.copy2(compiled_policy / "development_bounds.json", output / "policy_development_bounds.json")
    shutil.copy2(compiled_policy / "compile_audit.json", output / "policy_compile_audit.json")
    shutil.copy2(compiled_policy / "receipt.json", output / "policy_receipt.json")
    files = [p for p in output.iterdir() if p.is_file() and p.name != "checkpoint_receipt.json"]
    receipt = {"format": modeling.CHECKPOINT_FORMAT, "authority": modeling.AUTHORITY, "official_tensor_count": len(identity),
               "compiled_state_sha256": compiled_sha256, "scoring_state_sha256": scorer_receipt["compiled_sha256"],
               "attribution_state_sha256": attribution_receipt["weights_logical_sha256"], "policy_state_sha256": policy_receipt["policy_state_sha256"],
               "compiled_tensor_count": len(compiled_state), "files": {p.name: sha256_file(p) for p in sorted(files)},
               "packaging_runtime_seconds": time.perf_counter() - started}
    write_json(output / "checkpoint_receipt.json", receipt)
    return receipt
