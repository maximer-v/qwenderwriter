"""Merge the selected explanation adapter into the base checkpoint and write the final self-contained checkpoint."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .base import CHECKPOINT_FILES, logical_state_hash, sha256_file, write_json

EXAMPLE = '''import json
from transformers import AutoModelForImageTextToText, AutoTokenizer

model = AutoModelForImageTextToText.from_pretrained(".", trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
request = {"action": "ASSESS", "credit": {
    "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 39,
    "PAY_0": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 0,
    "BILL_AMT1": 117868, "BILL_AMT2": 114483, "BILL_AMT3": 118855, "BILL_AMT4": 125357, "BILL_AMT5": 121853, "BILL_AMT6": 124731,
    "PAY_AMT1": 0, "PAY_AMT2": 6216, "PAY_AMT3": 10000, "PAY_AMT4": 0, "PAY_AMT5": 5000, "PAY_AMT6": 4552}}
result = model.assess_credit(request, tokenizer)   # validate -> compiled score, TreeSHAP, policy, recourse -> one verified paragraph
print(result.status)                                # OK, or REFUSE for malformed input (no language model call)
assessment = json.loads(result.output)
print(assessment["decision"], assessment["model"], assessment["recourse"]["status"], assessment["explanation_status"])
print(assessment["explanation"])
'''


def merge_checkpoint(base_checkpoint: Path, adapter: Path, gate_report: Path, output: Path, checkpoint_code_dir: Path,
                     model_name: str, readme_text: str, model_card_text: str) -> dict:
    started = time.perf_counter()
    torch.set_num_threads(8)
    if output.exists():
        raise RuntimeError(f"refusing to replace an existing checkpoint: {output}")
    gate = json.loads(gate_report.read_text("utf-8"))
    if gate.get("gate") != "PASS":
        raise RuntimeError("the selected adapter did not pass the free-generation gate")
    adapter_hash = sha256_file(adapter / "adapter_model.safetensors")
    base = AutoModelForImageTextToText.from_pretrained(base_checkpoint, trust_remote_code=True, low_cpu_mem_usage=True, dtype=torch.bfloat16)
    expected = base.config.compiled_credit["compiled_sha256"]
    if logical_state_hash(base.compiled_credit.state_dict()) != expected:
        raise RuntimeError("base compiled-state hash mismatch")
    wrapped = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    targeted = sorted(name for name, module in wrapped.named_modules() if hasattr(module, "lora_A"))
    if not targeted or any("compiled_credit" in name for name in targeted):
        raise RuntimeError("adapter scope is empty or touches compiled-credit modules")
    merged = wrapped.merge_and_unload(safe_merge=True, progressbar=False)
    if any("lora_" in n or "base_layer" in n for n in merged.state_dict()):
        raise RuntimeError("PEFT tensors remain after merge")
    if logical_state_hash(merged.compiled_credit.state_dict()) != expected:
        raise RuntimeError("compiled state changed during merge")
    merged.config._name_or_path = model_name
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{model_name}-", dir=output.parent))
    try:
        merged.save_pretrained(temp_dir, safe_serialization=True, max_shard_size="5GB")
        output_weight = temp_dir / "model.safetensors"
        tensors = {}
        with safe_open(output_weight, framework="pt", device="cpu") as stream:
            for key in stream.keys():
                tensors[key] = stream.get_tensor(key)
        with safe_open(base_checkpoint / "model.safetensors", framework="pt", device="cpu") as stream:
            for key in sorted(set(stream.keys()) - set(tensors)):  # tensors the model class does not load (e.g. MTP head)
                tensors[key] = stream.get_tensor(key)
        replacement = temp_dir / "model.safetensors.replacement"
        save_file(tensors, replacement, metadata={"format": "pt"})
        replacement.replace(output_weight)
        del tensors
        AutoTokenizer.from_pretrained(base_checkpoint, trust_remote_code=True).save_pretrained(temp_dir)
        merged.generation_config.save_pretrained(temp_dir)
        config = json.loads((temp_dir / "config.json").read_text("utf-8"))
        config["model_type"] = "compiled_credit_llm"
        config["auto_map"] = {"AutoConfig": "configuration_compiled_credit_llm.CompiledCreditLLMConfig",
                              "AutoModelForCausalLM": "modeling_compiled_credit_llm.CompiledCreditLLMForConditionalGeneration",
                              "AutoModelForImageTextToText": "modeling_compiled_credit_llm.CompiledCreditLLMForConditionalGeneration"}
        write_json(temp_dir / "config.json", config)
        for item in base_checkpoint.iterdir():
            if item.is_file() and item.suffix in {".json", ".md", ".jinja", ".txt"} and not (temp_dir / item.name).exists():
                shutil.copy2(item, temp_dir / item.name)
        for name in CHECKPOINT_FILES:
            shutil.copy2(checkpoint_code_dir / name, temp_dir / name)
        (temp_dir / "__init__.py").write_text("")
        shutil.copy2(gate_report, temp_dir / "adapter_selection_gate.json")
        (temp_dir / "inference_example.py").write_text(EXAMPLE, "utf-8")
        (temp_dir / "README.md").write_text(readme_text, "utf-8")
        (temp_dir / "MODEL_CARD.md").write_text(model_card_text, "utf-8")
        write_json(temp_dir / "deployment_manifest.json", {
            "model_name": model_name, "format": "compiled-credit-llm/v1", "base_checkpoint": base_checkpoint.name,
            "explanation_adapter_sha256": adapter_hash, "merge_method": "PeftModel.merge_and_unload(safe_merge=True)",
            "targeted_linear_modules": len(targeted), "compiled_state_sha256": expected, "separate_adapter_required": False,
            "runtime_external_dependencies": "none (no XGBoost, SHAP, policy, renderer or verifier service)",
            "precision_note": "LoRA deltas folded into BF16 weights; generation quality is re-checked on the merged weights by the gate",
        })
        temp_dir.rename(output)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {"output": str(output), "compiled_state_sha256": expected, "targeted_linear_modules": len(targeted),
            "adapter_sha256": adapter_hash, "runtime_seconds": time.perf_counter() - started}
