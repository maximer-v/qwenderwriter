"""LoRA adaptation of the language backbone on verified explanation targets. Compiled buffers are frozen and their
hash is checked before and after. Teacher-forced loss is informational; selection uses free generation (gate.py)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

DEVELOPMENT_SPLITS = {"train", "validation", "calibration"}
DEFAULT_TARGET_MODULES = (
    r"^model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))$"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def logical_state_hash(state) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode() + b"\0"); digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode() + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def input_ids(value) -> list[int]:
    """Normalise the chat-template return value to one flat token list."""
    if hasattr(value, "keys") and "input_ids" in value.keys():
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise RuntimeError("chat template returned a batched token sequence")
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(token, int) for token in value):
        raise RuntimeError("chat template did not return a flat integer token sequence")
    return value


class ChatDataset(Dataset):
    """Supervised chat rows; loss only on the assistant turn."""

    def __init__(self, path: Path, tokenizer, max_length: int):
        self.records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
        if not self.records:
            raise RuntimeError(f"dataset is empty: {path.name}")
        if any(r.get("source", {}).get("source_split") not in DEVELOPMENT_SPLITS for r in self.records):
            raise RuntimeError("training rows must come from development splits")
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        messages = record["messages"]
        full = input_ids(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        prompt = input_ids(self.tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True))
        if len(full) > self.max_length:
            raise RuntimeError(f"{record['example_id']} exceeds max_sequence_length {self.max_length}")
        if not prompt or len(prompt) >= len(full) or full[:len(prompt)] != prompt:
            raise RuntimeError(f"{record['example_id']} chat-template prompt is not a strict prefix")
        return {"input_ids": full, "labels": [-100] * len(prompt) + full[len(prompt):], "attention_mask": [1] * len(full)}


class RightPadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        width = max(len(item["input_ids"]) for item in features)
        output = {"input_ids": [], "labels": [], "attention_mask": []}
        for item in features:
            pad = width - len(item["input_ids"])
            output["input_ids"].append(item["input_ids"] + [self.pad_token_id] * pad)
            output["labels"].append(item["labels"] + [-100] * pad)
            output["attention_mask"].append(item["attention_mask"] + [0] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in output.items()}


class ResourceGuard(TrainerCallback):
    """Wall-clock and GPU-temperature guard: pause at ``pause`` C, resume at ``resume`` C, abort at ``abort`` C."""

    def __init__(self, limits: dict):
        self.limits = limits; self.started = None; self.paused = 0.0

    @staticmethod
    def temperature() -> int | None:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5, check=True).stdout
            return max(int(x) for x in out.split())
        except Exception:
            return None

    def _wall_clock(self):
        if self.started is not None and time.monotonic() - self.started > self.limits["maximum_wall_clock_minutes"] * 60:
            raise RuntimeError("training exceeded the wall-clock envelope")

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic(); return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._wall_clock()
        t = self.temperature()
        if t is None:
            return control
        if t >= self.limits["thermal_abort_celsius"]:
            raise RuntimeError(f"thermal abort at {t} C")
        while t >= self.limits["thermal_pause_celsius"]:
            time.sleep(20); self.paused += 20
            if self.paused > self.limits.get("thermal_pause_budget_seconds", 3600):
                raise RuntimeError("thermal pause budget exceeded")
            self._wall_clock(); t = self.temperature()
            if t is None or t <= self.limits["thermal_resume_celsius"]:
                break
            if t >= self.limits["thermal_abort_celsius"]:
                raise RuntimeError(f"thermal abort at {t} C")
        return control


def train_explainer(config: dict, base_checkpoint: Path, dataset_dir: Path, output: Path) -> dict:
    trainer_config = config["trainer"]; lora = config["lora"]; limits = config["resource_limits"]
    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text("utf-8"))
    if manifest.get("refusal_examples") != 0 or manifest.get("test_rows_used") != 0:
        raise RuntimeError("dataset governance failed")
    train_path, loss_path = dataset_dir / "train.jsonl", dataset_dir / "validation.jsonl"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to reuse a non-empty adapter output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(int(trainer_config["seed"])); torch.manual_seed(int(trainer_config["seed"]))
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(base_checkpoint, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    compiled_before = logical_state_hash(model.compiled_credit.state_dict())
    if model.config.compiled_credit["compiled_sha256"] != compiled_before:
        raise RuntimeError("compiled scorer identity mismatch before training")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(r=int(lora["rank"]), lora_alpha=int(lora["alpha"]), lora_dropout=float(lora["dropout"]),
                                             bias="none", task_type=TaskType.CAUSAL_LM, target_modules=lora.get("target_modules_regex", DEFAULT_TARGET_MODULES)))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or "compiled_credit" in n for n, _ in trainable):
        raise RuntimeError("a non-adapter or compiled tensor became trainable")
    trainable_count = sum(p.numel() for _, p in trainable)
    target_modules = {n.rsplit(".lora_", 1)[0] for n, _ in trainable}
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    max_len = int(trainer_config["max_sequence_length"])
    train_dataset = ChatDataset(train_path, tokenizer, max_len)
    eval_dataset = ChatDataset(loss_path, tokenizer, max_len)
    micro, accumulation = int(trainer_config["per_device_train_batch_size"]), int(trainer_config["gradient_accumulation_steps"])
    steps = math.ceil(len(train_dataset) / (micro * accumulation)) * int(trainer_config.get("epochs", 1))
    arguments = TrainingArguments(
        output_dir=str(output), max_steps=steps, per_device_train_batch_size=micro, per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation, learning_rate=float(trainer_config["learning_rate"]),
        weight_decay=float(trainer_config["weight_decay"]), warmup_steps=int(trainer_config["warmup_steps"]),
        lr_scheduler_type=trainer_config["lr_scheduler_type"], max_grad_norm=float(trainer_config["max_grad_norm"]), bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, use_cache=False,
        eval_strategy="steps", eval_steps=int(trainer_config["eval_steps"]), save_strategy="steps", save_steps=int(trainer_config["save_steps"]),
        save_total_limit=int(trainer_config["save_total_limit"]), logging_steps=int(trainer_config["logging_steps"]), load_best_model_at_end=False,
        optim="adamw_torch", report_to=[], seed=int(trainer_config["seed"]), data_seed=int(trainer_config["seed"]),
        dataloader_num_workers=2, remove_unused_columns=False,
    )
    guard = ResourceGuard(limits)
    trainer = Trainer(model=model, args=arguments, train_dataset=train_dataset, eval_dataset=eval_dataset,
                      data_collator=RightPadCollator(tokenizer.pad_token_id), processing_class=tokenizer, callbacks=[guard])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    trainer.train()
    trainer.save_model(str(output / "final_adapter"))
    compiled_after = logical_state_hash(model.base_model.model.compiled_credit.state_dict())
    if compiled_after != compiled_before:
        raise RuntimeError("compiled scorer tensors changed during adaptation")
    report = {
        "base_checkpoint": str(base_checkpoint), "dataset_manifest_sha256": sha256_file(dataset_dir / "dataset_manifest.json"),
        "train_rows": len(train_dataset), "loss_rows": len(eval_dataset), "steps": steps,
        "compiled_state_sha256_before": compiled_before, "compiled_state_sha256_after": compiled_after,
        "trainable_parameter_count": trainable_count, "target_module_count": len(target_modules),
        "lora": lora, "trainer": trainer_config, "log_history": trainer.state.log_history,
        "runtime_seconds": time.monotonic() - started, "thermal_pause_seconds": guard.paused,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else None,
        "selection_policy": "free-generation gate only; teacher-forced loss is informational",
    }
    (output / "adapter_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
