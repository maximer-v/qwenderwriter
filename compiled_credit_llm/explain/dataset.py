"""Explanation targets: a teacher model writes one paragraph from the same facts the checkpoint will show at serving
time, and the checkpoint's own verifier decides whether the paragraph may become a training target.

The teacher never sees feature codes, raw contributions or margins. A paragraph that fails the verifier is
re-sampled (up to ``tries`` times) and otherwise dropped; the row's decision, recourse and receipt are unaffected."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch

from ..checkpoint.routing import SYSTEM_PROMPT, explanation_prompt, recourse, validate_raw_request, verify_explanation
from ..data.requests import RawTable

TEACHER_SYSTEM = (
    "You write short, clean explanations of credit assessments for credit analysts. You are given fixed facts in JSON. "
    "Write exactly one paragraph of 4 to 7 sentences (70 to 160 words), plain prose, no headings, lists, markdown or "
    "quotes. Copy numbers exactly as given (use the percent forms given, e.g. probability_pct and the *_pct values with "
    "two decimals, counts as integers); do not compute or round new numbers. Name features only by the exact 'feature' "
    "strings in drivers. State the outcome word from decision.outcome and every phrase in decision.controlling_rules "
    "verbatim. Mention the risk_band word. Mention each data_quality phrase verbatim if any. Finish with one sentence "
    "saying the probability is an association from a benchmark model, not a cause and not a guarantee. Vary sentence "
    "structure naturally; do not use the words 'synthetic rule matrix'. Never use quotation marks, parentheses or feature "
    "codes such as PAY_0, no currency signs; write feature names and rule phrases directly into the sentences as ordinary "
    "words. Do not address the reader (no 'remember', 'note that'). Amounts are in the benchmark's own currency units; "
    "state them without a symbol. Embed each controlling-rule phrase verbatim but naturally inside a sentence, for example "
    "'because a required input is missing' or 'since severe delinquency is present'; never write 'the rule X' or 'at the "
    "hard level'. Do not copy the decision.note text; paraphrase it in your own words. Each sentence must say something new; "
    "state the controlling rules once, not twice. The probability is the model's probability of DEFAULT (a bad outcome): "
    "higher means riskier; never describe it as a probability of approval, repayment, a good outcome or of the decision. "
    "Use the proximity facts for one or two sentences: if a better outcome is reachable, say how close the case is "
    "(use the proximity word), name that next better outcome exactly as given, and describe what_would_change_it in "
    "plain words with the exact numbers given; if the status says not reachable by behaviour, say that the route cannot "
    "be changed by the borrower's behaviour and why (the blockers), in plain words; if already best, say no further "
    "improvement applies. Do not invent any other change."
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_temperature() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
        return max(int(x) for x in out.split())
    except Exception:
        return None


def thermal_guard(pause: int = 78, resume: int = 70, abort: int = 84) -> None:
    t = gpu_temperature()
    if t is None:
        return
    if t >= abort:
        raise RuntimeError(f"thermal abort at {t} C")
    while t >= pause:
        time.sleep(15); t = gpu_temperature()
        if t >= abort:
            raise RuntimeError(f"thermal abort at {t} C")
        if t <= resume:
            break


def prepare_prompts(rows: list[dict], model, table: RawTable) -> list[dict]:
    """Attach request, recourse and the serving-time prompt to every row (CPU, compiled scorer only)."""
    metadata = model.config.compiled_credit
    labels = {i["raw_feature_name"]: i["factual_label"] for i in metadata["reason_metadata"]}
    thresholds = [int(v) for v in model.compiled_credit.demo_policy.thresholds.tolist()]
    age_od = int(model.compiled_credit.demo_policy.dev_maximum[4].item()) + 1
    out = []
    for row in rows:
        request = table.request(row, age_od)
        _, values, missing = validate_raw_request(request)
        rec = recourse(values, missing, model.score_credit, thresholds, feature_labels=labels)
        out.append({**row, "recourse": rec, "prompt": explanation_prompt(row["evidence"], labels, rec), "feature_labels": labels})
    return out


def write_targets(rows_by_partition: dict[str, list[dict]], teacher_id: str, output: Path, *, tries: int = 4,
                  temperature: float = 0.7, top_p: float = 0.9, batch_size: int = 16, max_new_tokens: int = 300,
                  limit: int | None = None, log=print) -> dict:
    """Generate and verify explanation targets with a teacher model on the GPU; write JSONL partitions and a manifest."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(teacher_id); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = AutoModelForCausalLM.from_pretrained(teacher_id, dtype=torch.bfloat16).to(device).eval()
    stats = Counter(); started = time.time(); outputs: dict[str, list[dict]] = {}
    for partition, rows in rows_by_partition.items():
        if limit:
            rows = rows[:limit]
        pending = list(rows)
        done: dict[str, dict] = {}
        for attempt in range(1, tries + 1):
            if not pending:
                break
            temp = temperature if attempt == 1 else min(1.0, temperature + 0.1 * attempt)
            next_pending = []
            for start in range(0, len(pending), batch_size):
                thermal_guard()
                batch = pending[start:start + batch_size]
                texts = [tok.apply_chat_template([{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": r["prompt"]}],
                                                 tokenize=False, add_generation_prompt=True) for r in batch]
                enc = tok(texts, return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    gen = teacher.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temp, top_p=top_p,
                                           pad_token_id=tok.pad_token_id)
                for row, seq in zip(batch, gen):
                    text = " ".join(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split())
                    problems = verify_explanation(text, row["evidence"], row["feature_labels"], row["recourse"])
                    stats[f"attempt{attempt}_{'ok' if not problems else 'fail'}"] += 1
                    if problems:
                        for p in problems:
                            stats["violation:" + p.split(":")[0]] += 1
                        next_pending.append(row)
                        continue
                    done[row["example_id"]] = {
                        "example_id": row["example_id"], "partition": partition, "evidence": row["evidence"], "recourse": row["recourse"],
                        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]},
                                     {"role": "assistant", "content": text}],
                        "source": {k: row[k] for k in ("opaque_row_id", "row_index", "source_split", "variant")} | {"teacher": teacher_id, "teacher_attempt": attempt},
                    }
                log(f"{partition}: attempt {attempt}, {len(done)}/{len(rows)} verified, {time.time() - started:.0f}s")
            pending = next_pending
        outputs[partition] = [done[r["example_id"]] for r in rows if r["example_id"] in done]
        stats[f"{partition}_kept"] = len(outputs[partition]); stats[f"{partition}_dropped"] = len(rows) - len(outputs[partition])
    output.mkdir(parents=True, exist_ok=True)
    for partition, rows in outputs.items():
        (output / f"{partition}.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows) + "\n", "utf-8")
    manifest = {
        "schema_version": "explanation-targets/v1", "teacher": teacher_id,
        "sampling": {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens, "tries": tries},
        "counts": {p: len(r) for p, r in outputs.items()}, "stats": dict(sorted(stats.items())),
        "verification": "every target passed routing.verify_explanation against its own evidence and recourse",
        "recourse": "computed by the compiled scorer per row and shown to the teacher as proximity facts",
        "refusal_examples": 0, "test_rows_used": 0, "test_labels_used": False,
        "artifact_hashes": {f"{p}.jsonl": sha(output / f"{p}.jsonl") for p in outputs},
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    return manifest
