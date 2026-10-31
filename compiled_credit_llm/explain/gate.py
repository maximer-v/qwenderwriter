"""Free-generation gate: run the real route (raw request in, checkpoint validates, scores, prompts, verifies) with the
adapter or the merged weights, and report the verified-explanation rate plus the seven refusal classes."""
from __future__ import annotations

from collections import Counter
import json
import time
from pathlib import Path

import torch

from ..checkpoint.routing import refusal, route_request
from ..data.requests import RawTable, malformed_mutations


def load_model(checkpoint: Path, adapter: Path | None, device: str):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, trust_remote_code=True, dtype=torch.bfloat16)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model.to(device).eval(), tokenizer


def run_gate(model, tokenizer, rows: list[dict], table: RawTable, *, max_new_tokens: int = 320, log=print) -> dict:
    inner = model if hasattr(model, "score_credit") else model.base_model.model
    device = next(inner.parameters()).device
    metadata = inner.config.compiled_credit
    labels = {i["raw_feature_name"]: i["factual_label"] for i in metadata["reason_metadata"]}
    thresholds = [int(v) for v in inner.compiled_credit.demo_policy.thresholds.tolist()]
    age_od = int(inner.compiled_credit.demo_policy.dev_maximum[4].item()) + 1
    scorer = lambda f, m: inner.score_credit(f.to(device), m.to(device))
    calls = Counter(); started = time.monotonic(); violations = Counter(); results = []

    def generate(system: str, user: str) -> str:
        calls["generate"] += 1
        enc = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    with torch.inference_mode():
        for row in rows:
            request = table.request(row["source"] if "source" in row else row, age_od)
            before = calls["generate"]
            result = route_request(request, scorer, generate, metadata["feature_names"], labels, thresholds)
            payload = json.loads(result.output)
            problems = payload.get("explanation_violations", [])
            for p in problems:
                violations[p.split(":")[0]] += 1
            results.append({"example_id": row["example_id"], "evidence": result.evidence, "evidence_matches_dataset": result.evidence == row["evidence"],
                            "explanation": payload["explanation"], "explanation_status": payload["explanation_status"], "violations": problems,
                            "verified": payload["explanation_status"] == "VERIFIED", "generation_calls": calls["generate"] - before, "assessment": payload})
            log(f"{row['example_id'][-40:]}: {payload['explanation_status']}")
        first = table.request(rows[0]["source"] if "source" in rows[0] else rows[0], age_od)
        before = calls["generate"]; classes = []
        for cls, bad in malformed_mutations(first, results[0]["evidence"]):
            result = route_request(bad, scorer, generate, metadata["feature_names"], labels, thresholds)
            classes.append({"class": cls, "status": result.status, "refusal_class": result.refusal_class,
                            "exact_text": result.output == refusal(cls), "generation_invoked": result.generation_invoked})
    verified = sum(r["verified"] for r in results)
    words = [len((r["explanation"] or "").split()) for r in results]
    refusals_passed = sum(c["status"] == "REFUSE" and c["refusal_class"] == c["class"] and c["exact_text"] and not c["generation_invoked"] for c in classes)
    return {
        "rows": len(results), "verified": verified, "verified_rate": verified / max(len(results), 1),
        "evidence_matches_dataset": sum(r["evidence_matches_dataset"] for r in results), "violations": dict(violations),
        "mean_words": sum(words) / max(len(words), 1),
        "refusals": {"classes_passed": refusals_passed, "malformed_generation_calls": calls["generate"] - before, "cases": classes},
        "generation_mode": "model.generate", "runtime_seconds": time.monotonic() - started,
        "gate": "PASS" if refusals_passed == 7 and (calls["generate"] - before) == 0 else "FAIL",
        "results": results,
    }
