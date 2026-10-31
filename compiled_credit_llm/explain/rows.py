"""Choose the development rows the explanation writer learns from and is tested on.

Rows come from the train split (training rows) and the validation split (validation, rare-trigger and smoke rows);
the test split is never read. Variants create the cases a natural row cannot: explicit nulls and contract
violations (invalid input), out-of-development ages (manual review) and natural rows carrying rare triggers."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from ..checkpoint.routing import OUTCOME_CODES, TRIGGER_CODES, evidence_from_score, validate_raw_request
from ..data.requests import RawTable, VARIANT_DESCRIPTIONS

SAMPLE_SEED = "compiled-credit-llm/rows/v1"
DEFAULT_PLAN = {
    "train": {"N_per_outcome": 200, "M": 100, "C": 100, "O": 100, "RU_N": 150, "RU_O": 60, "DZ_N": 40, "DX_N": 60},
    "validation": {"N_per_outcome": 40, "M": 20, "C": 20, "O": 20, "RU_N": 10, "RU_O": 6, "DZ_N": 3, "DX_N": 5},
}
RARE_VARIANTS = ("RU_N", "RU_O", "DZ_N", "DX_N")


def _order(split: str, opaque: str) -> str:
    return hashlib.sha256(f"{SAMPLE_SEED}:{split}:{opaque}".encode()).hexdigest()


def _score_all(model, table: RawTable, indices: list[int]) -> tuple[list[str], list[set[str]]]:
    outcomes, triggers = [], []
    features = torch.tensor([table.rows[i] for i in indices], dtype=torch.int64)
    with torch.inference_mode():
        for start in range(0, len(indices), 2000):
            block = features[start:start + 2000]
            result = model.score_credit(block, torch.zeros_like(block, dtype=torch.bool))
            for o, t in zip(result.policy_outcome_index.tolist(), result.policy_triggers.tolist()):
                outcomes.append(OUTCOME_CODES[int(o)])
                triggers.append({code for code, flag in zip(TRIGGER_CODES, t) if flag})
    return outcomes, triggers


def sample_rows(model, table: RawTable, plan: dict = DEFAULT_PLAN) -> dict[str, list[dict]]:
    """Return {partition: [row records]} for train, validation, rare_eval and smoke."""
    age_od = int(model.compiled_credit.demo_policy.dev_maximum[4].item()) + 1
    partitions: dict[str, list[dict]] = {"train": [], "validation": []}
    for split, counts in plan.items():
        indices = [i for i, s in enumerate(table.split) if s == split]
        indices.sort(key=lambda i: _order(split, table.opaque[i]))
        outcomes, triggers = _score_all(model, table, indices)
        taken: set[int] = set()
        used = Counter()

        def record(index: int, variant: str) -> dict:
            source = {"opaque_row_id": table.opaque[index], "row_index": index, "source_split": split, "variant": variant}
            request = table.request(source, age_od)
            error, values, missing = validate_raw_request(request)
            if error:
                raise RuntimeError(error)
            with torch.inference_mode():
                evidence = evidence_from_score(model.score_credit(torch.tensor([values]), torch.tensor([missing])))
            return {"example_id": f"{split}-{variant}-{table.opaque[index]}", "partition": split, **source, "evidence": evidence}

        def take(variant: str, cap: int, predicate):
            for position, index in enumerate(indices):
                if used[variant] >= cap:
                    break
                if index in taken or not predicate(outcomes[position], triggers[position]):
                    continue
                partitions[split].append(record(index, variant)); taken.add(index); used[variant] += 1

        for outcome in ("DD", "MR", "AC", "AS"):
            take("N", counts["N_per_outcome"], lambda o, t, want=outcome: o == want)
            used["N"] = 0
        take("M", counts["M"], lambda o, t: True)
        take("C", counts["C"], lambda o, t: True)
        take("O", counts["O"], lambda o, t: True)
        take("RU_N", counts["RU_N"], lambda o, t: "RU" in t)
        take("RU_O", counts["RU_O"], lambda o, t: "RU" in t)
        take("DZ_N", counts["DZ_N"], lambda o, t: "DZ" in t)
        take("DX_N", counts["DX_N"], lambda o, t: "DX" in t)
    validation = partitions["validation"]
    rare = [r for r in validation if r["variant"] in RARE_VARIANTS]
    smoke = []
    for outcome in OUTCOME_CODES:
        smoke.append(next(r for r in validation if f"|o={outcome}|" in r["evidence"]))
    out = {"train": partitions["train"], "validation": [r for r in validation if r["variant"] not in RARE_VARIANTS],
           "rare_eval": [{**r, "partition": "rare_eval"} for r in rare], "smoke": [{**r, "partition": "smoke"} for r in smoke]}
    ids = [r["opaque_row_id"] for name in ("train", "validation", "rare_eval") for r in out[name]]
    if len(ids) != len(set(ids)):
        raise RuntimeError("a source row appears in more than one of train / validation / rare_eval")
    return out


def write_rows(rows: dict[str, list[dict]], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for name in ("train", "validation", "rare_eval", "smoke"):
            for r in rows[name]:
                stream.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {name: {"rows": len(items), "variants": dict(Counter(r["variant"] for r in items)),
                      "outcomes": dict(Counter(r["evidence"].split("|o=")[1][:2] for r in items))} for name, items in rows.items()}
    summary["variant_descriptions"] = VARIANT_DESCRIPTIONS
    summary["sample_seed"] = SAMPLE_SEED
    summary["test_rows_used"] = 0
    return summary


def read_rows(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "validation": [], "rare_eval": [], "smoke": []}
    for line in path.read_text("utf-8").splitlines():
        if line:
            r = json.loads(line); out[r["partition"]].append(r)
    return out
