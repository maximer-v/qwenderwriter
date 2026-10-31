"""Checkpoint-owned request routing, deterministic assessment assembly, explanation verification and recourse.

This file ships inside the checkpoint directory next to the modeling code. Everything here runs before or
after the language model, never inside it:

* ``validate_raw_request`` checks the request envelope and the 23 raw fields. Malformed requests get an exact
  refusal and never reach the compiled scorer or the language model.
* ``evidence_from_score`` serialises one verified compiled-scorer result into a canonical evidence line.
* ``facts`` / ``assemble_assessment`` build the JSON assessment from that evidence alone.
* ``recourse`` searches the smallest behavioural change that reaches a better policy outcome, re-scoring every
  candidate with the compiled scorer.
* ``verify_explanation`` checks the one paragraph the language model writes against the evidence; a failed
  paragraph is withheld, the decision is never blocked.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
import re
from typing import Any, Callable, Mapping, Sequence

import torch

PROTOCOL = "compiled-credit-llm/assessment-v1"
POLICY_ID = "UCI_DEMO_POLICY_v1"
DISCLAIMER = "SYNTHETIC_NONREGULATORY_ASSOCIATIVE_NOT_CAUSAL"
EVIDENCE_PREFIX = "EVIDENCE1"
RECEIPT_PREFIX = "RECEIPT1"
ROUTE_MARKER = "ASSESSMENT_ROUTE|valid=1|validator=CHECKPOINT|verifier=CHECKPOINT"
FEATURE_NAMES = (
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2",
    "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2",
    "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1",
    "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)
BINDING_KEYS = ("p", "s", "q", "o", "b", "f", "t", "c", "xp", "xn", "r")
CALC_CODES = ("cu", "mu", "au", "dc", "rd", "ms", "pc", "bt", "bd", "pd", "zs", "pb", "pt")
FLAG_CODES = ("MI", "CI", "OD", "UC", "UR", "NF")
TRIGGER_CODES = ("IM", "IC", "DS", "DZ", "DX", "RU", "RO", "RH", "RM", "CD", "CU", "CP", "CB", "CE", "AS")
RISK_CODES = ("L", "M", "E", "H")
OUTCOME_CODES = ("II", "DD", "MR", "AC", "AS")
REFUSAL_REASONS = {
    "MISSING_FIELD": "Required raw credit field is missing.",
    "DUPLICATE_CONFLICT": "A duplicate raw credit field has conflicting values.",
    "OUT_OF_RANGE": "A raw credit value is outside the signed 64-bit input range.",
    "MALFORMED_TYPE": "Raw credit values must be integers or explicit nulls.",
    "EVIDENCE_MISMATCH": "Supplied evidence does not match checkpoint-computed evidence.",
    "UNSUPPORTED_REQUEST": "Only an unmodified credit assessment request is supported.",
    "OVERRIDE_ATTEMPT": "Checkpoint evidence and policy outcomes cannot be overridden.",
}
OUTCOME_WORDS = {"II": "invalid input", "DD": "demonstration decline", "MR": "manual review",
                 "AC": "approval with controls", "AS": "standard demonstration approval"}
RISK_WORDS = {"L": "low", "M": "moderate", "E": "elevated", "H": "high"}
FLAG_PHRASES = {
    "MI": "a required input is missing", "CI": "the hard input contract is invalid",
    "NF": "a value is nonfinite or has an invalid type", "OD": "a feature is outside the development range",
    "UC": "a demographic category is unknown", "UR": "a repayment code is undocumented in the UCI source",
}
TRIGGER_PHRASES = {
    "IM": "a required input is missing", "IC": "the input contract is invalid", "DS": "severe delinquency",
    "DZ": "zero payments with high utilization", "DX": "extreme utilization with delinquency",
    "RU": "an unknown category requires review", "RO": "a feature is outside the development range",
    "RH": "the model risk band requires review", "RM": "multiple soft controls require review",
    "CD": "repeated delinquency", "CU": "high utilization", "CP": "low payment coverage",
    "CB": "a rising balance trend", "CE": "elevated model risk", "AS": "standard approval",
}
HARD_CODES, REVIEW_CODES, SOFT_CODES = ("IM", "IC", "DS", "DZ", "DX"), ("RU", "RO", "RH", "RM"), ("CD", "CU", "CP", "CB", "CE")
LEVEL_OF = {"II": "hard", "DD": "hard", "MR": "review", "AC": "soft", "AS": "approve"}
OUTCOME_KEYWORDS = {"II": ("invalid",), "DD": ("decline",), "MR": ("manual review",), "AC": ("approv", "control"), "AS": ("approv",)}
CALC_LABELS = {
    "cu": ("current_utilization_pct", True), "mu": ("max_utilization_pct", True), "au": ("avg_utilization_pct", True),
    "dc": ("delinquent_statuses", False), "rd": ("repeated_delinquency", None), "ms": ("max_severity", False),
    "pc": ("payment_coverage_pct", True), "bt": ("balance_trend_pct", True), "bd": ("bill_dispersion", False),
    "pd": ("payment_dispersion", False), "zs": ("zero_payment_stress", None), "pb": ("positive_bill_total", False),
    "pt": ("payment_total", False),
}
SYSTEM_PROMPT = (
    "You are the explanation writer inside a compiled credit-assessment checkpoint. The decision, probability, "
    "policy rules, drivers and calculations are already fixed and verified; you only write one clear paragraph for "
    "a credit analyst that explains them. Use only the facts given. Never change, add or round beyond what is given. "
    "Plain prose, no headings, no lists, no markdown."
)
WRITING_BRIEF = (
    "Write one paragraph (4 to 7 sentences, 70 to 160 words) for a credit analyst: state the decision and the "
    "rule or rules that control it, give the model probability and band, name the main risk-increasing and "
    "risk-decreasing drivers, cite the calculations that matter for the active rules, mention any data-quality "
    "flag, say how close the case is to the next better outcome and what would change it (or that behaviour "
    "cannot change it), and close by noting the probability is a benchmark association, not a cause and not a guarantee. "
    "No quotation marks, no feature codes, no numbers other than those given."
)
BOUNDARY = ("Synthetic demonstration policy UCI_DEMO_POLICY_v1 on the historical 2005 UCI benchmark; model "
            "probability is a benchmark output, not a validated real-world probability of default; contributions "
            "are associations, not causes; not a lending, regulatory or adverse-action decision.")
_FORBIDDEN = (r"\bcaused\b", r"\bcauses\b", r"\bled to default\b", r"\bwill default\b", r"\bbecause the applicant\b",
              r"(?<!not a )\blegally valid adverse-action\b", r"\breal-world probability of default is\b", r"\bguarantees? (?:repayment|default|that)\b",
              r"\bprobability of (?:a |an |being )?(?:good|repay|approv|success|non-default|not default|positive)", r"\bgood credit risk\b",
              r"\bprobability of (?:this|the) (?:decision|outcome|assessment)\b", r"\blikelihood of (?:approval|repayment)\b")
_NUMBER = re.compile(r"(?<![A-Za-z_])[-+]?\d[\d,]*(?:\.\d+)?%?")
_FEATURE_CODE = re.compile(r"\b(?:LIMIT_BAL|SEX|EDUCATION|MARRIAGE|AGE|PAY_[023456]|BILL_AMT[1-6]|PAY_AMT[1-6])\b")
OUTCOME_RANK = {"II": 0, "DD": 1, "MR": 2, "AC": 3, "AS": 4}
BEHAVIOURAL = tuple(range(5, 23))  # PAY_0..PAY_6, BILL_AMT1..6, PAY_AMT1..6


@dataclass(frozen=True)
class RouteResult:
    status: str                      # "OK" or "REFUSE"
    output: str                      # JSON text (assessment or refusal)
    refusal_class: str | None
    generation_invoked: bool
    evidence: str | None = None
    prompt: str | None = None

    @property
    def payload(self) -> dict:
        return json.loads(self.output)


# ------------------------------------------------------------------------------------------------
# Request validation and refusals
# ------------------------------------------------------------------------------------------------

def refusal_receipt(code: str) -> str:
    return f"{RECEIPT_PREFIX}|status=REFUSE|e={code}|n={REFUSAL_REASONS[code]}|d={DISCLAIMER}"


def refusal(code: str) -> str:
    """Exact JSON refusal. Deterministic, no language model involved."""
    return json.dumps({
        "protocol": PROTOCOL, "status": "REFUSE", "refusal_class": code, "reason": REFUSAL_REASONS[code],
        "language_model_invoked": False, "receipt": refusal_receipt(code),
    }, indent=2)


def _pairs(value: Any) -> tuple[list[tuple[str, Any]], str | None]:
    if isinstance(value, Mapping):
        return list(value.items()), None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [], "MALFORMED_TYPE"
    pairs: list[tuple[str, Any]] = []
    for item in value:
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2 or not isinstance(item[0], str):
            return [], "MALFORMED_TYPE"
        pairs.append((item[0], item[1]))
    return pairs, None


def validate_raw_request(request: Any) -> tuple[str | None, list[int], list[bool]]:
    """Validate the envelope and the raw 23-field schema with fixed deterministic precedence."""
    if not isinstance(request, Mapping):
        return "MALFORMED_TYPE", [], []
    if "override" in request:
        return "OVERRIDE_ATTEMPT", [], []
    if request.get("action", "ASSESS") != "ASSESS" or set(request) - {"action", "credit", "bound_evidence"}:
        return "UNSUPPORTED_REQUEST", [], []
    if "credit" not in request:
        return "MISSING_FIELD", [], []
    pairs, error = _pairs(request["credit"])
    if error:
        return error, [], []
    seen: dict[str, Any] = {}
    duplicate_equal = False
    for name, value in pairs:
        if name in seen:
            if seen[name] != value:
                return "DUPLICATE_CONFLICT", [], []
            duplicate_equal = True
        else:
            seen[name] = value
    if duplicate_equal:  # duplicates are noncanonical even when equal
        return "DUPLICATE_CONFLICT", [], []
    if set(seen) != set(FEATURE_NAMES):
        return "MISSING_FIELD", [], []
    values: list[int] = []
    missing: list[bool] = []
    for name in FEATURE_NAMES:
        value = seen[name]
        if value is None:
            values.append(0); missing.append(True); continue
        if isinstance(value, bool) or not isinstance(value, int):
            return "MALFORMED_TYPE", [], []
        if value < -(2**63) or value > 2**63 - 1:
            return "OUT_OF_RANGE", [], []
        values.append(value); missing.append(False)
    return None, values, missing


# ------------------------------------------------------------------------------------------------
# Evidence serialisation and receipt
# ------------------------------------------------------------------------------------------------

def _active(values: Sequence[Any], codes: Sequence[str]) -> str:
    result = [code for code, active in zip(codes, values) if bool(active)]
    return ",".join(result) if result else "-"


def _top_contributions(values: Sequence[int], feature_names: Sequence[str]) -> tuple[str, str]:
    positive = sorted((i for i, value in enumerate(values) if value > 0), key=lambda i: (-values[i], i))[:2]
    negative = sorted((i for i, value in enumerate(values) if value < 0), key=lambda i: (values[i], i))[:2]
    encode = lambda indices: ",".join(f"{feature_names[i]}:{values[i]}" for i in indices) if indices else "-"
    return encode(positive), encode(negative)


def evidence_from_score(result: Any, feature_names: Sequence[str] = FEATURE_NAMES) -> str:
    """Serialise one verified score result into the canonical evidence line."""
    if result.verifier_passed.numel() != 1 or not bool(result.verifier_passed.item()):
        raise RuntimeError("checkpoint compiled scorer/verifier rejected evidence")
    scalar = lambda tensor: int(tensor.reshape(-1)[0].item())
    row = lambda tensor: [int(v) for v in tensor.reshape(tensor.shape[0], -1)[0].tolist()]
    score = scalar(result.score_fixed)
    probability = float(result.model_probability.reshape(-1)[0].item())
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise RuntimeError("checkpoint probability verifier failed")
    contributions = row(result.feature_contributions_fixed)
    xp, xn = _top_contributions(contributions, feature_names)
    fields = {
        "p": "UDP1", "s": str(score), "q": f"{probability:.12f}",
        "o": OUTCOME_CODES[scalar(result.policy_outcome_index)],
        "b": RISK_CODES[scalar(result.risk_band_index)],
        "f": _active(row(result.policy_flags), FLAG_CODES),
        "t": _active(row(result.policy_triggers), TRIGGER_CODES),
        "c": ",".join(f"{code}:{value}" for code, value in zip(CALC_CODES, row(result.policy_calculations))),
        "xp": xp, "xn": xn,
        "r": _active(row(result.evidence.policy_final_reasons), TRIGGER_CODES),
    }
    return f"{EVIDENCE_PREFIX}|" + "|".join(f"{key}={fields[key]}" for key in BINDING_KEYS)


def _fields(evidence: str) -> dict[str, str]:
    parts = evidence.split("|")
    if parts[0] != EVIDENCE_PREFIX or len(parts) != len(BINDING_KEYS) + 1:
        raise RuntimeError("noncanonical checkpoint evidence")
    fields = [part.split("=", 1) for part in parts[1:]]
    if tuple(key for key, _ in fields) != BINDING_KEYS:
        raise RuntimeError("noncanonical checkpoint evidence ordering")
    return dict(fields)


def expected_receipt(evidence: str) -> str:
    fields = _fields(evidence)
    return f"{RECEIPT_PREFIX}|" + "|".join(f"{key}={fields[key]}" for key in BINDING_KEYS) + f"|d={DISCLAIMER}"


# ------------------------------------------------------------------------------------------------
# Deterministic assessment (everything except the explanation paragraph)
# ------------------------------------------------------------------------------------------------

def facts(evidence: str, feature_labels: Mapping[str, str] | None = None, recourse_info: Mapping | None = None) -> dict:
    """Structured assessment computed from the evidence alone."""
    f = _fields(evidence)
    labels = dict(feature_labels or {})
    codes = lambda value: [] if value == "-" else value.split(",")
    phrases = lambda value, table: [table[c] for c in codes(value) if c in table]
    calc = {k: int(v) for k, v in (item.split(":", 1) for item in f["c"].split(","))}
    calculations = {}
    for code, (name, kind) in CALC_LABELS.items():
        value = calc[code]
        calculations[name] = bool(value) if kind is None else (round(value / 100, 2) if kind else value)
    drivers = lambda value: [
        {"feature": labels.get(item.split(":", 1)[0], item.split(":", 1)[0]), "code": item.split(":", 1)[0],
         "contribution": round(int(item.split(":", 1)[1]) / 1_000_000_000, 6)} for item in codes(value)]
    probability = float(f["q"])
    triggers = codes(f["t"])
    return {
        "protocol": PROTOCOL,
        "status": "OK",
        "decision": {"outcome_code": f["o"], "outcome": OUTCOME_WORDS[f["o"]], "level": LEVEL_OF[f["o"]],
                     "controlling_rules": phrases(f["r"], TRIGGER_PHRASES) if f["o"] != "AS" else [],
                     "note": "first active level in priority order hard > review > soft > approve" if f["o"] != "AS" else "no hard, review or soft rule is active, so the policy reaches standard approval"},
        "model": {"probability": f["q"], "probability_pct": f"{probability * 100:.1f}%", "risk_band": RISK_WORDS[f["b"]],
                  "raw_margin": round(int(f["s"]) / 1_000_000_000, 6)},
        "policy": {"id": POLICY_ID, "priority": ["hard", "review", "soft", "approve"],
                   "active": {"hard": [TRIGGER_PHRASES[c] for c in triggers if c in HARD_CODES],
                              "review": [TRIGGER_PHRASES[c] for c in triggers if c in REVIEW_CODES],
                              "soft": [TRIGGER_PHRASES[c] for c in triggers if c in SOFT_CODES]}},
        "drivers": {"increasing_risk": drivers(f["xp"]), "decreasing_risk": drivers(f["xn"])},
        "calculations": calculations,
        "data_quality": phrases(f["f"], FLAG_PHRASES),
        "recourse": dict(recourse_info) if recourse_info is not None else {"status": "NOT_COMPUTED"},
    }


def proximity_facts(rec: Mapping | None) -> dict:
    """Compact view of the recourse result for the language model."""
    if not rec or rec.get("status") == "NOT_COMPUTED":
        return {"status": "unknown"}
    out = {"status": rec["status"].lower().replace("_", " "), "blockers": list(rec.get("non_behavioural_blockers", []))}
    if rec.get("status") == "REACHABLE":
        nb = rec["next_better"]
        out.update({"proximity": rec["proximity"], "next_better_outcome": nb["outcome"], "moves_needed": nb["moves_needed"],
                    "what_would_change_it": nb["moves"], "model_after_change": nb["model_after"]})
        if rec["best_reachable"]["outcome_code"] != nb["outcome_code"]:
            out["best_reachable_outcome"] = rec["best_reachable"]["outcome"]; out["best_reachable_moves"] = rec["best_reachable"]["moves"]
    return out


def explanation_prompt(evidence: str, feature_labels: Mapping[str, str] | None = None, recourse_info: Mapping | None = None) -> str:
    """The user turn the language model sees: the deterministic facts, compactly, plus the writing brief."""
    fx = facts(evidence, feature_labels, recourse_info)
    shown = {k: fx[k] for k in ("decision", "model", "policy", "drivers", "calculations", "data_quality")}
    shown["proximity"] = proximity_facts(recourse_info)
    shown["model"] = {k: v for k, v in fx["model"].items() if k != "raw_margin"}
    shown["drivers"] = {k: [d["feature"] for d in v] for k, v in fx["drivers"].items()}  # labels only: no codes, no raw contributions
    brief = json.dumps(shown, separators=(",", ":"), ensure_ascii=False)
    return f"{ROUTE_MARKER}\n{evidence}\nFACTS={brief}\n{WRITING_BRIEF}"


# ------------------------------------------------------------------------------------------------
# Explanation verifier
# ------------------------------------------------------------------------------------------------

def licensed_numbers(evidence: str, recourse_info: Mapping | None = None) -> set[str]:
    f = _fields(evidence)
    extra: set[str] = set()
    if recourse_info and recourse_info.get("status") == "REACHABLE":
        for block in (recourse_info["next_better"], recourse_info["best_reachable"]):
            for text in block["moves"]:
                extra.update(m.group(0) for m in _NUMBER.finditer(text))
            extra.add(block["model_after"]["probability_pct"]); extra.add(str(block["moves_needed"]))
    probability = float(f["q"])
    allowed = {f["q"], f"{probability * 100:.1f}%", f"{probability * 100:.2f}%", f"{probability:.2f}", f"{probability:.3f}",
               f"{probability * 100:.0f}%", f"{probability * 100:.1f}", f"{probability * 100:.2f}"}
    calc = {k: int(v) for k, v in (item.split(":", 1) for item in f["c"].split(","))}
    for code, (name, kind) in CALC_LABELS.items():
        value = calc[code]
        if kind is None:
            continue
        if kind:
            for form in (f"{value / 100:.2f}%", f"{value / 100:.1f}%", f"{value / 100:.0f}%", f"{value / 100:.2f}", f"{value / 100:.1f}"):
                allowed.add(form)
        else:
            allowed.add(str(value)); allowed.add(f"{value:,}")
    allowed.update({"6", "13", "23", "2005", "1", "2", "3", "4", "5", "0"})  # month counts, feature counts, benchmark year, small ordinals
    allowed.update(extra)
    allowed.update(v.replace(",", "") for v in list(allowed))
    return allowed


def verify_explanation(text: str, evidence: str, feature_labels: Mapping[str, str] | None = None,
                       recourse_info: Mapping | None = None) -> list[str]:
    """Return the list of violations (empty means verified). Deterministic and checkpoint-owned."""
    f = _fields(evidence)
    labels = dict(feature_labels or {})
    problems: list[str] = []
    if not isinstance(text, str) or not text.strip():
        return ["empty"]
    clean = text.strip()
    if "\n" in clean or clean.startswith("#") or "**" in clean or clean.startswith("-"):
        problems.append("formatting: must be one plain paragraph")
    if any(ch in clean for ch in '"“”$()[]'):
        problems.append("formatting: no quotation marks, currency signs or brackets")
    words = clean.split()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()]
    if not 60 <= len(words) <= 175:
        problems.append(f"length: {len(words)} words")
    if not 3 <= len(sentences) <= 8:
        problems.append(f"sentences: {len(sentences)}")
    lower = clean.lower()
    if not all(k in lower for k in OUTCOME_KEYWORDS[f["o"]]):
        problems.append("outcome not stated")
    if f["o"] != "AS":
        for code in f["r"].split(","):
            if code in TRIGGER_PHRASES and TRIGGER_PHRASES[code].lower() not in lower:
                problems.append(f"controlling rule missing: {TRIGGER_PHRASES[code]}")
    probability = float(f["q"])
    if f["q"] not in clean and f"{probability * 100:.1f}%" not in clean and f"{probability * 100:.2f}%" not in clean:
        problems.append("probability missing")
    if RISK_WORDS[f["b"]] not in lower:
        problems.append("risk band missing")
    allowed = licensed_numbers(evidence, recourse_info)
    found = {m.group(0).rstrip(",.") for m in _NUMBER.finditer(clean)}
    bad = sorted(n for n in found if n and n not in allowed and n.rstrip("%") not in allowed)
    if bad:
        problems.append("unlicensed numbers: " + ",".join(bad))
    licensed = {labels[c.split(":", 1)[0]].lower() for key in ("xp", "xn") for c in f[key].split(",") if c != "-" and c.split(":", 1)[0] in labels}
    if recourse_info:
        recourse_text = json.dumps(recourse_info).lower()
        licensed |= {v.lower() for v in labels.values() if v.lower() in recourse_text}
    unlicensed = sorted(l for l in {v.lower() for v in labels.values()} - licensed if l in lower)
    if unlicensed or _FEATURE_CODE.search(clean):
        problems.append("unlicensed features: " + ",".join(unlicensed + _FEATURE_CODE.findall(clean)))
    if any(re.search(p, lower) for p in _FORBIDDEN):
        problems.append("causal/legal/guarantee claim")
    if not any(t in lower for t in ("not causal", "not a cause", "association", "benchmark", "not causation")):
        problems.append("association/benchmark caveat missing")
    if recourse_info and recourse_info.get("status") == "REACHABLE":
        nb = recourse_info["next_better"]
        if not all(k in lower for k in OUTCOME_KEYWORDS[nb["outcome_code"]]):
            problems.append("recourse: next better outcome not mentioned")
    elif recourse_info and recourse_info.get("status", "").startswith("NOT_REACHABLE") and "behaviour" not in lower and "behavior" not in lower and "review" not in lower:
        problems.append("recourse: non-behavioural blocker not acknowledged")
    return problems


def assemble_assessment(evidence: str, explanation: str | None, feature_labels: Mapping[str, str] | None = None,
                        violations: Sequence[str] | None = None, recourse_info: Mapping | None = None) -> dict:
    fx = facts(evidence, feature_labels, recourse_info)
    fx["explanation"] = explanation if not violations else None
    fx["explanation_status"] = "VERIFIED" if explanation and not violations else ("WITHHELD_BY_VERIFIER" if explanation else "NOT_GENERATED")
    if violations:
        fx["explanation_violations"] = list(violations)
    fx["boundary"] = BOUNDARY
    fx["receipt"] = expected_receipt(evidence)
    return fx


# ------------------------------------------------------------------------------------------------
# Recourse: the smallest behavioural change that reaches a better policy outcome
# ------------------------------------------------------------------------------------------------

def _moves(values: list[int], thresholds: Sequence[int]) -> list[tuple[str, str, list[int]]]:
    """Candidate single moves derived from the policy's own thresholds: (key, description, new feature vector)."""
    scale, high_util, extreme_util, low_cov, rising = thresholds[0], thresholds[2], thresholds[3], thresholds[4], thresholds[5]
    limit = max(values[0], 1)
    status, bills, pays = values[5:11], values[11:17], values[17:23]
    positive = [max(b, 0) for b in bills]
    pb, pt = sum(positive), sum(pays)
    moves: list[tuple[str, str, list[int]]] = []

    def with_(new_status=None, new_bills=None, new_pays=None):
        out = list(values)
        if new_status is not None: out[5:11] = new_status
        if new_bills is not None: out[11:17] = new_bills
        if new_pays is not None: out[17:23] = new_pays
        return out

    delinquent = [s for s in status if s > 0]
    if delinquent:
        moves.append(("statuses_current", "all six repayment statuses paid on time (no month in delay)", with_(new_status=[min(s, 0) for s in status])))
        if max(delinquent) >= 2:
            moves.append(("severity_le_1", "no repayment status later than one month (maximum delay 1)", with_(new_status=[min(s, 1) for s in status])))
        if len(delinquent) >= 2:
            first = next(i for i, s in enumerate(status) if s > 0)
            moves.append(("single_delay", "only one month in delay instead of %d" % len(delinquent),
                          with_(new_status=[s if i == first else min(s, 0) for i, s in enumerate(status)])))
    if pb > 0 and pt * scale // pb < low_cov:
        needed = -(-low_cov * pb // scale)  # ceil
        while needed * scale // pb < low_cov:
            needed += 1
        new_pays = list(pays); new_pays[0] += needed - pt
        moves.append(("raise_payments", f"total payments over the six months of at least {needed:,} instead of {pt:,} (payment coverage {low_cov / 100:.2f}% of positive bills)", with_(new_pays=new_pays)))
    elif pt == 0 and pb > 0:
        new_pays = list(pays); new_pays[0] = 1
        moves.append(("any_payment", "at least one payment made in the six months", with_(new_pays=new_pays)))
    max_bill = max(positive) if positive else 0
    for key, cap, label in (("balances_below_high", high_util, "below %.0f%% of the limit" % (high_util / 100)),
                            ("balances_below_extreme", extreme_util, "below %.0f%% of the limit" % (extreme_util / 100))):
        if max_bill * scale // limit >= cap:
            target_max = (cap * limit - 1) // scale  # largest bill with utilization < cap
            while target_max * scale // limit >= cap:
                target_max -= 1
            new_bills = [min(b, target_max) if b > 0 else b for b in bills]
            moves.append((key, f"every statement balance held to at most {target_max:,} ({label}; the highest was {max_bill:,})", with_(new_bills=new_bills)))
    if (bills[0] - bills[5]) * scale // limit >= rising:
        target_first = bills[5] + (rising * limit - 1) // scale
        while (target_first - bills[5]) * scale // limit >= rising:
            target_first -= 1
        new_bills = list(bills); new_bills[0] = target_first
        moves.append(("flatten_trend", f"most recent statement balance of at most {target_first:,} instead of {bills[0]:,} (balance growth below {rising / 100:.0f}% of the limit)", with_(new_bills=new_bills)))
    return moves


def recourse(values: list[int], missing: list[bool], scorer: Callable[[torch.Tensor, torch.Tensor], Any],
             thresholds: Sequence[int], feature_names: Sequence[str] = FEATURE_NAMES, max_moves: int = 3,
             feature_labels: Mapping[str, str] | None = None) -> dict:
    """Exact recourse through the compiled scorer. Only behavioural fields move; demographics and the limit never do."""
    with torch.inference_mode():
        base = scorer(torch.tensor([values], dtype=torch.int64), torch.tensor([missing], dtype=torch.bool))
    current = OUTCOME_CODES[int(base.policy_outcome_index.reshape(-1)[0].item())]
    flags = [code for code, on in zip(FLAG_CODES, [int(v) for v in base.policy_flags.reshape(-1).tolist()]) if on]
    blockers = []
    if any(missing):
        labels = dict(feature_labels or {})
        blockers.append("required field missing: " + ", ".join(labels.get(feature_names[i], feature_names[i]).lower() for i, m in enumerate(missing) if m))
    if "CI" in flags:
        blockers.append("input violates the hard contract (for example a negative payment or an impossible limit or age)")
    if "UC" in flags:
        blockers.append("a demographic category is unknown; this cannot be changed by borrower behaviour and routes to human review")
    if "OD" in flags:
        blockers.append("a value is outside the development range; this routes to human review regardless of behaviour")
    result = {"current_outcome": OUTCOME_WORDS[current], "current_outcome_code": current, "scope": "behavioural fields only: repayment statuses, statement balances, payments",
              "non_behavioural_blockers": blockers}
    if current == "AS":
        result["status"] = "ALREADY_BEST"
        return result
    if blockers and current in ("II", "MR"):
        result["status"] = "NOT_REACHABLE_BY_BEHAVIOUR"
        return result
    singles = _moves(values, thresholds)
    if not singles:
        result["status"] = "NO_BEHAVIOURAL_MOVE_AVAILABLE"
        return result
    candidates: dict[tuple[int, ...], tuple[tuple[str, ...], list[int]]] = {}
    for size in range(1, min(max_moves, len(singles)) + 1):
        for combo in itertools.combinations(range(len(singles)), size):
            vec = list(values)
            for index in combo:
                key, _, moved = singles[index]
                for m in _moves(vec, thresholds):  # re-derive each move on the modified vector so combinations compose exactly
                    if m[0] == key:
                        vec = m[2]; break
            tup = tuple(vec)
            if tup != tuple(values) and (tup not in candidates or len(combo) < len(candidates[tup][0])):
                candidates[tup] = (tuple(singles[i][0] for i in combo), vec)
    if not candidates:
        result["status"] = "NO_BEHAVIOURAL_MOVE_AVAILABLE"
        return result
    vectors = [vec for _, vec in candidates.values()]
    with torch.inference_mode():
        scored = scorer(torch.tensor(vectors, dtype=torch.int64), torch.tensor([missing] * len(vectors), dtype=torch.bool))
    outcomes = [int(v) for v in scored.policy_outcome_index.reshape(-1).tolist()]
    bands = [int(v) for v in scored.risk_band_index.reshape(-1).tolist()]
    probs = [float(v) for v in scored.model_probability.reshape(-1).tolist()]
    descriptions = {m[0]: m[1] for m in singles}
    options = []
    for (keys, vec), outcome, band, prob in zip(candidates.values(), outcomes, bands, probs):
        if outcome > OUTCOME_RANK[current]:
            changed = sum(1 for i in BEHAVIOURAL if vec[i] != values[i])
            options.append((outcome, len(keys), changed, keys, band, prob))
    if not options:
        result["status"] = "NOT_REACHABLE_WITHIN_%d_MOVES" % max_moves
        return result
    def pack(option):
        outcome, n, changed, keys, band, prob = option
        return {"outcome": OUTCOME_WORDS[OUTCOME_CODES[outcome]], "outcome_code": OUTCOME_CODES[outcome], "moves_needed": n,
                "moves": [descriptions[k] for k in keys], "fields_changed": changed,
                "model_after": {"risk_band": RISK_WORDS[RISK_CODES[band]], "probability_pct": f"{prob * 100:.1f}%"}}
    nearest = min(options, key=lambda o: (o[0], o[1], o[2]))            # smallest improvement with fewest moves
    best = min(options, key=lambda o: (-o[0], o[1], o[2]))              # best reachable outcome with fewest moves
    result["status"] = "REACHABLE"
    result["next_better"] = pack(nearest)
    result["best_reachable"] = pack(best)
    result["proximity"] = {1: "close", 2: "moderate"}.get(nearest[1], "far")
    return result


# ------------------------------------------------------------------------------------------------
# The route
# ------------------------------------------------------------------------------------------------

def route_request(request: Any, scorer: Callable[[torch.Tensor, torch.Tensor], Any],
                  explanation_generator: Callable[[str, str], str] | None, feature_names: Sequence[str] = FEATURE_NAMES,
                  feature_labels: Mapping[str, str] | None = None, thresholds: Sequence[int] | None = None) -> RouteResult:
    """Malformed -> exact refusal (no scorer, no language model). Valid -> deterministic JSON plus one verified paragraph.

    The decision is never blocked by the language model: a failed paragraph is withheld, not substituted. With
    ``explanation_generator=None`` the assessment is returned without a paragraph (``explanation_status: NOT_GENERATED``)."""
    error, values, missing = validate_raw_request(request)
    if error:
        return RouteResult("REFUSE", refusal(error), error, False)
    with torch.inference_mode():
        scored = scorer(torch.tensor([values], dtype=torch.int64), torch.tensor([missing], dtype=torch.bool))
    evidence = evidence_from_score(scored, feature_names)
    bound = request.get("bound_evidence")
    if bound is not None and bound != evidence:
        return RouteResult("REFUSE", refusal("EVIDENCE_MISMATCH"), "EVIDENCE_MISMATCH", False, evidence=evidence)
    rec = recourse(values, missing, scorer, thresholds, feature_names, feature_labels=feature_labels) if thresholds is not None else None
    prompt = explanation_prompt(evidence, feature_labels, rec)
    if explanation_generator is None:
        output = json.dumps(assemble_assessment(evidence, None, feature_labels, None, rec), indent=2, ensure_ascii=False)
        return RouteResult("OK", output, None, False, evidence=evidence, prompt=prompt)
    explanation = explanation_generator(SYSTEM_PROMPT, prompt).strip()
    violations = verify_explanation(explanation, evidence, feature_labels, rec)
    output = json.dumps(assemble_assessment(evidence, explanation, feature_labels, violations, rec), indent=2, ensure_ascii=False)
    return RouteResult("OK", output, None, True, evidence=evidence, prompt=prompt)


class AssessmentMixin:
    """Adds ``assess_credit`` to the checkpoint model class."""

    def assess_credit(self, request: Any, tokenizer: Any = None, explain: bool = True, **generation_kwargs: Any) -> RouteResult:
        """Validate, score, assemble the assessment and (optionally) write and verify the explanation paragraph."""
        device = next(self.parameters()).device
        metadata = self.config.compiled_credit
        labels = {item["raw_feature_name"]: item["factual_label"] for item in metadata["reason_metadata"]}
        scorer = lambda features, missing: self.score_credit(features.to(device), missing.to(device))
        thresholds = [int(v) for v in self.compiled_credit.demo_policy.thresholds.tolist()]
        generate = None
        if explain:
            if tokenizer is None:
                raise ValueError("a tokenizer is required to write the explanation paragraph; pass explain=False to skip it")

            def generate(system: str, user: str) -> str:
                encoded = tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                kwargs = {"do_sample": False, "max_new_tokens": 320, **generation_kwargs}
                output = self.generate(**encoded, **kwargs)
                return tokenizer.decode(output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return route_request(request, scorer, generate, metadata["feature_names"], labels, thresholds)
