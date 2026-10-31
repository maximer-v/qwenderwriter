"""Router, assessment assembly, explanation verifier and recourse, with a fake scorer (no checkpoint needed)."""
import json
import unittest
from types import SimpleNamespace

import torch

from compiled_credit_llm.checkpoint.routing import (
    EVIDENCE_PREFIX, FEATURE_NAMES, OUTCOME_CODES, RECEIPT_PREFIX, expected_receipt, facts, licensed_numbers, recourse,
    refusal, route_request, validate_raw_request, verify_explanation,
)

EVIDENCE = (f"{EVIDENCE_PREFIX}|p=UDP1|s=-2786075081|q=0.058081307163|o=AC|b=L|f=UR|t=CB|c=cu:6090,mu:7107,au:3608,dc:0,rd:0,"
            "ms:0,pc:2274,bt:5484,bd:24021,pd:2289,zs:0,pb:173225,pt:39396|xp=LIMIT_BAL:86801866,BILL_AMT3:75170530"
            "|xn=PAY_0:-463110267,PAY_AMT2:-244290662|r=CB")
LABELS = {"LIMIT_BAL": "Credit limit amount", "BILL_AMT3": "Statement balance three months ago", "PAY_0": "Most recent repayment status",
          "PAY_AMT2": "Payment amount two months ago", "AGE": "Age"}
GOOD = ("The synthetic policy approves this account with controls because a rising balance trend is active while no hard "
        "decline or review rule fired. The model places it in the low risk band with a benchmark probability of 5.8%. Credit "
        "limit amount and Statement balance three months ago push the model margin up, while Most recent repayment status and "
        "Payment amount two months ago pull it down. There are 0 delinquent statuses, current utilization is 60.90% and the "
        "balance trend is 54.84% of the limit, which is what triggers the control. One data-quality flag applies: a repayment "
        "code is undocumented in the UCI source. The probability is an association from a benchmark model, not a cause and not "
        "a guarantee of repayment.")


def raw_credit():
    return {name: 0 for name in FEATURE_NAMES}


def fake_score(outcome=4):
    triggers = torch.zeros((1, 15), dtype=torch.int64)
    triggers[0, 14 if outcome == 4 else (0 if outcome == 0 else 2 if outcome == 1 else 6 if outcome == 2 else 9)] = 1
    return SimpleNamespace(
        verifier_passed=torch.tensor([True]), score_fixed=torch.tensor([0]), model_probability=torch.tensor([0.5], dtype=torch.float64),
        policy_outcome_index=torch.tensor([outcome]), risk_band_index=torch.tensor([1]), policy_flags=torch.zeros((1, 6), dtype=torch.int64),
        policy_triggers=triggers, policy_calculations=torch.zeros((1, 13), dtype=torch.int64),
        feature_contributions_fixed=torch.tensor([[2, -1] + [0] * 21], dtype=torch.int64), evidence=SimpleNamespace(policy_final_reasons=triggers.clone()),
    )


class ValidationTests(unittest.TestCase):
    def test_all_seven_malformed_classes_are_exact_and_never_generate(self):
        base = raw_credit()
        missing = dict(base); missing.pop(FEATURE_NAMES[-1])
        out_of_range = dict(base); out_of_range[FEATURE_NAMES[0]] = 2 ** 63
        malformed = dict(base); malformed[FEATURE_NAMES[0]] = 1.25
        cases = {
            "MISSING_FIELD": {"credit": missing}, "DUPLICATE_CONFLICT": {"credit": list(base.items()) + [(FEATURE_NAMES[0], 1)]},
            "OUT_OF_RANGE": {"credit": out_of_range}, "MALFORMED_TYPE": {"credit": malformed},
            "EVIDENCE_MISMATCH": {"credit": base, "bound_evidence": f"{EVIDENCE_PREFIX}|tampered=1"},
            "UNSUPPORTED_REQUEST": {"credit": base, "action": "CHANGE_DECISION"}, "OVERRIDE_ATTEMPT": {"credit": base, "override": "IGNORE_CHECKPOINT"},
        }
        for expected, request in cases.items():
            def generate(_system, _prompt):
                raise AssertionError("malformed input reached language generation")
            result = route_request(request, lambda _f, _m: fake_score(), generate)
            self.assertEqual(result.status, "REFUSE"); self.assertEqual(result.refusal_class, expected)
            self.assertEqual(result.output, refusal(expected)); self.assertFalse(result.generation_invoked)
            payload = json.loads(result.output)
            self.assertEqual(payload["status"], "REFUSE"); self.assertFalse(payload["language_model_invoked"])
            self.assertTrue(payload["receipt"].startswith(f"{RECEIPT_PREFIX}|status=REFUSE|e={expected}|"))

    def test_explicit_null_is_policy_input_not_router_missing_field(self):
        credit = raw_credit(); credit[FEATURE_NAMES[3]] = None
        error, values, missing = validate_raw_request({"credit": credit})
        self.assertIsNone(error); self.assertTrue(missing[3]); self.assertEqual(values[3], 0)

    def test_booleans_are_not_integers(self):
        credit = raw_credit(); credit["SEX"] = True
        self.assertEqual(validate_raw_request({"credit": credit})[0], "MALFORMED_TYPE")


class AssessmentTests(unittest.TestCase):
    def test_facts_are_deterministic_and_complete(self):
        fx = facts(EVIDENCE, LABELS)
        self.assertEqual(fx["decision"]["outcome_code"], "AC"); self.assertEqual(fx["decision"]["level"], "soft")
        self.assertEqual(fx["decision"]["controlling_rules"], ["a rising balance trend"])
        self.assertEqual(fx["model"]["probability_pct"], "5.8%"); self.assertEqual(fx["calculations"]["current_utilization_pct"], 60.9)
        self.assertEqual(fx["drivers"]["increasing_risk"][0]["feature"], "Credit limit amount")
        self.assertEqual(fx["policy"]["active"], {"hard": [], "review": [], "soft": ["a rising balance trend"]})
        self.assertTrue(expected_receipt(EVIDENCE).startswith(f"{RECEIPT_PREFIX}|p=UDP1|s=-2786075081|"))

    def test_every_outcome_reaches_generation_exactly_once(self):
        for outcome, code in enumerate(OUTCOME_CODES):
            calls = {"score": 0, "generate": 0}
            def scorer(features, missing):
                calls["score"] += 1; return fake_score(outcome)
            def generate(system, prompt):
                calls["generate"] += 1
                self.assertIn(f"|o={code}|", prompt); self.assertIn("FACTS={", prompt)
                return "Too short."
            result = route_request({"action": "ASSESS", "credit": raw_credit()}, scorer, generate)
            self.assertEqual(result.status, "OK"); self.assertTrue(result.generation_invoked)
            self.assertEqual(calls, {"score": 1, "generate": 1})

    def test_failed_explanation_is_withheld_but_decision_delivered(self):
        result = route_request({"credit": raw_credit()}, lambda f, m: fake_score(0), lambda s, u: "Too short.")
        payload = json.loads(result.output)
        self.assertEqual(payload["status"], "OK"); self.assertEqual(payload["explanation_status"], "WITHHELD_BY_VERIFIER")
        self.assertIsNone(payload["explanation"]); self.assertIn("decision", payload); self.assertIn("receipt", payload)
        self.assertTrue(payload["explanation_violations"])

    def test_no_generator_means_no_language_model(self):
        result = route_request({"credit": raw_credit()}, lambda f, m: fake_score(3), None)
        payload = json.loads(result.output)
        self.assertFalse(result.generation_invoked); self.assertEqual(payload["explanation_status"], "NOT_GENERATED")


class VerifierTests(unittest.TestCase):
    def test_accepts_good_paragraph(self):
        self.assertEqual(verify_explanation(GOOD, EVIDENCE, LABELS), [])

    def test_rejects_unlicensed_number_and_feature(self):
        problems = verify_explanation(GOOD.replace("5.8%", "7.1%").replace("Credit limit amount", "Age"), EVIDENCE, LABELS)
        self.assertTrue(any(p.startswith("unlicensed numbers") for p in problems))
        self.assertTrue(any(p.startswith("unlicensed features") for p in problems))

    def test_rejects_causal_claim_and_feature_codes(self):
        self.assertIn("causal/legal/guarantee claim", verify_explanation(GOOD + " This caused the outcome.", EVIDENCE, LABELS))
        self.assertTrue(any(p.startswith("unlicensed features") for p in verify_explanation(GOOD.replace("Credit limit amount", "LIMIT_BAL"), EVIDENCE, LABELS)))

    def test_rejects_missing_controlling_rule_and_inverted_probability(self):
        self.assertIn("controlling rule missing: a rising balance trend", verify_explanation(GOOD.replace("a rising balance trend", "a rising trend"), EVIDENCE, LABELS))
        self.assertIn("causal/legal/guarantee claim", verify_explanation(GOOD.replace("benchmark probability of 5.8%", "probability of repayment of 5.8%"), EVIDENCE, LABELS))

    def test_recourse_numbers_are_licensed(self):
        rec = {"status": "REACHABLE", "proximity": "close", "non_behavioural_blockers": [],
               "next_better": {"outcome": "standard demonstration approval", "outcome_code": "AS", "moves_needed": 1,
                               "moves": ["most recent statement balance of at most 28,848 instead of 48,725"], "model_after": {"risk_band": "low", "probability_pct": "5.8%"}},
               "best_reachable": {"outcome": "standard demonstration approval", "outcome_code": "AS", "moves_needed": 1,
                                  "moves": ["most recent statement balance of at most 28,848 instead of 48,725"], "model_after": {"risk_band": "low", "probability_pct": "5.8%"}}}
        self.assertIn("28,848", licensed_numbers(EVIDENCE, rec)); self.assertIn("48725", licensed_numbers(EVIDENCE, rec))
        text = GOOD.replace("which is what triggers the control.", "which is what triggers the control, and the case is close to a standard demonstration approval if the most recent statement balance were at most 28,848 instead of 48,725.")
        self.assertEqual(verify_explanation(text, EVIDENCE, LABELS, rec), [])
        review = {**rec, "next_better": {**rec["next_better"], "outcome": "manual review", "outcome_code": "MR"}}
        self.assertIn("recourse: next better outcome not mentioned", verify_explanation(GOOD, EVIDENCE, LABELS, review))


class RecourseFakeScorerTests(unittest.TestCase):
    """Recourse mechanics with a scorer that approves once all statuses are on time."""

    THRESHOLDS = [10000, 8000, 9000, 15000, 1000, 3000, 2, 3, 4, 2, -1_000_000_000, 0, 1_000_000_000]

    @staticmethod
    def scorer(features, missing):
        status = features[:, 5:11]
        delinquent = (status > 0).sum(dim=1)
        outcome = torch.where(delinquent == 0, torch.full_like(delinquent, 4), torch.full_like(delinquent, 1))
        n = features.shape[0]
        return SimpleNamespace(policy_outcome_index=outcome, policy_flags=torch.zeros((n, 6), dtype=torch.int64),
                               risk_band_index=torch.zeros(n, dtype=torch.int64), model_probability=torch.full((n,), 0.1, dtype=torch.float64))

    def test_one_move_to_approval(self):
        values = [50000, 1, 1, 1, 30, 3, 2, 2, 2, 2, 0] + [1000] * 6 + [500] * 6
        rec = recourse(values, [False] * 23, self.scorer, self.THRESHOLDS)
        self.assertEqual(rec["status"], "REACHABLE"); self.assertEqual(rec["next_better"]["outcome_code"], "AS")
        self.assertEqual(rec["next_better"]["moves_needed"], 1); self.assertEqual(rec["proximity"], "close")
        self.assertIn("paid on time", rec["next_better"]["moves"][0])

    def test_already_best_and_missing_blocker(self):
        values = [50000, 1, 1, 1, 30, 0, 0, 0, 0, 0, 0] + [1000] * 6 + [500] * 6
        self.assertEqual(recourse(values, [False] * 23, self.scorer, self.THRESHOLDS)["status"], "ALREADY_BEST")
        blocked = [50000, 1, 1, 1, 30, 3, 0, 0, 0, 0, 0] + [1000] * 6 + [500] * 6
        def invalid(features, missing):
            n = features.shape[0]
            flags = torch.zeros((n, 6), dtype=torch.int64); flags[:, 0] = 1
            return SimpleNamespace(policy_outcome_index=torch.zeros(n, dtype=torch.int64), policy_flags=flags,
                                   risk_band_index=torch.zeros(n, dtype=torch.int64), model_probability=torch.full((n,), 0.5, dtype=torch.float64))
        rec = recourse(blocked, [True] + [False] * 22, invalid, self.THRESHOLDS, feature_labels={"LIMIT_BAL": "Credit limit amount"})
        self.assertEqual(rec["status"], "NOT_REACHABLE_BY_BEHAVIOUR"); self.assertIn("credit limit amount", rec["non_behavioural_blockers"][0])
