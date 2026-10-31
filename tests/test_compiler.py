import random
import tempfile
import unittest
from pathlib import Path

import torch

from compiled_credit_llm.compiler.compiler import compile_manifest, load_compiled, save_compiled
from compiled_credit_llm.compiler.schema import CompiledManifest, ManifestError
from test_schema import manifest_dict


def irs():
    return [
        {"format_version": "normalized-tree-v1", "output_scale": 100, "base_score": 7, "trees": [
            {"type": "split", "feature": 0, "threshold": 10, "missing": "left",
             "left": {"type": "split", "feature": 1, "threshold": 5, "missing": "right",
                      "left": {"type": "leaf", "value": -20, "reasons": {"INCOME": -20}},
                      "right": {"type": "leaf", "value": 30, "reasons": {"DEBT": 30}}},
             "right": {"type": "leaf", "value": 80, "reasons": {"INCOME": 80}}},
            {"type": "split", "feature": 1, "threshold": 0, "missing": "right",
             "left": {"type": "leaf", "value": -10, "reasons": {"DEBT": -10}},
             "right": {"type": "leaf", "value": 10, "reasons": {"DEBT": 10}}}]},
        {"format_version": "normalized-tree-v1", "output_scale": 100, "base_score": -3, "trees": [
            {"type": "split", "feature": 0, "threshold": 10, "missing": "right",
             "left": {"type": "leaf", "value": -8, "reasons": {"INCOME": -8}},
             "right": {"type": "leaf", "value": 12, "reasons": {"INCOME": 12}}}]},
    ]


def eval_node(node, x, missing):
    if node["type"] == "leaf":
        return node["value"], dict(node["reasons"])
    go_left = node["missing"] == "left" if missing[node["feature"]] else x[node["feature"]] < node["threshold"]
    return eval_node(node["left"] if go_left else node["right"], x, missing)


def oracle(manifest, all_irs, x, missing):
    scores, reasons = [], []
    codes = [r.code for r in manifest.reasons]
    for ir in all_irs:
        score = ir["base_score"]; reason = {code: 0 for code in codes}
        for tree in ir["trees"]:
            value, contributions = eval_node(tree, x, missing)
            score += value
            for code, contribution in contributions.items():
                reason[code] += contribution
        scores.append(score); reasons.append(reason)
    routing, cal = manifest.routing, manifest.calibration
    routed = round(sum(s * w for s, w in zip(scores, routing.weight_numerators)) / routing.denominator)
    score = max(cal.minimum, min(cal.maximum, round(routed * cal.multiplier / cal.divisor) + cal.offset))
    reason_values = [round(round(sum(r[code] * w for r, w in zip(reasons, routing.weight_numerators)) / routing.denominator) * cal.multiplier / cal.divisor) for code in codes]
    return score, scores, reason_values


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = CompiledManifest.from_dict(manifest_dict())
        self.irs = irs()
        self.module = compile_manifest(self.manifest, self.irs)

    def assert_case(self, x, missing):
        expected = oracle(self.manifest, self.irs, x, missing)
        output = self.module(torch.tensor([x]), torch.tensor([missing]))
        self.assertEqual(int(output.score[0]), expected[0])
        self.assertEqual([int(v) for v in output.expert_scores[0]], expected[1])
        self.assertEqual([int(v) for v in output.reason_contributions[0]], expected[2])
        self.assertEqual(int(output.grade_one_hot[0].sum()), 1)

    def test_boundary_and_missing_semantics(self):
        for x, missing in [([9, 4], [0, 0]), ([10, 4], [0, 0]), ([11, 5], [0, 0]), ([10, 0], [1, 0]), ([0, 0], [0, 1]), ([0, 0], [1, 1])]:
            with self.subTest(x=x, missing=missing):
                self.assert_case(x, missing)

    def test_deterministic_random_oracle_equivalence(self):
        rng = random.Random(73421)
        for _ in range(300):
            self.assert_case([rng.randint(-100, 100), rng.randint(-100, 100)], [rng.randint(0, 1), rng.randint(0, 1)])

    def test_mutated_weight_is_detected_by_oracle(self):
        x, missing = [0, 0], [0, 0]
        before = int(self.module(torch.tensor([x]), torch.tensor([missing])).score[0])
        with torch.no_grad():
            self.module.experts[0].leaf_value[0] += 4
        after = int(self.module(torch.tensor([x]), torch.tensor([missing])).score[0])
        self.assertNotEqual(before, after)
        self.assertNotEqual(after, oracle(self.manifest, self.irs, x, missing)[0])

    def test_save_reload_and_receipt_tamper(self):
        x = torch.tensor([[10, 5], [0, 0]]); missing = torch.tensor([[0, 0], [1, 1]])
        expected = self.module(x, missing).score
        with tempfile.TemporaryDirectory() as directory:
            receipt = save_compiled(directory, self.manifest, self.irs, self.module)
            loaded_manifest, loaded, loaded_receipt = load_compiled(directory)
            self.assertEqual(loaded_manifest, self.manifest); self.assertEqual(receipt, loaded_receipt)
            torch.testing.assert_close(loaded(x, missing).score, expected, rtol=0, atol=0)
            state_path = Path(directory) / "compiled_weights.pt"
            state = torch.load(state_path, weights_only=True)
            state["experts.0.leaf_value"][0] += 1
            torch.save(state, state_path)
            with self.assertRaises(ManifestError):
                load_compiled(directory)

    def test_lte_policy_includes_threshold(self):
        x, missing = [0, 0], [0, 0]
        threshold = oracle(self.manifest, self.irs, x, missing)[0]
        raw = manifest_dict(); raw["policy"]["approve_when"] = "lte"; raw["policy"]["decision_threshold"] = threshold
        module = compile_manifest(CompiledManifest.from_dict(raw), self.irs)
        self.assertEqual(int(module(torch.tensor([x]), torch.tensor([missing])).decision[0]), 1)

    def test_resource_limit_fails_closed(self):
        raw = manifest_dict(); raw["precision"]["max_depth"] = 1
        with self.assertRaises(ManifestError):
            compile_manifest(CompiledManifest.from_dict(raw), self.irs)
