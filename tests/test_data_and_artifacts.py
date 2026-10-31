"""Data contract, request reconstruction and, when the build artifacts exist, the compiled scorer and base checkpoint."""
import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from compiled_credit_llm import paths
from compiled_credit_llm.data.requests import RawTable, malformed_mutations
from compiled_credit_llm.data.uci import FEATURE_NAMES, validate_csv

HAVE_DATA = paths.DATA_CSV.exists() and paths.SPLIT_MANIFEST.exists()
HAVE_SCORER = (paths.SCORER / "xgboost_model.json").exists() and (paths.COMPILED_SCORER / "receipt.json").exists()
HAVE_BASE = (paths.BASE_CHECKPOINT / "config.json").exists()


@unittest.skipUnless(HAVE_DATA, "data files missing")
class DataTests(unittest.TestCase):
    def test_csv_and_splits(self):
        info = validate_csv(paths.DATA_CSV)
        self.assertEqual(info["rows"], 30_000); self.assertTrue(info["matches_reference"])
        manifest = pd.read_csv(paths.SPLIT_MANIFEST)
        self.assertEqual(manifest["split"].value_counts().to_dict(), {"train": 18_000, "validation": 4_500, "calibration": 3_000, "test": 4_500})
        self.assertEqual(manifest["opaque_row_id"].nunique(), 30_000)

    def test_request_reconstruction_and_variants(self):
        table = RawTable(paths.DATA_CSV, paths.SPLIT_MANIFEST)
        index = next(i for i, s in enumerate(table.split) if s == "train")
        source = {"opaque_row_id": table.opaque[index], "row_index": index, "source_split": "train", "variant": "N"}
        natural = table.request(source, 80)
        self.assertEqual(set(natural["credit"]), set(FEATURE_NAMES))
        self.assertEqual(table.request({**source, "variant": "O"}, 80)["credit"]["AGE"], 80)
        self.assertEqual(table.request({**source, "variant": "C"}, 80)["credit"]["PAY_AMT1"], -1)
        self.assertEqual(sum(v is None for v in table.request({**source, "variant": "M"}, 80)["credit"].values()), 1)
        test_index = next(i for i, s in enumerate(table.split) if s == "test")
        with self.assertRaises(RuntimeError):
            table.request({"opaque_row_id": table.opaque[test_index], "row_index": test_index, "source_split": "test", "variant": "N"}, 80)
        classes = [cls for cls, _ in malformed_mutations(natural, "EVIDENCE1|p=UDP1|o=AC")]
        self.assertEqual(len(classes), 7)


@unittest.skipUnless(HAVE_DATA and HAVE_SCORER, "scorer artifacts missing; run scripts 03 and 04")
class CompiledScorerTests(unittest.TestCase):
    def test_adapter_structure_and_parity(self):
        from compiled_credit_llm.compiler.compiler import load_compiled
        from compiled_credit_llm.compiler.ir import XGBoostJSONAdapter
        from compiled_credit_llm.compiler.parity import parity_statistics
        raw = json.loads((paths.SCORER / "xgboost_model.json").read_text())
        result = XGBoostJSONAdapter.adapt(raw, feature_scales=(1,) * 23, output_scale=10 ** 9)
        self.assertEqual(result.metadata["objective"], "binary:logistic"); self.assertGreater(result.metadata["tree_count"], 0)
        manifest, module, receipt = load_compiled(paths.COMPILED_SCORER)
        rows = pd.read_csv(paths.COMPILED_SCORER / "row_predictions.csv")
        test = rows["split"].eq("test").to_numpy()
        stats = parity_statistics(rows.loc[test, "native_raw_margin"], rows.loc[test, "compiled_raw_margin"], rows.loc[test, "target"])
        self.assertLessEqual(stats["max_absolute_margin_error"], 2.2e-6); self.assertEqual(stats["sign_inversions"], 0)
        self.assertEqual(stats["roc_auc_difference"], 0.0); self.assertEqual(stats["ks_difference"], 0.0)
        frame = pd.read_csv(paths.DATA_CSV, skiprows=1)
        X = torch.from_numpy(frame[list(FEATURE_NAMES)].to_numpy(np.int64)[:64])
        with torch.inference_mode():
            score = module(X, torch.zeros_like(X, dtype=torch.bool)).score
        torch.testing.assert_close(score, torch.from_numpy(np.rint(rows["compiled_raw_margin"].to_numpy()[:64] * 10 ** 9)), rtol=0, atol=0)


@unittest.skipUnless(HAVE_DATA and HAVE_BASE, "base checkpoint missing; run scripts 03 to 07")
class BaseCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        cls.model = AutoModelForImageTextToText.from_pretrained(paths.BASE_CHECKPOINT, trust_remote_code=True, low_cpu_mem_usage=True).eval()
        cls.tokenizer = AutoTokenizer.from_pretrained(paths.BASE_CHECKPOINT, trust_remote_code=True)
        cls.table = RawTable(paths.DATA_CSV, paths.SPLIT_MANIFEST)
        frame = pd.read_csv(paths.DATA_CSV, skiprows=1)
        cls.X = frame[list(FEATURE_NAMES)].to_numpy(np.int64)

    def test_policy_oracle_parity_on_two_thousand_rows(self):
        from compiled_credit_llm.policy.oracle import calculate_policy
        X = torch.from_numpy(self.X[:2000])
        with torch.inference_mode():
            result = self.model.score_credit(X, torch.zeros_like(X, dtype=torch.bool))
        policy = self.model.compiled_credit.demo_policy
        oracle = calculate_policy(self.X[:2000], result.score_fixed.to(torch.int64).numpy(), policy.dev_minimum.numpy(), policy.dev_maximum.numpy())
        for name, field in (("calculations", "policy_calculations"), ("flags", "policy_flags"), ("triggers", "policy_triggers"), ("outcome", "policy_outcome_index")):
            np.testing.assert_array_equal(getattr(result, field).numpy(), oracle[name])
        self.assertTrue(bool(result.verifier_passed.all()))
        np.testing.assert_array_equal(result.feature_contributions_fixed.sum(dim=1).numpy() + int(result.attribution_bias_fixed), result.score_fixed.to(torch.int64).numpy())

    def test_tamper_fails_closed_and_restores(self):
        X = torch.from_numpy(self.X[:2]); missing = torch.zeros_like(X, dtype=torch.bool)
        leaf = self.model.compiled_credit.experts[0].leaf_value
        original = leaf[0].clone()
        try:
            with torch.no_grad():
                leaf[0] += 1
            with self.assertRaisesRegex(RuntimeError, "tensor receipt mismatch"):
                self.model.score_credit(X, missing)
        finally:
            with torch.no_grad():
                leaf[0].copy_(original)
        with torch.inference_mode():
            self.assertTrue(bool(self.model.score_credit(X, missing).verifier_passed.all()))

    def test_assessment_without_language_model_and_refusals(self):
        request = self.table.natural(0)
        result = self.model.assess_credit(request, self.tokenizer, explain=False)
        payload = json.loads(result.output)
        self.assertEqual(result.status, "OK"); self.assertEqual(payload["explanation_status"], "NOT_GENERATED")
        self.assertIn(payload["decision"]["outcome_code"], ("II", "DD", "MR", "AC", "AS"))
        self.assertIn(payload["recourse"]["status"], ("REACHABLE", "ALREADY_BEST", "NOT_REACHABLE_BY_BEHAVIOUR", "NO_BEHAVIOURAL_MOVE_AVAILABLE", "NOT_REACHABLE_WITHIN_3_MOVES"))
        original = self.model.generate
        self.model.generate = lambda *a, **k: (_ for _ in ()).throw(AssertionError("language model called"))
        try:
            for cls, bad in malformed_mutations(request, result.evidence):
                refused = self.model.assess_credit(bad, self.tokenizer)
                self.assertEqual((refused.status, refused.refusal_class), ("REFUSE", cls))
        finally:
            self.model.generate = original

    def test_typed_inputs_fail_closed(self):
        X = torch.from_numpy(self.X[:4]); missing = torch.zeros_like(X, dtype=torch.bool)
        with self.assertRaises(TypeError):
            self.model.score_credit(X.double(), missing)
        with self.assertRaises(TypeError):
            self.model.score_credit(X[:, :-1], missing[:, :-1])
        with self.assertRaises(TypeError):
            self.model.score_credit(X, missing.to(torch.int64))
