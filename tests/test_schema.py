import copy
import unittest

from compiled_credit_llm.compiler.schema import CompiledManifest, ManifestError


def manifest_dict():
    return {
        "schema_version": "1.0", "model_version": "model-1", "policy_version": "policy-1",
        "features": [
            {"name": "income", "index": 0, "dtype": "fixed_int", "scale": 1, "minimum": -100, "maximum": 100, "missing_allowed": True},
            {"name": "debt", "index": 1, "dtype": "fixed_int", "scale": 1, "minimum": -100, "maximum": 100, "missing_allowed": True},
        ],
        "experts": [
            {"id": "trees-a", "kind": "tree_ensemble", "ir_path": "a.json", "output_scale": 100, "reason_namespace": "credit"},
            {"id": "trees-b", "kind": "tree_ensemble", "ir_path": "b.json", "output_scale": 100, "reason_namespace": "credit"},
        ],
        "routing": {"kind": "static_weighted", "weight_numerators": [3, 1], "denominator": 4, "attention_layers": 0},
        "calibration": {"input_scale": 100, "output_scale": 100, "multiplier": 2, "divisor": 1, "offset": 10, "minimum": -1000, "maximum": 1000},
        "policy": {"decision_threshold": 50, "approve_when": "gte", "grade_bands": [
            {"grade": "D", "minimum": -1000, "maximum": 49}, {"grade": "A", "minimum": 50, "maximum": 1000}]},
        "reasons": [{"code": "DEBT", "label": "Debt burden", "adverse_when": "negative"}, {"code": "INCOME", "label": "Income", "adverse_when": "negative"}],
        "precision": {"feature_bits": 16, "accumulator_bits": 32, "output_bits": 16, "max_abs_accumulator": 100000, "max_trees": 10, "max_depth": 5, "max_leaves": 30},
        "receipt": {"algorithm": "sha256", "manifest_sha256": "", "ir_sha256": ["", ""], "compiled_sha256": ""},
    }


class ManifestTests(unittest.TestCase):
    def test_round_trip(self):
        manifest = CompiledManifest.from_dict(manifest_dict())
        self.assertEqual(CompiledManifest.from_dict(manifest.to_dict()), manifest)

    def test_unknown_field_fails_closed(self):
        raw = manifest_dict(); raw["surprise"] = True
        with self.assertRaises(ManifestError):
            CompiledManifest.from_dict(raw)

    def test_nested_unknown_field_fails_closed(self):
        raw = manifest_dict(); raw["features"][0]["surprise"] = True
        with self.assertRaises(ManifestError):
            CompiledManifest.from_dict(raw)

    def test_invalid_router_fails_closed(self):
        raw = manifest_dict(); raw["routing"]["weight_numerators"] = [1, 1]
        with self.assertRaises(ManifestError):
            CompiledManifest.from_dict(raw)

    def test_noncontiguous_grade_bands_fail_closed(self):
        raw = manifest_dict(); raw["policy"]["grade_bands"][1]["minimum"] = 51
        with self.assertRaises(ManifestError):
            CompiledManifest.from_dict(raw)
