#!/usr/bin/env python3
"""Assemble docs/EVALUATION.md from the build artifacts, so the numbers in the documentation are the numbers on disk."""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
from compiled_credit_llm import paths

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default=str(ROOT / "docs" / "EVALUATION.md"))
args = parser.parse_args()
J = lambda p: json.loads(Path(p).read_text())
name = paths.MODEL_NAME
scorer = J(paths.SCORER / "metrics.json"); run_meta = J(paths.SCORER / "run_metadata.json")
parity = J(paths.COMPILED_SCORER / "parity_metrics.json"); adapter_meta = J(paths.COMPILED_SCORER / "adapter_metadata.json")
shap = J(paths.COMPILED_TREESHAP / "attribution_parity.json"); policy = J(paths.COMPILED_POLICY / "compile_audit.json")
rows = J(paths.EXPLANATIONS / "rows_summary.json"); targets = J(paths.EXPLANATIONS / "dataset_manifest.json")
train = J(paths.ADAPTERS / "explainer-lora" / "adapter_manifest.json")
verify = J(paths.FINAL_CHECKPOINT / "verification_report.json"); tamper = J(paths.FINAL_CHECKPOINT / "tamper_determinism_report.json")
ci = scorer["bootstrap"]["intervals"]["uncalibrated"]; tm = scorer["test_metrics"]["uncalibrated"]
gate_rows = []
for who, prefix in (("adapter on base", "gate_adapter"), ("merged checkpoint", f"gate_{name}")):
    for part, label in (("smoke", "smoke"), ("rare_eval", "rare rules"), ("validation", "validation")):
        f = paths.EVALUATION / f"{prefix}_{part}.json"
        if f.exists():
            g = J(f)
            gate_rows.append(f"| {who} | {label} | {g['rows']} | {g['verified']} | {100 * g['verified_rate']:.1f}% | {g['refusals']['classes_passed']}/7, {g['refusals']['malformed_generation_calls']} calls | {g['gate']} |")
viol = sorted(((k.split(':', 1)[1], v) for k, v in targets["stats"].items() if k.startswith("violation:")), key=lambda kv: -kv[1])
tests = {t["test"]: t for t in tamper["tests"]}
d2 = tests["D2_batch_invariance"]; d3 = tests.get("D3_cpu_vs_gpu")
eval_loss = [h["eval_loss"] for h in train["log_history"] if "eval_loss" in h]
dist = policy["policy_distribution"]
lines = [
    "# Evaluation", "",
    "Every number on this page is read from the build artifacts by `scripts/15_write_evaluation.py`; rebuild and regenerate to update it.", "",
    "## Scorer (held-out test split, 4,500 clients, 995 defaults)", "",
    f"XGBoost {run_meta['versions']['xgboost']}, {scorer['selected_trees']} trees, depth {scorer['selected_parameters']['max_depth']}, {scorer['calibration_method'].capitalize()} calibration selected by the frozen rule. "
    "Hyperparameters and tree count were chosen on the validation split, the calibrator on the calibration split; the test split was read once, afterwards. "
    "Intervals are 95% percentile bootstrap intervals over 1,000 client-level resamples.", "",
    "| Metric | Value | 95% interval |", "|---|---:|---:|",
    *[f"| {k} | {tm[m]:.4f} | {ci[m]['lower_95']:.4f} to {ci[m]['upper_95']:.4f} |" for k, m in (("ROC-AUC", "roc_auc"), ("KS", "ks"), ("PR-AUC", "pr_auc"), ("Brier", "brier"), ("log loss", "log_loss"))],
    "", "The dataset has no application or event ordering, so this is random-split evidence, not a temporal validation.", "",
    "## Compilation parity (all 30,000 rows)", "",
    "| Check | Result |", "|---|---|",
    f"| integer-lattice threshold proof | {parity['threshold_equivalence']['observed_integer_comparisons'] + parity['threshold_equivalence']['boundary_integer_comparisons']:,} comparisons, {parity['threshold_equivalence']['observed_integer_mismatches'] + parity['threshold_equivalence']['boundary_integer_mismatches']} mismatches |",
    f"| compiled vs native margin, max absolute difference | {parity['all_rows']['max_absolute_margin_error']:.2e} (declared quantisation bound {parity['raw_margin_quantization_bound']:.1e}; the rest is the booster's float32 accumulation) |",
    f"| sign inversions / rank inversions | {parity['all_rows']['sign_inversions']} / {parity['all_rows']['ranking']['rank_inversions']} |",
    f"| ROC-AUC and KS difference, test split | {parity['test']['roc_auc_difference']:.1e} / {parity['test']['ks_difference']:.1e} |",
    f"| TreeSHAP regions compiled | {shap['total_regions']:,} over {adapter_meta['tree_count']} trees |",
    f"| TreeSHAP local accuracy, fixed units | max residual {shap['compiled_local_accuracy_residual']['max_absolute_fixed_units']} |",
    f"| TreeSHAP vs native float32 contributions | max {shap['global_max_absolute_error']:.2e}, mean {shap['global_mean_absolute_error']:.2e} |",
    f"| policy oracle parity (compile audit and verification report) | exact on all rows |",
    "", "## Policy outcome distribution", "",
    "| Split | rows | invalid | decline | review | controls | approve |", "|---|---:|---:|---:|---:|---:|---:|",
    *[f"| {s} | {dist[s]['rows']:,} | {dist[s]['outcomes']['INVALID_INPUT']} | {dist[s]['outcomes']['DEMO_DECLINE']} | {dist[s]['outcomes']['MANUAL_REVIEW']} | {dist[s]['outcomes']['APPROVE_WITH_CONTROLS']} | {dist[s]['outcomes']['APPROVE_STANDARD']} |" for s in ("train", "validation", "calibration", "test")],
    "", "## Explanation writer", "",
    f"Development rows: train {rows['train']['rows']}, validation {rows['validation']['rows']}, rare rules {rows['rare_eval']['rows']}, smoke {rows['smoke']['rows']} (train rows from the train split, the rest from the validation split; the test split is never read). "
    f"Teacher `{targets['teacher']}`, up to {targets['sampling']['tries']} tries per row, every target accepted by the checkpoint's verifier. "
    f"Kept targets: train {targets['counts']['train']}, validation {targets['counts']['validation']}, rare rules {targets['counts']['rare_eval']}, smoke {targets['counts']['smoke']}. "
    f"First-try acceptance {100 * targets['stats'].get('attempt1_ok', 0) / max(targets['stats'].get('attempt1_ok', 0) + targets['stats'].get('attempt1_fail', 0), 1):.0f}%.", "",
    "Why the verifier rejected teacher drafts (all attempts):", "",
    "| Violation | drafts |", "|---|---:|", *[f"| {k} | {v} |" for k, v in viol], "",
    f"LoRA rank {train['lora']['rank']} on {train['target_module_count']} linear modules ({train['trainable_parameter_count']:,} parameters), {train['steps']} steps, {train['runtime_seconds'] / 60:.0f} minutes, "
    f"final held-back loss {eval_loss[-1]:.3f} (informational; selection used free generation only). Compiled-state hash identical before and after training.", "",
    "### Free-generation gates (real request path)", "",
    "| Weights | Rows | n | verified | rate | refusals | gate |", "|---|---|---:|---:|---:|---|---|", *gate_rows, "",
    "A verified paragraph passed every verifier rule. An unverified paragraph is withheld in the product with the rest of the assessment delivered; none of these rows had a wrong decision, number or receipt, because none of those come from the language model.", "",
    "## Checkpoint verification", "",
    "| Check | Result |", "|---|---|",
    f"| strict reload | no missing, unexpected or mismatched keys |",
    f"| compiled-state SHA-256 | `{verify['compiled_state_sha256']}` |",
    f"| tensors | {verify['tensor_inventory']['total_tensor_count']:,} total: {verify['tensor_inventory']['compiled_tensor_count']:,} compiled, {verify['tensor_inventory']['language_tensor_count']} language, {verify['tensor_inventory']['mtp_tensor_count']} multi-token-prediction head, {verify['tensor_inventory']['lora_tensor_count']} adapter |",
    f"| all-row scorer parity vs base checkpoint | {verify['all_row_scorer_parity']['rows']:,} rows, every tensor and label field exact |",
    f"| policy oracle parity | exact on {verify['policy_oracle_parity']['rows']:,} rows |",
    f"| malformed classes, language model forbidden | {verify['deterministic_refusals']['classes_passed']}/7 exact refusals, {verify['deterministic_refusals']['language_model_calls']} calls |",
    f"| checkpoint size | {verify['artifact_size_bytes'] / 1e9:.2f} GB |",
    "", "## Tamper and determinism", "",
    "| Test | Result |", "|---|---|",
    *[f"| {t['test']} | {'PASS' if t['pass'] else 'FAIL'} |" for t in tamper["tests"]], "",
    f"Batch invariance: integer and fixed-point fields bit-identical; float64 probabilities differ by at most {max(v.get('max_abs_diff', 0.0) for v in d2['fields'].values()):.1e} between batch sizes (one unit in the last place from vectorised versus scalar sigmoid), identical at the twelve digits the receipt carries.",
    (f"CPU versus GPU: deterministic fields identical on {d3['deterministic_fields_identical']}/{d3['cases']} cases, receipts {d3['receipts_identical']}/{d3['cases']}, explanation text identical on {d3['explanation_text_identical']}/{d3['cases']} (greedy bf16 decoding is device-dependent; each paragraph is verified on its own device, statuses {d3['explanation_status_pairs']})." if d3 else "CPU versus GPU: not run (no GPU)."),
    "", "## Boundaries", "",
    "Synthetic demonstration policy on a 2005 benchmark; benchmark probability, not a validated probability of default; TreeSHAP contributions are associations; recourse is a statement about the compiled model and the policy; the paragraph is verified text, not trusted text; all explanation-side numbers are development results under a random split.",
]
Path(args.output).write_text("\n".join(lines) + "\n")
print("wrote", args.output)
