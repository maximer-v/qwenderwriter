# Evaluation

Every number on this page is read from the build artifacts by `scripts/15_write_evaluation.py`; rebuild and regenerate to update it.

## Scorer (held-out test split, 4,500 clients, 995 defaults)

XGBoost 3.1.2, 113 trees, depth 4, Platt calibration selected by the frozen rule. Hyperparameters and tree count were chosen on the validation split, the calibrator on the calibration split; the test split was read once, afterwards. Intervals are 95% percentile bootstrap intervals over 1,000 client-level resamples.

| Metric | Value | 95% interval |
|---|---:|---:|
| ROC-AUC | 0.7779 | 0.7615 to 0.7947 |
| KS | 0.4366 | 0.4079 to 0.4694 |
| PR-AUC | 0.5456 | 0.5124 to 0.5776 |
| Brier | 0.1358 | 0.1289 to 0.1425 |
| log loss | 0.4325 | 0.4146 to 0.4494 |

The dataset has no application or event ordering, so this is random-split evidence, not a temporal validation.

## Compilation parity (all 30,000 rows)

| Check | Result |
|---|---|
| integer-lattice threshold proof | 11,959,257 comparisons, 0 mismatches |
| compiled vs native margin, max absolute difference | 2.11e-06 (declared quantisation bound 5.7e-08; the rest is the booster's float32 accumulation) |
| sign inversions / rank inversions | 0 / 59 |
| ROC-AUC and KS difference, test split | 0.0e+00 / 0.0e+00 |
| TreeSHAP regions compiled | 344,106 over 113 trees |
| TreeSHAP local accuracy, fixed units | max residual 0 |
| TreeSHAP vs native float32 contributions | max 8.26e-07, mean 1.19e-08 |
| policy oracle parity (compile audit and verification report) | exact on all rows |

## Policy outcome distribution

| Split | rows | invalid | decline | review | controls | approve |
|---|---:|---:|---:|---:|---:|---:|
| train | 18,000 | 0 | 1836 | 5567 | 3974 | 6623 |
| validation | 4,500 | 0 | 465 | 1428 | 1013 | 1594 |
| calibration | 3,000 | 0 | 287 | 917 | 681 | 1115 |
| test | 4,500 | 0 | 440 | 1419 | 1011 | 1630 |

## Explanation writer

Development rows: train 1390, validation 220, rare rules 24, smoke 5 (train rows from the train split, the rest from the validation split; the test split is never read). Teacher `Qwen/Qwen3-4B-Instruct-2507`, up to 4 tries per row, every target accepted by the checkpoint's verifier. Kept targets: train 1252, validation 206, rare rules 21, smoke 5. First-try acceptance 82%.

Why the verifier rejected teacher drafts (all attempts):

| Violation | drafts |
|---|---:|
| unlicensed numbers | 641 |
| controlling rule missing | 285 |
| sentences | 4 |
| length | 1 |

LoRA rank 16 on 186 linear modules (10,822,656 parameters), 157 steps, 21 minutes, final held-back loss 0.135 (informational; selection used free generation only). Compiled-state hash identical before and after training.

### Free-generation gates (real request path)

| Weights | Rows | n | verified | rate | refusals | gate |
|---|---|---:|---:|---:|---|---|
| adapter on base | smoke | 5 | 4 | 80.0% | 7/7, 0 calls | PASS |
| adapter on base | rare rules | 21 | 21 | 100.0% | 7/7, 0 calls | PASS |
| adapter on base | validation | 206 | 200 | 97.1% | 7/7, 0 calls | PASS |
| merged checkpoint | smoke | 5 | 4 | 80.0% | 7/7, 0 calls | PASS |
| merged checkpoint | rare rules | 21 | 21 | 100.0% | 7/7, 0 calls | PASS |
| merged checkpoint | validation | 206 | 202 | 98.1% | 7/7, 0 calls | PASS |

A verified paragraph passed every verifier rule. An unverified paragraph is withheld in the product with the rest of the assessment delivered; none of these rows had a wrong decision, number or receipt, because none of those come from the language model.

## Checkpoint verification

| Check | Result |
|---|---|
| strict reload | no missing, unexpected or mismatched keys |
| compiled-state SHA-256 | `4d62c3df39b4c25f8d85e9842f530ed3ab57dbd6c5d120a5cc12252aeaca57b6` |
| tensors | 3,542 total: 3,054 compiled, 473 language, 15 multi-token-prediction head, 0 adapter |
| all-row scorer parity vs base checkpoint | 30,000 rows, every tensor and label field exact |
| policy oracle parity | exact on 30,000 rows |
| malformed classes, language model forbidden | 7/7 exact refusals, 0 calls |
| checkpoint size | 1.84 GB |

## Tamper and determinism

| Test | Result |
|---|---|
| T1_compiled_leaf_tamper | PASS |
| T2_policy_bound_tamper | PASS |
| T3_config_hash_tamper | PASS |
| T0_restored_after_tamper | PASS |
| T4_language_weight_tamper_scores_unchanged | PASS |
| D1_same_request_twice_cpu | PASS |
| D2_batch_invariance | PASS |
| T5_file_bit_flip | PASS |
| D3_cpu_vs_gpu | PASS |

Batch invariance: integer and fixed-point fields bit-identical; float64 probabilities differ by at most 1.1e-16 between batch sizes (one unit in the last place from vectorised versus scalar sigmoid), identical at the twelve digits the receipt carries.
CPU versus GPU: deterministic fields identical on 5/5 cases, receipts 5/5, explanation text identical on 4/5 (greedy bf16 decoding is device-dependent; each paragraph is verified on its own device, statuses [['VERIFIED', 'VERIFIED'], ['VERIFIED', 'VERIFIED'], ['VERIFIED', 'VERIFIED'], ['VERIFIED', 'VERIFIED'], ['WITHHELD_BY_VERIFIER', 'WITHHELD_BY_VERIFIER']]).

## Boundaries

Synthetic demonstration policy on a 2005 benchmark; benchmark probability, not a validated probability of default; TreeSHAP contributions are associations; recourse is a statement about the compiled model and the policy; the paragraph is verified text, not trusted text; all explanation-side numbers are development results under a random split.
