# Model card: CompiledCreditLLM-0.8B-UCI (released as Qwenderwriter-0.8B)

## Summary

A Qwen3.5-0.8B checkpoint whose safetensors file also carries a compiled credit decision engine: an XGBoost scorer
compiled to exact fixed-point tensors, exact TreeSHAP attribution as per-tree region tables, the synthetic
`UCI_DEMO_POLICY_v1` in integer arithmetic, a recourse engine and an explanation verifier. The language backbone was
adapted with a merged LoRA to write one verified paragraph per assessment. The backbone and the compiled buffers share a
file but no arithmetic: changing a language weight leaves every score field bit-identical, and changing a compiled buffer
makes the checkpoint refuse to score.

## Intended use

Research and demonstration of verifiable, self-contained credit assessment with a language model confined to explanation.
Reference implementation for adapting the method to another portfolio (see `docs/ADAPTING_TO_YOUR_PORTFOLIO.md`).

## Out of scope

Any lending, regulatory, adverse-action or consumer-facing decision. The policy is synthetic, the data is a 2005
benchmark, the probability is a benchmark output, the booster uses demographic fields because the benchmark does, and
the paragraph is development-validated only.

## Inputs and outputs

Input: the 23 UCI fields as integers or explicit nulls (`docs/OUTPUT_SCHEMA.md`). Output: a JSON assessment with
decision, controlling rules, probability and band, active policy rules, drivers with signed contributions, thirteen
calculations, data-quality flags, recourse, boundary, receipt, and one paragraph with its verification status; or an exact
JSON refusal for malformed input.

## Training

* Scorer: XGBoost 3.1.2, bounded search on the validation split, Platt calibration on the calibration split, test split
  read once. Compiled with parity proven on all 30,000 rows.
* Explanation writer: LoRA (rank 16) on the language layers only. Targets were written by Qwen3-4B-Instruct-2507 from
  the serving-time facts JSON and kept only when the checkpoint's verifier accepted them. Selection by free generation on
  the real request path; teacher-forced loss was never a criterion. Rows came from the train and validation splits only.

## Evaluation

`docs/EVALUATION.md` in the repository and, inside the checkpoint, `verification_report.json`,
`tamper_determinism_report.json`, `adapter_selection_gate.json` and `merged_generation_gate/`. Headline: the compiled scorer
has the booster's held-out ROC-AUC and KS with zero sign inversions (one of 10.1 million ordered test pairs swaps, a near-tie); the merged checkpoint refuses all seven malformed
classes without a language-model call; tamper tests refuse 4/4 and the language-weight test leaves scores bit-identical;
the verified-paragraph rates per gate set are in the evaluation document.

## Limitations

The verifier checks numbers, names, rule phrases and forbidden constructions, not helpfulness. Explanation results are
development results under a random split. Paragraph wording can differ between CPU and GPU under greedy bf16 decoding;
each device's paragraph is verified on its own. English only.

## License and provenance

Checkpoint and code: CC BY-NC 4.0. Base model Qwen3.5-0.8B: Apache-2.0 (tensors copied unchanged, then adapted). Teacher
Qwen3-4B-Instruct-2507: Apache-2.0 (not part of the checkpoint). Data: UCI dataset 350, CC BY 4.0, I-Cheng Yeh.
