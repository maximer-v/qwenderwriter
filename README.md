<p align="center">
  <img src="docs/assets/overview.png" alt="Qwenderwriter-0.8B: deterministic assessment, language writer, verifier" width="100%">
</p>

<h1 align="center">Qwenderwriter-0.8B</h1>

<p align="center"><b>A complete credit decision system, from scorecard to verified explanation, compiled inside one 0.8B language model.</b></p>

<p align="center">
  <a href="LICENSE"><img alt="License: CC BY-NC 4.0" src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="Base model: Qwen3.5-0.8B" src="https://img.shields.io/badge/base%20model-Qwen3.5--0.8B-orange.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-43%20passing-brightgreen.svg">
  <img alt="Author: Javier Mancilla" src="https://img.shields.io/badge/author-Javier%20Mancilla-1f4ca0.svg">
  <img alt="Affiliation: SquareOne Labs" src="https://img.shields.io/badge/affiliation-SquareOne%20Labs-lightgrey.svg">
</p>

One `from_pretrained` call loads a Qwen3.5-0.8B model whose safetensors file also carries a gradient-boosted scorecard,
its exact TreeSHAP attribution tables and a written credit policy as fixed-point tensors. Send a raw 23-field credit
request and get back one JSON document: the decision and the rule that controlled it, the probability of default and its
band, the drivers with exact Shapley contributions, the thirteen quantities the policy used, data-quality flags, the
smallest behavioural change that would reach a better outcome, a cryptographic receipt, and one plain-language
paragraph that the model has checked against its own numbers before releasing it.

The score, the drivers, the policy outcome and the recourse come from the compiled tensors. The transformer writes the
paragraph. A deterministic verifier decides whether the paragraph is attached or withheld. The decision is delivered
either way.

## Paper

The method, the guarantees and the full evaluation are described in the paper, included in this repository:

**[Qwenderwriter: a complete credit decision system, from scorecard to verified explanation, compiled inside one 0.8B language model](paper/Qwenderwriter.pdf)**
(Javier Mancilla and Tomás Tagliani, SquareOne Labs, 2026). The LaTeX source and figures are in [`paper/source/`](paper/source/).

## One request, one answer

<p align="center">
  <img src="docs/assets/sample_io.png" alt="A raw request and the raw assessment returned by Qwenderwriter-0.8B" width="100%">
</p>

The applicant above is declined by a policy rule (severe delinquency), is shown to be **one behavioural step away from
approval with controls** (bring the six repayment statuses current; the probability would fall from 69.9% to 11.7%), and receives a
verified paragraph plus a receipt from which an auditor can reproduce the whole decision without running the language
layers. The complete JSON is in [`examples/outputs/`](examples/outputs/), one file per example request.

```python
import json
from transformers import AutoModelForImageTextToText, AutoTokenizer

path = "artifacts/checkpoints/CompiledCreditLLM-0.8B-UCI"          # the built artefact
model = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

request = json.load(open("examples/requests/decline_close_to_approval.json"))
result = model.assess_credit(request, tokenizer)                     # explain=False skips the paragraph
assessment = json.loads(result.output)

print(assessment["decision"]["outcome"], assessment["model"]["probability_pct"], assessment["recourse"]["proximity"])
print(assessment["explanation_status"], assessment["receipt"][:60], "...")
```

```
demonstration decline 69.9% close
VERIFIED RECEIPT1|p=UDP1|s=842436674|q=0.698978159399|o=DD|b=E|f=UR ...
```

A malformed request (a missing field, a string where an integer belongs, a duplicate field, an attempt to override the
outcome, ...) is refused with an exact JSON refusal before any model code runs:

```json
{"protocol": "compiled-credit-llm/assessment-v1", "status": "REFUSE", "refusal_class": "OVERRIDE_ATTEMPT",
 "reason": "Checkpoint evidence and policy outcomes cannot be overridden.", "language_model_invoked": false, "receipt": "..."}
```

## What the file guarantees

| Question | Answer, and how it is tested |
|---|---|
| Is the compiled scorecard the trained model? | Yes. Zero sign inversions and identical ROC-AUC and KS on all 30,000 UCI rows; the largest margin difference is the booster's own float32 noise (`04_compile_scorer.py`). |
| Can the language weights touch the score? | No. Changing a language weight leaves every score field bit-identical on 1,000 rows (`14_tamper_determinism.py`). |
| Can the file be tampered with? | Every call re-hashes the compiled buffers against a SHA-256 receipt in the config. One flipped bit in the compiled region makes the next call refuse to score. |
| Can the paragraph invent a number? | Every number and feature name in it must be licensed by the evidence; anything else is withheld and the violations are listed. |
| Is the answer reproducible? | Same request, same bytes. One-by-one and batched scoring agree; CPU and GPU agree on every deterministic field and receipt. |
| What would change the decision? | The recourse engine re-scores candidate behavioural changes with the same compiled scorecard and reports the exact steps. Demographic fields never change. |

## Results on the UCI benchmark

Every number below is read from the build artefacts by `scripts/15_write_evaluation.py` into
[`docs/EVALUATION.md`](docs/EVALUATION.md).

| | |
|---|---|
| Scorecard, sealed test split (4,500 clients) | ROC-AUC 0.778 (95% CI 0.762 to 0.795), KS 0.437, PR-AUC 0.546, Brier 0.136 |
| Compilation parity, all 30,000 rows | 0 sign inversions, identical ROC-AUC and KS, TreeSHAP local accuracy exact in integer units |
| Verified paragraphs, held-back applications | 202 of 206 (98.1%); rare-rule applications 21 of 21 |
| Malformed request classes | 7 of 7 refused with the exact JSON, 0 language-model calls |
| Tamper and determinism tests | 9 of 9 pass |
| Assessment without the paragraph | 48 ms on a CPU |
| Full build, raw data to tested artefact | 58 minutes on one workstation (teacher targets reused) |

## Repository layout

```
compiled_credit_llm/
  compiler/     manifest schema, XGBoost JSON adapter, normalised tree IR, exact tensor modules, receipts, parity
  checkpoint/   the three files that ship inside the artefact: config, modeling (scorecard + TreeSHAP + policy), routing
  policy/       UCI_DEMO_POLICY_v1 as a NumPy oracle, and its compiler
  scorer/       XGBoost training, scorecard compilation, TreeSHAP compilation
  data/         UCI download, conversion and splits; raw-request reconstruction; malformed request classes
  explain/      development-row sampling, teacher-written and verifier-filtered targets, LoRA training, free-generation gate
  package/      base artefact assembly, adapter merge, CPU verification, tamper and determinism tests
scripts/        numbered command-line entry points for every stage, plus assess.py
notebooks/      end_to_end.ipynb: the whole pipeline, executed
examples/       UCI example requests and their assessments
paper/          the paper (PDF) and its LaTeX source
docs/           architecture, policy, output schema, evaluation, model card, adapting the method to your own portfolio
tests/          unit tests (no GPU, no artefact needed) plus artefact tests that run when the build exists
```

## Build it yourself

The built artefact is 1.8 GB and is not committed; the repository rebuilds it end to end. Stages 01 to 08 and 12 to 14
run on the CPU in minutes; 09 (teacher), 10 (LoRA) and 11 (free-generation gates) use a GPU.

```bash
pip install -e ".[test,notebook]"
python scripts/02_make_splits.py && python scripts/03_train_scorer.py
python scripts/04_compile_scorer.py && python scripts/05_compile_treeshap.py && python scripts/06_compile_policy.py
python scripts/07_build_base_checkpoint.py --base Qwen/Qwen3.5-0.8B
python scripts/08_sample_rows.py && python scripts/09_write_explanations.py                      # GPU, about 2 h
python scripts/10_train_explainer.py                                                             # GPU, about 20 min
python scripts/11_gate_explainer.py --adapter artifacts/adapters/explainer-lora/final_adapter --partition smoke
python scripts/12_merge_checkpoint.py && python scripts/13_verify_checkpoint.py && python scripts/14_tamper_determinism.py
python scripts/15_write_evaluation.py
python scripts/assess.py examples/requests/*.json --output-dir examples/outputs
```

| Stage | Script | What it does |
|---|---|---|
| 01 | `01_download_uci.py` | fetch the official UCI XLS, verify its hash, convert to CSV (the CSV is shipped, so this is optional) |
| 02 | `02_make_splits.py` | stratified client-level splits: train 18,000 / validation 4,500 / calibration 3,000 / test 4,500 |
| 03 | `03_train_scorer.py` | XGBoost search on validation, calibration on the calibration split, locked test metrics |
| 04 | `04_compile_scorer.py` | booster to exact fixed-point tensors; parity on all 30,000 rows |
| 05 | `05_compile_treeshap.py` | exact TreeSHAP as per-tree region lookup tables, checked against the native booster on every row |
| 06 | `06_compile_policy.py` | policy thresholds, development bounds and calibrator into buffers; audited against the NumPy oracle |
| 07 | `07_build_base_checkpoint.py` | official Qwen3.5-0.8B tensors plus the `compiled_credit.*` buffers in one safetensors file |
| 08 | `08_sample_rows.py` | development rows and variants for the explanation writer (the test split is never read) |
| 09 | `09_write_explanations.py` | a teacher writes one paragraph per row from the serving-time facts; the verifier filters |
| 10 | `10_train_explainer.py` | LoRA on the language layers only; compiled buffers hashed before and after |
| 11 | `11_gate_explainer.py` | free generation on the real request path: verified rate, refusals with the language model forbidden |
| 12 | `12_merge_checkpoint.py` | merge the adapter into one self-contained artefact |
| 13 | `13_verify_checkpoint.py` | strict reload, compiled identity, 30,000-row parity, oracle parity, refusals |
| 14 | `14_tamper_determinism.py` | leaf, bound, digest and file-bit tamper tests; repeat, batch and CPU/GPU determinism |
| 15 | `15_write_evaluation.py` | `docs/EVALUATION.md` from the artefacts on disk |

The notebook [`notebooks/end_to_end.ipynb`](notebooks/end_to_end.ipynb) runs the same stages in order with commentary
and the resulting numbers; `notebooks/run_notebook.sh` executes it in place. Set `CCL_BASE_MODEL` to a local directory
with the official Qwen3.5-0.8B weights and `CCL_TEACHER` to a local copy of the teacher when running offline.

`examples/requests/` holds seven UCI-shaped requests: a decline close to approval, an approval with controls, a standard
approval, a manual review with a non-behavioural blocker, a missing field, an override attempt and a string where an
integer is required. `examples/outputs/` holds the assessment each one produces.

## Use it on your own portfolio

The artefact built here is for the UCI benchmark and its demonstration policy. The method transfers: the feature
contract, the policy (tensor code plus NumPy oracle) and the phrase tables that define the writer's vocabulary are the
three things you replace, and the same gates, verification and tamper tests then apply to your build.
[`docs/ADAPTING_TO_YOUR_PORTFOLIO.md`](docs/ADAPTING_TO_YOUR_PORTFOLIO.md) walks through it stage by stage. The
compiled region is a stack of tree experts, so several scorecards can share one file with a compiled router in front of
them.

## Documentation

| | |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | how trees, TreeSHAP and the policy become exact tensors, and how the receipt and verifier work |
| [`docs/POLICY.md`](docs/POLICY.md) | every calculation, flag, threshold and the decision priority of the demonstration policy |
| [`docs/OUTPUT_SCHEMA.md`](docs/OUTPUT_SCHEMA.md) | the assessment and refusal JSON, field by field |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | all numbers, generated from the artefacts |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | intended use, data, training, evaluation, boundaries |
| [`docs/ADAPTING_TO_YOUR_PORTFOLIO.md`](docs/ADAPTING_TO_YOUR_PORTFOLIO.md) | replacing the feature contract, the policy and the phrase tables |

## Boundaries

* The policy is a demonstration policy on a 2005 benchmark. Nothing here is a lending, regulatory or adverse-action
  decision, and the model probability is a benchmark output, not a validated probability of default.
* TreeSHAP contributions are associations in a frozen model, not causes.
* Recourse describes what the compiled scorecard and the policy would do with different behavioural inputs; it is not advice.
* The explanation paragraph is the only free text in the output. It is verified, not trusted; when verification fails it is withheld.

## Authorship

Qwenderwriter was conceived and built by **Javier Mancilla** (SquareOne Labs), who owns the code and the released model
and publishes them from a personal GitHub account. **Tomás Tagliani** (SquareOne Labs) is the pair reviewer of the design,
the code and the paper, and co-author of the paper. Contact: javier.m@squareonecap.com.

## Citation

```bibtex
@misc{qwenderwriter2026,
  title  = {Qwenderwriter: a complete credit decision system, from scorecard to verified explanation, compiled inside one 0.8B language model},
  author = {Mancilla, Javier and Tagliani, Tom\'as},
  year   = {2026},
  note   = {SquareOne Labs},
  url    = {https://github.com/maximer-v/qwenderwriter},
  note   = {Paper: paper/Qwenderwriter.pdf in the repository}
}
```

## License

Copyright 2026 Javier Mancilla. Code, documentation and the built artefact: [CC BY-NC 4.0](LICENSE) (non-commercial). The Qwen3.5 base model is
Apache-2.0, the teacher Qwen3-4B-Instruct-2507 is Apache-2.0 and the UCI data is CC BY 4.0; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

<p align="center"><sub>Authored and owned by <b>Javier Mancilla</b>. Pair review by <b>Tomás Tagliani</b>. Both at <b>SquareOne Labs</b>, a frontier technology lab applying generative AI, quantum computing and neuromorphic/thermodynamic computing to financial services.</sub></p>
