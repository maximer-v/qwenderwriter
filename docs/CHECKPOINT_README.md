# CompiledCreditLLM-0.8B-UCI (Qwenderwriter-0.8B)

One self-contained checkpoint: a Qwen3.5-0.8B language backbone plus, in the same `model.safetensors`, a compiled
XGBoost credit scorer, exact fixed-point TreeSHAP attribution, the thirteen calculations and rules of the synthetic
`UCI_DEMO_POLICY_v1`, a tensor verifier, a deterministic request router, a recourse engine and an explanation verifier.
No XGBoost, SHAP, policy, rendering or verification service runs at inference. The compiled tensors compute the score
and the decision; the language model writes one paragraph about them, which the checkpoint verifies before releasing it.

```python
import json
from transformers import AutoModelForImageTextToText, AutoTokenizer

model = AutoModelForImageTextToText.from_pretrained(".", trust_remote_code=True).eval()
tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
result = model.assess_credit({"action": "ASSESS", "credit": {...23 UCI fields...}}, tokenizer)
assessment = json.loads(result.output)   # see inference_example.py
```

`assess_credit` validates the raw 23-field request (seven malformed classes return an exact JSON refusal and never reach
the language model), runs the compiled scorer (score, probability, TreeSHAP contributions, calculations, flags, triggers,
policy outcome and reasons, with the compiled buffers re-hashed against the receipt in `config.json` on every call),
computes recourse over behavioural fields by re-scoring candidates with the same scorer, assembles the JSON assessment,
and finally lets the language model write the `explanation` paragraph, which is verified against the evidence; a failed
paragraph is withheld (`explanation_status: WITHHELD_BY_VERIFIER`) and the rest of the assessment is still returned.
`assess_credit(request, explain=False)` skips the language model entirely. `score_credit(features, missing_mask)` gives
typed tensor access to the scorer.

Files worth reading: `verification_report.json` (strict reload, compiled identity, 30,000-row parity, oracle parity,
refusals), `tamper_determinism_report.json`, `adapter_selection_gate.json` and `merged_generation_gate/` (free-generation
gates), `policy_specification.json` (every calculation, flag, threshold and the decision priority), `feature_labels.json`
(the only feature names the writer may use), `compiled_scorer_manifest.json` and the receipts.

`UCI_DEMO_POLICY_v1` is a synthetic demonstration policy on the 2005 UCI benchmark. This checkpoint is a research
artefact, not a lending, regulatory or adverse-action system; the model probability is a benchmark output, contributions
are associations, recourse is a statement about the compiled model and the policy, and the paragraph is verified text, not
trusted text. Licensed CC BY-NC 4.0; the Qwen3.5-0.8B base is Apache-2.0 and the UCI data is CC BY 4.0.
