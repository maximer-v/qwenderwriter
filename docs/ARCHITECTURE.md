# Architecture

## One checkpoint, two kinds of weights

The checkpoint directory is a standard Transformers checkpoint with `trust_remote_code` model classes. Its single
`model.safetensors` file holds two kinds of tensors:

* the language backbone (Qwen3.5-0.8B, adapted by a merged LoRA), under the usual `model.*` names;
* the compiled credit engine, under `compiled_credit.*`: the tree ensemble, the TreeSHAP region tables and the policy
  buffers.

The model class `CompiledCreditLLMForConditionalGeneration` subclasses the Qwen3.5 class and registers a
`CheckpointCompiledCredit` submodule. Nothing else is needed at runtime: no XGBoost, no SHAP library, no policy service,
no renderer, no external verifier.

```
request (23 raw fields)
   │
   ▼
validate_raw_request ──REFUSE──► exact JSON refusal (7 classes, no model call)
   │ ok
   ▼
score_credit  (tensor-only)
   ├─ CheckpointTreeExpert ........ exact tree traversal on the integer lattice → fixed-point margin
   ├─ CheckpointTreeShapAttributor  per-tree region lookup → 23 fixed-point contributions, local accuracy exact
   ├─ CheckpointDemoPolicy ........ 13 calculations, 6 flags, 4 risk bands, 15 triggers, outcome, reasons, probability
   └─ tensor verifier ............. receipt hash of all compiled buffers + consistency checks on every output
   │
   ▼
evidence line  EVIDENCE1|p=..|s=..|q=..|o=..|b=..|f=..|t=..|c=..|xp=..|xn=..|r=..
   │
   ├─► facts()  →  deterministic JSON (decision, model, policy, drivers, calculations, data_quality)
   ├─► recourse() → smallest behavioural change to a better outcome, re-scored by score_credit
   ├─► explanation_prompt() → language model → one paragraph → verify_explanation()  (withheld on failure)
   └─► receipt  RECEIPT1|...|d=SYNTHETIC_NONREGULATORY_ASSOCIATIVE_NOT_CAUSAL
```

## Exact tree traversal as tensor arithmetic

Features are integers on a lattice (scale 1 for the UCI fields). For integer `x` and integer threshold `t`,
`clamp(relu(t - x), 0, 1)` is exactly the predicate `x < t`, including equality going right as in XGBoost. A native
floating threshold is normalised to `ceil(threshold * scale)`, which keeps the predicate identical for every lattice value;
the compiler proves this over every observed value and five integers around every split boundary (11.96 million comparisons,
zero mismatches on the UCI model). Missing values follow the tree's default direction through the same gates. Each leaf
stores its path as a row of gates; an AND over the row activates exactly one leaf per tree, and a matrix product
accumulates leaf values at scale 10^9. Float64 carries these integers exactly. The compiled margin equals the booster's
float32 margin to within 2.2e-6 on all 30,000 rows with zero sign inversions and identical ROC-AUC and KS.

## Exact TreeSHAP as a lookup

TreeSHAP is not constant across the rows that reach the same leaf, because cold subtrees contribute. It is constant across
every cell of the Cartesian product of a tree's split thresholds, because such a cell fixes every branch decision. The
compiler enumerates those cells for each tree (344,106 in total for the UCI model), queries the native booster once per
cell at compile time, quantises at 10^9 and corrects a sub-unit rounding residual on the largest component so that the 23
contributions plus the bias reproduce the compiled margin exactly in integers. At inference each tree computes a mixed-radix
cell index with tensor comparisons and gathers its 23-vector. Local accuracy is exact in fixed units; against native
float32 TreeSHAP the largest difference is 8.3e-7.

## Policy in integer arithmetic

`UCI_DEMO_POLICY_v1` (see `POLICY.md`) is thirteen floor-division calculations, six validity and semantic flags, four
risk bands on the raw margin, fifteen triggers and a first-match priority: invalid input, decline, manual review, approval
with controls, standard approval. All of it runs as `torch.int64` operations on the same buffers that the verifier hashes.
The probability is `sigmoid(margin)` in float64; a frozen Platt map is also carried.

## Fail-closed verification

`score_credit` hashes every compiled buffer (name, dtype, shape, raw bytes) and compares the digest with the receipt stored
in the config, caching by tensor version so an in-place change invalidates the cache. It also recomputes grade, decision
and probability bounds from the outputs and checks local accuracy and exactly-one risk band and outcome. Any failure
raises before any evidence leaves the function. The tamper tests exercise an in-memory leaf change, a policy bound change,
a changed expected hash, and a single flipped bit inside the safetensors file.

## The language model's job

The language model receives a compact JSON of the facts (decision, controlling rules, probability and band, driver labels,
calculations, data quality, proximity to the next better outcome) and a writing brief. It writes one paragraph. The verifier
then requires the outcome word, every controlling-rule phrase verbatim, the probability and band, the association caveat
and the recourse statement, and forbids any number not licensed by the evidence or the recourse moves, any feature label not
in the evidence, raw feature codes, quotation marks, brackets, currency signs, and causal, legal or guarantee wording. The
decision is never blocked by the paragraph: a failed paragraph is withheld and the violations are listed.

The writer was adapted with LoRA on targets written by a teacher model from exactly the facts JSON the checkpoint sends at
serving time, each target filtered by the same verifier. Selection used free generation on the real route only; teacher-forced
loss was never a selection criterion.

## Recourse

Candidate single moves are derived from the policy's own thresholds: all statuses on time, no status later than one
month, one delayed month only, payments raised to the coverage threshold, balances held under the high or extreme
utilization caps, the most recent balance flattened under the rising-trend cap. Combinations of up to three moves are
re-derived on the progressively modified vector so they compose exactly, and every candidate is re-scored by
`score_credit`. The engine reports the smallest improvement with the fewest moves and the best reachable outcome with the
fewest moves, with the resulting band and probability. Demographic fields and the credit limit are never touched; a
missing field, a contract violation, an unknown category or an out-of-development value is reported as a non-behavioural
blocker instead.
