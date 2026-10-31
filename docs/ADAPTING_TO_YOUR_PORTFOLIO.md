# Adapting the method to your own portfolio

The released checkpoint scores the UCI benchmark under a synthetic policy. The method is portable: a tree model you
already trust, compiled into tensors with a proof of parity; exact attribution; a policy you can read; a recourse engine;
a language model that writes about the result and is checked against it. This page lists what changes and what stays.

## What stays

* `compiled_credit_llm/compiler/`: the manifest, the XGBoost JSON adapter, the exact tensor modules, receipts and parity
  statistics are feature-agnostic. Any `binary:logistic` gbtree booster with integer features (or features you quantise to
  an integer lattice with a declared scale) compiles unchanged.
* `scorer/treeshap.py`: region-table compilation works for any such booster (requires default-right splits, i.e. no
  missing values at training time, or a small extension for default-left).
* The verification pattern: receipt hash of compiled buffers, local-accuracy check, exactly-one outcome, refusals before
  the language model, the explanation verifier's structure.
* `explain/`: the teacher-and-verifier loop, the LoRA trainer and the free-generation gate are generic over the facts JSON.

## What you replace

1. **Feature contract.** `FEATURE_NAMES` in `checkpoint/routing.py` and `data/uci.py`, the feature labels in
   `scorer/treeshap.py` (`FEATURE_LABELS`, which become the `feature` strings the writer may use), and the request
   reconstruction in `data/requests.py`. Keep features integer-valued; quantise currency to whole units or cents and
   declare the scale in the manifest.
2. **Policy.** `CheckpointDemoPolicy` in `checkpoint/modeling_compiled_credit_llm.py` and its NumPy twin in
   `policy/oracle.py` are specific to the UCI fields (six months of statuses, bills and payments). Write your own pair:
   calculations in integer arithmetic, flags, bands, triggers, a first-match priority, and a `final_reasons` vector. Keep
   the oracle independent of the tensor code; the compile audit and the verification report compare them on every row.
3. **Phrase tables.** `FLAG_PHRASES`, `TRIGGER_PHRASES`, `OUTCOME_WORDS` and `CALC_LABELS` in `checkpoint/routing.py`
   are the only words the writer is allowed to use for rules, flags and calculations. The verifier checks the controlling
   phrases verbatim, so choose phrases that read naturally inside a sentence.
4. **Recourse moves.** `_moves` in `checkpoint/routing.py` derives candidate changes from the policy thresholds. Define
   the behavioural fields of your portfolio and the moves that make sense (and never include protected or immutable
   fields).
5. **Boundary text and policy id.** `BOUNDARY`, `POLICY_ID` and the disclaimer string.

## Stage by stage

| Stage | UCI release | Your portfolio |
|---|---|---|
| data | `01`, `02` | your table, your split manifest (keep a client-level, time-aware split if you have dates) |
| scorer | `03` trains a small XGBoost search | bring your production booster as `xgboost_model.json` (XGBoost 3.1.2 JSON, integer feature types) and a Platt calibrator |
| compile | `04`, `05`, `06` | unchanged, unchanged, your policy compiler |
| base checkpoint | `07` | unchanged (base model of your choice in the Qwen3.5 family; other families need a different parent class) |
| rows | `08` samples variants that exercise every outcome and rare trigger | write a sampler for your outcomes and rare rules |
| targets | `09` with a permissively licensed teacher | same; the verifier is yours now, so the targets follow your phrases |
| train, gate, merge, verify, tamper | `10` to `14` | unchanged |

## Things that are easy to get wrong

* **Thresholds that depend on labels.** The policy thresholds here are assumptions. If you fit thresholds, fit them on
  development rows and keep the test split sealed.
* **Letting the writer see numbers it may not use.** The facts JSON is the writer's whole world; keep raw contributions and
  margins out of it, as the UCI prompt does, and list every number you want quotable.
* **Trusting training loss.** Select adapters with the free-generation gate on the real route. A low teacher-forced loss
  tells you nothing about what the model emits when it is free.
* **Protected attributes.** The UCI booster uses demographic fields because the benchmark does. Your scorer should not,
  and the recourse engine must never propose changing them.
* **Model probability.** `sigmoid(margin)` is the booster's probability on its training distribution. Calibrate on your
  own calibration split and say so in the boundary text.

## Minimal checklist before you ship

1. `04` reports zero sign inversions and identical ROC-AUC and KS between booster and compiled scorer on all rows.
2. `05` reports exact local accuracy in fixed units on all rows.
3. `06` reports oracle parity on all rows and no label use in threshold selection.
4. `13` passes: strict reload, compiled identity, parity against the base checkpoint, refusals with the language model forbidden.
5. `14` passes: four tamper refusals, scores unchanged under a language-weight change, byte-identical repeats, batch and
   device agreement on every deterministic field.
6. The gate's verified rate on held-back development rows is one you are comfortable withholding the remainder of.
