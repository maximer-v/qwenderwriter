# Output schema

`model.assess_credit(request, tokenizer)` returns a `RouteResult` with `status` (`OK` or `REFUSE`), `output` (JSON text),
`refusal_class`, `generation_invoked`, `evidence` and `prompt`. `result.payload` parses the JSON.

## Request

```json
{"action": "ASSESS", "credit": {"LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 39,
 "PAY_0": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 2, "PAY_6": 0,
 "BILL_AMT1": 117868, "BILL_AMT2": 114483, "BILL_AMT3": 118855, "BILL_AMT4": 125357, "BILL_AMT5": 121853, "BILL_AMT6": 124731,
 "PAY_AMT1": 0, "PAY_AMT2": 6216, "PAY_AMT3": 10000, "PAY_AMT4": 0, "PAY_AMT5": 5000, "PAY_AMT6": 4552}}
```

* `action` is optional and must be `ASSESS`. `credit` must contain exactly the 23 fields, each an integer or `null`
  (an explicit null is a policy input and produces `INVALID_INPUT`; it is not a malformed request).
* `bound_evidence` is optional: a previously returned evidence line that the checkpoint re-derives and compares.

## Assessment (`status: OK`)

| Field | Content |
|---|---|
| `protocol` | `compiled-credit-llm/assessment-v1` |
| `status` | `OK` |
| `decision` | `outcome_code` (II, DD, MR, AC, AS), `outcome` in words, `level` (hard, review, soft, approve), `controlling_rules` (the rule phrases that selected the outcome), `note` |
| `model` | `probability` (12 digits, `sigmoid(raw margin)`), `probability_pct`, `risk_band` (low, moderate, elevated, high), `raw_margin` |
| `policy` | `id`, `priority`, and `active` rules per level (hard, review, soft), whether or not they controlled |
| `drivers` | `increasing_risk` and `decreasing_risk`: up to two features each with `feature` label, `code` and signed TreeSHAP `contribution` (margin units) |
| `calculations` | the thirteen policy calculations in readable units: utilizations and coverage in percent, counts, totals, two booleans |
| `data_quality` | the active flag phrases (missing input, contract invalid, out of development range, unknown category, undocumented repayment code) |
| `recourse` | see below |
| `explanation` | one paragraph written by the language model, or `null` |
| `explanation_status` | `VERIFIED`, `WITHHELD_BY_VERIFIER` or `NOT_GENERATED` (`explain=False`) |
| `explanation_violations` | present only when withheld: the verifier's reasons |
| `boundary` | the standing disclaimer |
| `receipt` | `RECEIPT1|p=UDP1|s=<fixed-point margin>|q=<probability>|o=<outcome>|b=<band>|f=<flags>|t=<triggers>|c=<13 calculations>|xp=<top positive drivers>|xn=<top negative drivers>|r=<controlling reasons>|d=SYNTHETIC_NONREGULATORY_ASSOCIATIVE_NOT_CAUSAL` |

### Recourse

| `status` | Meaning |
|---|---|
| `REACHABLE` | a better outcome is reachable by behavioural changes alone; `next_better` and `best_reachable` describe the moves, the number of fields changed and the model band and probability after the change; `proximity` is close (1 move), moderate (2) or far (3) |
| `ALREADY_BEST` | the case is a standard approval |
| `NOT_REACHABLE_BY_BEHAVIOUR` | a non-behavioural blocker applies: missing field, contract violation, unknown demographic category, value outside the development range (listed in `non_behavioural_blockers`) |
| `NO_BEHAVIOURAL_MOVE_AVAILABLE` | no candidate move applies to this case |
| `NOT_REACHABLE_WITHIN_3_MOVES` | candidates exist but none reaches a better outcome within three moves |

Only repayment statuses, statement balances and payments move. Demographic fields and the credit limit never do.

## Refusal (`status: REFUSE`)

```json
{"protocol": "compiled-credit-llm/assessment-v1", "status": "REFUSE", "refusal_class": "MALFORMED_TYPE",
 "reason": "Raw credit values must be integers or explicit nulls.", "language_model_invoked": false,
 "receipt": "RECEIPT1|status=REFUSE|e=MALFORMED_TYPE|n=Raw credit values must be integers or explicit nulls.|d=SYNTHETIC_NONREGULATORY_ASSOCIATIVE_NOT_CAUSAL"}
```

Classes, in precedence order: `OVERRIDE_ATTEMPT` (an `override` key), `UNSUPPORTED_REQUEST` (another action or an unknown
key), `MISSING_FIELD` (no `credit`, or a field set not equal to the 23), `MALFORMED_TYPE` (a non-integer, a Boolean, a
string), `DUPLICATE_CONFLICT` (a field given twice), `OUT_OF_RANGE` (outside signed 64-bit), `EVIDENCE_MISMATCH`
(`bound_evidence` differs from the recomputed evidence). The language model is never called for a refusal.

## Typed access

`model.score_credit(features, missing_mask)` takes a `[batch, 23]` integer tensor and a Boolean mask of the same shape
and returns the raw tensors: fixed-point margin, probability, contributions, calculations, flags, triggers, outcome,
reasons, verifier result. It never runs the language model.
