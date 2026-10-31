# UCI_DEMO_POLICY_v1

> **Synthetic demonstration policy.** This is not a bank policy, a lending standard, a credit recommendation or a regulatory
> adverse-action framework. It exists so the checkpoint has a complete, transparent decision path to compute and explain.

## Inputs

The 23 UCI fields, as integers, plus a Boolean missing mask. Currency fields are New Taiwan dollars.

| Fields | Meaning (UCI documentation) |
|---|---|
| `LIMIT_BAL` | granted credit limit |
| `SEX` | 1 male, 2 female |
| `EDUCATION` | 1 graduate school, 2 university, 3 high school, 4 other; observed 0, 5, 6 are undocumented |
| `MARRIAGE` | 1 married, 2 single, 3 other; observed 0 is undocumented |
| `AGE` | years |
| `PAY_0`, `PAY_2` to `PAY_6` | repayment status, September back to April 2005; -1 paid duly, 1 to 9 months of delay; observed -2 and 0 are undocumented |
| `BILL_AMT1` to `BILL_AMT6` | statement amounts, September back to April 2005 |
| `PAY_AMT1` to `PAY_AMT6` | previous payment amounts, September back to April 2005 |

Demographic fields are model inputs because the benchmark booster uses them; the policy does not assert that this is
appropriate anywhere else.

## Calculations (integer floor division, ratios in basis points)

| Calculation | Formula |
|---|---|
| current utilization | `max(BILL_AMT1,0)*10000 // max(LIMIT_BAL,1)` |
| maximum utilization | `max(max(BILL_AMT1..6,0))*10000 // max(LIMIT_BAL,1)` |
| average utilization | `sum(max(BILL_AMT1..6,0))*10000 // (6*max(LIMIT_BAL,1))` |
| delinquency count | count of `PAY_0..PAY_6 > 0` |
| repeated delinquency | delinquency count >= 2 |
| maximum severity | `max(max(PAY_0..PAY_6),0)` |
| payment coverage | `sum(PAY_AMT1..6)*10000 // max(sum(max(BILL_AMT1..6,0)),1)` |
| bill trend | `(BILL_AMT1-BILL_AMT6)*10000 // max(LIMIT_BAL,1)` |
| bill dispersion | `sum(abs(6*bill_i - sum(bills))) // 36` (floor mean absolute deviation) |
| payment dispersion | `sum(abs(6*payment_i - sum(payments))) // 36` |
| zero-payment stress | payment total is 0 and current utilization >= 8,000 bp |
| positive bill total | `sum(max(BILL_AMT1..6,0))` |
| payment total | `sum(PAY_AMT1..6)` |

## Flags

| Flag | Condition |
|---|---|
| missing input | any missing-mask bit set |
| hard contract invalid | limit <= 0, age outside 0 to 120, status outside -2 to 9, or a negative payment |
| out of development range | any feature outside the min/max of the train, validation and calibration rows |
| unknown demographic category | sex, education or marriage outside the documented codes |
| undocumented repayment code | any status equal to -2 or 0 (informational; common in the source) |
| nonfinite or type invalid | always 0 after the typed contract; non-integers fail before scoring |

## Risk bands on the raw margin

`LOW < -1`, `MODERATE [-1, 0)`, `ELEVATED [0, 1)`, `HIGH >= 1`. Descriptive model bands, not calibrated grades.

## Thresholds

| Threshold | Value |
|---|---:|
| current high utilization | 8,000 bp |
| maximum high utilization | 9,000 bp |
| extreme maximum utilization | 15,000 bp |
| low payment coverage | 1,000 bp |
| rising bill trend | 3,000 bp |
| soft delinquency severity | 2 |
| hard delinquency severity | 3 |
| hard delinquency count | 4 |
| multiple soft triggers | 2 |

Every threshold is a demonstration assumption. Development bounds come from the non-test rows only. No threshold was
tuned on labels.

## Decision priority (first match)

1. **INVALID_INPUT**: missing input or hard contract invalid.
2. **DEMO_DECLINE**: severity >= 3, or delinquency count >= 4, or zero-payment stress with maximum utilization >= 100%, or
   maximum utilization >= 150% with any delinquency.
3. **MANUAL_REVIEW**: unknown demographic code, out of development range, HIGH band, or at least two soft triggers.
4. **APPROVE_WITH_CONTROLS**: one soft trigger or ELEVATED band.
5. **APPROVE_STANDARD**: nothing above.

Soft triggers: repeated delinquency (or severity >= 2), maximum utilization >= 90%, payment coverage below 10% with a
positive bill total, bill trend >= 30% of the limit. The trigger vector and the final-reason vector are both kept, so the
priority never hides which conditions were active.

## Probability

`model_probability = sigmoid(raw margin)`; `frozen_platt_probability = sigmoid(intercept + coefficient * raw margin)`
with the Platt map fitted on the calibration split. Both are benchmark outputs, not validated probabilities of default.

## Where it runs

Every calculation, flag, band, trigger, outcome, reason and probability is computed by `CheckpointDemoPolicy` as
`torch.int64` operations on buffers that the tensor verifier hashes. `compiled_credit_llm/policy/oracle.py` is the
independent NumPy oracle used by the compiler audit, the verification report and the tests.
