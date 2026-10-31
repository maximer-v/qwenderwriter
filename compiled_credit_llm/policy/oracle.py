"""Independent NumPy oracle for synthetic UCI_DEMO_POLICY_v1."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)
CALCULATION_NAMES = (
    "current_utilization_bp", "maximum_utilization_bp", "average_utilization_bp",
    "delinquency_count", "repeated_delinquency", "maximum_delinquency_severity",
    "payment_coverage_bp", "bill_trend_bp", "bill_mean_absolute_deviation",
    "payment_mean_absolute_deviation", "zero_payment_high_utilization",
    "positive_bill_total", "payment_total",
)
FLAG_NAMES = (
    "missing_input", "hard_contract_invalid", "out_of_development_range",
    "unknown_demographic_category", "undocumented_repayment_code", "nonfinite_or_type_invalid",
)
RISK_BAND_NAMES = ("LOW", "MODERATE", "ELEVATED", "HIGH")
OUTCOME_NAMES = ("INVALID_INPUT", "DEMO_DECLINE", "MANUAL_REVIEW", "APPROVE_WITH_CONTROLS", "APPROVE_STANDARD")
TRIGGER_NAMES = (
    "INVALID_MISSING", "INVALID_CONTRACT", "DECLINE_SEVERE_DELINQUENCY",
    "DECLINE_ZERO_PAYMENT_HIGH_UTILIZATION", "DECLINE_EXTREME_UTILIZATION_WITH_DELINQUENCY",
    "REVIEW_UNKNOWN_CATEGORY", "REVIEW_OUT_OF_DEVELOPMENT_RANGE", "REVIEW_HIGH_MODEL_RISK",
    "REVIEW_MULTIPLE_SOFT_TRIGGERS", "CONTROL_REPEATED_DELINQUENCY", "CONTROL_HIGH_UTILIZATION",
    "CONTROL_LOW_PAYMENT_COVERAGE", "CONTROL_RISING_BILL_TREND", "CONTROL_ELEVATED_MODEL_RISK",
    "APPROVE_STANDARD_NO_ADVERSE_TRIGGER",
)
THRESHOLDS = {
    "utilization_scale_bp": 10_000,
    "current_high_utilization_bp": 8_000,
    "maximum_high_utilization_bp": 9_000,
    "maximum_extreme_utilization_bp": 15_000,
    "low_payment_coverage_bp": 1_000,
    "rising_bill_trend_bp": 3_000,
    "soft_delinquency_severity": 2,
    "hard_delinquency_severity": 3,
    "hard_delinquency_count": 4,
    "multiple_soft_trigger_count": 2,
    "risk_moderate_margin_fixed": -1_000_000_000,
    "risk_elevated_margin_fixed": 0,
    "risk_high_margin_fixed": 1_000_000_000,
}


def _mad_floor(values: np.ndarray) -> np.ndarray:
    count = values.shape[1]
    total = values.sum(axis=1)
    numerator = np.abs(values * count - total[:, None]).sum(axis=1)
    return np.floor_divide(numerator, count * count)


def calculate_policy(X: np.ndarray, margin_fixed: np.ndarray, dev_minimum: np.ndarray,
                     dev_maximum: np.ndarray, missing_mask: np.ndarray | None = None):
    X = np.asarray(X, dtype=np.int64)
    margin_fixed = np.asarray(margin_fixed, dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != 23 or margin_fixed.shape != (len(X),):
        raise ValueError("oracle expects [rows,23] and [rows]")
    missing = np.zeros_like(X, dtype=bool) if missing_mask is None else np.asarray(missing_mask, dtype=bool)
    limit = np.maximum(X[:, 0], 1)
    status = X[:, 5:11]
    bills = X[:, 11:17]
    payments = X[:, 17:23]
    positive_bills = np.maximum(bills, 0)
    positive_bill_total = positive_bills.sum(axis=1)
    payment_total = payments.sum(axis=1)
    current_utilization = np.floor_divide(positive_bills[:, 0] * 10_000, limit)
    maximum_utilization = np.floor_divide(positive_bills.max(axis=1) * 10_000, limit)
    average_utilization = np.floor_divide(positive_bill_total * 10_000, 6 * limit)
    delinquency_count = (status > 0).sum(axis=1).astype(np.int64)
    repeated_delinquency = (delinquency_count >= 2).astype(np.int64)
    maximum_severity = np.maximum(status.max(axis=1), 0)
    payment_coverage = np.floor_divide(payment_total * 10_000, np.maximum(positive_bill_total, 1))
    bill_trend = np.floor_divide((bills[:, 0] - bills[:, 5]) * 10_000, limit)
    bill_mad = _mad_floor(bills)
    payment_mad = _mad_floor(payments)
    zero_payment_stress = ((payment_total == 0) & (current_utilization >= 8_000)).astype(np.int64)
    calculations = np.column_stack((
        current_utilization, maximum_utilization, average_utilization, delinquency_count,
        repeated_delinquency, maximum_severity, payment_coverage, bill_trend, bill_mad,
        payment_mad, zero_payment_stress, positive_bill_total, payment_total,
    )).astype(np.int64)

    missing_input = missing.any(axis=1)
    hard_contract_invalid = ((X[:, 0] <= 0) | (X[:, 4] < 0) | (X[:, 4] > 120) |
                             (status < -2).any(axis=1) | (status > 9).any(axis=1) |
                             (payments < 0).any(axis=1))
    out_of_development = ((X < dev_minimum) | (X > dev_maximum)).any(axis=1)
    unknown_demographic = (~np.isin(X[:, 1], [1, 2]) | ~np.isin(X[:, 2], [1, 2, 3, 4]) |
                           ~np.isin(X[:, 3], [1, 2, 3]))
    undocumented_repayment = np.isin(status, [-2, 0]).any(axis=1)
    nonfinite_or_type_invalid = np.zeros(len(X), dtype=bool)
    flags = np.column_stack((missing_input, hard_contract_invalid, out_of_development,
                             unknown_demographic, undocumented_repayment,
                             nonfinite_or_type_invalid)).astype(np.int64)

    risk_band = np.zeros(len(X), dtype=np.int64)
    risk_band[margin_fixed >= -1_000_000_000] = 1
    risk_band[margin_fixed >= 0] = 2
    risk_band[margin_fixed >= 1_000_000_000] = 3

    severe = (maximum_severity >= 3) | (delinquency_count >= 4)
    zero_stress_decline = (zero_payment_stress == 1) & (maximum_utilization >= 10_000)
    extreme_with_delinquency = (maximum_utilization >= 15_000) & (delinquency_count >= 1)
    repeated = (repeated_delinquency == 1) | (maximum_severity >= 2)
    high_utilization = maximum_utilization >= 9_000
    low_coverage = (positive_bill_total > 0) & (payment_coverage < 1_000)
    rising_trend = bill_trend >= 3_000
    soft_count = np.column_stack((repeated, high_utilization, low_coverage, rising_trend)).sum(axis=1)
    triggers = np.column_stack((
        missing_input, hard_contract_invalid, severe, zero_stress_decline, extreme_with_delinquency,
        unknown_demographic, out_of_development, risk_band == 3, soft_count >= 2,
        repeated, high_utilization, low_coverage, rising_trend, risk_band == 2,
        np.zeros(len(X), dtype=bool),
    )).astype(np.int64)

    invalid = missing_input | hard_contract_invalid
    decline = severe | zero_stress_decline | extreme_with_delinquency
    review = unknown_demographic | out_of_development | (risk_band == 3) | (soft_count >= 2)
    controls = (soft_count >= 1) | (risk_band == 2)
    outcome = np.full(len(X), 4, dtype=np.int64)
    outcome[controls] = 3
    outcome[review] = 2
    outcome[decline] = 1
    outcome[invalid] = 0
    triggers[:, 14] = outcome == 4
    outcome_one_hot = np.eye(5, dtype=np.int64)[outcome]
    risk_one_hot = np.eye(4, dtype=np.int64)[risk_band]

    final_reasons = np.zeros_like(triggers)
    final_reasons[:, 0:2] = triggers[:, 0:2] * (outcome == 0)[:, None]
    final_reasons[:, 2:5] = triggers[:, 2:5] * (outcome == 1)[:, None]
    final_reasons[:, 5:14] = triggers[:, 5:14] * np.isin(outcome, [2, 3])[:, None]
    final_reasons[:, 14] = outcome == 4
    return {
        "calculations": calculations, "flags": flags, "risk_band": risk_band,
        "risk_one_hot": risk_one_hot, "triggers": triggers, "outcome": outcome,
        "outcome_one_hot": outcome_one_hot, "final_reasons": final_reasons,
    }
