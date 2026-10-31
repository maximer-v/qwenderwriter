"""The NumPy policy oracle against hand-computed boundary cases (no checkpoint needed)."""
import unittest

import numpy as np

from compiled_credit_llm.policy.oracle import OUTCOME_NAMES, TRIGGER_NAMES, calculate_policy

BASE = np.array([10000, 1, 1, 1, 30, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)
DEV_MIN = np.full(23, -10_000_000, dtype=np.int64)
DEV_MAX = np.full(23, 10_000_000, dtype=np.int64)


def outcome(row, margin=-2_000_000_000, missing=None):
    result = calculate_policy(row[None, :], np.array([margin]), DEV_MIN, DEV_MAX, None if missing is None else missing[None, :])
    return OUTCOME_NAMES[int(result["outcome"][0])], result


class PolicyOracleTests(unittest.TestCase):
    def test_clean_row_is_standard_approval(self):
        name, result = outcome(BASE)
        self.assertEqual(name, "APPROVE_STANDARD")
        self.assertEqual(int(result["final_reasons"][0, TRIGGER_NAMES.index("APPROVE_STANDARD_NO_ADVERSE_TRIGGER")]), 1)

    def test_severity_three_declines_and_severity_two_controls(self):
        row = BASE.copy(); row[5] = 3
        self.assertEqual(outcome(row)[0], "DEMO_DECLINE")
        row[5] = 2
        self.assertEqual(outcome(row)[0], "APPROVE_WITH_CONTROLS")

    def test_four_delayed_months_decline(self):
        row = BASE.copy(); row[5:9] = 1
        self.assertEqual(outcome(row)[0], "DEMO_DECLINE")
        row = BASE.copy(); row[5:7] = 1
        self.assertEqual(outcome(row)[0], "APPROVE_WITH_CONTROLS")  # repeated delinquency is one soft control

    def test_two_soft_controls_route_to_review(self):
        row = BASE.copy(); row[5:7] = 1; row[11] = 9500  # repeated delinquency + utilization 95%
        self.assertEqual(outcome(row)[0], "MANUAL_REVIEW")

    def test_risk_band_boundaries(self):
        for margin, band in ((-1_000_000_001, 0), (-1_000_000_000, 1), (-1, 1), (0, 2), (999_999_999, 2), (1_000_000_000, 3)):
            self.assertEqual(int(outcome(BASE, margin)[1]["risk_band"][0]), band)
        self.assertEqual(outcome(BASE, 1_000_000_000)[0], "MANUAL_REVIEW")
        self.assertEqual(outcome(BASE, 0)[0], "APPROVE_WITH_CONTROLS")

    def test_missing_and_contract_violations_are_invalid(self):
        missing = np.zeros(23, dtype=bool); missing[4] = True
        self.assertEqual(outcome(BASE, missing=missing)[0], "INVALID_INPUT")
        row = BASE.copy(); row[17] = -1
        self.assertEqual(outcome(row)[0], "INVALID_INPUT")
        row = BASE.copy(); row[4] = 121
        self.assertEqual(outcome(row)[0], "INVALID_INPUT")

    def test_unknown_demographic_and_out_of_development_review(self):
        row = BASE.copy(); row[2] = 0
        self.assertEqual(outcome(row)[0], "MANUAL_REVIEW")
        result = calculate_policy(BASE[None, :], np.array([-2_000_000_000]), DEV_MIN, np.full(23, 25, dtype=np.int64))
        self.assertEqual(OUTCOME_NAMES[int(result["outcome"][0])], "MANUAL_REVIEW")

    def test_calculations_are_floor_basis_points(self):
        row = BASE.copy(); row[11] = 3333; row[16] = 0; row[17] = 100
        result = outcome(row)[1]["calculations"][0]
        self.assertEqual(int(result[0]), 3333)   # current utilization bp
        self.assertEqual(int(result[6]), 300)    # payment coverage: 100 / 3333 -> 300 bp (floor)
        self.assertEqual(int(result[7]), 3333)   # bill trend bp
