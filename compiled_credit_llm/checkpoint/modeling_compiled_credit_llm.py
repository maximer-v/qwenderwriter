"""Qwen3.5 language backbone with an internal compiled credit scorer.

The scorer (gradient-boosted trees, TreeSHAP attribution and the policy) lives in persistent tensor buffers under
``compiled_credit.*`` inside the same checkpoint as the language model. ``score_credit`` runs only tensor
operations and never touches the language backbone; the language backbone never touches the score.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

import torch
from torch import Tensor, nn
from transformers import Qwen3_5ForConditionalGeneration

from .configuration_compiled_credit_llm import CompiledCreditLLMConfig
from .routing import AssessmentMixin

CHECKPOINT_FORMAT = "compiled-credit-llm/v1"
AUTHORITY = "COMPILED_XGBOOST_SCORER"
EVIDENCE_CONTRACT = "compiled-credit-evidence/v1"
_DTYPE = {"torch.float64": torch.float64, "torch.int64": torch.int64}
_TOP_BUFFERS = (
    "routing_weights", "routing_denominator", "calibration", "output_bounds",
    "policy_threshold", "approve_gte", "grade_bounds",
)
_EXPERT_BUFFERS = (
    "feature_index", "threshold", "missing_left", "edge_left", "path_mask",
    "path_length", "leaf_value", "leaf_reason", "base_score",
)
REQUIRED_METADATA = {
    "attribution_local_tolerance_fixed", "attribution_sha256", "attribution_tree_specs", "authority", "buffer_specs",
    "checkpoint_format_version", "compiled_sha256", "evidence_contract", "expert_count", "feature_count",
    "feature_maximums", "feature_minimums", "feature_names", "feature_scales", "grade_labels", "ir_sha256",
    "manifest_sha256", "max_model_reasons", "missing_allowed", "model_version", "output_scale", "policy_version",
    "reason_metadata", "scoring_compiled_sha256", "policy_id", "policy_status", "policy_sha256",
    "policy_calculation_names", "policy_flag_names", "risk_band_names", "policy_outcome_names",
    "policy_trigger_names", "policy_warning", "probability_warning",
}


def _empty(spec):
    dtype = _DTYPE.get(spec.get("dtype"))
    if dtype is None or not isinstance(spec.get("shape"), list):
        raise ValueError("unsupported compiled-credit buffer specification")
    return torch.zeros(tuple(spec["shape"]), dtype=dtype)


class CheckpointTreeExpert(nn.Module):
    """Exact tree-ensemble traversal on the integer lattice (one leaf per tree activates)."""

    def __init__(self, specs):
        super().__init__()
        if set(specs) != set(_EXPERT_BUFFERS):
            raise ValueError("compiled tree buffer schema mismatch")
        for name in _EXPERT_BUFFERS:
            self.register_buffer(name, _empty(specs[name]), persistent=True)

    def forward(self, features: Tensor, missing_mask: Tensor):
        x = features.to(dtype=torch.float64)
        missing = missing_mask.to(dtype=torch.float64)
        selected = x[:, self.feature_index]
        selected_missing = missing[:, self.feature_index]
        less = torch.clamp(torch.relu(self.threshold - selected), max=1.0)
        left_gate = self.missing_left * torch.clamp(less + selected_missing, max=1.0)
        left_gate = left_gate + (1.0 - self.missing_left) * torch.relu(less - selected_missing)
        edge_gate = self.edge_left * left_gate + (1.0 - self.edge_left) * (1.0 - left_gate)
        edge_gate = edge_gate * self.path_mask
        active = torch.clamp(torch.relu(edge_gate.sum(dim=-1) - self.path_length + 1.0), max=1.0)
        return active.matmul(self.leaf_value) + self.base_score, active.matmul(self.leaf_reason)


class CheckpointShapRegionFeature(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.register_buffer("feature_index", torch.zeros((), dtype=torch.int64), persistent=True)
        self.register_buffer("thresholds", torch.zeros((spec["threshold_count"],), dtype=torch.int64), persistent=True)
        self.register_buffer("stride", torch.zeros((), dtype=torch.int64), persistent=True)

    def forward(self, features: Tensor, missing_mask: Tensor) -> Tensor:
        values = features[:, self.feature_index].to(dtype=torch.int64)
        missing = missing_mask[:, self.feature_index]
        right = (values[:, None] >= self.thresholds[None, :]).to(dtype=torch.int64)
        right = torch.maximum(right, missing[:, None].to(dtype=torch.int64))
        return right.sum(dim=-1) * self.stride


class CheckpointShapTree(nn.Module):
    def __init__(self, spec, feature_count):
        super().__init__()
        self.features = nn.ModuleList([CheckpointShapRegionFeature(value) for value in spec["features"]])
        self.register_buffer(
            "contribution_table", torch.zeros((spec["region_count"], feature_count), dtype=torch.int64),
            persistent=True,
        )

    def forward(self, features: Tensor, missing_mask: Tensor) -> Tensor:
        region = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device)
        for feature in self.features:
            region = region + feature(features, missing_mask)
        return self.contribution_table[region]


class CheckpointTreeShapAttributor(nn.Module):
    """Exact TreeSHAP as a region lookup: every tree stores one fixed-point contribution vector per split region."""

    def __init__(self, specs, feature_count):
        super().__init__()
        self.trees = nn.ModuleList([CheckpointShapTree(value, feature_count) for value in specs])
        self.register_buffer("bias_fixed", torch.zeros((), dtype=torch.int64), persistent=True)

    def forward(self, features: Tensor, missing_mask: Tensor) -> Tensor:
        contributions = torch.zeros(
            (features.shape[0], self.trees[0].contribution_table.shape[1]),
            dtype=torch.int64, device=features.device,
        )
        for tree in self.trees:
            contributions = contributions + tree(features, missing_mask)
        return contributions


class CheckpointDemoPolicyOutput(NamedTuple):
    calculations: Tensor
    flags: Tensor
    risk_band_index: Tensor
    risk_band_one_hot: Tensor
    triggers: Tensor
    outcome_index: Tensor
    outcome_one_hot: Tensor
    final_reasons: Tensor
    model_probability: Tensor
    frozen_platt_probability: Tensor


class CheckpointDemoPolicy(nn.Module):
    """Synthetic demonstration policy UCI_DEMO_POLICY_v1 in integer tensor arithmetic; never a production or
    regulatory policy. See docs/POLICY.md for every formula and threshold."""

    def __init__(self, feature_count):
        super().__init__()
        self.register_buffer("dev_minimum", torch.zeros(feature_count, dtype=torch.int64), persistent=True)
        self.register_buffer("dev_maximum", torch.zeros(feature_count, dtype=torch.int64), persistent=True)
        self.register_buffer("thresholds", torch.zeros(13, dtype=torch.int64), persistent=True)
        self.register_buffer("platt", torch.zeros(2, dtype=torch.float64), persistent=True)
        self.register_buffer("declared_margin_bounds_fixed", torch.zeros(2, dtype=torch.int64), persistent=True)

    @staticmethod
    def _mad_floor(values: Tensor) -> Tensor:
        count = values.shape[1]
        total = values.sum(dim=1)
        numerator = torch.abs(values * count - total[:, None]).sum(dim=1)
        return torch.div(numerator, count * count, rounding_mode="floor")

    def forward(self, features: Tensor, missing_mask: Tensor, score_fixed: Tensor) -> CheckpointDemoPolicyOutput:
        x = features.to(dtype=torch.int64)
        missing = missing_mask.to(dtype=torch.bool)
        limit = torch.clamp(x[:, 0], min=1)
        status, bills, payments = x[:, 5:11], x[:, 11:17], x[:, 17:23]
        positive_bills = torch.clamp(bills, min=0)
        positive_bill_total = positive_bills.sum(dim=1)
        payment_total = payments.sum(dim=1)
        scale = self.thresholds[0]
        current_utilization = torch.div(positive_bills[:, 0] * scale, limit, rounding_mode="floor")
        maximum_utilization = torch.div(torch.amax(positive_bills, dim=1) * scale, limit, rounding_mode="floor")
        average_utilization = torch.div(positive_bill_total * scale, 6 * limit, rounding_mode="floor")
        delinquency_count = (status > 0).to(torch.int64).sum(dim=1)
        repeated_delinquency = (delinquency_count >= 2).to(torch.int64)
        maximum_severity = torch.clamp(torch.amax(status, dim=1), min=0)
        payment_coverage = torch.div(payment_total * scale, torch.clamp(positive_bill_total, min=1), rounding_mode="floor")
        bill_trend = torch.div((bills[:, 0] - bills[:, 5]) * scale, limit, rounding_mode="floor")
        bill_mad = self._mad_floor(bills)
        payment_mad = self._mad_floor(payments)
        zero_payment_stress = ((payment_total == 0) & (current_utilization >= self.thresholds[1])).to(torch.int64)
        calculations = torch.stack((
            current_utilization, maximum_utilization, average_utilization, delinquency_count,
            repeated_delinquency, maximum_severity, payment_coverage, bill_trend, bill_mad,
            payment_mad, zero_payment_stress, positive_bill_total, payment_total,
        ), dim=1)

        missing_input = torch.any(missing, dim=1)
        contract_invalid = ((x[:, 0] <= 0) | (x[:, 4] < 0) | (x[:, 4] > 120) |
                            torch.any(status < -2, dim=1) | torch.any(status > 9, dim=1) |
                            torch.any(payments < 0, dim=1))
        out_of_development = torch.any((x < self.dev_minimum) | (x > self.dev_maximum), dim=1)
        unknown_demographic = (~torch.isin(x[:, 1], x.new_tensor([1, 2])) |
                               ~torch.isin(x[:, 2], x.new_tensor([1, 2, 3, 4])) |
                               ~torch.isin(x[:, 3], x.new_tensor([1, 2, 3])))
        undocumented_repayment = torch.any(torch.isin(status, x.new_tensor([-2, 0])), dim=1)
        nonfinite_or_type_invalid = torch.zeros_like(missing_input)
        flags = torch.stack((missing_input, contract_invalid, out_of_development, unknown_demographic,
                             undocumented_repayment, nonfinite_or_type_invalid), dim=1).to(torch.int64)

        margin = score_fixed.to(torch.int64)
        risk_band = torch.zeros_like(margin)
        risk_band = torch.where(margin >= self.thresholds[10], torch.ones_like(risk_band), risk_band)
        risk_band = torch.where(margin >= self.thresholds[11], torch.full_like(risk_band, 2), risk_band)
        risk_band = torch.where(margin >= self.thresholds[12], torch.full_like(risk_band, 3), risk_band)
        risk_one_hot = torch.nn.functional.one_hot(risk_band, num_classes=4).to(torch.int64)

        severe = (maximum_severity >= self.thresholds[7]) | (delinquency_count >= self.thresholds[8])
        zero_stress_decline = (zero_payment_stress == 1) & (maximum_utilization >= scale)
        extreme_with_delinquency = (maximum_utilization >= self.thresholds[3]) & (delinquency_count >= 1)
        repeated = (repeated_delinquency == 1) | (maximum_severity >= self.thresholds[6])
        high_utilization = maximum_utilization >= self.thresholds[2]
        low_coverage = (positive_bill_total > 0) & (payment_coverage < self.thresholds[4])
        rising_trend = bill_trend >= self.thresholds[5]
        soft_count = torch.stack((repeated, high_utilization, low_coverage, rising_trend), dim=1).to(torch.int64).sum(dim=1)
        invalid = missing_input | contract_invalid
        decline = severe | zero_stress_decline | extreme_with_delinquency
        review = unknown_demographic | out_of_development | (risk_band == 3) | (soft_count >= self.thresholds[9])
        controls = (soft_count >= 1) | (risk_band == 2)
        outcome = torch.full_like(risk_band, 4)
        outcome = torch.where(controls, torch.full_like(outcome, 3), outcome)
        outcome = torch.where(review, torch.full_like(outcome, 2), outcome)
        outcome = torch.where(decline, torch.ones_like(outcome), outcome)
        outcome = torch.where(invalid, torch.zeros_like(outcome), outcome)
        outcome_one_hot = torch.nn.functional.one_hot(outcome, num_classes=5).to(torch.int64)
        triggers = torch.stack((
            missing_input, contract_invalid, severe, zero_stress_decline, extreme_with_delinquency,
            unknown_demographic, out_of_development, risk_band == 3, soft_count >= self.thresholds[9],
            repeated, high_utilization, low_coverage, rising_trend, risk_band == 2, outcome == 4,
        ), dim=1).to(torch.int64)
        final_reasons = torch.zeros_like(triggers)
        final_reasons[:, 0:2] = triggers[:, 0:2] * (outcome == 0).to(torch.int64)[:, None]
        final_reasons[:, 2:5] = triggers[:, 2:5] * (outcome == 1).to(torch.int64)[:, None]
        final_reasons[:, 5:14] = triggers[:, 5:14] * ((outcome == 2) | (outcome == 3)).to(torch.int64)[:, None]
        final_reasons[:, 14] = (outcome == 4).to(torch.int64)
        margin_float = score_fixed.to(torch.float64) / 1_000_000_000.0
        model_probability = torch.sigmoid(margin_float)
        frozen_platt_probability = torch.sigmoid(self.platt[1] + self.platt[0] * margin_float)
        return CheckpointDemoPolicyOutput(
            calculations, flags, risk_band, risk_one_hot, triggers, outcome, outcome_one_hot,
            final_reasons, model_probability, frozen_platt_probability,
        )


class CheckpointCompiledOutput(NamedTuple):
    score: Tensor
    expert_scores: Tensor
    decision: Tensor
    grade_index: Tensor
    grade_one_hot: Tensor
    reason_contributions: Tensor


class CheckpointCompiledCredit(nn.Module):
    """Tensor-only fixed-point tree, policy, grade, and reason-code path."""

    def __init__(self, metadata):
        super().__init__()
        specs = metadata["buffer_specs"]
        if set(specs) != set(_TOP_BUFFERS) | {"experts"}:
            raise ValueError("compiled-credit top-level buffer schema mismatch")
        expert_specs = specs["experts"]
        if not isinstance(expert_specs, list) or len(expert_specs) != metadata["expert_count"]:
            raise ValueError("compiled-credit expert count mismatch")
        self.experts = nn.ModuleList([CheckpointTreeExpert(value) for value in expert_specs])
        for name in _TOP_BUFFERS:
            self.register_buffer(name, _empty(specs[name]), persistent=True)
        self.attribution = CheckpointTreeShapAttributor(metadata["attribution_tree_specs"], metadata["feature_count"])
        self.demo_policy = CheckpointDemoPolicy(metadata["feature_count"])

    @staticmethod
    def _round_div(value, divisor):
        return torch.round(value / divisor)

    def forward(self, features: Tensor, missing_mask: Tensor) -> CheckpointCompiledOutput:
        results = [expert(features, missing_mask) for expert in self.experts]
        expert_scores = torch.stack([value[0] for value in results], dim=-1)
        expert_reasons = torch.stack([value[1] for value in results], dim=1)
        routed = self._round_div((expert_scores * self.routing_weights).sum(dim=-1), self.routing_denominator)
        calibrated = self._round_div(routed * self.calibration[0], self.calibration[1]) + self.calibration[2]
        score = torch.clamp(calibrated, min=self.output_bounds[0], max=self.output_bounds[1])
        reason_routed = self._round_div(
            (expert_reasons * self.routing_weights.view(1, -1, 1)).sum(dim=1), self.routing_denominator
        )
        reasons = self._round_div(reason_routed * self.calibration[0], self.calibration[1])
        gte = torch.clamp(torch.relu(score - self.policy_threshold + 1.0), max=1.0)
        lte = torch.clamp(torch.relu(self.policy_threshold - score + 1.0), max=1.0)
        decision = self.approve_gte * gte + (1.0 - self.approve_gte) * lte
        lower = torch.clamp(torch.relu(score[:, None] - self.grade_bounds[:, 0] + 1.0), max=1.0)
        upper = torch.clamp(torch.relu(self.grade_bounds[:, 1] - score[:, None] + 1.0), max=1.0)
        grade_one_hot = torch.relu(lower + upper - 1.0)
        indices = torch.arange(self.grade_bounds.shape[0], dtype=torch.float64, device=score.device)
        grade_index = (grade_one_hot * indices).sum(dim=-1)
        return CheckpointCompiledOutput(score, expert_scores, decision, grade_index, grade_one_hot, reasons)


@dataclass(frozen=True)
class ModelContributionReason:
    reason_code: str
    raw_feature_name: str
    factual_label: str
    contribution: float
    direction: str
    rank: int
    causal: bool = False
    regulatory_adverse_action_reason: bool = False


@dataclass(frozen=True)
class CreditEvidence:
    contract: str
    model_version: str
    policy_version: str
    compiled_sha256: str
    score_fixed: Tensor
    raw_margin: Tensor
    expert_scores_fixed: Tensor
    policy_decision: Tensor
    grade_index: Tensor
    grade_one_hot: Tensor
    feature_contributions_fixed: Tensor
    attribution_bias_fixed: Tensor
    local_accuracy_residual_fixed: Tensor
    ranked_feature_indices: Tensor
    ranked_positive_mask: Tensor
    policy_calculations: Tensor
    policy_flags: Tensor
    risk_band_index: Tensor
    risk_band_one_hot: Tensor
    policy_triggers: Tensor
    policy_outcome_index: Tensor
    policy_outcome_one_hot: Tensor
    policy_final_reasons: Tensor
    model_probability: Tensor
    frozen_platt_probability: Tensor
    attention_used: bool = False


@dataclass(frozen=True)
class CreditScoreResult:
    score_fixed: Tensor
    raw_margin: Tensor
    expert_scores_fixed: Tensor
    policy_decision: Tensor
    grade_index: Tensor
    grade_labels: tuple[str, ...]
    feature_contributions_fixed: Tensor
    feature_contributions: Tensor
    attribution_bias_fixed: Tensor
    attribution_bias: Tensor
    model_contribution_reasons: tuple[tuple[ModelContributionReason, ...], ...]
    policy_calculations: Tensor
    calculation_names: tuple[str, ...]
    policy_flags: Tensor
    flag_names: tuple[str, ...]
    risk_band_index: Tensor
    risk_band_labels: tuple[str, ...]
    policy_triggers: Tensor
    trigger_names: tuple[str, ...]
    policy_outcome_index: Tensor
    policy_outcome_labels: tuple[str, ...]
    policy_reason_codes: tuple[tuple[str, ...], ...]
    model_probability: Tensor
    frozen_platt_probability: Tensor
    probability_warning: str
    policy_warning: str
    evidence: CreditEvidence
    verifier_passed: Tensor


def _hash_state(state) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class CompiledCreditLLMForConditionalGeneration(AssessmentMixin, Qwen3_5ForConditionalGeneration):
    """Language backbone plus the compiled credit scorer. ``score_credit`` is tensor-only; ``assess_credit``
    (from ``AssessmentMixin``) validates a raw request, scores it, assembles the JSON assessment and lets the
    language model write one verified explanation paragraph."""

    config_class = CompiledCreditLLMConfig

    def __init__(self, config):
        metadata = config.compiled_credit
        if set(metadata) != REQUIRED_METADATA:
            raise ValueError("compiled-credit checkpoint metadata is missing or has unknown fields")
        if (metadata["checkpoint_format_version"] != CHECKPOINT_FORMAT or metadata["authority"] != AUTHORITY
                or metadata["expert_count"] != 1):
            raise ValueError("checkpoint does not declare the compiled XGBoost scorer as its authority")
        super().__init__(config)
        self.compiled_credit = CheckpointCompiledCredit(metadata)

    def _validate_credit_inputs(self, features: Tensor, missing_mask: Tensor):
        metadata = self.config.compiled_credit
        integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        if not isinstance(features, Tensor) or features.ndim != 2 or features.shape[1] != metadata["feature_count"]:
            raise TypeError(f"features must be a [batch, {metadata['feature_count']}] integer tensor")
        if features.dtype not in integer_dtypes:
            raise TypeError("features must use an integer dtype")
        if not isinstance(missing_mask, Tensor) or missing_mask.shape != features.shape or missing_mask.dtype != torch.bool:
            raise TypeError("missing_mask must be a bool tensor with the same shape as features")
        if features.device != missing_mask.device or features.shape[0] == 0:
            raise ValueError("features/missing_mask must share a device and contain at least one row")

    def _verify_authority_metadata(self):
        metadata = self.config.compiled_credit
        if (metadata.get("checkpoint_format_version") != CHECKPOINT_FORMAT or metadata.get("authority") != AUTHORITY
                or metadata.get("expert_count") != 1):
            raise RuntimeError("compiled-credit authority metadata mismatch")

    def _verify_compiled_receipt(self):
        """Hash every compiled buffer and compare with the receipt in the config. Any change (in memory or in the
        file) makes every later ``score_credit`` call fail closed."""
        state = self.compiled_credit.state_dict()
        versions = tuple((name, tensor._version) for name, tensor in state.items())
        if getattr(self, "_verified_compiled_versions", None) == versions:
            return
        actual = _hash_state(state)
        expected = self.config.compiled_credit["compiled_sha256"]
        if actual != expected:
            raise RuntimeError("compiled-credit tensor receipt mismatch")
        self._verified_compiled_versions = versions

    def _verify_evidence_tensor(self, output: CheckpointCompiledOutput, feature_contributions_fixed: Tensor,
                                local_accuracy_residual_fixed: Tensor, demo_policy: CheckpointDemoPolicyOutput) -> Tensor:
        score = output.score
        bounds_ok = (score >= self.compiled_credit.output_bounds[0]) & (score <= self.compiled_credit.output_bounds[1])
        lower = torch.clamp(torch.relu(score[:, None] - self.compiled_credit.grade_bounds[:, 0] + 1.0), max=1.0)
        upper = torch.clamp(torch.relu(self.compiled_credit.grade_bounds[:, 1] - score[:, None] + 1.0), max=1.0)
        expected_grade = torch.relu(lower + upper - 1.0)
        grade_ok = torch.all(expected_grade == output.grade_one_hot, dim=-1) & (output.grade_one_hot.sum(dim=-1) == 1)
        gte = torch.clamp(torch.relu(score - self.compiled_credit.policy_threshold + 1.0), max=1.0)
        lte = torch.clamp(torch.relu(self.compiled_credit.policy_threshold - score + 1.0), max=1.0)
        expected_decision = self.compiled_credit.approve_gte * gte + (1.0 - self.compiled_credit.approve_gte) * lte
        policy_ok = expected_decision == output.decision
        metadata = self.config.compiled_credit
        attribution_shape_ok = feature_contributions_fixed.shape[-1] == metadata["feature_count"]
        tolerance = int(metadata["attribution_local_tolerance_fixed"])
        local_accuracy_ok = torch.abs(local_accuracy_residual_fixed) <= tolerance
        if not attribution_shape_ok:
            local_accuracy_ok = torch.zeros_like(policy_ok, dtype=torch.bool)
        demo_shape_ok = (demo_policy.calculations.shape[-1] == len(metadata["policy_calculation_names"]) and
                         demo_policy.flags.shape[-1] == len(metadata["policy_flag_names"]) and
                         demo_policy.triggers.shape[-1] == len(metadata["policy_trigger_names"]))
        demo_ok = ((demo_policy.risk_band_one_hot.sum(dim=-1) == 1) &
                   (demo_policy.outcome_one_hot.sum(dim=-1) == 1) &
                   (demo_policy.risk_band_index >= 0) & (demo_policy.risk_band_index < 4) &
                   (demo_policy.outcome_index >= 0) & (demo_policy.outcome_index < 5) &
                   (demo_policy.model_probability >= 0) & (demo_policy.model_probability <= 1) &
                   (demo_policy.frozen_platt_probability >= 0) & (demo_policy.frozen_platt_probability <= 1))
        if not demo_shape_ok:
            demo_ok = torch.zeros_like(policy_ok, dtype=torch.bool)
        return bounds_ok & grade_ok & policy_ok & local_accuracy_ok & demo_ok

    def score_credit(self, features: Tensor, missing_mask: Tensor) -> CreditScoreResult:
        """Score typed integer features (``[batch, 23]`` int64) with the compiled scorer. Never runs the language model."""
        self._validate_credit_inputs(features, missing_mask)
        self._verify_authority_metadata()
        self._verify_compiled_receipt()
        output = self.compiled_credit(features, missing_mask)
        metadata = self.config.compiled_credit
        scale = float(metadata["output_scale"])
        raw_margin = output.score / scale
        feature_contributions_fixed = self.compiled_credit.attribution(features, missing_mask)
        attribution_bias_fixed = self.compiled_credit.attribution.bias_fixed
        local_accuracy_residual_fixed = (
            feature_contributions_fixed.sum(dim=-1) + attribution_bias_fixed - output.score.to(dtype=torch.int64)
        )
        ranked_feature_indices = torch.argsort(feature_contributions_fixed, dim=-1, descending=True)
        ranked_values = torch.gather(feature_contributions_fixed, 1, ranked_feature_indices)
        higher_risk = output.decision == 0
        rank_slots = torch.arange(ranked_values.shape[1], device=features.device)[None, :]
        ranked_positive_mask = (
            (ranked_values > 0) & higher_risk[:, None] & (rank_slots < int(metadata["max_model_reasons"]))
        )
        demo_policy = self.compiled_credit.demo_policy(features, missing_mask, output.score)
        grade_indices = output.grade_index.to(dtype=torch.int64).detach().cpu().tolist()
        grade_labels = tuple(metadata["grade_labels"][index] for index in grade_indices)
        ranked_cpu = ranked_feature_indices.detach().cpu().tolist()
        valid_cpu = ranked_positive_mask.detach().cpu().tolist()
        contributions_cpu = feature_contributions_fixed.detach().cpu()
        reason_metadata = metadata["reason_metadata"]
        model_reasons = []
        for row, (indices, valid) in enumerate(zip(ranked_cpu, valid_cpu)):
            records = []
            for position, (feature_index, include) in enumerate(zip(indices, valid)):
                if not include:
                    continue
                item = reason_metadata[feature_index]
                records.append(ModelContributionReason(
                    reason_code=item["reason_code"], raw_feature_name=item["raw_feature_name"],
                    factual_label=item["factual_label"],
                    contribution=float(contributions_cpu[row, feature_index].item() / scale),
                    direction="increases_model_margin", rank=position + 1,
                ))
            model_reasons.append(tuple(records))
        risk_indices = demo_policy.risk_band_index.detach().cpu().tolist()
        outcome_indices = demo_policy.outcome_index.detach().cpu().tolist()
        risk_labels = tuple(metadata["risk_band_names"][index] for index in risk_indices)
        outcome_labels = tuple(metadata["policy_outcome_names"][index] for index in outcome_indices)
        final_reason_rows = demo_policy.final_reasons.detach().cpu().tolist()
        policy_reason_codes = tuple(
            tuple(name for name, active in zip(metadata["policy_trigger_names"], row) if active)
            for row in final_reason_rows
        )
        evidence = CreditEvidence(
            contract=metadata["evidence_contract"], model_version=metadata["model_version"],
            policy_version=metadata["policy_version"], compiled_sha256=metadata["compiled_sha256"],
            score_fixed=output.score, raw_margin=raw_margin, expert_scores_fixed=output.expert_scores,
            policy_decision=output.decision, grade_index=output.grade_index,
            grade_one_hot=output.grade_one_hot,
            feature_contributions_fixed=feature_contributions_fixed,
            attribution_bias_fixed=attribution_bias_fixed,
            local_accuracy_residual_fixed=local_accuracy_residual_fixed,
            ranked_feature_indices=ranked_feature_indices, ranked_positive_mask=ranked_positive_mask,
            policy_calculations=demo_policy.calculations, policy_flags=demo_policy.flags,
            risk_band_index=demo_policy.risk_band_index, risk_band_one_hot=demo_policy.risk_band_one_hot,
            policy_triggers=demo_policy.triggers, policy_outcome_index=demo_policy.outcome_index,
            policy_outcome_one_hot=demo_policy.outcome_one_hot,
            policy_final_reasons=demo_policy.final_reasons,
            model_probability=demo_policy.model_probability,
            frozen_platt_probability=demo_policy.frozen_platt_probability,
        )
        verified = self._verify_evidence_tensor(output, feature_contributions_fixed, local_accuracy_residual_fixed, demo_policy)
        return CreditScoreResult(
            output.score, raw_margin, output.expert_scores, output.decision, output.grade_index,
            grade_labels, feature_contributions_fixed, feature_contributions_fixed.to(torch.float64) / scale,
            attribution_bias_fixed, attribution_bias_fixed.to(torch.float64) / scale,
            tuple(model_reasons), demo_policy.calculations, tuple(metadata["policy_calculation_names"]),
            demo_policy.flags, tuple(metadata["policy_flag_names"]), demo_policy.risk_band_index,
            risk_labels, demo_policy.triggers, tuple(metadata["policy_trigger_names"]),
            demo_policy.outcome_index, outcome_labels, policy_reason_codes,
            demo_policy.model_probability, demo_policy.frozen_platt_probability,
            metadata["probability_warning"], metadata["policy_warning"], evidence, verified,
        )
