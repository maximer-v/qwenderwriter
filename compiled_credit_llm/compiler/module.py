"""Generated-weight PyTorch modules: exact tree traversal on an integer lattice."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class CompiledOutput(NamedTuple):
    score: Tensor
    expert_scores: Tensor
    decision: Tensor
    grade_index: Tensor
    grade_one_hot: Tensor
    reason_contributions: Tensor


class CompiledTreeExpert(nn.Module):
    """Vectorized exact tree traversal on an integer lattice.

    All model arithmetic is represented by tensor affine, ReLU/clamp, gather,
    multiplication and reduction operations. Python does not inspect feature values.
    """

    def __init__(self, *, feature_index: Tensor, threshold: Tensor, missing_left: Tensor,
                 edge_left: Tensor, path_mask: Tensor, path_length: Tensor,
                 leaf_value: Tensor, leaf_reason: Tensor, base_score: int):
        super().__init__()
        for name, value in (
            ("feature_index", feature_index.long()), ("threshold", threshold.double()),
            ("missing_left", missing_left.double()), ("edge_left", edge_left.double()),
            ("path_mask", path_mask.double()), ("path_length", path_length.double()),
            ("leaf_value", leaf_value.double()), ("leaf_reason", leaf_reason.double()),
            ("base_score", torch.tensor(float(base_score), dtype=torch.float64)),
        ):
            self.register_buffer(name, value, persistent=True)

    def forward(self, features: Tensor, missing_mask: Tensor) -> tuple[Tensor, Tensor]:
        x = features.to(dtype=torch.float64)
        missing = missing_mask.to(dtype=torch.float64)
        selected = x[:, self.feature_index]
        selected_missing = missing[:, self.feature_index]
        # Quantized integer lattice: clamp(ReLU(t - x), 0, 1) is exactly [x < t].
        less = torch.clamp(torch.relu(self.threshold - selected), max=1.0)
        # default-left is OR(less, missing); default-right is less AND NOT(missing).
        left_gate = self.missing_left * torch.clamp(less + selected_missing, max=1.0)
        left_gate = left_gate + (1.0 - self.missing_left) * torch.relu(less - selected_missing)
        edge_gate = self.edge_left * left_gate + (1.0 - self.edge_left) * (1.0 - left_gate)
        edge_gate = edge_gate * self.path_mask
        active = torch.clamp(torch.relu(edge_gate.sum(dim=-1) - self.path_length + 1.0), max=1.0)
        score = active.matmul(self.leaf_value) + self.base_score
        reasons = active.matmul(self.leaf_reason)
        return score, reasons


class CompiledCreditModule(nn.Module):
    """Drop-in submodule for a custom Qwen class; contains all inference scoring."""

    def __init__(self, experts: list[CompiledTreeExpert], *, routing_weights: Tensor,
                 routing_denominator: int, calibration: Tensor, output_bounds: Tensor,
                 policy_threshold: int, approve_gte: bool, grade_bounds: Tensor):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        self.register_buffer("routing_weights", routing_weights.double(), persistent=True)
        self.register_buffer("routing_denominator", torch.tensor(float(routing_denominator), dtype=torch.float64), persistent=True)
        self.register_buffer("calibration", calibration.double(), persistent=True)  # multiplier, divisor, offset
        self.register_buffer("output_bounds", output_bounds.double(), persistent=True)
        self.register_buffer("policy_threshold", torch.tensor(float(policy_threshold), dtype=torch.float64), persistent=True)
        self.register_buffer("approve_gte", torch.tensor(1.0 if approve_gte else 0.0, dtype=torch.float64), persistent=True)
        self.register_buffer("grade_bounds", grade_bounds.double(), persistent=True)

    @staticmethod
    def _round_div(value: Tensor, divisor: Tensor) -> Tensor:
        return torch.round(value / divisor)

    def forward(self, features: Tensor, missing_mask: Tensor) -> CompiledOutput:
        if features.ndim != 2 or missing_mask.shape != features.shape:
            raise ValueError("features and missing_mask must have identical [batch, feature] shape")
        results = [expert(features, missing_mask) for expert in self.experts]
        expert_scores = torch.stack([x[0] for x in results], dim=-1)
        expert_reasons = torch.stack([x[1] for x in results], dim=1)
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
        grade_index = (grade_one_hot * torch.arange(
            self.grade_bounds.shape[0], dtype=torch.float64, device=score.device
        )).sum(dim=-1)
        return CompiledOutput(score, expert_scores, decision, grade_index, grade_one_hot, reasons)
