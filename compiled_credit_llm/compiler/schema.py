"""Strict, dependency-free manifest types for compiled credit checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence


class ManifestError(ValueError):
    """Raised when compilation inputs fail closed validation."""


def _strict(cls: type, raw: Mapping[str, Any]):
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{cls.__name__} must be an object")
    expected = {f.name for f in fields(cls)}
    unknown = set(raw) - expected
    missing = {f.name for f in fields(cls) if f.default is f.default_factory} - set(raw)
    if unknown or missing:
        raise ManifestError(f"{cls.__name__}: unknown={sorted(unknown)}, missing={sorted(missing)}")
    try:
        return cls(**raw)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"invalid {cls.__name__}: {exc}") from exc


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    index: int
    dtype: str
    scale: int
    minimum: int
    maximum: int
    missing_allowed: bool

    def __post_init__(self):
        _nonempty(self.name, "feature.name")
        if self.dtype not in {"fixed_int", "bool"} or self.index < 0 or self.scale <= 0:
            raise ManifestError("invalid feature dtype/index/scale")
        if self.minimum > self.maximum:
            raise ManifestError("feature minimum exceeds maximum")


@dataclass(frozen=True)
class ExpertSpec:
    id: str
    kind: str
    ir_path: str
    output_scale: int
    reason_namespace: str

    def __post_init__(self):
        _nonempty(self.id, "expert.id")
        if self.kind not in {"tree_ensemble", "scorecard", "logistic"}:
            raise ManifestError(f"unsupported expert kind in schema: {self.kind}")
        if self.output_scale <= 0:
            raise ManifestError("expert output_scale must be positive")


@dataclass(frozen=True)
class RoutingSpec:
    kind: str
    weight_numerators: tuple[int, ...]
    denominator: int
    attention_layers: int

    def __post_init__(self):
        if self.kind not in {"static_weighted", "internal_attention"}:
            raise ManifestError("routing.kind is invalid")
        if self.denominator <= 0 or not self.weight_numerators:
            raise ManifestError("routing weights/denominator are invalid")
        if any(x < 0 for x in self.weight_numerators) or sum(self.weight_numerators) != self.denominator:
            raise ManifestError("routing weights must be nonnegative and sum to denominator")
        if self.attention_layers < 0 or (self.kind == "static_weighted" and self.attention_layers != 0):
            raise ManifestError("invalid attention layer declaration")


@dataclass(frozen=True)
class StackCalibrationSpec:
    input_scale: int
    output_scale: int
    multiplier: int
    divisor: int
    offset: int
    minimum: int
    maximum: int

    def __post_init__(self):
        if min(self.input_scale, self.output_scale, self.divisor) <= 0:
            raise ManifestError("calibration scales/divisor must be positive")
        if self.minimum > self.maximum:
            raise ManifestError("calibration bounds invalid")


@dataclass(frozen=True)
class GradeBand:
    grade: str
    minimum: int
    maximum: int

    def __post_init__(self):
        _nonempty(self.grade, "grade")
        if self.minimum > self.maximum:
            raise ManifestError("grade band bounds invalid")


@dataclass(frozen=True)
class PolicySpec:
    decision_threshold: int
    approve_when: str
    grade_bands: tuple[GradeBand, ...]

    def __post_init__(self):
        if self.approve_when not in {"gte", "lte"}:
            raise ManifestError("policy.approve_when must be gte or lte")
        if not self.grade_bands:
            raise ManifestError("at least one grade band is required")
        ordered = sorted(self.grade_bands, key=lambda b: b.minimum)
        for left, right in zip(ordered, ordered[1:]):
            if left.maximum + 1 != right.minimum:
                raise ManifestError("grade bands must be contiguous and non-overlapping")


@dataclass(frozen=True)
class ReasonSpec:
    code: str
    label: str
    adverse_when: str

    def __post_init__(self):
        _nonempty(self.code, "reason.code")
        _nonempty(self.label, "reason.label")
        if self.adverse_when not in {"positive", "negative", "absolute"}:
            raise ManifestError("invalid reason direction")


@dataclass(frozen=True)
class PrecisionSpec:
    feature_bits: int
    accumulator_bits: int
    output_bits: int
    max_abs_accumulator: int
    max_trees: int
    max_depth: int
    max_leaves: int

    def __post_init__(self):
        if not (1 <= self.feature_bits <= 53 and 1 <= self.accumulator_bits <= 53 and 1 <= self.output_bits <= 53):
            raise ManifestError("bit widths must be in [1, 53] for exact float64 integer arithmetic")
        if min(self.max_abs_accumulator, self.max_trees, self.max_depth, self.max_leaves) <= 0:
            raise ManifestError("resource limits must be positive")
        if self.max_abs_accumulator >= 2 ** (self.accumulator_bits - 1):
            raise ManifestError("accumulator bound does not fit declared bits")


@dataclass(frozen=True)
class ReceiptSpec:
    algorithm: str
    manifest_sha256: str
    ir_sha256: tuple[str, ...]
    compiled_sha256: str

    def __post_init__(self):
        if self.algorithm != "sha256":
            raise ManifestError("only sha256 receipts are supported")
        for digest in (self.manifest_sha256, self.compiled_sha256, *self.ir_sha256):
            if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ManifestError("receipt digest must be lowercase sha256 hex or empty before compilation")


@dataclass(frozen=True)
class CompiledManifest:
    schema_version: str
    model_version: str
    policy_version: str
    features: tuple[FeatureSpec, ...]
    experts: tuple[ExpertSpec, ...]
    routing: RoutingSpec
    calibration: StackCalibrationSpec
    policy: PolicySpec
    reasons: tuple[ReasonSpec, ...]
    precision: PrecisionSpec
    receipt: ReceiptSpec

    def __post_init__(self):
        for value, name in ((self.schema_version, "schema_version"), (self.model_version, "model_version"), (self.policy_version, "policy_version")):
            _nonempty(value, name)
        if self.schema_version != "1.0":
            raise ManifestError("unsupported schema_version")
        if not self.features or not self.experts:
            raise ManifestError("features and experts cannot be empty")
        if len(self.experts) != len(self.routing.weight_numerators):
            raise ManifestError("one routing weight is required per expert")
        if len(self.receipt.ir_sha256) != len(self.experts):
            raise ManifestError("receipt requires one IR digest slot per expert")
        bands = sorted(self.policy.grade_bands, key=lambda b: b.minimum)
        if bands[0].minimum != self.calibration.minimum or bands[-1].maximum != self.calibration.maximum:
            raise ManifestError("grade bands must exactly cover calibrated output bounds")
        if not self.calibration.minimum <= self.policy.decision_threshold <= self.calibration.maximum:
            raise ManifestError("policy threshold must be inside output bounds")
        signed_output_limit = 2 ** (self.precision.output_bits - 1)
        if self.calibration.minimum < -signed_output_limit or self.calibration.maximum >= signed_output_limit:
            raise ManifestError("calibrated output bounds do not fit output_bits")
        signed_feature_limit = 2 ** (self.precision.feature_bits - 1)
        if any(f.minimum < -signed_feature_limit or f.maximum >= signed_feature_limit for f in self.features):
            raise ManifestError("feature bounds do not fit feature_bits")
        if [f.index for f in self.features] != list(range(len(self.features))):
            raise ManifestError("feature indices must be unique, ordered and contiguous")
        for group, name in ((self.features, "feature names"), (self.experts, "expert ids"), (self.reasons, "reason codes")):
            values = [getattr(x, "name", getattr(x, "id", getattr(x, "code", None))) for x in group]
            if len(values) != len(set(values)):
                raise ManifestError(f"duplicate {name}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CompiledManifest":
        if not isinstance(raw, Mapping):
            raise ManifestError("CompiledManifest must be an object")
        expected = {f.name for f in fields(cls)}
        if set(raw) != expected:
            raise ManifestError(f"CompiledManifest: unknown={sorted(set(raw) - expected)}, missing={sorted(expected - set(raw))}")
        try:
            features = tuple(_strict(FeatureSpec, x) for x in raw["features"])
            experts = tuple(_strict(ExpertSpec, x) for x in raw["experts"])
            routing_raw = dict(raw["routing"])
            routing_raw["weight_numerators"] = tuple(routing_raw["weight_numerators"])
            routing = _strict(RoutingSpec, routing_raw)
            calibration = _strict(StackCalibrationSpec, raw["calibration"])
            policy_raw = dict(raw["policy"])
            policy_raw["grade_bands"] = tuple(_strict(GradeBand, x) for x in policy_raw["grade_bands"])
            policy = _strict(PolicySpec, policy_raw)
            reasons = tuple(_strict(ReasonSpec, x) for x in raw["reasons"])
            precision = _strict(PrecisionSpec, raw["precision"])
            receipt_raw = dict(raw["receipt"])
            receipt_raw["ir_sha256"] = tuple(receipt_raw["ir_sha256"])
            receipt = _strict(ReceiptSpec, receipt_raw)
            return cls(raw["schema_version"], raw["model_version"], raw["policy_version"], features, experts, routing, calibration, policy, reasons, precision, receipt)
        except (KeyError, TypeError) as exc:
            raise ManifestError(f"malformed nested manifest: {exc}") from exc
