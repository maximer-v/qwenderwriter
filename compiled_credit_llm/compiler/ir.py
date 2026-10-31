"""Normalized tree ensemble IR and a contract stub for XGBoost ingestion."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
import math
import re
from typing import Any, Mapping

from .schema import ManifestError


@dataclass(frozen=True)
class SplitNode:
    feature: int
    threshold: int
    missing: str
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class LeafNode:
    value: int
    reasons: Mapping[str, int]


Node = SplitNode | LeafNode


@dataclass(frozen=True)
class TreeEnsembleIR:
    format_version: str
    output_scale: int
    base_score: int
    trees: tuple[Node, ...]

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TreeEnsembleIR":
        allowed = {"format_version", "output_scale", "base_score", "trees"}
        if not isinstance(raw, Mapping) or set(raw) != allowed:
            raise ManifestError(f"tree IR requires exactly {sorted(allowed)}")
        if raw["format_version"] != "normalized-tree-v1" or not isinstance(raw["output_scale"], int) or raw["output_scale"] <= 0:
            raise ManifestError("invalid normalized tree format version/output scale")
        if not isinstance(raw["base_score"], int) or not isinstance(raw["trees"], list) or not raw["trees"]:
            raise ManifestError("base_score must be int and trees a nonempty list")
        return cls(raw["format_version"], raw["output_scale"], raw["base_score"], tuple(_node(x) for x in raw["trees"]))


def _node(raw: Any) -> Node:
    if not isinstance(raw, Mapping) or "type" not in raw:
        raise ManifestError("each node must be an object with type")
    if raw["type"] == "leaf":
        if set(raw) != {"type", "value", "reasons"} or not isinstance(raw["value"], int) or not isinstance(raw["reasons"], Mapping):
            raise ManifestError("leaf requires integer value and reasons object")
        reasons = dict(raw["reasons"])
        if any(not isinstance(k, str) or not k or not isinstance(v, int) for k, v in reasons.items()):
            raise ManifestError("leaf reasons must map nonempty codes to integer contributions")
        return LeafNode(raw["value"], reasons)
    if raw["type"] == "split":
        required = {"type", "feature", "threshold", "missing", "left", "right"}
        if set(raw) != required or not isinstance(raw["feature"], int) or not isinstance(raw["threshold"], int):
            raise ManifestError("split fields are invalid")
        if raw["feature"] < 0 or raw["missing"] not in {"left", "right"}:
            raise ManifestError("split feature/missing direction invalid")
        return SplitNode(raw["feature"], raw["threshold"], raw["missing"], _node(raw["left"]), _node(raw["right"]))
    raise ManifestError(f"unknown node type: {raw['type']}")


@dataclass(frozen=True)
class XGBoostAdapterResult:
    normalized_ir: Mapping[str, Any]
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]


class XGBoostJSONAdapter:
    """Fail-closed adapter for the native XGBoost 3.1.2 ``gbtree`` JSON format.

    This adapter intentionally supports one source version/objective. Integer-lattice
    splits are exact: for quantized integer ``q`` with real value ``q / scale``, the
    native predicate ``value < t`` becomes ``q < ceil(t * scale)``.
    """

    source_format = "xgboost-json-3.1.2"
    target_format = "normalized-tree-v1"
    _TREE_KEYS = {
        "base_weights", "categories", "categories_nodes", "categories_segments",
        "categories_sizes", "default_left", "id", "left_children", "loss_changes",
        "parents", "right_children", "split_conditions", "split_indices", "split_type",
        "sum_hessian", "tree_param",
    }

    @classmethod
    def normalize(cls, raw: Mapping[str, Any], *, feature_scales: tuple[int, ...],
                  output_scale: int) -> Mapping[str, Any]:
        return cls.adapt(raw, feature_scales=feature_scales, output_scale=output_scale).normalized_ir

    @classmethod
    def adapt(cls, raw: Mapping[str, Any], *, feature_scales: tuple[int, ...],
              output_scale: int) -> XGBoostAdapterResult:
        if not isinstance(raw, Mapping) or set(raw) != {"learner", "version"}:
            raise ManifestError("native XGBoost JSON must contain exactly learner/version")
        if raw["version"] != [3, 1, 2]:
            raise ManifestError(f"adapter requires XGBoost JSON version [3, 1, 2], got {raw['version']!r}")
        learner = raw["learner"]
        learner_keys = {"attributes", "feature_names", "feature_types", "gradient_booster", "learner_model_param", "objective"}
        if not isinstance(learner, Mapping) or set(learner) != learner_keys or learner["attributes"] != {}:
            raise ManifestError("unsupported learner fields or attributes")
        names, types = learner["feature_names"], learner["feature_types"]
        if (not isinstance(names, list) or not names or len(names) != len(set(names)) or
                any(not isinstance(x, str) or not x for x in names)):
            raise ManifestError("feature_names must be unique nonempty strings")
        if types != ["int"] * len(names):
            raise ManifestError("this exact adapter requires integer XGBoost feature types")
        if len(feature_scales) != len(names) or any(not isinstance(x, int) or x <= 0 for x in feature_scales):
            raise ManifestError("one positive integer scale is required per feature")
        if not isinstance(output_scale, int) or output_scale <= 0 or output_scale > 10 ** 12:
            raise ManifestError("output_scale must be an integer in [1, 10^12]")

        param = learner["learner_model_param"]
        expected_param = {"base_score", "boost_from_average", "num_class", "num_feature", "num_target"}
        if not isinstance(param, Mapping) or set(param) != expected_param:
            raise ManifestError("unsupported learner_model_param structure")
        if (param["num_class"] != "0" or param["num_target"] != "1" or
                param["num_feature"] != str(len(names)) or param["boost_from_average"] != "1"):
            raise ManifestError("only one-target boosted-average binary models with matching feature count are supported")
        objective = learner["objective"]
        if objective != {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}:
            raise ManifestError("only unweighted binary:logistic objective is supported")
        match = re.fullmatch(r"\[([^,\]]+)\]", param["base_score"])
        if not match:
            raise ManifestError("base_score must be a scalar vector string")
        try:
            base_probability = float(match.group(1))
        except ValueError as exc:
            raise ManifestError("base_score is not numeric") from exc
        if not math.isfinite(base_probability) or not 0.0 < base_probability < 1.0:
            raise ManifestError("binary:logistic base_score must be in (0, 1)")
        base_margin = math.log(base_probability / (1.0 - base_probability))

        booster = learner["gradient_booster"]
        if not isinstance(booster, Mapping) or set(booster) != {"model", "name"} or booster["name"] != "gbtree":
            raise ManifestError("only native gbtree boosters are supported")
        model = booster["model"]
        model_keys = {"cats", "gbtree_model_param", "iteration_indptr", "tree_info", "trees"}
        if not isinstance(model, Mapping) or set(model) != model_keys:
            raise ManifestError("unsupported gbtree model structure")
        if model["cats"] != {"enc": [], "feature_segments": [], "sorted_idx": []}:
            raise ManifestError("categorical split metadata is unsupported")
        trees = model["trees"]
        tree_count = len(trees) if isinstance(trees, list) else -1
        if model["gbtree_model_param"] != {"num_parallel_tree": "1", "num_trees": str(tree_count)}:
            raise ManifestError("parallel trees or inconsistent tree count are unsupported")
        if model["iteration_indptr"] != list(range(tree_count + 1)) or model["tree_info"] != [0] * tree_count:
            raise ManifestError("only one binary tree per boosting iteration is supported")
        if tree_count <= 0:
            raise ManifestError("model has no trees")

        normalized_trees = []
        leaf_count = split_count = fractional_count = default_left_count = 0
        max_depth = 0
        max_leaf_error = 0.5 / output_scale
        for expected_id, tree in enumerate(trees):
            if not isinstance(tree, Mapping) or set(tree) != cls._TREE_KEYS or tree["id"] != expected_id:
                raise ManifestError("tree fields/ids do not match XGBoost 3.1.2 format")
            tree_param = tree["tree_param"]
            if (not isinstance(tree_param, Mapping) or set(tree_param) != {"num_deleted", "num_feature", "num_nodes", "size_leaf_vector"} or
                    tree_param["num_deleted"] != "0" or tree_param["num_feature"] != str(len(names)) or tree_param["size_leaf_vector"] != "1"):
                raise ManifestError("unsupported tree_param")
            node_count = int(tree_param["num_nodes"])
            arrays = ("base_weights", "default_left", "left_children", "loss_changes", "parents", "right_children",
                      "split_conditions", "split_indices", "split_type", "sum_hessian")
            if node_count <= 0 or any(not isinstance(tree[k], list) or len(tree[k]) != node_count for k in arrays):
                raise ManifestError("tree arrays do not match num_nodes")
            if any(tree[k] != [] for k in ("categories", "categories_nodes", "categories_segments", "categories_sizes")):
                raise ManifestError("categorical tree nodes are unsupported")
            if tree["parents"][0] != 2147483647 or any(x not in (0, 1) for x in tree["default_left"]):
                raise ManifestError("invalid root parent or default_left encoding")
            if any(x != 0 for x in tree["split_type"]):
                raise ManifestError("categorical split_type is unsupported")
            visited: set[int] = set()

            def convert(node_id: int, depth: int) -> dict[str, Any]:
                nonlocal leaf_count, split_count, fractional_count, default_left_count, max_depth
                if node_id < 0 or node_id >= node_count or node_id in visited:
                    raise ManifestError("tree contains invalid child index, cycle, or shared child")
                visited.add(node_id)
                max_depth = max(max_depth, depth)
                left, right = tree["left_children"][node_id], tree["right_children"][node_id]
                condition = tree["split_conditions"][node_id]
                if not isinstance(condition, (int, float)) or isinstance(condition, bool) or not math.isfinite(condition):
                    raise ManifestError("split/leaf condition must be finite numeric")
                base_weight = tree["base_weights"][node_id]
                if not isinstance(base_weight, (int, float)) or isinstance(base_weight, bool) or not math.isfinite(base_weight):
                    raise ManifestError("base_weights must be finite numeric")
                if left == -1 or right == -1:
                    if left != -1 or right != -1:
                        raise ManifestError("leaf must have both children set to -1")
                    if float(base_weight) != float(condition):
                        raise ManifestError("XGBoost 3.1.2 leaf base_weight/split_condition mismatch")
                    leaf_count += 1
                    quantized = round(float(condition) * output_scale)
                    if abs(quantized) >= 2 ** 53:
                        raise ManifestError("quantized leaf exceeds exact float64 integer range")
                    return {"type": "leaf", "value": quantized, "reasons": {}}
                feature = tree["split_indices"][node_id]
                if not isinstance(feature, int) or isinstance(feature, bool) or not 0 <= feature < len(names):
                    raise ManifestError("split references invalid feature")
                if tree["parents"][left] != node_id or tree["parents"][right] != node_id:
                    raise ManifestError("child parent pointer mismatch")
                split_count += 1
                default_left_count += int(tree["default_left"][node_id])
                scaled = Decimal(str(condition)) * feature_scales[feature]
                boundary = int(scaled.to_integral_value(rounding=ROUND_CEILING))
                fractional_count += int(scaled != scaled.to_integral_value())
                return {
                    "type": "split", "feature": feature, "threshold": boundary,
                    "missing": "left" if tree["default_left"][node_id] else "right",
                    "left": convert(left, depth + 1), "right": convert(right, depth + 1),
                }

            normalized_trees.append(convert(0, 0))
            if len(visited) != node_count:
                raise ManifestError("tree has unreachable nodes")
        quantized_base = round(base_margin * output_scale)
        if abs(quantized_base) >= 2 ** 53:
            raise ManifestError("quantized base margin exceeds exact float64 integer range")
        ir = {
            "format_version": "normalized-tree-v1", "output_scale": output_scale,
            "base_score": quantized_base, "trees": normalized_trees,
        }
        metadata = {
            "source_format": cls.source_format, "source_version": [3, 1, 2],
            "objective": "binary:logistic", "base_probability": base_probability,
            "base_margin": base_margin, "tree_count": tree_count, "leaf_count": leaf_count,
            "split_count": split_count, "max_depth": max_depth,
            "fractional_scaled_threshold_count": fractional_count,
            "default_left_split_count": default_left_count,
            "threshold_normalization": "q < ceil(native_threshold * feature_scale)",
            "leaf_fixed_point_scale": output_scale,
            "per_leaf_absolute_quantization_bound": max_leaf_error,
            "raw_margin_absolute_quantization_bound": (tree_count + 1) * max_leaf_error,
        }
        return XGBoostAdapterResult(ir, tuple(names), metadata)
