"""Model-to-generated-weights compiler for normalized tree ensembles."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .ir import LeafNode, Node, SplitNode, TreeEnsembleIR
from .module import CompiledCreditModule, CompiledTreeExpert
from .receipts import canonical_json_bytes, hash_state_dict, sha256_json
from .schema import CompiledManifest, ManifestError


def _leaf_paths(node: Node, path=()):
    if isinstance(node, LeafNode):
        yield path, node
        return
    yield from _leaf_paths(node.left, path + ((node.feature, node.threshold, node.missing == "left", True),))
    yield from _leaf_paths(node.right, path + ((node.feature, node.threshold, node.missing == "left", False),))


def _compile_expert(ir: TreeEnsembleIR, manifest: CompiledManifest) -> CompiledTreeExpert:
    precision = manifest.precision
    if len(ir.trees) > precision.max_trees:
        raise ManifestError("tree count exceeds resource limit")
    rows: list[tuple[tuple, LeafNode]] = []
    conservative = abs(ir.base_score)
    for tree in ir.trees:
        leaves = list(_leaf_paths(tree))
        if not leaves:
            raise ManifestError("tree has no leaves")
        conservative += max(abs(leaf.value) for _, leaf in leaves)
        rows.extend(leaves)
    if len(rows) > precision.max_leaves:
        raise ManifestError("leaf count exceeds resource limit")
    max_depth = max((len(path) for path, _ in rows), default=0)
    if max_depth > precision.max_depth:
        raise ManifestError("tree depth exceeds resource limit")
    if conservative > precision.max_abs_accumulator:
        raise ManifestError("conservative accumulator bound exceeds resource limit")
    width = max(1, max_depth)
    feature_index, threshold, missing_left, edge_left, path_mask, path_length = [], [], [], [], [], []
    leaf_value, leaf_reason = [], []
    reason_index = {reason.code: i for i, reason in enumerate(manifest.reasons)}
    for path, leaf in rows:
        fi = [0] * width
        th = [0] * width
        ml = [0] * width
        el = [0] * width
        pm = [0] * width
        for i, (feature, split, default_left, take_left) in enumerate(path):
            if feature >= len(manifest.features):
                raise ManifestError(f"tree references unknown feature index {feature}")
            fi[i], th[i], ml[i], el[i], pm[i] = feature, split, int(default_left), int(take_left), 1
        unknown = set(leaf.reasons) - set(reason_index)
        if unknown:
            raise ManifestError(f"leaf references unknown reason codes: {sorted(unknown)}")
        rv = [0] * len(reason_index)
        for code, value in leaf.reasons.items():
            rv[reason_index[code]] = value
        feature_index.append(fi); threshold.append(th); missing_left.append(ml)
        edge_left.append(el); path_mask.append(pm); path_length.append(len(path))
        leaf_value.append(leaf.value); leaf_reason.append(rv)
    return CompiledTreeExpert(
        feature_index=torch.tensor(feature_index), threshold=torch.tensor(threshold),
        missing_left=torch.tensor(missing_left), edge_left=torch.tensor(edge_left),
        path_mask=torch.tensor(path_mask), path_length=torch.tensor(path_length),
        leaf_value=torch.tensor(leaf_value),
        leaf_reason=torch.tensor(leaf_reason if manifest.reasons else [[] for _ in rows]),
        base_score=ir.base_score,
    )


def compile_manifest(manifest: CompiledManifest, irs: Sequence[TreeEnsembleIR | Mapping[str, Any]]) -> CompiledCreditModule:
    """Compile validated IR into an inference-only tensor module."""
    if len(irs) != len(manifest.experts):
        raise ManifestError("one IR is required per expert")
    if manifest.routing.kind != "static_weighted":
        raise ManifestError("internal_attention is reserved by the interface but not implemented by this backend")
    compiled = []
    for spec, raw in zip(manifest.experts, irs):
        if spec.kind != "tree_ensemble":
            raise ManifestError(f"first backend compiles tree_ensemble only, got {spec.kind}")
        ir = raw if isinstance(raw, TreeEnsembleIR) else TreeEnsembleIR.from_dict(raw)
        if ir.output_scale != spec.output_scale or ir.output_scale != manifest.calibration.input_scale:
            raise ManifestError("expert/IR/calibration input scales must match")
        compiled.append(_compile_expert(ir, manifest))
    cal = manifest.calibration
    bands = sorted(manifest.policy.grade_bands, key=lambda x: x.minimum)
    module = CompiledCreditModule(
        compiled, routing_weights=torch.tensor(manifest.routing.weight_numerators),
        routing_denominator=manifest.routing.denominator,
        calibration=torch.tensor([cal.multiplier, cal.divisor, cal.offset]),
        output_bounds=torch.tensor([cal.minimum, cal.maximum]),
        policy_threshold=manifest.policy.decision_threshold,
        approve_gte=manifest.policy.approve_when == "gte",
        grade_bounds=torch.tensor([[x.minimum, x.maximum] for x in bands]),
    )
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def save_compiled(directory: str | Path, manifest: CompiledManifest,
                  irs: Sequence[TreeEnsembleIR | Mapping[str, Any]], module: CompiledCreditModule) -> dict[str, Any]:
    """Save architecture metadata, generated tensors, and deterministic logical receipts."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    manifest_raw = manifest.to_dict()
    ir_raw = [_ir_to_dict(x) if isinstance(x, TreeEnsembleIR) else dict(x) for x in irs]
    (target / "manifest.json").write_bytes(canonical_json_bytes(manifest_raw) + b"\n")
    (target / "normalized_ir.json").write_bytes(canonical_json_bytes(ir_raw) + b"\n")
    torch.save(module.state_dict(), target / "compiled_weights.pt")
    receipt = {
        "algorithm": "sha256",
        "manifest_sha256": sha256_json(manifest_raw),
        "ir_sha256": [sha256_json(x) for x in ir_raw],
        "compiled_sha256": hash_state_dict(module.state_dict()),
        "model_version": manifest.model_version,
        "policy_version": manifest.policy_version,
    }
    (target / "receipt.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
    return receipt


def load_compiled(directory: str | Path) -> tuple[CompiledManifest, CompiledCreditModule, dict[str, Any]]:
    """Fail closed on metadata or generated-weight receipt mismatch."""
    source = Path(directory)
    manifest_raw = json.loads((source / "manifest.json").read_text("ascii"))
    ir_raw = json.loads((source / "normalized_ir.json").read_text("ascii"))
    receipt = json.loads((source / "receipt.json").read_text("ascii"))
    if receipt.get("algorithm") != "sha256" or receipt.get("manifest_sha256") != sha256_json(manifest_raw):
        raise ManifestError("manifest receipt mismatch")
    if receipt.get("ir_sha256") != [sha256_json(x) for x in ir_raw]:
        raise ManifestError("IR receipt mismatch")
    manifest = CompiledManifest.from_dict(manifest_raw)
    if receipt.get("model_version") != manifest.model_version or receipt.get("policy_version") != manifest.policy_version:
        raise ManifestError("version receipt mismatch")
    module = compile_manifest(manifest, ir_raw)
    state = torch.load(source / "compiled_weights.pt", map_location="cpu", weights_only=True)
    if hash_state_dict(state) != receipt.get("compiled_sha256"):
        raise ManifestError("compiled weight receipt mismatch")
    module.load_state_dict(state, strict=True)
    module.eval()
    return manifest, module, receipt


def _ir_to_dict(ir: TreeEnsembleIR) -> dict[str, Any]:
    def node(value: Node) -> dict[str, Any]:
        if isinstance(value, LeafNode):
            return {"type": "leaf", "value": value.value, "reasons": dict(value.reasons)}
        return {"type": "split", "feature": value.feature, "threshold": value.threshold,
                "missing": "left" if value.missing == "left" else "right",
                "left": node(value.left), "right": node(value.right)}
    return {"format_version": ir.format_version, "output_scale": ir.output_scale,
            "base_score": ir.base_score, "trees": [node(x) for x in ir.trees]}
