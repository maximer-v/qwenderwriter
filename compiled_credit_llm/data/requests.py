"""Raw 23-field requests from the UCI table, the development-row variants used to teach and test the explanation
writer, and the seven malformed request classes the router must refuse."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .uci import DEVELOPMENT_SPLITS, FEATURE_NAMES

VARIANT_SEED = "compiled-credit-llm/variants/v1"
VARIANTS = ("N", "M", "O", "C", "NULL", "RU_N", "RU_O", "DZ_N", "DX_N")
VARIANT_DESCRIPTIONS = {
    "N": "natural development row", "M": "one field set to an explicit null", "O": "age above the development maximum",
    "C": "contract-invalid negative payment", "NULL": "one field set to an explicit null (rare-trigger augmentation)",
    "RU_N": "natural row with an unknown demographic category", "RU_O": "unknown demographic category and age above the development maximum",
    "DZ_N": "natural row with zero payments and high utilization", "DX_N": "natural row with extreme utilization and delinquency",
}


def stable_hash(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode()).hexdigest()


class RawTable:
    """The processed UCI table plus split manifest, with deterministic request reconstruction."""

    def __init__(self, csv_path: Path, manifest_path: Path):
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream)
            next(reader)
            header = next(reader)
            columns = {name: header.index(name) for name in FEATURE_NAMES}
            self.rows = [[int(row[columns[name]]) for name in FEATURE_NAMES] for row in reader]
        with Path(manifest_path).open(newline="") as stream:
            manifest = list(csv.DictReader(stream))
        if len(manifest) != len(self.rows):
            raise RuntimeError("split manifest and processed table are misaligned")
        self.split = [row["split"] for row in manifest]
        self.opaque = [row["opaque_row_id"] for row in manifest]

    def natural(self, index: int) -> dict:
        return {"action": "ASSESS", "credit": dict(zip(FEATURE_NAMES, self.rows[index]))}

    def request(self, source: dict, age_out_of_development: int) -> dict:
        """Rebuild the request of a dataset row from its ``source`` record (row_index, split, variant)."""
        index = int(source["row_index"])
        if self.split[index] not in DEVELOPMENT_SPLITS or self.split[index] != source["source_split"]:
            raise RuntimeError("raw request reconstruction refuses non-development rows")
        if self.opaque[index] != source["opaque_row_id"]:
            raise RuntimeError("opaque row identity mismatch")
        variant = source["variant"]
        if variant not in VARIANTS:
            raise RuntimeError(f"unknown variant {variant}")
        request = self.natural(index)
        credit = request["credit"]
        if variant in ("O", "RU_O"):
            credit["AGE"] = age_out_of_development
        elif variant in ("M", "NULL"):
            credit[FEATURE_NAMES[int(stable_hash(VARIANT_SEED, "null", self.opaque[index])[:8], 16) % 23]] = None
        elif variant == "C":
            credit["PAY_AMT1"] = -1
        return request


def malformed_mutations(valid_request: dict, evidence: str) -> list[tuple[str, dict]]:
    """The seven malformed request classes, derived from one valid request and its evidence line."""
    credit = dict(valid_request["credit"])
    missing = {"action": "ASSESS", "credit": {k: v for k, v in credit.items() if k != "PAY_0"}}
    duplicate = {"action": "ASSESS", "credit": list(credit.items()) + [("PAY_0", (credit["PAY_0"] or 0) + 1)]}
    out_of_range = {"action": "ASSESS", "credit": {**credit, "LIMIT_BAL": 2 ** 63}}
    malformed = {"action": "ASSESS", "credit": {**credit, "BILL_AMT1": "1000"}}
    parts = evidence.split("|")
    swapped = "|".join(part if not part.startswith("o=") else ("o=AS" if part != "o=AS" else "o=MR") for part in parts)
    mismatch = {"action": "ASSESS", "credit": credit, "bound_evidence": swapped}
    unsupported = {"action": "CHANGE_DECISION", "credit": credit}
    override = {"action": "ASSESS", "credit": credit, "override": "IGNORE_CHECKPOINT"}
    return [
        ("MISSING_FIELD", missing), ("DUPLICATE_CONFLICT", duplicate), ("OUT_OF_RANGE", out_of_range),
        ("MALFORMED_TYPE", malformed), ("EVIDENCE_MISMATCH", mismatch), ("UNSUPPORTED_REQUEST", unsupported),
        ("OVERRIDE_ATTEMPT", override),
    ]
