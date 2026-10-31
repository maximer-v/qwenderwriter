"""UCI dataset 350, *Default of Credit Card Clients*: download, conversion and frozen splits.

The raw XLS is fetched from the official UCI host, hash-checked and converted to the working CSV with the
two-row UCI header (``X1..X23, Y`` on the first row, field names on the second). The repository ships the
converted CSV and the split manifest so every later step runs offline; this module reproduces them.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
REPOSITORY_URL = "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients"
EXPECTED_XLS_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
EXPECTED_XLS_BYTES = 5_539_328
EXPECTED_CSV_SHA256 = "bf15b02eb5a2c34f01f6d071083ecb7b8f4e803a4280a7ef497903d84c791c48"
TARGET = "default payment next month"
FEATURE_NAMES = (
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
)
COLUMNS = ["ID", *FEATURE_NAMES, TARGET]
SPLIT_SEED = 20261030
SPLIT_SIZES = {"train": 18_000, "validation": 4_500, "calibration": 3_000, "test": 4_500}
DEVELOPMENT_SPLITS = ("train", "validation", "calibration")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_xls(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "compiled-credit-llm/1.0"})
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            os.replace(tmp_path, destination)
        finally:
            tmp_path.unlink(missing_ok=True)
    actual = sha256(destination)
    if actual != EXPECTED_XLS_SHA256 or destination.stat().st_size != EXPECTED_XLS_BYTES:
        raise RuntimeError(f"raw XLS mismatch: sha256={actual}, bytes={destination.stat().st_size}")
    return destination


def convert_xls(xls: Path, csv_path: Path) -> Path:
    """Write the working CSV: UCI's two header rows, then 30,000 integer rows."""
    frame = pd.read_excel(xls, header=None, engine="xlrd")
    variables = [str(v) if not (isinstance(v, float) and np.isnan(v)) else "" for v in frame.iloc[0].tolist()]
    names = [str(v) for v in frame.iloc[1].tolist()]
    if variables[1:] != [f"X{i}" for i in range(1, 24)] + ["Y"] or names != COLUMNS:
        raise RuntimeError(f"unexpected UCI header rows: {variables[:3]}... / {names[:3]}...")
    body = frame.iloc[2:].astype("int64")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(variables)
        writer.writerow(names)
        for row in body.itertuples(index=False):
            writer.writerow([int(v) for v in row])
    validate_csv(csv_path)
    return csv_path


def validate_csv(csv_path: Path) -> dict:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.reader(stream)
        variables = next(rows)
        names = next(rows)
        body = list(rows)
    if variables[1:] != [f"X{i}" for i in range(1, 24)] + ["Y"] or names != COLUMNS:
        raise RuntimeError("unexpected CSV header rows")
    if len(body) != 30_000 or any(len(r) != 25 for r in body):
        raise RuntimeError("working CSV must contain exactly 30,000 rows and 25 columns")
    positives = sum(int(r[-1]) for r in body)
    if positives != 6_636:
        raise RuntimeError(f"unexpected positive target count: {positives}")
    return {"rows": len(body), "positives": positives, "sha256": sha256(csv_path), "matches_reference": sha256(csv_path) == EXPECTED_CSV_SHA256}


def load_table(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path, skiprows=1)
    if frame.shape != (30_000, 25) or list(frame.columns) != COLUMNS:
        raise RuntimeError(f"unexpected data shape/columns: {frame.shape}")
    return frame


def opaque_key(index: int) -> str:
    return "client_" + hashlib.sha256(f"uci350-row-{index}".encode()).hexdigest()[:20]


def make_splits(csv_path: Path, manifest_path: Path, summary_path: Path | None = None) -> pd.DataFrame:
    """Deterministic stratified client-level splits with fixed sizes (hierarchical, seed-locked)."""
    frame = load_table(csv_path)
    y = frame[TARGET].to_numpy()
    if set(np.unique(y)) != {0, 1} or int(y.sum()) != 6_636:
        raise RuntimeError("target identity/prevalence check failed")
    index = np.arange(len(y))
    development, test = train_test_split(index, test_size=SPLIT_SIZES["test"], random_state=SPLIT_SEED, stratify=y)
    trainval, calibration = train_test_split(development, test_size=SPLIT_SIZES["calibration"], random_state=SPLIT_SEED + 1, stratify=y[development])
    train, validation = train_test_split(trainval, test_size=SPLIT_SIZES["validation"], random_state=SPLIT_SEED + 2, stratify=y[trainval])
    split = np.empty(len(y), dtype=object)
    for name, rows in (("train", train), ("validation", validation), ("calibration", calibration), ("test", test)):
        split[rows] = name
    manifest = pd.DataFrame({"opaque_row_id": [opaque_key(i) for i in range(len(y))], "split": split, "target": frame[TARGET].astype("int8")})
    sizes = manifest.groupby("split", sort=True).size().to_dict()
    if sizes != SPLIT_SIZES:
        raise RuntimeError(f"split size mismatch: {sizes}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    if summary_path is not None:
        counts = {}
        for name in SPLIT_SIZES:
            s = manifest[manifest["split"] == name]["target"]
            counts[name] = {"rows": int(len(s)), "defaults": int(s.sum()), "non_defaults": int((1 - s).sum()), "prevalence": float(s.mean())}
        summary_path.write_text(json.dumps({
            "seed": SPLIT_SEED, "method": "hierarchical stratified train_test_split with fixed exact sizes",
            "split_sizes": SPLIT_SIZES, "class_counts": counts, "overall_prevalence": float(manifest["target"].mean()),
            "feature_policy": "ID excluded; all 23 variables retained; no target-derived features",
            "evidence_scope": "random-split development evidence; the dataset has no application ordering for a temporal holdout",
        }, indent=2) + "\n")
    return manifest


def write_source_metadata(path: Path, xls: Path, csv_path: Path) -> None:
    path.write_text(json.dumps({
        "dataset": "Default of Credit Card Clients", "uci_id": 350, "doi": "10.24432/C55S3H",
        "repository_url": REPOSITORY_URL, "source_url": SOURCE_URL,
        "retrieved_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "raw_bytes": xls.stat().st_size, "raw_sha256": sha256(xls), "working_csv_sha256": sha256(csv_path),
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "license_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "creator": "I-Cheng Yeh", "citation_doi": "10.1016/j.eswa.2007.12.020",
    }, indent=2) + "\n")
