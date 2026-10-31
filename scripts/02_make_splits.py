#!/usr/bin/env python3
"""Create the frozen stratified splits (train 18,000 / validation 4,500 / calibration 3,000 / test 4,500)."""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401
from compiled_credit_llm import paths
from compiled_credit_llm.data import uci

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--csv", default=str(paths.DATA_CSV))
parser.add_argument("--manifest", default=str(paths.SPLIT_MANIFEST))
parser.add_argument("--summary", default=str(paths.SPLIT_SUMMARY))
args = parser.parse_args()
uci.make_splits(Path(args.csv), Path(args.manifest), Path(args.summary))
print(json.dumps(json.loads(Path(args.summary).read_text())["class_counts"], indent=2))
