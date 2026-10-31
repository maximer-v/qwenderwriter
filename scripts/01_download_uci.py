#!/usr/bin/env python3
"""Fetch the official UCI XLS, verify its hash, convert it to the working CSV and record provenance.

The repository already ships data/default_of_credit_card_clients.csv, so this step is optional; run it to
re-derive the file from the source."""
import argparse
import json

from _common import ROOT  # noqa: F401
from compiled_credit_llm import paths
from compiled_credit_llm.data import uci

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--xls", default=str(paths.DATA / "raw" / "default_of_credit_card_clients.xls"))
parser.add_argument("--csv", default=str(paths.DATA_CSV))
args = parser.parse_args()
xls = uci.download_xls(uci.Path(args.xls))
csv_path = uci.convert_xls(xls, uci.Path(args.csv))
uci.write_source_metadata(paths.DATA / "uci_source_metadata.json", xls, csv_path)
print(json.dumps(uci.validate_csv(csv_path), indent=2))
