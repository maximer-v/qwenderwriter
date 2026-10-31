#!/bin/bash
# Execute the notebook in place (all stages). Expects the repository's Python environment to be active.
cd "$(dirname "$0")/.."
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/end_to_end.ipynb
