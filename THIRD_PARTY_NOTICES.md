# Third-party notices

| Component | Role | License |
|---|---|---|
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | language backbone of the checkpoint; its tensors are copied unchanged into the base checkpoint and adapted by a merged LoRA | Apache License 2.0 |
| [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | offline teacher that wrote the explanation targets (not part of the checkpoint) | Apache License 2.0 |
| [UCI dataset 350, Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (I-Cheng Yeh) | training data for the scorer, development rows for the explanation writer, example inputs | CC BY 4.0 |
| XGBoost, PyTorch, Transformers, PEFT, scikit-learn, pandas, NumPy, SciPy, safetensors | build and runtime libraries | Apache 2.0 / BSD |

The dataset is described in Yeh and Lien, *Expert Systems with Applications* 36(2), 2009, doi:10.1016/j.eswa.2007.12.020.
`data/default_of_credit_card_clients.csv` is a lossless CSV conversion of the official XLS; the two UCI header rows are kept.

This repository and the released checkpoint are licensed under CC BY-NC 4.0 (see `LICENSE`). The Apache-2.0 and CC BY 4.0
terms of the components above continue to apply to those components.
