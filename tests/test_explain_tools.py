"""Training-data plumbing: chat dataset masking, collator, and the gate's bookkeeping with a fake tokenizer."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from compiled_credit_llm.explain.train import ChatDataset, RightPadCollator, input_ids


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        tokens = []
        for m in messages:
            tokens += [1] + [ord(c) % 50 + 2 for c in m["content"]] + [1]
        if add_generation_prompt:
            tokens += [1]
        return tokens


class ExplainToolTests(unittest.TestCase):
    def test_input_ids_normalises_shapes(self):
        self.assertEqual(input_ids([[1, 2, 3]]), [1, 2, 3])
        self.assertEqual(input_ids(torch.tensor([[4, 5]])), [4, 5])
        self.assertEqual(input_ids({"input_ids": [7, 8]}), [7, 8])
        with self.assertRaises(RuntimeError):
            input_ids([[1], [2]])

    def test_dataset_masks_prompt_and_collator_pads_right(self):
        rows = [{"example_id": f"r{i}", "source": {"source_split": "train"},
                 "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "facts " * (i + 1)}, {"role": "assistant", "content": "answer"}]}
                for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            dataset = ChatDataset(path, FakeTokenizer(), 512)
            item = dataset[0]
            prompt_len = len(input_ids(FakeTokenizer().apply_chat_template(rows[0]["messages"][:-1], add_generation_prompt=True)))
            self.assertEqual(item["labels"][:prompt_len], [-100] * prompt_len)
            self.assertEqual(item["labels"][prompt_len:], item["input_ids"][prompt_len:])
            batch = RightPadCollator(0)([dataset[0], dataset[2]])
            self.assertEqual(batch["input_ids"].shape, batch["labels"].shape)
            self.assertEqual(int(batch["attention_mask"][0].sum()), len(dataset[0]["input_ids"]))
            self.assertEqual(int(batch["labels"][0, -1]), -100)
            with self.assertRaises(RuntimeError):
                ChatDataset(path, FakeTokenizer(), 8)[0]

    def test_dataset_rejects_test_rows(self):
        rows = [{"example_id": "r", "source": {"source_split": "test"}, "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train.jsonl"; path.write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaises(RuntimeError):
                ChatDataset(path, FakeTokenizer(), 512)
