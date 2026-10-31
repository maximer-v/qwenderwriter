"""Configuration for the Compiled Credit LLM checkpoint: the official Qwen3.5 configuration plus immutable
compiled-credit metadata (feature contract, buffer specifications, receipts, policy and reason labels)."""
from __future__ import annotations

from transformers import Qwen3_5Config


class CompiledCreditLLMConfig(Qwen3_5Config):
    model_type = "compiled_credit_llm"

    def __init__(self, compiled_credit=None, **kwargs):
        super().__init__(**kwargs)
        self.compiled_credit = compiled_credit or {}
