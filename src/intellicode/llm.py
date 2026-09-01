"""HuggingFace text-generation backend implementing the agent's LLM protocol.

Isolated here so the core :mod:`intellicode.agent` depends only on the small
:class:`~intellicode.agent.LLMBackend` protocol — not on torch/transformers.
This keeps import time low and the agent unit-testable without a model.
"""

from __future__ import annotations

import logging

from intellicode.config import Settings

logger = logging.getLogger(__name__)


class QwenLLM:
    """Causal-LM backend (default: Qwen2.5-3B-Instruct) via transformers.

    The model is loaded eagerly in ``__init__`` so failures surface at startup.
    Use :meth:`from_pretrained` for the common construction path.

    Args:
        model: A loaded ``AutoModelForCausalLM``.
        tokenizer: The matching tokenizer.
        settings: Generation configuration.
    """

    def __init__(self, model, tokenizer, settings: Settings | None = None) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings or Settings()

    @classmethod
    def from_pretrained(cls, settings: Settings | None = None) -> QwenLLM:
        """Load model + tokenizer from the configured ``llm_model_id``.

        Args:
            settings: Configuration (uses defaults when ``None``).

        Returns:
            A ready :class:`QwenLLM`.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        settings = settings or Settings()
        logger.info("Loading LLM '%s' …", settings.llm_model_id)
        tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            settings.llm_model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("LLM loaded.")
        return cls(model, tokenizer, settings)

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        """Generate a completion for *prompt* using the chat template.

        Args:
            prompt: The user prompt.
            max_new_tokens: Token budget (defaults to ``settings.llm_max_new_tokens``).

        Returns:
            The generated text with the prompt stripped.
        """
        import torch

        max_new_tokens = max_new_tokens or self._settings.llm_max_new_tokens
        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self._settings.llm_temperature,
                do_sample=self._settings.llm_temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
