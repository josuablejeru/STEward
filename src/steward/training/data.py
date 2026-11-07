"""Turn SFT records into prompt/completion text for assistant-only loss.

The prompt is the model's chat template applied to the system and user
messages with ``add_generation_prompt=True`` and thinking disabled. For
Qwen3 this ends with an *empty* ``<think></think>`` block, which is the
template's own non-thinking format. The completion is only the STE target
plus the end-of-turn token. TRL's ``completion_only_loss`` then computes the
loss on completion tokens only: instruction, input, and template tokens are
masked. No reasoning traces are trained.

This module has no heavy imports; the tokenizer is passed in.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from steward.datasets.sft import SftRecord
from steward.generation.prompts import PromptTemplate


class ChatTokenizer(Protocol):
    """The tokenizer methods used here (Hugging Face compatible)."""

    eos_token: str

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        """Render a conversation."""
        ...


def messages_for(template: PromptTemplate, source: str) -> list[dict[str, str]]:
    """System and user messages for one input (same as inference)."""
    rendered = template.render(source_text=source)
    return [
        {"role": "system", "content": rendered.system},
        {"role": "user", "content": rendered.user},
    ]


def end_of_turn(tokenizer: ChatTokenizer) -> str:
    """The token that ends an assistant turn (``<|im_end|>`` for Qwen3)."""
    return tokenizer.eos_token


def to_prompt_completion(
    records: Sequence[SftRecord], template: PromptTemplate, tokenizer: ChatTokenizer
) -> list[dict[str, str]]:
    """Return ``{"prompt", "completion", "sft_id"}`` rows for TRL."""
    eot = end_of_turn(tokenizer)
    rows = []
    for record in records:
        prompt = tokenizer.apply_chat_template(
            messages_for(template, record.input),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rows.append(
            {"prompt": prompt, "completion": record.target.strip() + eot, "sft_id": record.sft_id}
        )
    return rows


def check_completion_mask(
    input_ids: Sequence[int],
    completion_mask: Sequence[int],
    expected_completion: str,
    decode: Any,
) -> tuple[bool, str]:
    """Check that exactly the completion tokens are unmasked.

    Returns ``(ok, decoded_unmasked_text)``.
    """
    unmasked = [t for t, m in zip(input_ids, completion_mask, strict=True) if m]
    text = decode(unmasked)
    return text.strip() == expected_completion.strip(), text
