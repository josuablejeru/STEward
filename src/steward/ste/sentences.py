"""Simple sentence splitting and word counting.

This is a rule-based approximation, not a full parser. Known limitations:
abbreviations with periods (for example "approx.") can split a sentence,
and list items without final punctuation count as one sentence per line.
"""

from __future__ import annotations

import re

_LINE_ITEM_RE = re.compile(r"^\s*(?:[-*•]|\(?[0-9A-Za-z]{1,3}[.)])\s+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?:])\s+(?=[\"'(\[]?[A-Z0-9])")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'./][A-Za-z0-9]+)*")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences. Each non-empty line is split separately."""
    sentences: list[str] = []
    for line in text.splitlines():
        line = _LINE_ITEM_RE.sub("", line).strip()
        if line:
            sentences.extend(s.strip() for s in _SENTENCE_END_RE.split(line) if s.strip())
    return sentences


def words(text: str) -> list[str]:
    """Return word tokens. Hyphenated words and numbers like ``2.5`` count as one."""
    return _WORD_RE.findall(text)


def word_count(text: str) -> int:
    """Return the number of word tokens in ``text``."""
    return len(words(text))
