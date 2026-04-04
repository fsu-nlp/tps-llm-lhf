from __future__ import annotations

"""
Dataset loaders for SFT and preference training from JSONL files.

Supports:
  • SFT: either {prompt, response} records or {messages} (chat turns).
  • Preference: {chosen, rejected} chat-turn lists; extracts the prompt
    (first user message) and the last assistant replies for both arms.
Returns Hugging Face `datasets.Dataset` objects.
"""

import json
from typing import List, Iterable
from datasets import Dataset


def _read_jsonl(path: str) -> Iterable[dict]:
    """Yield JSON objects from a UTF-8 JSONL file, skipping blanks/invalid lines."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue


def _first_user(msgs: List[dict]) -> str:
    """Return the content of the first 'user' message in a chat transcript."""
    for m in msgs:
        if m.get("role") == "user":
            return str(m.get("content", "")).strip()
    return ""


def load_sft_dataset(path: str) -> Dataset:
    """Load an SFT dataset from JSONL into a Hugging Face `Dataset`.

    Accepts records of form:
      • {"prompt": str, "response": str}
      • {"chosen": [...]}  (chat list assigned to key 'messages' in output)

    Returns:
        Dataset with rows containing either {prompt, response} or {messages}.
    """
    rows = []
    for rec in _read_jsonl(path):
        prompt = rec.get("prompt")
        response = rec.get("response")
        if isinstance(prompt, str) and isinstance(response, str):
            rows.append({"prompt": prompt, "response": response})
            continue
        chosen = rec.get("chosen", [])
        if isinstance(chosen, list) and chosen:
            rows.append({"messages": chosen})
    return Dataset.from_list(rows)


def load_pref_dataset(path: str) -> Dataset:
    """Load a preference (pairwise) dataset from JSONL into a `Dataset`.

    Expects records with lists of chat turns:
      • {"chosen": [...], "rejected": [...]}

    Extracts:
      prompt  := first user message (from chosen or rejected)
      chosen  := last assistant message in chosen
      rejected:= last assistant message in rejected

    Returns:
        Dataset with columns {prompt, chosen, rejected}.
    """
    rows = []
    for rec in _read_jsonl(path):
        chosen, rejected = rec.get("chosen", []), rec.get("rejected", [])
        if not isinstance(chosen, list) or not isinstance(rejected, list):
            continue

        def last_assistant(msgs: List[dict]) -> str:
            for m in reversed(msgs):
                if m.get("role") == "assistant":
                    return str(m.get("content", ""))
            return ""

        ch, rj = last_assistant(chosen), last_assistant(rejected)
        if not ch or not rj:
            continue
        prompt = _first_user(chosen) or _first_user(rejected)
        rows.append({"prompt": prompt, "chosen": ch, "rejected": rj})
    return Dataset.from_list(rows)
