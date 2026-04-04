from __future__ import annotations
import json
from typing import Dict, Iterator, List, Literal

Record = Dict[str, object]
Side = Literal["chosen", "rejected", "both"]

def read_jsonl(path: str) -> Iterator[Record]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                continue

def iter_assistant_texts(rec: Record, side: Side = "both") -> Iterator[str]:
    sides: List[str] = ["chosen", "rejected"] if side == "both" else [side]
    for s in sides:
        msgs = rec.get(s, [])
        if isinstance(msgs, list):
            for msg in msgs:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        yield content

def stream_texts(path: str, side: Side = "both") -> Iterator[str]:
    for rec in read_jsonl(path):
        yield from iter_assistant_texts(rec, side=side)
