#!/usr/bin/env python3
"""
Clean meta-comments/loops from second halves, line-by-line, via an OpenAI-compatible Chat Completions API,
then (optionally) strip simple <...> tags via regex (no spaces inside; up to a max length).

Determinism: defaults enforce deterministic decoding (temperature=0, top_p=1, n=1, no penalties).
If a provider rejects any param, the code drops only that param and retries while preserving determinism.

Inputs:
  - out/<PHASE>/**/*.jsonl (pass with --in-out-dir)
  - optional human halves via --in-human-halves
Outputs:
  - out_clean/<mirrored>.clean.jsonl
  - out_clean/cleaning_summary.{json,md}
  - out_clean/cleaning_diffs.jsonl (verbose per-item log)

Restart-safety:
  - Use --resume to skip inputs whose cleaned outputs already exist and to append to an existing diff log.

Throughput:
  - Parallelised with a thread pool (default --concurrency=32).
  - Optional --rpm limiter (global, across threads) to respect per-minute request limits.

Post-processing:
  - Optional regex-based stripping of simple angle-bracket tokens: '<([^\\s<>]{1,MAXLEN})>' (default MAXLEN=16).
    Captures items like <|user|>, <tag>, <X1>, but NOT strings with spaces ('< a b >') or attributes ('<ref id=3>').
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
import difflib
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Iterable, List, Optional, Sequence, Tuple, Pattern
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---- OpenAI client (Chat Completions) ----
try:
    from openai import OpenAI
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionSystemMessageParam,
        ChatCompletionUserMessageParam,
    )
except Exception:
    OpenAI = None  # type: ignore[assignment]

THIS_FILE = Path(__file__).resolve()
DEFAULT_REPO_ROOT = THIS_FILE.parent.parent  # assumes script in <repo>/scripts/

HUMAN_FIELDS = ["second_half", "human_second_half", "gold_second_half", "gold", "reference", "text"]
MODEL_FIELDS = ["text", "second_half", "completion", "response", "model_output", "output"]

CLEAN_INSTRUCTIONS = (
    "ROLE: Editorial cleaner for SCIENTIFIC ABSTRACT CONTINUATIONS (mid-abstract).\n"
    "ACTIONS: ONLY DELETE text; NEVER paraphrase, reorder, merge, or add. Keep sentence wording and order. "
    "If uncertain whether to delete, KEEP.\n"
    "\n"
    "DELETE:\n"
    "  1) Meta/AI persona: e.g., 'Certainly, here is ...', 'as an AI model', apologies, instructions, tool/safety notes.\n"
    "  2) Conversation turns & scaffolding (only if truly dialogic): "
    "Pseudo-dialogue markers <|user|>, <|assistant|>, </|user|>, </|assistant|> ONLY IF followed by chat-like material "
    "(greeting, instruction, apology, question to reader); delete the marker and that span; otherwise delete markers only.\n"
    "  4) Obvious repetition/loops: remove verbatim or near-verbatim repeats, KEEP one copy.\n"
    "  5) First/second-person META sentences using 'I/me/my' or direct address 'you/your'. Do NOT delete 'we/our/us'.\n"
    "\n"
    "PRESERVE:\n"
    "  Keep phrases like 'In conclusion'/'concluding' when embedded in a normal sentence.\n"
    "  - All scientific content and phrasing (incl. 'we/our/us').\n"
    "  - Angle-bracketed tokens in general (operators, tags, gene symbols, XML-like markup) EXCEPT the pseudo-dialogue markers listed above.\n"
    "  - Original wording, punctuation, and order. Do NOT fix grammar, reflow text, or change casing.\n"
    "\n"
    "OUTPUT: cleaned text ONLY (no quotes, no notes). If nothing but commentary remains, output an empty string."
)

def render_user_prompt(raw: str) -> str:
    return (
        "Clean the following MID-ABSTRACT CONTINUATION by applying ONLY the deletion rules. "
        "Do not paraphrase or rewrite; return the cleaned text only.\n\n"
        f"INPUT:\n{raw}\n\n"
        "OUTPUT (cleaned text only):"
    )


# ---------- Simple RPM limiter (global across threads) ----------
class RpmLimiter:
    """
    Enforces a minimum spacing between requests: interval = 60 / rpm seconds.
    Thread-safe. If rpm is falsy/None/<=0, limiter is disabled.
    """
    def __init__(self, rpm: Optional[int]):
        if rpm is None or rpm <= 0:
            self.enabled = False
            self.interval = 0.0
        else:
            self.enabled = True
            self.interval = 60.0 / float(rpm)
        self._lock = threading.Lock()
        self._next_time = time.monotonic()

    def wait(self):
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            if now < self._next_time:
                time.sleep(self._next_time - now)
                now = self._next_time
            self._next_time = now + self.interval

# ---------- OpenAI cleaner ----------
@dataclass
class CleanerCfg:
    model: str = "gpt-4o-mini"
    max_retries: int = 6
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None             # was 1.0
    frequency_penalty: Optional[float] = None # was 0.0
    presence_penalty: Optional[float] = None  # was 0.0
    n: int = 1
    seed: Optional[int] = 42

class Cleaner:
    def __init__(self, cfg: CleanerCfg):
        if OpenAI is None:
            raise RuntimeError("Missing dependency: `pip install -U openai`")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI()
        self.cfg = cfg

    def _make_kwargs(self, messages: List[ChatCompletionMessageParam]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "n": self.cfg.n,
        }
        # Deterministic decoding knobs (explicitly set)
        if self.cfg.temperature is not None:       kwargs["temperature"] = self.cfg.temperature
        if self.cfg.top_p is not None:             kwargs["top_p"] = self.cfg.top_p
        if self.cfg.frequency_penalty is not None: kwargs["frequency_penalty"] = self.cfg.frequency_penalty
        if self.cfg.presence_penalty is not None:  kwargs["presence_penalty"] = self.cfg.presence_penalty
        if self.cfg.seed is not None: kwargs["seed"] = int(self.cfg.seed)
        return kwargs

    def call(self, text: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Returns (cleaned_text, system_fingerprint, model_version_or_id)
        """
        last: Optional[Exception] = None
        sys_msg: ChatCompletionSystemMessageParam = {"role": "system", "content": CLEAN_INSTRUCTIONS}
        user_msg: ChatCompletionUserMessageParam = {"role": "user", "content": render_user_prompt(text)}
        messages: List[ChatCompletionMessageParam] = [sys_msg, user_msg]

        # Parameters that, if rejected by the server, we will drop one-by-one (but keep determinism).
        maybe_drop_keys = ["temperature", "top_p", "frequency_penalty", "presence_penalty"]
        dropped: set[str] = set()

        for attempt in range(self.cfg.max_retries):
            try:
                kwargs = self._make_kwargs(messages)
                # remove any previously dropped keys
                for k in list(dropped):
                    kwargs.pop(k, None)

                resp = self.client.chat.completions.create(**kwargs)
                content = (resp.choices[0].message.content or "").strip()
                fingerprint = getattr(resp, "system_fingerprint", None)
                model_id = getattr(resp, "model", None)
                return content, fingerprint, model_id
            except Exception as e:
                msg = getattr(e, "message", "") or str(e)
                # If the provider rejects a specific param, drop it and retry
                lowered = msg.lower()
                dropped_this_round = False
                for k in maybe_drop_keys:
                    if k in lowered or f"'{k}'" in lowered or "not supported" in lowered:
                        dropped.add(k)
                        dropped_this_round = True
                last = e
                # Exponential backoff, capped
                time.sleep(min(60, 2 ** attempt))
                if not dropped_this_round and attempt + 1 == self.cfg.max_retries:
                    break
        raise RuntimeError(f"OpenAI call failed after {self.cfg.max_retries} retries: {last}")

# ---------- Regex tag stripping ----------
@dataclass
class TagStripCfg:
    enabled: bool = True
    maxlen: int = 16

def compile_tag_regex(maxlen: int) -> Pattern[str]:
    # Match <...> with no whitespace and length 1..maxlen (non-greedy by construction via char class)
    return re.compile(rf"<([^\s<>]{{1,{maxlen}}})>")

def strip_angle_tags(text: str, tag_re: Optional[Pattern[str]]) -> Tuple[str, List[str], int]:
    """
    Remove all matches of tag_re from text. Returns (new_text, removed_tags, removed_char_count).
    """
    if tag_re is None:
        return text, [], 0
    removed: List[str] = [m.group(0) for m in tag_re.finditer(text)]
    if not removed:
        return text, [], 0
    new_text = tag_re.sub("", text)
    removed_chars = sum(len(t) for t in removed)
    return new_text, removed, removed_chars

# ---------- JSONL I/O ----------
def read_jsonl(p: Path):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)

def write_jsonl_line(out_fh, rec: Dict[str, Any]):
    out_fh.write(json.dumps(rec, ensure_ascii=False))
    out_fh.write("\n")

def append_jsonl(p: Path, recs: Iterable[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as out:
        for r in recs:
            out.write(json.dumps(r, ensure_ascii=False))
            out.write("\n")

def mirror_model_dest(src_file: Path, in_root: Path, out_root: Path) -> Path:
    rel = src_file.relative_to(in_root)
    dest = (out_root / rel).with_name(f"{rel.stem}.clean.jsonl")
    return dest

def mirror_human_dest(src_file: Path, out_root: Path) -> Path:
    return out_root / "human" / f"{src_file.stem}.clean.jsonl"

def pick_field(rec: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        v = rec.get(k)
        if isinstance(v, str):
            return k
    return None

# ---------- Diff utilities (verbose logging) ----------
def removed_spans(original: str, cleaned: str) -> Tuple[List[str], int]:
    sm = difflib.SequenceMatcher(a=original, b=cleaned, autojunk=False)
    removed: List[str] = []
    total = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "delete":
            seg = original[i1:i2]
            if seg:
                removed.append(seg)
                total += len(seg)
        elif tag == "replace":
            seg = original[i1:i2]
            if seg:
                removed.append(seg)
                total += len(seg)
    return removed, total

# ---------- Cleaning passes (parallel) ----------
def clean_file(src: Path, cleaner: Cleaner, mode: str, in_root: Path, out_root: Path,
               log_path: Path, concurrency: int, rpm_limiter: Optional[RpmLimiter],
               tag_cfg: TagStripCfg, tag_re: Optional[Pattern[str]]) -> Dict[str, Any]:
    if mode == "model":
        dest = mirror_model_dest(src, in_root, out_root)
    else:
        dest = mirror_human_dest(src, out_root)
    dest.parent.mkdir(parents=True, exist_ok=True)

    stats: Dict[str, Any] = {
        "source": str(src),
        "dest": str(dest),
        "items": 0,
        "sent": 0,
        "changed": 0,
        "chars_removed": 0,
        "missing_field": 0,
        "tags_removed": 0,
        "tag_chars_removed": 0,
    }
    fields = HUMAN_FIELDS if mode == "human" else MODEL_FIELDS

    # Log a stable hash of the system+prompt template for full reproducibility
    prompt_signature = hashlib.sha256(
        (CLEAN_INSTRUCTIONS + "\n" + "TASK_PROMPT_TEMPLATE:" + render_user_prompt("{{TEXT}}")).encode("utf-8")
    ).hexdigest()

    # Prepare work items
    recs = list(read_jsonl(src))
    total_items = len(recs)
    stats["items"] = total_items

    # Thread-safe pieces for buffered logging & ordered writing
    next_to_write = 0
    results: Dict[int, Tuple[Dict[str, Any], Optional[Dict[str, Any]], Dict[str, int]]] = {}
    log_buffer: List[Dict[str, Any]] = []

    # Worker function
    def work(idx: int, rec: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Optional[Dict[str, Any]], Dict[str, int]]:
        per_stats = {"sent": 0, "changed": 0, "chars_removed": 0, "missing_field": 0, "tags_removed": 0, "tag_chars_removed": 0}
        field = pick_field(rec, fields)
        if not field:
            per_stats["missing_field"] = 1
            # pass through unchanged
            return idx, rec, None, per_stats

        original = rec[field]

        # Throttle if requested
        if rpm_limiter is not None:
            rpm_limiter.wait()

        # API call (GPT-based cleaning)
        gpt_cleaned, system_fingerprint, model_version = cleaner.call(original)
        per_stats["sent"] = 1

        # Optional regex tag stripping (applied AFTER GPT cleaning)
        if tag_cfg.enabled:
            final_cleaned, removed_tags, tag_removed_chars_only = strip_angle_tags(gpt_cleaned, tag_re)
        else:
            final_cleaned, removed_tags, tag_removed_chars_only = gpt_cleaned, [], 0

        # Per-stage and overall diffs
        _, gpt_removed_chars = removed_spans(original, gpt_cleaned)
        _, tag_stage_removed_chars = removed_spans(gpt_cleaned, final_cleaned)
        removed_list_total, removed_chars_total = removed_spans(original, final_cleaned)

        changed = final_cleaned != original
        if changed:
            per_stats["changed"] = 1
            per_stats["chars_removed"] = max(0, removed_chars_total)
        if removed_tags:
            per_stats["tags_removed"] = len(removed_tags)
            per_stats["tag_chars_removed"] = max(0, tag_stage_removed_chars)

        # Enrich record
        out_rec = dict(rec)
        out_rec["cleaned_text"] = final_cleaned
        out_rec.setdefault("cleaning", {})
        out_rec["cleaning"].update({
            "source_field": field,
            "orig_len": len(original),
            "clean_len": len(final_cleaned),
            "chars_removed": max(0, removed_chars_total),
            "orig_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "clean_sha256": hashlib.sha256(final_cleaned.encode("utf-8")).hexdigest(),
            "model_used": cleaner.cfg.model,
            "api": "chat.completions",
            "system_fingerprint": system_fingerprint,
            "model_version": model_version,
            "prompt_signature": prompt_signature,
            "decoding": {
                "temperature": cleaner.cfg.temperature,
                "top_p": cleaner.cfg.top_p,
                "frequency_penalty": cleaner.cfg.frequency_penalty,
                "presence_penalty": cleaner.cfg.presence_penalty,
                "n": cleaner.cfg.n
            },
            "stage_removed_chars": {
                "gpt": max(0, gpt_removed_chars),
                "regex_tags": max(0, tag_stage_removed_chars),
                "total": max(0, removed_chars_total),
            },
            "regex_tag_strip": {
                "enabled": bool(tag_cfg.enabled),
                "pattern": tag_re.pattern if (tag_cfg.enabled and tag_re is not None) else "",
                "maxlen": tag_cfg.maxlen,
                "count": len(removed_tags),
                "removed": removed_tags,
                "removed_chars": max(0, tag_stage_removed_chars),
            },
        })

        log_entry = {
            "src_file": str(src),
            "dest_file": str(dest),
            "mode": mode,
            "item_index": idx,
            "record_id": rec.get("id", None),
            "source_field": field,
            "model_used": cleaner.cfg.model,
            "system_fingerprint": system_fingerprint,
            "model_version": model_version,
            "prompt_signature": prompt_signature,
            "changed": changed,
            "orig_len": len(original),
            "clean_len": len(final_cleaned),
            "chars_removed": max(0, removed_chars_total),
            "removed_substrings": removed_list_total,
            "original_text": original,
            "cleaned_text": final_cleaned,
            "decoding": {
                "temperature": cleaner.cfg.temperature,
                "top_p": cleaner.cfg.top_p,
                "frequency_penalty": cleaner.cfg.frequency_penalty,
                "presence_penalty": cleaner.cfg.presence_penalty,
                "n": cleaner.cfg.n
            },
            "stage_removed_chars": {
                "gpt": max(0, gpt_removed_chars),
                "regex_tags": max(0, tag_stage_removed_chars),
                "total": max(0, removed_chars_total),
            },
            "regex_tag_removed": {
                "pattern": tag_re.pattern if (tag_cfg.enabled and tag_re is not None) else "",
                "maxlen": tag_cfg.maxlen,
                "count": len(removed_tags),
                "tags": removed_tags,
                "removed_chars": max(0, tag_stage_removed_chars),
            },
        }
        return idx, out_rec, log_entry, per_stats

    # Execute with bounded concurrency; maintain original order on disk
    with open(dest, "w", encoding="utf-8") as out_fh:
        with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as ex:
            futures = [ex.submit(work, i, recs[i]) for i in range(total_items)]
            for fut in as_completed(futures):
                idx, out_rec, log_entry, per_stats = fut.result()
                results[idx] = (out_rec, log_entry, per_stats)

                # Flush in-order records as they become available
                while next_to_write in results:
                    w_rec, w_log, w_stats = results.pop(next_to_write)
                    # update stats
                    for k in ["sent", "changed", "chars_removed", "missing_field", "tags_removed", "tag_chars_removed"]:
                        stats[k] += w_stats[k]
                    # write record
                    write_jsonl_line(out_fh, w_rec)
                    # buffer log
                    if w_log is not None:
                        log_buffer.append(w_log)
                        if len(log_buffer) >= 1000:
                            append_jsonl(log_path, log_buffer)
                            log_buffer = []
                    next_to_write += 1

        # Flush any remaining logs
        if log_buffer:
            append_jsonl(log_path, log_buffer)

    return stats

def path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False

def clean_all(cleaner: Cleaner, repo_root: Path, in_out_dirs: Sequence[Path],
              human_files: Sequence[Path], out_root: Path, log_diffs: Optional[Path],
              resume: bool, concurrency: int, rpm: Optional[int],
              tag_cfg: TagStripCfg) -> Dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    log_path = (log_diffs or (out_root / "cleaning_diffs.jsonl")).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Only truncate the log if not resuming
    if not resume:
        with open(log_path, "w", encoding="utf-8"):
            pass

    # Compile regex once
    tag_re = compile_tag_regex(tag_cfg.maxlen) if tag_cfg.enabled else None

    summary: Dict[str, Any] = {
        "config": asdict(cleaner.cfg),
        "repo_root": str(repo_root),
        "human": [],
        "files": [],
        "totals": {"items":0,"sent":0,"changed":0,"chars_removed":0,"missing_field":0,"tags_removed":0,"tag_chars_removed":0},
        "by_model_name": defaultdict(lambda: {"items":0,"sent":0,"changed":0,"chars_removed":0,"tags_removed":0,"tag_chars_removed":0}),
        "log_diffs": str(log_path),
        "system_fingerprints": set(),
        "model_versions": set(),
        "concurrency": concurrency,
        "rpm": rpm or 0,
        "tag_strip": {"enabled": bool(tag_cfg.enabled), "pattern": (tag_re.pattern if tag_re else ""), "maxlen": tag_cfg.maxlen},
    }

    rpm_limiter = RpmLimiter(rpm)

    # human halves
    for human_path in human_files:
        if human_path.exists():
            if resume:
                dest_h = mirror_human_dest(human_path, out_root)
                if dest_h.exists() and dest_h.stat().st_size > 0:
                    continue
            hstats = clean_file(human_path, cleaner, mode="human",
                                in_root=repo_root, out_root=out_root,
                                log_path=log_path, concurrency=concurrency, rpm_limiter=rpm_limiter,
                                tag_cfg=tag_cfg, tag_re=tag_re)
            summary["human"].append(hstats)
            for k in ["items","sent","changed","chars_removed","missing_field","tags_removed","tag_chars_removed"]:
                summary["totals"][k] += hstats[k]

    for in_root in in_out_dirs:
        if not in_root.exists():
            continue
        if path_is_within(in_root, out_root):
            continue
        for src in sorted(in_root.rglob("*.jsonl")):
            if path_is_within(src, out_root):
                continue
            if resume:
                dest_m = mirror_model_dest(src, in_root, out_root)
                if dest_m.exists() and dest_m.stat().st_size > 0:
                    continue
            fstats = clean_file(src, cleaner, mode="model",
                                in_root=in_root, out_root=out_root,
                                log_path=log_path, concurrency=concurrency, rpm_limiter=rpm_limiter,
                                tag_cfg=tag_cfg, tag_re=tag_re)
            model_name = src.parent.name if src.parent else "unknown"
            fstats["model_name"] = model_name
            summary["files"].append(fstats)
            for k in ["items","sent","changed","chars_removed","missing_field","tags_removed","tag_chars_removed"]:
                summary["totals"][k] += fstats[k]
            bm = summary["by_model_name"][model_name]
            bm["items"] += fstats["items"]
            bm["sent"] += fstats["sent"]
            bm["changed"] += fstats["changed"]
            bm["chars_removed"] += fstats["chars_removed"]
            bm["tags_removed"] += fstats["tags_removed"]
            bm["tag_chars_removed"] += fstats["tag_chars_removed"]

    # scan diff log once to collect fingerprints/versions (cheap linear pass)
    try:
        fps, vers = set(), set()
        with open(summary["log_diffs"], "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("system_fingerprint"):
                    fps.add(d["system_fingerprint"])
                if d.get("model_version"):
                    vers.add(d["model_version"])
        summary["system_fingerprints"] = sorted(fps)
        summary["model_versions"] = sorted(vers)
    except Exception:
        pass

    with open(out_root / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# Cleaning summary\n\n")
    md.append(f"- Model: `{cleaner.cfg.model}`\n")
    md.append(f"- Concurrency: {summary['concurrency']}  |  RPM limit: {summary['rpm']}\n")
    md.append(f"- Diff log: `{summary['log_diffs']}`\n")
    md.append(f"- System fingerprints: {', '.join(summary.get('system_fingerprints', [])) or 'n/a'}\n")
    md.append(f"- Model versions: {', '.join(summary.get('model_versions', [])) or 'n/a'}\n")
    md.append(f"- Tag strip: enabled={summary['tag_strip']['enabled']} | pattern=`{summary['tag_strip']['pattern']}` | maxlen={summary['tag_strip']['maxlen']}\n")
    md.append("## Totals\n")
    t = summary["totals"]
    md.append(f"- Items: {t['items']}\n- Sent: {t['sent']}\n- Changed: {t['changed']}\n"
              f"- Characters removed: {t['chars_removed']}\n- Records missing target field: {t['missing_field']}\n"
              f"- Tags removed: {t['tags_removed']}\n- Tag characters removed: {t['tag_chars_removed']}\n\n")
    if summary["human"]:
        md.append("## Human halves\n")
        for h in summary["human"]:
            md.append(f"- {h['source']} → {h['dest']} | Items: {h['items']}; Sent: {h['sent']}; Changed: {h['changed']}; Removed: {h['chars_removed']}; Tags: {h['tags_removed']}\n")
        md.append("\n")
    md.append("## By folder (immediate parent of each input file)\n")
    for m, v in sorted(summary["by_model_name"].items()):
        md.append(f"- **{m}** — Items: {v['items']}; Sent: {v['sent']}; Changed: {v['changed']}; Removed: {v['chars_removed']}; "
                  f"Tags: {v['tags_removed']}; Tag chars: {v['tag_chars_removed']}\n")
    with open(out_root / "cleaning_summary.md", "w", encoding="utf-8") as f:
        f.write("".join(md))

    return summary

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Clean meta-commentary/loops from JSONL generations (repo-relative paths), then optionally strip <...> tags.")
    ap.add_argument("--model", default="gpt-4o-mini", help="OpenAI model id (default: gpt-4o-mini)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Sampling temperature (default: 0.0 for determinism).")
    ap.add_argument("--top-p", dest="top_p", type=float, default=1.0,
                    help="Nucleus sampling top_p (default: 1.0). With T=0 this does not introduce randomness.")
    ap.add_argument("--frequency-penalty", type=float, default=0.0,
                    help="Frequency penalty (default: 0.0).")
    ap.add_argument("--presence-penalty", type=float, default=0.0,
                    help="Presence penalty (default: 0.0).")
    ap.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT,
                    help=f"Repository root (default: {DEFAULT_REPO_ROOT})")
    ap.add_argument("--in-out-dir", type=Path, action="append", default=None,
                    help="Input dir(s) for model outputs. Pass e.g. out/pre and/or out/post. May be given multiple times.")
    ap.add_argument("--in-human-halves", type=Path, action="append", default=None,
                    help="Path(s) to human halves JSONL (optional; may be given multiple times).")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="Output root (default: <repo_root>/out_clean)")
    ap.add_argument("--log-diffs", type=Path, default=None,
                    help="Path to verbose JSONL diff log (default: <out_root>/cleaning_diffs.jsonl)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip inputs whose cleaned outputs already exist; append to existing diff log.")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="Max parallel requests (default: 32).")
    ap.add_argument("--rpm", type=int, default=0,
                    help="Optional global requests-per-minute cap across threads (default: 0=disabled).")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic seed for decoding (recommended for reproducibility).")

    # Tag stripping controls
    ap.add_argument("--strip-angle-tags", dest="strip_angle_tags", action="store_true",
                    help="Enable regex-based removal of simple <...> tags (no spaces, length <= tag-maxlen). (default: enabled)")
    ap.add_argument("--no-strip-angle-tags", dest="strip_angle_tags", action="store_false",
                    help="Disable regex-based tag stripping.")
    ap.add_argument("--tag-maxlen", type=int, default=16,
                    help="Maximum characters allowed between < and > for a tag to be stripped (default: 16).")
    ap.set_defaults(strip_angle_tags=True)

    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    in_out_dirs = [p.resolve() for p in (args.in_out_dir or [repo_root / "out"])]
    human_files = [p.resolve() for p in (args.in_human_halves or [])]
    out_root = (args.out_root or (repo_root / "out_clean")).resolve()

    cfg = CleanerCfg(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        seed=args.seed,
    )
    tag_cfg = TagStripCfg(enabled=bool(args.strip_angle_tags), maxlen=int(args.tag_maxlen))

    cleaner = Cleaner(cfg)
    summary = clean_all(cleaner, repo_root, in_out_dirs, human_files, out_root,
                        args.log_diffs, args.resume, args.concurrency, args.rpm,
                        tag_cfg=tag_cfg)

    t = summary["totals"]
    print("== Done ==")
    print(f"Items: {t['items']} | Sent: {t['sent']} | Changed: {t['changed']} | Removed chars: {t['chars_removed']}")
    print(f"Tags removed: {t['tags_removed']} | Tag chars removed: {t['tag_chars_removed']}")
    print(f"Concurrency: {summary['concurrency']} | RPM limit: {summary['rpm']}")
    print(f"Tag strip: enabled={summary['tag_strip']['enabled']} pattern='{summary['tag_strip']['pattern']}' maxlen={summary['tag_strip']['maxlen']}")
    print(f"Summary: {out_root / 'cleaning_summary.md'}")
    print(f"Verbose diff log: {summary['log_diffs']}")

if __name__ == "__main__":
    main()
