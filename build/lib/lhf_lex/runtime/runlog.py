
from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

def stable_hash(obj: Any) -> str:
    """SHA256 over a canonical JSON serialisation."""
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return sha256(data).hexdigest()

def get_git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None, stderr=subprocess.DEVNULL, timeout=2)
        return out.decode("utf-8").strip()
    except Exception:
        return None

def get_env_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    # Python & OS
    info["python"] = sys.version.split()[0]
    info["platform"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
    }
    info["hostname"] = socket.gethostname()

    # Libraries
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_compiled"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            try:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
                info["gpu_count"] = torch.cuda.device_count()
            except Exception:
                pass
    except Exception:
        info["torch"] = None

    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None

    try:
        import peft
        info["peft"] = peft.__version__
    except Exception:
        info["peft"] = None

    try:
        import huggingface_hub as hf_hub
        info["huggingface_hub"] = hf_hub.__version__
    except Exception:
        info["huggingface_hub"] = None

    info["git_sha"] = get_git_sha(Path.cwd())
    return info

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))

def write_per_prompt_record(dirpath: Path, prompt_id: int, record: Dict[str, Any]) -> Path:
    """Atomically write one JSON file per prompt id, return final path."""
    dirpath.mkdir(parents=True, exist_ok=True)
    fn = f"{prompt_id:06d}.json"
    p_final = dirpath / fn
    data = json.dumps(record, ensure_ascii=False)
    atomic_write_text(p_final, data + "\n")
    return p_final

def merge_jsonl(parts_dir: Path, out_jsonl: Path) -> None:
    """Deterministically merge per-prompt JSON files into a single JSONL (sorted by id)."""
    files = sorted(p for p in parts_dir.glob("*.json"))
    with open(out_jsonl.with_suffix(out_jsonl.suffix + ".partial"), "w", encoding="utf-8") as out:
        for p in files:
            with open(p, "r", encoding="utf-8") as f:
                out.write(f.read())
    os.replace(out_jsonl.with_suffix(out_jsonl.suffix + ".partial"), out_jsonl)

def load_prompts(path: Path, limit: Optional[int] = None) -> List[str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                lines.append(ln)
                if limit is not None and len(lines) >= limit:
                    break
    return lines

def compute_prompt_sha(prompt: str) -> str:
    return sha256(prompt.encode("utf-8")).hexdigest()

def make_run_id() -> str:
    t = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    rnd = os.urandom(3).hex()
    return f"{t}-{rnd}"

def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

def scan_completed(parts_dir: Path) -> List[int]:
    ids = []
    for p in parts_dir.glob("*.json"):
        try:
            i = int(p.stem)
            ids.append(i)
        except Exception:
            continue
    return sorted(ids)
