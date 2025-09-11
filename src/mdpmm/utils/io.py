from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | os.PathLike[str]) -> str:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def save_json(obj: Any, path: str | os.PathLike[str]) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(record: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

