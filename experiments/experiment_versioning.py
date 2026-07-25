from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GENERATED_PATH_PREFIXES = ("data/simulated/", "data/real_like/", "results/", "paper/figures/")
ROOT = Path(__file__).resolve().parents[1]


def _run_git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _source_marker(name: str) -> str:
    marker = Path(name)
    if marker.exists() and marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    return ""


def _status_path(line: str) -> str:
    parts = line.split(maxsplit=1)
    path = parts[1] if len(parts) == 2 else line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"').replace("\\", "/")


def git_metadata(ignore_paths: list[str | Path] | None = None) -> dict[str, Any]:
    status = _run_git(["status", "--short"])
    ignored = {Path(p).as_posix() for p in (ignore_paths or [])}
    status_lines = status.splitlines()
    generated = [
        line
        for line in status_lines
        if _status_path(line) in ignored or _status_path(line).startswith(GENERATED_PATH_PREFIXES)
    ]
    filtered = [line for line in status_lines if line not in generated]
    commit = _run_git(["rev-parse", "HEAD"]) or os.environ.get("SOURCE_GIT_COMMIT", "") or _source_marker(".source_commit")
    branch = (
        _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        or os.environ.get("SOURCE_GIT_BRANCH", "")
        or _source_marker(".source_branch")
    )
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(filtered),
        "status_short": filtered,
        "generated_status_short": generated,
        "status_short_all": status.splitlines(),
    }


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    exists = p.exists()
    try:
        display_path = p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        display_path = str(p)
    return {
        "path": display_path,
        "exists": exists,
        "size_bytes": p.stat().st_size if exists and p.is_file() else None,
        "sha256": sha256_file(p),
    }


def metadata_path(output_path: str | Path) -> Path:
    p = Path(output_path)
    return p.with_suffix(p.suffix + ".meta.json")


def write_run_metadata(
    output_path: str | Path,
    *,
    script: str,
    args: dict[str, Any],
    inputs: list[str | Path],
    outputs: list[str | Path],
    notes: str = "",
) -> Path:
    out = metadata_path(output_path)
    ignored_status_paths = [*outputs, out]
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": script,
        "argv": sys.argv,
        "args": args,
        "python": sys.version,
        "platform": platform.platform(),
        "git": git_metadata(ignore_paths=ignored_status_paths),
        "inputs": [file_record(p) for p in inputs],
        "outputs": [file_record(p) for p in outputs],
        "notes": notes,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
