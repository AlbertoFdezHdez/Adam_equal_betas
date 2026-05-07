from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional


def bootstrap_code_root() -> Path:
    code_root = Path(__file__).resolve().parents[1]
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    return code_root


def get_env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def require_dependency(module_name: str, install_hint: Optional[str] = None) -> None:
    try:
        __import__(module_name)
    except ImportError as exc:
        hint = f" Install hint: {install_hint}" if install_hint else ""
        raise RuntimeError(f"Missing optional dependency '{module_name}'.{hint}") from exc


def get_nanogpt_root() -> Path:
    env_value = os.environ.get("NANOGPT_ROOT")
    if env_value:
        return Path(env_value).expanduser()

    candidates = [
        Path("/scratch1/hernanal/nanoGPT"),
        Path("/opt/nanoGPT"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def nanogpt_repo_status(root: Optional[Path] = None) -> Dict[str, object]:
    root = root or get_nanogpt_root()
    required_paths = {
        "train.py": root / "train.py",
        "model.py": root / "model.py",
        "sample.py": root / "sample.py",
        "config": root / "config",
    }
    exists = {name: path.exists() for name, path in required_paths.items()}
    ok = all(exists.values())
    return {
        "root": str(root),
        "required_paths": {name: str(path) for name, path in required_paths.items()},
        "exists": exists,
        "ok": ok,
    }


def load_nanogpt_symbols():
    root = get_nanogpt_root()
    status = nanogpt_repo_status(root)
    if not status["ok"]:
        raise RuntimeError(
            f"NanoGPT repo not ready at {root}. Expected train.py, model.py, sample.py, and config/. "
            f"Set NANOGPT_ROOT explicitly if the repo lives elsewhere."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from model import GPT, GPTConfig
    except ImportError as exc:
        raise RuntimeError(
            f"Failed to import GPT/GPTConfig from nanoGPT repo at {root}. Check the repo checkout and dependencies."
        ) from exc
    return GPT, GPTConfig
