from __future__ import annotations

import shutil
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "paper_artifacts"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    ensure_dir(dst.parent)
    try:
        shutil.copy2(src, dst)
    except OSError as exc:
        print(f"Skipped {src} -> {dst}: {exc}")


def copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    ensure_dir(dst_dir)
    for src in sorted(src_dir.glob(pattern)):
        if src.is_file():
            copy_file(src, dst_dir / src.name)


def copy_relative_files(src_root: Path, dst_root: Path, suffixes: set[str]) -> None:
    if not src_root.exists():
        return
    for src in sorted(src_root.rglob("*")):
        if src.is_file() and src.suffix.lower() in suffixes:
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            if len(str(dst)) > 240:
                digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
                short_stem = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in src.stem)
                short_stem = short_stem[:70]
                short_name = f"{short_stem}__{digest}{src.suffix}"
                dst = dst_root / "_long_paths" / short_name
            copy_file(src, dst)


def main() -> None:
    ensure_dir(ARTIFACTS)

    copy_glob(ROOT / "display" / "plots", "*.pdf", ARTIFACTS / "figures")
    copy_glob(ROOT / "display" / "display_code" / "grid5_outputs", "*", ARTIFACTS / "tables" / "grid5")

    copy_relative_files(
        ROOT / "legacy" / "for_visualizing" / "legacy_analysis_outputs",
        ARTIFACTS / "tables" / "legacy_grid3",
        {".csv", ".txt", ".md", ".json", ".tex"},
    )

    copy_relative_files(
        ROOT / "validacion_teoria" / "results",
        ARTIFACTS / "theory_diagnostics",
        {".csv", ".txt", ".md", ".json", ".tex", ".pdf"},
    )

    print(f"Collected public artifacts in: {ARTIFACTS}")


if __name__ == "__main__":
    main()
