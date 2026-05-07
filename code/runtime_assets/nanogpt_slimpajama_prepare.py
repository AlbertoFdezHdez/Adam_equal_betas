import glob
import os
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset


SLIMPAJAMA_DIR = Path(os.environ.get("SLIMPAJAMA_DIR", "/data/slimpajama-mini")).expanduser()
OUTPUT_DIR = Path(__file__).resolve().parent
LIMIT_TRAIN_DOCS = int(os.environ.get("LIMIT_TRAIN_DOCS", "200000"))
LIMIT_VAL_DOCS = int(os.environ.get("LIMIT_VAL_DOCS", "2000"))


def _load_stream(pattern: str):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files matched pattern: {pattern}")
    return load_dataset("json", data_files=files, split="train", streaming=True), files


def _write_split(name: str, pattern: str, limit_docs: int, enc) -> dict:
    dataset, files = _load_stream(pattern)
    output_path = OUTPUT_DIR / f"{name}.bin"
    docs = 0
    tokens = 0
    preview = None

    with output_path.open("wb") as f:
        for example in dataset:
            text = example.get("text", "")
            if preview is None:
                preview = text[:160]
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)
            arr = np.asarray(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            docs += 1
            tokens += int(arr.size)
            if limit_docs > 0 and docs >= limit_docs:
                break

    return {
        "output_path": str(output_path),
        "input_pattern": pattern,
        "matched_files": len(files),
        "docs_written": docs,
        "tokens_written": tokens,
        "preview": preview or "",
    }


def main():
    enc = tiktoken.get_encoding("gpt2")
    train_pattern = str(SLIMPAJAMA_DIR / "train" / "chunk1" / "*.jsonl.zst")
    val_pattern = str(SLIMPAJAMA_DIR / "validation" / "chunk1" / "*.jsonl.zst")

    train_info = _write_split("train", train_pattern, LIMIT_TRAIN_DOCS, enc)
    val_info = _write_split("val", val_pattern, LIMIT_VAL_DOCS, enc)

    print(
        {
            "slimpajama_dir": str(SLIMPAJAMA_DIR),
            "limit_train_docs": LIMIT_TRAIN_DOCS,
            "limit_val_docs": LIMIT_VAL_DOCS,
            "train": train_info,
            "val": val_info,
        }
    )


if __name__ == "__main__":
    main()
