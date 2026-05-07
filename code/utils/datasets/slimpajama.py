import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..runtime import get_nanogpt_root


DEFAULT_DATASET_NAME = "slimpajama"
DEFAULT_EXTERNAL_DATA_DIR = "/scratch1/hernanal/nanoGPT/data/slimpajama"


def resolve_nanogpt_data_dir(data_dir: Optional[str] = None, dataset: str = DEFAULT_DATASET_NAME) -> Path:
    if data_dir:
        return Path(data_dir).expanduser()

    env_dir = os.environ.get("NANOGPT_DATA_DIR")
    if env_dir:
        return Path(env_dir).expanduser()

    nanogpt_root = get_nanogpt_root()
    repo_relative = nanogpt_root / "data" / dataset
    if repo_relative.exists():
        return repo_relative

    if dataset == DEFAULT_DATASET_NAME:
        return Path(DEFAULT_EXTERNAL_DATA_DIR)

    return Path("data") / dataset


def _bin_paths(data_dir: Optional[str] = None, dataset: str = DEFAULT_DATASET_NAME) -> Tuple[Path, Path, Path]:
    root = resolve_nanogpt_data_dir(data_dir=data_dir, dataset=dataset)
    return root, root / "train.bin", root / "val.bin"


def inspect_slimpajama_bins(data_dir: Optional[str] = None, dataset: str = DEFAULT_DATASET_NAME) -> Dict[str, object]:
    root, train_path, val_path = _bin_paths(data_dir=data_dir, dataset=dataset)
    if not train_path.is_file():
        raise RuntimeError(f"train.bin not found at {train_path}")
    if not val_path.is_file():
        raise RuntimeError(f"val.bin not found at {val_path}")

    train_tokens = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_tokens = np.memmap(val_path, dtype=np.uint16, mode="r")

    return {
        "data_dir": str(root),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
        "dtype": "uint16",
        "train_bin_size_bytes": train_path.stat().st_size,
        "val_bin_size_bytes": val_path.stat().st_size,
        "train_tokens": int(train_tokens.shape[0]),
        "val_tokens": int(val_tokens.shape[0]),
    }


class NanoGPTBinDataset(Dataset):
    def __init__(
        self,
        split: str,
        block_size: int,
        data_dir: Optional[str] = None,
        dataset: str = DEFAULT_DATASET_NAME,
        max_windows: Optional[int] = None,
    ):
        if split not in ("train", "val", "validation"):
            raise ValueError("split must be 'train', 'val', or 'validation'")

        root, train_path, val_path = _bin_paths(data_dir=data_dir, dataset=dataset)
        actual_split = "val" if split == "validation" else split
        bin_path = train_path if actual_split == "train" else val_path
        if not bin_path.is_file():
            raise RuntimeError(f"{actual_split}.bin not found at {bin_path}")

        self.data_dir = root
        self.split = actual_split
        self.block_size = int(block_size)
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        if self.data.shape[0] <= self.block_size + 1:
            raise RuntimeError(
                f"{bin_path} is too short for block_size={self.block_size}. "
                f"Tokens available: {self.data.shape[0]}"
            )
        self.max_start = int(self.data.shape[0] - self.block_size - 1)
        self.length = self.max_start + 1
        if max_windows is not None:
            self.length = min(int(max_windows), self.length)
        self._stride = max(1, (self.max_start + 1) // max(1, self.length))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        start = min(int(idx) * self._stride, self.max_start)
        x = np.array(self.data[start : start + self.block_size], dtype=np.int64)
        y = np.array(self.data[start + 1 : start + 1 + self.block_size], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_slimpajama(
    batch_size: int = 8,
    max_length: int = 1024,
    num_workers: int = 0,
    pin_memory: bool = True,
    out_dir: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    percent: float = 0.51,
    seed: int = 1337,
    force_redownload: bool = False,
    revision: Optional[str] = None,
    tokenizer_name: str = "gpt2",
    subset_size_train: Optional[int] = None,
    subset_size_val: Optional[int] = None,
    data_dir: Optional[str] = None,
    dataset: str = DEFAULT_DATASET_NAME,
) -> Tuple[DataLoader, DataLoader, int]:
    del out_dir, hf_cache_dir, percent, seed, force_redownload, revision, tokenizer_name

    train_ds = NanoGPTBinDataset(
        split="train",
        block_size=max_length,
        data_dir=data_dir,
        dataset=dataset,
        max_windows=subset_size_train,
    )
    val_ds = NanoGPTBinDataset(
        split="val",
        block_size=max_length,
        data_dir=data_dir,
        dataset=dataset,
        max_windows=subset_size_val,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, 50257


__all__ = [
    "DEFAULT_DATASET_NAME",
    "DEFAULT_EXTERNAL_DATA_DIR",
    "inspect_slimpajama_bins",
    "load_slimpajama",
    "resolve_nanogpt_data_dir",
]
