from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = BASE_DIR / "descargas" / "adam_gradient_cifar10"
DEFAULT_DATA_DIR = BASE_DIR / "descargas" / "data"


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    val_size: int
    probe_batch_size: int | None
    grad_dtype: str
    save_gradients: str
    saved_gradient_source: str
    scheduler: str
    num_workers: int


class TinyCIFARCNN(nn.Module):
    """Small enough for CPU experiments, expressive enough to learn CIFAR-10."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a toy CIFAR-10 CNN with Adam while streaming per-batch "
            "gradients to disk for beta1/beta2 validation experiments."
        )
    )
    parser.add_argument("--mode", choices=["single", "grid"], default="single")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--beta-values", type=float, nargs="+", default=[0.9, 0.999])
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=None,
        help=(
            "Batch size for the fixed validation probe batch. Defaults to --batch-size. "
            "Only affects --saved-gradient-source probe and validation DataLoader batching."
        ),
    )
    parser.add_argument("--grad-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--save-gradients",
        choices=["full", "stats-only", "none"],
        default="full",
        help="full writes all flattened gradients to a .npy memmap; stats-only keeps CSV summaries.",
    )
    parser.add_argument(
        "--saved-gradient-source",
        choices=["train", "probe"],
        default="train",
        help=(
            "train saves the gradient used by Adam; probe saves the gradient of one fixed "
            "validation batch, while Adam still steps with the train-batch gradient."
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine"],
        default="none",
        help="Default is constant LR. Cosine is available for later comparisons.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def set_reproducible_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))


def make_loaders(
    data_dir: Path,
    batch_size: int,
    val_batch_size: int,
    val_size: int,
    seed: int,
    num_workers: int,
    download: bool,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    full_train_aug = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=download, transform=train_transform
    )
    full_train_eval = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=download, transform=eval_transform
    )

    if not 0 < val_size < len(full_train_aug):
        raise ValueError(f"--val-size must be between 1 and {len(full_train_aug) - 1}")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(full_train_aug), generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]

    train_dataset = torch.utils.data.Subset(full_train_aug, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_eval, val_indices)

    loader_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


def param_metadata(model: nn.Module) -> tuple[list[dict[str, object]], int]:
    rows: list[dict[str, object]] = []
    offset = 0
    for name, parameter in model.named_parameters():
        n = parameter.numel()
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": n,
                "start": offset,
                "stop": offset + n,
            }
        )
        offset += n
    return rows, offset


def bytes_per_value(dtype_name: str) -> int:
    if dtype_name == "float16":
        return 2
    if dtype_name == "float32":
        return 4
    raise ValueError(dtype_name)


def fmt_float_for_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def estimate_storage(
    *,
    model: nn.Module,
    epochs: int,
    batch_size: int,
    train_size: int,
    beta_count: int,
    grad_dtype: str,
    save_gradients: str,
    saved_gradient_source: str,
) -> dict[str, object]:
    metadata, n_params = param_metadata(model)
    train_batches_per_epoch = math.ceil(train_size / batch_size)
    train_steps_per_run = train_batches_per_epoch * epochs
    per_run_bytes = (
        n_params * train_steps_per_run * bytes_per_value(grad_dtype)
        if save_gradients == "full"
        else 0
    )
    return {
        "model": "TinyCIFARCNN",
        "n_parameter_tensors": len(metadata),
        "n_trainable_parameters": n_params,
        "gradient_dtype": grad_dtype,
        "saved_gradient_source": saved_gradient_source,
        "bytes_per_gradient_value": bytes_per_value(grad_dtype),
        "train_size": train_size,
        "batch_size": batch_size,
        "train_batches_per_epoch": train_batches_per_epoch,
        "epochs": epochs,
        "train_steps_per_run": train_steps_per_run,
        "number_of_runs": beta_count,
        "full_gradients_enabled": save_gradients == "full",
        "estimated_gradient_storage_per_run_bytes": per_run_bytes,
        "estimated_gradient_storage_all_runs_bytes": per_run_bytes * beta_count,
        "estimated_gradient_storage_per_run_mib": per_run_bytes / 1024**2,
        "estimated_gradient_storage_all_runs_mib": (per_run_bytes * beta_count) / 1024**2,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def open_grad_memmap(path: Path, shape: tuple[int, int], dtype_name: str) -> np.memmap:
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if dtype_name == "float16" else np.float32
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def flattened_gradient_f32(model: nn.Module) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            chunks.append(np.zeros(parameter.numel(), dtype=np.float32))
        else:
            chunks.append(parameter.grad.detach().cpu().reshape(-1).numpy().astype(np.float32, copy=False))
    return np.concatenate(chunks)


def cast_gradient_for_storage(grad_f32: np.ndarray, dtype_name: str) -> np.ndarray:
    if dtype_name == "float16":
        return grad_f32.astype(np.float16)
    return grad_f32


def clone_current_grads(model: nn.Module) -> list[torch.Tensor | None]:
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in model.parameters()]


def restore_grads(model: nn.Module, grads: list[torch.Tensor | None]) -> None:
    for parameter, grad in zip(model.parameters(), grads):
        parameter.grad = None if grad is None else grad


def first_probe_batch(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = next(iter(loader))
    return images.to(device), targets.to(device)


def probe_gradient_f32(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[np.ndarray, float, float]:
    was_training = model.training
    model.eval()
    logits = model(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    grad_f32 = flattened_gradient_f32(model)
    accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
    if was_training:
        model.train()
    return grad_f32, float(loss.item()), accuracy


def grad_stats(grad_f32: np.ndarray, previous_grad_f32: np.ndarray | None) -> dict[str, float]:
    norm = float(np.linalg.norm(grad_f32))
    if previous_grad_f32 is None:
        cosine = float("nan")
    else:
        denom = norm * float(np.linalg.norm(previous_grad_f32))
        cosine = float(np.dot(grad_f32, previous_grad_f32) / denom) if denom > 0 else float("nan")
    return {
        "grad_l2": norm,
        "grad_mean": float(np.mean(grad_f32)),
        "grad_abs_mean": float(np.mean(np.abs(grad_f32))),
        "grad_max_abs": float(np.max(np.abs(grad_f32))),
        "grad_cosine_to_previous_batch": cosine,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_seen += int(targets.numel())
    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


def run_one(
    config: ExperimentConfig,
    output_dir: Path,
    data_dir: Path,
    download: bool,
) -> dict[str, object]:
    set_reproducible_seed(config.seed)
    device = torch.device("cpu")
    train_loader, val_loader = make_loaders(
        data_dir=data_dir,
        batch_size=config.batch_size,
        val_batch_size=config.probe_batch_size or config.batch_size,
        val_size=config.val_size,
        seed=config.seed,
        num_workers=config.num_workers,
        download=download,
    )
    model = TinyCIFARCNN().to(device)
    probe_images: torch.Tensor | None = None
    probe_targets: torch.Tensor | None = None
    if config.saved_gradient_source == "probe":
        probe_images, probe_targets = first_probe_batch(val_loader, device)

    metadata, n_params = param_metadata(model)
    train_steps = len(train_loader) * config.epochs

    probe_suffix = (
        f"_pbs{config.probe_batch_size or config.batch_size}"
        if config.saved_gradient_source == "probe"
        else ""
    )
    run_name = (
        f"{config.dataset}_tinycnn"
        f"_b1{fmt_float_for_name(config.beta1)}"
        f"_b2{fmt_float_for_name(config.beta2)}"
        f"_bs{config.batch_size}"
        f"_{config.saved_gradient_source}"
        f"{probe_suffix}"
        f"_s{config.seed}"
    )
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "config.json", asdict(config))
    write_json(run_dir / "param_metadata.json", metadata)
    if probe_targets is not None:
        write_json(
            run_dir / "probe_batch_metadata.json",
            {
                "source": "first validation batch after deterministic train/val split",
                "batch_size": int(probe_targets.numel()),
                "requested_probe_batch_size": config.probe_batch_size or config.batch_size,
                "uses_eval_mode": True,
                "uses_data_augmentation": False,
                "note": (
                    "Probe gradients are computed before optimizer.step() at each iteration. "
                    "They are saved, then train gradients are restored for Adam."
                ),
            },
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        if config.scheduler == "cosine"
        else None
    )

    gradients = None
    if config.save_gradients == "full":
        gradients = open_grad_memmap(
            run_dir / "gradients.npy",
            shape=(train_steps, n_params),
            dtype_name=config.grad_dtype,
        )

    metrics_path = run_dir / "metrics_epoch.csv"
    grad_stats_path = run_dir / "grad_stats_batch.csv"
    batch_metrics_path = run_dir / "metrics_batch.csv"
    probe_metrics_path = run_dir / "probe_metrics_batch.csv"

    start_time = time.time()
    global_step = 0
    previous_grad: np.ndarray | None = None
    history: list[dict[str, float | int]] = []

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file, grad_stats_path.open(
        "w", newline="", encoding="utf-8"
    ) as grad_file, batch_metrics_path.open("w", newline="", encoding="utf-8") as batch_file, probe_metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as probe_file:
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=["epoch", "lr", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        grad_writer = csv.DictWriter(
            grad_file,
            fieldnames=[
                "global_step",
                "epoch",
                "batch_in_epoch",
                "grad_l2",
                "grad_mean",
                "grad_abs_mean",
                "grad_max_abs",
                "grad_cosine_to_previous_batch",
            ],
        )
        batch_writer = csv.DictWriter(
            batch_file,
            fieldnames=["global_step", "epoch", "batch_in_epoch", "loss", "accuracy", "lr"],
        )
        probe_writer = csv.DictWriter(
            probe_file,
            fieldnames=["global_step", "epoch", "batch_in_epoch", "loss", "accuracy"],
        )
        metrics_writer.writeheader()
        grad_writer.writeheader()
        batch_writer.writeheader()
        probe_writer.writeheader()

        for epoch in range(1, config.epochs + 1):
            model.train()
            epoch_loss_sum = 0.0
            epoch_correct = 0
            epoch_seen = 0

            for batch_idx, (images, targets) in enumerate(train_loader, start=1):
                images = images.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = F.cross_entropy(logits, targets)
                loss.backward()

                if config.saved_gradient_source == "train":
                    grad_f32 = flattened_gradient_f32(model)
                    probe_loss = None
                    probe_accuracy = None
                else:
                    if probe_images is None or probe_targets is None:
                        raise RuntimeError("Probe batch was not initialized.")
                    train_grads = clone_current_grads(model)
                    optimizer.zero_grad(set_to_none=True)
                    grad_f32, probe_loss, probe_accuracy = probe_gradient_f32(
                        model=model,
                        images=probe_images,
                        targets=probe_targets,
                    )
                    restore_grads(model, train_grads)

                if gradients is not None:
                    gradients[global_step, :] = cast_gradient_for_storage(grad_f32, config.grad_dtype)
                stats = grad_stats(grad_f32, previous_grad)
                previous_grad = grad_f32.copy()

                optimizer.step()

                batch_correct = int((logits.argmax(dim=1) == targets).sum().item())
                batch_seen = int(targets.numel())
                batch_loss_sum = float(loss.item()) * batch_seen
                epoch_loss_sum += batch_loss_sum
                epoch_correct += batch_correct
                epoch_seen += batch_seen
                lr = float(optimizer.param_groups[0]["lr"])

                grad_writer.writerow(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_idx,
                        **stats,
                    }
                )
                batch_writer.writerow(
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_idx,
                        "loss": batch_loss_sum / batch_seen,
                        "accuracy": batch_correct / batch_seen,
                        "lr": lr,
                    }
                )
                if probe_loss is not None and probe_accuracy is not None:
                    probe_writer.writerow(
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "batch_in_epoch": batch_idx,
                            "loss": probe_loss,
                            "accuracy": probe_accuracy,
                        }
                    )
                global_step += 1

            if gradients is not None:
                gradients.flush()
            if scheduler is not None:
                scheduler.step()

            train_metrics = {
                "loss": epoch_loss_sum / max(epoch_seen, 1),
                "accuracy": epoch_correct / max(epoch_seen, 1),
            }
            val_metrics = evaluate(model, val_loader, device)
            epoch_row = {
                "epoch": epoch,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
            metrics_writer.writerow(epoch_row)
            metrics_file.flush()
            grad_file.flush()
            batch_file.flush()
            probe_file.flush()
            history.append(epoch_row)
            print(
                f"[{run_name}] epoch {epoch:03d}/{config.epochs} "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}",
                flush=True,
            )

    torch.save(model.state_dict(), run_dir / "model_state.pt")
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config": asdict(config),
        "n_trainable_parameters": n_params,
        "train_steps": train_steps,
        "elapsed_seconds": time.time() - start_time,
        "final_metrics": history[-1] if history else {},
        "artifacts": {
            "gradients": str(run_dir / "gradients.npy") if config.save_gradients == "full" else None,
            "param_metadata": str(run_dir / "param_metadata.json"),
            "grad_stats_batch": str(grad_stats_path),
            "metrics_batch": str(batch_metrics_path),
            "probe_metrics_batch": str(probe_metrics_path)
            if config.saved_gradient_source == "probe"
            else None,
            "metrics_epoch": str(metrics_path),
            "model_state": str(run_dir / "model_state.pt"),
        },
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def beta_pairs(args: argparse.Namespace) -> Iterable[tuple[float, float]]:
    if args.mode == "single":
        return [(args.beta1, args.beta2)]
    return [(beta1, beta2) for beta1 in args.beta_values for beta2 in args.beta_values]


def main() -> None:
    args = parse_args()
    pairs = list(beta_pairs(args))
    model = TinyCIFARCNN()
    train_size = 50_000 - args.val_size
    estimate = estimate_storage(
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_size=train_size,
        beta_count=len(pairs),
        grad_dtype=args.grad_dtype,
        save_gradients=args.save_gradients,
        saved_gradient_source=args.saved_gradient_source,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "storage_estimate.json", estimate)
    print(json.dumps(estimate, indent=2), flush=True)

    if args.estimate_only:
        print(f"Storage estimate written to: {args.output_dir / 'storage_estimate.json'}")
        return

    summaries = []
    for beta1, beta2 in pairs:
        config = ExperimentConfig(
            dataset=args.dataset,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            beta1=beta1,
            beta2=beta2,
            val_size=args.val_size,
            probe_batch_size=args.probe_batch_size,
            grad_dtype=args.grad_dtype,
            save_gradients=args.save_gradients,
            saved_gradient_source=args.saved_gradient_source,
            scheduler=args.scheduler,
            num_workers=args.num_workers,
        )
        summaries.append(
            run_one(
                config=config,
                output_dir=args.output_dir,
                data_dir=args.data_dir,
                download=not args.no_download,
            )
        )
    write_json(args.output_dir / "latest_summaries.json", summaries)


if __name__ == "__main__":
    main()
