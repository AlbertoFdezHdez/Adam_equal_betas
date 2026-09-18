"""Generate small, fully reproducible gradient trajectories on sklearn digits.

The optimizer uses the exact full training gradient at every step, and that same
gradient is stored before the parameter update.  This removes minibatch noise
without replacing the learning problem by a synthetic signal.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--activation", choices=["tanh", "relu", "softplus"], default="tanh")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Use 0 for full-batch training with the stored training gradient, "
            "a positive value for minibatches plus a fixed probe, or -1 for a "
            "full optimization split plus the same fixed probe."
        ),
    )
    parser.add_argument(
        "--probe-size",
        type=int,
        default=256,
        help="Fixed probe size used only when minibatch training is enabled.",
    )
    parser.add_argument("--log-every", type=int, default=250)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_model(hidden: int, activation: str) -> nn.Module:
    activation_layer: type[nn.Module]
    if activation == "tanh":
        activation_layer = nn.Tanh
    elif activation == "relu":
        activation_layer = nn.ReLU
    else:
        activation_layer = nn.Softplus
    return nn.Sequential(
        nn.Linear(64, hidden),
        activation_layer(),
        nn.Linear(hidden, hidden),
        activation_layer(),
        nn.Linear(hidden, 10),
    )


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    return torch.cat([parameter.grad.detach().reshape(-1) for parameter in model.parameters()])


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((model(x).argmax(dim=1) == y).float().mean())


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(min(10, torch.get_num_threads()))

    dataset = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.data.astype(np.float32),
        dataset.target.astype(np.int64),
        test_size=0.2,
        random_state=args.seed,
        stratify=dataset.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(y_train)
    test_x = torch.from_numpy(x_test)
    test_y = torch.from_numpy(y_test)

    model = make_model(args.hidden, args.activation)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gradients_path = args.output_dir / "gradients.npy"
    gradients = np.lib.format.open_memmap(
        gradients_path,
        mode="w+",
        dtype=np.float32,
        shape=(args.steps, parameter_count),
    )
    minibatch_rng = np.random.default_rng(args.seed + 104729)
    if args.batch_size != 0:
        split_order = minibatch_rng.permutation(train_x.shape[0])
        reserved_for_optimization = 1 if args.batch_size < 0 else args.batch_size
        probe_count = min(args.probe_size, train_x.shape[0] - reserved_for_optimization)
        probe_indices = split_order[:probe_count]
        optimization_indices = split_order[probe_count:]
        epoch_order = minibatch_rng.permutation(optimization_indices)
        cursor = 0
        probe_x = train_x[probe_indices]
        probe_y = train_y[probe_indices]
    else:
        probe_x = train_x
        probe_y = train_y

    losses: list[float] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        if args.batch_size != 0:
            probe_loss = criterion(model(probe_x), probe_y)
            probe_loss.backward()
            gradients[step] = flatten_gradients(model).cpu().numpy()
            optimizer.zero_grad(set_to_none=True)
            if args.batch_size < 0:
                batch_indices = optimization_indices
            else:
                if cursor + args.batch_size > epoch_order.size:
                    epoch_order = minibatch_rng.permutation(optimization_indices)
                    cursor = 0
                batch_indices = epoch_order[cursor : cursor + args.batch_size]
                cursor += args.batch_size
            loss = criterion(model(train_x[batch_indices]), train_y[batch_indices])
            loss.backward()
        else:
            loss = criterion(model(train_x), train_y)
            loss.backward()
            gradients[step] = flatten_gradients(model).cpu().numpy()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % args.log_every == 0:
            print(f"step={step + 1:5d} loss={losses[-1]:.6f}", flush=True)
    gradients.flush()

    metadata = {
        "dataset": "sklearn_digits",
        "split": "stratified_80_20",
        "train_examples": int(train_x.shape[0]),
        "test_examples": int(test_x.shape[0]),
        "input_preprocessing": "StandardScaler fit on training split",
        "training_gradient": (
            "full training split; stored before every optimizer step"
            if args.batch_size == 0
            else (
                "full non-probe training split"
                if args.batch_size < 0
                else f"minibatches of {args.batch_size} from the non-probe training split"
            )
        ),
        "stored_gradient": (
            "full training split"
            if args.batch_size == 0
            else f"fixed probe of {probe_x.shape[0]} training examples; before optimizer step"
        ),
        "model": f"MLP-64-{args.hidden}-{args.hidden}-10-{args.activation}",
        "parameter_count": parameter_count,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "optimizer": "torch.optim.Adam",
        "betas": [args.beta1, args.beta2],
        "epsilon": args.epsilon,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "gradient_dtype": "float32",
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "train_accuracy": accuracy(model, train_x, train_y),
        "test_accuracy": accuracy(model, test_x, test_y),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
