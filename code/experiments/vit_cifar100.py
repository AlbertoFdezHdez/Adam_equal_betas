import argparse
import time
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from utils import Accuracy, LRScheduler, Model, get_weight_gradients, get_weight_params, save_experiment_run
from utils.datasets import load_cifar100_vit
from utils.datasets.utils import get_reduced_dataloader
from utils.models import load_vit, sinusoidal_init


torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

EXPERIMENT_NAME = "vit_cifar100"

parser = argparse.ArgumentParser()
parser.add_argument("--beta1", required=True, help="Beta1 for Adam optimizer")
parser.add_argument("--beta2", required=True, help="Beta2 for Adam optimizer")
parser.add_argument("--seed", required=True, help="Seed for random number generators")
parser.add_argument("--smoke-test", action="store_true", help="Run a tiny validation training")
args = parser.parse_args()

SMOKE_TEST = bool(args.smoke_test)
EPOCHS = 2 if SMOKE_TEST else 60
WARMUP_EPOCHS = 1 if SMOKE_TEST else 0.1 * EPOCHS
LR = 3e-4
WD = 0.2
BETAS = (float(args.beta1), float(args.beta2))
SEED = int(args.seed)
SKIP_NONFINITE_LOSS = True
EXPERIMENT_OUTPUT_NAME = "vit_cifar100_smoke" if SMOKE_TEST else EXPERIMENT_NAME


def train_step(self, batch):
    inputs, targets = batch
    inputs, targets = inputs.to(self.device), targets.to(self.device)
    self.optimizer.zero_grad()
    outputs = self.forward(inputs)
    loss = self.loss(outputs, targets)
    if not torch.isfinite(loss):
        if SKIP_NONFINITE_LOSS:
            self.optimizer.zero_grad()
            if not hasattr(self, "history"):
                self.history = {"updates_norm": [], "grads_norm": [], "train_step_loss": []}
            self.history["updates_norm"].append(0.0)
            self.history["grads_norm"].append(0.0)
            self.history["train_step_loss"].append(float("inf"))
            return outputs, float("inf")
        raise Exception("Loss is NaN or Inf! Stopping training.")
    if self.use_amp:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
    else:
        loss.backward()

    if self.clipping:
        nn.utils.clip_grad_norm_(self.parameters(), 1, error_if_nonfinite=False)

    grads_nonfinite = False
    for p in self.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grads_nonfinite = True

    if grads_nonfinite:
        for state in self.optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    value.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    param1 = get_weight_params(self)
    grad = get_weight_gradients(self)

    if self.use_amp:
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        self.optimizer.step()

    param2 = get_weight_params(self)
    if not hasattr(self, "history"):
        self.history = {"updates_norm": [], "grads_norm": [], "train_step_loss": []}
    lr = self.optimizer.param_groups[0]["lr"]
    self.history["updates_norm"].append(torch.norm((param2 - param1) / lr, p=2).cpu().item())
    self.history["grads_norm"].append(torch.norm(grad, p=2).cpu().item())
    self.history["train_step_loss"].append(loss.item())
    return outputs, loss.item()


def main():
    wall_start = time.perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader, test_loader, input_shape, num_classes = load_cifar100_vit(batch_size=8 if SMOKE_TEST else 256, size=224)
    if SMOKE_TEST:
        train_loader = get_reduced_dataloader(train_loader, n_samples=64)
        test_loader = get_reduced_dataloader(test_loader, n_samples=32)
        print(f"[vit_cifar100] smoke train_batches={len(train_loader)} val_batches={len(test_loader)}", flush=True)

    model = load_vit(input_shape=input_shape, num_classes=num_classes)
    model.apply(sinusoidal_init)
    model = Model(predefined_model=model)
    model.train_step = MethodType(train_step, model)
    model.compile(
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW,
        optimizer_params={"lr": LR, "weight_decay": WD, "betas": BETAS},
        metrics=Accuracy(),
        use_amp=False,
    )

    warmup_epochs = WARMUP_EPOCHS
    cosine_epochs = max(1, EPOCHS - warmup_epochs)
    seq_scheduler = optim.lr_scheduler.SequentialLR(
        model.optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(model.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=cosine_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )
    complete_scheduler = LRScheduler(seq_scheduler)
    model.add_callback([complete_scheduler])
    history = model.train_model(train_loader, test_loader, num_epochs=EPOCHS, verbose=True)
    history = {
        "model": "ViT-B_16",
        "dataset": "CIFAR-100",
        "initialization": "sinusoidal",
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "beta1": BETAS[0],
        "beta2": BETAS[1],
        "seed": SEED,
        "loss": "CrossEntropyLoss",
        "lr": LR,
        "wd": WD,
        **history,
        **model.history,
    }
    history["smoke_test"] = SMOKE_TEST
    history["wall_time_total_sec"] = time.perf_counter() - wall_start
    save_experiment_run(EXPERIMENT_OUTPUT_NAME, history)


if __name__ == "__main__":
    main()
