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
from utils.datasets import load_cifar100
from utils.datasets.utils import get_reduced_dataloader
from utils.models import load_resnet18, sinusoidal_init


EXPERIMENT_NAME = "resnet18_cifar100"

parser = argparse.ArgumentParser()
parser.add_argument("--beta1", required=True, help="Beta1 for Adam optimizer")
parser.add_argument("--beta2", required=True, help="Beta2 for Adam optimizer")
parser.add_argument("--seed", required=True, help="Seed for random number generators")
parser.add_argument("--smoke-test", action="store_true", help="Run a tiny validation training")
args = parser.parse_args()

SMOKE_TEST = bool(args.smoke_test)
EPOCHS = 2 if SMOKE_TEST else 100
LR = 2e-2
WD = 1e-3
BETAS = (float(args.beta1), float(args.beta2))
SEED = int(args.seed)
WARMUP_EPOCHS = 1 if SMOKE_TEST else 10
EXPERIMENT_OUTPUT_NAME = "resnet18_cifar100_smoke" if SMOKE_TEST else EXPERIMENT_NAME


def train_step(self, batch):
    inputs, targets = batch
    inputs, targets = inputs.to(self.device), targets.to(self.device)
    self.optimizer.zero_grad()
    outputs = self.forward(inputs)
    loss = self.loss(outputs, targets)
    loss.backward()
    param1 = get_weight_params(self)
    grad = get_weight_gradients(self)
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

    train_loader, test_loader, input_shape, num_classes = load_cifar100(batch_size=16 if SMOKE_TEST else 64)
    if SMOKE_TEST:
        train_loader = get_reduced_dataloader(train_loader, n_samples=128)
        test_loader = get_reduced_dataloader(test_loader, n_samples=64)
        print(f"[resnet18_cifar100] smoke train_batches={len(train_loader)} val_batches={len(test_loader)}", flush=True)

    model = load_resnet18(input_shape=input_shape, num_classes=num_classes)
    model.apply(sinusoidal_init)
    model = Model(predefined_model=model)
    model.train_step = MethodType(train_step, model)
    model.compile(
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam,
        optimizer_params={"lr": LR, "weight_decay": WD, "betas": BETAS},
        metrics=Accuracy(),
    )

    warmup_epochs = WARMUP_EPOCHS
    cosine_epochs = max(1, EPOCHS - warmup_epochs)
    seq_scheduler = optim.lr_scheduler.SequentialLR(
        model.optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(model.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=cosine_epochs, eta_min=0),
        ],
        milestones=[warmup_epochs],
    )
    complete_scheduler = LRScheduler(seq_scheduler)
    model.add_callback([complete_scheduler])
    history = model.train_model(train_loader, test_loader, num_epochs=EPOCHS, verbose=True)
    history = {
        "model": "ResNet18",
        "dataset": "CIFAR-100",
        "initialization": "sinusoidal",
        "epochs": EPOCHS,
        "optimizer": "Adam",
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
