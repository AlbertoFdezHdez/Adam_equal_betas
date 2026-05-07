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
from utils.datasets import load_squad
from utils.models import load_t5, sinusoidal_init


torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

EXPERIMENT_NAME = "t5_squad"

parser = argparse.ArgumentParser()
parser.add_argument("--beta1", required=True, help="Beta1 for Adam optimizer")
parser.add_argument("--beta2", required=True, help="Beta2 for Adam optimizer")
parser.add_argument("--seed", required=True, help="Seed for random number generators")
parser.add_argument("--smoke-test", action="store_true", help="Run a tiny validation training")
args = parser.parse_args()

SMOKE_TEST = bool(args.smoke_test)
EPOCHS = 2 if SMOKE_TEST else 60
WARMUP_EPOCHS = 1 if SMOKE_TEST else 0.2 * EPOCHS
LR = 1e-3
WD = 1e-2
BETAS = (float(args.beta1), float(args.beta2))
SEED = int(args.seed)
EXPERIMENT_OUTPUT_NAME = "t5_squad_smoke" if SMOKE_TEST else EXPERIMENT_NAME


def train_step(self, batch):
    inputs, targets = batch
    inputs, targets = inputs.to(self.device), targets.to(self.device)
    self.optimizer.zero_grad()
    outputs = self.forward(inputs)
    loss = self.loss(outputs, targets)

    if self.use_amp:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
    else:
        loss.backward()

    if self.clipping:
        nn.utils.clip_grad_norm_(self.parameters(), 1, error_if_nonfinite=True)

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


def evaluate_step_transformer(self, batch):
    with torch.no_grad():
        batch = {k: v.to(self.device) for k, v in batch.items()}
        outputs = self.forward(**batch)
        loss = outputs.loss
    return outputs, loss.item()


def train_step_transformer(self, batch):
    self.optimizer.zero_grad()
    batch = {k: v.to(self.device) for k, v in batch.items()}
    outputs = self.forward(**batch)
    loss = outputs.loss
    if not torch.isfinite(loss):
        raise Exception("Loss is NaN or Inf! Stopping training.")
    if self.use_amp:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
    else:
        loss.backward()
    param1 = get_weight_params(self)
    grad = get_weight_gradients(self)
    if self.clipping:
        nn.utils.clip_grad_norm_(self.parameters(), 1, error_if_nonfinite=True)

    for p in self.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise Exception("Gradients are not finite. Stopping the execution.")

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


def training_one_epoch_transformer(self, train_data, num_epochs, epoch, verbose):
    self.train()
    running_loss = 0.0
    total_loss = 0.0
    self.metrics.reset_train()

    n_steps = len(train_data)
    for i, batch in enumerate(train_data):
        outputs, loss = self.train_step(batch)
        preds = outputs.logits.argmax(dim=-1)
        mask = batch["labels"] != -100
        preds = preds[mask].to(self.device)
        targets = batch["labels"].to(self.device)[mask]

        total_loss += loss
        running_loss += loss
        self.metrics.update_train(preds, targets, done=True)

        if ((i + 1) % 20 == 0) and verbose:
            avg_loss = running_loss / 20
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} - Step {i+1}/{n_steps}: "
                f"Train Acc: {self.metrics.train_acc:.2f}%, Loss: {avg_loss:.4f}",
                end="\n",
                flush=True,
            )
            running_loss = 0

    return total_loss / len(train_data)


def evaluate_model_transformer(self, val_data, verbose=True):
    self.to(self.device)
    self.eval()
    val_loss = 0.0
    self.metrics.reset_val()
    if val_data:
        with torch.no_grad():
            for batch in val_data:
                outputs, loss = self.evaluate_step(batch)
                preds = outputs.logits.argmax(dim=-1)
                mask = batch["labels"] != -100
                preds = preds[mask].to(self.device)
                targets = batch["labels"].to(self.device)[mask]

                val_loss += loss
                self.metrics.update_val(preds, targets, done=True)

        if verbose:
            print(self.metrics)

        return {"loss": val_loss / len(val_data), "metric": self.metrics.val}


def main():
    wall_start = time.perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = load_t5(input_shape=None, num_classes=None)
    train_loader, test_loader = load_squad(
        model,
        batch_size=2 if SMOKE_TEST else 8,
        max_train_samples=16 if SMOKE_TEST else None,
        max_val_samples=8 if SMOKE_TEST else None,
    )
    if SMOKE_TEST:
        print(f"[t5_squad] smoke train_batches={len(train_loader)} val_batches={len(test_loader)}", flush=True)

    model.apply(sinusoidal_init)
    model = Model(predefined_model=model)
    model.train_step = MethodType(train_step_transformer, model)
    model.training_one_epoch = MethodType(training_one_epoch_transformer, model)
    model.evaluate_step = MethodType(evaluate_step_transformer, model)
    model.evaluate_model = MethodType(evaluate_model_transformer, model)
    model.compile(
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam,
        optimizer_params={"lr": LR, "weight_decay": WD, "betas": BETAS},
        metrics=Accuracy(),
        use_amp=False,
    )

    warmup_epochs = WARMUP_EPOCHS
    cosine_epochs = max(1, EPOCHS - warmup_epochs)
    seq_scheduler = optim.lr_scheduler.SequentialLR(
        model.optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(model.optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs),
            optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=cosine_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs],
    )
    complete_scheduler = LRScheduler(seq_scheduler)
    model.add_callback([complete_scheduler])
    history = model.train_model(train_loader, test_loader, num_epochs=EPOCHS, verbose=True)
    history = {
        "model": "T5",
        "dataset": "SQuAD",
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
