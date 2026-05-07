import argparse
import time
from types import MethodType
from typing import Optional

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
from utils.datasets.wikitext import load_wikitext2 as load_wikitext
from utils.models import sinusoidal_init
from utils.runtime import load_nanogpt_symbols


torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

EXPERIMENT_NAME = "nanogpt_wikitext"

parser = argparse.ArgumentParser()
parser.add_argument("--beta1", required=True, help="Beta1 for Adam optimizer")
parser.add_argument("--beta2", required=True, help="Beta2 for Adam optimizer")
parser.add_argument("--seed", required=True, help="Seed for random number generators")
parser.add_argument("--smoke-test", action="store_true", help="Run a tiny validation training")
args = parser.parse_args()

SMOKE_TEST = bool(args.smoke_test)
EPOCHS = 1 if SMOKE_TEST else 60
LR = 3e-4
WD = 1e-4
BETAS = (float(args.beta1), float(args.beta2))
SEED = int(args.seed)
BLOCK_SIZE = 128 if SMOKE_TEST else 256
BATCH_SIZE = 2 if SMOKE_TEST else 8
MAX_TRAIN_SAMPLES = 24 if SMOKE_TEST else None
MAX_VAL_SAMPLES = 8 if SMOKE_TEST else None
EXPERIMENT_OUTPUT_NAME = "nanogpt_wikitext_smoke" if SMOKE_TEST else EXPERIMENT_NAME

GPT, GPTConfig = load_nanogpt_symbols()


def train_step_transformer(self, batch):
    inputs, targets = batch
    inputs, targets = inputs.to(self.device), targets.to(self.device)
    self.optimizer.zero_grad()
    logits, loss = self.forward(inputs, targets)
    if not torch.isfinite(loss):
        raise Exception("Loss is NaN or Inf! Stopping training.")
    if self.use_amp:
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
    else:
        loss.backward()

    if self.clipping:
        nn.utils.clip_grad_norm_(self.parameters(), 1, error_if_nonfinite=True)

    for p in self.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise Exception("Gradients are not finite. Stopping the execution.")

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
    return logits, loss.item()


def training_one_epoch_transformer(self, train_data, num_epochs, epoch, verbose):
    self.train()
    running_loss = 0.0
    total_loss = 0.0
    self.metrics.reset_train()

    n_steps = len(train_data)
    for i, batch in enumerate(train_data):
        logits, loss = self.train_step(batch)
        targets = batch[1].to(self.device)
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)
        mask = targets_flat != -1
        total_loss += loss
        running_loss += loss
        self.metrics.update_train(logits_flat[mask], targets_flat[mask])

        if ((i + 1) % 20 == 0) and verbose:
            avg_loss = running_loss / 20
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs} - Step {i+1}/{n_steps}: "
                f"Train Acc: {self.metrics.train_acc:.2f}%, Loss: {avg_loss:.4f}",
                end="\n",
                flush=True,
            )
            running_loss = 0
            for callback in self.callbacks:
                callback.on_epoch_end(self)

    return total_loss / len(train_data)


def evaluate_step_transformer(self, batch):
    with torch.no_grad():
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        logits, loss = self.forward(inputs, targets)
    return logits, loss.item()


def evaluate_model_transformer(self, val_data, verbose=True):
    self.to(self.device)
    self.eval()
    val_loss = 0.0
    self.metrics.reset_val()
    if val_data:
        with torch.no_grad():
            for batch in val_data:
                logits, loss = self.evaluate_step(batch)
                targets = batch[1].to(self.device)
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)
                val_loss += loss
                self.metrics.update_val(logits_flat, targets_flat)

        if verbose:
            print(self.metrics)

        return {"loss": val_loss / len(val_data), "metric": self.metrics.val}


def create_nanogpt_model(vocab_size: int, config_name: str = "gpt2-small", device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    configs = {
        "gpt2-small": {
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "block_size": 1024,
            "bias": True,
            "vocab_size": vocab_size,
            "dropout": 0.1,
        },
        "gpt2-medium": {
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 1024,
            "block_size": 1024,
            "bias": True,
            "vocab_size": vocab_size,
            "dropout": 0.1,
        },
        "gpt2-large": {
            "n_layer": 36,
            "n_head": 20,
            "n_embd": 1280,
            "block_size": 1024,
            "bias": True,
            "vocab_size": vocab_size,
            "dropout": 0.1,
        },
        "gpt2-xl": {
            "n_layer": 48,
            "n_head": 25,
            "n_embd": 1600,
            "block_size": 1024,
            "bias": True,
            "vocab_size": vocab_size,
            "dropout": 0.1,
        },
        "tiny": {
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 128,
            "block_size": 256,
            "bias": True,
            "vocab_size": vocab_size,
            "dropout": 0.1,
        },
    }

    if config_name not in configs:
        raise ValueError(f"Config {config_name} not found. Choose from: {list(configs.keys())}")

    model_config = GPTConfig(**configs[config_name])
    model = GPT(model_config)

    def _init_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    model.apply(_init_weights)
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created with config: {config_name}")
    print(f"Total parameters: {param_count:,}")
    print(f"Model device: {device}")

    return model


def main():
    wall_start = time.perf_counter()
    class NanoGPTCrossEntropyLoss(nn.Module):
        def __init__(self, ignore_index: int = -1, reduction: str = "mean"):
            super().__init__()
            self.ignore_index = int(ignore_index)
            if reduction not in ("mean", "sum", "none"):
                raise ValueError("reduction must be one of: 'mean', 'sum', 'none'")
            self.reduction = reduction

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            if logits.ndim != 3:
                raise ValueError(f"logits must be (B,T,V), got shape {tuple(logits.shape)}")
            if targets.ndim != 2:
                raise ValueError(f"targets must be (B,T), got shape {tuple(targets.shape)}")

            batch, steps, vocab = logits.shape
            if targets.shape != (batch, steps):
                raise ValueError(f"targets shape {tuple(targets.shape)} does not match (B,T)=({batch},{steps})")

            logits_flat = logits.reshape(batch * steps, vocab)
            targets_flat = targets.reshape(batch * steps)
            import torch.nn.functional as F

            return F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
            )

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_loader, test_loader, vocab_size = load_wikitext(
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        tokenizer_name="gpt2",
        max_train_samples=MAX_TRAIN_SAMPLES,
        max_val_samples=MAX_VAL_SAMPLES,
        nanogpt_pairs=True,
    )
    if SMOKE_TEST:
        print(f"[nanogpt_wikitext] smoke train_batches={len(train_loader)} val_batches={len(test_loader)}", flush=True)
    model = create_nanogpt_model(vocab_size=vocab_size, config_name="gpt2-small")
    model.apply(sinusoidal_init)
    model = Model(predefined_model=model)
    model.train_step = MethodType(train_step_transformer, model)
    model.training_one_epoch = MethodType(training_one_epoch_transformer, model)
    model.evaluate_step = MethodType(evaluate_step_transformer, model)
    model.evaluate_model = MethodType(evaluate_model_transformer, model)
    model.compile(
        loss=NanoGPTCrossEntropyLoss(),
        optimizer=optim.Adam,
        optimizer_params={"lr": LR, "weight_decay": WD, "betas": BETAS},
        metrics=Accuracy(),
        use_amp=False,
    )
    model.clipping = True

    total_steps = int(EPOCHS * len(train_loader) / 20)
    warmup_steps = int(0.2 * total_steps)
    cosine_steps = max(1, total_steps - warmup_steps)

    seq_scheduler = torch.optim.lr_scheduler.SequentialLR(
        model.optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(model.optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(model.optimizer, T_max=cosine_steps, eta_min=1e-6),
        ],
        milestones=[warmup_steps],
    )

    complete_scheduler = LRScheduler(seq_scheduler)
    model.add_callback([complete_scheduler])
    history = model.train_model(train_loader, test_loader, num_epochs=EPOCHS, verbose=True)
    history = {
        "model": "NanoGPT",
        "dataset": "WikiText",
        "initialization": "sinusoidal",
        "epochs": EPOCHS,
        "optimizer": "Adam",
        "beta1": BETAS[0],
        "beta2": BETAS[1],
        "seed": SEED,
        "loss": "CrossEntropyLoss",
        "lr": LR,
        "wd": WD,
        "block_size": BLOCK_SIZE,
        "batch_size": BATCH_SIZE,
        "smoke_test": SMOKE_TEST,
        **history,
        **model.history,
    }
    history["wall_time_total_sec"] = time.perf_counter() - wall_start
    save_experiment_run(EXPERIMENT_OUTPUT_NAME, history)


if __name__ == "__main__":
    main()
